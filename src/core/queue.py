"""SQLite apply queue: READY → CLAIMED → DONE (feeds parallel browser workers)."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.core.host_limiter import host_bucket_key, spacing_interval_for
from src.core.memory import _connect, init_db

logger = logging.getLogger(__name__)

QUEUE_STATUSES = frozenset({"READY", "CLAIMED", "DONE", "CANCELLED"})
_RECLAIM_RE = re.compile(r"(?:^|;)\s*reclaim=(\d+)\s*(?:;|$)", re.I)

# Soft safety cap — eligibility is host-aware across this window, not a fixed peek of 25.
CLAIM_SCAN_LIMIT_DEFAULT = 500
# Legacy name kept for imports; eligibility no longer stops at 25.
CLAIM_PEEK_LIMIT = CLAIM_SCAN_LIMIT_DEFAULT
WAIT_SENTINEL_KEY = "__wait__"


def claim_scan_limit() -> int:
    raw = (os.environ.get("ACE_CLAIM_SCAN_LIMIT") or str(CLAIM_SCAN_LIMIT_DEFAULT)).strip()
    try:
        return max(50, int(raw))
    except ValueError:
        return CLAIM_SCAN_LIMIT_DEFAULT


def _ensure_queue_columns(conn) -> None:
    rows = conn.execute("PRAGMA table_info(apply_queue)").fetchall()
    cols = {str(r["name"] if hasattr(r, "keys") else r[1]) for r in rows}
    if "job_title" not in cols:
        conn.execute("ALTER TABLE apply_queue ADD COLUMN job_title TEXT")
    if "attempt_count" not in cols:
        conn.execute(
            "ALTER TABLE apply_queue ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
        )
    if "host_key" not in cols:
        conn.execute("ALTER TABLE apply_queue ADD COLUMN host_key TEXT")


def _ensure_host_schedule(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS host_schedule (
            host_key TEXT PRIMARY KEY,
            next_eligible_at REAL NOT NULL DEFAULT 0,
            jitter_sec REAL NOT NULL DEFAULT 0
        )
        """
    )


def init_queue(db_path: Path | None = None) -> Path:
    path = init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS apply_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                job_url TEXT NOT NULL UNIQUE,
                track TEXT,
                jd_text TEXT,
                pdf_path TEXT NOT NULL,
                yaml_snapshot TEXT,
                status TEXT NOT NULL DEFAULT 'READY',
                worker_id TEXT,
                error_log TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                job_title TEXT,
                host_key TEXT
            )
            """
        )
        _ensure_queue_columns(conn)
        _ensure_host_schedule(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_apply_queue_status "
            "ON apply_queue(status, id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_apply_queue_host "
            "ON apply_queue(status, host_key, id)"
        )
        # Backfill missing host_key for older READY/CLAIMED rows.
        rows = conn.execute(
            "SELECT id, job_url FROM apply_queue "
            "WHERE host_key IS NULL OR host_key = ''"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE apply_queue SET host_key=? WHERE id=?",
                (host_bucket_key(str(row["job_url"] or "")), int(row["id"])),
            )
        conn.commit()
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enqueue_apply(
    *,
    company: str,
    job_url: str,
    track: str,
    jd_text: str,
    pdf_path: str,
    yaml_snapshot: str = "",
    job_title: str = "",
    db_path: Path | None = None,
) -> int:
    """Insert or refresh a READY apply item. Returns row id."""
    init_queue(db_path)
    url = (job_url or "").strip()
    ts = _now()
    title = (job_title or "").strip()
    hkey = host_bucket_key(url)
    with _connect(db_path) as conn:
        _ensure_queue_columns(conn)
        _ensure_host_schedule(conn)
        conn.execute(
            """
            INSERT INTO apply_queue (
                company, job_url, track, jd_text, pdf_path, yaml_snapshot,
                status, worker_id, error_log, created_at, updated_at, job_title,
                host_key
            )
            VALUES (?, ?, ?, ?, ?, ?, 'READY', NULL, '', ?, ?, ?, ?)
            ON CONFLICT(job_url) DO UPDATE SET
                company=excluded.company,
                track=excluded.track,
                jd_text=excluded.jd_text,
                pdf_path=excluded.pdf_path,
                yaml_snapshot=excluded.yaml_snapshot,
                job_title=COALESCE(NULLIF(excluded.job_title, ''), apply_queue.job_title),
                host_key=excluded.host_key,
                status='READY',
                worker_id=NULL,
                error_log='',
                updated_at=excluded.updated_at
            WHERE apply_queue.status IN ('READY', 'CANCELLED')
            """,
            (
                (company or "Unknown").strip() or "Unknown",
                url,
                (track or "").strip(),
                jd_text or "",
                pdf_path,
                yaml_snapshot or "",
                ts,
                ts,
                title,
                hkey,
            ),
        )
        row = conn.execute(
            "SELECT id FROM apply_queue WHERE job_url = ?", (url,)
        ).fetchone()
        conn.commit()
    return int(row["id"]) if row else 0


def _schedule_ready_in(conn, host_key: str, *, now: float) -> float:
    row = conn.execute(
        "SELECT next_eligible_at FROM host_schedule WHERE host_key = ?",
        (host_key,),
    ).fetchone()
    if not row:
        return 0.0
    try:
        nxt = float(row["next_eligible_at"] or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, nxt - now)


def _reserve_host(conn, host_key: str, *, now: float) -> float:
    """Upsert next_eligible_at for host_key. Returns interval used."""
    interval = spacing_interval_for(host_key)
    nxt = float(now) + interval
    conn.execute(
        """
        INSERT INTO host_schedule (host_key, next_eligible_at, jitter_sec)
        VALUES (?, ?, ?)
        ON CONFLICT(host_key) DO UPDATE SET
            next_eligible_at=excluded.next_eligible_at,
            jitter_sec=excluded.jitter_sec
        """,
        (host_key, nxt, interval),
    )
    return interval


def reserve_host(
    host_key: str,
    *,
    db_path: Path | None = None,
    now: float | None = None,
) -> float:
    """Public helper: reserve host spacing (tests / dual-write)."""
    init_queue(db_path)
    t = float(now if now is not None else time.time())
    key = (host_key or "default").strip() or "default"
    with _connect(db_path) as conn:
        _ensure_host_schedule(conn)
        interval = _reserve_host(conn, key, now=t)
        conn.commit()
    return interval


def claim_next(
    worker_id: str,
    db_path: Path | None = None,
    *,
    host_limiter: object | None = None,
    now: float | None = None,
) -> dict | None:
    """Claim the oldest READY row whose host is off cooldown (durable schedule).

    Scans up to ``ACE_CLAIM_SCAN_LIMIT`` READY rows (default 500) so a free host
    beyond a cooling prefix is still claimed. When every scanned host is cooling,
    returns ``{"__wait__": seconds}`` without claiming.
    """
    init_queue(db_path)
    ts = _now()
    wid = (worker_id or "w").strip() or "w"
    clock = float(now if now is not None else time.time())
    scan = claim_scan_limit()
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_queue_columns(conn)
        _ensure_host_schedule(conn)
        rows = conn.execute(
            """
            SELECT * FROM apply_queue
            WHERE status = 'READY'
            ORDER BY id ASC
            LIMIT ?
            """,
            (scan,),
        ).fetchall()
        if not rows:
            conn.commit()
            return None

        pick = None
        best_wait: float | None = None
        ready_in_mem = getattr(host_limiter, "ready_in", None) if host_limiter else None

        for row in rows:
            hkey = str(row["host_key"] or "").strip() or host_bucket_key(
                str(row["job_url"] or "")
            )
            wait_sched = _schedule_ready_in(conn, hkey, now=clock)
            wait_mem = 0.0
            if callable(ready_in_mem):
                try:
                    wait_mem = float(ready_in_mem(str(row["job_url"] or "")) or 0.0)
                except Exception:
                    wait_mem = 0.0
            wait = max(wait_sched, wait_mem)
            if wait <= 0:
                pick = row
                break
            best_wait = wait if best_wait is None else min(best_wait, wait)

        if pick is None:
            conn.commit()
            return {
                WAIT_SENTINEL_KEY: float(
                    best_wait if best_wait is not None else 1.0
                )
            }

        hkey = str(pick["host_key"] or "").strip() or host_bucket_key(
            str(pick["job_url"] or "")
        )
        conn.execute(
            """
            UPDATE apply_queue
            SET status='CLAIMED', worker_id=?, updated_at=?, host_key=?
            WHERE id=? AND status='READY'
            """,
            (wid, ts, hkey, pick["id"]),
        )
        if conn.total_changes == 0:
            conn.commit()
            return None
        _reserve_host(conn, hkey, now=clock)
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM apply_queue WHERE id = ?", (pick["id"],)
        ).fetchone()

    # Keep optional in-memory limiter in sync for callers that still pass it.
    if host_limiter is not None and claimed is not None:
        mark = getattr(host_limiter, "mark_dispatched", None)
        if callable(mark):
            try:
                mark(str(claimed["job_url"] or ""))
            except Exception:
                logger.debug("host_limiter.mark_dispatched sync failed", exc_info=True)

    return dict(claimed) if claimed else None


def mark_done(
    queue_id: int,
    *,
    error_log: str = "",
    db_path: Path | None = None,
) -> None:
    init_queue(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE apply_queue
            SET status='DONE', error_log=?, updated_at=?
            WHERE id=?
            """,
            (error_log or "", _now(), int(queue_id)),
        )
        conn.commit()


def requeue_claimed(
    queue_id: int,
    *,
    error_log: str = "circuit_cooldown",
    bump_attempt: bool = False,
    db_path: Path | None = None,
) -> bool:
    """Return a CLAIMED row to READY (ban-armor circuit / pacing / policy retry)."""
    init_queue(db_path)
    with _connect(db_path) as conn:
        _ensure_queue_columns(conn)
        if bump_attempt:
            cur = conn.execute(
                """
                UPDATE apply_queue
                SET status='READY',
                    worker_id=NULL,
                    error_log=?,
                    updated_at=?,
                    attempt_count=COALESCE(attempt_count, 0) + 1
                WHERE id=? AND status='CLAIMED'
                """,
                (error_log or "circuit_cooldown", _now(), int(queue_id)),
            )
        else:
            cur = conn.execute(
                """
                UPDATE apply_queue
                SET status='READY', worker_id=NULL, error_log=?, updated_at=?
                WHERE id=? AND status='CLAIMED'
                """,
                (error_log or "circuit_cooldown", _now(), int(queue_id)),
            )
        conn.commit()
        return cur.rowcount > 0


def get_attempt_count(job_url: str, db_path: Path | None = None) -> int:
    """Current attempt_count for a queue row (0 if missing)."""
    init_queue(db_path)
    url = (job_url or "").strip()
    if not url:
        return 0
    with _connect(db_path) as conn:
        _ensure_queue_columns(conn)
        row = conn.execute(
            "SELECT attempt_count FROM apply_queue WHERE job_url = ? LIMIT 1",
            (url,),
        ).fetchone()
    if not row:
        return 0
    try:
        return max(0, int(row["attempt_count"] or 0))
    except (TypeError, ValueError, KeyError):
        return 0


def bump_attempt_on_claim(queue_id: int, db_path: Path | None = None) -> int:
    """Increment attempt_count when a worker starts an apply. Returns new count."""
    init_queue(db_path)
    with _connect(db_path) as conn:
        _ensure_queue_columns(conn)
        conn.execute(
            """
            UPDATE apply_queue
            SET attempt_count=COALESCE(attempt_count, 0) + 1, updated_at=?
            WHERE id=?
            """,
            (_now(), int(queue_id)),
        )
        row = conn.execute(
            "SELECT attempt_count FROM apply_queue WHERE id = ?",
            (int(queue_id),),
        ).fetchone()
        conn.commit()
    if not row:
        return 1
    try:
        return max(1, int(row["attempt_count"] or 1))
    except (TypeError, ValueError, KeyError):
        return 1


def _parse_reclaim_count(error_log: str | None) -> int:
    m = _RECLAIM_RE.search(str(error_log or ""))
    if not m:
        return 0
    try:
        return max(0, int(m.group(1)))
    except ValueError:
        return 0


def _with_reclaim_count(error_log: str | None, n: int) -> str:
    base = _RECLAIM_RE.sub("", str(error_log or "")).strip(" ;")
    prefix = f"reclaim={int(n)}"
    return f"{prefix};{base}" if base else prefix


def reclaim_max() -> int:
    raw = (os.environ.get("ACE_CLAIM_RECLAIM_MAX") or "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def stale_claim_sec() -> int:
    raw = (os.environ.get("ACE_CLAIM_STALE_SEC") or "3600").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 3600


def requeue_stale_claims(max_age_sec: int | None = None, db_path: Path | None = None) -> int:
    """Return CLAIMED rows stuck too long back to READY (crashed workers)."""
    init_queue(db_path)
    age = int(max_age_sec) if max_age_sec is not None else stale_claim_sec()
    cutoff = datetime.now(timezone.utc).timestamp() - max(60, age)
    n = 0
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, updated_at, error_log FROM apply_queue WHERE status='CLAIMED'"
        ).fetchall()
        for row in rows:
            try:
                ts = datetime.fromisoformat(str(row["updated_at"]))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts.timestamp() >= cutoff:
                    continue
            except ValueError:
                pass
            nxt = _parse_reclaim_count(row["error_log"]) + 1
            conn.execute(
                """
                UPDATE apply_queue
                SET status='READY', worker_id=NULL, error_log=?, updated_at=?
                WHERE id=? AND status='CLAIMED'
                """,
                (_with_reclaim_count(row["error_log"], nxt), _now(), row["id"]),
            )
            n += 1
        conn.commit()
    return n


def reclaim_orphaned_claims(
    db_path: Path | None = None,
    *,
    max_reclaims: int | None = None,
) -> dict[str, int]:
    """On cold boot: reclaim every CLAIMED row (previous worker died).

    No schema change — reclaim count lives in ``error_log`` as ``reclaim=N;``.
    Under the max → READY (retry). At/over max → DONE + applications
    ``FAILED_INTERRUPTED``.
    """
    init_queue(db_path)
    limit = reclaim_max() if max_reclaims is None else max(0, int(max_reclaims))
    retried = 0
    abandoned = 0
    ts = _now()
    to_log: list[dict] = []
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, company, job_url, track, jd_text, yaml_snapshot, error_log
            FROM apply_queue WHERE status='CLAIMED'
            """
        ).fetchall()
        for row in rows:
            n = _parse_reclaim_count(row["error_log"])
            if n < limit:
                nxt = n + 1
                conn.execute(
                    """
                    UPDATE apply_queue
                    SET status='READY', worker_id=NULL, error_log=?, updated_at=?
                    WHERE id=? AND status='CLAIMED'
                    """,
                    (_with_reclaim_count(row["error_log"], nxt), ts, row["id"]),
                )
                retried += 1
            else:
                conn.execute(
                    """
                    UPDATE apply_queue
                    SET status='DONE', worker_id=NULL, error_log=?, updated_at=?
                    WHERE id=? AND status='CLAIMED'
                    """,
                    (
                        _with_reclaim_count(
                            f"interrupted;{row['error_log'] or ''}", n
                        ),
                        ts,
                        row["id"],
                    ),
                )
                abandoned += 1
                to_log.append(dict(row))
        conn.commit()

    if to_log:
        from src.core.memory import log_application

        for row in to_log:
            try:
                log_application(
                    str(row["company"] or "Unknown"),
                    str(row["job_url"] or ""),
                    str(row["track"] or ""),
                    "FAILED_INTERRUPTED",
                    error_log="orphaned_claim_reclaim_exhausted",
                    jd_text=str(row["jd_text"] or ""),
                    tailored_yaml_snapshot=str(row["yaml_snapshot"] or ""),
                    db_path=db_path,
                )
            except Exception:
                logger.warning(
                    "FAILED_INTERRUPTED log failed for queue id=%s",
                    row.get("id"),
                    exc_info=True,
                )
    if retried or abandoned:
        logger.info(
            "Orphan reclaim: retried=%s abandoned=%s (max_reclaims=%s)",
            retried,
            abandoned,
            limit,
        )
    return {"retried": retried, "abandoned": abandoned}


def queue_counts(db_path: Path | None = None) -> dict[str, int]:
    init_queue(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM apply_queue GROUP BY status"
        ).fetchall()
    out = {"READY": 0, "CLAIMED": 0, "DONE": 0, "CANCELLED": 0}
    for r in rows:
        out[str(r["status"])] = int(r["n"])
    out["total"] = sum(out.values())
    return out


def ready_count(db_path: Path | None = None) -> int:
    counts = queue_counts(db_path)
    return int(counts.get("READY", 0)) + int(counts.get("CLAIMED", 0))


def company_active_apply_count(
    company: str,
    *,
    db_path: Path | None = None,
) -> int:
    """Count READY/CLAIMED rows for this company (shared build-dir peers)."""
    needle = (company or "").strip().lower()
    if not needle:
        return 0
    init_queue(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT company FROM apply_queue
            WHERE status IN ('READY', 'CLAIMED')
            ORDER BY id DESC
            LIMIT 400
            """
        ).fetchall()
    return sum(
        1
        for row in rows
        if str(row["company"] or "").strip().lower() == needle
    )


def company_has_peer_apply(company: str, *, db_path: Path | None = None) -> bool:
    """True if ≥2 active queue rows share the company (caller is usually one of them)."""
    return company_active_apply_count(company, db_path=db_path) > 1
