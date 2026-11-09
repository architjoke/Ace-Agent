"""SQLite apply queue: READY → CLAIMED → DONE (feeds parallel browser workers)."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from src.core.memory import _connect, init_db

logger = logging.getLogger(__name__)

QUEUE_STATUSES = frozenset({"READY", "CLAIMED", "DONE", "CANCELLED"})
_RECLAIM_RE = re.compile(r"(?:^|;)\s*reclaim=(\d+)\s*(?:;|$)", re.I)


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
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_apply_queue_status "
            "ON apply_queue(status, id)"
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
    db_path: Path | None = None,
) -> int:
    """Insert or refresh a READY apply item. Returns row id."""
    init_queue(db_path)
    url = (job_url or "").strip()
    ts = _now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO apply_queue (
                company, job_url, track, jd_text, pdf_path, yaml_snapshot,
                status, worker_id, error_log, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'READY', NULL, '', ?, ?)
            ON CONFLICT(job_url) DO UPDATE SET
                company=excluded.company,
                track=excluded.track,
                jd_text=excluded.jd_text,
                pdf_path=excluded.pdf_path,
                yaml_snapshot=excluded.yaml_snapshot,
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
            ),
        )
        row = conn.execute(
            "SELECT id FROM apply_queue WHERE job_url = ?", (url,)
        ).fetchone()
        conn.commit()
    return int(row["id"]) if row else 0


CLAIM_PEEK_LIMIT = 25
WAIT_SENTINEL_KEY = "__wait__"


def claim_next(
    worker_id: str,
    db_path: Path | None = None,
    *,
    host_limiter: object | None = None,
) -> dict | None:
    """Atomically claim a READY row, preferring hosts that are off cooldown.

    When ``host_limiter`` is set and every peeked READY host is cooling down,
    returns ``{"__wait__": seconds}`` without claiming a row so workers can
    sleep and retry (domain interleaving).
    """
    init_queue(db_path)
    ts = _now()
    wid = (worker_id or "w").strip() or "w"
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT * FROM apply_queue
            WHERE status = 'READY'
            ORDER BY id ASC
            LIMIT ?
            """,
            (CLAIM_PEEK_LIMIT,),
        ).fetchall()
        if not rows:
            conn.commit()
            return None

        pick = None
        best_wait: float | None = None
        if host_limiter is None:
            pick = rows[0]
        else:
            ready_in = getattr(host_limiter, "ready_in", None)
            if not callable(ready_in):
                pick = rows[0]
            else:
                for row in rows:
                    wait = float(ready_in(str(row["job_url"] or "")) or 0.0)
                    if wait <= 0:
                        pick = row
                        break
                    best_wait = wait if best_wait is None else min(best_wait, wait)
                if pick is None:
                    conn.commit()
                    return {WAIT_SENTINEL_KEY: float(best_wait if best_wait is not None else 1.0)}

        conn.execute(
            """
            UPDATE apply_queue
            SET status='CLAIMED', worker_id=?, updated_at=?
            WHERE id=? AND status='READY'
            """,
            (wid, ts, pick["id"]),
        )
        if conn.total_changes == 0:
            conn.commit()
            return None
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM apply_queue WHERE id = ?", (pick["id"],)
        ).fetchone()
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
