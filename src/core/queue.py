"""SQLite apply queue: READY → CLAIMED → DONE (feeds parallel browser workers)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.core.memory import _connect, init_db

QUEUE_STATUSES = frozenset({"READY", "CLAIMED", "DONE", "CANCELLED"})


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


def claim_next(worker_id: str, db_path: Path | None = None) -> dict | None:
    """Atomically claim the oldest READY row for a browser worker."""
    init_queue(db_path)
    ts = _now()
    wid = (worker_id or "w").strip() or "w"
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM apply_queue
            WHERE status = 'READY'
            ORDER BY id ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE apply_queue
            SET status='CLAIMED', worker_id=?, updated_at=?
            WHERE id=? AND status='READY'
            """,
            (wid, ts, row["id"]),
        )
        if conn.total_changes == 0:
            conn.commit()
            return None
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM apply_queue WHERE id = ?", (row["id"],)
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


def requeue_stale_claims(max_age_sec: int = 3600, db_path: Path | None = None) -> int:
    """Return CLAIMED rows stuck too long back to READY (crashed workers)."""
    init_queue(db_path)
    cutoff = datetime.now(timezone.utc).timestamp() - max(60, int(max_age_sec))
    n = 0
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, updated_at FROM apply_queue WHERE status='CLAIMED'"
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
            conn.execute(
                """
                UPDATE apply_queue
                SET status='READY', worker_id=NULL, updated_at=?
                WHERE id=? AND status='CLAIMED'
                """,
                (_now(), row["id"]),
            )
            n += 1
        conn.commit()
    return n


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
