"""Status-only SQLite mutations for Ace Copilot chat tools.

Chat must never DROP/DELETE application rows or run arbitrary SQL.
Only allowlisted column updates for user-confirmed submit / undo.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

# Chat-facing status transitions only.
ALLOWED_CHAT_STATUSES = frozenset({"SUCCESS", "MANUAL_ACTION"})
ALLOWED_CHAT_ERROR_LOGS = frozenset(
    {
        "USER_MARKED_SUBMITTED",
        "REVERTED_BULK_MARK_MISTAKE",
    }
)


def jailed_update_application_status(
    conn: sqlite3.Connection,
    app_id: int,
    *,
    status: str,
    error_log: str,
    job_title: str | None = None,
) -> None:
    """UPDATE status/error_log/timestamp/(optional job_title) for one row only."""
    st = (status or "").strip().upper()
    err = (error_log or "").strip()
    if st not in ALLOWED_CHAT_STATUSES:
        raise PermissionError(f"DB jail: status {st!r} not allowlisted for chat")
    if err not in ALLOWED_CHAT_ERROR_LOGS:
        raise PermissionError(f"DB jail: error_log {err!r} not allowlisted for chat")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if job_title is not None:
        conn.execute(
            """
            UPDATE applications
            SET status=?,
                error_log=?,
                timestamp=?,
                job_title=COALESCE(NULLIF(?, ''), job_title)
            WHERE id=?
            """,
            (st, err, ts, job_title, int(app_id)),
        )
    else:
        conn.execute(
            """
            UPDATE applications
            SET status=?, error_log=?, timestamp=?
            WHERE id=?
            """,
            (st, err, ts, int(app_id)),
        )


def assert_no_destructive_sql(sql: str) -> None:
    """Defense-in-depth if a future helper builds dynamic SQL."""
    low = (sql or "").lower()
    for needle in ("drop ", "delete from", "truncate ", "alter ", "attach "):
        if needle in low:
            raise PermissionError(f"DB jail: forbidden SQL fragment {needle!r}")
