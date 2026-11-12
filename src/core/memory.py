"""SQLite long-term memory: apply status + recruiter outcome tracking."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.core.settings import get_settings

VALID_STATUSES = frozenset(
    {
        "SUCCESS",
        "FAILED",
        "MANUAL_ACTION",
        "ALREADY_APPLIED",
        "FAILED_TIMEOUT_OR_STUCK",
        "FAILED_INTERRUPTED",
        "REQUIRES_HUMAN_INTERVENTION",
        "REQUIRES_HUMAN_CITIZENSHIP_REVIEW",
        "JOB_CLOSED",
        "TOKEN_LIMIT_REACHED",
    }
)
VALID_OUTCOMES = frozenset(
    {"PENDING", "REJECTED", "INTERVIEW_INVITE", "ASSESSMENT", "UNCLEAR", "SELECTED"}
)


def _db_path(db_path: Path | None = None) -> Path:
    return Path(db_path) if db_path else get_settings().memory_db


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout + WAL: parallel triage/tailor threads + browser workers share this DB
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.Error:
        pass
    return conn


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(applications)").fetchall()
    return {str(r["name"]) for r in rows}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn)
    migrations = {
        "outcome_status": "TEXT NOT NULL DEFAULT 'PENDING'",
        "jd_text": "TEXT",
        "tailored_yaml_snapshot": "TEXT",
        "outcome_reason": "TEXT",
        "outcome_updated_at": "TEXT",
        "action_url": "TEXT",
        "action_deadline": "TEXT",
        "job_title": "TEXT",
    }
    for name, decl in migrations.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {name} {decl}")


def init_db(db_path: Path | None = None) -> Path:
    """Create / migrate the applications table. Returns the DB path."""
    path = _db_path(db_path)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                job_url TEXT NOT NULL UNIQUE,
                track TEXT,
                status TEXT NOT NULL,
                error_log TEXT,
                timestamp TEXT NOT NULL,
                outcome_status TEXT NOT NULL DEFAULT 'PENDING',
                jd_text TEXT,
                tailored_yaml_snapshot TEXT,
                outcome_reason TEXT,
                outcome_updated_at TEXT,
                action_url TEXT,
                action_deadline TEXT
            )
            """
        )
        _ensure_columns(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_applications_status_ts "
            "ON applications(status, timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_applications_outcome "
            "ON applications(outcome_status)"
        )
        conn.commit()
    return path


def has_applied(job_url: str, db_path: Path | None = None) -> bool:
    """Return True if this job_url was already logged (any status).

    Diagnostics / legacy only — triage must use ``should_suppress_apply``.
    """
    url = (job_url or "").strip()
    if not url:
        return False
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM applications WHERE job_url = ? LIMIT 1",
            (url,),
        ).fetchone()
    return row is not None


def has_successful_apply(job_url: str, db_path: Path | None = None) -> bool:
    """True if this job_url already has status SUCCESS."""
    url = (job_url or "").strip()
    if not url:
        return False
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM applications WHERE job_url = ? AND status = 'SUCCESS' LIMIT 1",
            (url,),
        ).fetchone()
    return row is not None


def has_terminal_apply_status(job_url: str, db_path: Path | None = None) -> bool:
    """True when memory says never retry this URL (success / already / closed)."""
    from src.core.apply_policy import SUPPRESS_MEMORY_STATUSES

    url = (job_url or "").strip()
    if not url:
        return False
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM applications WHERE job_url = ? LIMIT 1",
            (url,),
        ).fetchone()
    if not row:
        return False
    return str(row["status"] or "").strip().upper() in SUPPRESS_MEMORY_STATUSES


def count_failed_apply_attempts(job_url: str, db_path: Path | None = None) -> int:
    """Count non-terminal logged attempts for retry budgeting.

    ``applications`` is unique on job_url, so this is 0 or 1 for a failed row.
    Prefer queue ``attempt_count`` when available; this backs triage when the
    queue row is gone.
    """
    from src.core.apply_policy import SUPPRESS_MEMORY_STATUSES

    url = (job_url or "").strip()
    if not url:
        return 0
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, error_log FROM applications WHERE job_url = ? LIMIT 1",
            (url,),
        ).fetchone()
    if not row:
        return 0
    status = str(row["status"] or "").strip().upper()
    if status in SUPPRESS_MEMORY_STATUSES:
        return 0
    # Parse attempt=N from error_log when present; else one failed attempt.
    err = str(row["error_log"] or "")
    m = re.search(r"(?:^|;)\s*attempt=(\d+)\s*(?:;|$)", err, re.I)
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            pass
    if status:
        return 1
    return 0


def should_suppress_apply(job_url: str, db_path: Path | None = None) -> bool:
    """Triage gate: skip URL only for terminal memory or retry exhaustion."""
    from src.core.apply_policy import should_suppress_apply as _policy_suppress

    return _policy_suppress(job_url, db_path=db_path)


def _norm_company(name: str) -> str:
    text = (name or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(
        r"\b(inc|llc|ltd|pvt|private|limited|corp|corporation|technologies|tech|labs|recruiting|careers|team|hr)\b",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _norm_role(title: str) -> str:
    """Normalize job role for company+role matching."""
    from src.core.tailor import normalize_role_title

    return normalize_role_title(title or "")


def _row_age_days(ts_raw: str, *, now: datetime | None = None) -> float | None:
    raw = (ts_raw or "").strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ref = now or datetime.now(timezone.utc)
        return (ref - ts).total_seconds() / 86400.0
    except ValueError:
        return None


def has_successful_company_apply(
    company: str,
    *,
    within_days: int = 30,
    db_path: Path | None = None,
) -> bool:
    """Legacy: SUCCESS/ALREADY_APPLIED for this company within within_days (any role).

    Prefer ``has_recent_company_role_apply`` when a job title is known.
    """
    return has_recent_company_role_apply(
        company, "", within_days=within_days, db_path=db_path
    )


def has_recent_company_role_apply(
    company: str,
    job_title: str,
    *,
    within_days: int = 30,
    db_path: Path | None = None,
) -> bool:
    """True if SUCCESS/ALREADY_APPLIED for same company(+role) within within_days.

    With ``job_title``: match normalized company + role. Empty title: company-wide.
    After TTL, same company+role is allowed again for a new URL (exact URL still
    blocked forever by ``has_applied``).
    """
    needle = _norm_company(company)
    if not needle:
        return False
    role_needle = _norm_role(job_title) if (job_title or "").strip() else ""
    days = max(1, int(within_days))
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            """
            SELECT company, job_title, status, timestamp FROM applications
            WHERE status IN ('SUCCESS', 'ALREADY_APPLIED')
            ORDER BY id DESC
            LIMIT 800
            """
        ).fetchall()
    now = datetime.now(timezone.utc)
    for row in rows:
        if _norm_company(str(row["company"] or "")) != needle:
            continue
        age = _row_age_days(str(row["timestamp"] or ""), now=now)
        in_window = True if age is None else age <= days
        if not in_window:
            continue
        stored_title = str(row["job_title"] or "").strip()
        if role_needle:
            if stored_title:
                if _norm_role(stored_title) == role_needle:
                    return True
                continue
            # Legacy SUCCESS without title still blocks that company while in window
            if str(row["status"] or "").upper() == "SUCCESS":
                return True
            continue
        return True
    return False


def list_submitted_company_roles(
    *,
    within_days: int | None = 30,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[dict]:
    """Distinct submitted company · role rows (SUCCESS), newest first."""
    init_db(db_path)
    days = None if within_days is None else max(1, int(within_days))
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            """
            SELECT company, job_title, timestamp, job_url, status
            FROM applications
            WHERE status = 'SUCCESS'
            ORDER BY id DESC
            LIMIT 800
            """
        ).fetchall()
    now = datetime.now(timezone.utc)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in rows:
        company = str(row["company"] or "").strip() or "Unknown"
        title = str(row["job_title"] or "").strip()
        if days is not None:
            age = _row_age_days(str(row["timestamp"] or ""), now=now)
            if age is not None and age > days:
                continue
        key = (_norm_company(company), _norm_role(title) if title else "")
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "company": company,
                "job_title": title or "(role unknown)",
                "timestamp": str(row["timestamp"] or ""),
                "job_url": str(row["job_url"] or ""),
            }
        )
        if len(out) >= max(1, min(int(limit), 500)):
            break
    return out


def mark_company_role_submitted(
    company: str,
    job_title: str = "",
    *,
    job_url: str = "",
    track: str = "",
    db_path: Path | None = None,
) -> dict:
    """Record user-confirmed submit for company(+role). Returns summary dict."""
    init_db(db_path)
    company_s = (company or "").strip() or "Unknown"
    title_s = (job_title or "").strip()
    needle = _norm_company(company_s)
    role_needle = _norm_role(title_s) if title_s else ""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated_ids: list[int] = []
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            """
            SELECT id, company, job_title, job_url, status FROM applications
            ORDER BY id DESC
            LIMIT 800
            """
        ).fetchall()
        # Empty role (Manual_Action bulk mark): only flip open packages, never
        # every historical row for the company.
        _OPEN_FOR_MARK = frozenset(
            {
                "MANUAL_ACTION",
                "REQUIRES_HUMAN_INTERVENTION",
                "REQUIRES_HUMAN_CITIZENSHIP_REVIEW",
            }
        )
        for row in rows:
            if _norm_company(str(row["company"] or "")) != needle:
                continue
            stored = str(row["job_title"] or "").strip()
            if role_needle and stored and _norm_role(stored) != role_needle:
                continue
            status = str(row["status"] or "").strip().upper()
            if not role_needle and status not in _OPEN_FOR_MARK:
                continue
            from src.core.db_jail import jailed_update_application_status

            jailed_update_application_status(
                conn,
                int(row["id"]),
                status="SUCCESS",
                error_log="USER_MARKED_SUBMITTED",
                job_title=title_s or stored,
            )
            updated_ids.append(int(row["id"]))
            if role_needle:
                break

        if not updated_ids:
            url = (job_url or "").strip()
            if not url:
                slug_c = re.sub(r"[^a-z0-9]+", "-", needle).strip("-") or "company"
                slug_r = role_needle or "role"
                url = f"ace://submitted/{slug_c}/{slug_r}"
            conn.execute(
                """
                INSERT INTO applications (
                    company, job_url, track, status, error_log, timestamp,
                    outcome_status, jd_text, tailored_yaml_snapshot, job_title
                )
                VALUES (?, ?, ?, 'SUCCESS', 'USER_MARKED_SUBMITTED', ?, 'PENDING', '', '', ?)
                ON CONFLICT(job_url) DO UPDATE SET
                    company=excluded.company,
                    track=excluded.track,
                    status='SUCCESS',
                    error_log='USER_MARKED_SUBMITTED',
                    timestamp=excluded.timestamp,
                    job_title=COALESCE(NULLIF(excluded.job_title, ''), applications.job_title)
                """,
                (company_s, url, (track or "").strip(), ts, title_s),
            )
            row = conn.execute(
                "SELECT id FROM applications WHERE job_url = ?", (url,)
            ).fetchone()
            if row:
                updated_ids.append(int(row["id"]))
        conn.commit()
    return {
        "company": company_s,
        "job_title": title_s or "(role unknown)",
        "updated": len(updated_ids),
        "ids": updated_ids,
    }


def log_application(
    company: str,
    job_url: str,
    track: str,
    status: str,
    error_log: str = "",
    *,
    jd_text: str = "",
    tailored_yaml_snapshot: str = "",
    outcome_status: str = "PENDING",
    job_title: str = "",
    db_path: Path | None = None,
) -> None:
    """Record one application execution. Upserts on job_url."""
    status = (status or "").strip().upper()
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")

    outcome = (outcome_status or "PENDING").strip().upper()
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome_status must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}"
        )

    url = (job_url or "").strip()
    if not url:
        raise ValueError("job_url is required")

    title = (job_title or "").strip()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        conn.execute(
            """
            INSERT INTO applications (
                company, job_url, track, status, error_log, timestamp,
                outcome_status, jd_text, tailored_yaml_snapshot, job_title
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_url) DO UPDATE SET
                company=excluded.company,
                track=excluded.track,
                status=excluded.status,
                error_log=excluded.error_log,
                timestamp=excluded.timestamp,
                jd_text=COALESCE(NULLIF(excluded.jd_text, ''), applications.jd_text),
                tailored_yaml_snapshot=COALESCE(
                    NULLIF(excluded.tailored_yaml_snapshot, ''),
                    applications.tailored_yaml_snapshot
                ),
                job_title=COALESCE(NULLIF(excluded.job_title, ''), applications.job_title)
            """,
            (
                (company or "Unknown").strip() or "Unknown",
                url,
                (track or "").strip(),
                status,
                error_log or "",
                ts,
                outcome,
                jd_text or "",
                tailored_yaml_snapshot or "",
                title,
            ),
        )
        conn.commit()


def update_application_outcome(
    company_name: str,
    outcome: str,
    reason: str = "",
    db_path: Path | None = None,
    *,
    action_url: str = "",
    action_deadline: str = "",
) -> int:
    """Fuzzy-match company_name to PENDING applications; set outcome_status.

    Optionally stores action_url / action_deadline (Calendly, HackerRank, …).
    Returns number of rows updated.
    """
    outcome = (outcome or "").strip().upper()
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}"
        )

    needle = _norm_company(company_name)
    if not needle:
        return 0

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    url = (action_url or "").strip() or None
    deadline = (action_deadline or "").strip() or None
    updated = 0
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            """
            SELECT id, company FROM applications
            WHERE outcome_status = 'PENDING'
            ORDER BY id DESC
            """
        ).fetchall()
        for row in rows:
            target = _norm_company(row["company"])
            if not target:
                continue
            if needle in target or target in needle or needle.split()[0] == target.split()[0]:
                conn.execute(
                    """
                    UPDATE applications
                    SET outcome_status = ?,
                        outcome_reason = ?,
                        outcome_updated_at = ?,
                        action_url = COALESCE(?, action_url),
                        action_deadline = COALESCE(?, action_deadline)
                    WHERE id = ?
                    """,
                    (outcome, reason or "", ts, url, deadline, row["id"]),
                )
                updated += 1
                break  # one email → one best match
        conn.commit()
    return updated


def recent_failures(
    hours: int = 24,
    db_path: Path | None = None,
) -> list[dict]:
    """Return FAILED / MANUAL_ACTION rows from the last ``hours`` (UTC)."""
    return _recent_by_status(
        statuses=("FAILED", "MANUAL_ACTION"),
        hours=hours,
        db_path=db_path,
        status_column="status",
    )


def recent_rejections(
    hours: int = 48,
    db_path: Path | None = None,
) -> list[dict]:
    """Return REJECTED outcome rows from the last ``hours`` (UTC)."""
    return _recent_by_status(
        statuses=("REJECTED",),
        hours=hours,
        db_path=db_path,
        status_column="outcome_status",
        ts_column="outcome_updated_at",
    )


def _recent_by_status(
    *,
    statuses: tuple[str, ...],
    hours: int,
    db_path: Path | None,
    status_column: str,
    ts_column: str = "timestamp",
) -> list[dict]:
    cutoff = datetime.now(timezone.utc).timestamp() - max(hours, 1) * 3600
    placeholders = ",".join("?" for _ in statuses)
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            f"""
            SELECT id, company, job_url, track, status, error_log, timestamp,
                   outcome_status, jd_text, tailored_yaml_snapshot,
                   outcome_reason, outcome_updated_at
            FROM applications
            WHERE {status_column} IN ({placeholders})
            ORDER BY id DESC
            """,
            statuses,
        ).fetchall()

    out: list[dict] = []
    for row in rows:
        raw_ts = row[ts_column] or row["timestamp"]
        try:
            ts = datetime.fromisoformat(str(raw_ts))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts.timestamp() < cutoff:
                continue
        except ValueError:
            continue
        out.append(dict(row))
    return out


def application_counts(db_path: Path | None = None) -> dict[str, int]:
    """All-time counts by apply status + unique companies."""
    init_db(db_path)
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM applications GROUP BY status"
        ).fetchall()
        companies = conn.execute(
            "SELECT COUNT(DISTINCT company) FROM applications"
        ).fetchone()[0]
        pending_outcomes = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE outcome_status = 'PENDING'"
        ).fetchone()[0]
    by_status = {str(r["status"]): int(r["n"]) for r in rows}
    staged = by_status.get("SUCCESS", 0)
    manual = by_status.get("MANUAL_ACTION", 0)
    failed = by_status.get("FAILED", 0)
    already = by_status.get("ALREADY_APPLIED", 0)
    stuck = by_status.get("FAILED_TIMEOUT_OR_STUCK", 0)
    interrupted = by_status.get("FAILED_INTERRUPTED", 0)
    human = by_status.get("REQUIRES_HUMAN_INTERVENTION", 0)
    citizen = by_status.get("REQUIRES_HUMAN_CITIZENSHIP_REVIEW", 0)
    closed = by_status.get("JOB_CLOSED", 0)
    return {
        "staged_ready": staged,
        "manual_action": manual,
        "failed": failed,
        "already_applied": already,
        "failed_timeout_or_stuck": stuck,
        "failed_interrupted": interrupted,
        "requires_human_intervention": human,
        "requires_human_citizenship_review": citizen,
        "job_closed": closed,
        "attempted": staged
        + manual
        + failed
        + already
        + stuck
        + interrupted
        + human
        + citizen
        + closed,
        "unique_companies": int(companies or 0),
        "pending_outcomes": int(pending_outcomes or 0),
        **{f"status_{k.lower()}": v for k, v in by_status.items()},
    }


def list_applications(
    *,
    status: str | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[dict]:
    """Newest applications first; optional status filter."""
    init_db(db_path)
    clause = ""
    params: list[object] = []
    if status:
        clause = "WHERE status = ?"
        params.append(status.strip().upper())
    params.append(max(1, min(int(limit), 500)))
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            f"""
            SELECT id, company, job_url, track, status, error_log, timestamp,
                   outcome_status, outcome_reason, job_title
            FROM applications
            {clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def stats_since(hours: int = 24, db_path: Path | None = None) -> dict:
    """Aggregate counts for the daily brief."""
    cutoff = datetime.now(timezone.utc).timestamp() - max(hours, 1) * 3600
    with _connect(db_path) as conn:
        _ensure_columns(conn)
        rows = conn.execute(
            """
            SELECT status, outcome_status, timestamp, outcome_updated_at
            FROM applications
            """
        ).fetchall()

    applied = rejected = invites = assessments = pending = 0
    for row in rows:
        try:
            ts = datetime.fromisoformat(str(row["timestamp"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            in_window = ts.timestamp() >= cutoff
        except ValueError:
            in_window = False

        if in_window and row["status"] in ("SUCCESS", "MANUAL_ACTION", "FAILED"):
            if row["status"] == "SUCCESS":
                applied += 1
            elif row["status"] == "MANUAL_ACTION":
                applied += 1  # staged/fallback still counts as attempted apply path

        outcome = (row["outcome_status"] or "PENDING").upper()
        outcome_ts = row["outcome_updated_at"] or row["timestamp"]
        try:
            ots = datetime.fromisoformat(str(outcome_ts))
            if ots.tzinfo is None:
                ots = ots.replace(tzinfo=timezone.utc)
            outcome_in = ots.timestamp() >= cutoff
        except ValueError:
            outcome_in = False

        if not outcome_in:
            continue
        if outcome == "REJECTED":
            rejected += 1
        elif outcome == "INTERVIEW_INVITE":
            invites += 1
        elif outcome == "ASSESSMENT":
            assessments += 1
        elif outcome == "PENDING":
            pending += 1

    return {
        "applied_today": applied,
        "rejections_today": rejected,
        "interview_invites_today": invites,
        "assessments_today": assessments,
        "pending_outcomes": pending,
    }
