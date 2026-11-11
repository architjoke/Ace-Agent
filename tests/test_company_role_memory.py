"""Company + role submit ledger: mark, gate TTL, list."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_mark_submitted_blocks_same_role_allows_other(tmp_path):
    from src.core import memory as mem

    db = tmp_path / "m.db"
    mem.init_db(db)
    mem.mark_company_role_submitted(
        "Acme Labs",
        "SDE Intern",
        job_url="https://example.com/jobs/1",
        db_path=db,
    )
    assert mem.has_recent_company_role_apply("Acme Labs", "SDE Intern", db_path=db)
    assert mem.has_recent_company_role_apply("Acme", "Software Development Intern", db_path=db)
    assert not mem.has_recent_company_role_apply(
        "Acme Labs", "ML Engineer Intern", db_path=db
    )


def test_ttl_expires_allows_new_url(tmp_path):
    from src.core import memory as mem

    db = tmp_path / "m.db"
    mem.init_db(db)
    old = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat(timespec="seconds")
    mem.log_application(
        "Beta Co",
        "https://example.com/jobs/old",
        "SDE",
        "SUCCESS",
        job_title="Backend Engineer",
        db_path=db,
    )
    with mem._connect(db) as conn:
        conn.execute(
            "UPDATE applications SET timestamp=? WHERE job_url=?",
            (old, "https://example.com/jobs/old"),
        )
        conn.commit()
    assert not mem.has_recent_company_role_apply(
        "Beta Co", "Backend Engineer", within_days=30, db_path=db
    )
    # Exact old URL still forever-blocked
    assert mem.has_applied("https://example.com/jobs/old", db_path=db)


def test_manual_action_does_not_block_until_marked(tmp_path):
    from src.core import memory as mem

    db = tmp_path / "m.db"
    mem.init_db(db)
    mem.log_application(
        "Shure",
        "https://careers-shure.icims.com/jobs/5037/job",
        "SDE",
        "MANUAL_ACTION",
        error_log="captcha",
        job_title="Software Engineer Intern",
        db_path=db,
    )
    assert not mem.has_recent_company_role_apply(
        "Shure", "Software Engineer Intern", db_path=db
    )
    mem.mark_company_role_submitted(
        "Shure", "Software Engineer Intern", db_path=db
    )
    assert mem.has_recent_company_role_apply(
        "Shure", "Software Engineer Intern", db_path=db
    )


def test_list_submitted_company_roles(tmp_path):
    from src.core import memory as mem

    db = tmp_path / "m.db"
    mem.init_db(db)
    mem.mark_company_role_submitted("Gamma", "Data Analyst", db_path=db)
    mem.log_application(
        "Gamma",
        "https://example.com/g2",
        "ML",
        "SUCCESS",
        job_title="ML Intern",
        db_path=db,
    )
    rows = mem.list_submitted_company_roles(within_days=30, db_path=db)
    pairs = {(r["company"], r["job_title"]) for r in rows}
    assert ("Gamma", "Data Analyst") in pairs
    assert ("Gamma", "ML Intern") in pairs


def test_mark_creates_ledger_when_no_rows(tmp_path):
    from src.core import memory as mem

    db = tmp_path / "m.db"
    mem.init_db(db)
    summary = mem.mark_company_role_submitted(
        "Brand New Co", "Product Intern", db_path=db
    )
    assert summary["updated"] >= 1
    assert mem.has_recent_company_role_apply(
        "Brand New Co", "Product Intern", db_path=db
    )
    listed = mem.list_submitted_company_roles(db_path=db)
    assert any(r["company"] == "Brand New Co" for r in listed)


def test_empty_role_mark_only_flips_open_packages(tmp_path):
    """Bulk Manual_Action mark must not SUCCESS every historical company row."""
    from src.core import memory as mem

    db = tmp_path / "m.db"
    mem.init_db(db)
    mem.log_application(
        "Acme",
        "https://example.com/old-success",
        "SDE",
        "SUCCESS",
        job_title="Backend Engineer",
        db_path=db,
    )
    mem.log_application(
        "Acme",
        "https://example.com/failed",
        "SDE",
        "FAILED",
        job_title="Platform Engineer",
        db_path=db,
    )
    mem.log_application(
        "Acme",
        "https://example.com/manual",
        "SDE",
        "MANUAL_ACTION",
        job_title="SDE Intern",
        db_path=db,
    )
    summary = mem.mark_company_role_submitted("Acme", "", db_path=db)
    assert summary["updated"] == 1
    with mem._connect(db) as conn:
        statuses = {
            str(r["job_url"]): str(r["status"])
            for r in conn.execute(
                "SELECT job_url, status, error_log FROM applications WHERE company='Acme'"
            )
        }
        logs = {
            str(r["job_url"]): str(r["error_log"] or "")
            for r in conn.execute(
                "SELECT job_url, error_log FROM applications WHERE company='Acme'"
            )
        }
    assert statuses["https://example.com/manual"] == "SUCCESS"
    assert "USER_MARKED_SUBMITTED" in logs["https://example.com/manual"]
    assert statuses["https://example.com/old-success"] == "SUCCESS"
    assert statuses["https://example.com/failed"] == "FAILED"
    assert "USER_MARKED_SUBMITTED" not in logs["https://example.com/old-success"]
