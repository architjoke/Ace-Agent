"""Unit tests for CDP handoff helpers, soft-fail status, and apply_job wiring."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_read_cdp_url_parses_devtools_port(tmp_path):
    from src.playbooks.session import read_cdp_url

    port_file = tmp_path / "DevToolsActivePort"
    port_file.write_text("9333\n/devtools/browser/abc\n", encoding="utf-8")
    assert read_cdp_url(tmp_path, wait_s=0.5) == "http://127.0.0.1:9333"


def test_read_cdp_url_missing(tmp_path):
    from src.playbooks.session import read_cdp_url

    assert read_cdp_url(tmp_path, wait_s=0.2) is None


def test_task_handoff_format():
    from src.core.prompts import TASK_HANDOFF

    text = TASK_HANDOFF.format(
        pdf_abs_path=r"C:\tmp\resume.pdf",
        pdf_stem="resume",
        submitted_token="APPLICATION_SUBMITTED",
    )
    assert "HANDOFF MODE" in text
    assert "Do not navigate away" in text
    assert "APPLICATION_SUBMITTED" in text


def test_browser_profile_cdp_flags():
    from src.core.executor import _browser_profile_cdp

    bp = _browser_profile_cdp("http://127.0.0.1:9222")
    assert bp.cdp_url == "http://127.0.0.1:9222"
    assert bp.is_local is True
    assert bp.keep_alive is True
    assert bp.accept_downloads is False
    assert list(bp.permissions or []) == []


def test_application_counts_includes_human(tmp_path):
    from src.core.memory import application_counts, init_db, log_application

    db = tmp_path / "mem.db"
    init_db(db)
    log_application(
        "Co",
        "http://127.0.0.1/job",
        "swe",
        "REQUIRES_HUMAN_INTERVENTION",
        error_log="soft",
        db_path=db,
    )
    counts = application_counts(db)
    assert counts["requires_human_intervention"] == 1
    assert counts["attempted"] >= 1


def test_discard_modal_clicks_discard():
    from src.playbooks.session import PlaybookSession

    clicks: list[str] = []

    class _Btn:
        def count(self):
            return 1

        def is_visible(self, timeout=0):
            return True

        def click(self, timeout=0):
            clicks.append("discard")

    class _Page:
        def get_by_role(self, role, name=None):
            return SimpleNamespace(first=_Btn())

        def wait_for_timeout(self, ms):
            return None

        @property
        def keyboard(self):
            return SimpleNamespace(press=lambda k: clicks.append(f"key:{k}"))

    session = PlaybookSession(
        context=None,
        page=_Page(),
        user_data_dir=Path("."),
        cdp_url=None,
    )
    session.discard_modal()
    assert clicks == ["discard"]


def test_cdp_handoff_soft_fails_without_port(tmp_path, monkeypatch):
    from src.core.executor import RequiresHumanInterventionError, _execute_cdp_handoff
    from src.playbooks.session import PlaybookSession

    closed = {"n": 0}
    discarded = {"n": 0}
    detached = {"n": 0}

    class _Session(PlaybookSession):
        def discard_modal(self):
            discarded["n"] += 1

        def detach(self):
            detached["n"] += 1

        def close(self):
            closed["n"] += 1

    session = _Session(
        context=None,
        page=None,
        user_data_dir=tmp_path,
        cdp_url=None,
    )
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.1\n%%EOF\n")
    master = tmp_path / "master.yaml"
    master.write_text("cv: {}\napplication_answers: {}\n", encoding="utf-8")

    monkeypatch.setenv("ACE_BROWSER_KEEP_ALIVE", "1")
    with pytest.raises(RequiresHumanInterventionError):
        asyncio.run(
            _execute_cdp_handoff(
                session,
                "stuck",
                "http://127.0.0.1/job",
                pdf,
                master,
            )
        )
    assert discarded["n"] == 1
    assert detached["n"] == 1
    assert closed["n"] == 0


def test_apply_job_requires_human(tmp_path, monkeypatch):
    from src.core.executor import RequiresHumanInterventionError
    from src.core import pipeline

    db = tmp_path / "mem.db"
    monkeypatch.setattr(pipeline, "has_successful_apply", lambda url: False)
    monkeypatch.setattr(
        pipeline,
        "has_recent_company_role_apply",
        lambda company, title="", **kw: False,
    )
    monkeypatch.setattr(pipeline, "is_aggregator_listing", lambda url: False)

    async def _boom(*args, **kwargs):
        raise RequiresHumanInterventionError("modal stuck")

    monkeypatch.setattr(pipeline, "execute_application", _boom)

    logged: list[str] = []

    def _log(company, job_url, track, status, **kwargs):
        logged.append(status)
        return 1

    monkeypatch.setattr(pipeline, "log_application", _log)

    outcome = asyncio.run(
        pipeline.apply_job(
            company="MockCo",
            job_url="http://127.0.0.1/job",
            track="swe",
            jd_text="jd",
            pdf_path=str(tmp_path / "x.pdf"),
            yaml_snapshot="",
            master_yaml=tmp_path / "m.yaml",
        )
    )
    assert outcome == "requires_human"
    assert logged == ["REQUIRES_HUMAN_INTERVENTION"]


def test_desk_mentions_human_soft_fail(monkeypatch):
    from src.core import desk

    monkeypatch.setattr(
        desk,
        "application_counts",
        lambda: {
            "staged_ready": 0,
            "manual_action": 0,
            "failed": 0,
            "already_applied": 0,
            "failed_timeout_or_stuck": 0,
            "requires_human_intervention": 3,
            "attempted": 3,
            "unique_companies": 1,
            "pending_outcomes": 0,
        },
    )
    monkeypatch.setattr(desk, "manual_packages", lambda: [])
    monkeypatch.setattr(desk, "init_db", lambda: None)
    monkeypatch.setattr(
        "src.core.browser_pool.read_worker_status",
        lambda: {"workers": {}},
    )
    monkeypatch.setattr("src.core.queue.init_queue", lambda: None)
    monkeypatch.setattr(
        "src.core.queue.queue_counts",
        lambda: {"READY": 0, "CLAIMED": 0, "DONE": 0},
    )
    text = desk.format_status()
    assert "Needs human (soft-fail):  3" in text
