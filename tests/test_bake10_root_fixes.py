"""Bake10 root-cause fixes: greenlet pin, external bridge, portal DNS, triage cache."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_slot_executor_pins_owner_thread():
    from src.core.browser_warmup import WorkerBrowserSlot

    slot = WorkerBrowserSlot(user_data_dir=Path("data/_test_slot_pin"))
    seen: list[int] = []

    def _mark() -> int:
        ident = threading.get_ident()
        seen.append(ident)
        if slot.creator_thread_ident is None:
            slot.creator_thread_ident = ident
        return ident

    a = slot.run(_mark)
    b = slot.run(_mark)
    assert a == b == seen[0] == seen[1]
    assert slot.creator_thread_ident == a
    # Off-thread ensure must refuse
    with pytest.raises(RuntimeError, match="owner thread"):
        slot.ensure()
    slot.shutdown()


def test_portal_load_markers_before_quota():
    from src.core.executor import (
        ExternalPortalLoadError,
        ApplicationTimeoutError,
        _raise_if_blocked,
        is_portal_load_failure,
    )

    assert is_portal_load_failure("net::ERR_NAME_NOT_RESOLVED at careers.example")
    assert is_portal_load_failure("external application portal failed to load")

    hist = SimpleNamespace(
        final_result=lambda: (
            "the external company application site that failed to load (dns/blank page). "
            "also saw 503 unavailable noise"
        ),
        errors=lambda: [],
        judgement=lambda: None,
    )
    with pytest.raises(ExternalPortalLoadError, match="FAILED_EXTERNAL_LOAD"):
        _raise_if_blocked(hist)

    quota = SimpleNamespace(
        final_result=lambda: "503 unavailable high demand resource_exhausted",
        errors=lambda: [],
        judgement=lambda: None,
    )
    with pytest.raises(ApplicationTimeoutError, match="quota/capacity"):
        _raise_if_blocked(quota)


def test_goto_dom_raises_portal_dns(monkeypatch):
    from src.playbooks import _common as common
    from src.core.executor import ExternalPortalLoadError

    page = MagicMock()
    page.url = "about:blank"
    page.goto.side_effect = Exception("net::ERR_NAME_NOT_RESOLVED")

    with pytest.raises(ExternalPortalLoadError, match="FAILED_PORTAL_DNS"):
        common.goto_dom(page, "https://dead.portal.example/jobs/1", force=True)


def test_probe_on_page_rethrows_portal_dns(monkeypatch):
    from src.core import job_closed as jc
    from src.core.executor import ExternalPortalLoadError

    page = MagicMock()

    def _boom(*a, **k):
        raise ExternalPortalLoadError("FAILED_PORTAL_DNS: dead host")

    monkeypatch.setattr("src.playbooks._common.goto_dom", _boom)
    with pytest.raises(ExternalPortalLoadError, match="FAILED_PORTAL_DNS"):
        jc.probe_on_page(page, "https://dead.portal.example/x")


def test_gemini_capacity_excludes_portal_dns():
    from src.core.executor import ExternalPortalLoadError, _is_gemini_capacity_error

    assert _is_gemini_capacity_error(ExternalPortalLoadError("FAILED_PORTAL_DNS: x")) is False
    assert (
        _is_gemini_capacity_error(RuntimeError("net::ERR_NAME_NOT_RESOLVED")) is False
    )
    assert _is_gemini_capacity_error(RuntimeError("503 unavailable high demand")) is True


def test_classify_google_forms_and_blocked_bridge():
    from src.playbooks import classify_ats
    from src.core.discovery import is_blocked_board
    from src.core.executor import _handoff_should_skip_cdp

    assert (
        classify_ats(
            "https://docs.google.com/forms/d/e/1FAIpQLScQzIGDwY3p/viewform"
        )
        == "google_forms"
    )
    assert (
        classify_ats("https://paypal.eightfold.ai/careers/job/123") == "career_site"
    )
    assert (
        classify_ats(
            "https://selah.zohorecruit.in/jobs/Careers/213761000001539342/x"
        )
        == "career_site"
    )
    assert classify_ats("https://careers.bempu.com/jobs/Careers/1/x") == "career_site"
    from src.playbooks.career_site import _APPLY, _clean_career_url

    cleaned = _clean_career_url(
        "https://faveohelpdesk.zohorecruit.in/jobs/Careers/1/x"
        "?source=LinkedIn-Basic&embedsource=LinkedIn%2BLimited%2BListings"
    )
    assert cleaned == "https://faveohelpdesk.zohorecruit.in/jobs/Careers/1/x"
    assert _APPLY.search("I'm interested")
    assert _APPLY.search("Apply Now")
    assert is_blocked_board(
        "https://internshala.com/internship/detail/voice-ai-internship"
    )
    assert _handoff_should_skip_cdp(
        "linkedin playbook: Easy Apply not clickable (main/aria:not_ready)"
    )
    assert _handoff_should_skip_cdp("SKIP_BLOCKED_BOARD: internshala")
    assert not _handoff_should_skip_cdp(
        "linkedin playbook: Next/Review/Submit missing — Gemini"
    )


def test_triage_skips_internshala_in_jd(monkeypatch):
    from src.core import pipeline as pl

    monkeypatch.setattr(pl, "should_suppress_apply", lambda *a, **k: False)
    monkeypatch.setattr(pl, "has_recent_company_role_apply", lambda *a, **k: False)
    job = {
        "company": "Jarvis",
        "title": "Voice AI Intern",
        "track": "AI",
        "job_url": "https://www.linkedin.com/jobs/view/1",
        "description": "Apply via https://internshala.com/internship/detail/foo",
    }
    assert pl.triage_job(job, "master", min_fit=72) is None


def test_linkedin_external_bridge_dispatches_greenhouse(monkeypatch):
    from src.playbooks import linkedin as li

    page = MagicMock()
    ctx = MagicMock()
    btn = MagicMock()
    btn.count.return_value = 1
    btn.is_visible.return_value = True
    btn.get_attribute.return_value = "https://boards.greenhouse.io/acme/jobs/1"
    btn.click = MagicMock()

    page.get_by_role.return_value = MagicMock(
        or_=MagicMock(return_value=MagicMock(or_=MagicMock(return_value=MagicMock(first=btn))))
    )
    # Simpler: stub helpers directly
    monkeypatch.setattr(li, "_external_apply_locator", lambda p: btn)
    monkeypatch.setattr(li, "_page_shows_external_apply", lambda p: True)
    monkeypatch.setattr(li, "_scrape_offsite_apply_url", lambda p: "")
    monkeypatch.setattr(li, "goto_dom", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.playbooks._common.capture_popup_url_then_close",
        lambda *a, **k: "https://boards.greenhouse.io/acme/jobs/1",
    )
    monkeypatch.setattr(
        "src.playbooks._common.close_extra_pages", lambda *a, **k: None
    )
    called = {}

    def _dispatch(kind, *a, **k):
        called["kind"] = kind
        return True

    monkeypatch.setattr("src.playbooks.aggregator._dispatch_ats", _dispatch)
    monkeypatch.setattr(
        "src.playbooks.classify_ats",
        lambda url: "greenhouse" if "greenhouse" in (url or "") else "other",
    )

    pdf = Path("x.pdf")
    master = Path("y.yaml")
    ok = li._try_external_company_bridge(
        page,
        ctx,
        pdf=pdf,
        master=master,
        user_data_dir=None,
        yaml_snapshot=None,
        cdp_assist=None,
    )
    assert ok is True
    assert called.get("kind") == "greenhouse"


def test_fit_triage_cache_roundtrip(tmp_path, monkeypatch):
    from src.core import evaluate as ev

    monkeypatch.setattr(ev, "_TRIAGE_CACHE_PATH", tmp_path / "fit.sqlite")
    monkeypatch.setattr(ev, "_hard_config_reject", lambda jd: None)

    calls = {"n": 0}

    def fake_invoke(role, prompt, **kwargs):
        calls["n"] += 1
        return {
            "fit_score": 88,
            "reason": "strong match",
            "rule_broken": None,
            "matched_skills": ["python"],
            "missing_must_haves": [],
            "recommended_track": "SDE",
            "company_provided_salary": False,
            "salary_min": 0,
            "salary_max": 0,
            "salary_format": "ANNUAL",
            "includes_equity_or_bonus": False,
        }

    monkeypatch.setattr(ev, "invoke_with_failover", fake_invoke)
    monkeypatch.setattr(
        "src.core.salary_engine.compensation_from_fit",
        lambda raw: {
            "company_provided_salary": False,
            "salary_min": 0,
            "salary_max": 0,
            "salary_format": "ANNUAL",
            "includes_equity_or_bonus": False,
            "jd_salary_format": "ANNUAL",
        },
    )
    monkeypatch.setattr(
        "src.core.preferences.apply_preference_gate",
        lambda jd: (None, 0, [], False),
    )

    jd = "Software engineer intern Python FastAPI AI agents"
    a = ev.evaluate_fit(jd, "fact_bank_v1", min_score=72)
    b = ev.evaluate_fit(jd, "fact_bank_v1", min_score=72)
    assert a["passed"] and b["passed"]
    assert a["fit_score"] == 88
    assert calls["n"] == 1  # second call served from SQLite cache
