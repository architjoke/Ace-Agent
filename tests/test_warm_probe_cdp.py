"""Warm-slot probe IoC + non-destructive CDP detach."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_probe_on_page_no_launch(monkeypatch):
    from src.core import job_closed as jc

    launched = {"n": 0}

    def boom(*a, **k):
        launched["n"] += 1
        raise AssertionError("must not launch")

    monkeypatch.setattr("playwright.sync_api.sync_playwright", boom)
    monkeypatch.setattr("src.playbooks._common.goto_dom", lambda *a, **k: None)
    monkeypatch.setattr(jc, "page_shows_job_closed", lambda *a, **k: False)
    monkeypatch.setattr(
        "src.core.already_applied.page_shows_already_applied", lambda *a, **k: False
    )

    page = MagicMock()
    assert jc.probe_on_page(page, "https://jobs.example.com/x") == "ok"
    assert launched["n"] == 0
    assert jc.probe_job_page("https://jobs.example.com/x", page=page) == "ok"


def test_resolve_ats_on_warm_page(monkeypatch):
    from src.playbooks import resolve_ats_kind

    page = MagicMock()
    page.url = "https://boards.greenhouse.io/acme/jobs/1"
    assert resolve_ats_kind("https://careers.example.com/x", page=page) == "greenhouse"


def test_session_detach_keeps_refs(monkeypatch):
    from src.playbooks.session import PlaybookSession

    ctx, page, pw = object(), object(), object()
    monkeypatch.setattr(
        "src.core.browser_warmup.reset_context_state",
        lambda c, p: p,
    )
    s = PlaybookSession(context=ctx, page=page, user_data_dir=Path("."), playwright=pw)
    s.detach()
    assert s.context is ctx and s.playwright is pw


def test_cdp_handoff_detach_not_close(monkeypatch):
    from src.core import executor as ex

    closed, detached, invalidated = {"n": 0}, {"n": 0}, {"n": 0}

    class _S:
        user_data_dir = Path(".")
        cdp_url = "http://127.0.0.1:9222"
        playwright = object()

        def discard_modal(self):
            pass

        def detach(self):
            detached["n"] += 1

        def close(self):
            closed["n"] += 1

    async def fake_agent(*a, **k):
        return [SimpleNamespace(result="APPLICATION_SUBMITTED")]

    monkeypatch.setattr(ex, "_run_agent", fake_agent)
    monkeypatch.setattr(ex, "_browser_profile_cdp", lambda u: object())
    monkeypatch.setattr(ex, "_reached_token", lambda *a, **k: True)
    monkeypatch.setattr(ex, "_application_answers_block", lambda *a, **k: "")
    monkeypatch.setattr(ex, "_step_budget", lambda u: (5, 60))
    monkeypatch.setattr(
        "src.core.browser_warmup.invalidate_slot",
        lambda *a, **k: invalidated.__setitem__("n", invalidated["n"] + 1),
    )
    monkeypatch.setenv("ACE_BROWSER_KEEP_ALIVE", "1")

    ok = asyncio.run(
        ex._execute_cdp_handoff(
            _S(),
            "stuck",
            "https://jobs.example.com/x",
            Path("r.pdf"),
            Path("m.yaml"),
        )
    )
    assert ok is True
    assert detached["n"] == 1
    assert closed["n"] == 0
    assert invalidated["n"] == 0
