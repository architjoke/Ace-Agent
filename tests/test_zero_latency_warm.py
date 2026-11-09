"""Zero-latency warm path: goto skip, semantic DI, CDP vision, TTL, cycle blank."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_urls_equivalent_greenhouse_utm_slash():
    from src.playbooks._common import urls_equivalent

    base = "https://boards.greenhouse.io/acme/jobs/123"
    assert urls_equivalent(base, base)
    assert urls_equivalent(base + "/", base)
    assert urls_equivalent(base + "?utm_source=li", base)
    assert urls_equivalent(
        "https://job-boards.greenhouse.io/acme/jobs/123/application",
        "https://boards.greenhouse.io/acme/jobs/123",
    )
    assert not urls_equivalent(
        "https://boards.greenhouse.io/acme/jobs/999",
        "https://boards.greenhouse.io/acme/jobs/123",
    )


def test_urls_equivalent_linkedin_www():
    from src.playbooks._common import urls_equivalent

    assert urls_equivalent(
        "https://www.linkedin.com/jobs/view/1",
        "https://linkedin.com/jobs/view/1",
    )


def test_urls_equivalent_lever_ashby_job_id():
    from src.playbooks._common import urls_equivalent

    assert urls_equivalent(
        "https://jobs.lever.co/acme/abc-123",
        "https://jobs.ashbyhq.com/acme/abc-123",
    )
    assert not urls_equivalent(
        "https://jobs.lever.co/acme/abc-123",
        "https://jobs.ashbyhq.com/acme/other-id",
    )


def test_goto_dom_skips_equivalent(monkeypatch):
    from src.playbooks import _common as common

    page = MagicMock()
    page.url = "https://boards.greenhouse.io/acme/jobs/123?utm_source=x"
    page.goto = MagicMock(side_effect=AssertionError("must not goto"))
    common.goto_dom(page, "https://boards.greenhouse.io/acme/jobs/123/")
    page.goto.assert_not_called()


def test_goto_dom_navigates_different_job():
    from src.playbooks import _common as common

    page = MagicMock()
    page.url = "https://boards.greenhouse.io/acme/jobs/111"
    page.goto = MagicMock()
    common.goto_dom(page, "https://boards.greenhouse.io/acme/jobs/222")
    page.goto.assert_called_once()


def test_semantic_warm_no_playwright(monkeypatch, tmp_path):
    from src.core import semantic_dom as sd

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    master = tmp_path / "master.yaml"
    master.write_text("contact:\n  email: a@b.c\n", encoding="utf-8")

    launched = {"n": 0}

    def boom(*a, **k):
        launched["n"] += 1
        raise AssertionError("must not sync_playwright")

    monkeypatch.setattr("playwright.sync_api.sync_playwright", boom)
    monkeypatch.setattr(sd, "env_truthy", lambda *a, **k: True)
    monkeypatch.setattr(sd, "_answers_dict", lambda p: {"email": "a@b.c"})
    monkeypatch.setattr(sd, "_semantic_fill", lambda *a, **k: True)

    page = MagicMock()
    assert sd.try_semantic_apply(
        "https://jobs.example.com/x", pdf, master, page=page
    )
    assert launched["n"] == 0


def test_vision_cdp_prefer_teardown_false(monkeypatch, tmp_path):
    from src.core import executor as ex

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    master = tmp_path / "master.yaml"
    master.write_text("x: 1\n", encoding="utf-8")
    profile = tmp_path / "prof"
    profile.mkdir()

    captured: dict = {}

    async def fake_run(*a, **k):
        captured.update(k)
        return SimpleNamespace(
            is_successful=lambda: True,
            final_result=lambda: "APPLICATION_SUBMITTED",
            is_done=lambda: True,
        )

    monkeypatch.setattr(ex, "_run_agent", fake_run)
    monkeypatch.setattr(ex, "_browser_profile_cdp", lambda u: object())
    monkeypatch.setattr(
        "src.playbooks.session.read_cdp_url", lambda *a, **k: "http://127.0.0.1:9222"
    )
    monkeypatch.setattr(ex, "_reached_token", lambda *a, **k: True)
    monkeypatch.setattr(ex, "_application_answers_block", lambda *a, **k: "")
    monkeypatch.setattr(ex, "load_lessons", lambda: "")
    monkeypatch.setattr(ex, "_step_budget", lambda u: (5, 60))
    monkeypatch.setattr(
        "src.core.browser_warmup.get_or_create_slot",
        lambda d: SimpleNamespace(reset=lambda: None),
    )
    monkeypatch.setenv("ACE_BROWSER_KEEP_ALIVE", "1")

    ok = asyncio.run(
        ex._execute_with_browser_use(
            "https://jobs.example.com/x",
            pdf,
            master,
            auto_submit=True,
            cleanup_linkedin_resume=False,
            user_data_dir=profile,
            prefer_cdp=True,
        )
    )
    assert ok is True
    assert captured.get("teardown") is False
    assert captured.get("browser_profile") is not None


def test_slot_ttl_relaunches(monkeypatch, tmp_path):
    from src.core import browser_warmup as bw

    launches = {"n": 0}

    monkeypatch.setattr(bw, "browser_alive", lambda c: c is not None)
    monkeypatch.setenv("ACE_BROWSER_SLOT_TTL_HOURS", "0.0001")  # 0.36s

    class _PW:
        def start(self):
            return self

        def stop(self):
            pass

    def fake_launch(pw, **k):
        launches["n"] += 1
        return object(), object()

    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright", lambda: _PW()
    )
    monkeypatch.setattr(
        "src.playbooks._common.launch_persistent_context", fake_launch
    )
    monkeypatch.setattr(
        "src.playbooks._common.close_context_safe", lambda *a, **k: None
    )

    slot = bw.WorkerBrowserSlot(user_data_dir=tmp_path)
    slot.context = object()
    slot.page = object()
    slot.playwright = _PW()
    slot.born_at = time.monotonic() - 10.0  # aged past TTL
    ctx, page = slot.ensure()
    assert launches["n"] == 1
    assert ctx is not None and page is not None
    assert slot.born_at is not None


def test_blank_all_slots_keeps_registry(monkeypatch, tmp_path):
    from src.core import browser_warmup as bw

    key = bw.slot_key(tmp_path)
    slot = bw.WorkerBrowserSlot(user_data_dir=Path(key))
    slot.context = object()
    slot.page = object()
    slot.born_at = time.monotonic()
    resets = {"n": 0}

    monkeypatch.setattr(bw, "browser_alive", lambda c: True)

    def fake_reset(self):
        resets["n"] += 1

    monkeypatch.setattr(bw.WorkerBrowserSlot, "reset", fake_reset)

    with bw._SLOTS_LOCK:
        bw._SLOTS.clear()
        bw._SLOTS[key] = slot

    n = bw.blank_all_slots()
    assert n == 1
    assert resets["n"] == 1
    with bw._SLOTS_LOCK:
        assert key in bw._SLOTS
        bw._SLOTS.clear()


def test_blank_slot_preserves_registry(monkeypatch, tmp_path):
    from src.core import browser_warmup as bw

    key = bw.slot_key(tmp_path)
    slot = bw.WorkerBrowserSlot(user_data_dir=Path(key))
    slot.context = object()
    slot.page = object()
    resets = {"n": 0}
    monkeypatch.setattr(bw, "browser_alive", lambda c: True)
    monkeypatch.setattr(
        bw.WorkerBrowserSlot, "reset", lambda self: resets.__setitem__("n", resets["n"] + 1)
    )
    with bw._SLOTS_LOCK:
        bw._SLOTS.clear()
        bw._SLOTS[key] = slot
    assert bw.blank_slot(tmp_path) is True
    assert resets["n"] == 1
    with bw._SLOTS_LOCK:
        assert key in bw._SLOTS
        assert bw._SLOTS[key].context is not None
        bw._SLOTS.clear()


def test_pool_exit_blanks_not_invalidate(monkeypatch, tmp_path):
    """Normal worker-loop exit must soft-blank when keep-alive is on."""
    import asyncio
    from src.core import browser_pool as bp

    calls = {"blank": 0, "invalidate": 0}
    monkeypatch.setenv("ACE_BROWSER_KEEP_ALIVE", "1")
    monkeypatch.setattr(
        "src.core.browser_warmup.blank_slot",
        lambda p: calls.__setitem__("blank", calls["blank"] + 1) or True,
    )
    monkeypatch.setattr(
        "src.core.browser_warmup.invalidate_slot",
        lambda p: calls.__setitem__("invalidate", calls["invalidate"] + 1),
    )
    monkeypatch.setattr(bp, "worker_profile_dir", lambda i: tmp_path / f"w{i}")
    monkeypatch.setattr(bp, "update_one_worker", lambda *a, **k: None)
    monkeypatch.setattr(bp, "clear_singleton_locks", lambda *a, **k: None)

    async def _noop_apply(**k):
        return "skipped"

    done = asyncio.Event()
    done.set()
    stop = asyncio.Event()
    stop.set()

    asyncio.run(
        bp.browser_worker_loop(
            worker_id="1",
            master_yaml=tmp_path / "m.yaml",
            apply_fn=_noop_apply,
            stop_event=stop,
            producers_done=done,
        )
    )
    assert calls["blank"] == 1
    assert calls["invalidate"] == 0
