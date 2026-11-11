"""Warm Chromium path: keep-alive, probe/CDP, goto skip, pool telemetry."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakePage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False
        self.gotos: list[str] = []
        self.evaluated = False

    def close(self) -> None:
        self.closed = True

    def evaluate(self, _script: str) -> None:
        self.evaluated = True

    def goto(self, url: str, **_kwargs) -> None:
        self.gotos.append(url)


class _FakeContext:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages
        self.clear_cookies_called = False
        self._created: list[_FakePage] = []

    def clear_cookies(self) -> None:
        self.clear_cookies_called = True

    def new_page(self) -> _FakePage:
        p = _FakePage("new")
        self.pages.append(p)
        self._created.append(p)
        return p


def test_reset_closes_extras_never_clear_cookies():
    from src.core.browser_warmup import reset_context_state

    main = _FakePage("main")
    extra = _FakePage("extra")
    ctx = _FakeContext([main, extra])
    out = reset_context_state(ctx, main)
    assert out is main
    assert extra.closed is True
    assert main.closed is False
    assert main.evaluated is True
    assert main.gotos == ["about:blank"]
    assert ctx.clear_cookies_called is False


def test_reset_raises_when_context_missing():
    from src.core.browser_warmup import reset_context_state

    with pytest.raises(RuntimeError, match="unhealthy"):
        reset_context_state(None, None)


def _patch_browser_profile(monkeypatch, ex, profile):
    monkeypatch.setattr(
        ex,
        "get_settings",
        lambda: SimpleNamespace(
            browser_user_data_dir=str(profile),
            browser_profile_directory="Default",
            browser_headless=True,
        ),
    )
    captured: dict = {}

    class _BP:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

    monkeypatch.setattr(ex, "BrowserProfile", _BP)
    monkeypatch.setattr("src.core.network.browser_proxy_settings", lambda: None)
    monkeypatch.setattr(
        "src.core.browser_identity.pinned_user_agent", lambda **kw: "UA"
    )
    monkeypatch.setattr(
        "src.core.browser_identity.pinned_viewport",
        lambda: {"width": 1920, "height": 1080},
    )
    monkeypatch.setattr(
        "src.core.browser_identity.pinned_device_scale_factor", lambda: 1
    )
    return captured


def test_browser_profile_keep_alive_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ACE_BROWSER_KEEP_ALIVE", "1")
    from src.core import executor as ex

    profile = tmp_path / "browser_profile"
    profile.mkdir()
    captured = _patch_browser_profile(monkeypatch, ex, profile)

    bp = ex._browser_profile(profile)
    assert bp.keep_alive is True
    assert captured.get("accept_downloads") is False


def test_browser_profile_keep_alive_default_on(monkeypatch, tmp_path):
    monkeypatch.delenv("ACE_BROWSER_KEEP_ALIVE", raising=False)
    from src.core import executor as ex

    profile = tmp_path / "browser_profile"
    profile.mkdir()
    _patch_browser_profile(monkeypatch, ex, profile)
    bp = ex._browser_profile(profile)
    assert bp.keep_alive is True


def test_worker_setdefault_does_not_override_explicit_zero(monkeypatch):
    monkeypatch.setenv("ACE_BROWSER_KEEP_ALIVE", "0")
    # Mirror worker loop pin capture
    user_pin = (
        __import__("os").environ["ACE_BROWSER_KEEP_ALIVE"]
        if "ACE_BROWSER_KEEP_ALIVE" in __import__("os").environ
        else None
    )
    __import__("os").environ.setdefault("ACE_BROWSER_KEEP_ALIVE", "1")
    assert user_pin == "0"
    assert __import__("os").environ["ACE_BROWSER_KEEP_ALIVE"] == "0"

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


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

    # job_url must classify as "other" so warm page.url is consulted.
    page = MagicMock()
    page.url = "https://boards.greenhouse.io/acme/jobs/1"
    assert resolve_ats_kind("https://example.org/openings/x", page=page) == "greenhouse"


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
        "src.playbooks.session.cdp_endpoint_alive", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "src.playbooks.session.read_live_cdp_url",
        lambda *a, **k: "http://127.0.0.1:9222",
    )
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

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


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
    monkeypatch.setattr(
        "src.playbooks.session.cdp_endpoint_alive", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "src.playbooks.session.read_live_cdp_url",
        lambda *a, **k: "http://127.0.0.1:9222",
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

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clear_slots():
    from src.core import browser_warmup as bw

    bw._SLOTS.clear()
    yield
    for slot in list(bw._SLOTS.values()):
        try:
            slot.close()
        except Exception:
            pass
    bw._SLOTS.clear()


def test_telemetry_time_stage_writes_jsonl(tmp_path, monkeypatch):
    from src.core import telemetry as tel

    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(tel, "_TELEMETRY_FILE", path)
    monkeypatch.setattr(tel, "ensure_runtime_dirs", lambda: None)

    with tel.time_stage("playbook_execution_ms", kind="lever"):
        pass

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["stage"] == "playbook_execution_ms"
    assert row["ok"] is True
    assert row["kind"] == "lever"
    assert "ms" in row


def test_slot_reuse_skips_second_launch(tmp_path, monkeypatch):
    from src.core.browser_warmup import WorkerBrowserSlot, get_or_create_slot

    launches = {"n": 0}
    fake_ctx = MagicMock()
    fake_page = MagicMock()
    fake_ctx.pages = [fake_page]

    def fake_launch(pw, **kwargs):
        launches["n"] += 1
        return fake_ctx, fake_page

    monkeypatch.setattr(
        "src.playbooks._common.launch_persistent_context", fake_launch
    )

    class _PW:
        def stop(self):
            pass

    class _SP:
        def start(self):
            return _PW()

    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright", lambda: _SP()
    )
    monkeypatch.setattr(
        "src.core.browser_warmup.browser_alive", lambda ctx: ctx is fake_ctx
    )

    profile = tmp_path / "prof"
    profile.mkdir()
    slot = get_or_create_slot(profile)
    c1, p1 = slot.ensure()
    c2, p2 = slot.ensure()
    assert launches["n"] == 1
    assert c1 is c2 is fake_ctx
    assert p1 is p2 is fake_page
    assert isinstance(slot, WorkerBrowserSlot)


def test_playbook_injected_page_skips_owned_launch(tmp_path, monkeypatch):
    """Warm path: context/page injected → no sync_playwright().start()."""
    from src.playbooks import lever as lev

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    master = tmp_path / "master.yaml"
    master.write_text(
        "cv:\n  name: Ada Lovelace\nanswers:\n  email: ada@example.com\n  phone: '1'\n",
        encoding="utf-8",
    )

    started = {"n": 0}

    class _Boom:
        def start(self):
            started["n"] += 1
            raise AssertionError("owned launch should not run")

    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _Boom())
    monkeypatch.setattr(lev, "load_applicant_profile", lambda *a, **k: {
        "full_name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "1",
        "linkedin": "",
        "github": "",
    })
    monkeypatch.setattr("src.playbooks._common.goto_dom", lambda *a, **k: None)
    monkeypatch.setattr(
        "src.core.job_closed.page_shows_job_closed", lambda *a, **k: False
    )
    monkeypatch.setattr(lev, "_click_apply", lambda *a, **k: None)
    monkeypatch.setattr(lev, "_fill_lever", lambda *a, **k: True)
    monkeypatch.setattr(lev, "upload_resume_file", lambda *a, **k: True)
    monkeypatch.setattr(lev, "_unanswered_custom", lambda *a, **k: False)
    monkeypatch.setattr(lev, "_submit", lambda *a, **k: True)
    monkeypatch.setattr(lev, "success_text", lambda *a, **k: True)
    monkeypatch.setattr(
        "src.core.browser_warmup.reset_context_state",
        lambda ctx, page: page,
    )

    page = MagicMock()
    page.inner_text.return_value = "thank you"
    page.wait_for_timeout = lambda *_: None
    ctx = MagicMock()

    ok = lev.run_lever(
        "https://jobs.lever.co/acme/x",
        pdf,
        master,
        context=ctx,
        page=page,
    )
    assert ok is True
    assert started["n"] == 0


def test_cdp_assist_returns_bool_without_close(monkeypatch):
    import asyncio
    from src.core import executor as ex

    closed = {"n": 0}

    class _Session:
        user_data_dir = Path(".")
        cdp_url = "http://127.0.0.1:9222"
        playwright = None

        def close(self):
            closed["n"] += 1

        def discard_modal(self):
            pass

    async def fake_agent(*args, **kwargs):
        return []

    monkeypatch.setattr(ex, "_run_agent", fake_agent)
    monkeypatch.setattr(ex, "_browser_profile_cdp", lambda url: object())
    monkeypatch.setattr(
        "src.playbooks.session.cdp_endpoint_alive", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "src.playbooks.session.read_live_cdp_url",
        lambda *a, **k: "http://127.0.0.1:9222",
    )

    async def _run():
        ok = await ex._execute_cdp_assist(_Session(), "stuck field", max_steps=2)
        assert ok is True
        assert closed["n"] == 0

    asyncio.run(_run())


def test_make_cdp_assist_bool(monkeypatch):
    import asyncio
    from src.core.executor import make_cdp_assist

    async def fake_assist(session, reason, *, max_steps=2, timeout_sec=60):
        assert max_steps == 2
        assert "stuck" in reason
        return True

    monkeypatch.setattr(
        "src.core.executor._execute_cdp_assist", fake_assist
    )

    async def _main():
        loop = asyncio.get_running_loop()
        assist = make_cdp_assist(loop, lambda: object())
        # Run sync callback off the loop thread via to_thread
        return await asyncio.to_thread(assist, "stuck: custom")

    assert asyncio.run(_main()) is True

def test_close_all_slots(tmp_path):
    from src.core import browser_warmup as bw

    bw._SLOTS.clear()
    key = str((tmp_path / "prof").resolve())
    (tmp_path / "prof").mkdir()
    slot = bw.WorkerBrowserSlot(user_data_dir=Path(key))
    slot.context = object()
    slot.page = object()
    slot.playwright = None
    bw._SLOTS[key] = slot

    n = bw.close_all_slots()
    assert n == 1
    assert bw._SLOTS == {}
    assert slot.context is None
    assert slot.page is None
