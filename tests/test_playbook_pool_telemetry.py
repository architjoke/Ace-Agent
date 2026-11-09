"""Deep playbook pooling + CDP assist + telemetry."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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


def test_keep_alive_default_is_one(monkeypatch, tmp_path):
    monkeypatch.delenv("ACE_BROWSER_KEEP_ALIVE", raising=False)
    from src.core import executor as ex

    profile = tmp_path / "browser_profile"
    profile.mkdir()
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

    bp = ex._browser_profile(profile)
    assert bp.keep_alive is True
    assert captured.get("keep_alive") is True


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
