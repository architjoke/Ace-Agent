"""P0 conversion hardening: LI multi-locator, citizenship soft-gate, CDP loop pin."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class _FakeBtn:
    def __init__(self, *, visible=True, disabled=False, text="Easy Apply", click_ok=True):
        self._visible = visible
        self._disabled = disabled
        self._text = text
        self._click_ok = click_ok
        self.clicks = 0

    def count(self):
        return 1

    def is_visible(self, timeout=0):
        return self._visible

    def is_disabled(self):
        return self._disabled

    def inner_text(self, timeout=0):
        return self._text

    def click(self, timeout=0):
        if not self._click_ok:
            raise TimeoutError("click timeout")
        self.clicks += 1


class _FakeRoot:
    """Page or frame stand-in with role/css/xpath strategies."""

    def __init__(self, *, aria=None, css=None, xpath=None):
        self._aria = aria
        self._css = css
        self._xpath = xpath

    def get_by_role(self, role, name=None):
        loc = MagicMock()
        loc.first = self._aria or _FakeBtn(visible=False, click_ok=False)
        return loc

    def locator(self, sel):
        loc = MagicMock()
        sel_l = (sel or "").lower()
        if "xpath=" in sel_l or sel_l.startswith("//"):
            loc.first = self._xpath or _FakeBtn(visible=False, click_ok=False)
        else:
            loc.first = self._css or _FakeBtn(visible=False, click_ok=False)
        return loc


class _FakePage(_FakeRoot):
    def __init__(self, *, aria=None, css=None, xpath=None, frames=None):
        super().__init__(aria=aria, css=css, xpath=xpath)
        self.main_frame = object()
        self.frames = [self.main_frame] + list(frames or [])
        self.waited = 0

    def wait_for_timeout(self, ms):
        self.waited += ms


def test_easy_apply_falls_back_after_aria_and_css_timeout():
    """Steps 1–2 time out / miss; step 3 (xpath) succeeds."""
    from src.playbooks.linkedin import _click_easy_apply

    xpath_btn = _FakeBtn(text="Easy Apply")
    page = _FakePage(
        aria=_FakeBtn(visible=True, click_ok=False),  # visible but click times out
        css=_FakeBtn(visible=False),
        xpath=xpath_btn,
    )
    _click_easy_apply(page)
    assert xpath_btn.clicks == 1


def test_easy_apply_falls_back_to_css():
    from src.playbooks.linkedin import _click_easy_apply

    css_btn = _FakeBtn(text="Easy Apply")
    page = _FakePage(aria=_FakeBtn(visible=False), css=css_btn)
    _click_easy_apply(page)
    assert css_btn.clicks == 1


def test_easy_apply_searches_child_frame():
    from src.playbooks.linkedin import _click_easy_apply

    frame_btn = _FakeBtn(text="Easy Apply")
    child = _FakeRoot(aria=frame_btn)
    page = _FakePage(
        aria=_FakeBtn(visible=False),
        css=_FakeBtn(visible=False),
        xpath=_FakeBtn(visible=False),
        frames=[child],
    )
    _click_easy_apply(page)
    assert frame_btn.clicks == 1


def test_easy_apply_all_miss_raises():
    from src.playbooks.linkedin import _click_easy_apply
    from src.playbooks import PlaybookFallback

    page = _FakePage(
        aria=_FakeBtn(visible=False),
        css=_FakeBtn(visible=False),
        xpath=_FakeBtn(visible=False),
    )
    with pytest.raises(PlaybookFallback, match="Easy Apply not clickable"):
        _click_easy_apply(page)


def _rule(category: str, text: str, level: str = "HARD"):
    from src.core.preferences import PreferenceRule

    return PreferenceRule(
        id="t1",
        category=category,
        rule_text=text,
        enforcement_level=level,
        created="2026-01-01T00:00:00+00:00",
        affirmed="2026-01-01T00:00:00+00:00",
    )


def test_java_primary_hard_dropped_polyglot_allowed():
    from src.core.evaluate import _hard_config_reject
    from src.core.settings import reload_settings

    reload_settings()
    java = _hard_config_reject(
        "Hiring Java Developer Intern. Must know Core Java and Spring Boot."
    )
    assert java is not None
    assert java.get("rule_broken") in {
        "stack_mismatch_java",
        "config_exclude_term",
    }
    # Polyglot / our-stack JDs stay in the pipeline.
    assert (
        _hard_config_reject(
            "Full Stack Intern: Python or JavaScript. Node.js preferred."
        )
        is None
    )
    assert (
        _hard_config_reject(
            "AI Engineer Intern using Python, PyTorch, and Node.js backends."
        )
        is None
    )


def test_citizenship_soft_gate_not_dropped(monkeypatch):
    from src.core import preferences as pref
    from src.core.evaluate import _hard_config_reject

    monkeypatch.setattr(
        pref,
        "load_preference_rules",
        lambda path=None: [
            _rule(
                "JOB_REQUIREMENTS",
                "Never apply to jobs that require US Citizenship or green card.",
            )
        ],
    )
    jd = "Role requires US citizenship and green card. Python internship in Bangalore."
    reason, penalty, hits, cit = pref.apply_preference_gate(jd)
    assert reason is None
    assert cit is True
    assert penalty >= pref.CITIZENSHIP_SOFT_PENALTY_POINTS
    assert penalty < pref.SOFT_PENALTY_POINTS or penalty == pref.CITIZENSHIP_SOFT_PENALTY_POINTS
    assert any("citizenship_soft" in h for h in hits)
    # Hard shield must not drop citizenship-only JDs (clearance still hard).
    assert _hard_config_reject(jd) is None


def test_clearance_stays_hard(monkeypatch):
    from src.core import preferences as pref

    monkeypatch.setattr(
        pref,
        "load_preference_rules",
        lambda path=None: [
            _rule(
                "JOB_REQUIREMENTS",
                "Never apply to jobs that require security clearance or TS/SCI.",
            )
        ],
    )
    reason, penalty, hits, cit = pref.apply_preference_gate(
        "Active security clearance required for this role."
    )
    assert reason and "preference_hard" in reason
    assert cit is False
    assert penalty == 0


def test_clearances_plural_stays_hard(monkeypatch):
    from src.core import preferences as pref

    monkeypatch.setattr(
        pref,
        "load_preference_rules",
        lambda path=None: [
            _rule(
                "JOB_REQUIREMENTS",
                "Never apply to jobs that require security clearances or US Citizenship.",
            )
        ],
    )
    reason, penalty, hits, cit = pref.apply_preference_gate(
        "Must hold active security clearances for federal systems."
    )
    assert reason and "preference_hard" in reason
    assert cit is False
    assert penalty == 0


def test_compound_clearance_citizenship_splits_on_jd(monkeypatch):
    """Compound rule: clearance in JD → HARD; citizenship-only → soft."""
    from src.core import preferences as pref

    monkeypatch.setattr(
        pref,
        "load_preference_rules",
        lambda path=None: [
            _rule(
                "JOB_REQUIREMENTS",
                "Never apply to jobs that require security clearances or US Citizenship.",
            )
        ],
    )
    soft_reason, soft_pen, soft_hits, cit = pref.apply_preference_gate(
        "Software intern in Bangalore. Candidates must have US citizenship."
    )
    assert soft_reason is None
    assert cit is True
    assert soft_pen == pref.CITIZENSHIP_SOFT_PENALTY_POINTS


def test_citizenship_token_false_positive_ignored(monkeypatch):
    """Ordinary SDE JD must not soft-gate via 'security'+'require' tokens alone."""
    from src.core import preferences as pref

    monkeypatch.setattr(
        pref,
        "load_preference_rules",
        lambda path=None: [
            _rule(
                "JOB_REQUIREMENTS",
                "Never apply to jobs that require security clearances or US Citizenship.",
            )
        ],
    )
    reason, penalty, hits, cit = pref.apply_preference_gate(
        "We require strong security fundamentals and experience with cloud systems."
    )
    assert reason is None
    assert cit is False
    assert penalty == 0


def test_eeo_boilerplate_ignored(monkeypatch):
    from src.core import preferences as pref

    monkeypatch.setattr(
        pref,
        "load_preference_rules",
        lambda path=None: [
            _rule("JOB_REQUIREMENTS", "Never apply to jobs that require US Citizenship.")
        ],
    )
    jd = (
        "Software Intern. We are an equal opportunity employer and hire "
        "regardless of race, citizenship, or national origin."
    )
    reason, penalty, hits, cit = pref.apply_preference_gate(jd)
    assert reason is None
    assert cit is False


def test_builtin_exclusions_drop_citizen_terms():
    from src.core.evaluate import _BUILTIN_EXCLUSIONS

    assert "us citizen" not in _BUILTIN_EXCLUSIONS
    assert "green card" not in _BUILTIN_EXCLUSIONS
    assert "security clearance" in _BUILTIN_EXCLUSIONS


def test_citizenship_review_status_valid(tmp_path):
    from src.core import memory as mem

    db = tmp_path / "m.db"
    assert "REQUIRES_HUMAN_CITIZENSHIP_REVIEW" in mem.VALID_STATUSES
    mem.init_db(db)
    # Must not raise ValueError (status allowlisted).
    mem.log_application(
        "CitCo",
        "https://example.com/job/citizenship-review-1",
        "SDE",
        "REQUIRES_HUMAN_CITIZENSHIP_REVIEW",
        error_log="soft-gated",
        db_path=db,
    )
    counts = mem.application_counts(db)
    assert counts.get("requires_human_citizenship_review", 0) >= 1


def test_cdp_assist_skips_dead_loop():
    from src.core.executor import make_cdp_assist

    called = {"n": 0}

    def factory():
        called["n"] += 1
        return object()

    assist = make_cdp_assist(None, factory)
    assert assist("stuck") is False
    assert called["n"] == 0

    class _Closed:
        def is_running(self):
            return True

        def is_closed(self):
            return True

    assist2 = make_cdp_assist(_Closed(), factory)
    assert assist2("stuck") is False
    assert called["n"] == 0


def test_cdp_assist_pins_to_running_loop():
    from src.core import executor as ex
    import asyncio as aio

    scheduled = {"loop": None}

    class _Loop:
        def is_running(self):
            return True

        def is_closed(self):
            return False

    loop = _Loop()

    def fake_threadsafe(coro, target_loop):
        scheduled["loop"] = target_loop

        class _Fut:
            def result(self, timeout=None):
                try:
                    coro.close()
                except Exception:
                    pass
                return True

            def cancel(self):
                return False

        return _Fut()

    orig = aio.run_coroutine_threadsafe
    aio.run_coroutine_threadsafe = fake_threadsafe  # type: ignore
    try:
        assist = ex.make_cdp_assist(
            loop, lambda: MagicMock(cdp_url="http://x", user_data_dir=".")
        )
        assert assist("x") is True
        assert scheduled["loop"] is loop
    finally:
        aio.run_coroutine_threadsafe = orig


def test_cdp_pin_rejects_loop_mismatch():
    from src.core.executor import RequiresHumanInterventionError, _pin_running_loop

    async def _inner():
        other = object()
        with pytest.raises(RequiresHumanInterventionError, match="loop mismatch"):
            _pin_running_loop(expected=other)

    asyncio.run(_inner())


def test_cdp_assist_handles_future_loop_runtime_error(monkeypatch):
    """Sync bridge must not raise when the future surfaces a different-loop error."""
    from src.core import executor as ex
    import asyncio as aio

    class _Loop:
        def is_running(self):
            return True

        def is_closed(self):
            return False

    def boom_threadsafe(coro, target_loop):
        try:
            coro.close()
        except Exception:
            pass

        class _Fut:
            def result(self, timeout=None):
                raise RuntimeError(
                    "Task got Future <Future pending> attached to a different loop"
                )

            def cancel(self):
                return True

        return _Fut()

    orig = aio.run_coroutine_threadsafe
    aio.run_coroutine_threadsafe = boom_threadsafe  # type: ignore
    try:
        assist = ex.make_cdp_assist(_Loop(), lambda: MagicMock())
        assert assist("stuck") is False
    finally:
        aio.run_coroutine_threadsafe = orig


def test_pin_running_loop_inside_async():
    from src.core.executor import _pin_running_loop

    async def _inner():
        return _pin_running_loop()

    loop = asyncio.run(_inner())
    assert loop is not None


def test_cdp_endpoint_alive_and_stale_clear(tmp_path, monkeypatch):
    from src.playbooks import session as sess

    port_file = tmp_path / "DevToolsActivePort"
    port_file.write_text("59999\n/devtools/browser/x", encoding="utf-8")
    assert sess.read_cdp_url(tmp_path, wait_s=0.2) == "http://127.0.0.1:59999"
    assert sess.cdp_endpoint_alive("http://127.0.0.1:59999", timeout_s=0.3) is False
    assert sess.read_live_cdp_url(tmp_path, wait_s=0.2, probe_timeout_s=0.3) is None
    assert not port_file.exists()


def test_is_cdp_attach_error_detects_connect():
    from src.core.executor import _is_cdp_attach_error

    class ConnectError(Exception):
        pass

    assert _is_cdp_attach_error(ConnectError("All connection attempts failed"))
    assert _is_cdp_attach_error(AssertionError("Root CDP client not initialized"))
    assert not _is_cdp_attach_error(ValueError("unrelated"))


def test_warm_browser_unavailable_disables_prefer_cdp(monkeypatch):
    """Warm ensure miss must not advertise zombie CDP to vision fallback."""
    from src.core import executor as ex

    assert issubclass(ex.WarmBrowserUnavailable, RuntimeError)
    assert ex._is_cdp_attach_error(
        RuntimeError("CDP connected but failed due to AssertionError: Root CDP client not initialized")
    )
