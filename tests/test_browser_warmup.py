"""Warm keep-alive reset + env defaults."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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
