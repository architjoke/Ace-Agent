"""Playbook browser session with CDP endpoint for Gemini hot-swap."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_cdp_url(user_data_dir: Path | str | None, *, wait_s: float = 2.0) -> str | None:
    """Read Chromium DevToolsActivePort → http://127.0.0.1:{port}."""
    if not user_data_dir:
        return None
    port_file = Path(user_data_dir).expanduser().resolve() / "DevToolsActivePort"
    deadline = time.monotonic() + max(0.2, float(wait_s))
    while time.monotonic() < deadline:
        try:
            if port_file.is_file():
                lines = port_file.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    port = int(str(lines[0]).strip())
                    if 1 <= port <= 65535:
                        return f"http://127.0.0.1:{port}"
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    logger.debug("DevToolsActivePort not ready under %s", user_data_dir)
    return None


@dataclass
class PlaybookSession:
    context: Any
    page: Any
    user_data_dir: Path
    cdp_url: str | None = None
    playwright: Any = field(default=None, repr=False)

    @classmethod
    def from_live(
        cls,
        context: Any,
        page: Any,
        user_data_dir: Path | str | None,
        *,
        playwright: Any = None,
    ) -> PlaybookSession:
        from src.core.settings import get_settings

        profile = (
            Path(user_data_dir).expanduser().resolve()
            if user_data_dir
            else Path(get_settings().browser_user_data_dir).expanduser().resolve()
        )
        return cls(
            context=context,
            page=page,
            user_data_dir=profile,
            cdp_url=read_cdp_url(profile),
            playwright=playwright,
        )

    def discard_modal(self) -> None:
        """Best-effort Close/Discard/Dismiss on the live page."""
        page = self.page
        if page is None:
            return
        import re

        for pattern in (
            re.compile(r"^(Close|Discard|Dismiss|Cancel)$", re.I),
            re.compile(r"discard\s+application", re.I),
        ):
            try:
                btn = page.get_by_role("button", name=pattern).first
                if btn.count() and btn.is_visible(timeout=800):
                    btn.click(timeout=2000)
                    page.wait_for_timeout(300)
                    return
            except Exception:
                continue
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

    def close(self) -> None:
        from src.playbooks._common import close_context_safe

        close_context_safe(self.context, user_data_dir=self.user_data_dir)
        self.context = None
        self.page = None
        pw = self.playwright
        self.playwright = None
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                logger.debug("playwright.stop after handoff failed", exc_info=True)


class PlaybookHandoff(Exception):
    """Playbook stuck — keep Chromium open for CDP Gemini takeover."""

    def __init__(self, reason: str, session: PlaybookSession) -> None:
        super().__init__(reason)
        self.session = session
        self.reason = reason
