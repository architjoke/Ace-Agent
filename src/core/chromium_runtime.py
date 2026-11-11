"""Guarantee a durable Playwright Chromium binary for warm/cold apply paths.

Cursor sandbox caches under %TEMP%\\cursor-sandbox-cache are ephemeral — a wiped
cache makes ``launch_persistent_context`` fail with ``Executable doesn't exist``,
which collapses every apply into MANUAL_ACTION. Prefer a stable local cache.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENSURED = False


def _chrome_exe(browsers_root: Path) -> Path | None:
    """Best-effort locate chrome.exe under a Playwright browsers root."""
    if not browsers_root.is_dir():
        return None
    # chromium-XXXX/chrome-win64/chrome.exe (Windows) or chrome-linux/chrome
    for pattern in (
        "chromium-*/chrome-win64/chrome.exe",
        "chromium-*/chrome-win/chrome.exe",
        "chromium-*/chrome-linux/chrome",
        "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    ):
        hits = sorted(browsers_root.glob(pattern), reverse=True)
        for hit in hits:
            if hit.is_file():
                return hit
    return None


def durable_browsers_path() -> Path:
    """Stable install root (not Temp\\cursor-sandbox-cache)."""
    override = (os.environ.get("ACE_PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("HOME") or str(Path.home())
    return Path(local).expanduser().resolve() / "ace-playwright"


def _path_looks_ephemeral(path: Path | None) -> bool:
    if path is None:
        return True
    s = str(path).lower().replace("\\", "/")
    return "cursor-sandbox-cache" in s or "/temp/" in s or "/tmp/" in s


def ensure_playwright_chromium(*, force_install: bool = False) -> Path:
    """Point PLAYWRIGHT_BROWSERS_PATH at a durable cache and ensure Chromium exists.

    Safe to call repeatedly; subsequent calls are no-ops once verified.
    """
    global _ENSURED
    if _ENSURED and not force_install:
        root = Path(
            os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or durable_browsers_path()
        )
        exe = _chrome_exe(root)
        if exe is not None:
            return exe

    current = (os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    current_path = Path(current).expanduser().resolve() if current else None
    current_exe = _chrome_exe(current_path) if current_path else None

    if current_exe is not None and not _path_looks_ephemeral(current_path):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(current_path)
        _ENSURED = True
        return current_exe

    durable = durable_browsers_path()
    durable.mkdir(parents=True, exist_ok=True)
    durable_exe = _chrome_exe(durable)

    # Reuse ephemeral install if present (avoids re-download) by keeping it only
    # when durable is empty — still prefer copying path to durable via install.
    if durable_exe is None or force_install:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(durable)
        logger.warning(
            "Ensuring Playwright Chromium under durable path %s", durable
        )
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=False,
                timeout=600,
                capture_output=True,
                text=True,
            )
        except Exception:
            logger.exception("playwright install chromium failed")
        durable_exe = _chrome_exe(durable)

    if durable_exe is None and current_exe is not None:
        # Last resort: keep working ephemeral binary for this process.
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(current_path)
        logger.warning(
            "Durable Chromium missing — using ephemeral cache %s", current_exe
        )
        _ENSURED = True
        return current_exe

    if durable_exe is None:
        raise FileNotFoundError(
            f"Playwright Chromium not found under {durable}. "
            "Run: python -m playwright install chromium"
        )

    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(durable)
    logger.info("Playwright Chromium ready: %s", durable_exe)
    _ENSURED = True
    return durable_exe
