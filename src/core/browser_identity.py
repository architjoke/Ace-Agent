"""Pinned browser identity — stable UA/viewport for single-IP stealth."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Single stable desktop Chrome UA (never rotate on one IP).
_PINNED_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_WEBDRIVER_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
"""


def _ua_from_profile(profile_dir: Path | None) -> str | None:
    """Best-effort Chrome version → UA from persistent profile Local State."""
    if profile_dir is None:
        return None
    root = Path(profile_dir).expanduser()
    for name in ("Local State", "Preferences"):
        path = root / name
        if not path.is_file():
            # Preferences lives under Default/
            alt = root / "Default" / name if name == "Preferences" else None
            path = alt if alt and alt.is_file() else path
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Local State: browser.last_redirect_origin / executable version strings
        blob = json.dumps(data)[:50_000]
        match = re.search(r"Chrome/(\d+\.\d+\.\d+\.\d+)", blob)
        if not match:
            match = re.search(r'"[^"]*chrome[^"]*?(\d{2,3}\.\d+\.\d+\.\d+)"', blob, re.I)
        if match:
            ver = match.group(1) if match.lastindex else match.group(0)
            ver = ver.replace("Chrome/", "")
            return (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                f"(KHTML, like Gecko) Chrome/{ver} Safari/537.36"
            )
    return None


def pinned_user_agent(*, user_data_dir: Path | str | None = None) -> str:
    """UA pin order: env → profile Chrome version → stable constant."""
    env = (os.environ.get("ACE_BROWSER_USER_AGENT") or "").strip()
    if env:
        return env
    try:
        from src.core.settings import get_settings

        profile = Path(user_data_dir) if user_data_dir else get_settings().browser_user_data_dir
    except Exception:
        profile = Path(user_data_dir) if user_data_dir else None
    from_profile = _ua_from_profile(profile)
    if from_profile:
        return from_profile
    return _PINNED_UA


def pinned_viewport() -> dict[str, int]:
    """Viewport pin: ACE_BROWSER_VIEWPORT=WxH or 1920x1080."""
    raw = (os.environ.get("ACE_BROWSER_VIEWPORT") or "1920x1080").strip().lower()
    match = re.match(r"(\d+)\s*[x×]\s*(\d+)", raw)
    if match:
        return {"width": int(match.group(1)), "height": int(match.group(2))}
    return {"width": 1920, "height": 1080}


def pinned_device_scale_factor() -> float:
    raw = (os.environ.get("ACE_BROWSER_DEVICE_SCALE") or "1").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.0


def pinned_locale() -> str:
    """BCP-47 locale aligned with sticky proxy geo (ACE_BROWSER_LOCALE).

    Default en-IN for India candidates. Mismatch vs residential IP is a CF signal.
    """
    return (os.environ.get("ACE_BROWSER_LOCALE") or "en-IN").strip() or "en-IN"


def pinned_timezone_id() -> str:
    """IANA timezone for Chromium (ACE_BROWSER_TIMEZONE). Default Asia/Kolkata."""
    return (os.environ.get("ACE_BROWSER_TIMEZONE") or "Asia/Kolkata").strip() or "Asia/Kolkata"


def webdriver_stealth_script() -> str:
    return _WEBDRIVER_STEALTH_JS
