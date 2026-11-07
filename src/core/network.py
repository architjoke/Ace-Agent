"""Proxy helpers for JobSpy rotation and sticky Chromium egress."""

from __future__ import annotations

import logging
import os
import random
import threading
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_rr_lock = threading.Lock()
_rr_index = 0


def _env_proxy() -> str:
    return (os.environ.get("PROXY_URL") or "").strip()


def _config_proxies() -> list[str]:
    try:
        from src.core.settings import get_settings

        raw = get_settings().raw or {}
        net = raw.get("network") if isinstance(raw, dict) else None
        if not isinstance(net, dict):
            return []
        items = net.get("proxies") or []
        if not isinstance(items, list):
            return []
        return [str(p).strip() for p in items if str(p).strip()]
    except Exception:
        return []


def _rotate_discovery_enabled() -> bool:
    try:
        from src.core.settings import get_settings

        raw = get_settings().raw or {}
        net = raw.get("network") if isinstance(raw, dict) else None
        if isinstance(net, dict) and "rotate_discovery" in net:
            return bool(net.get("rotate_discovery"))
    except Exception:
        pass
    return True


def list_proxies() -> list[str]:
    """Merge PROXY_URL + config network.proxies (deduped, order preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in [_env_proxy(), *_config_proxies()]:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def sticky_proxy_url() -> str | None:
    """Preferred Chromium egress: PROXY_URL, else first config proxy."""
    env = _env_proxy()
    if env:
        return env
    cfg = _config_proxies()
    return cfg[0] if cfg else None


def pick_discovery_proxy() -> str | None:
    """Rotate (or pick random) proxy for JobSpy when rotate_discovery is on."""
    proxies = list_proxies()
    if not proxies:
        return None
    if not _rotate_discovery_enabled():
        return proxies[0]
    global _rr_index
    with _rr_lock:
        if len(proxies) == 1:
            return proxies[0]
        # Prefer round-robin; occasional shuffle avoids sticky scrape IP
        if random.random() < 0.15:
            return random.choice(proxies)
        proxy = proxies[_rr_index % len(proxies)]
        _rr_index += 1
        return proxy


def discovery_proxies_arg() -> list[str] | None:
    """JobSpy ``proxies=`` argument — None means direct egress."""
    proxies = list_proxies()
    if not proxies:
        return None
    if _rotate_discovery_enabled() and len(proxies) > 1:
        # JobSpy accepts a list; pass full pool so its internals can rotate
        return proxies
    picked = pick_discovery_proxy()
    return [picked] if picked else None


def parse_proxy_url(url: str) -> dict[str, str | None]:
    """Split proxy URL into server / username / password for Playwright."""
    raw = (url or "").strip()
    if not raw:
        return {"server": None, "username": None, "password": None}
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.hostname:
        # Bare host:port → assume http
        if "://" not in raw:
            return {"server": f"http://{raw}", "username": None, "password": None}
        return {"server": raw, "username": None, "password": None}
    port = f":{parsed.port}" if parsed.port else ""
    server = f"{parsed.scheme}://{parsed.hostname}{port}"
    return {
        "server": server,
        "username": parsed.username,
        "password": parsed.password,
    }


def browser_proxy_settings() -> Any | None:
    """browser-use ProxySettings for sticky egress, or None for direct."""
    url = sticky_proxy_url()
    if not url:
        return None
    parts = parse_proxy_url(url)
    if not parts.get("server"):
        return None
    try:
        from browser_use.browser.profile import ProxySettings

        return ProxySettings(
            server=parts["server"],
            username=parts.get("username"),
            password=parts.get("password"),
        )
    except Exception as exc:
        logger.warning("Could not build ProxySettings from PROXY_URL: %s", exc)
        return None


def playwright_proxy_dict() -> dict[str, str] | None:
    """Dict suitable for Playwright ``proxy=`` launch/context option."""
    url = sticky_proxy_url()
    if not url:
        return None
    parts = parse_proxy_url(url)
    if not parts.get("server"):
        return None
    out: dict[str, str] = {"server": str(parts["server"])}
    if parts.get("username"):
        out["username"] = str(parts["username"])
    if parts.get("password"):
        out["password"] = str(parts["password"])
    return out
