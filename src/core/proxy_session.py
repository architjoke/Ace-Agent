"""Crawlee-inspired proxy session pool: burn on block, rotate, bounded retry."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, TypeVar

from src.core.network import list_proxies, parse_proxy_url

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BLOCK_NEEDLES = (
    "cf-challenge",
    "cf-browser-verification",
    "attention required",
    "just a moment",
    "checking your browser",
    "access denied",
    "403 forbidden",
    "captcha",
    "recaptcha",
    "hcaptcha",
    "verify you are human",
    "unusual traffic",
    "blocked by bot",
    "net::err_proxy",
    "err_proxy_connection_failed",
    "proxy connection",
)


def _extra_env_proxies() -> list[str]:
    raw = (os.environ.get("PROXY_URLS") or "").strip()
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def pool_proxies() -> list[str]:
    """PROXY_URL + PROXY_URLS + config network.proxies (deduped)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in [*list_proxies(), *_extra_env_proxies()]:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def looks_blocked(text: str | Exception | None = None, *, status: int | None = None) -> bool:
    """True if response/error indicates proxy burn / bot wall."""
    if status is not None and int(status) in {403, 429, 503}:
        return True
    blob = str(text or "").lower()
    if not blob:
        return False
    return any(n in blob for n in _BLOCK_NEEDLES)


class ProxySessionManager:
    """Sticky proxy session with burn/rotate for apply/scrape resilience."""

    def __init__(self, proxies: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._pool = list(proxies) if proxies is not None else pool_proxies()
        self._index = 0
        self._burned: set[str] = set()

    def available(self) -> list[str]:
        with self._lock:
            return [p for p in self._pool if p not in self._burned]

    def current(self) -> str | None:
        alive = self.available()
        if not alive:
            return None
        with self._lock:
            self._index %= len(alive)
            return alive[self._index]

    def playwright_proxy(self) -> dict[str, str] | None:
        url = self.current()
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

    def browser_use_proxy(self) -> Any | None:
        url = self.current()
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
            logger.warning("Could not build ProxySettings: %s", exc)
            return None

    def mark_burned(self, reason: str = "") -> None:
        url = self.current()
        if not url:
            return
        with self._lock:
            self._burned.add(url)
        logger.warning("Proxy burned (%s): %s", reason or "blocked", url.split("@")[-1][:80])

    def rotate(self) -> str | None:
        alive = self.available()
        if not alive:
            logger.warning("No proxies left to rotate to")
            return None
        with self._lock:
            self._index = (self._index + 1) % len(alive)
            nxt = alive[self._index]
        logger.info("Proxy rotated → %s", nxt.split("@")[-1][:80])
        return nxt

    def with_retries(
        self,
        fn: Callable[[], T],
        *,
        max_attempts: int = 3,
        is_block: Callable[[Exception], bool] | None = None,
    ) -> T:
        """Run fn; on block markers burn proxy, rotate, retry."""
        attempts = max(1, int(max_attempts))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                blocked = looks_blocked(exc) or (is_block(exc) if is_block else False)
                if not blocked or attempt >= attempts - 1:
                    raise
                self.mark_burned(str(exc)[:120])
                if not self.rotate():
                    raise
                logger.info(
                    "Retrying after proxy rotate (attempt %s/%s)",
                    attempt + 2,
                    attempts,
                )
        assert last_exc is not None
        raise last_exc
