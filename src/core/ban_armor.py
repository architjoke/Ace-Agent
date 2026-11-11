"""Account-ban armor: quiet hours, CF/captcha circuit breaker, apply pacing helpers.

Ban priority over conversion — never solves Turnstile/OTP; pauses instead.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def apply_timezone_name() -> str:
    """Timezone for quiet-hours + browser pin (ACE_BROWSER_TIMEZONE)."""
    return (os.environ.get("ACE_BROWSER_TIMEZONE") or "Asia/Kolkata").strip() or "Asia/Kolkata"


def in_apply_hours(*, now: datetime | None = None) -> bool:
    """True if local time is inside ACE_APPLY_HOURS (default 8-22, inclusive start).

    Format: ``HH-HH`` in 24h local time (ACE_BROWSER_TIMEZONE). Empty / ``0-24`` = always on.
    Outside hours the worker should sleep — overnight bursts look bot-like.
    """
    raw = (os.environ.get("ACE_APPLY_HOURS") or "8-22").strip()
    if not raw or raw in {"0-24", "00-24", "all", "*"}:
        return True
    try:
        start_s, end_s = raw.split("-", 1)
        start_h = int(start_s.strip())
        end_h = int(end_s.strip())
    except ValueError:
        return True
    start_h = max(0, min(23, start_h))
    end_h = max(0, min(24, end_h))
    if start_h == end_h:
        return True
    try:
        tz = ZoneInfo(apply_timezone_name())
    except Exception:
        tz = ZoneInfo("UTC")
    local = now or datetime.now(tz)
    if local.tzinfo is None:
        local = local.replace(tzinfo=tz)
    else:
        local = local.astimezone(tz)
    hour = local.hour
    if start_h < end_h:
        return start_h <= hour < end_h
    # Overnight window e.g. 22-6
    return hour >= start_h or hour < end_h


def seconds_until_apply_hours(*, now: datetime | None = None) -> float:
    """Seconds to sleep until the next apply-hours open (0 if already open)."""
    if in_apply_hours(now=now):
        return 0.0
    try:
        tz = ZoneInfo(apply_timezone_name())
    except Exception:
        tz = ZoneInfo("UTC")
    local = now or datetime.now(tz)
    if local.tzinfo is None:
        local = local.replace(tzinfo=tz)
    else:
        local = local.astimezone(tz)

    raw = (os.environ.get("ACE_APPLY_HOURS") or "8-22").strip()
    try:
        start_s, _ = raw.split("-", 1)
        start_h = max(0, min(23, int(start_s.strip())))
    except ValueError:
        return 60.0

    candidate = local.replace(hour=start_h, minute=0, second=0, microsecond=0)
    if candidate <= local:
        from datetime import timedelta

        candidate = candidate + timedelta(days=1)
    return max(30.0, (candidate - local).total_seconds())


class CaptchaCircuitBreaker:
    """Trip after N CF/captcha blocks in a window; cool that host (or global) down.

    Env:
      ACE_CF_TRIP_THRESHOLD — failures to trip (default 3)
      ACE_CF_TRIP_WINDOW_SEC — rolling window (default 1800)
      ACE_CF_COOLDOWN_SEC — pause after trip (default 3600)
    """

    def __init__(
        self,
        *,
        threshold: int | None = None,
        window_sec: float | None = None,
        cooldown_sec: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.threshold = max(1, int(threshold if threshold is not None else _env_int("ACE_CF_TRIP_THRESHOLD", 3)))
        self.window_sec = float(
            window_sec if window_sec is not None else _env_int("ACE_CF_TRIP_WINDOW_SEC", 1800)
        )
        self.cooldown_sec = float(
            cooldown_sec if cooldown_sec is not None else _env_int("ACE_CF_COOLDOWN_SEC", 3600)
        )
        self._clock = clock or time.monotonic
        self._hits: dict[str, deque[float]] = {}
        self._cooldown_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def ready_in(self, host_key: str) -> float:
        key = (host_key or "global").lower()
        now = float(self._clock())
        with self._lock:
            until = self._cooldown_until.get(key) or self._cooldown_until.get("global")
            if until is None:
                return 0.0
            return max(0.0, until - now)

    def record_block(self, host_key: str, *, reason: str = "") -> bool:
        """Record a CF/captcha hard-exit. Returns True if this call tripped the breaker."""
        key = (host_key or "global").lower()
        now = float(self._clock())
        tripped = False
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            hits.append(now)
            while hits and (now - hits[0]) > self.window_sec:
                hits.popleft()
            if len(hits) >= self.threshold:
                self._cooldown_until[key] = now + self.cooldown_sec
                # Global soft pause so one board storm doesn't thrash the whole pool.
                global_until = self._cooldown_until.get("global", 0.0)
                self._cooldown_until["global"] = max(
                    global_until, now + min(self.cooldown_sec, 900.0)
                )
                hits.clear()
                tripped = True
        if tripped:
            logger.warning(
                "Captcha circuit OPEN for %s (cooldown %.0fs)%s",
                key,
                self.cooldown_sec,
                f" — {reason}" if reason else "",
            )
        return tripped

    def is_cf_block_message(self, message: str) -> bool:
        low = (message or "").lower()
        needles = (
            "cloudflare",
            "turnstile",
            "cf-mitigated",
            "captcha",
            "verify you are human",
            "requires human",
        )
        return any(n in low for n in needles)


_BREAKER: CaptchaCircuitBreaker | None = None
_BREAKER_LOCK = threading.Lock()


def get_captcha_breaker() -> CaptchaCircuitBreaker:
    global _BREAKER
    with _BREAKER_LOCK:
        if _BREAKER is None:
            _BREAKER = CaptchaCircuitBreaker()
        return _BREAKER


def reset_captcha_breaker_for_tests() -> CaptchaCircuitBreaker:
    global _BREAKER
    with _BREAKER_LOCK:
        _BREAKER = CaptchaCircuitBreaker()
        return _BREAKER
