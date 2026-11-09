"""Per-host dispatch spacing with jitter (IP-block / CAPTCHA soft armor)."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from urllib.parse import urlparse

# host_suffix -> (base_seconds, jitter_seconds)
HOST_LIMITS: dict[str, tuple[float, float]] = {
    "linkedin.com": (90.0, 15.0),
    "myworkdayjobs.com": (45.0, 10.0),
    "lever.co": (25.0, 5.0),
    "greenhouse.io": (25.0, 5.0),
}
DEFAULT_LIMIT: tuple[float, float] = (15.0, 3.0)

_HOST_KEYS_SORTED = tuple(sorted(HOST_LIMITS.keys(), key=len, reverse=True))


def host_bucket_key(job_url: str) -> str:
    """Map a job URL hostname to a limiter bucket (suffix match)."""
    host = (urlparse(job_url or "").hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return "default"
    for key in _HOST_KEYS_SORTED:
        if host == key or host.endswith("." + key):
            return key
    return host


def _limits_for(key: str) -> tuple[float, float]:
    if key in HOST_LIMITS:
        return HOST_LIMITS[key]
    for listed, lim in HOST_LIMITS.items():
        if key == listed or key.endswith("." + listed):
            return lim
    return DEFAULT_LIMIT


class HostRateLimiter:
    """In-memory per-host token spacing shared across browser workers."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._last: dict[str, float] = {}
        self._interval: dict[str, float] = {}
        self._lock = threading.Lock()

    def host_key(self, job_url: str) -> str:
        return host_bucket_key(job_url)

    def _effective_interval(self, key: str) -> float:
        base, jitter = _limits_for(key)
        delay = base + random.uniform(-jitter, jitter)
        return max(1.0, float(delay))

    def ready_in(self, job_url: str) -> float:
        """Seconds until this host can dispatch again (0 if ready)."""
        key = self.host_key(job_url)
        now = float(self._clock())
        with self._lock:
            last = self._last.get(key)
            if last is None:
                return 0.0
            interval = self._interval.get(key) or self._effective_interval(key)
            remaining = (last + interval) - now
            return max(0.0, float(remaining))

    def mark_dispatched(self, job_url: str) -> None:
        key = self.host_key(job_url)
        with self._lock:
            self._last[key] = float(self._clock())
            self._interval[key] = self._effective_interval(key)
