"""Per-host dispatch spacing with jitter (IP-block / CAPTCHA soft armor)."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections.abc import Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# host_suffix -> (base_seconds, jitter_seconds)
# Sensitive boards (CF / account flags) get LinkedIn-class spacing.
_DEFAULT_HOST_LIMITS: dict[str, tuple[float, float]] = {
    "linkedin.com": (90.0, 15.0),
    # Indeed Soft Apply is crowded — heavy spacing if ACE_ALLOW_INDEED=1.
    "indeed.com": (150.0, 30.0),
    "smartapply.indeed.com": (150.0, 30.0),
    "myworkdayjobs.com": (45.0, 10.0),
    "lever.co": (25.0, 5.0),
    "greenhouse.io": (25.0, 5.0),
    "wellfound.com": (95.0, 20.0),
    "angel.co": (95.0, 20.0),
    "glassdoor.com": (120.0, 25.0),
    "glassdoor.co.in": (120.0, 25.0),
    "naukri.com": (50.0, 12.0),
    "workatastartup.com": (75.0, 15.0),
    "weworkremotely.com": (20.0, 5.0),
    "himalayas.app": (20.0, 5.0),
    "hiring.cafe": (25.0, 5.0),
}
DEFAULT_LIMIT: tuple[float, float] = (15.0, 3.0)


def _parse_host_limits_env(raw: str) -> dict[str, tuple[float, float]]:
    """Parse ACE_HOST_LIMITS=linkedin.com=120:20,naukri.com=60:15."""
    out: dict[str, tuple[float, float]] = {}
    for part in (raw or "").replace("\n", ",").split(","):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        host, rest = chunk.split("=", 1)
        host = host.strip().lower().lstrip(".")
        rest = rest.strip()
        if not host or ":" not in rest:
            continue
        base_s, jit_s = rest.split(":", 1)
        try:
            base = float(base_s.strip())
            jitter = float(jit_s.strip())
        except ValueError:
            continue
        if base < 1.0:
            continue
        out[host] = (base, max(0.0, jitter))
    return out


def _build_host_limits() -> dict[str, tuple[float, float]]:
    limits = dict(_DEFAULT_HOST_LIMITS)
    env = (os.environ.get("ACE_HOST_LIMITS") or "").strip()
    if env:
        overrides = _parse_host_limits_env(env)
        if overrides:
            limits.update(overrides)
            logger.info("ACE_HOST_LIMITS overrides applied: %s", sorted(overrides))
    return limits


HOST_LIMITS: dict[str, tuple[float, float]] = _build_host_limits()
_HOST_KEYS_SORTED = tuple(sorted(HOST_LIMITS.keys(), key=len, reverse=True))


def reload_host_limits_for_tests() -> None:
    """Re-read ACE_HOST_LIMITS into module globals (tests only)."""
    global HOST_LIMITS, _HOST_KEYS_SORTED
    HOST_LIMITS = _build_host_limits()
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
