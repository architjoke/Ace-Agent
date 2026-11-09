"""Clone Chromium login profiles and run N parallel browser apply workers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from src.core.config import PROJECT_ROOT, WORKER_STATUS
from src.core.host_limiter import HostRateLimiter
from src.core.queue import (
    WAIT_SENTINEL_KEY,
    claim_next,
    mark_done,
    queue_counts,
    reclaim_orphaned_claims,
    requeue_stale_claims,
)
from src.core.settings import get_settings
from src.core.utils import env_truthy

logger = logging.getLogger(__name__)

REPO_ROOT = PROJECT_ROOT
WORKER_STATUS_PATH = WORKER_STATUS

_LOCK_NAMES = ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile")
_COPY_IGNORE = shutil.ignore_patterns(
    "SingletonLock",
    "SingletonCookie",
    "SingletonSocket",
    "lockfile",
    "Crashpad",
    "Crash Reports",
    "BrowserMetrics",
    "Code Cache",
    "GPUCache",
    "GrShaderCache",
    "ShaderCache",
)


def master_profile_dir() -> Path:
    return Path(get_settings().browser_user_data_dir).expanduser().resolve()


def worker_profile_dir(worker_id: str) -> Path:
    """``data/browser_profile_w1`` … from master ``data/browser_profile``."""
    master = master_profile_dir()
    wid = (worker_id or "w1").strip().lower()
    if not wid.startswith("w"):
        wid = f"w{wid}"
    return master.parent / f"{master.name}_{wid}"


def clear_singleton_locks(profile_dir: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    for name in _LOCK_NAMES:
        lock = profile_dir / name
        try:
            lock.unlink(missing_ok=True)
        except TypeError:
            if lock.exists():
                try:
                    lock.unlink()
                except OSError:
                    pass
        except OSError:
            pass


def clone_profiles_from_master(
    n_workers: int | None = None,
    *,
    force: bool = False,
) -> list[Path]:
    """Copy master login profile → ``browser_profile_w{i}`` for each worker.

    Call after ``ace login`` (and before ``ace start``). Skips copy when the
    worker folder already looks populated unless ``force=True``.
    """
    settings = get_settings()
    n = int(n_workers if n_workers is not None else settings.browser_workers)
    n = max(1, min(8, n))
    master = master_profile_dir()
    if not master.is_dir():
        raise FileNotFoundError(
            f"Master browser profile missing at {master}. Run: ace login"
        )

    clear_singleton_locks(master)
    out: list[Path] = []
    for i in range(1, n + 1):
        wid = f"w{i}"
        dest = worker_profile_dir(wid)
        marker = dest / "Default" / "Preferences"
        needs_copy = force or not marker.is_file()
        if needs_copy:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            logger.info("Cloning browser profile %s → %s", master.name, dest.name)
            shutil.copytree(master, dest, ignore=_COPY_IGNORE, dirs_exist_ok=False)
        else:
            logger.info("Reusing existing worker profile %s", dest.name)
        clear_singleton_locks(dest)
        out.append(dest)
    write_worker_status(
        {
            f"w{i}": {"status": "idle", "company": "", "updated_at": _now()}
            for i in range(1, n + 1)
        }
    )
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_worker_status(workers: dict[str, dict[str, Any]]) -> None:
    WORKER_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now(),
        "workers": workers,
        "queue": queue_counts(),
    }
    try:
        WORKER_STATUS_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except OSError:
        logger.exception("Could not write worker status")


def read_worker_status() -> dict[str, Any]:
    if not WORKER_STATUS_PATH.is_file():
        return {"workers": {}, "queue": queue_counts()}
    try:
        data = json.loads(WORKER_STATUS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["queue"] = queue_counts()
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"workers": {}, "queue": queue_counts()}


def update_one_worker(worker_id: str, **fields: Any) -> None:
    data = read_worker_status()
    workers = dict(data.get("workers") or {})
    cur = dict(workers.get(worker_id) or {})
    cur.update(fields)
    cur["updated_at"] = _now()
    workers[worker_id] = cur
    write_worker_status(workers)


class LinkedInRateLimiter:
    """Soft cap Easy Applies / hour / account across parallel workers."""

    def __init__(self, max_per_hour: int = 10) -> None:
        self.max_per_hour = max(1, int(max_per_hour))
        self._ts: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Reserve a slot; sleep *outside* the lock so peers can progress."""
        while True:
            wait = 0.0
            async with self._lock:
                now = time.monotonic()
                self._ts = [t for t in self._ts if now - t < 3600]
                if len(self._ts) < self.max_per_hour:
                    self._ts.append(now)
                    return
                wait = min(120.0, max(5.0, 3600 - (now - self._ts[0]) + 1))
            logger.info(
                "LinkedIn soft rate limit (%s/hour) — sleeping %.0fs",
                self.max_per_hour,
                wait,
            )
            await asyncio.sleep(wait)


ApplyFn = Callable[..., Awaitable[str]]


_BROWSER_CRASH_NEEDLES = (
    "target closed",
    "browser has been closed",
    "chromium",
    "disconnected",
    "protocol error",
    "browsercontext",
)


async def browser_worker_loop(
    worker_id: str,
    *,
    master_yaml: Path,
    apply_fn: ApplyFn,
    stop_event: asyncio.Event,
    producers_done: asyncio.Event,
    linkedin_limiter: LinkedInRateLimiter | None = None,
    host_limiter: HostRateLimiter | None = None,
    idle_poll_sec: float = 1.5,
    launch_stagger_sec: float = 0.0,
) -> dict[str, int]:
    """Pull READY queue items until producers finish and the queue is empty."""
    from src.core.executor import ApplicationBlockedError, ApplicationTimeoutError

    # Pool default: warm Chromium keep-alive unless user explicitly set 0.
    user_keep_alive_pin = (
        os.environ["ACE_BROWSER_KEEP_ALIVE"]
        if "ACE_BROWSER_KEEP_ALIVE" in os.environ
        else None
    )
    os.environ.setdefault("ACE_BROWSER_KEEP_ALIVE", "1")

    profile = worker_profile_dir(worker_id)
    clear_singleton_locks(profile)
    counts = {"success": 0, "fallback": 0, "failed": 0, "skipped": 0}
    update_one_worker(worker_id, status="idle", company="", profile=str(profile))
    force_cold_next = False

    # Stagger first Chromium launch so 3 windows don't slam open together.
    if launch_stagger_sec > 0:
        logger.info("[%s] waiting %.1fs before first apply (smooth launch)", worker_id, launch_stagger_sec)
        await asyncio.sleep(launch_stagger_sec)

    while True:
        if stop_event.is_set():
            break
        from src.core.telemetry import time_stage

        with time_stage("queue_claim_wait_ms", worker=worker_id):
            item = claim_next(worker_id, host_limiter=host_limiter)
        if item is None:
            if producers_done.is_set():
                # READY empty → peers own any remaining CLAIMED rows; we can stop.
                if queue_counts().get("READY", 0) == 0:
                    break
            await asyncio.sleep(idle_poll_sec)
            continue

        if WAIT_SENTINEL_KEY in item and "id" not in item:
            wait_s = min(30.0, float(item.get(WAIT_SENTINEL_KEY) or 1.0))
            logger.info("[%s] all peeked hosts cooling — sleep %.1fs", worker_id, wait_s)
            with time_stage(
                "queue_claim_wait_ms", worker=worker_id, wait_sentinel=True
            ):
                await asyncio.sleep(wait_s)
            continue

        company = str(item.get("company") or "Unknown")
        job_url = str(item.get("job_url") or "")
        track = str(item.get("track") or "")
        jd_text = str(item.get("jd_text") or "")
        pdf_path = str(item.get("pdf_path") or "")
        yaml_snapshot = str(item.get("yaml_snapshot") or "")
        qid = int(item["id"])

        if host_limiter is not None:
            host_limiter.mark_dispatched(job_url)

        update_one_worker(
            worker_id, status="busy", company=company, job_url=job_url[:180]
        )
        logger.info(
            "[%s] Applying %s @ %s → %s",
            worker_id,
            track or "?",
            company,
            job_url[:120],
        )

        if "linkedin.com" in job_url.lower() and linkedin_limiter is not None:
            await linkedin_limiter.acquire()

        if force_cold_next:
            os.environ["ACE_BROWSER_KEEP_ALIVE"] = "0"
            clear_singleton_locks(profile)
            force_cold_next = False
        elif user_keep_alive_pin is not None:
            os.environ["ACE_BROWSER_KEEP_ALIVE"] = user_keep_alive_pin
        else:
            os.environ["ACE_BROWSER_KEEP_ALIVE"] = "1"

        try:
            outcome = await apply_fn(
                company=company,
                job_url=job_url,
                track=track,
                jd_text=jd_text,
                pdf_path=pdf_path,
                yaml_snapshot=yaml_snapshot,
                master_yaml=master_yaml,
                user_data_dir=profile,
                cleanup_linkedin_resume=False,
            )
        except (ApplicationBlockedError, ApplicationTimeoutError, Exception) as exc:
            logger.warning(
                "[%s] apply crash %s: %s: %s",
                worker_id,
                company,
                type(exc).__name__,
                exc,
            )
            outcome = "failed"
            err_l = f"{type(exc).__name__} {exc}".lower()
            if any(n in err_l for n in _BROWSER_CRASH_NEEDLES):
                force_cold_next = True
                try:
                    from src.core.browser_warmup import invalidate_slot

                    invalidate_slot(profile)
                except Exception:
                    logger.debug("[%s] slot invalidate after crash failed", worker_id, exc_info=True)

        counts[outcome] = counts.get(outcome, 0) + 1
        mark_done(qid, error_log=outcome)
        update_one_worker(worker_id, status="idle", company="", last_outcome=outcome)

        # Between warm jobs, avoid panic lock clears that race a live Chromium.
        if not env_truthy("ACE_BROWSER_KEEP_ALIVE", "1") or force_cold_next:
            clear_singleton_locks(profile)
            if force_cold_next:
                try:
                    from src.core.browser_warmup import invalidate_slot

                    invalidate_slot(profile)
                except Exception:
                    logger.debug("[%s] cold slot invalidate failed", worker_id, exc_info=True)

    # Normal pool exit: soft-blank so the next hunt cycle reuses Chromium.
    # Crash/TTL still invalidate; process exit uses close_all_slots (atexit/SIGTERM).
    try:
        from src.core.browser_warmup import blank_slot, invalidate_slot

        if env_truthy("ACE_BROWSER_KEEP_ALIVE", "1"):
            blank_slot(profile)
        else:
            invalidate_slot(profile)
    except Exception:
        logger.debug("[%s] exit slot cleanup failed", worker_id, exc_info=True)
    update_one_worker(worker_id, status="stopped", company="")
    logger.info("[%s] worker exit counts=%s", worker_id, counts)
    return counts


async def run_browser_pool(
    *,
    master_yaml: Path,
    apply_fn: ApplyFn,
    n_workers: int | None = None,
    producers_done: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    linkedin_max_per_hour: int | None = None,
) -> dict[str, int]:
    """Spawn N browser workers; return merged outcome counts."""
    settings = get_settings()
    n = int(n_workers if n_workers is not None else settings.browser_workers)
    n = max(1, min(8, n))
    try:
        reclaim_orphaned_claims()
    except Exception:
        logger.exception("Pool startup orphan reclaim failed")
    requeue_stale_claims()
    clone_profiles_from_master(n)
    stop = stop_event or asyncio.Event()
    lim = LinkedInRateLimiter(
        max_per_hour=(
            linkedin_max_per_hour
            if linkedin_max_per_hour is not None
            else settings.linkedin_max_per_hour
        )
    )
    host_lim = HostRateLimiter()
    tasks = [
        asyncio.create_task(
            browser_worker_loop(
                f"w{i}",
                master_yaml=master_yaml,
                apply_fn=apply_fn,
                stop_event=stop,
                producers_done=producers_done,
                linkedin_limiter=lim,
                host_limiter=host_lim,
                launch_stagger_sec=float(i - 1) * 2.5,
            ),
            name=f"browser-w{i}",
        )
        for i in range(1, n + 1)
    ]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        try:
            from src.core.browser_warmup import blank_all_slots

            blank_all_slots()
        except Exception:
            logger.debug("blank_all_slots after pool failed", exc_info=True)
    merged = {"success": 0, "fallback": 0, "failed": 0, "skipped": 0}
    for r in results:
        if isinstance(r, dict):
            for k, v in r.items():
                merged[k] = merged.get(k, 0) + int(v)
        elif isinstance(r, Exception):
            logger.exception("Browser worker crashed: %s", r)
    return merged
