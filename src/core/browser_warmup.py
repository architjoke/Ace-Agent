"""Auth-safe Chromium context reset + worker warm slots for playbook reuse."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _slot_ttl_seconds() -> float:
    try:
        hours = float(os.environ.get("ACE_BROWSER_SLOT_TTL_HOURS", "6") or "6")
    except ValueError:
        hours = 6.0
    if hours <= 0:
        return float("inf")  # disabled
    return hours * 3600.0

# Documented allowlist — we never clear cookie jars for these hosts.
AUTH_COOKIE_HOST_SUFFIXES = (
    "google.com",
    "accounts.google.com",
    "linkedin.com",
    "www.linkedin.com",
    "github.com",
)

_SLOTS: dict[str, "WorkerBrowserSlot"] = {}
_SLOTS_LOCK = threading.Lock()


def browser_alive(context: Any) -> bool:
    """True if the Playwright context still looks usable."""
    if context is None:
        return False
    try:
        list(context.pages)
    except Exception:
        return False
    try:
        browser = getattr(context, "browser", None)
        if browser is not None and hasattr(browser, "is_connected"):
            if not browser.is_connected():
                return False
    except Exception:
        return False
    return True


def reset_context_state(context: Any, page: Any = None) -> Any:
    """Close extra tabs and blank the survivor without wiping auth cookies.

    NEVER calls ``context.clear_cookies()`` — persistent profile Cookie DB
    (Google / LinkedIn sessions) must survive between jobs.
    """
    if context is None:
        raise RuntimeError("warm browser unhealthy: missing context")

    # Refuse cookie-store wipes if a caller monkeypatched something odd.
    clear_cookies = getattr(context, "clear_cookies", None)

    pages = []
    try:
        pages = list(context.pages)
    except Exception:
        pages = []

    survivor = page if page is not None and page in pages else (pages[0] if pages else None)
    for p in pages:
        if p is survivor:
            continue
        try:
            p.close()
        except Exception:
            logger.debug("extra tab close failed", exc_info=True)

    if survivor is None:
        try:
            survivor = context.new_page()
        except Exception as exc:
            raise RuntimeError("warm browser unhealthy after reset") from exc

    # Storage clear on the *current* document only (not cookie jar).
    try:
        survivor.evaluate(
            "() => { try { sessionStorage.clear(); } catch (e) {} "
            "try { localStorage.clear(); } catch (e) {} }"
        )
    except Exception:
        logger.debug("storage clear skipped", exc_info=True)

    try:
        survivor.goto("about:blank", wait_until="domcontentloaded", timeout=5000)
    except Exception:
        logger.debug("about:blank navigation skipped", exc_info=True)

    # Explicit non-call: never invoke clear_cookies
    del clear_cookies

    if not browser_alive(context):
        raise RuntimeError("warm browser unhealthy after reset")
    return survivor


@dataclass
class WorkerBrowserSlot:
    """Warm Playwright persistent context keyed by worker profile path.

    Playwright sync API is greenlet-bound to the thread that created the
    browser. All Chromium work for a slot MUST run on that owner thread via
    ``executor()`` / ``run()`` — never via a shared ``asyncio.to_thread`` pool.
    """

    user_data_dir: Path
    playwright: Any = None
    context: Any = None
    page: Any = None
    born_at: float | None = None
    _headless: bool | None = field(default=None, repr=False)
    creator_thread_ident: int | None = field(default=None, repr=False)
    _owner_executor: ThreadPoolExecutor | None = field(default=None, repr=False)

    def executor(self) -> ThreadPoolExecutor:
        """Single-thread executor pinned for this slot's Playwright lifetime."""
        if self._owner_executor is None:
            name = Path(self.user_data_dir).name[:24] or "slot"
            self._owner_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"ace-pw-{name}",
            )
        return self._owner_executor

    def run(self, fn: Callable[..., _T], *args: Any, timeout: float | None = 600.0, **kwargs: Any) -> _T:
        """Execute ``fn`` on the slot owner thread (direct call if already there)."""
        if (
            self.creator_thread_ident is not None
            and threading.get_ident() == self.creator_thread_ident
        ):
            return fn(*args, **kwargs)

        def _wrapped() -> _T:
            if self.creator_thread_ident is None:
                self.creator_thread_ident = threading.get_ident()
            elif threading.get_ident() != self.creator_thread_ident:
                # Executor thread changed (should not happen with max_workers=1).
                self.creator_thread_ident = threading.get_ident()
            return fn(*args, **kwargs)

        return self.executor().submit(_wrapped).result(timeout=timeout)

    def _ttl_expired(self) -> bool:
        if self.born_at is None:
            return False
        return (time.monotonic() - self.born_at) > _slot_ttl_seconds()

    def ensure(
        self,
        *,
        headless: bool | None = None,
        timeout_ms: int = 30_000,
    ) -> tuple[Any, Any]:
        """Return (context, page), launching Chromium once if needed.

        Must be called from this slot's owner thread (see ``run`` / ``executor``).
        """
        if self.creator_thread_ident is None:
            self.creator_thread_ident = threading.get_ident()
        elif threading.get_ident() != self.creator_thread_ident:
            raise RuntimeError(
                "WorkerBrowserSlot.ensure called off owner thread "
                f"(owner={self.creator_thread_ident}, current={threading.get_ident()}). "
                "Use slot.run(...) / slot.executor()."
            )

        if self.context is not None and browser_alive(self.context) and not self._ttl_expired():
            pages = list(self.context.pages)
            self.page = self.page if self.page in pages else (pages[0] if pages else None)
            if self.page is None:
                self.page = self.context.new_page()
            return self.context, self.page

        if self._ttl_expired():
            logger.info(
                "warm slot TTL expired (%.1fh) — relaunching %s",
                _slot_ttl_seconds() / 3600.0,
                str(self.user_data_dir)[:120],
            )

        self._close_browser_only()
        from playwright.sync_api import sync_playwright

        from src.core.chromium_runtime import ensure_playwright_chromium
        from src.playbooks._common import launch_persistent_context

        ensure_playwright_chromium()
        self.playwright = sync_playwright().start()
        self.context, self.page = launch_persistent_context(
            self.playwright,
            user_data_dir=self.user_data_dir,
            headless=headless if headless is not None else self._headless,
            timeout_ms=timeout_ms,
        )
        self._headless = headless if headless is not None else self._headless
        self.born_at = time.monotonic()
        self.creator_thread_ident = threading.get_ident()
        return self.context, self.page

    def reset(self) -> None:
        if self.context is None:
            return
        try:
            self.page = reset_context_state(self.context, self.page)
        except Exception:
            logger.warning("warm slot reset failed — closing slot", exc_info=True)
            self.close()

    def _close_browser_only(self) -> None:
        from src.playbooks._common import close_context_safe

        try:
            close_context_safe(self.context, user_data_dir=self.user_data_dir)
        except Exception:
            logger.debug("slot context close failed", exc_info=True)
        self.context = None
        self.page = None
        self.born_at = None
        pw = self.playwright
        self.playwright = None
        if pw is not None:
            try:
                pw.stop()
            except Exception:
                logger.debug("slot playwright.stop failed", exc_info=True)

    def close(self) -> None:
        """Close Chromium; keep executor alive so the next ensure reuses the thread."""
        if (
            self.creator_thread_ident is not None
            and threading.get_ident() != self.creator_thread_ident
            and self._owner_executor is not None
        ):
            try:
                self.run(self._close_browser_only, timeout=60.0)
                return
            except Exception:
                logger.debug("slot.close marshal failed — closing inline", exc_info=True)
        self._close_browser_only()

    def shutdown(self) -> None:
        """Close browser and destroy the owner executor (slot leave registry)."""
        try:
            self.close()
        finally:
            ex = self._owner_executor
            self._owner_executor = None
            self.creator_thread_ident = None
            if ex is not None:
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    logger.debug("slot executor shutdown failed", exc_info=True)


def slot_key(user_data_dir: Path | str) -> str:
    return str(Path(user_data_dir).expanduser().resolve())


def get_or_create_slot(user_data_dir: Path | str) -> WorkerBrowserSlot:
    key = slot_key(user_data_dir)
    with _SLOTS_LOCK:
        slot = _SLOTS.get(key)
        if slot is None:
            slot = WorkerBrowserSlot(user_data_dir=Path(key))
            _SLOTS[key] = slot
        return slot


def invalidate_slot(user_data_dir: Path | str | None) -> None:
    """Destroy slot Chromium. Crash / TTL recreate / process exit only — not cycle ends."""
    if not user_data_dir:
        return
    key = slot_key(user_data_dir)
    with _SLOTS_LOCK:
        slot = _SLOTS.pop(key, None)
    if slot is not None:
        slot.shutdown()
    try:
        from src.playbooks.session import clear_stale_devtools_port

        clear_stale_devtools_port(key)
    except Exception:
        logger.debug("clear_stale_devtools_port after invalidate failed", exc_info=True)


def blank_slot(user_data_dir: Path | str | None) -> bool:
    """Soft-reset one slot to about:blank; keep registry + Chromium for next cycle."""
    if not user_data_dir:
        return False
    key = slot_key(user_data_dir)
    with _SLOTS_LOCK:
        slot = _SLOTS.get(key)
    if slot is None:
        return False
    if slot.context is None or not browser_alive(slot.context):
        invalidate_slot(key)
        return False
    try:
        slot.reset()
        return True
    except Exception:
        logger.debug("blank_slot failed — invalidate", exc_info=True)
        invalidate_slot(key)
        return False


def blank_all_slots() -> int:
    """Reset live slots to about:blank; drop dead ones. Keeps registry for next cycle."""
    with _SLOTS_LOCK:
        keys = list(_SLOTS.keys())
    return sum(1 for key in keys if blank_slot(key))


def close_all_slots() -> int:
    """Close every warm WorkerBrowserSlot and clear the process registry.

    Call on process exit / crash only — not between hunt cycles.
    """
    with _SLOTS_LOCK:
        items = list(_SLOTS.items())
        _SLOTS.clear()
    n = 0
    for _key, slot in items:
        try:
            slot.shutdown()
            n += 1
        except Exception:
            logger.debug("close_all_slots: slot close failed", exc_info=True)
    return n
