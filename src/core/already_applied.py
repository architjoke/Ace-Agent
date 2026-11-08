"""Deterministic Playwright probe: detect 'already applied' before LLM burn."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# High-confidence banners / labels (not lone JD prose).
_BANNER_RE = re.compile(
    r"(already\s+applied|you\s+applied\s+on|you\s+have\s+already\s+(applied|submitted)|"
    r"already\s+submitted\s+an?\s+application|application\s+already\s+submitted)",
    re.IGNORECASE,
)

# Confirmation pages that imply apply already done (ATS thank-you).
_CONFIRM_RE = re.compile(
    r"(thanks?\s+for\s+applying|thank\s+you\s+for\s+applying|"
    r"we\s+(have\s+)?received\s+your\s+application|application\s+received)",
    re.IGNORECASE,
)


def _locator_hit(page, locator_factory, *, timeout_ms: int = 1200) -> bool:
    try:
        loc = locator_factory()
        return bool(loc.first.is_visible(timeout=timeout_ms))
    except Exception:
        return False


def _disabled_applied_control(page) -> bool:
    """LinkedIn / ATS Applied button that is disabled or aria-disabled."""
    candidates = (
        lambda: page.get_by_role("button", name=re.compile(r"^applied$", re.I)),
        lambda: page.locator('button:has-text("Applied")'),
        lambda: page.locator('[aria-label*="Applied" i]'),
    )
    for factory in candidates:
        try:
            loc = factory().first
            if loc.count() == 0:
                continue
            if not loc.is_visible(timeout=800):
                continue
            disabled = loc.is_disabled()
            aria = (loc.get_attribute("aria-disabled") or "").strip().lower()
            if disabled or aria in {"true", "1"}:
                return True
            # Visible Applied control with no Easy Apply sibling → treat as done
            if _locator_hit(
                page,
                lambda: page.get_by_role(
                    "button", name=re.compile(r"easy\s*apply|apply\s*now", re.I)
                ),
                timeout_ms=400,
            ):
                continue
            return True
        except Exception:
            continue
    return False


def _banner_hit(page) -> bool:
    checks = (
        lambda: page.get_by_text(re.compile(r"already\s+applied", re.I)),
        lambda: page.get_by_text(re.compile(r"you\s+applied\s+on", re.I)),
        lambda: page.get_by_text(
            re.compile(r"you\s+have\s+already\s+(applied|submitted)", re.I)
        ),
        lambda: page.get_by_text(
            re.compile(r"already\s+submitted\s+an?\s+application", re.I)
        ),
        lambda: page.get_by_text(
            re.compile(r"application\s+already\s+submitted", re.I)
        ),
    )
    return any(_locator_hit(page, c) for c in checks)


def page_shows_already_applied(page) -> bool:
    """Return True when the loaded job page clearly indicates a prior apply."""
    if _banner_hit(page):
        return True
    if _disabled_applied_control(page):
        return True
    try:
        body = page.locator("main, [role='main'], body").first.inner_text(timeout=2000)
    except Exception:
        try:
            body = page.content()
        except Exception:
            body = ""
    text = body or ""
    if _BANNER_RE.search(text):
        return True
    # Confirmation-only pages (post-submit ATS) — require short page / header signal
    if _CONFIRM_RE.search(text) and len(text) < 4000:
        return True
    return False


def probe_already_applied(
    job_url: str,
    *,
    user_data_dir: Path | str | None = None,
    timeout_ms: int = 15_000,
) -> bool:
    """Open job_url with profile cookies; True if already applied. Always closes context."""
    url = (job_url or "").strip()
    if not url:
        return False

    from playwright.sync_api import sync_playwright

    from src.playbooks._common import goto_dom, launch_persistent_context

    context = None
    try:
        with sync_playwright() as p:
            context, page = launch_persistent_context(
                p,
                user_data_dir=user_data_dir,
                timeout_ms=timeout_ms,
            )
            try:
                goto_dom(page, url, timeout_ms=timeout_ms)
                page.wait_for_timeout(400)
                hit = page_shows_already_applied(page)
                if hit:
                    logger.info("DOM fast-fail: already applied detected at %s", url[:160])
                return hit
            finally:
                try:
                    context.close()
                except Exception:
                    logger.exception("Failed to close already-applied probe context")
                context = None
    except Exception:
        logger.warning("Already-applied DOM probe failed for %s — continuing apply", url[:160])
        return False
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
