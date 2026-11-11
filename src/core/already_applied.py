"""Deterministic Playwright probe: detect 'already applied' before LLM burn."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# High-confidence banners / labels (not lone JD prose).
_BANNER_RE = re.compile(
    r"(already\s+applied|you\s+applied\s+on|you\s+have\s+already\s+(applied|submitted)|"
    r"already\s+submitted\s+an?\s+application|application\s+already\s+submitted|"
    r"application\s+submitted|your\s+application\s+(was|has\s+been)\s+submitted|"
    r"your\s+application\s+was\s+sent|application\s+sent|"
    r"you\s+applied\s+(for\s+this|to\s+this))",
    re.IGNORECASE,
)

# Confirmation pages that imply apply already done (ATS thank-you).
_CONFIRM_RE = re.compile(
    r"(thanks?\s+for\s+applying|thank\s+you\s+for\s+applying|"
    r"we\s+(have\s+)?received\s+your\s+application|application\s+received|"
    r"application\s+submitted|your\s+application\s+(was|has\s+been)\s+submitted)",
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
        # DigiPlus / LinkedIn post-submit chrome often says this without "already".
        lambda: page.get_by_text(
            re.compile(r"^application\s+submitted$", re.I)
        ),
        lambda: page.get_by_text(
            re.compile(
                r"your\s+application\s+(was|has\s+been)\s+(submitted|sent)",
                re.I,
            )
        ),
        lambda: page.get_by_text(re.compile(r"application\s+sent", re.I)),
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


# probe_already_applied removed — use src.core.job_closed.probe_job_page
# (closed + already-applied in one Chromium open).
