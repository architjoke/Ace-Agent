"""Detect closed / missing job postings before spending apply tokens."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

JobProbeResult = Literal["ok", "already_applied", "closed"]

_CLOSED_RE = re.compile(
    r"("
    r"does\s+not\s+exist|"
    r"page\s+does\s+not\s+exist|"
    r"job\s+(is\s+)?(no\s+longer\s+available|not\s+available|closed|expired|removed)|"
    r"this\s+(job|position|posting|role)\s+(has\s+been\s+)?(closed|filled|removed|expired)|"
    r"no\s+longer\s+accepting\s+applications|"
    r"position\s+has\s+been\s+filled|"
    r"sorry[,!]?\s+this\s+job|"
    r"404\s*[-–|]?\s*not\s+found|"
    r"job\s+not\s+found|page\s+not\s+found|"
    r"we\s+can'?t\s+find\s+(this|that)\s+(job|page)|"
    r"can'?t\s+find\s+this\s+job|"
    r"listing\s+(is\s+)?no\s+longer\s+active"
    r")",
    re.IGNORECASE,
)


def page_shows_job_closed(page) -> bool:
    """True when the loaded page clearly indicates the job is gone/closed."""
    try:
        title = (page.title() or "").strip()
    except Exception:
        title = ""
    if _CLOSED_RE.search(title):
        return True
    try:
        body = page.locator("main, [role='main'], body").first.inner_text(timeout=2000)
    except Exception:
        try:
            body = page.content()
        except Exception:
            body = ""
    text = f"{title}\n{body or ''}"
    if _CLOSED_RE.search(text):
        if len(text) < 6000 or re.search(
            r"does\s+not\s+exist|404|no\s+longer\s+available|job\s+closed",
            text,
            re.I,
        ):
            return True
    return False


def probe_on_page(page, job_url: str, *, timeout_ms: int = 15_000) -> JobProbeResult:
    """Closed / already-applied / ok on an existing warm page (no launch/close)."""
    url = (job_url or "").strip()
    if not url or page is None:
        return "ok"
    from src.core.already_applied import page_shows_already_applied
    from src.playbooks._common import goto_dom

    try:
        goto_dom(page, url, timeout_ms=timeout_ms)
        page.wait_for_timeout(400)
        if page_shows_job_closed(page):
            logger.info("DOM fast-fail: job closed/missing at %s", url[:160])
            return "closed"
        if page_shows_already_applied(page):
            logger.info("DOM fast-fail: already applied at %s", url[:160])
            return "already_applied"
        return "ok"
    except Exception:
        logger.warning("Job-page DOM probe failed for %s — continuing apply", url[:160])
        return "ok"


def probe_job_page(
    job_url: str,
    *,
    user_data_dir: Path | str | None = None,
    timeout_ms: int = 15_000,
    page=None,
) -> JobProbeResult:
    """Probe closed/already-applied. Pass ``page`` to reuse warm Chromium."""
    if page is not None:
        return probe_on_page(page, job_url, timeout_ms=timeout_ms)

    url = (job_url or "").strip()
    if not url:
        return "ok"

    from playwright.sync_api import sync_playwright

    from src.playbooks._common import close_context_safe, launch_persistent_context

    context = None
    try:
        with sync_playwright() as p:
            context, pg = launch_persistent_context(
                p, user_data_dir=user_data_dir, timeout_ms=timeout_ms
            )
            try:
                return probe_on_page(pg, url, timeout_ms=timeout_ms)
            finally:
                close_context_safe(context, user_data_dir=user_data_dir)
                context = None
    except Exception:
        logger.warning("Job-page DOM probe failed for %s — continuing apply", url[:160])
        return "ok"
    finally:
        if context is not None:
            try:
                close_context_safe(context, user_data_dir=user_data_dir)
            except Exception:
                pass
