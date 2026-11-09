"""Deterministic ATS playbooks (DOM) with Gemini vision fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PlaybookFallback(Exception):
    """Raised when a DOM playbook cannot finish — hand off to Gemini."""


# Re-export handoff types for callers (session keeps Playwright alive for CDP).
from src.playbooks.session import (  # noqa: E402
    PlaybookHandoff as PlaybookHandoff,
    PlaybookSession as PlaybookSession,
)


def classify_ats(job_url: str) -> str:
    """Return ats kind: linkedin | greenhouse | lever | ashby | workday | other."""
    host = (urlparse(job_url or "").hostname or "").lower()
    path = (urlparse(job_url or "").path or "").lower()
    blob = f"{host}{path}"
    if "linkedin.com" in host:
        return "linkedin"
    if "boards.greenhouse.io" in host or host.endswith("greenhouse.io"):
        return "greenhouse"
    if "greenhouse.io" in blob and "/jobs/" in path:
        return "greenhouse"
    if "ashbyhq.com" in host or "jobs.ashbyhq.com" in host:
        return "ashby"
    if "jobs.lever.co" in host or host.endswith("lever.co"):
        return "lever"
    if "myworkdayjobs.com" in host or ("workday.com" in host and "/job/" in path):
        return "workday"
    if host.startswith("wd") and "myworkdayjobs.com" in host:
        return "workday"
    return "other"


def _iframe_ats_kind(srcs: list[str]) -> str | None:
    for src in srcs:
        kind = classify_ats(src)
        if kind in {"greenhouse", "lever", "ashby", "workday"}:
            return kind
    return None


def _resolve_ats_on_page(page, url: str) -> str:
    """Inspect current/final URL + iframes on an open page."""
    from src.playbooks._common import goto_dom

    try:
        if url and (not getattr(page, "url", None) or url not in (page.url or "")):
            goto_dom(page, url)
    except Exception:
        pass
    resolved = classify_ats(page.url or url)
    if resolved != "other":
        logger.info("ATS resolve via final URL → %s (%s)", resolved, (page.url or "")[:120])
        return resolved
    srcs: list[str] = []
    try:
        frames = page.locator("iframe[src]").evaluate_all(
            "els => els.map(e => e.src || '')"
        )
        if isinstance(frames, list):
            srcs = [str(s) for s in frames if s]
    except Exception:
        for i in range(min(page.locator("iframe").count(), 12)):
            try:
                src = page.locator("iframe").nth(i).get_attribute("src") or ""
                if src:
                    srcs.append(src)
            except Exception:
                continue
    embedded = _iframe_ats_kind(srcs)
    if embedded:
        logger.info("ATS resolve via iframe → %s", embedded)
        return embedded
    return "other"


def resolve_ats_kind(
    job_url: str,
    *,
    user_data_dir: Path | str | None = None,
    page=None,
) -> str:
    """Classify ATS. Pass ``page`` to reuse warm Chromium (no launch/close)."""
    url = (job_url or "").strip()
    kind = classify_ats(url)
    if kind != "other":
        return kind
    if page is not None:
        try:
            return _resolve_ats_on_page(page, url)
        except Exception:
            logger.warning("resolve_ats_kind on warm page failed for %s", url[:160], exc_info=True)
            return "other"
    try:
        from playwright.sync_api import sync_playwright

        from src.playbooks._common import close_context_safe, launch_persistent_context

        with sync_playwright() as p:
            context, pg = launch_persistent_context(
                p, user_data_dir=user_data_dir, timeout_ms=15_000
            )
            try:
                return _resolve_ats_on_page(pg, url)
            finally:
                close_context_safe(context, user_data_dir=user_data_dir)
    except Exception:
        logger.warning("resolve_ats_kind probe failed for %s", url[:160], exc_info=True)
    return "other"
