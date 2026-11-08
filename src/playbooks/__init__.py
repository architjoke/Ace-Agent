"""Deterministic ATS playbooks (DOM) with Gemini vision fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PlaybookFallback(Exception):
    """Raised when a DOM playbook cannot finish — hand off to Gemini."""


# Re-export handoff types for callers (session keeps Playwright alive for CDP).
from src.playbooks.session import PlaybookHandoff, PlaybookSession  # noqa: E402


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


def resolve_ats_kind(
    job_url: str,
    *,
    user_data_dir: Path | str | None = None,
) -> str:
    """Classify ATS using host heuristics, then resolved URL + iframe scan if needed."""
    url = (job_url or "").strip()
    kind = classify_ats(url)
    if kind != "other":
        return kind

    # Careers-page embeds: open briefly and inspect final URL / iframes.
    try:
        from playwright.sync_api import sync_playwright

        from src.playbooks._common import close_context_safe, goto_dom, launch_persistent_context

        context = None
        with sync_playwright() as p:
            context, page = launch_persistent_context(
                p, user_data_dir=user_data_dir, timeout_ms=15_000
            )
            try:
                goto_dom(page, url)
                resolved = classify_ats(page.url or url)
                if resolved != "other":
                    logger.info("ATS resolve via final URL → %s (%s)", resolved, page.url[:120])
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
            finally:
                close_context_safe(context, user_data_dir=user_data_dir)
                context = None
    except Exception:
        logger.warning("resolve_ats_kind probe failed for %s", url[:160], exc_info=True)
    return "other"
