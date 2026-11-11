"""Redirect-board bridge: WWR / Himalayas / Hiring Cafe → ATS on primary warm tab.

Critical: new Apply tabs are captured, URL resolved, popup closed, then the
primary WorkerBrowserSlot page navigates to the ATS URL (no tab leaks).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from src.playbooks import PlaybookFallback, classify_ats
from src.playbooks._common import (
    capture_popup_url_then_close,
    close_context_safe,
    close_extra_pages,
    goto_dom,
    launch_persistent_context,
    try_cdp_assist,
    wait_for_any,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_APPLY = re.compile(
    r"^(Apply(\s+for\s+this\s+position)?|Apply\s+now|Apply\s+here|"
    r"Submit\s+application|Apply\s+for\s+this\s+job)$",
    re.I,
)
_COOKIE = re.compile(
    r"^(Accept(\s+all)?(\s+cookies)?|Agree|Got\s+it|OK|I\s+agree)$", re.I
)
_BUDGET_S = 45
_ATS = frozenset(
    {
        "greenhouse",
        "lever",
        "ashby",
        "workday",
        "indeed",
        "linkedin",
        "google_forms",
        "career_site",
    }
)


def _dismiss_cookies(page) -> None:
    try:
        btn = page.get_by_role("button", name=_COOKIE).first
        btn.wait_for(state="visible", timeout=700)
        btn.click(timeout=2500)
    except Exception:
        pass


def _apply_locator(page):
    """Fallback-chained Apply control (link | button | known CSS)."""
    return (
        page.get_by_role("link", name=_APPLY)
        .or_(page.get_by_role("button", name=_APPLY))
        .or_(
            page.locator(
                "a[href^='mailto:'], a[href*='greenhouse'], a[href*='lever.co'], "
                "a[href*='ashbyhq'], a[href*='myworkdayjobs'], "
                "a.application-link, a.apply-button, button.apply, "
                "a:has-text('Apply'), button:has-text('Apply')"
            )
        )
        .first
    )


def _click_apply(page) -> None:
    loc = _apply_locator(page)
    try:
        loc.wait_for(state="visible", timeout=4000)
        loc.scroll_into_view_if_needed(timeout=800)
        loc.click(timeout=5000)
        return
    except Exception as exc:
        raise PlaybookFallback(
            f"aggregator bridge: Apply not clickable ({type(exc).__name__})"
        ) from exc


def _mailto_from_page(page) -> str | None:
    try:
        loc = page.locator("a[href^='mailto:']").first
        if loc.count():
            href = loc.get_attribute("href") or ""
            if href.lower().startswith("mailto:"):
                return href
    except Exception:
        pass
    return None


def _dispatch_ats(kind: str, page, ctx, job_url, pdf, master, **kwargs) -> bool:
    """Hand off to an existing ATS playbook on the same warm page (no executor import)."""
    run = None
    if kind == "greenhouse":
        from src.playbooks.greenhouse import run_greenhouse as run
    elif kind in {"lever", "ashby"}:
        from src.playbooks.lever import run_lever as run
    elif kind == "workday":
        from src.playbooks.workday import run_workday as run
    elif kind == "indeed":
        from src.playbooks.indeed import run_indeed as run
    elif kind == "linkedin":
        from src.playbooks.linkedin import run_linkedin as run
    elif kind == "google_forms":
        from src.playbooks.google_forms import run_google_forms as run
    elif kind == "career_site":
        from src.playbooks.career_site import run_career_site as run
    if run is None:
        return False
    local = dict(kwargs)
    local.update({"page": page, "context": ctx})
    if kind in {"lever", "ashby"}:
        local["ats"] = kind
    return bool(run(job_url, pdf, master, **local))


def run_aggregator(
    job_url: str,
    pdf_path: Path | str,
    master_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
    yaml_snapshot: str | None = None,
    context=None,
    page=None,
    cdp_assist=None,
    company: str = "",
    job_title: str = "",
    jd_text: str = "",
    **_kwargs,
) -> bool:
    """Follow Apply → mailto / known ATS on primary tab; else PlaybookFallback."""
    pdf = Path(pdf_path)
    master = Path(master_path)
    owns = page is None or context is None
    handoff = False
    ctx = context
    pg = page
    pw = None
    deadline = time.monotonic() + _BUDGET_S
    try:
        if owns:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            ctx, pg = launch_persistent_context(pw, user_data_dir=user_data_dir)
        assert pg is not None and ctx is not None

        goto_dom(pg, job_url)
        _dismiss_cookies(pg)
        wait_for_any(
            pg,
            [
                "a:has-text('Apply')",
                "button:has-text('Apply')",
                "a[href^='mailto:']",
            ],
            timeout_ms=5000,
        )

        kind0 = classify_ats(pg.url or job_url)
        if kind0 in _ATS:
            return _dispatch_ats(
                kind0,
                pg,
                ctx,
                pg.url,
                pdf,
                master,
                user_data_dir=user_data_dir,
                yaml_snapshot=yaml_snapshot,
                cdp_assist=cdp_assist,
            )

        mailto = _mailto_from_page(pg)
        try:
            final_url = capture_popup_url_then_close(
                ctx, pg, lambda: _click_apply(pg), popup_timeout_ms=6000
            )
        except PlaybookFallback as exc:
            if try_cdp_assist(cdp_assist, str(exc)):
                return True
            raise

        if (final_url or "").lower().startswith("mailto:"):
            mailto = final_url
        if not mailto:
            mailto = _mailto_from_page(pg)

        if mailto:
            from src.core.mailto_apply import send_mailto_application

            close_extra_pages(ctx, keep=pg)
            send_mailto_application(
                mailto_url=mailto,
                company=company or urlparse(job_url).hostname or "Company",
                job_title=job_title or "",
                jd_text=jd_text or "",
                pdf_path=pdf,
                yaml_snapshot=yaml_snapshot or "",
            )
            return True

        # Same-tab apply may already be on ATS; else navigate primary warm tab.
        kind = classify_ats(final_url or pg.url or "")
        if kind == "other" and final_url and final_url != (pg.url or ""):
            goto_dom(pg, final_url)
            kind = classify_ats(pg.url or final_url)
        elif kind in {"greenhouse", "lever", "ashby", "workday"} and final_url:
            if classify_ats(pg.url or "") != kind:
                goto_dom(pg, final_url)

        close_extra_pages(ctx, keep=pg)
        kind = classify_ats(pg.url or final_url or "")
        if kind in {"greenhouse", "lever", "ashby", "workday"}:
            logger.info("Aggregator bridge → %s on primary tab (%s)", kind, (pg.url or "")[:120])
            return _dispatch_ats(
                kind,
                pg,
                ctx,
                pg.url or final_url,
                pdf,
                master,
                user_data_dir=user_data_dir,
                yaml_snapshot=yaml_snapshot,
                cdp_assist=cdp_assist,
            )

        if time.monotonic() > deadline:
            raise PlaybookFallback("aggregator bridge: budget exhausted")
        reason = f"aggregator bridge: unresolved destination {(final_url or pg.url or '')[:160]}"
        if try_cdp_assist(cdp_assist, reason):
            return True
        raise PlaybookFallback(reason)
    except PlaybookFallback as exc:
        handoff = ctx is not None and pg is not None
        raise_handoff(
            str(exc),
            context=ctx,
            page=pg,
            user_data_dir=user_data_dir,
            playwright=pw,
            cause=exc,
        )
    finally:
        if owns and not handoff:
            close_context_safe(ctx, user_data_dir=user_data_dir)
            if pw is not None:
                try:
                    pw.stop()
                except Exception:
                    pass
    return False
