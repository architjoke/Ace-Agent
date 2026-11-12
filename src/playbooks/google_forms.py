"""Google Forms apply — FormFillEngine fill, upload resume, submit."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    close_context_safe,
    confirm_application_success,
    fill_ats_form,
    force_upload_resume,
    goto_dom,
    launch_persistent_context,
    load_applicant_profile,
    settle,
    success_text,
)

logger = logging.getLogger(__name__)

_BUDGET_S = 90
_SUBMIT = re.compile(r"^(Submit|Send|Submit\s+application)$", re.I)


def _looks_like_google_form(url: str) -> bool:
    u = (url or "").lower()
    return "docs.google.com/forms" in u or "forms.gle/" in u


def run_google_forms(
    job_url: str,
    pdf_path: Path | str,
    master_yaml_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
    headless: bool | None = None,
    yaml_snapshot: str | None = None,
    context=None,
    page=None,
    cdp_assist=None,
    **kwargs,
) -> bool:
    """Fill a Google Form application and submit via shared FormFillEngine."""
    _ = kwargs
    _ = cdp_assist
    pdf = Path(pdf_path).expanduser().resolve()
    master = Path(master_yaml_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not master.is_file():
        raise FileNotFoundError(f"master.yaml not found: {master}")

    answers = load_applicant_profile(master, yaml_snapshot=yaml_snapshot)
    owned = context is None or page is None
    pw = None
    started = time.monotonic()
    try:
        if owned:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            context, page = launch_persistent_context(
                pw, user_data_dir=user_data_dir, headless=headless
            )
        assert page is not None and context is not None

        logger.info("Google Forms playbook navigating %s", job_url[:160])
        goto_dom(page, job_url)
        if not _looks_like_google_form(page.url or job_url):
            raise PlaybookFallback(
                f"google_forms: not a Google Form URL ({(page.url or job_url)[:160]})"
            )

        stats = fill_ats_form(
            page,
            profile=answers,
            yaml_snapshot=yaml_snapshot,
            max_pages=3,
        )
        filled = int(stats.get("filled") or 0)
        if time.monotonic() - started > _BUDGET_S:
            raise PlaybookFallback("google_forms: budget exceeded during fill")

        if pdf.is_file():
            try:
                force_upload_resume(page, pdf)
            except Exception:
                logger.debug("google_forms resume upload miss", exc_info=True)

        submitted = False
        try:
            btn = page.get_by_role("button", name=_SUBMIT).first
            if btn.count() and btn.is_visible(timeout=1500):
                btn.click(timeout=4000)
                submitted = True
        except Exception:
            pass
        if not submitted:
            try:
                page.locator("div[role='button']").filter(has_text=_SUBMIT).first.click(
                    timeout=4000
                )
                submitted = True
            except Exception as exc:
                raise PlaybookFallback(
                    f"google_forms: Submit not clickable ({type(exc).__name__})"
                ) from exc

        settle(page, timeout_ms=2000)
        try:
            body = page.inner_text("body", timeout=3000) or ""
        except Exception:
            body = ""
        if (
            confirm_application_success(page)
            or success_text(body)
            or any(
                m in body.lower()
                for m in (
                    "your response has been recorded",
                    "response has been recorded",
                    "thanks for",
                    "thank you",
                )
            )
        ):
            logger.info("Google Forms submit success for %s", job_url[:160])
            return True
        raise PlaybookFallback(
            f"google_forms: no confirmation after submit (filled={filled}, submitted={submitted})"
        )
    finally:
        if owned:
            close_context_safe(context, user_data_dir=user_data_dir)
            if pw is not None:
                try:
                    pw.stop()
                except Exception:
                    pass
        else:
            try:
                from src.core.browser_warmup import reset_context_state

                reset_context_state(context, page)
            except Exception:
                logger.debug("google_forms warm reset failed", exc_info=True)
