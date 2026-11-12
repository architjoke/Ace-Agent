"""Deterministic Indeed Soft / Easy Apply playbook (sync Playwright)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    close_context_safe,
    fill_css,
    force_upload_resume,
    goto_dom,
    launch_persistent_context,
    load_applicant_profile,
    settle,
    success_text,
    try_cdp_assist,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_WIDGET_MS = 8_000
_MAX_STEPS = 14
_OVERALL_BUDGET_S = 150

_APPLY = re.compile(
    r"^(Apply(\s+now)?|Easily\s+apply|Easy\s+Apply|Continue\s+applying)$",
    re.I,
)
_NEXT = re.compile(r"^(Continue|Next|Save\s+and\s+continue|Review)$", re.I)
_SUBMIT = re.compile(
    r"^(Submit(\s+your)?\s+application|Submit\s+application|Submit)$",
    re.I,
)
_PREPARING = re.compile(r"preparing\s+review", re.I)
_COOKIE = re.compile(r"^(Accept(\s+all)?(\s+cookies)?|Agree|Got\s+it|OK)$", re.I)


def _dismiss_cookies(page) -> None:
    for pattern in (_COOKIE, re.compile(r"accept\s+all", re.I)):
        try:
            btn = page.get_by_role("button", name=pattern).first
            if btn.count() and btn.is_visible(timeout=800):
                btn.click(timeout=3000)
                page.wait_for_timeout(400)
                return
        except Exception:
            continue


def _click_apply(page) -> None:
    errors: list[str] = []
    for factory in (
        lambda: page.get_by_role("button", name=_APPLY).first,
        lambda: page.get_by_role("link", name=_APPLY).first,
        lambda: page.locator(
            "button[id*='indeed-apply'], a[id*='indeed-apply'], "
            "button.jobsearch-IndeedApplyButton, "
            "#indeedApplyButton, button[aria-label*='Apply']"
        ).first,
    ):
        try:
            loc = factory()
            if loc.count() == 0:
                errors.append("miss")
                continue
            if not loc.is_visible(timeout=2000):
                errors.append("hidden")
                continue
            try:
                loc.scroll_into_view_if_needed(timeout=1000)
            except Exception:
                pass
            loc.click(timeout=5000)
            page.wait_for_timeout(800)
            return
        except Exception as exc:
            errors.append(type(exc).__name__)
    raise PlaybookFallback(
        "indeed playbook: Apply control not clickable ("
        + "; ".join(errors[:8])
        + ")"
    )


def _wait_preparing_review(page, *, timeout_s: float = 45.0) -> None:
    """Wait out Soft Apply 'Preparing review' without stuck-FP abort."""
    deadline = time.monotonic() + max(5.0, timeout_s)
    while time.monotonic() < deadline:
        try:
            body = (page.inner_text("body", timeout=1500) or "").lower()
        except Exception:
            body = ""
        if not _PREPARING.search(body):
            return
        # Still preparing — poll; do not treat as stuck
        try:
            page.wait_for_timeout(1500)
        except Exception:
            time.sleep(1.0)
    logger.warning("indeed playbook: Preparing review still visible after wait")


def _fill_identity(page, answers: dict) -> None:
    mapping = [
        (["input[name*='name' i]", "input[autocomplete='name']", "#input-applicant.name"], answers.get("full_name") or ""),
        (["input[type='email']", "input[name*='email' i]", "input[autocomplete='email']"], answers.get("email") or ""),
        (
            ["input[type='tel']", "input[name*='phone' i]", "input[autocomplete='tel']"],
            answers.get("phone_display")
            or answers.get("phone_e164")
            or answers.get("phone_national")
            or "",
        ),
    ]
    for selectors, value in mapping:
        if value:
            fill_css(page, selectors, str(value))


def _advance(page) -> str:
    for kind, pattern in (("submit", _SUBMIT), ("next", _NEXT)):
        try:
            btn = page.get_by_role("button", name=pattern).first
            if btn.count() == 0 or not btn.is_visible(timeout=1000):
                continue
            btn.click(timeout=_WIDGET_MS)
            page.wait_for_timeout(700)
            return kind
        except Exception:
            continue
    return "none"


def _confirm_success(page) -> bool:
    try:
        body = page.inner_text("body", timeout=2500) or ""
    except Exception:
        body = ""
    if success_text(body):
        return True
    lower = body.lower()
    return any(
        m in lower
        for m in (
            "application submitted",
            "your application has been submitted",
            "successfully applied",
            "you applied",
        )
    )


def _amazon_or_hard_captcha(page) -> bool:
    try:
        body = (page.inner_text("body", timeout=1500) or "").lower()
        title = (page.title() or "").lower()
    except Exception:
        return False
    blob = f"{title}\n{body}"
    needles = (
        "amzn-captcha",
        "amazon captcha",
        "confirm you are human",
        "let's confirm you are human",
        "solve this puzzle",
        "unusual activity from your device",
    )
    return any(n in blob for n in needles)


def run_indeed(
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
    """Indeed Soft/Easy Apply. Raises PlaybookHandoff / PlaybookFallback / closed errors."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    _ = kwargs
    pdf = Path(pdf_path).expanduser().resolve()
    master = Path(master_yaml_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not master.is_file():
        raise FileNotFoundError(f"master.yaml not found: {master}")

    answers = load_applicant_profile(master, yaml_snapshot=yaml_snapshot)
    owned = context is None
    handoff = False
    pw = None
    if owned:
        pw = sync_playwright().start()
    started = time.monotonic()
    try:
        if owned:
            context, page = launch_persistent_context(
                pw,
                user_data_dir=user_data_dir,
                headless=headless,
                timeout_ms=30_000,
            )
        logger.info("Indeed playbook navigating %s", job_url)
        goto_dom(page, job_url)
        from src.core.job_closed import page_shows_job_closed

        if page_shows_job_closed(page):
            from src.core.executor import JobClosedError

            raise JobClosedError(f"Indeed job closed/missing at {job_url[:160]}")

        if _amazon_or_hard_captcha(page):
            from src.core.human_assist import ensure_captcha_cleared

            ensure_captcha_cleared(
                page,
                company=str(kwargs.get("company") or "Indeed"),
                job_url=job_url,
                site="indeed",
            )

        _dismiss_cookies(page)
        _click_apply(page)
        settle(page, timeout_ms=1200)

        if _amazon_or_hard_captcha(page):
            from src.core.human_assist import ensure_captcha_cleared

            ensure_captcha_cleared(
                page,
                company=str(kwargs.get("company") or "Indeed"),
                job_url=job_url,
                site="indeed",
            )

        # Soft Apply often needs Edit (beside CV on review) → upload other resume.
        uploaded = force_upload_resume(page, pdf)
        if not uploaded:
            reason = "indeed playbook: resume upload failed"
            if try_cdp_assist(cdp_assist, reason, page=page):
                uploaded = force_upload_resume(page, pdf)
            if not uploaded:
                raise PlaybookFallback(reason)

        _fill_identity(page, answers)

        miss = 0
        for step in range(_MAX_STEPS):
            if time.monotonic() - started > _OVERALL_BUDGET_S:
                raise PlaybookFallback(
                    f"indeed playbook: overall budget exceeded on {job_url}"
                )
            _wait_preparing_review(page, timeout_s=20.0)
            # Review step: Edit beside CV → upload tailored PDF → Continue.
            force_upload_resume(page, pdf)
            _fill_identity(page, answers)
            action = _advance(page)
            if action == "submit":
                _wait_preparing_review(page, timeout_s=25.0)
                settle(page, timeout_ms=1500)
                if _confirm_success(page):
                    logger.info("Indeed Soft Apply success for %s", job_url[:160])
                    return True
                raise PlaybookFallback(
                    f"indeed playbook: submit clicked without confirmation for {job_url}"
                )
            if action == "none":
                miss += 1
                if miss >= 2:
                    reason = "indeed playbook: Continue/Submit missing — Gemini"
                    if try_cdp_assist(cdp_assist, reason, page=page):
                        miss = 0
                        continue
                    raise PlaybookFallback(reason)
            else:
                miss = 0

        raise PlaybookFallback("indeed playbook: too many steps without submit — Gemini")
    except PlaywrightTimeout as exc:
        handoff = context is not None and page is not None
        raise_handoff(
            f"indeed playbook timed out on {job_url}",
            context=context,
            page=page,
            user_data_dir=user_data_dir,
            playwright=pw,
            cause=exc,
            fallback=TimeoutError(f"indeed playbook timed out on {job_url}"),
        )
    except PlaybookFallback as exc:
        handoff = context is not None and page is not None
        raise_handoff(
            str(exc),
            context=context,
            page=page,
            user_data_dir=user_data_dir,
            playwright=pw,
            cause=exc,
        )
    finally:
        if not handoff:
            if owned:
                close_context_safe(context, user_data_dir=user_data_dir)
                if pw is not None:
                    try:
                        pw.stop()
                    except Exception:
                        pass
            else:
                from src.core.browser_warmup import reset_context_state

                try:
                    reset_context_state(context, page)
                except Exception:
                    logger.debug("warm indeed reset failed", exc_info=True)
