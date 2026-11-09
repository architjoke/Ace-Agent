"""Deterministic LinkedIn Easy Apply state machine (sync Playwright)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    close_context_safe,
    goto_dom,
    launch_persistent_context,
    load_applicant_profile,
    success_text,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_WIDGET_TIMEOUT_MS = 5_000
_MAX_STEPS = 12
_OVERALL_BUDGET_S = 90

_EASY_APPLY = re.compile(r"^Easy Apply$", re.I)
_APPLIED = re.compile(r"^Applied$", re.I)
_SUBMIT = re.compile(r"^Submit application$", re.I)
_REVIEW = re.compile(r"^Review$", re.I)
_NEXT = re.compile(r"^(Next|Continue|Save and continue)$", re.I)


def _already_applied_visible(page) -> bool:
    try:
        if page.get_by_role("button", name=_APPLIED).first.is_visible(timeout=800):
            return True
    except Exception:
        pass
    try:
        if page.get_by_text(re.compile(r"you applied on", re.I)).first.is_visible(
            timeout=600
        ):
            return True
    except Exception:
        pass
    return False


def _click_easy_apply(page) -> None:
    btn = page.get_by_role("button", name=_EASY_APPLY).first
    try:
        btn.click(timeout=_WIDGET_TIMEOUT_MS)
    except Exception as exc:
        raise PlaybookFallback(
            f"linkedin playbook: Easy Apply not clickable ({exc})"
        ) from exc
    page.wait_for_timeout(500)


def _fill_step(page, *, answers: dict, pdf: Path) -> None:
    from src.core.semantic_dom import extract_interactive, plan_semantic_actions

    elements = extract_interactive(page)
    if not elements:
        return
    actions = plan_semantic_actions(
        elements, answers=answers, resume_path=str(pdf)
    )
    if not actions:
        return
    # Cap per-action timeouts for stuck detection
    for step in actions:
        ace_id = step.get("id")
        action = step.get("action")
        loc = page.locator(f'[data-ace-id="{ace_id}"]').first
        try:
            if loc.count() == 0:
                continue
            if action == "fill":
                loc.fill(str(step.get("text") or ""), timeout=_WIDGET_TIMEOUT_MS)
            elif action == "click":
                loc.click(timeout=_WIDGET_TIMEOUT_MS)
            elif action == "upload" and pdf.is_file():
                try:
                    tag = (loc.evaluate("el => el.tagName") or "").lower()
                except Exception:
                    tag = ""
                if tag == "input":
                    loc.set_input_files(str(pdf), timeout=_WIDGET_TIMEOUT_MS)
                else:
                    page.locator('input[type="file"]').first.set_input_files(
                        str(pdf), timeout=_WIDGET_TIMEOUT_MS
                    )
        except Exception as exc:
            raise PlaybookFallback(
                f"linkedin playbook: widget stuck action={action} id={ace_id} ({exc})"
            ) from exc


def _advance(page) -> str:
    """Click Submit / Review / Next. Returns submit|review|next|none."""
    for kind, pattern in (
        ("submit", _SUBMIT),
        ("review", _REVIEW),
        ("next", _NEXT),
    ):
        try:
            btn = page.get_by_role("button", name=pattern).first
            if btn.count() == 0:
                continue
            if not btn.is_visible(timeout=800):
                continue
            btn.click(timeout=_WIDGET_TIMEOUT_MS)
            page.wait_for_timeout(600)
            return kind
        except Exception:
            continue
    return "none"


def _confirm_success(page) -> bool:
    try:
        body = page.inner_text("body", timeout=2000) or ""
    except Exception:
        body = ""
    lower = body.lower()
    if success_text(body):
        return True
    markers = (
        "application sent",
        "your application was sent",
        "application submitted",
        "applied",
    )
    return any(m in lower for m in markers)


def run_linkedin(
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
    """Easy Apply state machine. Raises TimeoutError / PlaybookHandoff / AlreadyAppliedError."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    from src.playbooks._common import try_cdp_assist

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
        logger.info("LinkedIn playbook navigating %s", job_url)
        goto_dom(page, job_url)
        from src.core.job_closed import page_shows_job_closed

        if page_shows_job_closed(page):
            from src.core.executor import JobClosedError

            raise JobClosedError(f"LinkedIn job closed/missing at {job_url[:160]}")

        if _already_applied_visible(page):
            from src.core.executor import AlreadyAppliedError

            raise AlreadyAppliedError(
                f"LinkedIn already applied UI at {job_url[:160]}"
            )

        _click_easy_apply(page)

        miss_advance = 0
        for step in range(_MAX_STEPS):
            if time.monotonic() - started > _OVERALL_BUDGET_S:
                raise PlaybookFallback(
                    f"linkedin playbook: overall budget exceeded on {job_url}"
                )

            try:
                _fill_step(page, answers=answers, pdf=pdf)
            except PlaybookFallback as fill_exc:
                if try_cdp_assist(cdp_assist, str(fill_exc)):
                    try:
                        _fill_step(page, answers=answers, pdf=pdf)
                    except PlaybookFallback:
                        raise
                else:
                    raise
            except Exception as exc:
                reason = f"linkedin playbook: step fill failed ({exc})"
                if try_cdp_assist(cdp_assist, reason):
                    try:
                        _fill_step(page, answers=answers, pdf=pdf)
                    except Exception as retry_exc:
                        raise PlaybookFallback(
                            f"linkedin playbook: step fill failed ({retry_exc})"
                        ) from retry_exc
                else:
                    raise PlaybookFallback(reason) from exc

            action = _advance(page)
            if action == "submit":
                page.wait_for_timeout(1200)
                if _confirm_success(page):
                    logger.info("LinkedIn Easy Apply success for %s", job_url[:160])
                    return True
                logger.info(
                    "LinkedIn submit clicked — counting success for %s",
                    job_url[:160],
                )
                return True
            if action == "none":
                miss_advance += 1
                if miss_advance >= 2:
                    reason = "linkedin playbook: Next/Review/Submit missing — Gemini"
                    if try_cdp_assist(cdp_assist, reason):
                        miss_advance = 0
                        continue
                    raise PlaybookFallback(reason)
            else:
                miss_advance = 0

        raise PlaybookFallback(
            "linkedin playbook: too many steps without submit — Gemini"
        )
    except PlaywrightTimeout as exc:
        handoff = context is not None and page is not None
        raise_handoff(
            f"linkedin playbook timed out on {job_url}",
            context=context,
            page=page,
            user_data_dir=user_data_dir,
            playwright=pw,
            cause=exc,
            fallback=TimeoutError(f"linkedin playbook timed out on {job_url}"),
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
                    logger.debug("warm linkedin reset failed", exc_info=True)
