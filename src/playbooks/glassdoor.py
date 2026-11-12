"""Glassdoor — external ATS on primary tab; Easy Apply A/B; CF hard-exit; force upload."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback, classify_ats
from src.playbooks._common import (
    capture_popup_url_then_close,
    close_context_safe,
    close_extra_pages,
    force_upload_resume,
    goto_dom,
    human_type,
    launch_persistent_context,
    raise_if_cloudflare,
    resolve_pitch_short,
    success_text,
    confirm_application_success,
    try_cdp_assist,
    wait_for_any,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_EASY = re.compile(r"^(Easy\s+Apply|Apply\s+now|Apply)$", re.I)
_EXTERNAL = re.compile(
    r"^(Apply\s+on\s+company\s+(site|website)|Continue\s+to\s+apply|"
    r"Apply\s+on\s+employer\s+site)$",
    re.I,
)
_NEXT = re.compile(r"^(Next|Continue|Review|Save\s+and\s+continue)$", re.I)
_SUBMIT = re.compile(r"^(Submit(\s+application)?)$", re.I)
_BUDGET_S = 130
_MAX_STEPS = 12


def run_glassdoor(
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
    pdf = Path(pdf_path)
    master = Path(master_path)
    owns = page is None or context is None
    handoff = False
    ctx, pg, pw = context, page, None
    deadline = time.monotonic() + _BUDGET_S
    try:
        if owns:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            ctx, pg = launch_persistent_context(pw, user_data_dir=user_data_dir)
        assert pg is not None and ctx is not None

        goto_dom(pg, job_url)
        raise_if_cloudflare(pg, "glassdoor")

        # External company apply — capture popup URL, close tab, run ATS on primary
        ext = (
            pg.get_by_role("button", name=_EXTERNAL)
            .or_(pg.get_by_role("link", name=_EXTERNAL))
            .or_(
                pg.locator(
                    "a[href*='greenhouse'], a[href*='lever.co'], a[href*='ashbyhq']"
                )
            )
            .first
        )
        try:
            if ext.count() and ext.is_visible(timeout=1200):

                def _click_ext() -> None:
                    ext.click(timeout=5000)

                final_url = capture_popup_url_then_close(
                    ctx, pg, _click_ext, popup_timeout_ms=5000
                )
                kind = classify_ats(final_url or pg.url or "")
                if kind in {"greenhouse", "lever", "ashby", "workday", "indeed"}:
                    if final_url and classify_ats(pg.url or "") != kind:
                        goto_dom(pg, final_url)
                    close_extra_pages(ctx, keep=pg)
                    from src.playbooks.aggregator import _dispatch_ats

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
        except Exception:
            logger.debug("glassdoor external path miss", exc_info=True)

        # Easy Apply / Apply Now A/B
        easy = (
            pg.get_by_role("button", name=_EASY)
            .or_(pg.get_by_role("link", name=_EASY))
            .or_(
                pg.locator(
                    "button[data-test='easyApplyClick'], "
                    "button[data-test*='apply'], "
                    "button:has-text('Easy Apply'), "
                    "button:has-text('Apply Now')"
                )
            )
            .first
        )
        try:
            easy.wait_for(state="visible", timeout=4000)
            easy.click(timeout=5000)
        except Exception as exc:
            reason = f"glassdoor: no Easy Apply / Apply Now ({type(exc).__name__})"
            unblocked = try_cdp_assist(cdp_assist, reason, page=pg)
            if unblocked and confirm_application_success(pg):
                return True
            if not unblocked:
                raise PlaybookFallback(reason) from exc

        wait_for_any(
            pg,
            [
                "input[type='file']",
                "textarea",
                "button:has-text('Next')",
                "button:has-text('Continue')",
                "button:has-text('Submit')",
            ],
            timeout_ms=6000,
        )
        raise_if_cloudflare(pg, "glassdoor")

        # Modal loop: force upload early + before submit
        force_upload_resume(pg, pdf)
        pitch = resolve_pitch_short(
            yaml_snapshot=yaml_snapshot,
            master_path=master,
            company=company or "the team",
            job_title=job_title or "",
            jd_text=jd_text or "",
        )
        try:
            box = pg.locator("textarea").first
            if box.count() and box.is_visible(timeout=500):
                human_type(box, pitch, delay_ms=10)
        except Exception:
            pass

        for step in range(_MAX_STEPS):
            if time.monotonic() > deadline:
                raise PlaybookFallback("glassdoor: budget exhausted")
            raise_if_cloudflare(pg, "glassdoor")
            try:
                body = pg.inner_text("body", timeout=1200) or ""
            except Exception:
                body = ""
            if success_text(body):
                return True

            # Re-assert resume each cycle (GD sometimes resets file input)
            if step in (0, 1, 3) or step == _MAX_STEPS - 2:
                force_upload_resume(pg, pdf)

            advanced = False
            for pattern in (_SUBMIT, _NEXT):
                try:
                    btn = pg.get_by_role("button", name=pattern).first
                    if btn.count() and btn.is_visible(timeout=600):
                        if pattern is _SUBMIT:
                            force_upload_resume(pg, pdf)
                        btn.click(timeout=4500)
                        wait_for_any(
                            pg,
                            [
                                "text=/thank you/i",
                                "text=/application submitted/i",
                                "button:has-text('Next')",
                                "button:has-text('Submit')",
                                "button:has-text('Review')",
                            ],
                            timeout_ms=4000,
                        )
                        advanced = True
                        break
                except Exception:
                    continue
            if not advanced:
                reason = f"glassdoor: stuck step {step}"
                if try_cdp_assist(cdp_assist, reason, page=pg):
                    force_upload_resume(pg, pdf)
                    continue
                raise PlaybookFallback(reason)

        raise PlaybookFallback("glassdoor: too many steps")
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
