"""Wellfound native apply — pitch injection, CF hard-exit, clean CDP yield."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback, classify_ats
from src.playbooks._common import (
    close_context_safe,
    close_extra_pages,
    force_upload_resume,
    goto_dom,
    human_type,
    launch_persistent_context,
    raise_if_cloudflare,
    resolve_pitch_short,
    success_text,
    try_cdp_assist,
    wait_for_any,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_APPLY = re.compile(r"^(Apply|Apply\s+now|Send\s+application)$", re.I)
_SEND = re.compile(r"^(Send(\s+application)?|Submit(\s+application)?)$", re.I)
_BUDGET_S = 100


def _profile_human_blocker(page) -> str | None:
    try:
        body = (page.inner_text("body", timeout=1500) or "").lower()
    except Exception:
        body = ""
    if "complete your profile" in body or "missing_skills" in body:
        return "wellfound: profile incomplete"
    if "sign in" in body and "password" in body:
        return "wellfound: login required"
    return None


def run_wellfound(
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
        raise_if_cloudflare(pg, "wellfound")
        human = _profile_human_blocker(pg)
        if human:
            from src.core.executor import RequiresHumanInterventionError

            raise RequiresHumanInterventionError(human)

        # Offsite ATS listing — stay on primary tab
        try:
            offsite = pg.locator(
                "a[href*='greenhouse'], a[href*='lever.co'], a[href*='ashbyhq']"
            ).first
            if offsite.count() and offsite.is_visible(timeout=600):
                href = offsite.get_attribute("href") or ""
                if href:
                    goto_dom(pg, href)
                    raise_if_cloudflare(pg, "wellfound")
                    kind = classify_ats(pg.url)
                    if kind in {"greenhouse", "lever", "ashby", "workday"}:
                        from src.playbooks.aggregator import _dispatch_ats

                        return _dispatch_ats(
                            kind,
                            pg,
                            ctx,
                            pg.url,
                            pdf,
                            master,
                            user_data_dir=user_data_dir,
                            yaml_snapshot=yaml_snapshot,
                            cdp_assist=cdp_assist,
                        )
        except Exception:
            pass

        apply_btn = (
            pg.get_by_role("button", name=_APPLY)
            .or_(pg.locator("[data-test*='Apply'], button:has-text('Apply')"))
            .first
        )
        try:
            apply_btn.wait_for(state="visible", timeout=4000)
            apply_btn.click(timeout=5000)
        except Exception as exc:
            reason = f"wellfound: Apply not clickable ({type(exc).__name__})"
            if try_cdp_assist(cdp_assist, reason):
                return True
            raise PlaybookFallback(reason) from exc

        wait_for_any(
            pg,
            [
                "textarea[name='userNote']",
                "textarea[placeholder*='note' i]",
                "textarea",
                "[data-test='JobApplicationModal--SubmitButton']",
                "button:has-text('Send')",
            ],
            timeout_ms=6000,
        )
        raise_if_cloudflare(pg, "wellfound")
        human = _profile_human_blocker(pg)
        if human:
            from src.core.executor import RequiresHumanInterventionError

            raise RequiresHumanInterventionError(human)

        pitch = resolve_pitch_short(
            yaml_snapshot=yaml_snapshot,
            master_path=master,
            company=company or "the team",
            job_title=job_title or "",
            jd_text=jd_text or "",
        )
        note = (
            pg.locator("textarea[name='userNote']")
            .or_(pg.locator("textarea[placeholder*='note' i]"))
            .or_(pg.locator("textarea[placeholder*='Cover' i]"))
            .or_(pg.locator("textarea"))
            .first
        )
        try:
            if note.count() and note.is_visible(timeout=800):
                human_type(note, pitch, delay_ms=10)
        except Exception:
            logger.debug("wellfound pitch type missed", exc_info=True)

        force_upload_resume(pg, pdf)
        close_extra_pages(ctx, keep=pg)

        for step in range(6):
            if time.monotonic() > deadline:
                raise PlaybookFallback("wellfound: budget exhausted")
            raise_if_cloudflare(pg, "wellfound")
            try:
                body = pg.inner_text("body", timeout=1200) or ""
            except Exception:
                body = ""
            if success_text(body) or "application sent" in body.lower():
                return True
            try:
                btn = pg.get_by_role("button", name=_SEND).first
                if btn.count() and btn.is_visible(timeout=600):
                    btn.click(timeout=5000)
                    wait_for_any(
                        pg,
                        [
                            "text=/application sent/i",
                            "text=/thank you/i",
                            "text=/you applied/i",
                        ],
                        timeout_ms=4000,
                    )
                    continue
            except Exception:
                pass
            # Unknown custom question → CDP assist once, then handoff (keep warm session)
            reason = f"wellfound: custom question / send blocked (step {step})"
            if try_cdp_assist(cdp_assist, reason):
                return True
            raise PlaybookFallback(reason)

        raise PlaybookFallback("wellfound: too many steps")
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
