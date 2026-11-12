"""YC Work at a Startup — pitch injection, profile gates as human exits, CDP yield."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback
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
    confirm_application_success,
    try_cdp_assist,
    wait_for_any,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_APPLY = re.compile(r"^(Apply|Reach\s+out|Send\s+application)$", re.I)
_SEND = re.compile(r"^(Send|Submit|Send\s+application)$", re.I)
_BUDGET_S = 100


def _profile_gate(page) -> str | None:
    try:
        body = (page.inner_text("body", timeout=1500) or "").lower()
    except Exception:
        body = ""
    url = (page.url or "").lower()
    if "sign in" in body or "/login" in url:
        return "yc_waas: login required"
    if "finish profile" in body or "complete your profile" in body:
        return "yc_waas: profile incomplete — finish WaAS profile wizard"
    if "hide my profile" in body and "change status" in body:
        return "yc_waas: profile hidden / not looking — set Actively looking"
    if "change status to apply" in body:
        return "yc_waas: status not looking — set Actively looking or Open"
    if "weekly" in body and "limit" in body:
        return "yc_waas: weekly application limit"
    return None


def run_yc_waas(
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
        assert pg is not None

        goto_dom(pg, job_url)
        raise_if_cloudflare(pg, "yc_waas")
        gate = _profile_gate(pg)
        if gate:
            from src.core.executor import RequiresHumanInterventionError

            raise RequiresHumanInterventionError(gate)

        apply_btn = (
            pg.get_by_role("button", name=_APPLY)
            .or_(pg.get_by_role("link", name=_APPLY))
            .first
        )
        try:
            apply_btn.wait_for(state="visible", timeout=4000)
            apply_btn.click(timeout=5000)
        except Exception as exc:
            reason = f"yc_waas: Apply not clickable ({type(exc).__name__})"
            unblocked = try_cdp_assist(cdp_assist, reason, page=pg)
            if unblocked and confirm_application_success(pg):
                return True
            if not unblocked:
                raise PlaybookFallback(reason) from exc

        wait_for_any(
            pg,
            [
                "textarea[name='message']",
                "textarea[placeholder*='message' i]",
                "textarea",
                "button:has-text('Send')",
            ],
            timeout_ms=6000,
        )
        gate = _profile_gate(pg)
        if gate:
            from src.core.executor import RequiresHumanInterventionError

            raise RequiresHumanInterventionError(gate)

        pitch = resolve_pitch_short(
            yaml_snapshot=yaml_snapshot,
            master_path=master,
            company=company or "the startup",
            job_title=job_title or "",
            jd_text=jd_text or "",
        )
        if len(pitch) < 50:
            pitch = f"{pitch} I'm excited to contribute and learn with the team."

        box = (
            pg.locator("textarea[name='message']")
            .or_(pg.locator("textarea[placeholder*='message' i]"))
            .or_(pg.locator("textarea[placeholder*='note' i]"))
            .or_(pg.locator("textarea"))
            .first
        )
        try:
            if box.count() and box.is_visible(timeout=800):
                human_type(box, pitch, delay_ms=10)
        except Exception:
            logger.debug("yc pitch type missed", exc_info=True)

        force_upload_resume(pg, pdf)
        close_extra_pages(ctx, keep=pg)

        for step in range(6):
            if time.monotonic() > deadline:
                raise PlaybookFallback("yc_waas: budget exhausted")
            try:
                body = pg.inner_text("body", timeout=1200) or ""
            except Exception:
                body = ""
            low = body.lower()
            if success_text(body) or "application sent" in low or "you applied" in low:
                return True
            gate = _profile_gate(pg)
            if gate:
                from src.core.executor import RequiresHumanInterventionError

                raise RequiresHumanInterventionError(gate)
            try:
                btn = pg.get_by_role("button", name=_SEND).first
                if btn.count() and btn.is_visible(timeout=600):
                    btn.click(timeout=5000)
                    wait_for_any(
                        pg,
                        [
                            "text=/application sent/i",
                            "text=/you applied/i",
                            "text=/thank you/i",
                        ],
                        timeout_ms=4000,
                    )
                    continue
            except Exception:
                pass
            reason = f"yc_waas: custom question or send blocked (step {step})"
            unblocked = try_cdp_assist(cdp_assist, reason, page=pg)
            if unblocked and confirm_application_success(pg):
                return True
            if unblocked:
                continue
            raise PlaybookFallback(reason)

        raise PlaybookFallback("yc_waas: too many steps")
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
