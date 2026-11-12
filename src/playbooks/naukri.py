"""Deterministic Naukri apply — preflight nag shield, one-click / form, clean CDP yield."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback, classify_ats
from src.playbooks._common import (
    close_context_safe,
    force_upload_resume,
    goto_dom,
    human_type,
    launch_persistent_context,
    load_applicant_profile,
    raise_if_cloudflare,
    success_text,
    confirm_application_success,
    try_cdp_assist,
    wait_for_any,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_APPLY = re.compile(r"^(Apply(\s+now)?|Easy\s+Apply|I\s+am\s+interested)$", re.I)
_SUBMIT = re.compile(r"^(Submit|Send\s+application|Apply)$", re.I)
_NEXT = re.compile(r"^(Next|Continue|Save\s+and\s+continue)$", re.I)
_MAX_STEPS = 12
_BUDGET_S = 120


def preflight_dismiss_naukri(page) -> None:
    """Aggressively close chat widgets / profile nags before Apply (locator.or_ chain)."""
    dismiss = (
        page.get_by_role(
            "button",
            name=re.compile(
                r"^(Not\s+now|Skip|Later|Close|No\s+thanks|Cancel|Maybe\s+later|"
                r"Update\s+later|Remind\s+me\s+later|Dismiss|Minimize)$",
                re.I,
            ),
        )
        .or_(page.locator("div.chatbot_close, .crossIcon, #chatbot-close, .layer-close"))
        .or_(page.locator("button[aria-label='Close'], button[aria-label='close']"))
        .or_(page.locator(".chatbot_Icon, #chat-widget, .naukri-chat-widget"))
    )
    for _ in range(5):
        try:
            loc = dismiss.first
            if not loc.count() or not loc.is_visible(timeout=350):
                break
            # Prefer clicking a close control inside/near the widget
            try:
                closer = loc.locator(
                    "button, [role='button'], .crossIcon, .chatbot_close"
                ).first
                if closer.count() and closer.is_visible(timeout=200):
                    closer.click(timeout=1200)
                else:
                    loc.click(timeout=1200)
            except Exception:
                loc.click(timeout=1200)
            # Event-driven: wait until that overlay is hidden/detached
            try:
                loc.wait_for(state="hidden", timeout=800)
            except Exception:
                pass
        except Exception:
            break


def _already_applied(page) -> bool:
    try:
        body = (page.inner_text("body", timeout=1500) or "").lower()
    except Exception:
        body = ""
    return any(
        m in body for m in ("already applied", "you have applied", "application sent")
    )


def _exotic_human_blocker(page) -> str | None:
    try:
        body = (page.inner_text("body", timeout=1200) or "").lower()
    except Exception:
        return None
    if "audio intro" in body or "record a video" in body or "video introduction" in body:
        return "naukri: exotic media intro required"
    if "login" in (page.url or "").lower() and "naukri" in (page.url or "").lower():
        return "naukri: login wall"
    return None


def run_naukri(
    job_url: str,
    pdf_path: Path | str,
    master_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
    yaml_snapshot: str | None = None,
    context=None,
    page=None,
    cdp_assist=None,
    **_kwargs,
) -> bool:
    pdf = Path(pdf_path)
    master = Path(master_path)
    owns = page is None or context is None
    handoff = False
    ctx, pg, pw = context, page, None
    profile = load_applicant_profile(master, yaml_snapshot=yaml_snapshot)
    deadline = time.monotonic() + _BUDGET_S
    try:
        if owns:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            ctx, pg = launch_persistent_context(pw, user_data_dir=user_data_dir)
        assert pg is not None

        goto_dom(pg, job_url)
        raise_if_cloudflare(pg, "naukri")
        preflight_dismiss_naukri(pg)
        if _already_applied(pg):
            from src.core.executor import AlreadyAppliedError

            raise AlreadyAppliedError("Naukri already applied")

        apply_btn = (
            pg.get_by_role("button", name=_APPLY)
            .or_(pg.get_by_role("link", name=_APPLY))
            .or_(pg.locator("#apply-button, .apply-button, button.waves-effect"))
            .first
        )
        try:
            apply_btn.wait_for(state="visible", timeout=4000)
            apply_btn.click(timeout=5000)
        except Exception as exc:
            reason = f"naukri: Apply not clickable ({type(exc).__name__})"
            unblocked = try_cdp_assist(cdp_assist, reason, page=pg)
            if unblocked and confirm_application_success(pg):
                return True
            if not unblocked:
                raise PlaybookFallback(reason) from exc

        # Wait for post-apply UI (success toast, form, or redirect)
        wait_for_any(
            pg,
            [
                "text=/already applied/i",
                "text=/application sent/i",
                "textarea",
                "input[type='file']",
                "button:has-text('Submit')",
            ],
            timeout_ms=5000,
        )
        preflight_dismiss_naukri(pg)
        raise_if_cloudflare(pg, "naukri")

        human = _exotic_human_blocker(pg)
        if human:
            from src.core.executor import RequiresHumanInterventionError

            raise RequiresHumanInterventionError(human)

        kind = classify_ats(pg.url or "")
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

        force_upload_resume(pg, pdf)
        for step in range(_MAX_STEPS):
            if time.monotonic() > deadline:
                raise PlaybookFallback("naukri: budget exhausted")
            preflight_dismiss_naukri(pg)
            try:
                body = pg.inner_text("body", timeout=1200) or ""
            except Exception:
                body = ""
            if success_text(body) or _already_applied(pg):
                return True
            human = _exotic_human_blocker(pg)
            if human:
                from src.core.executor import RequiresHumanInterventionError

                raise RequiresHumanInterventionError(human)

            for label, value in (
                ("name", profile.get("full_name") or profile.get("name")),
                ("email", profile.get("email")),
                ("mobile", profile.get("phone_national") or profile.get("phone")),
            ):
                if not value:
                    continue
                try:
                    inp = pg.get_by_label(re.compile(label, re.I)).first
                    if inp.count() and inp.is_visible(timeout=350):
                        human_type(inp, str(value), delay_ms=10)
                except Exception:
                    pass

            advanced = False
            for pattern in (_SUBMIT, _NEXT, _APPLY):
                try:
                    btn = pg.get_by_role("button", name=pattern).first
                    if btn.count() and btn.is_visible(timeout=500):
                        btn.click(timeout=4000)
                        wait_for_any(
                            pg,
                            [
                                "text=/application sent/i",
                                "text=/thank you/i",
                                "button:has-text('Submit')",
                                "button:has-text('Next')",
                            ],
                            timeout_ms=3000,
                        )
                        advanced = True
                        break
                except Exception:
                    continue
            if not advanced:
                # Unknown custom question / stuck DOM → CDP assist once, else handoff
                reason = f"naukri: unknown form / stuck at step {step}"
                if try_cdp_assist(cdp_assist, reason, page=pg):
                    force_upload_resume(pg, pdf)
                    continue
                raise PlaybookFallback(reason)

        raise PlaybookFallback("naukri: too many steps without success")
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
