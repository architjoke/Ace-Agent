"""Pure Playwright Workday apply playbook — ARIA selectors only (no #id / .class)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    launch_persistent_context,
    load_applicant_profile,
    success_text,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 30_000
_OVERALL_BUDGET_S = 120

# Labels we know how to fill — anything else required → Gemini
_KNOWN_LABEL_HINTS = re.compile(
    r"resume|cv|upload|legal\s*name|first\s*name|last\s*name|given\s*name|"
    r"family\s*name|email|phone|mobile|address|city|postal|zip|state|province|"
    r"country|linkedin|how\s+did\s+you\s+hear",
    re.I,
)


def _click_role(page, role: str, name_pat: re.Pattern, timeout: int = 8000) -> bool:
    loc = page.get_by_role(role, name=name_pat)
    try:
        if loc.count() and loc.first.is_visible(timeout=min(2000, timeout)):
            loc.first.click(timeout=timeout)
            return True
    except Exception:
        pass
    return False


def _fill_label(page, label_pat: re.Pattern, value: str) -> bool:
    if not value:
        return False
    try:
        loc = page.get_by_label(label_pat)
        if loc.count() == 0:
            return False
        target = loc.first
        if not target.is_visible(timeout=2000):
            return False
        target.fill(value, timeout=8000)
        return True
    except Exception:
        return False


def _upload_resume_aria(page, pdf_path: Path) -> bool:
    """Upload via Resume/CV label region; file input is the only type=file exception."""
    label_pat = re.compile(r"Resume|CV|Upload\s*(Resume|CV)?", re.I)
    try:
        labeled = page.get_by_label(label_pat)
        if labeled.count():
            # Try set_input_files on the labeled control itself
            try:
                labeled.first.set_input_files(str(pdf_path), timeout=10_000)
                return True
            except Exception:
                pass
            # Nested file input under the labeled widget's ancestor
            try:
                root = labeled.first.locator("xpath=ancestor-or-self::*[.//input[@type='file']][1]")
                file_inp = root.locator('input[type="file"]').first
                if file_inp.count():
                    file_inp.set_input_files(str(pdf_path), timeout=10_000)
                    return True
            except Exception:
                pass
    except Exception:
        pass
    # Last resort: any file input (no class/id) — allowed exception for uploads
    try:
        inputs = page.locator('input[type="file"]')
        n = inputs.count()
        for i in range(n):
            try:
                inputs.nth(i).set_input_files(str(pdf_path), timeout=10_000)
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _start_apply(page) -> None:
    for pat in (
        r"Apply",
        r"Apply Now",
        r"Autofill with Resume",
        r"Start",
        r"Begin",
    ):
        if _click_role(page, "button", re.compile(pat, re.I)):
            page.wait_for_timeout(1000)
            return
        if _click_role(page, "link", re.compile(pat, re.I)):
            page.wait_for_timeout(1000)
            return
        try:
            t = page.get_by_text(re.compile(pat, re.I))
            if t.count() and t.first.is_visible(timeout=1000):
                t.first.click(timeout=5000)
                page.wait_for_timeout(1000)
                return
        except Exception:
            continue


def _wait_parser(page) -> None:
    try:
        page.get_by_text(re.compile(r"success|uploaded|parsed|complete", re.I)).first.wait_for(
            state="visible", timeout=8000
        )
    except Exception:
        page.wait_for_timeout(2500)


def _fill_known_fields(page, fields: dict) -> None:
    pairs = [
        (re.compile(r"Legal Name|Full Name", re.I), fields.get("full_name") or ""),
        (re.compile(r"First Name|Given Name", re.I), fields.get("first_name") or ""),
        (re.compile(r"Last Name|Family Name", re.I), fields.get("last_name") or ""),
        (re.compile(r"^Email|Email Address", re.I), fields.get("email") or ""),
        (re.compile(r"Phone|Mobile|Telephone", re.I), fields.get("phone") or ""),
        (re.compile(r"Address Line 1|Address(?!\s*Line\s*2)", re.I), fields.get("address_line") or fields.get("location") or ""),
        (re.compile(r"^City", re.I), fields.get("city") or ""),
        (re.compile(r"Postal|Zip|PIN", re.I), fields.get("pincode") or ""),
        (re.compile(r"State|Province|Region", re.I), fields.get("state") or ""),
        (re.compile(r"^Country", re.I), fields.get("country") or ""),
        (re.compile(r"LinkedIn", re.I), fields.get("linkedin") or ""),
    ]
    for pat, value in pairs:
        if value:
            _fill_label(page, pat, value)


def _click_next_or_submit(page) -> str | None:
    """Return 'submit' | 'next' | None."""
    if _click_role(page, "button", re.compile(r"^Submit$|Submit Application|Send", re.I)):
        return "submit"
    if _click_role(page, "button", re.compile(r"^Next$|Continue|Save and Continue", re.I)):
        return "next"
    return None


def _custom_unanswered(page) -> bool:
    """Required textboxes whose accessible name is not in our known set."""
    try:
        boxes = page.get_by_role("textbox")
        n = boxes.count()
    except Exception:
        return False
    for i in range(min(n, 20)):
        box = boxes.nth(i)
        try:
            if not box.is_visible(timeout=500):
                continue
            name = (box.get_attribute("aria-label") or "") + " " + (
                box.evaluate(
                    """e => {
                      const id = e.getAttribute('aria-labelledby');
                      if (!id) return '';
                      const el = document.getElementById(id);
                      return el ? el.textContent : '';
                    }"""
                )
                or ""
            )
            # Also try associated label via accessible name
            try:
                accessible = box.evaluate("e => e.labels ? [...e.labels].map(l => l.textContent).join(' ') : ''") or ""
                name = f"{name} {accessible}"
            except Exception:
                pass
            name_l = name.lower()
            if _KNOWN_LABEL_HINTS.search(name_l):
                continue
            # required?
            required = False
            try:
                required = bool(
                    box.evaluate(
                        "e => e.required || e.getAttribute('aria-required') === 'true'"
                    )
                )
            except Exception:
                required = False
            if not required:
                continue
            val = ""
            try:
                val = box.input_value(timeout=400)
            except Exception:
                pass
            if not str(val or "").strip():
                return True
        except Exception:
            continue
    return False


def run_workday(
    job_url: str,
    pdf_path: Path | str,
    master_yaml_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
    headless: bool | None = None,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    yaml_snapshot: str | None = None,
    context=None,
    page=None,
    cdp_assist=None,
    **kwargs,
) -> bool:
    """Workday multi-step apply via ARIA. Raises TimeoutError / PlaybookFallback."""
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

    fields = load_applicant_profile(master, yaml_snapshot=yaml_snapshot)
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
                timeout_ms=timeout_ms,
            )
        logger.info("Workday playbook navigating %s", job_url)
        from src.playbooks._common import goto_dom

        goto_dom(page, job_url)
        from src.core.job_closed import page_shows_job_closed

        if page_shows_job_closed(page):
            from src.core.executor import JobClosedError

            raise JobClosedError(f"Workday job closed/missing at {job_url[:160]}")
        _start_apply(page)

        if not _upload_resume_aria(page, pdf):
            reason = "Workday playbook: Resume/CV upload control not found"
            if try_cdp_assist(cdp_assist, reason) and _upload_resume_aria(page, pdf):
                pass
            else:
                raise PlaybookFallback(reason)
        _wait_parser(page)

        for step in range(12):
            if time.monotonic() - started > _OVERALL_BUDGET_S:
                raise PlaybookFallback(
                    f"Workday playbook overall budget exceeded on {job_url}"
                )

            _fill_known_fields(page, fields)

            if _custom_unanswered(page):
                reason = "Workday playbook: unanswered custom/required fields — Gemini"
                if try_cdp_assist(cdp_assist, reason) and not _custom_unanswered(page):
                    pass
                else:
                    raise PlaybookFallback(reason)

            action = _click_next_or_submit(page)
            page.wait_for_timeout(1200)

            try:
                body = page.inner_text("body") or ""
            except Exception:
                body = ""
            if success_text(body) or page.get_by_text(
                re.compile(r"thank you|application submitted|successfully submitted", re.I)
            ).count():
                logger.info("Workday playbook success for %s (step=%s)", job_url, step)
                return True

            if action == "submit":
                page.wait_for_timeout(2000)
                try:
                    body = page.inner_text("body") or ""
                except Exception:
                    body = ""
                if success_text(body):
                    return True
                logger.info("Workday playbook submit clicked — counting success")
                return True

            if action is None:
                reason = "Workday playbook: Next/Submit control not found — Gemini"
                if try_cdp_assist(cdp_assist, reason):
                    continue
                raise PlaybookFallback(reason)

        raise PlaybookFallback("Workday playbook: too many steps without success — Gemini")
    except PlaywrightTimeout as exc:
        from src.playbooks.session import raise_handoff

        handoff = context is not None and page is not None
        raise_handoff(
            f"Workday playbook timed out on {job_url}",
            context=context,
            page=page,
            user_data_dir=user_data_dir,
            playwright=pw,
            cause=exc,
            fallback=TimeoutError(f"Workday playbook timed out on {job_url}"),
        )
    except PlaybookFallback as exc:
        from src.playbooks.session import raise_handoff

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
                from src.playbooks._common import close_context_safe

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
                    logger.debug("warm workday reset failed", exc_info=True)
