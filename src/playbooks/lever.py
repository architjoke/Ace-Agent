"""Pure Playwright Lever (+ Ashby host branch) apply playbook."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    fill_css,
    launch_persistent_context,
    load_applicant_profile,
    success_text,
    upload_resume_file,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 30_000


def _is_ashby(job_url: str) -> bool:
    host = (urlparse(job_url or "").hostname or "").lower()
    return "ashbyhq.com" in host


def _click_apply(page) -> None:
    for pattern in (
        r"^Apply$",
        r"Apply for this job",
        r"Submit application",
        r"Start application",
        r"Apply now",
    ):
        for role in ("button", "link"):
            loc = page.get_by_role(role, name=re.compile(pattern, re.I))
            try:
                if loc.count() and loc.first.is_visible():
                    loc.first.click(timeout=5000)
                    page.wait_for_timeout(700)
                    return
            except Exception:
                continue


def _fill_lever(page, fields: dict) -> bool:
    filled = False
    filled |= fill_css(
        page,
        ['input[name="name"]', 'input[name="fullName"]', 'input[autocomplete="name"]'],
        fields["full_name"],
    )
    filled |= fill_css(
        page,
        ['input[name="email"]', 'input[type="email"]', 'input[autocomplete="email"]'],
        fields["email"],
    )
    filled |= fill_css(
        page,
        ['input[name="phone"]', 'input[type="tel"]', 'input[autocomplete="tel"]'],
        fields["phone"],
    )
    if fields.get("linkedin"):
        fill_css(
            page,
            [
                'input[name="urls[LinkedIn]"]',
                'input[name*="LinkedIn" i]',
                'input[placeholder*="LinkedIn" i]',
            ],
            fields["linkedin"],
        )
    if fields.get("github"):
        fill_css(
            page,
            ['input[name="urls[GitHub]"]', 'input[name*="GitHub" i]'],
            fields["github"],
        )
    return filled


def _fill_ashby(page, fields: dict) -> bool:
    """Ashby often uses labeled inputs / test ids — CSS + label fallbacks."""
    filled = _fill_lever(page, fields)
    # Label-oriented extras when name= attributes differ
    label_map = [
        (re.compile(r"name|full name", re.I), fields["full_name"]),
        (re.compile(r"email", re.I), fields["email"]),
        (re.compile(r"phone|mobile", re.I), fields["phone"]),
        (re.compile(r"linkedin", re.I), fields.get("linkedin") or ""),
    ]
    for pat, value in label_map:
        if not value:
            continue
        try:
            loc = page.get_by_label(pat)
            if loc.count() and loc.first.is_visible(timeout=1200):
                loc.first.fill(value, timeout=5000)
                filled = True
        except Exception:
            continue
    # data-testid common on Ashby
    filled |= fill_css(
        page,
        [
            '[data-testid*="name" i] input',
            '[data-testid*="email" i] input',
            'input[data-testid*="email" i]',
        ],
        fields["email"] if fields["email"] else fields["full_name"],
    )
    return filled


def _unanswered_custom(page) -> bool:
    try:
        required = page.locator(
            "textarea[required], select[required], "
            "input[required]:not([type='hidden']):not([type='file']):"
            "not([type='checkbox']):not([type='radio'])"
        )
        count = required.count()
    except Exception:
        return False
    for i in range(min(count, 12)):
        el = required.nth(i)
        try:
            name = ((el.get_attribute("name") or "") + (el.get_attribute("id") or "")).lower()
            if re.search(r"name|email|phone|linkedin|github|resume|url", name):
                continue
            val = el.input_value(timeout=400)
            if not str(val or "").strip():
                return True
        except Exception:
            continue
    return False


def _submit(page) -> bool:
    for pattern in (r"Submit application", r"^Submit$", r"Send application", r"Apply"):
        btn = page.get_by_role("button", name=re.compile(pattern, re.I))
        try:
            if btn.count() and btn.first.is_visible():
                btn.first.click(timeout=8000)
                return True
        except Exception:
            continue
    for sel in ('button[type="submit"]', 'input[type="submit"]'):
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                loc.click(timeout=8000)
                return True
        except Exception:
            continue
    return False


def run_lever(
    job_url: str,
    pdf_path: Path | str,
    master_yaml_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
    headless: bool | None = None,
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ats: str | None = None,
) -> bool:
    """Fill Lever (or Ashby) apply form. Raises TimeoutError / PlaybookFallback."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    pdf = Path(pdf_path).expanduser().resolve()
    master = Path(master_yaml_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not master.is_file():
        raise FileNotFoundError(f"master.yaml not found: {master}")

    ashby = (ats == "ashby") or _is_ashby(job_url)
    fields = load_applicant_profile(master)
    context = None
    try:
        with sync_playwright() as p:
            context, page = launch_persistent_context(
                p,
                user_data_dir=user_data_dir,
                headless=headless,
                timeout_ms=timeout_ms,
            )
            kind = "ashby" if ashby else "lever"
            logger.info("%s playbook navigating %s", kind.capitalize(), job_url)
            from src.playbooks._common import goto_dom

            goto_dom(page, job_url)
            _click_apply(page)

            filled = _fill_ashby(page, fields) if ashby else _fill_lever(page, fields)
            uploaded = upload_resume_file(page, pdf)
            if not uploaded:
                raise PlaybookFallback(f"{kind} playbook: resume file input not found")
            if not filled and not fields.get("email"):
                raise PlaybookFallback(f"{kind} playbook: no profile fields filled")

            if _unanswered_custom(page):
                raise PlaybookFallback(
                    f"{kind} playbook: unanswered custom/required fields — Gemini"
                )

            if not _submit(page):
                raise PlaybookFallback(f"{kind} playbook: submit control not found")

            page.wait_for_timeout(2000)
            try:
                body = page.inner_text("body") or ""
            except Exception:
                body = ""
            if success_text(body):
                logger.info("%s playbook success for %s", kind, job_url)
                return True
            if "required" in body.lower() and "error" in body.lower():
                raise PlaybookFallback(f"{kind} playbook: validation errors after submit")
            logger.info("%s playbook submitted without clear thank-you — counting success", kind)
            return True
    except PlaywrightTimeout as exc:
        raise TimeoutError(f"Lever/Ashby playbook timed out on {job_url}") from exc
    finally:
        from src.playbooks._common import close_context_safe

        close_context_safe(context, user_data_dir=user_data_dir)
