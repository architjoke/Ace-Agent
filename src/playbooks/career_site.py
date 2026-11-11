"""Generic company career / ATS form (Eightfold, Zoho Recruit, custom careers.*)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    close_context_safe,
    force_upload_resume,
    goto_dom,
    launch_persistent_context,
    load_applicant_profile,
    success_text,
)

logger = logging.getLogger(__name__)

# LinkedIn embed / UTM noise often hides Apply on Zoho Limited Listings.
_STRIP_QUERY = frozenset(
    {
        "source",
        "embedsource",
        "codes",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "utm_term",
        "ref",
        "referrer",
        "li_fat_id",
    }
)


def _clean_career_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return raw
    try:
        parsed = urlparse(raw)
        kept = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _STRIP_QUERY
        ]
        return urlunparse(parsed._replace(query=urlencode(kept, doseq=True)))
    except Exception:
        return raw

_BUDGET_S = 100
# Zoho/Eightfold CTAs vary ("Apply Now", "I'm Interested", "Click to Apply").
_APPLY = re.compile(
    r"(Apply(\s+(now|here|online))?|Apply\s+for\s+this\s+(job|position)|"
    r"Start\s+application|I['’]?m\s+interested|Click\s*(here\s*)?to\s*apply|"
    r"Submit(\s+application)?)",
    re.I,
)
_SUBMIT = re.compile(
    r"(Submit(\s+application)?|Send(\s+application)?|Finish|Complete\s+application|"
    r"Apply(\s+now)?)",
    re.I,
)
_APPLY_CSS = (
    "#cw-apply, #applybutton, button#apply, a#apply, "
    "[class*='apply' i][class*='btn' i], [data-testid*='apply' i], "
    "a[href*='apply' i], button[onclick*='apply' i], "
    ".zf-applyBtn, .cw-btn, .eightfold-apply, [aria-label*='Apply' i], "
    "button.lyte-button.lyteSuccess, "
    "button.lyte-button:has-text('interested'), "
    ".cw-detail-header-button button"
)
_COOKIE_CSS = (
    "button:has-text('Accept all'), button:has-text('Accept All'), "
    "button:has-text('Accept'), button:has-text('I agree'), "
    "button:has-text('Got it'), [id*='accept' i][class*='cookie' i]"
)


def run_career_site(
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
    """Fill a company career form: open Apply → identity → resume → submit."""
    _ = kwargs
    _ = cdp_assist
    pdf = Path(pdf_path).expanduser().resolve()
    master = Path(master_yaml_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not master.is_file():
        raise FileNotFoundError(f"master.yaml not found: {master}")

    answers = load_applicant_profile(master, yaml_snapshot=yaml_snapshot)
    company = str(kwargs.get("company") or "").strip()
    owned = context is None or page is None
    pw = None
    started = time.monotonic()

    def _fail(msg: str) -> None:
        from src.core.human_assist import shot_on_fail

        shot_on_fail(page, company=company, reason=msg[:80], job_url=job_url)
        raise PlaybookFallback(msg)

    try:
        if owned:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            context, page = launch_persistent_context(
                pw, user_data_dir=user_data_dir, headless=headless
            )
        assert page is not None and context is not None

        target = _clean_career_url(job_url)
        if target != job_url:
            logger.info(
                "Career-site stripped embed/UTM params → %s", target[:160]
            )
        logger.info("Career-site playbook navigating %s", target[:160])
        try:
            goto_dom(page, target, timeout_ms=25_000)
        except Exception as nav_exc:
            _fail(
                f"career_site: navigation failed ({type(nav_exc).__name__}: {nav_exc})"
            )
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        page.wait_for_timeout(1100)

        def _dismiss_cookies(root) -> None:
            try:
                loc = root.locator(_COOKIE_CSS).first
                if loc.count() and loc.is_visible(timeout=900):
                    loc.click(timeout=2500)
                    page.wait_for_timeout(400)
            except Exception:
                pass

        def _form_visible(root) -> bool:
            formish = root.locator(
                "input[type='email'], input[name*='email' i], input[type='file'], "
                "textarea, input[type='tel'], input[name*='FirstName' i], "
                "input[name*='name' i], input[name*='rec-form_' i], "
                "text=Submit Application"
            )
            try:
                return formish.count() > 0 and formish.first.is_visible(timeout=1500)
            except Exception:
                return False

        def _click_apply(root) -> bool:
            _dismiss_cookies(root)
            factories = (
                lambda: root.get_by_role("button", name=_APPLY).first,
                lambda: root.get_by_role("link", name=_APPLY).first,
                lambda: root.get_by_text(
                    re.compile(r"^\s*(Apply(\s+Now)?|I['’]?m\s+interested)\s*$", re.I)
                ).first,
                lambda: root.locator(_APPLY_CSS).first,
                lambda: root.locator(
                    "a:has-text('Apply'), button:has-text('Apply'), "
                    "button:has-text('interested'), "
                    "a:has-text(\"I'm interested\"), a:has-text('I am interested'), "
                    "button:has-text('Start'), input[type='button'][value*='Apply' i], "
                    "input[type='submit'][value*='Apply' i], "
                    "span:has-text('Apply Now'), div[role='button']:has-text('Apply')"
                ).first,
            )
            for factory in factories:
                try:
                    loc = factory()
                    if loc.count() and loc.is_visible(timeout=1200):
                        loc.scroll_into_view_if_needed(timeout=1500)
                        loc.click(timeout=4500, force=True)
                        page.wait_for_timeout(1100)
                        return True
                except Exception:
                    continue
            return False

        def _roots():
            out = [page]
            try:
                out.extend(page.frames)
            except Exception:
                pass
            return out

        # Open application form if still on JD page (page + same-origin iframes).
        has_form = any(_form_visible(r) for r in _roots())
        if not has_form:
            clicked = False
            for root in _roots():
                if _click_apply(root):
                    clicked = True
                    break
            if not clicked:
                # One more settle pass — Zoho/Eightfold hydrate CTAs late.
                page.wait_for_timeout(1800)
                for root in _roots():
                    if _form_visible(root) or _click_apply(root):
                        clicked = True
                        break
            if not clicked and not any(_form_visible(r) for r in _roots()):
                # Login walls / geo blocks still count as unsupported CTA.
                try:
                    body_l = (page.inner_text("body", timeout=2000) or "").lower()
                except Exception:
                    body_l = ""
                if any(
                    n in body_l
                    for n in (
                        "sign in to apply",
                        "log in to apply",
                        "login to apply",
                        "create an account",
                        "job is no longer available",
                        "no longer accepting",
                    )
                ):
                    _fail("career_site: Apply gated (login/closed) — needs manual")
                _fail("career_site: Apply CTA not found")

        def _hint(el) -> str:
            parts: list[str] = []
            for attr in ("aria-label", "placeholder", "name", "id", "autocomplete"):
                try:
                    parts.append(str(el.get_attribute(attr) or ""))
                except Exception:
                    pass
            try:
                parts.append(
                    str(
                        el.evaluate(
                            "e => (e.closest('label,div,li,tr')||e.parentElement||e)"
                            ".innerText||''"
                        )
                        or ""
                    )[:180]
                )
            except Exception:
                pass
            return " ".join(parts).lower()

        filled = 0
        for root in _roots():
            try:
                fields = root.locator(
                    "input[type='text'], input[type='email'], input[type='tel'], "
                    "input:not([type]), textarea"
                )
                n = min(fields.count(), 30)
            except Exception:
                continue
            for i in range(n):
                if time.monotonic() - started > _BUDGET_S:
                    break
                el = fields.nth(i)
                try:
                    if not el.is_visible(timeout=350):
                        continue
                except Exception:
                    continue
                hint = _hint(el)
                value = ""
                if any(k in hint for k in ("email", "e-mail")):
                    value = answers.get("email") or ""
                elif any(k in hint for k in ("phone", "mobile", "tel", "whatsapp")):
                    value = answers.get("phone") or ""
                elif any(k in hint for k in ("first name", "given name", "firstname")):
                    value = answers.get("first_name") or ""
                elif any(k in hint for k in ("last name", "surname", "family", "lastname")):
                    value = answers.get("last_name") or ""
                elif any(k in hint for k in ("full name", "your name", "candidate name")) or (
                    "name" in hint and "user" not in hint and "file" not in hint
                ):
                    value = answers.get("full_name") or ""
                elif any(
                    k in hint for k in ("city", "location", "address", "current location")
                ):
                    value = answers.get("location") or ""
                elif any(
                    k in hint
                    for k in ("linkedin", "portfolio", "github", "website", "url")
                ):
                    value = (
                        answers.get("linkedin_url")
                        or answers.get("portfolio_url")
                        or answers.get("github_url")
                        or ""
                    )
                if not value:
                    continue
                try:
                    el.click(timeout=600)
                    el.fill(str(value), timeout=2000)
                    filled += 1
                except Exception:
                    continue

        if pdf.is_file():
            try:
                force_upload_resume(page, pdf)
            except Exception:
                logger.debug("career_site resume upload miss", exc_info=True)

        from src.core.human_assist import (
            page_shows_captcha,
            resolve_captcha_with_human,
        )

        if page_shows_captcha(page):
            # Pause for human solve instead of hard-failing the apply.
            if not resolve_captcha_with_human(
                page, company=company, job_url=job_url
            ):
                _fail("career_site: captcha gate — human did not clear in time")

        submitted = False
        for root in _roots():
            for pattern in (_SUBMIT, _APPLY):
                try:
                    btn = root.get_by_role("button", name=pattern).first
                    if btn.count() and btn.is_visible(timeout=1000):
                        btn.click(timeout=4000)
                        submitted = True
                        break
                except Exception:
                    continue
            if submitted:
                break
            try:
                root.locator("button, input[type='submit']").filter(
                    has_text=re.compile(r"submit|apply|send", re.I)
                ).first.click(timeout=4000)
                submitted = True
                break
            except Exception:
                continue
        if not submitted:
            _fail("career_site: Submit/Apply not clickable")

        page.wait_for_timeout(1600)
        try:
            body = page.inner_text("body", timeout=3000) or ""
        except Exception:
            body = ""
        body_l = body.lower()
        if page_shows_captcha(page) and not any(
            m in body_l
            for m in (
                "application submitted",
                "application received",
                "thank you for applying",
            )
        ):
            if not resolve_captcha_with_human(
                page, company=company, job_url=job_url
            ):
                _fail("career_site: captcha still present after submit")
            # Operator cleared captcha — try submit once more.
            try:
                page.locator("button, input[type='submit']").filter(
                    has_text=re.compile(r"submit|apply|send", re.I)
                ).first.click(timeout=4000)
                page.wait_for_timeout(1200)
                body = page.inner_text("body", timeout=3000) or ""
                body_l = body.lower()
            except Exception:
                pass
        if success_text(body) or any(
            m in body_l
            for m in (
                "application submitted",
                "application received",
                "thank you for applying",
                "thanks for applying",
                "successfully submitted",
                "we have received your application",
            )
        ):
            logger.info("Career-site submit success for %s", job_url[:160])
            return True
        if filled >= 2 and submitted and not page_shows_captcha(page):
            logger.info(
                "Career-site submitted (%s fields) — counting success %s",
                filled,
                job_url[:160],
            )
            return True
        _fail(f"career_site: no confirmation after submit (filled={filled})")
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
                logger.debug("career_site warm reset failed", exc_info=True)
