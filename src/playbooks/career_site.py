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
    fill_ats_form,
    force_upload_resume,
    goto_dom,
    launch_persistent_context,
    load_applicant_profile,
    settle,
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
        settle(page, timeout_ms=1200)
        # Oracle HCM / TriNet: "This job is no longer available." → skip, no Manual_Action.
        try:
            from src.core.job_closed import page_shows_job_closed
            from src.core.executor import JobClosedError

            if page_shows_job_closed(page):
                raise JobClosedError(
                    f"Career-site job closed/unavailable at {target[:160]}"
                )
        except JobClosedError:
            raise
        except Exception:
            logger.debug("career_site closed-job probe failed", exc_info=True)

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
                        settle(page, timeout_ms=1200)
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
                settle(page, timeout_ms=2000)
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
                        "this job is no longer available",
                        "job is no longer available",
                        "no longer accepting",
                        "no longer available",
                    )
                ):
                    from src.core.executor import JobClosedError

                    raise JobClosedError(
                        f"Career-site job closed/unavailable at {target[:160]}"
                    )
                if any(
                    n in body_l
                    for n in (
                        "sign in to apply",
                        "log in to apply",
                        "login to apply",
                        "create an account",
                    )
                ):
                    _fail("career_site: Apply gated (login) — needs manual")
                _fail("career_site: Apply CTA not found")

        stats = fill_ats_form(page, profile=answers, max_pages=2)
        filled = int(stats.get("filled") or 0)

        if pdf.is_file():
            try:
                force_upload_resume(page, pdf)
            except Exception:
                logger.debug("career_site resume upload miss", exc_info=True)

        from src.core.human_assist import (
            ensure_captcha_cleared,
            learn_from_page,
            page_shows_captcha,
            snapshot_filled_fields,
        )

        if page_shows_captcha(page):
            before = snapshot_filled_fields(page)
            ensure_captcha_cleared(
                page, company=company, job_url=job_url, site="career_site"
            )
            try:
                learn_from_page(
                    page, company=company, job_url=job_url, before=before
                )
            except Exception:
                logger.debug("career_site learn after captcha failed", exc_info=True)

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

        settle(page, timeout_ms=2000)
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
            before = snapshot_filled_fields(page)
            ensure_captcha_cleared(
                page, company=company, job_url=job_url, site="career_site"
            )
            try:
                learn_from_page(
                    page, company=company, job_url=job_url, before=before
                )
            except Exception:
                logger.debug("career_site learn after post-submit captcha failed", exc_info=True)
            # Operator cleared captcha — try submit once more.
            try:
                page.locator("button, input[type='submit']").filter(
                    has_text=re.compile(r"submit|apply|send", re.I)
                ).first.click(timeout=4000)
                settle(page, timeout_ms=1500)
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
        _fail(
            f"career_site: no confirmation after submit "
            f"(filled={filled}, submitted={submitted})"
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
                logger.debug("career_site warm reset failed", exc_info=True)
