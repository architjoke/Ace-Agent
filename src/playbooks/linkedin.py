"""Deterministic LinkedIn Easy Apply state machine (sync Playwright)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from src.playbooks import PlaybookFallback
from src.playbooks._common import (
    close_context_safe,
    force_upload_resume,
    goto_dom,
    launch_persistent_context,
    load_applicant_profile,
    success_text,
)
from src.playbooks.session import raise_handoff

logger = logging.getLogger(__name__)

_WIDGET_TIMEOUT_MS = 5_000
# Deep Easy Apply modals (7+ screens) need headroom beyond the old 12-step cap.
_MAX_STEPS = 15
_OVERALL_BUDGET_S = 120

_EASY_APPLY = re.compile(r"Easy Apply", re.I)
_APPLIED = re.compile(r"^Applied$", re.I)
_EXTERNAL_APPLY = re.compile(
    r"Apply\s+on\s+company\s+(website|site)|Apply\s+on\s+employer\s+site|"
    r"Continue\s+to\s+apply|Company\s+website",
    re.I,
)
_SUBMIT = re.compile(r"^Submit application$", re.I)
_REVIEW = re.compile(r"^Review$", re.I)
_NEXT = re.compile(r"^(Next|Continue|Save and continue)$", re.I)
_ERROR_BANNER_TEXT = re.compile(
    r"(please\s+fix|something\s+went\s+wrong|an\s+error\s+occurred|"
    r"fix\s+the\s+(errors?|highlighted)|required\s+field|"
    r"there\s+was\s+a\s+problem|try\s+again)",
    re.I,
)
_ERROR_BANNER_SELS = (
    ".artdeco-inline-feedback--error",
    "[data-test-form-element-error-message]",
    "[data-test-form-element-error]",
    ".jobs-easy-apply-content .artdeco-inline-feedback--error",
    "[role='alert']",
)

# Easy Apply click budget (was ~3s — too tight on warm LI).
_EA_VISIBLE_MS = 1800
_EA_CLICK_MS = 1500

_EA_CSS = (
    "button.jobs-apply-button--top-card, "
    "button[data-control-name*=\"easy_apply\"], "
    "button.jobs-apply-button"
)
_EA_XPATH = (
    '//button[contains(translate(normalize-space(.), '
    '"EASY APPLY", "easy apply"), "easy apply")]'
)


def _already_applied_visible(page) -> bool:
    """Skip when Applied / Application submitted is already showing (DigiPlus case)."""
    try:
        from src.core.already_applied import page_shows_already_applied

        if page_shows_already_applied(page):
            return True
    except Exception:
        pass
    try:
        if page.get_by_role("button", name=_APPLIED).first.is_visible(timeout=800):
            return True
    except Exception:
        pass
    markers = (
        re.compile(r"you applied on", re.I),
        re.compile(r"^application\s+submitted$", re.I),
        re.compile(r"your\s+application\s+(was|has\s+been)\s+(submitted|sent)", re.I),
        re.compile(r"application\s+sent", re.I),
    )
    for pat in markers:
        try:
            if page.get_by_text(pat).first.is_visible(timeout=500):
                return True
        except Exception:
            continue
    return False


def _ea_button_ok(loc) -> bool:
    """True if locator looks like a clickable Easy Apply (not Applied/disabled)."""
    try:
        if loc.count() == 0:
            return False
        if not loc.is_visible(timeout=_EA_VISIBLE_MS):
            return False
        try:
            if loc.is_disabled():
                return False
        except Exception:
            pass
        try:
            label = (loc.inner_text(timeout=400) or "").strip().lower()
        except Exception:
            label = ""
        if label.startswith("applied") or "already applied" in label:
            return False
        return True
    except Exception:
        return False


def _try_click_locator(loc) -> bool:
    if not _ea_button_ok(loc):
        return False
    try:
        try:
            loc.scroll_into_view_if_needed(timeout=1000)
        except Exception:
            pass
        loc.click(timeout=_EA_CLICK_MS)
        return True
    except Exception:
        return False


def _ea_search_roots(page):
    """Page + child frames (Easy Apply often lives in a nested frame).

    Playwright locators pierce open shadow roots by default; closed shadow is
    rare on LinkedIn job cards — frames are the common miss.
    """
    yield ("main", page)
    try:
        main = getattr(page, "main_frame", None)
        frames = list(getattr(page, "frames", []) or [])
    except Exception:
        return
    for i, fr in enumerate(frames):
        if main is not None and fr is main:
            continue
        yield (f"frame{i}", fr)


def _external_apply_locator(page):
    """Apply-on-company-website control (not Easy Apply modal)."""
    return (
        page.get_by_role("button", name=_EXTERNAL_APPLY)
        .or_(page.get_by_role("link", name=_EXTERNAL_APPLY))
        .or_(
            page.locator(
                "a[data-control-name*='apply'], button[data-control-name*='apply'], "
                "a.jobs-apply-button, button.jobs-apply-button"
            ).filter(has_text=_EXTERNAL_APPLY)
        )
        .first
    )


def _page_shows_external_apply(page) -> bool:
    """True only when an offsite Apply control is *actually* visible.

    Require ``is_visible is True`` so MagicMock / odd locator chains never
    look like a positive (would wrongly raise ExternalPortalLoadError).
    """
    try:
        loc = _external_apply_locator(page)
        try:
            count = int(loc.count())
        except (TypeError, ValueError):
            return False
        if count == 0:
            return False
        return loc.is_visible(timeout=900) is True
    except Exception:
        return False


def _scrape_offsite_apply_url(page) -> str:
    """Pull company apply URL from LinkedIn embedded JSON / apply href (no click)."""
    try:
        raw = page.evaluate(
            """() => {
              const keys = [
                'companyApplyUrl', 'companyApplyURI', 'applyUrl',
                'offsiteApplyUrl', 'externalApplyUrl', 'externalUrl'
              ];
              const html = document.documentElement
                ? document.documentElement.innerHTML
                : '';
              for (const k of keys) {
                const re = new RegExp('"' + k + '"\\\\s*:\\\\s*"(https?:[^"]+)"', 'i');
                const m = html.match(re);
                if (m && m[1] && !/linkedin\\.com/i.test(m[1])) {
                  return m[1]
                    .replace(/\\\\u0026/g, '&')
                    .replace(/\\\\u003d/g, '=');
                }
              }
              const sels = [
                'a.jobs-apply-button[href^="http"]',
                'a[data-control-name*="apply"][href^="http"]',
                'a[href*="greenhouse"]', 'a[href*="lever.co"]',
                'a[href*="ashbyhq"]', 'a[href*="myworkdayjobs"]',
                'a[href*="zohorecruit"]', 'a[href*="eightfold"]',
                'a[href*="careers."][href*="jobs"]'
              ];
              for (const s of sels) {
                const a = document.querySelector(s);
                if (a && a.href && !/linkedin\\.com/i.test(a.href)) return a.href;
              }
              return '';
            }"""
        )
        url = (raw or "").strip()
        if url and "linkedin.com" not in url.lower():
            return url
    except Exception:
        logger.debug("offsite apply URL scrape failed", exc_info=True)
    return ""


def _try_external_company_bridge(
    page,
    context,
    *,
    pdf: Path,
    master: Path,
    user_data_dir: Path | str | None,
    yaml_snapshot: str | None,
    cdp_assist,
) -> bool:
    """Bridge 'Apply on company website' → Greenhouse/Lever/Ashby/Workday on primary tab.

    Returns True on ATS success. Raises ExternalPortalLoadError on dead portals.
    Raises PlaybookFallback when no external CTA / unknown host (caller may Easy Apply).
    """
    from src.playbooks import classify_ats
    from src.playbooks._common import (
        capture_popup_url_then_close,
        close_extra_pages,
    )
    from src.playbooks.aggregator import _dispatch_ats

    loc = _external_apply_locator(page)
    try:
        visible = loc.count() > 0 and loc.is_visible(timeout=1200)
    except Exception:
        visible = False
    if not visible:
        raise PlaybookFallback("linkedin playbook: no external company-apply CTA")

    # Prefer embedded / href company URL before clicking (avoids popup races).
    href = ""
    try:
        href = (loc.get_attribute("href") or "").strip()
    except Exception:
        href = ""
    scraped = _scrape_offsite_apply_url(page)
    final_url = ""
    for candidate in (href, scraped):
        c = (candidate or "").strip()
        if c.startswith("http") and "linkedin.com" not in c.lower():
            final_url = c
            break

    if not final_url:

        def _click_ext() -> None:
            loc.click(timeout=5000)

        final_url = capture_popup_url_then_close(
            context, page, _click_ext, popup_timeout_ms=7000, nav_timeout_ms=14_000
        )

    final_url = (final_url or "").strip()
    if not final_url or "linkedin.com" in final_url.lower():
        # Same-tab may have navigated; re-scrape after click.
        try:
            final_url = (page.url or "").strip()
        except Exception:
            final_url = final_url or ""
        if not final_url or "linkedin.com" in final_url.lower():
            final_url = _scrape_offsite_apply_url(page) or final_url
    if not final_url or "linkedin.com" in final_url.lower():
        from src.core.executor import ExternalPortalLoadError
        from src.core.human_assist import shot_on_fail

        shot_on_fail(
            page,
            company="",
            reason="linkedin_no_company_url",
            job_url=(page.url if page else "") or "",
        )
        raise ExternalPortalLoadError(
            "FAILED_EXTERNAL_LOAD: LinkedIn external apply did not yield a company URL"
        )

    from src.core.discovery import is_blocked_board
    from src.core.executor import ExternalPortalLoadError

    # Internshala / scam boards — hard skip (never CDP, never Easy Apply hunt).
    if is_blocked_board(final_url):
        raise ExternalPortalLoadError(
            f"SKIP_BLOCKED_BOARD: LinkedIn bridge landed on blocked board "
            f"({final_url[:160]})"
        )

    kind = classify_ats(final_url)
    logger.info(
        "LinkedIn external apply bridge → %s (%s)",
        kind,
        final_url[:160],
    )
    _BRIDGEABLE = {
        "greenhouse",
        "lever",
        "ashby",
        "workday",
        "indeed",
        "google_forms",
        "career_site",
    }
    if kind in _BRIDGEABLE:
        if classify_ats(page.url or "") != kind:
            goto_dom(page, final_url)
        close_extra_pages(context, keep=page)
        try:
            return bool(
                _dispatch_ats(
                    kind,
                    page,
                    context,
                    page.url or final_url,
                    pdf,
                    master,
                    user_data_dir=user_data_dir,
                    yaml_snapshot=yaml_snapshot,
                    cdp_assist=cdp_assist,
                )
            )
        except PlaybookFallback as exc:
            raise ExternalPortalLoadError(
                f"FAILED_EXTERNAL_LOAD: bridged {kind} playbook failed ({exc})"
            ) from exc

    # Unknown career host — still navigate primary tab; do not Easy-Apply-hunt.
    try:
        goto_dom(page, final_url)
    except Exception as exc:
        from src.core.executor import is_portal_load_failure

        if isinstance(exc, ExternalPortalLoadError) or is_portal_load_failure(exc):
            raise
        raise ExternalPortalLoadError(
            f"FAILED_EXTERNAL_LOAD: company portal navigation failed ({exc})"
        ) from exc
    close_extra_pages(context, keep=page)
    if is_blocked_board(page.url or final_url):
        raise ExternalPortalLoadError(
            f"SKIP_BLOCKED_BOARD: LinkedIn bridge landed on blocked board "
            f"({(page.url or final_url)[:160]})"
        )
    kind2 = classify_ats(page.url or final_url)
    if kind2 in _BRIDGEABLE:
        try:
            return bool(
                _dispatch_ats(
                    kind2,
                    page,
                    context,
                    page.url or final_url,
                    pdf,
                    master,
                    user_data_dir=user_data_dir,
                    yaml_snapshot=yaml_snapshot,
                    cdp_assist=cdp_assist,
                )
            )
        except PlaybookFallback as exc:
            raise ExternalPortalLoadError(
                f"FAILED_EXTERNAL_LOAD: bridged {kind2} playbook failed ({exc})"
            ) from exc

    # Last resort: try generic career-site form fill on the company URL.
    logger.info(
        "LinkedIn bridge → career_site fallback (%s)",
        (page.url or final_url)[:160],
    )
    try:
        return bool(
            _dispatch_ats(
                "career_site",
                page,
                context,
                page.url or final_url,
                pdf,
                master,
                user_data_dir=user_data_dir,
                yaml_snapshot=yaml_snapshot,
                cdp_assist=cdp_assist,
            )
        )
    except PlaybookFallback as exc:
        raise ExternalPortalLoadError(
            f"FAILED_EXTERNAL_LOAD: career_site fallback failed ({exc}) "
            f"url={(page.url or final_url)[:160]}"
        ) from exc


def _click_easy_apply(page) -> None:
    """Multi-strategy Easy Apply click; combined budget ~3s then PlaybookFallback.

    If Easy Apply is missing because the job already shows Application submitted /
    Applied, raise AlreadyAppliedError instead of burning CDP Gemini.
    """

    def _strategies(root):
        return (
            ("aria", lambda: root.get_by_role("button", name=_EASY_APPLY).first),
            ("css", lambda: root.locator(_EA_CSS).first),
            ("xpath", lambda: root.locator(f"xpath={_EA_XPATH}").first),
        )

    errors: list[str] = []
    for root_name, root in _ea_search_roots(page):
        for name, factory in _strategies(root):
            key = f"{root_name}/{name}"
            try:
                loc = factory()
                if _try_click_locator(loc):
                    logger.debug("Easy Apply clicked via %s", key)
                    try:
                        page.wait_for_timeout(400)
                    except Exception:
                        pass
                    return
                errors.append(f"{key}:not_ready")
            except Exception as exc:
                errors.append(f"{key}:{type(exc).__name__}")
    # DigiPlus-class: submit chrome present, Easy Apply gone — skip CDP burn.
    if _already_applied_visible(page):
        from src.core.executor import AlreadyAppliedError

        raise AlreadyAppliedError(
            "linkedin playbook: Application submitted / Applied visible "
            "(Easy Apply not clickable)"
        )
    if _page_shows_external_apply(page):
        from src.core.executor import ExternalPortalLoadError

        raise ExternalPortalLoadError(
            "FAILED_EXTERNAL_LOAD: Apply on company website visible but Easy Apply missing "
            "(caller should have bridged)"
        )
    raise PlaybookFallback(
        "linkedin playbook: Easy Apply not clickable ("
        + "; ".join(errors[:12])
        + ")"
    )


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
                if not force_upload_resume(page, pdf):
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


def _form_error_banner(page) -> bool:
    """True when Easy Apply shows a validation/error banner after Next."""
    for sel in _ERROR_BANNER_SELS:
        try:
            loc = page.locator(sel).first
            if loc.count() == 0:
                continue
            if not loc.is_visible(timeout=500):
                continue
            try:
                text = (loc.inner_text(timeout=300) or "").strip()
            except Exception:
                text = ""
            sel_l = sel.lower()
            looks_error_sel = "error" in sel_l or "feedback--error" in sel_l
            if looks_error_sel:
                return True
            # Generic alerts need matching copy — avoid false positives.
            if text and (
                _ERROR_BANNER_TEXT.search(text)
                or "error" in text.lower()
                or "required" in text.lower()
            ):
                return True
        except Exception:
            continue
    try:
        loc = page.get_by_text(_ERROR_BANNER_TEXT).first
        if loc.count() and loc.is_visible(timeout=400):
            return True
    except Exception:
        pass
    return False


def _advance(page) -> str:
    """Click Submit first (short-circuit), then Review, then Next.

    Returns submit|review|next|none.
    """
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


def _easy_apply_modal_open(page) -> bool:
    """True when LinkedIn Easy Apply modal/content is visible (CDP may still help)."""
    sels = (
        ".jobs-easy-apply-modal",
        ".jobs-easy-apply-content",
        "[data-test-modal-id='easy-apply-modal']",
        "div[role='dialog'] .jobs-easy-apply-content",
    )
    for sel in sels:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=400):
                return True
        except Exception:
            continue
    return False


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

        # External "Apply on company website" → ATS bridge BEFORE Easy Apply hunt.
        if _page_shows_external_apply(page):
            logger.info(
                "LinkedIn external company apply detected — bridging off Easy Apply path"
            )
            return _try_external_company_bridge(
                page,
                context,
                pdf=pdf,
                master=master,
                user_data_dir=user_data_dir,
                yaml_snapshot=yaml_snapshot,
                cdp_assist=cdp_assist,
            )

        try:
            _click_easy_apply(page)
        except Exception as ea_exc:
            from src.core.executor import ExternalPortalLoadError

            # Race: external CTA appeared / Easy Apply miss on company-site jobs.
            if _page_shows_external_apply(page) or isinstance(
                ea_exc, ExternalPortalLoadError
            ):
                return _try_external_company_bridge(
                    page,
                    context,
                    pdf=pdf,
                    master=master,
                    user_data_dir=user_data_dir,
                    yaml_snapshot=yaml_snapshot,
                    cdp_assist=cdp_assist,
                )
            if isinstance(ea_exc, PlaybookFallback):
                raise
            raise

        # Always force a fresh disk PDF before answering questions / Next.
        if pdf.is_file():
            if not force_upload_resume(page, pdf):
                logger.debug(
                    "LinkedIn force_upload_resume deferred — file input may appear later"
                )

        miss_advance = 0
        uploaded_once = False
        for step in range(_MAX_STEPS):
            if time.monotonic() - started > _OVERALL_BUDGET_S:
                raise PlaybookFallback(
                    f"linkedin playbook: overall budget exceeded on {job_url}"
                )

            if pdf.is_file() and not uploaded_once:
                if force_upload_resume(page, pdf):
                    uploaded_once = True

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

            # Prefer Submit → Review → Next so deep modals short-circuit when ready.
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
            if action == "next" and _form_error_banner(page):
                raise PlaybookFallback(
                    "linkedin playbook: error banner after Next — Gemini"
                )
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
        from src.core.executor import ExternalPortalLoadError

        msg = str(exc)
        msg_l = msg.lower()
        # Dead ends — never burn Gemini CDP; still snapshot for human review.
        from src.core.human_assist import shot_on_fail

        if "easy apply not clickable" in msg_l:
            shot_on_fail(page, reason="skip_no_easy_apply", job_url=job_url)
            raise ExternalPortalLoadError(f"SKIP_NO_EASY_APPLY: {msg}") from exc
        if "overall budget exceeded" in msg_l and not _easy_apply_modal_open(page):
            shot_on_fail(page, reason="skip_budget_no_modal", job_url=job_url)
            raise ExternalPortalLoadError(f"SKIP_BUDGET_NO_MODAL: {msg}") from exc
        shot_on_fail(page, reason="linkedin_fallback", job_url=job_url)
        handoff = context is not None and page is not None
        raise_handoff(
            msg,
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
