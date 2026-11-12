"""Shared playbook helpers (profile load + persistent Chromium)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse

import yaml

logger = logging.getLogger(__name__)

# Anti-detection flags for Google OAuth / ATS persistent Chromium.
CHROMIUM_STEALTH_ARGS: list[str] = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    # Ephemeral CDP port for Gemini hot-swap (DevToolsActivePort in profile dir).
    "--remote-debugging-port=0",
]

_TRACKING_Q = frozenset(
    {"ref", "source", "gh_src", "trk", "trkinfo", "lipi", "originalSubdomain"}
)
_TRACKING_Q_RE = re.compile(r"^utm_", re.I)


def pick_chrome_user_agent() -> str:
    """Back-compat alias — always returns the pinned (non-random) UA."""
    from src.core.browser_identity import pinned_user_agent

    return pinned_user_agent()


# DOM-ready navigation — ignore tracking-pixel networkidle hangs.
GOTO_DOM_TIMEOUT_MS = 15_000


def try_cdp_assist(
    cdp_assist,
    reason: str,
    *,
    page=None,
    company: str = "",
    job_url: str = "",
) -> bool:
    """Run optional CDP assist. True means *unblocked/ran*, NOT application success.

    When ``page`` is provided, snapshots fields before assist and learns afterward
    (promote fixed keys into local master). Callers must still confirm submit via
    ``confirm_application_success`` / ``assist_and_confirm``.
    """
    if cdp_assist is None:
        return False
    before: list[tuple[str, str]] | None = None
    if page is not None:
        try:
            from src.core.human_assist import snapshot_filled_fields

            before = snapshot_filled_fields(page)
        except Exception:
            before = None
    try:
        ok = bool(cdp_assist(str(reason or "stuck")))
    except Exception:
        logger.debug("cdp_assist raised", exc_info=True)
        return False
    if ok and page is not None:
        try:
            from src.core.human_assist import learn_from_page

            learn_from_page(
                page, company=company or "", job_url=job_url or "", before=before
            )
        except Exception:
            logger.debug("learn_from_page after cdp_assist failed", exc_info=True)
    return ok


def assist_and_confirm(
    page,
    cdp_assist,
    reason: str,
    *,
    company: str = "",
    job_url: str = "",
) -> bool:
    """Assist then verify submission markers. Never treat assist alone as applied."""
    if not try_cdp_assist(
        cdp_assist, reason, page=page, company=company, job_url=job_url
    ):
        return False
    return confirm_application_success(page)


def _norm_host(host: str) -> str:
    h = (host or "").lower().removeprefix("www.")
    return h


def _host_peers(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.endswith(".greenhouse.io") and b.endswith(".greenhouse.io"):
        return True
    if a.endswith(".myworkdayjobs.com") and b.endswith(".myworkdayjobs.com"):
        return a == b
    # lever ↔ ashby only with matching path job token (checked in urls_equivalent)
    return False


def _filter_q(query: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in parse_qsl(query or "", keep_blank_values=True):
        lk = k.lower()
        if lk in _TRACKING_Q or _TRACKING_Q_RE.match(lk):
            continue
        out[lk] = v
    return out


def urls_equivalent(current: str, target: str) -> bool:
    """True when current page is already the target job (redirects / utm / trailing slash)."""
    try:
        c, t = urlparse((current or "").strip()), urlparse((target or "").strip())
    except Exception:
        return False
    if not t.netloc or not c.netloc:
        return False
    ch, th = _norm_host(c.hostname or ""), _norm_host(t.hostname or "")
    lever_ashby = {ch, th} <= {"jobs.lever.co", "jobs.ashbyhq.com"} and ch != th
    if not (_host_peers(ch, th) or lever_ashby):
        return False
    cp = unquote((c.path or "/").rstrip("/") or "/")
    tp = unquote((t.path or "/").rstrip("/") or "/")
    if lever_ashby:
        # require last path segment (job id) equal
        if (cp.rsplit("/", 1)[-1] or "") != (tp.rsplit("/", 1)[-1] or ""):
            return False
    elif not (cp == tp or cp.startswith(tp + "/")):
        return False
    tq, cq = _filter_q(t.query), _filter_q(c.query)
    return all(cq.get(k) == v for k, v in tq.items())


def _is_playwright_thread_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "different thread" in msg or "cannot switch to a different" in msg


def _is_portal_navigation_error(exc: BaseException) -> bool:
    """DNS / connection failures on company portals (not greenlet / soft timeout)."""
    msg = str(exc).lower()
    markers = (
        "err_name_not_resolved",
        "err_connection_refused",
        "err_connection_timed_out",
        "err_address_unreachable",
        "err_internet_disconnected",
        "err_connection_reset",
        "err_timed_out",
        "net::err_",
        "name_not_resolved",
        "dns_probe_finished",
    )
    return any(m in msg for m in markers)


def goto_dom(
    page,
    url: str,
    *,
    timeout_ms: int = GOTO_DOM_TIMEOUT_MS,
    force: bool = False,
) -> None:
    """Navigate with wait_until=domcontentloaded; skip if already on equivalent URL."""
    if not force:
        try:
            cur = page.url or ""
        except Exception:
            cur = ""
        if cur and urls_equivalent(cur, url):
            logger.debug("goto_dom skip — already on %s", (cur or "")[:160])
            return
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception as first_exc:
        # Greenlet/thread mismatch: reload would also fail — surface cleanly.
        if _is_playwright_thread_error(first_exc):
            raise TimeoutError(
                f"DOM navigation failed for {url[:160]}: {first_exc}"
            ) from first_exc
        if _is_portal_navigation_error(first_exc):
            from src.core.executor import ExternalPortalLoadError

            raise ExternalPortalLoadError(
                f"FAILED_PORTAL_DNS: navigation failed for {url[:160]}: {first_exc}"
            ) from first_exc
        logger.warning(
            "goto_dom timeout/fail on %s (%s) — reloading once",
            (url or "")[:160],
            type(first_exc).__name__,
        )
        try:
            page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as reload_exc:
            if _is_portal_navigation_error(reload_exc):
                from src.core.executor import ExternalPortalLoadError

                raise ExternalPortalLoadError(
                    f"FAILED_PORTAL_DNS: reload failed for {url[:160]}: {reload_exc}"
                ) from reload_exc
            raise TimeoutError(
                f"DOM navigation failed for {url[:160]}: {first_exc}; reload={reload_exc}"
            ) from reload_exc


def load_applicant_profile(
    master_yaml_path: Path,
    *,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
) -> dict[str, Any]:
    data = yaml.safe_load(master_yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    answers = (
        data.get("application_answers")
        if isinstance(data.get("application_answers"), dict)
        else {}
    )
    try:
        from src.core.form_answers import sanitize_application_answers
        from src.core.salary import inject_salary_answers

        answers = inject_salary_answers(
            answers if isinstance(answers, dict) else {},
            jd_text=jd_text,
            yaml_snapshot=yaml_snapshot,
        )
        answers = sanitize_application_answers(answers)
    except Exception:
        answers = answers if isinstance(answers, dict) else {}
    try:
        from src.core.learned_answers import merge_into_answers

        answers = merge_into_answers(answers if isinstance(answers, dict) else {})
    except Exception:
        pass
    cv = data.get("cv") if isinstance(data.get("cv"), dict) else {}
    name = str(cv.get("name") or answers.get("full_name") or "").strip()
    parts = name.split(None, 1)
    first = str(answers.get("first_name") or (parts[0] if parts else "")).strip()
    last = str(
        answers.get("last_name") or (parts[1] if len(parts) > 1 else "")
    ).strip()
    full = str(answers.get("full_name") or name or f"{first} {last}".strip()).strip()
    phone = str(
        answers.get("phone_display")
        or answers.get("phone_e164")
        or cv.get("phone")
        or ""
    ).strip()
    email = str(answers.get("email") or cv.get("email") or "").strip()
    profile: dict[str, Any] = {
        "full_name": full,
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": phone,
        "location": str(
            answers.get("full_location")
            or answers.get("address_line")
            or cv.get("location")
            or ""
        ).strip(),
        "address_line": str(answers.get("address_line") or "").strip(),
        "city": str(answers.get("city") or "").strip(),
        "state": str(answers.get("state") or "").strip(),
        "pincode": str(answers.get("pincode") or answers.get("postal_code") or "").strip(),
        "country": str(answers.get("country") or "India").strip(),
        "linkedin": str(answers.get("linkedin_url") or "").strip(),
        "github": str(answers.get("github_url") or "").strip(),
        # Never reuse GitHub for portfolio/website screening questions.
        "portfolio": str(answers.get("portfolio_url") or answers.get("website_url") or "").strip(),
        "portfolio_url": str(answers.get("portfolio_url") or "").strip(),
        "website_url": str(answers.get("website_url") or "").strip(),
        "expected_ctc": str(
            answers.get("expected_ctc_inr")
            or answers.get("expected_ctc_digits")
            or answers.get("expected_ctc")
            or ""
        ).strip(),
        "expected_ctc_inr": str(
            answers.get("expected_ctc_inr") or answers.get("expected_ctc_digits") or ""
        ).strip(),
        "current_ctc": str(
            answers.get("current_ctc_inr")
            or answers.get("current_ctc_digits")
            or answers.get("current_ctc")
            or "0"
        ).strip(),
        "current_ctc_inr": str(
            answers.get("current_ctc_inr") or answers.get("current_ctc_digits") or "0"
        ).strip(),
        "salary_expectation": str(
            answers.get("expected_ctc_inr")
            or answers.get("salary_expectation")
            or ""
        ).strip(),
        "years_of_experience": str(answers.get("years_of_experience") or "0").strip(),
        "additional_months_of_experience": str(
            answers.get("additional_months_of_experience") or "0"
        ).strip(),
    }
    # Pass through remaining application_answers for form engine kinds.
    for key, value in (answers or {}).items():
        k = str(key)
        if k in profile and profile[k]:
            continue
        if value is None or value == "":
            continue
        profile[k] = (
            value if isinstance(value, (int, float, bool)) else str(value).strip()
        )
    try:
        from src.core.form_engine import enrich_profile_skills

        profile = enrich_profile_skills(profile, data)
    except Exception:
        pass
    return profile


def fill_application_form(
    page,
    master_yaml_path: Path | str,
    *,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
) -> dict[str, Any]:
    """Shared playbook entry: load personal profile → FormFillEngine (general rules)."""
    from src.core.form_engine import FormFillEngine

    master = Path(master_yaml_path)
    profile = load_applicant_profile(
        master, yaml_snapshot=yaml_snapshot, jd_text=jd_text
    )
    master_data = None
    try:
        raw = yaml.safe_load(master.read_text(encoding="utf-8")) or {}
        master_data = raw if isinstance(raw, dict) else None
    except Exception:
        master_data = None
    return FormFillEngine(
        profile, jd_text=jd_text, master_data=master_data
    ).fill_visible(page)


_NEXT_PAGE = re.compile(r"^(Next|Continue|Next\s*page)$", re.I)


def fill_ats_form(
    page,
    master_yaml_path: Path | str | None = None,
    *,
    profile: dict[str, Any] | None = None,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
    max_pages: int = 2,
) -> dict[str, Any]:
    """FormFillEngine + optional Next/Continue for multi-page ATS / Google Forms."""
    from src.core.form_engine import FormFillEngine

    if profile is None:
        if master_yaml_path is None:
            raise ValueError("fill_ats_form requires profile or master_yaml_path")
        return_stats = fill_application_form(
            page,
            master_yaml_path,
            yaml_snapshot=yaml_snapshot,
            jd_text=jd_text,
        )
        engine = None
    else:
        engine = FormFillEngine(profile, jd_text=jd_text)
        return_stats = engine.fill_visible(page)

    total_filled = int(return_stats.get("filled") or 0)
    unknown = list(return_stats.get("unknown_labels") or [])

    for _ in range(max(0, int(max_pages) - 1)):
        advanced = False
        try:
            btn = page.get_by_role("button", name=_NEXT_PAGE).first
            if btn.count() and btn.is_visible(timeout=800):
                btn.click(timeout=3000)
                advanced = True
        except Exception:
            advanced = False
        if not advanced:
            break
        settle(page, timeout_ms=1500)
        if engine is None and master_yaml_path is not None:
            stats = fill_application_form(
                page,
                master_yaml_path,
                yaml_snapshot=yaml_snapshot,
                jd_text=jd_text,
            )
        elif engine is not None:
            stats = engine.fill_visible(page)
        else:
            break
        total_filled += int(stats.get("filled") or 0)
        unknown.extend(stats.get("unknown_labels") or [])

    return {"filled": total_filled, "unknown_labels": unknown}


def clear_profile_locks(profile: Path) -> None:
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock = profile / name
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def close_context_safe(context, *, user_data_dir: Path | str | None = None) -> None:
    """Close Playwright context and clear Chromium singleton locks."""
    from src.core.settings import get_settings

    try:
        if context is not None:
            context.close()
    except Exception:
        logger.exception("Failed closing Playwright context")
    profile = Path(user_data_dir) if user_data_dir else Path(get_settings().browser_user_data_dir)
    try:
        clear_profile_locks(profile.expanduser().resolve())
    except Exception:
        pass


def launch_persistent_context(
    playwright,
    *,
    user_data_dir: Path | str | None = None,
    headless: bool | None = None,
    timeout_ms: int = GOTO_DOM_TIMEOUT_MS,
    proxy: dict[str, str] | None = None,
):
    """Launch Chromium persistent context (cookies) + sticky/override proxy."""
    from src.core.network import playwright_proxy_dict
    from src.core.settings import get_settings

    settings = get_settings()
    profile = Path(user_data_dir) if user_data_dir else Path(settings.browser_user_data_dir)
    profile = profile.expanduser().resolve()
    profile.mkdir(parents=True, exist_ok=True)
    clear_profile_locks(profile)

    from src.core.browser_identity import (
        pinned_device_scale_factor,
        pinned_locale,
        pinned_timezone_id,
        pinned_user_agent,
        pinned_viewport,
        webdriver_stealth_script,
    )

    use_headless = settings.browser_headless if headless is None else bool(headless)
    proxy_cfg = proxy if proxy is not None else playwright_proxy_dict()
    vp = pinned_viewport()
    launch_kwargs: dict[str, Any] = {
        "user_data_dir": str(profile),
        "headless": use_headless,
        "args": list(CHROMIUM_STEALTH_ARGS),
        "user_agent": pinned_user_agent(user_data_dir=profile),
        "viewport": vp,
        "device_scale_factor": pinned_device_scale_factor(),
        # Align Chromium locale/TZ with sticky residential geo (ban armor).
        "locale": pinned_locale(),
        "timezone_id": pinned_timezone_id(),
        # Zero-Trust: no downloads; no geo/camera/mic grants.
        "accept_downloads": False,
        "permissions": [],
    }
    if proxy_cfg:
        launch_kwargs["proxy"] = proxy_cfg
    from src.core.telemetry import time_stage

    with time_stage("chromium_startup_ms", profile=str(profile)):
        context = playwright.chromium.launch_persistent_context(**launch_kwargs)
    try:
        context.add_init_script(webdriver_stealth_script())
    except Exception:
        logger.debug("webdriver stealth init_script failed", exc_info=True)
    try:
        # Belt-and-suspenders if the browser still prompts for media/geo.
        context.clear_permissions()
    except Exception:
        logger.debug("clear_permissions unavailable", exc_info=True)
    page = context.pages[0] if context.pages else context.new_page()
    page.set_default_timeout(timeout_ms)
    return context, page


def fill_css(page, selectors: list[str], value: str) -> bool:
    if not value:
        return False
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            if loc.count() == 0:
                continue
            if not loc.is_visible(timeout=1500):
                continue
            loc.fill(value, timeout=5000)
            return True
        except Exception:
            continue
    return False


def upload_resume_file(page, pdf_path: Path) -> bool:
    inputs = page.locator('input[type="file"]')
    try:
        n = inputs.count()
    except Exception:
        return False
    for i in range(n):
        try:
            inputs.nth(i).set_input_files(str(pdf_path), timeout=10_000)
            return True
        except Exception:
            continue
    return False


_UPLOAD_BTN = re.compile(
    r"(upload(\s+(a\s+)?(new\s+|other\s+|different\s+)?(resume|cv|file))?|"
    r"replace(\s+(resume|cv|file))?|"
    r"change(\s+(resume|cv|file))?|choose\s+file|browse|"
    r"attach(\s+(resume|cv|file))?|use\s+a\s+different\s+(resume|cv|file))",
    re.I,
)
_SAVED_RESUME_CARD = re.compile(
    r"(select\s+(this\s+)?resume|use\s+(this\s+)?resume|choose\s+from\s+library|"
    r"previously\s+uploaded|saved\s+resume)",
    re.I,
)
_RESUME_EDIT_BTN = re.compile(r"^(Edit|Replace|Change)$", re.I)
_RESUME_SECTION = re.compile(r"(resume|cv|curriculum)", re.I)


def _click_resume_edit_affordance(page) -> bool:
    """Indeed Soft review: click Edit beside the attached CV to open re-upload."""
    # Aria / testid first (Indeed Soft Apply)
    for sel in (
        "button[aria-label*='Edit' i][aria-label*='resume' i]",
        "button[aria-label*='Edit' i][aria-label*='CV' i]",
        "[data-testid*='resume' i] button:has-text('Edit')",
        "[data-testid*='Resume' i] button:has-text('Edit')",
        "button:near(:text('Resume'), 80):has-text('Edit')",
        "button:near(:text('CV'), 80):has-text('Edit')",
    ):
        try:
            loc = page.locator(sel).first
            if loc.count() == 0 or not loc.is_visible(timeout=700):
                continue
            loc.click(timeout=4000)
            try:
                page.wait_for_timeout(450)
            except Exception:
                pass
            return True
        except Exception:
            continue
    # Role-name Edit near resume copy
    try:
        edits = page.get_by_role("button", name=_RESUME_EDIT_BTN)
        n = min(edits.count(), 8)
    except Exception:
        n = 0
    for i in range(n):
        btn = edits.nth(i)
        try:
            if not btn.is_visible(timeout=500):
                continue
            # Prefer edits whose nearby text mentions resume/CV
            try:
                nearby = (btn.evaluate(
                    """el => (el.closest('section,form,div')||el.parentElement||el)
                        .innerText||''"""
                ) or "")
            except Exception:
                nearby = ""
            if nearby and not _RESUME_SECTION.search(nearby):
                # Still try if only a couple Edit buttons exist
                if n > 3:
                    continue
            btn.click(timeout=4000)
            try:
                page.wait_for_timeout(450)
            except Exception:
                pass
            return True
        except Exception:
            continue
    return False


def _click_upload_affordance(page) -> bool:
    """Click Upload/Replace/Change — never saved-resume library cards.

    On Indeed Soft review, prefer Edit-beside-CV first so 'upload other resume'
    becomes available.
    """
    _click_resume_edit_affordance(page)
    # Prefer explicit buttons/links with upload/replace wording
    for role in ("button", "link"):
        try:
            loc = page.get_by_role(role, name=_UPLOAD_BTN)
            n = min(loc.count(), 6)
        except Exception:
            n = 0
        for i in range(n):
            btn = loc.nth(i)
            try:
                label = (btn.inner_text(timeout=400) or "").strip()
            except Exception:
                label = ""
            if _SAVED_RESUME_CARD.search(label):
                continue
            try:
                if not btn.is_visible(timeout=800):
                    continue
                btn.click(timeout=4000)
                try:
                    page.wait_for_timeout(350)
                except Exception:
                    pass
                return True
            except Exception:
                continue
    # Text fallback
    try:
        loc = page.get_by_text(_UPLOAD_BTN)
        n = min(loc.count(), 4)
    except Exception:
        n = 0
    for i in range(n):
        el = loc.nth(i)
        try:
            label = (el.inner_text(timeout=300) or "").strip()
            if _SAVED_RESUME_CARD.search(label):
                continue
            if el.is_visible(timeout=600):
                el.click(timeout=4000)
                return True
        except Exception:
            continue
    return False


def _resume_stem_visible(page, pdf_path: Path) -> bool:
    stem = pdf_path.stem.lower()
    name = pdf_path.name.lower()
    try:
        body = (page.inner_text("body", timeout=1500) or "").lower()
    except Exception:
        body = ""
    if stem and stem in body:
        return True
    if name and name in body:
        return True
    return False


_RESUME_ATTACHED_UI = re.compile(
    r"(upload\s+complete|resume\s+attached|cv\s+attached|file\s+uploaded|"
    r"successfully\s+uploaded|resume\s+uploaded)",
    re.I,
)


def _dispatch_file_input_react_events(page) -> None:
    """Force React/controlled inputs to register set_input_files via change/input."""
    try:
        page.evaluate(
            """() => {
                const inputs = Array.from(
                    document.querySelectorAll('input[type="file"]')
                );
                for (const el of inputs) {
                    try {
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    } catch (e) {}
                }
                return inputs.length;
            }"""
        )
    except Exception:
        logger.debug("file input React event dispatch failed", exc_info=True)


def _file_input_has_files(page) -> bool:
    try:
        return bool(
            page.evaluate(
                """() => Array.from(document.querySelectorAll('input[type="file"]'))
                    .some(el => el.files && el.files.length > 0)"""
            )
        )
    except Exception:
        return False


def _resume_ui_attached(page, pdf_path: Path) -> bool:
    """True when UI shows attached resume (stem, Indeed card, or generic copy)."""
    if _resume_stem_visible(page, pdf_path):
        return True
    try:
        card = page.locator(
            '[data-testid="attached-resume-card"], '
            '[data-testid*="resume" i][data-testid*="attach" i], '
            '[data-test*="attached-resume" i]'
        ).first
        if card.count() and card.is_visible(timeout=600):
            return True
    except Exception:
        pass
    try:
        loc = page.get_by_text(_RESUME_ATTACHED_UI).first
        if loc.count() and loc.is_visible(timeout=600):
            return True
    except Exception:
        pass
    return False


def _confirm_resume_upload(page, pdf_path: Path, *, wait_ms: int = 1200) -> bool:
    """Event-driven confirm: UI markers, else input.files.length > 0."""
    deadline_markers = (
        '[data-testid="attached-resume-card"]',
        "text=/upload complete/i",
        "text=/resume attached/i",
        "text=/cv attached/i",
    )
    try:
        # Prefer waiting for a visible attach affordance rather than sleep.
        combined = None
        for sel in deadline_markers:
            loc = page.locator(sel).first
            combined = loc if combined is None else combined.or_(loc)
        if combined is not None:
            try:
                combined.wait_for(state="visible", timeout=wait_ms)
            except Exception:
                pass
    except Exception:
        pass
    if _resume_ui_attached(page, pdf_path):
        return True
    if _file_input_has_files(page):
        logger.debug("force_upload_resume confirmed via input.files.length")
        return True
    return False


def force_upload_resume(page, pdf_path: Path | str) -> bool:
    """Force a fresh disk PDF upload — never reuse a saved/library resume.

    After ``set_input_files``, dispatches input/change so React registers the
    file, then verifies via UI text / Indeed attach card / ``el.files.length``.
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        return False
    _click_upload_affordance(page)
    if upload_resume_file(page, pdf):
        _dispatch_file_input_react_events(page)
        if _confirm_resume_upload(page, pdf):
            logger.debug("force_upload_resume confirmed stem=%s", pdf.stem)
            return True
        logger.debug(
            "force_upload_resume set files but UI/files not confirmed yet (%s)",
            pdf.stem,
        )
    # Second affordance click then retry (modal may open after first click)
    _click_upload_affordance(page)
    if upload_resume_file(page, pdf):
        _dispatch_file_input_react_events(page)
        if _confirm_resume_upload(page, pdf):
            logger.info("force_upload_resume succeeded on retry for %s", pdf.name)
            return True
    return False


def success_text(body: str) -> bool:
    blob = (body or "").lower()
    markers = (
        "thank you for applying",
        "thanks for applying",
        "application submitted",
        "your application was sent",
        "application was sent",
        "we have received your application",
        "application received",
        "successfully submitted",
        "you applied",
        "application has been submitted",
        "thanks for your application",
        "your application has been received",
    )
    return any(m in blob for m in markers)


_SUCCESS_URL_NEEDLES = (
    "application-submitted",
    "application_submitted",
    "applicationsubmitted",
    "/thank",
    "thank-you",
    "thankyou",
    "confirmation",
    "applied=true",
)


def confirm_application_success(page) -> bool:
    """True only when the page shows verified post-submit evidence.

    Never treat a Submit click alone as success.
    """
    if page is None:
        return False
    try:
        body = page.inner_text("body", timeout=2500) or ""
    except Exception:
        body = ""
    if success_text(body):
        return True
    try:
        url = (page.url or "").lower()
    except Exception:
        url = ""
    if url and any(n in url for n in _SUCCESS_URL_NEEDLES):
        return True
    try:
        title = (page.title() or "").lower()
    except Exception:
        title = ""
    if title and any(
        m in title
        for m in ("thank you", "application submitted", "application received")
    ):
        return True
    return False


def submission_result(page, *, submitted_click: bool = False) -> bool:
    """Shared submit contract: click is ignored; confirmation required."""
    _ = submitted_click  # documented so callers pass intent without changing semantics
    return confirm_application_success(page)


# --- Enterprise harden helpers (new boards) ---------------------------------


_CF_IFRAME = (
    "iframe[src*='challenges.cloudflare.com'], "
    "iframe[src*='turnstile'], "
    "iframe[title*='Cloudflare' i], "
    "div.cf-turnstile, "
    "[data-sitekey]"
)


def cloudflare_turnstile_present(page) -> bool:
    """True when a Cloudflare / Turnstile challenge is visible or in DOM."""
    try:
        loc = page.locator(_CF_IFRAME).first
        if loc.count() and loc.is_visible(timeout=400):
            return True
    except Exception:
        pass
    try:
        body = (page.inner_text("body", timeout=800) or "").lower()
    except Exception:
        body = ""
    blob = f"{(page.url or '').lower()} {body}"
    needles = (
        "cf-turnstile",
        "cf-browser-verification",
        "challenges.cloudflare.com",
        "one more step before you proceed",
        "verify you are human",
        "attention required",
        "checking your browser",
    )
    return any(n in blob for n in needles)


def raise_if_cloudflare(
    page,
    site: str,
    *,
    company: str = "",
    job_url: str = "",
) -> None:
    """CF/Turnstile: human assist first; unresolved → Manual_Action (no Gemini).

    Circuit recording happens in browser_worker_loop on ApplicationBlockedError
    so we do not double-count the same wall.
    """
    if not cloudflare_turnstile_present(page):
        return
    from src.core.human_assist import ensure_captcha_cleared

    ensure_captcha_cleared(
        page,
        company=company or site,
        job_url=job_url,
        site=site,
    )


def human_type(locator, text: str, *, delay_ms: int = 10) -> bool:
    """Type like a human (delay between keys). Prefer press_sequentially, else type."""
    value = str(text or "")
    if not value:
        return False
    try:
        locator.click(timeout=2000)
    except Exception:
        pass
    try:
        locator.fill("", timeout=1500)
    except Exception:
        pass
    try:
        if hasattr(locator, "press_sequentially"):
            locator.press_sequentially(value, delay=max(1, int(delay_ms)))
            return True
    except Exception:
        logger.debug("press_sequentially failed — falling back to type", exc_info=True)
    try:
        locator.type(value, delay=max(1, int(delay_ms)), timeout=30_000)
        return True
    except Exception:
        try:
            locator.fill(value, timeout=5000)
            return True
        except Exception:
            return False


def resolve_pitch_short(
    *,
    yaml_snapshot: str | None,
    master_path: Path | str | None = None,
    company: str = "",
    job_title: str = "",
    jd_text: str = "",
) -> str:
    """Prefer embedded pitch_short → application_answers → template (no mid-apply LLM)."""
    from src.core.pitch import extract_pitch, _fallback_pitch

    existing = extract_pitch(yaml_snapshot)
    if existing and len(existing) >= 40:
        return existing
    answers: dict[str, Any] = {}
    if master_path:
        try:
            answers = load_applicant_profile(master_path, yaml_snapshot=yaml_snapshot)
        except Exception:
            answers = {}
    for key in ("why_this_role", "additional_information", "pitch_short", "cover_letter"):
        val = str(answers.get(key) or "").strip()
        if len(val) >= 40:
            co = (company or "").strip()
            if co and co.lower() not in val.lower():
                return f"{val} Excited about {co}."
            return val
    return _fallback_pitch(company or "the team", job_title or "")


def wait_for_any(page, selectors: list[str], *, timeout_ms: int = 5000):
    """Event-driven wait for the first matching visible locator (or None)."""
    if not selectors:
        return None
    combined = None
    for sel in selectors:
        loc = page.locator(sel).first
        combined = loc if combined is None else combined.or_(loc)
    if combined is None:
        return None
    try:
        combined.wait_for(state="visible", timeout=timeout_ms)
        return combined
    except Exception:
        return None


def settle(
    page,
    *,
    timeout_ms: int = 2000,
    selectors: list[str] | None = None,
) -> bool:
    """Wait until a selector is visible, success markers appear, or timeout.

    Prefer event-driven readiness; never burns the full budget if ready early.
    """
    if page is None:
        return False
    budget = max(100, int(timeout_ms))
    if selectors:
        return wait_for_any(page, selectors, timeout_ms=budget) is not None

    try:
        start_url = (page.url or "").strip()
    except Exception:
        start_url = ""

    deadline = time.monotonic() + (budget / 1000.0)
    slice_ms = 150
    while time.monotonic() < deadline:
        try:
            if confirm_application_success(page):
                return True
        except Exception:
            pass
        try:
            cur = (page.url or "").strip()
            if start_url and cur and cur != start_url:
                return True
        except Exception:
            pass
        try:
            ready = page.evaluate("() => document.readyState") or ""
            if ready == "complete":
                remaining = deadline - time.monotonic()
                if remaining <= 0.2:
                    return True
        except Exception:
            pass
        try:
            page.wait_for_timeout(slice_ms)
        except Exception:
            time.sleep(slice_ms / 1000.0)
    try:
        return confirm_application_success(page)
    except Exception:
        return False


def close_extra_pages(context, keep) -> None:
    """Close every page except ``keep`` to prevent tab/memory leaks."""
    if context is None or keep is None:
        return
    try:
        pages = list(context.pages)
    except Exception:
        return
    for p in pages:
        if p is keep:
            continue
        try:
            if not p.is_closed():
                p.close()
        except Exception:
            logger.debug("close_extra_pages failed", exc_info=True)


def capture_popup_url_then_close(
    context,
    primary_page,
    click_fn,
    *,
    popup_timeout_ms: int = 6000,
    nav_timeout_ms: int = 12_000,
) -> str:
    """Click CTA; if a new tab opens, read its URL, close it, return URL.

    Same-tab navigations return the primary page URL. Never leaves popup tabs open.
    Sync Playwright only (warm worker thread) — no asyncio.run.

    Important: do not click twice. ``expect_page`` still runs the click when no
    popup appears; a second click was wiping the offsite URL race (bake15b).
    """
    before = ""
    try:
        before = (primary_page.url or "").strip()
    except Exception:
        before = ""

    popup = None
    try:
        with context.expect_page(timeout=popup_timeout_ms) as wait_popup:
            click_fn()
        popup = wait_popup.value
    except Exception:
        # Click already happened inside expect_page (or click itself failed).
        try:
            primary_page.wait_for_function(
                """(prev) => {
                    const u = location.href || '';
                    if (!u || u === 'about:blank') return false;
                    if (prev && u !== prev && !u.includes('linkedin.com')) return true;
                    if (prev && u !== prev) return true;
                    return false;
                }""",
                arg=before,
                timeout=min(nav_timeout_ms, 8000),
            )
        except Exception:
            try:
                primary_page.wait_for_load_state(
                    "domcontentloaded", timeout=min(nav_timeout_ms, 5000)
                )
            except Exception:
                pass
        return (primary_page.url or "").strip()

    try:
        popup.wait_for_load_state("domcontentloaded", timeout=nav_timeout_ms)
    except Exception:
        pass
    # Offsite tabs often land about:blank then redirect — wait briefly.
    try:
        popup.wait_for_function(
            """() => {
                const u = location.href || '';
                return !!u && u !== 'about:blank' && !u.startsWith('chrome-error:');
            }""",
            timeout=5000,
        )
    except Exception:
        pass
    final_url = (popup.url or "").strip()
    try:
        if not popup.is_closed():
            popup.close()
    except Exception:
        logger.debug("popup close failed", exc_info=True)
    close_extra_pages(context, keep=primary_page)
    return final_url
