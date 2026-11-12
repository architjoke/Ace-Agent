"""Pure Playwright Greenhouse Easy Apply / job board form playbook."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.playbooks import PlaybookFallback

logger = logging.getLogger(__name__)

# Per-action timeout (goto uses goto_dom's 15s). Keep fills/submits forgiving.
_DEFAULT_TIMEOUT_MS = 30_000


def _load_profile(master_yaml_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(master_yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    answers = data.get("application_answers") if isinstance(data.get("application_answers"), dict) else {}
    cv = data.get("cv") if isinstance(data.get("cv"), dict) else {}
    name = str(cv.get("name") or answers.get("full_name") or "").strip()
    parts = name.split(None, 1)
    first = str(answers.get("first_name") or (parts[0] if parts else "")).strip()
    last = str(
        answers.get("last_name") or (parts[1] if len(parts) > 1 else "")
    ).strip()
    phone = str(
        answers.get("phone_display")
        or answers.get("phone_e164")
        or cv.get("phone")
        or ""
    ).strip()
    email = str(answers.get("email") or cv.get("email") or "").strip()
    location = str(
        answers.get("full_location") or answers.get("address_line") or cv.get("location") or ""
    ).strip()
    return {
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": phone,
        "location": location,
        "linkedin": str(answers.get("linkedin_url") or "").strip(),
        "github": str(answers.get("github_url") or "").strip(),
    }


def _fill_first(page, selectors: list[str], value: str) -> bool:
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


def _upload_resume(page, pdf_path: Path) -> bool:
    from src.playbooks._common import force_upload_resume

    if force_upload_resume(page, pdf_path):
        return True
    inputs = page.locator('input[type="file"]')
    try:
        n = inputs.count()
    except Exception:
        n = 0
    if n == 0:
        return False
    for i in range(n):
        inp = inputs.nth(i)
        try:
            inp.set_input_files(str(pdf_path), timeout=10_000)
            return True
        except Exception:
            continue
    return False


def _has_unanswered_custom(page) -> bool:
    """Heuristic: required textareas / selects still empty → fallback to vision."""
    try:
        required = page.locator(
            "textarea[required], select[required], "
            "input[required]:not([type='hidden']):not([type='file']):not([type='checkbox']):not([type='radio'])"
        )
        count = required.count()
    except Exception:
        return False
    for i in range(min(count, 12)):
        el = required.nth(i)
        try:
            tag = (el.evaluate("e => e.tagName") or "").lower()
            val = el.input_value(timeout=500) if tag != "select" else el.input_value(timeout=500)
            if not str(val or "").strip():
                # Skip if our known fields
                name = (el.get_attribute("name") or "") + (el.get_attribute("id") or "")
                if re.search(r"first.?name|last.?name|email|phone|resume", name, re.I):
                    continue
                return True
        except Exception:
            continue
    return False


def run_greenhouse(
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
    """Fill a Greenhouse board form via Playwright DOM.

    Raises TimeoutError / PlaybookFallback on failure so caller can Gemini-fallback.
    Owned launches close Chromium; warm slots only reset context state.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    from src.core.network import playwright_proxy_dict
    from src.core.settings import get_settings
    from src.playbooks._common import try_cdp_assist

    _ = kwargs
    pdf = Path(pdf_path).expanduser().resolve()
    master = Path(master_yaml_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if not master.is_file():
        raise FileNotFoundError(f"master.yaml not found: {master}")

    settings = get_settings()
    profile = Path(user_data_dir) if user_data_dir else Path(settings.browser_user_data_dir)
    profile = profile.expanduser().resolve()
    profile.mkdir(parents=True, exist_ok=True)

    use_headless = settings.browser_headless if headless is None else bool(headless)
    proxy = playwright_proxy_dict()
    _ = yaml_snapshot  # accepted for playbook kwargs parity; no salary fields yet
    fields = _load_profile(master)

    owned = context is None
    handoff = False
    pw = None
    if owned:
        # Clear singleton locks so Playwright can attach
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            lock = profile / name
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
        pw = sync_playwright().start()
    try:
        if owned:
            from src.core.browser_identity import pinned_user_agent, pinned_viewport
            from src.playbooks._common import CHROMIUM_STEALTH_ARGS

            launch_kwargs: dict[str, Any] = {
                "user_data_dir": str(profile),
                "headless": use_headless,
                "args": list(CHROMIUM_STEALTH_ARGS),
                "user_agent": pinned_user_agent(user_data_dir=profile),
                "viewport": pinned_viewport(),
                "accept_downloads": False,
                "permissions": [],
            }
            if proxy:
                launch_kwargs["proxy"] = proxy
            context = pw.chromium.launch_persistent_context(**launch_kwargs)
            try:
                context.clear_permissions()
            except Exception:
                pass
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(timeout_ms)

        logger.info("Greenhouse playbook navigating %s", job_url)
        from src.playbooks._common import goto_dom

        goto_dom(page, job_url)
        from src.core.job_closed import page_shows_job_closed

        if page_shows_job_closed(page):
            from src.core.executor import JobClosedError

            raise JobClosedError(f"Greenhouse job closed/missing at {job_url[:160]}")

        # Only open the form when identity fields are not already present.
        # Avoid clicking "Submit application" on boards that land on the form URL.
        form_ready = False
        try:
            form_ready = bool(
                page.locator(
                    "#first_name, input[name='job_application[first_name]'], "
                    "input[type='email'], #email"
                ).count()
            )
        except Exception:
            form_ready = False
        if not form_ready:
            for label in (
                "Apply",
                "Apply for this job",
                "Start application",
            ):
                btn = page.get_by_role("button", name=re.compile(label, re.I))
                link = page.get_by_role("link", name=re.compile(label, re.I))
                try:
                    if btn.count() and btn.first.is_visible():
                        btn.first.click(timeout=5000)
                        page.wait_for_timeout(800)
                        break
                    if link.count() and link.first.is_visible():
                        link.first.click(timeout=5000)
                        page.wait_for_timeout(800)
                        break
                except Exception:
                    continue

        filled_any = False
        filled_any |= _fill_first(
            page,
            [
                "#first_name",
                "input[name='job_application[first_name]']",
                "input[autocomplete='given-name']",
                "input[name*='first_name' i]",
            ],
            fields["first_name"],
        )
        filled_any |= _fill_first(
            page,
            [
                "#last_name",
                "input[name='job_application[last_name]']",
                "input[autocomplete='family-name']",
                "input[name*='last_name' i]",
            ],
            fields["last_name"],
        )
        filled_any |= _fill_first(
            page,
            [
                "#email",
                "input[name='job_application[email]']",
                "input[type='email']",
                "input[autocomplete='email']",
            ],
            fields["email"],
        )
        filled_any |= _fill_first(
            page,
            [
                "#phone",
                "input[name='job_application[phone]']",
                "input[type='tel']",
                "input[autocomplete='tel']",
            ],
            fields["phone"],
        )
        if fields["location"]:
            _fill_first(
                page,
                [
                    "#job_application_location",
                    "input[name='job_application[location]']",
                    "input[placeholder*='Location' i]",
                ],
                fields["location"],
            )
        if fields["linkedin"]:
            _fill_first(
                page,
                [
                    "input[name*='linkedin' i]",
                    "input[id*='linkedin' i]",
                ],
                fields["linkedin"],
            )

        uploaded = _upload_resume(page, pdf)
        if not uploaded:
            reason = "Greenhouse playbook: resume file input not found"
            if try_cdp_assist(cdp_assist, reason, page=page) and _upload_resume(page, pdf):
                pass
            else:
                raise PlaybookFallback(reason)
        if not filled_any and not fields["email"]:
            raise PlaybookFallback("Greenhouse playbook: no profile fields to fill")

        if _has_unanswered_custom(page):
            reason = "Greenhouse playbook: unanswered custom/required fields — Gemini"
            if try_cdp_assist(cdp_assist, reason, page=page) and not _has_unanswered_custom(page):
                pass
            else:
                raise PlaybookFallback(reason)

        submitted = False
        for name in (
            "Submit application",
            "Submit Application",
            "Submit",
            "Apply",
            "Send application",
        ):
            btn = page.get_by_role("button", name=re.compile(rf"^{re.escape(name)}$", re.I))
            try:
                if btn.count() and btn.first.is_visible():
                    btn.first.click(timeout=8000)
                    submitted = True
                    break
            except Exception:
                continue
        if not submitted:
            for sel in (
                "input[type='submit']",
                "button[type='submit']",
                "#submit_app",
            ):
                try:
                    loc = page.locator(sel).first
                    if loc.count() and loc.is_visible():
                        loc.click(timeout=8000)
                        submitted = True
                        break
                except Exception:
                    continue

        if not submitted:
            reason = "Greenhouse playbook: submit control not found"
            if try_cdp_assist(cdp_assist, reason, page=page):
                for name in (
                    "Submit application",
                    "Submit Application",
                    "Submit",
                    "Apply",
                    "Send application",
                ):
                    btn = page.get_by_role(
                        "button", name=re.compile(rf"^{re.escape(name)}$", re.I)
                    )
                    try:
                        if btn.count() and btn.first.is_visible():
                            btn.first.click(timeout=8000)
                            submitted = True
                            break
                    except Exception:
                        continue
            if not submitted:
                raise PlaybookFallback(reason)

        from src.playbooks._common import settle

        settle(page, timeout_ms=2500)
        body = ""
        try:
            body = (page.inner_text("body") or "").lower()
        except Exception:
            pass
        success_markers = (
            "thank you for applying",
            "application submitted",
            "thanks for applying",
            "we have received your application",
            "application received",
        )
        if any(m in body for m in success_markers):
            logger.info("Greenhouse playbook success for %s", job_url)
            return True

        if "error" in body and "required" in body:
            raise PlaybookFallback("Greenhouse playbook: validation errors after submit")
        raise PlaybookFallback(
            "Greenhouse playbook: submitted without confirmation — Gemini"
        )

    except PlaywrightTimeout as exc:
        from src.playbooks.session import raise_handoff

        handoff = context is not None and page is not None
        raise_handoff(
            f"Greenhouse playbook timed out on {job_url}",
            context=context,
            page=page,
            user_data_dir=user_data_dir or profile,
            playwright=pw,
            cause=exc,
            fallback=TimeoutError(f"Greenhouse playbook timed out on {job_url}"),
        )
    except PlaybookFallback as exc:
        from src.playbooks.session import raise_handoff

        handoff = context is not None and page is not None
        raise_handoff(
            str(exc),
            context=context,
            page=page,
            user_data_dir=user_data_dir or profile,
            playwright=pw,
            cause=exc,
        )
    finally:
        if not handoff:
            if owned:
                from src.playbooks._common import close_context_safe

                close_context_safe(context, user_data_dir=user_data_dir or profile)
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
                    logger.debug("warm greenhouse reset failed", exc_info=True)
