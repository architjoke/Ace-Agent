"""Shared playbook helpers (profile load + persistent Chromium)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Anti-detection flags for Google OAuth / ATS persistent Chromium.
CHROMIUM_STEALTH_ARGS: list[str] = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
]

_CHROME_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36",
)


def pick_chrome_user_agent() -> str:
    """Stable-enough desktop Chrome UA (rotates lightly per process)."""
    import random

    return random.choice(_CHROME_USER_AGENTS)

# DOM-ready navigation — ignore tracking-pixel networkidle hangs.
GOTO_DOM_TIMEOUT_MS = 15_000


def goto_dom(page, url: str, *, timeout_ms: int = GOTO_DOM_TIMEOUT_MS) -> None:
    """Navigate with wait_until=domcontentloaded; one reload retry on timeout."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    except Exception as first_exc:
        logger.warning(
            "goto_dom timeout/fail on %s (%s) — reloading once",
            (url or "")[:160],
            type(first_exc).__name__,
        )
        try:
            page.reload(wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception as reload_exc:
            raise TimeoutError(
                f"DOM navigation failed for {url[:160]}: {first_exc}; reload={reload_exc}"
            ) from reload_exc


def load_applicant_profile(master_yaml_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(master_yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    answers = (
        data.get("application_answers")
        if isinstance(data.get("application_answers"), dict)
        else {}
    )
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
    return {
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
    }


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

    use_headless = settings.browser_headless if headless is None else bool(headless)
    proxy_cfg = proxy if proxy is not None else playwright_proxy_dict()
    launch_kwargs: dict[str, Any] = {
        "user_data_dir": str(profile),
        "headless": use_headless,
        "args": list(CHROMIUM_STEALTH_ARGS),
        "user_agent": pick_chrome_user_agent(),
    }
    if proxy_cfg:
        launch_kwargs["proxy"] = proxy_cfg
    context = playwright.chromium.launch_persistent_context(**launch_kwargs)
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


def success_text(body: str) -> bool:
    blob = (body or "").lower()
    markers = (
        "thank you for applying",
        "thanks for applying",
        "application submitted",
        "we have received your application",
        "application received",
        "successfully submitted",
        "you applied",
    )
    return any(m in blob for m in markers)
