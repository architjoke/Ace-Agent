"""Shared playbook helpers (profile load + persistent Chromium)."""

from __future__ import annotations

import logging
import re
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


def try_cdp_assist(cdp_assist, reason: str) -> bool:
    """Invoke optional sync CDP assist callback; False on missing/failure."""
    if cdp_assist is None:
        return False
    try:
        return bool(cdp_assist(str(reason or "stuck")))
    except Exception:
        logger.debug("cdp_assist raised", exc_info=True)
        return False


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
        from src.core.salary import inject_salary_answers

        answers = inject_salary_answers(
            answers if isinstance(answers, dict) else {},
            jd_text=jd_text,
            yaml_snapshot=yaml_snapshot,
        )
    except Exception:
        answers = answers if isinstance(answers, dict) else {}
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
        "expected_ctc": str(answers.get("expected_ctc") or "").strip(),
        "salary_expectation": str(answers.get("salary_expectation") or "").strip(),
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

    from src.core.browser_identity import (
        pinned_device_scale_factor,
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
