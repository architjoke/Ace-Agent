"""Failure screenshots + human captcha pause (Windows popup / toast).

When apply is about to fail or hit a captcha wall, save a PNG under
``output/Failure_Shots/`` and (for captcha) pause so the operator can solve it
in the live browser, then continue toward SUCCESS.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CAPTCHA_NEEDLES = (
    "type below image",
    "verify you are human",
    "confirm you are human",
    "recaptcha",
    "hcaptcha",
    "captcha",
    "amzn-captcha",
    "solve this puzzle",
    "unusual traffic",
    "cf-challenge",
    "cloudflare",
)


def _output_root() -> Path:
    try:
        from src.core.settings import get_settings

        return Path(get_settings().output_dir)
    except Exception:
        from src.core.config import OUTPUT_DIR

        return Path(OUTPUT_DIR)


def _safe_name(text: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", (text or "").strip()).strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned or "unknown")[:80]


def failure_shots_dir() -> Path:
    d = _output_root() / "Failure_Shots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def captcha_human_enabled() -> bool:
    """Default on — set ACE_CAPTCHA_HUMAN=0 to keep hard-fail after screenshot."""
    from src.core.utils import env_truthy

    return env_truthy("ACE_CAPTCHA_HUMAN", "1")


def captcha_wait_seconds() -> float:
    try:
        return max(30.0, float(os.environ.get("ACE_CAPTCHA_WAIT_S") or "300"))
    except ValueError:
        return 300.0


def page_shows_captcha(page: Any) -> bool:
    """True when visible page text/DOM looks like a captcha / bot wall."""
    if page is None:
        return False
    try:
        body = (page.inner_text("body", timeout=1500) or "").lower()
    except Exception:
        body = ""
    try:
        title = (page.title() or "").lower()
    except Exception:
        title = ""
    blob = f"{title}\n{body}"
    if any(n in blob for n in _CAPTCHA_NEEDLES):
        return True
    try:
        for sel in (
            "iframe[src*='recaptcha']",
            "iframe[src*='hcaptcha']",
            "iframe[title*='captcha' i]",
            "#captcha",
            ".g-recaptcha",
            "[data-sitekey]",
        ):
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=400):
                return True
    except Exception:
        pass
    return False


def capture_failure_shot(
    page: Any,
    *,
    company: str = "",
    reason: str = "",
    job_url: str = "",
) -> Path | None:
    """Save full-page PNG + sidecar note. Safe no-op if page is gone."""
    if page is None:
        return None
    try:
        if getattr(page, "is_closed", None) and page.is_closed():
            return None
    except Exception:
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    company_dir = failure_shots_dir() / _safe_name(company or "unknown")
    company_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_name(reason or "fail")[:48] or "fail"
    png = company_dir / f"{ts}_{slug}.png"
    note = company_dir / f"{ts}_{slug}.txt"

    try:
        url = ""
        try:
            url = (page.url or "").strip()
        except Exception:
            url = ""
        page.screenshot(path=str(png), full_page=True, timeout=15_000)
        note.write_text(
            "\n".join(
                [
                    f"time_utc={ts}",
                    f"company={company}",
                    f"reason={reason}",
                    f"job_url={job_url or url}",
                    f"page_url={url}",
                    f"shot={png}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        logger.warning(
            "Failure screenshot saved → %s (%s)",
            png,
            (reason or "")[:120],
        )
        return png
    except Exception:
        logger.debug("failure screenshot failed", exc_info=True)
        return None


def _toast_windows(title: str, message: str) -> None:
    """Best-effort non-blocking toast (PowerShell). Never raises."""
    try:
        import subprocess

        # Escape for single-quoted PowerShell string.
        t = (title or "Ace").replace("'", "''")[:80]
        m = (message or "").replace("'", "''")[:240]
        script = (
            f"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            f"ContentType = WindowsRuntime] > $null; "
            f"$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
            f"[Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
            f"$text = $template.GetElementsByTagName('text'); "
            f"$text.Item(0).AppendChild($template.CreateTextNode('{t}')) | Out-Null; "
            f"$text.Item(1).AppendChild($template.CreateTextNode('{m}')) | Out-Null; "
            f"$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
            f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Ace Agent')"
            f".Show($toast)"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        logger.debug("toast notify failed", exc_info=True)


def _message_box(title: str, message: str) -> None:
    """Blocking Windows MessageBox so the operator can finish the captcha first."""
    try:
        import ctypes

        # MB_TOPMOST | MB_ICONWARNING | MB_OK
        flags = 0x00040000 | 0x00000030 | 0x00000000
        ctypes.windll.user32.MessageBoxW(0, message[:900], title[:120], flags)
        return
    except Exception:
        logger.debug("MessageBox unavailable", exc_info=True)
    # Fallback: brief console pause file for headless/dev.
    try:
        flag = failure_shots_dir() / "_captcha_continue.flag"
        if flag.exists():
            flag.unlink()
        logger.warning(
            "%s — solve captcha in the browser, then create %s to continue",
            title,
            flag,
        )
        deadline = time.time() + min(captcha_wait_seconds(), 120)
        while time.time() < deadline:
            if flag.exists():
                try:
                    flag.unlink()
                except OSError:
                    pass
                return
            time.sleep(1.5)
    except Exception:
        pass


def notify_captcha(company: str, shot: Path | None) -> None:
    title = "Ace — CAPTCHA needed"
    company_s = (company or "this job").strip() or "this job"
    shot_s = str(shot) if shot else "(screenshot unavailable)"
    message = (
        f"Solve the CAPTCHA for {company_s} in the Ace browser window, "
        f"then click OK so Ace can finish submitting.\n\n"
        f"Screenshot: {shot_s}"
    )
    _toast_windows(title, f"Solve captcha for {company_s}, then click OK in the dialog.")
    # MessageBox must run on a thread that can pump UI — worker thread is OK on Win32.
    _message_box(title, message)


def resolve_captcha_with_human(
    page: Any,
    *,
    company: str = "",
    job_url: str = "",
    timeout_s: float | None = None,
) -> bool:
    """Pause for human captcha solve. Returns True if captcha cleared in time."""
    if page is None or not captcha_human_enabled():
        capture_failure_shot(
            page, company=company, reason="captcha_no_human", job_url=job_url
        )
        return False

    budget = float(timeout_s if timeout_s is not None else captcha_wait_seconds())
    deadline = time.time() + budget
    shot = capture_failure_shot(
        page, company=company, reason="captcha_wait", job_url=job_url
    )
    logger.warning(
        "CAPTCHA gate for %s — waiting up to %.0fs for human solve (shot=%s)",
        company or "unknown",
        budget,
        shot,
    )
    notify_captcha(company, shot)

    while time.time() < deadline:
        try:
            if not page_shows_captcha(page):
                logger.info("CAPTCHA cleared for %s — continuing apply", company or "")
                return True
        except Exception:
            logger.debug("captcha poll failed", exc_info=True)
        # Another prompt if still blocked (operator may need more time).
        remaining = deadline - time.time()
        if remaining < 5:
            break
        notify_captcha(company, shot)
        try:
            page.wait_for_timeout(1500)
        except Exception:
            time.sleep(1.5)

    capture_failure_shot(
        page, company=company, reason="captcha_timeout", job_url=job_url
    )
    logger.warning("CAPTCHA not cleared in time for %s", company or "unknown")
    return False


def shot_on_fail(
    page: Any,
    *,
    company: str = "",
    reason: str = "",
    job_url: str = "",
) -> Path | None:
    """Convenience alias used by playbooks / executor before raising."""
    return capture_failure_shot(
        page, company=company, reason=reason, job_url=job_url
    )
