"""Failure screenshots + human pause (Windows popup / toast).

When apply is about to fail, hit a captcha wall, or face an unknown screening
question, save a PNG under ``output/Failure_Shots/`` and pause so the operator
can take the lead in the live browser. Human-filled answers are learned for
future applies.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
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


def unknown_answer_human_enabled() -> bool:
    """Default on — set ACE_UNKNOWN_HUMAN=0 to skip popup and hard-fail."""
    from src.core.utils import env_truthy

    return env_truthy("ACE_UNKNOWN_HUMAN", "1")


def captcha_wait_seconds() -> float:
    try:
        return max(30.0, float(os.environ.get("ACE_CAPTCHA_WAIT_S") or "300"))
    except ValueError:
        return 300.0


def unknown_wait_seconds() -> float:
    try:
        return max(60.0, float(os.environ.get("ACE_UNKNOWN_WAIT_S") or "600"))
    except ValueError:
        return 600.0


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


def _message_box_blocking(title: str, message: str) -> None:
    """Blocking Windows MessageBox (call from a daemon thread only)."""
    try:
        import ctypes

        # MB_TOPMOST | MB_ICONWARNING | MB_OK
        flags = 0x00040000 | 0x00000030 | 0x00000000
        ctypes.windll.user32.MessageBoxW(0, message[:900], title[:120], flags)
    except Exception:
        logger.debug("MessageBox unavailable", exc_info=True)


def wait_for_operator(
    *,
    title: str,
    message: str,
    budget_s: float,
    page: Any = None,
    flag_name: str = "_operator_continue.flag",
    done_check: Callable[[], bool] | None = None,
) -> bool:
    """Wait for human help without exceeding ``budget_s``.

    Shows a toast + MessageBox on a daemon thread. Returns True if the dialog
    was dismissed, a continue flag appeared, ``done_check`` fired, or the page
    looks submitted — all before the budget expires. Returns False on timeout.
    """
    budget = max(5.0, float(budget_s))
    dismissed = threading.Event()
    flag = failure_shots_dir() / flag_name
    try:
        if flag.exists():
            flag.unlink()
    except OSError:
        pass

    _toast_windows(title, (message or "")[:240])

    def _run_box() -> None:
        try:
            _message_box_blocking(title, message)
        finally:
            dismissed.set()

    threading.Thread(target=_run_box, name="ace-operator-mbox", daemon=True).start()
    logger.warning(
        "Operator wait started (budget=%.0fs, flag=%s) — %s",
        budget,
        flag.name,
        (title or "")[:80],
    )
    # If MessageBox is unavailable the thread exits immediately; rely on flag/file.
    if dismissed.is_set():
        logger.warning(
            "No MessageBox — create %s after finishing in the browser to continue",
            flag,
        )

    deadline = time.time() + budget
    while time.time() < deadline:
        if dismissed.is_set():
            return True
        if flag.exists():
            try:
                flag.unlink()
            except OSError:
                pass
            return True
        if done_check is not None:
            try:
                if done_check():
                    return True
            except Exception:
                pass
        if page is not None:
            try:
                if form_looks_submitted(page):
                    return True
            except Exception:
                pass
        time.sleep(1.0)
    logger.warning("Operator wait timed out after %.0fs (%s)", budget, title[:80])
    return False


def notify_captcha(company: str, shot: Path | None) -> tuple[str, str]:
    """Build captcha notify copy (toast+box run via ``wait_for_operator``)."""
    title = "Ace — CAPTCHA needed"
    company_s = (company or "this job").strip() or "this job"
    shot_s = str(shot) if shot else "(screenshot unavailable)"
    message = (
        f"Solve the CAPTCHA for {company_s} in the Ace browser window, "
        f"then click OK so Ace can finish submitting.\n\n"
        f"Screenshot: {shot_s}"
    )
    return title, message


def notify_unknown_answer(
    company: str,
    shot: Path | None,
    *,
    reason: str = "",
) -> tuple[str, str]:
    """Build unknown-answer notify copy (toast+box via ``wait_for_operator``)."""
    title = "Ace — need your answer"
    company_s = (company or "this job").strip() or "this job"
    shot_s = str(shot) if shot else "(screenshot unavailable)"
    reason_s = (reason or "unknown screening field").strip()[:220]
    message = (
        f"Ace does not know how to answer a field for {company_s}.\n\n"
        f"1. Switch to the Ace browser window\n"
        f"2. Fill the unanswered / highlighted fields (and click Next/Submit if ready)\n"
        f"3. Click OK here — Ace will learn your answers for future jobs\n\n"
        f"Stuck on: {reason_s}\n"
        f"Screenshot: {shot_s}"
    )
    return title, message


_SNAPSHOT_JS = """
(() => {
  const sels = 'input, textarea, select';
  const nodes = Array.from(document.querySelectorAll(sels));
  const clip = (s, n) => {
    s = (s || '').replace(/\\s+/g, ' ').trim();
    return s.length > n ? s.slice(0, n) : s;
  };
  const isVisible = (el) => {
    if (!el || el.getAttribute('aria-hidden') === 'true') return false;
    const st = window.getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden' || Number(st.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const labelFor = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return clip(aria, 120);
    if (el.id) {
      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lab) return clip(lab.innerText || lab.textContent, 120);
    }
    const wrap = el.closest('label');
    if (wrap) return clip(wrap.innerText || wrap.textContent, 120);
    const prev = el.closest('div,li,fieldset');
    if (prev) {
      const lab = prev.querySelector('label,legend,span,p');
      if (lab) return clip(lab.innerText || lab.textContent, 120);
    }
    return clip(el.getAttribute('name') || el.getAttribute('placeholder') || '', 120);
  };
  const out = [];
  for (const el of nodes) {
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (['hidden', 'password', 'file', 'submit', 'button', 'image', 'reset'].includes(type)) continue;
    if (!isVisible(el)) continue;
    let value = '';
    if (el.tagName === 'SELECT') {
      const opt = el.options && el.selectedIndex >= 0 ? el.options[el.selectedIndex] : null;
      value = clip((opt && (opt.text || opt.value)) || el.value || '', 200);
    } else if (type === 'checkbox' || type === 'radio') {
      if (!el.checked) continue;
      value = clip(el.value || 'Yes', 80);
    } else {
      value = clip(el.value || '', 200);
    }
    if (!value) continue;
    out.push({ label: labelFor(el), value, type: type || (el.tagName || '').toLowerCase() });
    if (out.length >= 60) break;
  }
  return out;
})()
"""


def snapshot_filled_fields(page: Any) -> list[tuple[str, str]]:
    """Return (label, value) pairs currently filled on the visible form."""
    if page is None:
        return []
    try:
        raw = page.evaluate(_SNAPSHOT_JS)
    except Exception:
        logger.debug("form snapshot failed", exc_info=True)
        return []
    if not isinstance(raw, list):
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        value = str(item.get("value") or "").strip()
        if not label or not value:
            continue
        key = f"{label.lower()}::{value.lower()}"
        if key in seen:
            continue
        seen.add(key)
        pairs.append((label, value))
    return pairs


def _diff_new_values(
    before: list[tuple[str, str]],
    after: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    before_map = {label.lower(): value for label, value in before}
    learned: list[tuple[str, str]] = []
    for label, value in after:
        prev = before_map.get(label.lower())
        if prev is None or prev != value:
            learned.append((label, value))
    # If nothing changed but form has values, still teach from current fills
    # (human may have filled before the popup appeared).
    if not learned and after:
        return list(after)
    return learned


def learn_from_page(
    page: Any,
    *,
    company: str = "",
    job_url: str = "",
    before: list[tuple[str, str]] | None = None,
) -> int:
    """Compare current form to ``before`` and persist new answers. Returns count."""
    after = snapshot_filled_fields(page)
    pairs = _diff_new_values(before or [], after)
    if not pairs:
        return 0
    from src.core.learned_answers import append_lesson_from_learning, remember_many
    from src.core.profile_promote import promote_learned_to_master

    n = remember_many(pairs, company=company, job_url=job_url)
    # Catch-up: mirror eligible fixed keys into local master.yaml application_answers.
    try:
        promote_learned_to_master()
    except Exception:
        logger.debug("promote_learned_to_master after teach failed", exc_info=True)
    # One lesson line for the newest / longest question (signal, not spam).
    if pairs:
        q, a = max(pairs, key=lambda p: len(p[0]))
        append_lesson_from_learning(q, a)
    return n


def form_looks_submitted(page: Any) -> bool:
    """Best-effort success detection after human takeover."""
    if page is None:
        return False
    try:
        from src.playbooks._common import success_text

        body = page.inner_text("body", timeout=2000) or ""
        if success_text(body):
            return True
        lower = body.lower()
        markers = (
            "application sent",
            "your application was sent",
            "application submitted",
            "successfully applied",
        )
        return any(m in lower for m in markers)
    except Exception:
        return False


def resolve_unknown_with_human(
    page: Any,
    *,
    company: str = "",
    job_url: str = "",
    reason: str = "",
    timeout_s: float | None = None,
) -> bool:
    """Pause for human to answer unknown fields; learn answers.

    Returns True only when answers were learned or the form looks submitted.
    Empty dismiss / budget expiry without progress returns False.
    """
    if page is None or not unknown_answer_human_enabled():
        capture_failure_shot(
            page,
            company=company,
            reason="unknown_no_human",
            job_url=job_url,
        )
        return False

    budget = float(timeout_s if timeout_s is not None else unknown_wait_seconds())
    shot = capture_failure_shot(
        page,
        company=company,
        reason="unknown_answer_wait",
        job_url=job_url,
    )
    before = snapshot_filled_fields(page)
    title, message = notify_unknown_answer(company, shot, reason=reason)
    logger.warning(
        "Unknown-answer gate for %s — waiting for human (reason=%s, shot=%s, budget=%.0fs)",
        company or "unknown",
        (reason or "")[:160],
        shot,
        budget,
    )
    wait_for_operator(
        title=title,
        message=message,
        budget_s=budget,
        page=page,
        flag_name="_unknown_continue.flag",
    )

    learned_n = 0
    try:
        learned_n = learn_from_page(
            page, company=company, job_url=job_url, before=before
        )
    except Exception:
        logger.debug("learn_from_page failed", exc_info=True)

    submitted = False
    try:
        submitted = form_looks_submitted(page)
    except Exception:
        submitted = False

    helped = bool(submitted or learned_n > 0)
    logger.info(
        "Human unknown-answer session done for %s (learned=%s, submitted=%s, helped=%s)",
        company or "unknown",
        learned_n,
        submitted,
        helped,
    )
    return helped


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
    shot = capture_failure_shot(
        page, company=company, reason="captcha_wait", job_url=job_url
    )
    title, message = notify_captcha(company, shot)
    logger.warning(
        "CAPTCHA gate for %s — waiting up to %.0fs for human solve (shot=%s)",
        company or "unknown",
        budget,
        shot,
    )

    def _cleared() -> bool:
        return not page_shows_captcha(page)

    cleared = wait_for_operator(
        title=title,
        message=message,
        budget_s=budget,
        page=page,
        flag_name="_captcha_continue.flag",
        done_check=_cleared,
    )
    try:
        if cleared and not page_shows_captcha(page):
            logger.info("CAPTCHA cleared for %s — continuing apply", company or "")
            return True
        if not page_shows_captcha(page):
            logger.info("CAPTCHA cleared for %s — continuing apply", company or "")
            return True
    except Exception:
        logger.debug("captcha post-wait check failed", exc_info=True)

    capture_failure_shot(
        page, company=company, reason="captcha_timeout", job_url=job_url
    )
    logger.warning("CAPTCHA not cleared in time for %s", company or "unknown")
    return False


def ensure_captcha_cleared(
    page: Any,
    *,
    company: str = "",
    job_url: str = "",
    site: str = "",
    timeout_s: float | None = None,
) -> None:
    """Human-first captcha/CF gate; unresolved → Manual_Action via ApplicationBlockedError.

    Call from playbooks whenever a bot wall is detected. Never leave the worker
    spinning: either the human clears it, or we package Manual_Action and move on.
    """
    from src.core.executor import ApplicationBlockedError

    if page is None:
        return

    cf = False
    try:
        from src.playbooks._common import cloudflare_turnstile_present

        cf = bool(cloudflare_turnstile_present(page))
    except Exception:
        cf = False

    if not page_shows_captcha(page) and not cf:
        return

    label = (site or company or "apply").strip() or "apply"
    if not captcha_human_enabled():
        capture_failure_shot(
            page,
            company=company or label,
            reason="captcha_human_disabled",
            job_url=job_url,
        )
        raise ApplicationBlockedError(
            f"{label}: captcha/CF wall (human assist disabled) — Manual_Action"
        )

    if resolve_captcha_with_human(
        page,
        company=company or label,
        job_url=job_url,
        timeout_s=timeout_s,
    ):
        # Confirm wall is gone (incl. CF) before continuing.
        still_cf = False
        try:
            from src.playbooks._common import cloudflare_turnstile_present

            still_cf = bool(cloudflare_turnstile_present(page))
        except Exception:
            still_cf = False
        if not page_shows_captcha(page) and not still_cf:
            return

    raise ApplicationBlockedError(
        f"{label}: captcha/CF unresolved after human wait — Manual_Action"
    )


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


def is_mechanical_stuck_reason(reason: str) -> bool:
    """True for upload/click/timeout mechanics — never interrupt a human for these."""
    r = (reason or "").lower()
    needles = (
        "action=upload",
        "set_input_files",
        "upload resume",
        "force_upload",
        "disk resume",
        "file input",
        "locator.set_input_files",
        "no file upload",
        "resume not attached",
        "widget stuck action=upload",
    )
    return any(n in r for n in needles)


def try_human_unknown(
    page: Any,
    *,
    company: str = "",
    job_url: str = "",
    reason: str = "",
) -> bool:
    """Pause + learn only for unknown screening answers — never for resume upload."""
    if is_mechanical_stuck_reason(reason):
        logger.info(
            "Skipping human popup for mechanical stuck reason: %s",
            (reason or "")[:160],
        )
        return False
    try:
        return bool(
            resolve_unknown_with_human(
                page,
                company=company,
                job_url=job_url,
                reason=reason,
            )
        )
    except Exception:
        logger.debug("try_human_unknown failed", exc_info=True)
        return False
