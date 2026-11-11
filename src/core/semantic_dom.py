"""Skyvern-inspired semantic DOM apply: JS extract → Groq plan → Playwright execute."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.core.proxy_session import ProxySessionManager, looks_blocked
from src.core.utils import env_truthy

logger = logging.getLogger(__name__)

EXTRACT_INTERACTIVE_JS = """
(() => {
  const sels = 'input, textarea, select, button, a[href], [role="button"], [contenteditable="true"]';
  const nodes = Array.from(document.querySelectorAll(sels));
  const out = [];
  let aceId = 1;
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
    if (aria) return clip(aria, 80);
    if (el.id) {
      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lab) return clip(lab.innerText || lab.textContent, 80);
    }
    const wrap = el.closest('label');
    if (wrap) return clip(wrap.innerText || wrap.textContent, 80);
    return '';
  };
  for (const el of nodes) {
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (type === 'hidden') continue;
    if (!isVisible(el)) continue;
    el.setAttribute('data-ace-id', String(aceId));
    out.push({
      id: aceId,
      tag: (el.tagName || '').toLowerCase(),
      type: type || null,
      role: el.getAttribute('role') || null,
      name: clip(el.getAttribute('name') || el.getAttribute('aria-label') || '', 60),
      label: labelFor(el),
      placeholder: clip(el.getAttribute('placeholder') || '', 60),
      value: clip(el.value || '', 40),
      href: el.tagName === 'A' ? clip(el.getAttribute('href') || '', 80) : null,
      disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
      text: clip(el.innerText || el.textContent || '', 60),
    });
    aceId += 1;
    if (out.length >= 80) break;
  }
  return out;
})()
"""


def extract_interactive(page) -> list[dict[str, Any]]:
    """Run DOM inventory JS; return compact interactive element list."""
    try:
        raw = page.evaluate(EXTRACT_INTERACTIVE_JS)
    except Exception:
        logger.exception("semantic DOM extract failed")
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:80]:
        if isinstance(item, dict) and item.get("id") is not None:
            out.append(item)
    return out


def _answers_dict(master_path: Path) -> dict[str, Any]:
    from src.playbooks._common import load_applicant_profile

    return load_applicant_profile(master_path)


def plan_semantic_actions(
    elements: list[dict[str, Any]],
    *,
    answers: dict[str, Any],
    resume_path: str,
) -> list[dict[str, Any]]:
    """Ask Groq/text failover for a compact action list keyed by ace id."""
    if not elements:
        return []
    from src.core.llm_router import DEFAULT_GROQ_FALLBACK_MODEL, invoke_with_failover
    from src.core.llm_router import message_text

    slim = [
        {
            "id": e.get("id"),
            "tag": e.get("tag"),
            "type": e.get("type"),
            "label": e.get("label") or e.get("name") or e.get("placeholder") or e.get("text"),
            "disabled": e.get("disabled"),
        }
        for e in elements
    ]
    prompt = (
        "You plan Playwright actions for a job application form.\n"
        "Return ONLY a JSON array of actions. Allowed actions:\n"
        '{"action":"fill","id":N,"text":"..."}, '
        '{"action":"click","id":N}, '
        '{"action":"upload","id":N}\n'
        "Fill ALL visible identity fields from ANSWERS in ONE list. "
        "Use upload once for the resume file input — NEVER click a saved/library "
        "resume card; always Upload/Replace from disk. "
        "Click Apply/Submit/Continue/Next only after fields are filled. "
        "Skip disabled elements. Do not invent facts.\n\n"
        f"RESUME_PATH: {resume_path}\n"
        f"ANSWERS: {json.dumps(answers, ensure_ascii=False)[:2500]}\n"
        f"ELEMENTS: {json.dumps(slim, ensure_ascii=False)[:6000]}\n"
    )
    try:
        # Role "semantic": Groq-first short JSON; prefer 20b for TPM headroom.
        raw = invoke_with_failover(
            "semantic",
            prompt,
            temperature=0,
            model=DEFAULT_GROQ_FALLBACK_MODEL,
        )
        text = message_text(getattr(raw, "content", raw) if raw is not None else "")
    except Exception:
        logger.exception("semantic plan LLM failed")
        return []
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    blob = fence.group(1).strip() if fence else text
    match = re.search(r"\[[\s\S]*\]", blob)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        if action not in {"fill", "click", "upload"}:
            continue
        try:
            ace_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        entry: dict[str, Any] = {"action": action, "id": ace_id}
        if action == "fill":
            entry["text"] = str(item.get("text") or "")
        actions.append(entry)
    return actions[:40]


def execute_semantic_actions(page, actions: list[dict[str, Any]], resume_path: str) -> bool:
    """Execute planned actions against data-ace-id nodes. Returns True if any ran."""
    if not actions:
        return False
    ran = False
    pdf = Path(resume_path)
    for step in actions:
        ace_id = step.get("id")
        action = step.get("action")
        loc = page.locator(f'[data-ace-id="{ace_id}"]').first
        try:
            if loc.count() == 0:
                continue
            if action == "fill":
                loc.fill(str(step.get("text") or ""), timeout=5000)
                ran = True
            elif action == "click":
                loc.click(timeout=5000)
                ran = True
                page.wait_for_timeout(300)
            elif action == "upload" and pdf.is_file():
                from src.playbooks._common import force_upload_resume

                if force_upload_resume(page, pdf):
                    ran = True
                    continue
                # Prefer a nearby file input if the id is a button
                tag = ""
                try:
                    tag = (loc.evaluate("el => el.tagName") or "").lower()
                except Exception:
                    pass
                if tag == "input":
                    loc.set_input_files(str(pdf), timeout=10_000)
                else:
                    page.locator('input[type="file"]').first.set_input_files(
                        str(pdf), timeout=10_000
                    )
                ran = True
        except Exception:
            logger.debug("semantic action failed action=%s id=%s", action, ace_id, exc_info=True)
            continue
    return ran


def page_blocked(page) -> bool:
    try:
        body = page.inner_text("body", timeout=2000) or ""
    except Exception:
        body = ""
    try:
        title = page.title() or ""
    except Exception:
        title = ""
    return looks_blocked(f"{title}\n{body}")


def _semantic_fill(page, job_url: str, pdf: Path, answers: dict) -> bool:
    from src.playbooks._common import force_upload_resume, goto_dom, success_text

    goto_dom(page, job_url)
    if page_blocked(page):
        raise RuntimeError("blocked: captcha/cloudflare on semantic apply")
    # Prefer fresh disk upload before planning clicks that might hit library cards.
    if pdf.is_file():
        force_upload_resume(page, pdf)
    elements = extract_interactive(page)
    if len(elements) < 2:
        logger.info("semantic DOM: too few elements (%s) — skip", len(elements))
        return False
    actions = plan_semantic_actions(elements, answers=answers, resume_path=str(pdf))
    if not actions:
        logger.info("semantic DOM: empty plan — skip")
        return False
    execute_semantic_actions(page, actions, str(pdf))
    elements2 = extract_interactive(page)
    if elements2:
        actions2 = plan_semantic_actions(
            elements2, answers=answers, resume_path=str(pdf)
        )
        if actions2:
            execute_semantic_actions(page, actions2, str(pdf))
    page.wait_for_timeout(800)
    if page_blocked(page):
        raise RuntimeError("blocked: captcha after semantic actions")
    try:
        body = page.inner_text("body") or ""
    except Exception:
        body = ""
    if success_text(body):
        logger.info("semantic DOM apply succeeded for %s", job_url[:160])
        return True
    logger.info("semantic DOM apply did not confirm success — fall through")
    return False


def try_semantic_apply(
    job_url: str,
    pdf_path: Path | str,
    master_yaml_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
    page=None,
) -> bool:
    """Groq-first DOM apply. Pass ``page`` to reuse warm Chromium (no launch)."""
    if not env_truthy("ACE_SEMANTIC_DOM", "1"):
        return False

    from src.playbooks._common import close_context_safe, launch_persistent_context

    pdf = Path(pdf_path).expanduser().resolve()
    master = Path(master_yaml_path).expanduser().resolve()
    if not pdf.is_file() or not master.is_file():
        return False

    answers = _answers_dict(master)

    if page is not None:
        try:
            return _semantic_fill(page, job_url, pdf, answers)
        except Exception:
            logger.warning(
                "semantic DOM (warm) aborted for %s", job_url[:160], exc_info=True
            )
            return False

    from playwright.sync_api import sync_playwright

    manager = ProxySessionManager()

    def _once() -> bool:
        context = None
        try:
            with sync_playwright() as p:
                context, pg = launch_persistent_context(
                    p,
                    user_data_dir=user_data_dir,
                    proxy=manager.playwright_proxy(),
                )
                return _semantic_fill(pg, job_url, pdf, answers)
        finally:
            close_context_safe(context, user_data_dir=user_data_dir)

    try:
        return bool(
            manager.with_retries(
                _once,
                max_attempts=3 if manager.available() else 1,
                is_block=lambda exc: looks_blocked(exc),
            )
        )
    except Exception:
        logger.warning("semantic DOM apply aborted for %s", job_url[:160], exc_info=True)
        return False
