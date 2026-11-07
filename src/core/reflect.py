"""End-of-day reflection: apply failures + rejection-based resume retrain."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.core.llm_router import invoke_with_failover
from src.core.llm_text import message_text
from src.core.memory import recent_failures, recent_rejections
from src.core.settings import get_settings

logger = logging.getLogger(__name__)
_SEED_LESSONS = Path(__file__).resolve().parent / "lessons_learned.md"

_REFLECT_PROMPT = """You are an AI Job Agent analyzing your daily failures.
Look at these errors (e.g., DOM parsing failures, missing resume skills, CAPTCHAs).
Identify patterns. Write 2-3 new, strict operating rules to prevent these errors tomorrow.

Rules must be:
- Concrete and actionable (portal quirks, form fields, keyword mapping, abort conditions)
- Permanent (written so a future run can follow them without today's context)
- Non-contradictory with: never invent experience; verify tailored resume upload before submit; abort on CAPTCHA

Output ONLY the new rules as a Markdown numbered list. No preamble.
"""

_REJECTION_PROMPT = """You are an AI Job Agent evaluating why these resumes resulted in rejections.
Compare each Job Description to the Tailored Resume YAML.
Did we miss hidden keywords? Was the tone too generic? Did we fail to highlight SDE or ML skills properly?
Write 3 strict, new resume tailoring rules to fix these gaps.

Constraints:
- Never invent employers, titles, or metrics absent from the fact bank
- Prefer XYZ bullets and exact JD phrasing when truthful
- Output ONLY a Markdown numbered list of rules. No preamble.
"""


def lessons_path() -> Path:
    return get_settings().lessons_learned


def _ensure_lessons_file() -> Path:
    path = lessons_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        if _SEED_LESSONS.is_file():
            shutil.copyfile(_SEED_LESSONS, path)
        else:
            path.write_text(
                "# Lessons Learned (agent self-training)\n\n",
                encoding="utf-8",
            )
    return path


def load_lessons() -> str:
    """Return lessons_learned.md contents (seeds data/ from core/ on first use)."""
    path = _ensure_lessons_file()
    return path.read_text(encoding="utf-8").strip()


def _append_rules(title: str, rules: str, incident_count: int) -> None:
    path = _ensure_lessons_file()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    section = f"\n\n## {title} {stamp} ({incident_count} incidents)\n\n{rules}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(section)
    logger.info("Appended %s rules to %s", title, path)


def _format_failures(rows: list[dict]) -> str:
    blocks: list[str] = []
    for i, row in enumerate(rows, start=1):
        blocks.append(
            f"### Failure {i}\n"
            f"- company: {row.get('company')}\n"
            f"- track: {row.get('track')}\n"
            f"- status: {row.get('status')}\n"
            f"- url: {row.get('job_url')}\n"
            f"- error_log:\n{(row.get('error_log') or '(empty)')[:2000]}\n"
        )
    return "\n".join(blocks)


def _format_rejections(rows: list[dict]) -> str:
    blocks: list[str] = []
    for i, row in enumerate(rows, start=1):
        blocks.append(
            f"### Rejection {i}\n"
            f"- company: {row.get('company')}\n"
            f"- track: {row.get('track')}\n"
            f"- outcome_reason: {(row.get('outcome_reason') or '')[:500]}\n"
            f"- JD:\n{(row.get('jd_text') or '(missing)')[:3500]}\n"
            f"- Tailored YAML snapshot:\n"
            f"{(row.get('tailored_yaml_snapshot') or '(missing)')[:4500]}\n"
        )
    return "\n".join(blocks)


def prune_soft_preferences() -> str:
    """Forgetting curve: archive SOFT prefs older than 30d without re-affirm."""
    from src.core.preferences import prune_decayed_preferences

    return prune_decayed_preferences()


def daily_checkout_and_train(hours: int = 24) -> str:
    """Mine apply-layer failures → append operating rules."""
    # Dynamic Preference Engine — decay stale SOFT preferences first.
    try:
        prune_msg = prune_soft_preferences()
        logger.info("Preference prune: %s", prune_msg)
    except Exception:
        logger.exception("Preference prune failed — continuing reflection")

    rows = recent_failures(hours=hours)
    if not rows:
        msg = f"No FAILED/MANUAL_ACTION rows in the last {hours}h — skipping apply reflection."
        logger.info(msg)
        return msg

    prompt = (
        f"{_REFLECT_PROMPT}\n\n"
        f"## Failures (last {hours}h, n={len(rows)})\n{_format_failures(rows)}"
    )
    logger.info("Reflecting on %s apply failure(s)", len(rows))
    response = invoke_with_failover("reflect", prompt, temperature=0.2)
    rules = message_text(getattr(response, "content", response)).strip()
    if not rules:
        msg = "Apply reflection LLM returned empty rules — nothing appended."
        logger.warning(msg)
        return msg
    _append_rules("Apply Reflection", rules, len(rows))
    return rules


def analyze_rejections_and_retrain(hours: int = 48) -> str:
    """Mine REJECTED outcomes + JD/resume snapshots → append tailoring rules."""
    rows = recent_rejections(hours=hours)
    usable = [
        r
        for r in rows
        if (r.get("jd_text") or "").strip()
        and (r.get("tailored_yaml_snapshot") or "").strip()
    ]
    if not usable:
        msg = (
            f"No REJECTED rows with JD+YAML snapshots in the last {hours}h — "
            "skipping rejection retrain."
        )
        logger.info(msg)
        return msg

    prompt = (
        f"{_REJECTION_PROMPT}\n\n"
        f"## Rejections (last {hours}h, n={len(usable)})\n{_format_rejections(usable)}"
    )
    logger.info("Retraining from %s rejection(s)", len(usable))
    response = invoke_with_failover("reflect", prompt, temperature=0.2)
    rules = message_text(getattr(response, "content", response)).strip()
    if not rules:
        msg = "Rejection retrain LLM returned empty rules — nothing appended."
        logger.warning(msg)
        return msg
    _append_rules("Rejection Retrain", rules, len(usable))
    return rules
