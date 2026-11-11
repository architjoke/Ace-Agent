"""Short humanized application pitch for YC / Wellfound / mailto / Glassdoor notes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

PITCH_MARKER = "# ace_pitch: "

_FALLBACK = (
    "Hi — I'm Archit Gupta, a CSE student (CGPA 8.8) focused on Python/JS/TS, "
    "Node, and applied AI. I've shipped internships and projects with LLMs, "
    "backends, and product-minded engineering. I'd love to contribute to {company} "
    "on this role and learn fast with the team."
)


def _fallback_pitch(company: str, role: str = "") -> str:
    co = (company or "your team").strip() or "your team"
    base = _FALLBACK.format(company=co)
    role = (role or "").strip()
    if role:
        return f"{base} The {role} opening looks like a strong match for my stack."
    return base


def generate_pitch(
    *,
    company: str,
    job_title: str = "",
    jd_text: str = "",
    max_words: int = 140,
) -> str:
    """Return a short, humanized pitch (LLM when available, else template)."""
    company = (company or "").strip() or "the team"
    title = (job_title or "").strip()
    jd = re.sub(r"\s+", " ", (jd_text or "").strip())[:1800]
    prompt = (
        "Write a short, natural application note (not a formal cover letter). "
        f"Max {max_words} words. Sound human, specific, and concise. "
        "No emojis, no bullet lists, no 'Dear Hiring Manager'. "
        "Mention 1–2 concrete skills (Python/JS/TS/Node/AI) only if relevant. "
        f"Company: {company}\nRole: {title or 'Software / AI intern'}\n"
        f"JD excerpt: {jd or '(none)'}\n"
        "Return ONLY the note text."
    )
    try:
        from src.core.llm_router import invoke_with_failover

        text = invoke_with_failover(role="draft", prompt=prompt, temperature=0.5)
        if hasattr(text, "content"):
            text = getattr(text, "content")
        if isinstance(text, list):
            text = " ".join(
                str(getattr(p, "text", p) if not isinstance(p, str) else p) for p in text
            )
        body = str(text or "").strip().strip('"').strip()
        # Strip accidental fences / labels
        body = re.sub(r"^```\w*\n?", "", body)
        body = re.sub(r"\n?```$", "", body).strip()
        words = body.split()
        if len(words) > max_words + 20:
            body = " ".join(words[:max_words])
        if len(body) >= 40:
            return body
    except Exception:
        logger.debug("pitch LLM unavailable — using template", exc_info=True)
    return _fallback_pitch(company, title)


def embed_pitch(yaml_snapshot: str, pitch: str | None) -> str:
    """Insert pitch marker after compensation marker (compensation stays line 1)."""
    text = re.sub(
        rf"^{re.escape(PITCH_MARKER)}.*\n?",
        "",
        yaml_snapshot or "",
        count=1,
        flags=re.M,
    )
    # Also strip pitch if it was placed mid-header
    lines = text.splitlines()
    lines = [ln for ln in lines if not ln.startswith(PITCH_MARKER)]
    text = "\n".join(lines)
    note = (pitch or "").strip()
    if not note:
        return text
    blob = json.dumps({"pitch_short": note}, separators=(",", ":"))
    pitch_line = f"{PITCH_MARKER}{blob}"
    if text.startswith("ACE_COMPENSATION_JSON:"):
        first, _, rest = text.partition("\n")
        return f"{first}\n{pitch_line}\n{rest}" if rest else f"{first}\n{pitch_line}\n"
    return f"{pitch_line}\n{text}"


def extract_pitch(yaml_snapshot: str | None) -> str:
    text = yaml_snapshot or ""
    for line in text.splitlines()[:6]:
        if line.startswith(PITCH_MARKER):
            try:
                data = json.loads(line[len(PITCH_MARKER) :])
                if isinstance(data, dict) and data.get("pitch_short"):
                    return str(data["pitch_short"]).strip()
            except (json.JSONDecodeError, TypeError):
                continue
    return ""


def pitch_from_snapshot_or_generate(
    yaml_snapshot: str | None,
    *,
    company: str,
    job_title: str = "",
    jd_text: str = "",
) -> str:
    existing = extract_pitch(yaml_snapshot)
    if existing:
        return existing
    return generate_pitch(company=company, job_title=job_title, jd_text=jd_text)
