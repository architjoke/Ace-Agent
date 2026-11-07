"""Pre-tailor fit triage: hard policy filters + skill fit scoring."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field, ValidationError

from src.core.ats_playbook import ATS_MATCHING_DOCTRINE
from src.core.llm_router import invoke_with_failover
from src.core.settings import get_settings

logger = logging.getLogger(__name__)

FIT_PASS_SCORE = 80

# Always-on free exclusions (0 tokens) — unioned with config.yaml exclude_terms.
_BUILTIN_EXCLUSIONS = {
    "unpaid",
    "volunteer",
    "pay to apply",
    "security clearance",
    "us citizen",
    "u.s. citizen",
    "must be a citizen",
    "green card",
}

# Extra hard patterns beyond exclude_terms (deterministic, pre-LLM).
_SENIOR_PATTERNS = (
    r"\b5\+\s*years?\b",
    r"\b3\+\s*years?\b",
    r"\b4\+\s*years?\b",
    r"\b6\+\s*years?\b",
    r"\b7\+\s*years?\b",
    r"\b8\+\s*years?\b",
    r"\b10\+\s*years?\b",
    r"\bsenior\s+(software|engineer|developer|sde)\b",
    r"\bstaff\s+engineer\b",
    r"\bprincipal\s+engineer\b",
    r"\bsenior\b",
)


class FitAssessment(BaseModel):
    """Structured LLM judgment of candidate↔JD fit."""

    fit_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Holistic fit from 0-100. Must be 0 if any hard policy rule is broken."
        ),
    )
    reason: str = Field(
        ...,
        description=(
            "If a hard rule failed, start with 'RULE BROKEN: ...' naming the rule. "
            "Otherwise cite matched skills/projects and key gaps."
        ),
    )
    rule_broken: str | None = Field(
        default=None,
        description=(
            "Which hard rule failed, if any: unpaid_or_pay_to_apply | "
            "relocation_outside_india | senior_experience_required. Null if none."
        ),
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Exact JD skills/tools evidenced in the fact bank.",
    )
    missing_must_haves: list[str] = Field(
        default_factory=list,
        description="Critical JD requirements not supported by the fact bank.",
    )
    recommended_track: str = Field(
        default="",
        description="Best track for this JD among SDE, AI, ML (or empty).",
    )


class FitResult(TypedDict):
    fit_score: int
    reason: str
    passed: bool
    rule_broken: str | None
    matched_skills: list[str]
    missing_must_haves: list[str]
    recommended_track: str


def _hard_config_reject(jd_text: str) -> FitResult | None:
    """Pure-Python shield — 0 tokens. Drops doomed JDs before any LLM call."""
    jd = (jd_text or "").lower()
    if not jd.strip():
        return {
            "fit_score": 0,
            "reason": "RULE BROKEN: empty_jd. Job description missing.",
            "passed": False,
            "rule_broken": "empty_jd",
            "matched_skills": [],
            "missing_must_haves": [],
            "recommended_track": "",
        }

    try:
        settings = get_settings()
        config_terms = [
            str(t).strip().lower()
            for t in (settings.exclude_terms or [])
            if str(t).strip()
        ]
        require_paid = bool(settings.require_paid)
    except Exception:
        config_terms = []
        require_paid = True

    exclude_terms = sorted(set(config_terms) | _BUILTIN_EXCLUSIONS)
    for term in exclude_terms:
        if term and term in jd:
            # Don't drop true intern roles solely on a bare "senior" substring
            # inside "senior design project" noise — still drop role-title senior.
            if term == "senior" and "intern" in jd and not re.search(
                r"\bsenior\s+(software|engineer|developer|sde|role|position)\b", jd
            ):
                continue
            reason = f"RULE BROKEN: config_exclude_term. Matched {term!r}."
            logger.info("Hard shield rejected JD — %s", reason)
            return {
                "fit_score": 0,
                "reason": reason,
                "passed": False,
                "rule_broken": "config_exclude_term",
                "matched_skills": [],
                "missing_must_haves": [],
                "recommended_track": "",
            }

    if require_paid and (
        re.search(r"\bunpaid\b", jd)
        or re.search(r"\bvolunteer\b", jd)
        or "pay to apply" in jd
    ):
        reason = "RULE BROKEN: unpaid_or_pay_to_apply. Deterministic compensation shield."
        logger.info("Hard shield rejected JD — %s", reason)
        return {
            "fit_score": 0,
            "reason": reason,
            "passed": False,
            "rule_broken": "unpaid_or_pay_to_apply",
            "matched_skills": [],
            "missing_must_haves": [],
            "recommended_track": "",
        }

    for pattern in _SENIOR_PATTERNS:
        if re.search(pattern, jd, flags=re.IGNORECASE):
            if "intern" in jd and "senior" not in jd and re.search(
                r"\b\d\+\s*years?\b", jd
            ):
                continue
            if pattern == r"\bsenior\b" and "intern" in jd:
                continue
            reason = (
                f"RULE BROKEN: senior_experience_required. "
                f"Matched pattern={pattern!r}."
            )
            logger.info("Hard shield rejected JD — %s", reason)
            return {
                "fit_score": 0,
                "reason": reason,
                "passed": False,
                "rule_broken": "senior_experience_required",
                "matched_skills": [],
                "missing_must_haves": [],
                "recommended_track": "",
            }

    # Dynamic Preference Engine — HARD dealbreakers only (SOFT applied post-LLM).
    try:
        from src.core.preferences import apply_preference_gate

        hard_reason, _, _ = apply_preference_gate(jd_text)
        if hard_reason:
            logger.info("HARD preference rejected JD — %s", hard_reason[:200])
            return {
                "fit_score": 0,
                "reason": hard_reason,
                "passed": False,
                "rule_broken": "preference_hard",
                "matched_skills": [],
                "missing_must_haves": [],
                "recommended_track": "",
            }
    except Exception:
        logger.exception("Preference HARD gate crashed — continuing triage")

    return None


def _salvage_fit_from_text(raw: Any) -> dict[str, Any]:
    """Parse markdown-fenced or messy JSON into a FitAssessment-shaped dict."""
    from src.core.llm_text import message_text

    text = message_text(getattr(raw, "content", raw) if raw is not None else "")
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty fit triage response")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    blob = fence.group(1).strip() if fence else text
    # First {...} object
    brace = re.search(r"\{[\s\S]*\}", blob)
    if not brace:
        raise ValueError(f"No JSON object in fit triage reply: {text[:240]!r}")
    try:
        data = json.loads(brace.group(0))
    except json.JSONDecodeError:
        # Trailing commas / single quotes — light cleanup
        cleaned = re.sub(r",\s*}", "}", brace.group(0))
        cleaned = re.sub(r",\s*]", "]", cleaned)
        cleaned = cleaned.replace("'", '"')
        data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Fit triage JSON was not an object")
    try:
        return FitAssessment.model_validate(data).model_dump()
    except ValidationError:
        # Best-effort coerce
        return {
            "fit_score": int(data.get("fit_score") or 0),
            "reason": str(data.get("reason") or "salvaged"),
            "rule_broken": data.get("rule_broken"),
            "matched_skills": list(data.get("matched_skills") or []),
            "missing_must_haves": list(data.get("missing_must_haves") or []),
            "recommended_track": str(data.get("recommended_track") or ""),
        }


def evaluate_fit(
    jd_text: str,
    master_yaml: str | Path,
    *,
    min_score: int = FIT_PASS_SCORE,
) -> FitResult:
    """Score one JD against the master fact bank via text LLM (Groq/Ollama).

    Deterministic Python shield runs FIRST (0 tokens). Then LLM skill fit.
    """
    shielded = _hard_config_reject(jd_text)
    if shielded is not None:
        return shielded

    if isinstance(master_yaml, Path) or (
        isinstance(master_yaml, str) and Path(master_yaml).is_file()
    ):
        fact_bank = Path(master_yaml).read_text(encoding="utf-8")
    else:
        fact_bank = str(master_yaml)

    prompt = f"""You are a strict technical recruiter for an India-based CS student
(graduation 2027, seeking PAID internships / entry-level roles only).

{ATS_MATCHING_DOCTRINE}

## HARD RULES — evaluate these FIRST, before skill fit
If ANY rule fails, you MUST return fit_score=0, set rule_broken to the matching
code, and begin reason with "RULE BROKEN: <rule>".

1. unpaid_or_pay_to_apply — Discard if unpaid, volunteer-only, stipend-free
   "experience" roles, or any job that requires the candidate to pay/fees.
2. relocation_outside_india — Discard if the role requires physical relocation
   or onsite attendance outside India. India onsite/hybrid/remote is OK.
   Remote roles that hire and pay people located in India are OK.
   Remote roles restricted to US/EU/other countries (not India) are NOT OK.
3. senior_experience_required — Discard if the role clearly requires senior /
   mid-senior full-time experience of 3+ years (or equivalent senior titles).
   Internships, new-grad, and entry-level with 0–2 years are OK.

Only if ALL hard rules pass, score skill/project fit from 1–100:
- Extract JD must-have tools/languages/methods with exact phrasing.
- matched_skills = JD terms with clear evidence in the fact bank.
- missing_must_haves = critical JD requirements with no evidence.
- recommended_track = SDE, AI, or ML based on JD emphasis.
- High scores require overlap on repeated JD hard skills + relevant projects.
- Do NOT invent experience. A role needing mostly unsupported stack should score <60.

Return structured fields only.

## Job Description
---
{jd_text}
---

## Master Fact Bank (RenderCV YAML)
---
{fact_bank}
---
"""

    try:
        assessment = invoke_with_failover(
            "triage",
            prompt,
            temperature=0,
            structured=FitAssessment,
        )
    except Exception as structured_exc:
        logger.warning(
            "Structured fit triage failed (%s) — retrying free-form JSON salvage",
            structured_exc,
        )
        assessment = _salvage_fit_from_text(
            invoke_with_failover(
                "triage",
                prompt
                + "\n\nReply with ONLY a JSON object with keys: "
                "fit_score, reason, rule_broken, matched_skills, "
                "missing_must_haves, recommended_track.",
                temperature=0,
            )
        )

    if isinstance(assessment, dict):
        score = int(assessment.get("fit_score", 0) or 0)
        reason = str(assessment.get("reason") or "").strip()
        rule_broken = assessment.get("rule_broken")
        matched = [str(x) for x in (assessment.get("matched_skills") or [])]
        missing = [str(x) for x in (assessment.get("missing_must_haves") or [])]
        track = str(assessment.get("recommended_track") or "").strip().upper()
    else:
        score = int(assessment.fit_score)
        reason = str(assessment.reason).strip()
        rule_broken = assessment.rule_broken
        matched = [str(x) for x in (assessment.matched_skills or [])]
        missing = [str(x) for x in (assessment.missing_must_haves or [])]
        track = str(assessment.recommended_track or "").strip().upper()

    if rule_broken:
        score = 0
        if not reason.upper().startswith("RULE BROKEN"):
            reason = f"RULE BROKEN: {rule_broken}. {reason}".strip()

    if track not in {"SDE", "AI", "ML"}:
        track = ""

    score = max(0, min(100, score))

    # Serendipity triage: SOFT preferences penalize but do not hard-reject.
    # A very strong match can still clear the bar after the penalty.
    if not rule_broken:
        try:
            from src.core.preferences import apply_preference_gate

            _, soft_penalty, soft_hits = apply_preference_gate(jd_text)
            if soft_penalty > 0:
                before = score
                score = max(0, score - soft_penalty)
                note = (
                    f" SOFT preference penalty -{soft_penalty} "
                    f"({before}→{score}): " + "; ".join(soft_hits[:3])
                )
                reason = (reason + note).strip()
                logger.info(
                    "SOFT preference penalty -%s (%s→%s) hits=%s",
                    soft_penalty,
                    before,
                    score,
                    soft_hits[:3],
                )
        except Exception:
            logger.exception("SOFT preference gate crashed — continuing")

    passed = score >= min_score and not rule_broken
    logger.info(
        "Fit triage score=%s passed=%s rule_broken=%s track=%s matched=%s — %s",
        score,
        passed,
        rule_broken,
        track or "?",
        matched[:8],
        reason[:240],
    )
    return {
        "fit_score": score,
        "reason": reason,
        "passed": passed,
        "rule_broken": str(rule_broken) if rule_broken else None,
        "matched_skills": matched,
        "missing_must_haves": missing,
        "recommended_track": track,
    }
