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
from src.core.prompts import EVALUATE_FIT, EVALUATE_FIT_JSON_HINT
from src.core.settings import get_settings
from src.core.utils import fit_reject

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
    """Structured LLM judgment of candidate↔JD fit + compensation extract."""

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
    company_provided_salary: bool = Field(
        default=False,
        description="True if the JD states any salary/CTC/stipend figure.",
    )
    salary_min: float = Field(
        default=0.0,
        description="Lower bound as written (INR). Use lakhs as absolute if JD says LPA.",
    )
    salary_max: float = Field(
        default=0.0,
        description="Upper bound as written (INR). Same units as salary_min.",
    )
    salary_format: str = Field(
        default="ANNUAL",
        description="ANNUAL | MONTHLY | STIPEND as stated in the JD.",
    )
    includes_equity_or_bonus: bool = Field(
        default=False,
        description="True if equity, RSUs, variable pay, or joining bonus is mentioned.",
    )


class FitResult(TypedDict):
    fit_score: int
    reason: str
    passed: bool
    rule_broken: str | None
    matched_skills: list[str]
    missing_must_haves: list[str]
    recommended_track: str
    company_provided_salary: bool
    salary_min: float
    salary_max: float
    salary_format: str
    includes_equity_or_bonus: bool
    jd_salary_format: str


def _reject(rule: str, reason: str) -> FitResult:
    """Log + return a hard-shield FitResult."""
    logger.info("Hard shield rejected JD — %s", reason[:200])
    return fit_reject(rule, reason)  # type: ignore[return-value]


def _hard_config_reject(jd_text: str) -> FitResult | None:
    """Pure-Python shield — 0 tokens. Drops doomed JDs before any LLM call."""
    jd = (jd_text or "").lower()
    if not jd.strip():
        return _reject("empty_jd", "RULE BROKEN: empty_jd. Job description missing.")

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

    # Scam shields first — clearer rule_broken codes than generic exclude_terms
    if re.search(
        r"whatsapp\s*(group|community|channel|link)|"
        r"join\s+(our\s+)?(whatsapp|wa\.me)|"
        r"wa\.me/|"
        r"chat\.whatsapp\.com",
        jd,
        flags=re.IGNORECASE,
    ):
        return _reject(
            "scam_whatsapp",
            "RULE BROKEN: scam_whatsapp. "
            "Listing funnels to WhatsApp group/community — treat as scam.",
        )

    training_placement = bool(
        re.search(
            r"training\s*(and|&|/)?\s*placement|placement\s*(and|&|/)?\s*training|"
            r"training\s+cum\s+placement|pay\s+after\s+placement",
            jd,
            flags=re.IGNORECASE,
        )
    )
    asks_money = bool(
        re.search(
            r"\b(registration\s+fee|course\s+fee|training\s+fee|program\s+fee|"
            r"admission\s+fee|security\s+deposit|pay\s+(a\s+)?fee|fees?\s+applicable|"
            r"₹\s*\d|rs\.?\s*\d|inr\s*\d|pay\s+to\s+(join|apply|register)|"
            r"amount\s+(to\s+be\s+)?paid|one[- ]time\s+(fee|payment))\b",
            jd,
            flags=re.IGNORECASE,
        )
    )
    if training_placement and asks_money:
        return _reject(
            "scam_paid_training_placement",
            "RULE BROKEN: scam_paid_training_placement. "
            "Training/placement internship asking for money — treat as scam.",
        )

    for term in sorted(set(config_terms) | _BUILTIN_EXCLUSIONS):
        if not term or term not in jd:
            continue
        # Don't drop true intern roles solely on a bare "senior" substring
        # inside "senior design project" noise — still drop role-title senior.
        if term == "senior" and "intern" in jd and not re.search(
            r"\bsenior\s+(software|engineer|developer|sde|role|position)\b", jd
        ):
            continue
        return _reject(
            "config_exclude_term",
            f"RULE BROKEN: config_exclude_term. Matched {term!r}.",
        )

    if require_paid and (
        re.search(r"\bunpaid\b", jd)
        or re.search(r"\bvolunteer\b", jd)
        or "pay to apply" in jd
    ):
        return _reject(
            "unpaid_or_pay_to_apply",
            "RULE BROKEN: unpaid_or_pay_to_apply. Deterministic compensation shield.",
        )

    for pattern in _SENIOR_PATTERNS:
        if not re.search(pattern, jd, flags=re.IGNORECASE):
            continue
        if "intern" in jd and "senior" not in jd and re.search(
            r"\b\d\+\s*years?\b", jd
        ):
            continue
        if pattern == r"\bsenior\b" and "intern" in jd:
            continue
        return _reject(
            "senior_experience_required",
            f"RULE BROKEN: senior_experience_required. Matched pattern={pattern!r}.",
        )

    # Dynamic Preference Engine — HARD dealbreakers only (SOFT applied post-LLM).
    try:
        from src.core.preferences import apply_preference_gate

        hard_reason, _, _ = apply_preference_gate(jd_text)
        if hard_reason:
            return _reject("preference_hard", hard_reason)
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
            "company_provided_salary": bool(data.get("company_provided_salary") or False),
            "salary_min": float(data.get("salary_min") or 0),
            "salary_max": float(data.get("salary_max") or 0),
            "salary_format": str(data.get("salary_format") or "ANNUAL"),
            "includes_equity_or_bonus": bool(data.get("includes_equity_or_bonus") or False),
            "jd_salary_format": str(data.get("jd_salary_format") or data.get("salary_format") or "ANNUAL"),
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

    from src.core.prompts import UNTRUSTED_INPUT_GUARDRAIL
    from src.core.security import wrap_untrusted

    prompt = EVALUATE_FIT.format(
        guardrail=UNTRUSTED_INPUT_GUARDRAIL,
        doctrine=ATS_MATCHING_DOCTRINE,
        jd_text=wrap_untrusted("job_description", jd_text),
        fact_bank=fact_bank,
    )

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
                prompt + EVALUATE_FIT_JSON_HINT,
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
        comp_raw = assessment
    else:
        score = int(assessment.fit_score)
        reason = str(assessment.reason).strip()
        rule_broken = assessment.rule_broken
        matched = [str(x) for x in (assessment.matched_skills or [])]
        missing = [str(x) for x in (assessment.missing_must_haves or [])]
        track = str(assessment.recommended_track or "").strip().upper()
        comp_raw = assessment.model_dump()

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
    from src.core.salary_engine import compensation_from_fit

    compensation = compensation_from_fit(comp_raw)
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
        "company_provided_salary": bool(compensation.get("company_provided_salary")),
        "salary_min": float(compensation.get("salary_min") or 0),
        "salary_max": float(compensation.get("salary_max") or 0),
        "salary_format": str(compensation.get("salary_format") or "ANNUAL"),
        "includes_equity_or_bonus": bool(compensation.get("includes_equity_or_bonus")),
        "jd_salary_format": str(
            compensation.get("jd_salary_format")
            or compensation.get("salary_format")
            or "ANNUAL"
        ),
    }
