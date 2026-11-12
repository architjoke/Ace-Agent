"""Expected CTC helpers — always express salary as Indian ₹ per annum."""

from __future__ import annotations

import re
from typing import Any

# Default band when JD/company does not publish a figure (entry-level India).
DEFAULT_ANNUAL_MIN_L = 6.0
DEFAULT_ANNUAL_MAX_L = 7.0
DEFAULT_ANNUAL_TEXT = "₹6,00,000 – ₹7,00,000 per annum"


def _lakhs_to_inr(lakhs: float) -> int:
    return int(round(float(lakhs) * 100_000))


def format_inr_pa(amount: int | float) -> str:
    n = int(round(float(amount)))
    # Indian grouping: 6,00,000
    s = f"{n:,}"
    # Convert Western thousands to Indian-ish for 6–7 digit INR
    if n >= 100_000:
        # e.g. 600000 → 6,00,000
        core = str(n)
        last3 = core[-3:]
        rest = core[:-3]
        parts: list[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts + [last3])
    return f"₹{s} per annum"


def format_band_pa(min_l: float, max_l: float) -> str:
    lo = format_inr_pa(_lakhs_to_inr(min_l)).replace(" per annum", "")
    hi = format_inr_pa(_lakhs_to_inr(max_l))
    return f"{lo} – {hi}"


def default_expected_ctc() -> str:
    return DEFAULT_ANNUAL_TEXT


_LPA_RE = re.compile(
    r"(?P<a>\d+(?:\.\d+)?)\s*(?:-|–|to|/)?\s*(?P<b>\d+(?:\.\d+)?)?\s*"
    r"(?:lpa|lakh|lakhs|lac|lacs)\b",
    re.I,
)
_INR_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*(?P<a>[\d,]+(?:\.\d+)?)"
    r"(?:\s*(?:-|–|to)\s*(?:₹|rs\.?|inr)?\s*(?P<b>[\d,]+(?:\.\d+)?))?",
    re.I,
)
_MONTHLY_HINT = re.compile(r"\b(per\s*month|/month|monthly|pcm)\b", re.I)
_ANNUAL_HINT = re.compile(r"\b(per\s*annum|per\s*year|/year|annual|ctc|p\.?a\.?)\b", re.I)


def _parse_number(raw: str) -> float | None:
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def annual_from_text(blob: str | None) -> str | None:
    """Derive an annual ₹ band from JD/company text; None if nothing usable."""
    text = (blob or "").strip()
    if not text:
        return None

    m = _LPA_RE.search(text)
    if m:
        a = float(m.group("a"))
        b = float(m.group("b") or a)
        lo, hi = sorted((a, b))
        # Keep candidate expectation inside/near published band, prefer 6–7 overlap.
        pref_lo = max(lo, DEFAULT_ANNUAL_MIN_L)
        pref_hi = min(hi, DEFAULT_ANNUAL_MAX_L)
        if pref_lo <= pref_hi:
            return format_band_pa(pref_lo, pref_hi)
        # Published band outside 6–7: pick the nearer edge of the posted range.
        mid = (lo + hi) / 2.0
        target = min(max(mid, DEFAULT_ANNUAL_MIN_L), DEFAULT_ANNUAL_MAX_L)
        # If company max < 6L, use company max; if min > 7L, use company min.
        if hi < DEFAULT_ANNUAL_MIN_L:
            return format_band_pa(lo, hi)
        if lo > DEFAULT_ANNUAL_MAX_L:
            return format_inr_pa(_lakhs_to_inr(lo))
        return format_inr_pa(_lakhs_to_inr(target))

    m = _INR_RE.search(text)
    if m:
        a = _parse_number(m.group("a"))
        b = _parse_number(m.group("b") or "") if m.group("b") else a
        if a is None:
            return None
        if b is None:
            b = a
        lo, hi = sorted((a, b))
        monthly = bool(_MONTHLY_HINT.search(text)) and not _ANNUAL_HINT.search(text)
        if monthly or (hi < 100_000 and not _ANNUAL_HINT.search(text)):
            lo, hi = lo * 12, hi * 12
        # Express as lakhs for formatting
        return format_band_pa(lo / 100_000.0, hi / 100_000.0)

    return None


def resolve_expected_ctc(
    *,
    answers: dict[str, Any] | None = None,
    jd_text: str | None = None,
) -> str:
    """Prefer JD-derived annual figure; else stored answers; else 6–7 LPA."""
    derived = annual_from_text(jd_text)
    if derived:
        return derived
    answers = answers if isinstance(answers, dict) else {}
    for key in ("expected_ctc", "salary_expectation", "expected_salary", "ctc"):
        raw = str(answers.get(key) or "").strip()
        if not raw:
            continue
        # Soft norms → default band
        if re.search(r"as per|negotiable|company norms", raw, re.I):
            continue
        parsed = annual_from_text(raw)
        if parsed:
            return parsed
        if re.search(r"per\s*annum|lpa|lakh|₹|inr", raw, re.I):
            return raw
    return default_expected_ctc()


def inject_salary_answers(
    answers: dict[str, Any],
    *,
    jd_text: str | None = None,
    compensation: dict[str, Any] | None = None,
    yaml_snapshot: str | None = None,
) -> dict[str, Any]:
    """Return a shallow copy with annual expected CTC from SalaryCalculator."""
    from src.core.salary_engine import (
        SalaryCalculator,
        extract_compensation,
        normalize_compensation,
    )

    out = dict(answers or {})
    comp = compensation
    if comp is None and yaml_snapshot:
        comp = extract_compensation(yaml_snapshot)
    if not isinstance(comp, dict) or not comp:
        # Legacy JD-text parse when no structured blob was passed.
        if jd_text:
            legacy = resolve_expected_ctc(answers=out, jd_text=jd_text)
            if legacy and legacy != default_expected_ctc():
                digits = re.sub(r"[^\d]", "", legacy.split("–")[0].split("-")[0]) or ""
                if len(digits) >= 5:
                    out["expected_ctc"] = digits
                    out["expected_ctc_inr"] = digits
                    out["salary_expectation"] = digits
                    out["expected_salary"] = digits
                else:
                    out["expected_ctc"] = legacy
                    out["salary_expectation"] = legacy
                    out.setdefault("expected_salary", legacy)
                out["expected_ctc_display"] = legacy
                out.setdefault("expected_ctc_per_annum", legacy)
                out.setdefault("current_ctc_inr", str(out.get("current_ctc_inr") or "0"))
                return out
        comp = {}

    decision = SalaryCalculator().compute(
        normalize_compensation(comp),
        field_format="ANNUAL",
        field_kind="CTC",
    )
    annual = int(decision.annual_inr)
    digits = str(annual)
    ctc = f"₹{decision.display} per annum"
    # LinkedIn/ATS numeric CTC boxes (Example: 350000, often max 20 chars).
    out["expected_ctc_inr"] = digits
    out["expected_ctc_digits"] = digits
    out["salary_annual_inr"] = annual
    out.setdefault("current_ctc_inr", str(out.get("current_ctc_inr") or "0"))
    out.setdefault("current_ctc_digits", out["current_ctc_inr"])
    # Keep a human display form, but prefer digits when the field example is numeric.
    out["expected_ctc_display"] = ctc
    out["expected_ctc"] = digits
    out["salary_expectation"] = digits
    out["expected_salary"] = digits
    out["expected_ctc_per_annum"] = ctc
    out["salary_rationale"] = decision.rationale
    return out
