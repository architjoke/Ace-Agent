"""Normalize application_answers for ATS / Easy Apply field types.

Narrative prefs like "7–14 days" or "₹6,50,000 per annum" must NEVER be typed
into numeric boxes. This module derives single-value form fields and labels
which keys are safe for constrained inputs.
"""

from __future__ import annotations

import re
from typing import Any

# Keys that must be a single integer (or empty) when sent to forms.
_NUMERIC_KEYS = frozenset(
    {
        "notice_period_days",
        "notice_period_days_numeric",
        "current_ctc",
        "current_ctc_inr",
        "current_ctc_digits",
        "expected_ctc",
        "expected_ctc_inr",
        "expected_ctc_digits",
        "salary_expectation",
        "expected_salary",
        "years_of_experience",
        "additional_months_of_experience",
        "graduation_year",
        "cgpa_numeric",
        "phone_national",
        "country_calling_code",
        "pincode",
        "postal_code",
        "zip_code",
    }
)

# Human prose — useful context, but never paste into digit-only / short boxes.
_NARRATIVE_KEYS = frozenset(
    {
        "notice_period",
        "earliest_join_date",
        "available_to_join",
        "joining_flexibility",
        "years_of_experience_label",
        "experience_level",
        "expected_ctc_display",
        "expected_ctc_per_annum",
        "salary_rationale",
        "why_this_role",
        "additional_information",
        "work_authorization",
        "preferred_locations",
        "preferred_work_mode",
    }
)


def first_int(raw: Any, *, default: str = "0") -> str:
    """Extract the first integer from ranges like '7-14', '7 to 14 days', '0-1'."""
    text = str(raw or "").strip()
    if not text:
        return default
    # Prefer standalone digit string.
    if re.fullmatch(r"\d+", text):
        return text
    m = re.search(r"\d+", text)
    return m.group(0) if m else default


def digits_only(raw: Any, *, default: str = "0") -> str:
    """Strip currency / commas / words → plain digits (first amount if a band)."""
    text = str(raw or "").strip()
    if not text:
        return default
    if re.fullmatch(r"\d+", text):
        return text
    # "₹6,50,000 per annum" / "6,00,000 – 7,00,000" → first INR-like group.
    amounts = re.findall(r"\d[\d,]*", text)
    for amt in amounts:
        digits = re.sub(r"[^\d]", "", amt)
        if digits:
            return digits
    digits = re.sub(r"[^\d]", "", text)
    return digits if digits else default


def normalize_phone_fields(answers: dict[str, Any] | None) -> dict[str, Any]:
    """Shared phone contract: national + calling code → display / e164 / phone.

    Invariant: never invent a trunk-0 national; never hardcode ``+91`` when
    ``country_calling_code`` / ``phone_country_code`` is present.
    """
    out: dict[str, Any] = dict(answers or {})
    cc = digits_only(
        out.get("country_calling_code") or out.get("phone_country_code"),
        default="",
    )
    if cc == "0":
        cc = ""

    national = digits_only(out.get("phone_national"), default="")
    if national == "0":
        national = ""
    # Trunk prefix 0 on an otherwise 10-digit mobile (e.g. 08572855980).
    if national.startswith("0") and len(national) == 11:
        national = national[1:]
    if cc and national.startswith(cc) and len(national) == len(cc) + 10:
        national = national[len(cc) :]

    raw_digits = digits_only(
        out.get("phone") or out.get("phone_e164") or out.get("phone_display"),
        default="",
    )
    if raw_digits == "0":
        raw_digits = ""

    if not national and raw_digits:
        if raw_digits.startswith("0") and len(raw_digits) == 11:
            national = raw_digits[1:]
        elif cc and raw_digits.startswith(cc) and len(raw_digits) > len(cc):
            national = raw_digits[len(cc) :]
        elif len(raw_digits) == 10:
            national = raw_digits
        elif len(raw_digits) > 10:
            # +CCNNNNNNNNNN → last 10 as national when CC unknown/ambiguous.
            national = raw_digits[-10:]
            if not cc:
                cc = raw_digits[:-10]

    if cc and national.startswith(cc) and len(national) == len(cc) + 10:
        national = national[len(cc) :]

    if not cc:
        e164_src = str(out.get("phone_e164") or out.get("phone") or "").strip()
        if e164_src.startswith("+"):
            dig = digits_only(e164_src, default="")
            if national and dig.endswith(national) and len(dig) > len(national):
                cc = dig[: len(dig) - len(national)]
            elif len(dig) > 10:
                cc = dig[:-10]
                if not national:
                    national = dig[-10:]
        # Locale default only when we have a clean 10-digit national and no CC.
        if not cc and len(national) == 10:
            cc = "91"

    out["phone_national"] = national
    out["country_calling_code"] = cc
    if cc:
        out["phone_country_code"] = f"+{cc}"
    if national and cc:
        display = f"+{cc} {national}"
        e164 = f"+{cc}{national}"
        out["phone_display"] = display
        out["phone_e164"] = e164
        raw_phone = digits_only(out.get("phone"), default="")
        if not str(out.get("phone") or "").strip() or raw_phone.startswith("0") or (
            raw_phone == national or (raw_phone.endswith(national) if national else False)
        ):
            out["phone"] = display
    return out


def sanitize_application_answers(answers: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy with constrained form fields normalized to single values."""
    out: dict[str, Any] = dict(answers or {})
    # Notice: "7 to 14 days" → days field = 7 (not "7-14").
    out["notice_period_days"] = first_int(
        out.get("notice_period_days_numeric")
        or out.get("notice_period_days")
        or out.get("notice_period"),
        default="7",
    )
    out["notice_period_days_numeric"] = out["notice_period_days"]

    # Experience dropdowns / numeric experience.
    out["years_of_experience"] = first_int(out.get("years_of_experience"), default="0")
    out["additional_months_of_experience"] = first_int(
        out.get("additional_months_of_experience"), default="0"
    )

    # CTC: plain INR digits for Example: 200000 boxes.
    out["current_ctc_inr"] = digits_only(
        out.get("current_ctc_inr") or out.get("current_ctc_digits") or out.get("current_ctc"),
        default="0",
    )
    out["current_ctc_digits"] = out["current_ctc_inr"]
    out["current_ctc"] = out["current_ctc_inr"]

    expected_src = (
        out.get("expected_ctc_inr")
        or out.get("expected_ctc_digits")
        or out.get("salary_annual_inr")
        or out.get("expected_ctc")
        or out.get("salary_expectation")
    )
    out["expected_ctc_inr"] = digits_only(expected_src, default="650000")
    out["expected_ctc_digits"] = out["expected_ctc_inr"]
    out["expected_ctc"] = out["expected_ctc_inr"]
    out["salary_expectation"] = out["expected_ctc_inr"]
    out["expected_salary"] = out["expected_ctc_inr"]

    # CGPA numeric when forms ask for a number.
    cgpa = str(out.get("cgpa") or "")
    m = re.search(r"(\d+(?:\.\d+)?)", cgpa)
    if m:
        out["cgpa_numeric"] = m.group(1)

    out["graduation_year"] = first_int(out.get("graduation_year"), default="2027")
    out.update(normalize_phone_fields(out))
    out["pincode"] = digits_only(out.get("pincode") or out.get("postal_code"), default="")
    out["postal_code"] = out["pincode"]
    out["zip_code"] = out["pincode"]

    # Portfolio must never fall back to GitHub.
    portfolio = str(out.get("portfolio_url") or out.get("website_url") or "").strip()
    if "github.com" in portfolio.lower():
        portfolio = ""
    out["portfolio_url"] = portfolio
    out["website_url"] = str(out.get("website_url") or "").strip()
    if "github.com" in out["website_url"].lower():
        out["website_url"] = ""

    return out


def form_field_rules_block() -> str:
    return (
        "- form_rule: Ranges like '7-14 days' or '0-1 years' are PREFERENCES only — "
        "numeric boxes get a SINGLE integer (notice_period_days, years_of_experience).\n"
        "- form_rule: CTC Example 200000/350000 → type expected_ctc_inr / current_ctc_inr "
        "digits only (no ₹, commas, 'per annum').\n"
        "- form_rule: Portfolio URL → portfolio_url only; never GitHub; leave blank if empty.\n"
        "- form_rule: Prefer *_numeric / *_inr / *_days keys over narrative twins when the "
        "field is short, digit-only, or shows an Example: number.\n"
        "- form_rule: Education month/year dropdowns → university_start_month/year "
        "(August/2024), university_end_month/year (May/2027); mark currently attending."
    )


def answers_prompt_block(answers: dict[str, Any]) -> str:
    """Split constrained vs narrative so the LLM does not paste prose into digit boxes."""
    merged = dict(answers or {})
    learned_block = ""
    try:
        from src.core.learned_answers import learned_qa_prompt_block, merge_into_answers

        merged = merge_into_answers(merged)
        learned_block = learned_qa_prompt_block() or ""
    except Exception:
        pass

    clean = sanitize_application_answers(merged)
    constrained: list[str] = []
    narrative: list[str] = []
    other: list[str] = []
    for key, value in clean.items():
        if value is None or value == "":
            continue
        if str(key).startswith("learned_q:"):
            continue
        line = f"- {key}: {value}"
        if key in _NUMERIC_KEYS or key.endswith("_inr") or key.endswith("_numeric") or key.endswith("_days"):
            constrained.append(line)
        elif key in _NARRATIVE_KEYS:
            narrative.append(line)
        else:
            other.append(line)

    parts = ["## APPLICATION FORM ANSWERS (authoritative)"]
    if constrained:
        parts.append("### Constrained / numeric fields (use these on short digit boxes)")
        parts.extend(constrained)
    if other:
        parts.append("### Standard fields")
        parts.extend(other)
    if narrative:
        parts.append(
            "### Narrative context ONLY (do NOT paste into digit/short Example boxes)"
        )
        parts.extend(narrative)
    parts.append(form_field_rules_block())
    if learned_block:
        parts.append(learned_block)
    return "\n".join(parts)
