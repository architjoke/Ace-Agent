"""General form field kinds + label classification (no personal facts)."""

from __future__ import annotations

import re
from enum import Enum


class FieldKind(str, Enum):
    IDENTITY_FULL_NAME = "identity_full_name"
    IDENTITY_FIRST_NAME = "identity_first_name"
    IDENTITY_LAST_NAME = "identity_last_name"
    IDENTITY_EMAIL = "identity_email"
    IDENTITY_PHONE = "identity_phone"
    IDENTITY_CITY = "identity_city"
    IDENTITY_LINKEDIN = "identity_linkedin"
    IDENTITY_GITHUB = "identity_github"
    IDENTITY_PORTFOLIO = "identity_portfolio"
    COMP_CURRENT_CTC = "compensation_current_ctc"
    COMP_EXPECTED_CTC = "compensation_expected_ctc"
    AVAIL_NOTICE_DAYS = "availability_notice_days"
    EXP_YEARS = "experience_years"
    EXP_MONTHS = "experience_months"
    EDU_START_MONTH = "education_start_month"
    EDU_START_YEAR = "education_start_year"
    EDU_END_MONTH = "education_end_month"
    EDU_END_YEAR = "education_end_year"
    EDU_ATTENDING = "education_currently_attending"
    EDU_GRAD_YEAR = "education_graduation_year"
    YES_NO_ELIGIBILITY = "yes_no_eligibility"
    SKILLS_TECH = "skills_tech"
    FREEFORM = "freeform"
    UPLOAD_RESUME = "upload_resume"
    UNKNOWN = "unknown"


# Canonical profile keys for each kind (read from application_answers / profile dict).
KIND_PROFILE_KEYS: dict[FieldKind, tuple[str, ...]] = {
    FieldKind.IDENTITY_FULL_NAME: ("full_name",),
    FieldKind.IDENTITY_FIRST_NAME: ("first_name",),
    FieldKind.IDENTITY_LAST_NAME: ("last_name",),
    FieldKind.IDENTITY_EMAIL: ("email",),
    FieldKind.IDENTITY_PHONE: ("phone", "phone_national", "phone_display"),
    FieldKind.IDENTITY_CITY: ("city", "full_location"),
    FieldKind.IDENTITY_LINKEDIN: ("linkedin", "linkedin_url"),
    FieldKind.IDENTITY_GITHUB: ("github", "github_url"),
    FieldKind.IDENTITY_PORTFOLIO: ("portfolio_url", "website_url"),
    FieldKind.COMP_CURRENT_CTC: ("current_ctc_inr", "current_ctc", "current_ctc_digits"),
    FieldKind.COMP_EXPECTED_CTC: (
        "expected_ctc_inr",
        "expected_ctc",
        "expected_ctc_digits",
        "salary_expectation",
    ),
    FieldKind.AVAIL_NOTICE_DAYS: ("notice_period_days", "notice_period_days_numeric"),
    FieldKind.EXP_YEARS: ("years_of_experience",),
    FieldKind.EXP_MONTHS: ("additional_months_of_experience",),
    FieldKind.EDU_START_MONTH: ("university_start_month",),
    FieldKind.EDU_START_YEAR: ("university_start_year",),
    FieldKind.EDU_END_MONTH: ("university_end_month",),
    FieldKind.EDU_END_YEAR: ("university_end_year",),
    FieldKind.EDU_ATTENDING: ("university_currently_attending",),
    FieldKind.EDU_GRAD_YEAR: ("graduation_year",),
    FieldKind.FREEFORM: ("why_this_role", "additional_information"),
}

# Label → kind (general patterns — not person-specific).
_KIND_RULES: tuple[tuple[re.Pattern[str], FieldKind], ...] = (
    (re.compile(r"upload.*(resume|cv)|resume.*upload|attach.*(resume|cv)", re.I), FieldKind.UPLOAD_RESUME),
    (re.compile(r"first\s*name", re.I), FieldKind.IDENTITY_FIRST_NAME),
    (re.compile(r"last\s*name|surname|family\s*name", re.I), FieldKind.IDENTITY_LAST_NAME),
    (re.compile(r"full\s*name|^name$", re.I), FieldKind.IDENTITY_FULL_NAME),
    (re.compile(r"e-?mail", re.I), FieldKind.IDENTITY_EMAIL),
    (re.compile(r"phone|mobile|cell", re.I), FieldKind.IDENTITY_PHONE),
    (re.compile(r"enter\s*city|city|location", re.I), FieldKind.IDENTITY_CITY),
    (re.compile(r"linkedin", re.I), FieldKind.IDENTITY_LINKEDIN),
    (re.compile(r"github", re.I), FieldKind.IDENTITY_GITHUB),
    (re.compile(r"portfolio|personal\s*website|figma|online\s*portfolio", re.I), FieldKind.IDENTITY_PORTFOLIO),
    (re.compile(r"current\s*(ctc|salary|compensation)", re.I), FieldKind.COMP_CURRENT_CTC),
    (re.compile(r"expected\s*(ctc|salary|compensation)|salary\s*expectation", re.I), FieldKind.COMP_EXPECTED_CTC),
    (re.compile(r"notice\s*period", re.I), FieldKind.AVAIL_NOTICE_DAYS),
    (re.compile(r"year.*experience|how many years", re.I), FieldKind.EXP_YEARS),
    (re.compile(r"month.*experience|additional month|how many month", re.I), FieldKind.EXP_MONTHS),
    # Skills before yes/no — "have you worked with X technologies" is free text.
    (
        re.compile(
            r"technolog|AI\s*/?\s*LLM|tools have you worked|skills? you (have )?used|"
            r"list technolo|tech\s*stack|programming\s*language",
            re.I,
        ),
        FieldKind.SKILLS_TECH,
    ),
    # Yes/No before education date patterns — "2 months" must not match end-month.
    (
        re.compile(
            r"\b(are you|do you|have you|can you|will you|would you|open to|"
            r"available to|willing to|commit|internship|hours?\s*a?\s*day|"
            r"prompt engineering|built any|worked on)\b",
            re.I,
        ),
        FieldKind.YES_NO_ELIGIBILITY,
    ),
    (re.compile(r"dates attended.*from.*month|(university|college|school).*(start|from).*month|^start\s*month$|^from\s*month$", re.I), FieldKind.EDU_START_MONTH),
    (re.compile(r"dates attended.*from.*year|(university|college|school).*(start|from).*year|^start\s*year$|^from\s*year$", re.I), FieldKind.EDU_START_YEAR),
    (re.compile(r"dates attended.*to.*month|(university|college|school).*(end|to|grad).*month|^end\s*month$|^to\s*month$", re.I), FieldKind.EDU_END_MONTH),
    (re.compile(r"dates attended.*to.*year|(university|college|school).*(end|to|grad).*year|^end\s*year$|^to\s*year$|graduation\s*year", re.I), FieldKind.EDU_END_YEAR),
    (re.compile(r"currently\s*attend|still\s*(enrolled|studying)", re.I), FieldKind.EDU_ATTENDING),
    (re.compile(r"why (do )?you|cover letter|additional information|anything else", re.I), FieldKind.FREEFORM),
)

_SKIP = re.compile(
    r"password|otp|captcha|browse|submit|next|continue|review|cancel|close|dismiss|back$",
    re.I,
)


def classify_label(label: str) -> FieldKind:
    """Map a visible form label to a general field kind."""
    blob = (label or "").strip()
    if not blob or _SKIP.search(blob):
        return FieldKind.UNKNOWN
    for pattern, kind in _KIND_RULES:
        if pattern.search(blob):
            return kind
    return FieldKind.UNKNOWN


def kind_to_profile_keys(kind: FieldKind) -> tuple[str, ...]:
    return KIND_PROFILE_KEYS.get(kind, ())
