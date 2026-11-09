"""Dynamic compensation negotiation — deterministic INR targeting."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

USER_DEFAULT_MIN = 600_000
USER_DEFAULT_MAX = 700_000
USER_DEFAULT_MEDIAN = 650_000
USER_HARD_FLOOR = 300_000
BASE_SHARE = 0.75
RANGE_PERCENTILE = 0.60

FieldFormat = Literal["ANNUAL", "MONTHLY", "STIPEND", "LAKHS"]
FieldKind = Literal["CTC", "BASE", "UNKNOWN"]
SalaryFormat = Literal["ANNUAL", "MONTHLY", "STIPEND"]

COMP_MARKER = "ACE_COMPENSATION_JSON:"

_EMPTY_COMP: dict[str, Any] = {
    "company_provided_salary": False,
    "salary_min": 0.0,
    "salary_max": 0.0,
    "salary_format": "ANNUAL",
    "includes_equity_or_bonus": False,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce LLM / JSON junk to float without raising."""
    if value is None or value is False:
        return float(default)
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return float(default)
        return float(value)
    text = str(value).strip().replace(",", "").replace("₹", "").replace("Rs", "")
    text = re.sub(r"[^\d.\-]", "", text)
    if not text or text in {".", "-", "-."}:
        return float(default)
    try:
        out = float(text)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _norm_format(raw: Any) -> SalaryFormat:
    s = str(raw or "ANNUAL").strip().upper()
    if s in {"MONTHLY", "MONTH", "PCM", "PER_MONTH"}:
        return "MONTHLY"
    if s in {"STIPEND", "INTERNSHIP"}:
        return "STIPEND"
    return "ANNUAL"


def _norm_field_format(raw: Any) -> FieldFormat:
    s = str(raw or "ANNUAL").strip().upper()
    if s in {"MONTHLY", "MONTH", "PCM", "STIPEND"}:
        return "STIPEND" if "STIPEND" in s else "MONTHLY"
    if s in {"LAKH", "LAKHS", "LPA", "L"}:
        return "LAKHS"
    return "ANNUAL"


def _norm_field_kind(raw: Any) -> FieldKind:
    s = str(raw or "").strip().lower()
    if re.search(r"\bbase\b", s) and not re.search(r"\b(ctc|total|gross)\b", s):
        return "BASE"
    if re.search(r"\b(ctc|total|gross|package)\b", s):
        return "CTC"
    return "UNKNOWN"


def normalize_compensation(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize extraction; amounts stored as annual INR floats (idempotent)."""
    src = raw if isinstance(raw, dict) else {}
    provided = _safe_bool(src.get("company_provided_salary"), False)
    orig_fmt = _norm_format(src.get("salary_format"))
    jd_fmt = _norm_format(src.get("jd_salary_format") or src.get("salary_format"))
    lo = _safe_float(src.get("salary_min"), 0.0)
    hi = _safe_float(src.get("salary_max"), 0.0)
    if hi < lo:
        lo, hi = hi, lo
    # Heuristic: small absolute figures stated as annual → treat as lakhs
    if provided and hi > 0 and hi <= 100 and orig_fmt == "ANNUAL":
        lo, hi = lo * 100_000.0, hi * 100_000.0
    # Only annualize when the *stored* format is still monthly/stipend (not re-run)
    if provided and orig_fmt in {"MONTHLY", "STIPEND"}:
        lo, hi = lo * 12.0, hi * 12.0
    equity = _safe_bool(src.get("includes_equity_or_bonus"), False)
    if not provided or hi <= 0:
        return dict(_EMPTY_COMP)
    return {
        "company_provided_salary": True,
        "salary_min": float(max(0.0, lo)),
        "salary_max": float(max(0.0, hi)),
        "salary_format": "ANNUAL",
        "includes_equity_or_bonus": equity,
        "jd_salary_format": jd_fmt,
    }


def compensation_from_fit(fit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(fit, dict):
        return dict(_EMPTY_COMP)
    keys = (
        "company_provided_salary",
        "salary_min",
        "salary_max",
        "salary_format",
        "includes_equity_or_bonus",
    )
    if not any(k in fit for k in keys):
        return dict(_EMPTY_COMP)
    return normalize_compensation({k: fit.get(k) for k in keys})


def embed_compensation(yaml_snapshot: str, compensation: dict[str, Any] | None) -> str:
    """Prefix snapshot with a one-line JSON marker (no DB migration)."""
    body = re.sub(
        rf"^{re.escape(COMP_MARKER)}.*\n?",
        "",
        yaml_snapshot or "",
        count=1,
        flags=re.M,
    )
    blob = json.dumps(normalize_compensation(compensation), separators=(",", ":"))
    return f"{COMP_MARKER}{blob}\n{body}"


def extract_compensation(yaml_snapshot: str | None) -> dict[str, Any]:
    text = yaml_snapshot or ""
    m = re.match(rf"^{re.escape(COMP_MARKER)}(\{{.*\}})\s*\n?", text)
    if not m:
        return dict(_EMPTY_COMP)
    try:
        data = json.loads(m.group(1))
    except (json.JSONDecodeError, TypeError):
        return dict(_EMPTY_COMP)
    return normalize_compensation(data if isinstance(data, dict) else None)


def strip_compensation_marker(yaml_snapshot: str | None) -> str:
    text = yaml_snapshot or ""
    return re.sub(rf"^{re.escape(COMP_MARKER)}.*\n?", "", text, count=1, flags=re.M)


@dataclass
class SalaryDecision:
    annual_inr: int
    amount: float
    display: str
    field_format: str
    field_kind: str
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SalaryCalculator:
    """Compute the exact figure to type into an ATS salary field."""

    def __init__(
        self,
        *,
        default_median: int = USER_DEFAULT_MEDIAN,
        hard_floor: int = USER_HARD_FLOOR,
        percentile: float = RANGE_PERCENTILE,
        base_share: float = BASE_SHARE,
    ) -> None:
        self.default_median = int(default_median)
        self.hard_floor = int(hard_floor)
        self.percentile = float(percentile)
        self.base_share = float(base_share)

    def target_annual(
        self,
        compensation: dict[str, Any] | None,
        *,
        field_kind: str | FieldKind = "UNKNOWN",
    ) -> tuple[int, str]:
        comp = normalize_compensation(compensation)
        kind = _norm_field_kind(field_kind)

        if not comp.get("company_provided_salary"):
            annual = self.default_median
            why = "no_company_range→default_median"
        else:
            lo = _safe_float(comp.get("salary_min"), 0.0)
            hi = _safe_float(comp.get("salary_max"), 0.0)
            if hi <= 0 and lo <= 0:
                annual = self.default_median
                why = "invalid_range→default_median"
            elif hi < self.hard_floor:
                annual = int(round(hi))
                why = "company_max_below_floor→take_company_max"
            else:
                span = max(0.0, hi - lo)
                annual = int(round(lo + span * self.percentile))
                annual = int(min(hi, max(lo, annual)))
                if annual < self.hard_floor <= hi:
                    annual = int(min(hi, self.hard_floor))
                    why = "p60_then_bump_to_floor_inside_band"
                else:
                    why = "company_range_p60"

        if kind == "BASE" and _safe_bool(comp.get("includes_equity_or_bonus"), False):
            annual = int(round(annual * self.base_share))
            why = f"{why}+base_x{self.base_share}"

        annual = max(0, int(annual))
        return annual, why

    def format_amount(self, annual_inr: int, field_format: str | FieldFormat) -> tuple[float, str]:
        fmt = _norm_field_format(field_format)
        annual = max(0, int(_safe_float(annual_inr, 0)))
        if fmt in {"MONTHLY", "STIPEND"}:
            amount = float(int(round(annual / 12.0)))
            display = str(int(amount))
        elif fmt == "LAKHS":
            lakhs = round(annual / 100_000.0, 1)
            amount = float(lakhs)
            display = f"{lakhs:.1f}".rstrip("0").rstrip(".")
        else:
            amount = float(annual)
            display = _indian_commas(annual)
        return amount, display

    def compute(
        self,
        compensation: dict[str, Any] | None = None,
        *,
        field_format: str = "ANNUAL",
        field_kind: str = "UNKNOWN",
    ) -> SalaryDecision:
        try:
            annual, why = self.target_annual(compensation, field_kind=field_kind)
            amount, display = self.format_amount(annual, field_format)
            return SalaryDecision(
                annual_inr=annual,
                amount=amount,
                display=display,
                field_format=_norm_field_format(field_format),
                field_kind=_norm_field_kind(field_kind),
                rationale=why,
            )
        except Exception as exc:
            logger.exception("SalaryCalculator failed — using default median")
            annual = self.default_median
            amount, display = self.format_amount(annual, field_format)
            return SalaryDecision(
                annual_inr=annual,
                amount=amount,
                display=display,
                field_format=_norm_field_format(field_format),
                field_kind=_norm_field_kind(field_kind),
                rationale=f"error_fallback:{type(exc).__name__}",
            )


def _indian_commas(n: int) -> str:
    n = abs(int(n))
    core = str(n)
    if n < 1000:
        return core
    last3 = core[-3:]
    rest = core[:-3]
    parts: list[str] = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ",".join(parts + [last3])


def classify_salary_field(label: str | None) -> tuple[FieldFormat, FieldKind]:
    """Heuristic (no LLM) for ATS label → format/kind."""
    text = str(label or "").lower()
    kind = _norm_field_kind(text)
    if re.search(r"lakh|lpa|\bl\b", text):
        fmt: FieldFormat = "LAKHS"
    elif re.search(r"month|stipend|pcm", text):
        fmt = "MONTHLY"
    else:
        fmt = "ANNUAL"
    return fmt, kind
