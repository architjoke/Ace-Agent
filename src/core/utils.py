"""Small shared helpers — keep this file intentionally thin."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, TypeVar

# Re-export Zero-Trust helpers for convenient imports.
from src.core.security import (  # noqa: F401
    redact_secrets,
    sanitize_untrusted_input,
    validate_safe_path,
    wrap_untrusted,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_HDR = re.compile(r"^(?:#{1,3}\s*)?(?P<body>.+)\s*$", re.I)
_STRIP_EEO = re.compile(
    r"^(?:equal\s+employment\s+opportunity(?:\s*\(eeo\))?|eeo\s+statement|"
    r"eeo|affirmative\s+action|"
    r"diversity(?:\s*[,&]\s*|\s+)equity(?:\s*[,&]\s*|\s+)inclusion|"
    r"our\s+commitment\s+to\s+diversity)\b",
    re.I,
)
_STRIP_BENEFITS = re.compile(
    r"^(?:benefits(?:\s+and\s+perks)?|perks(?:\s+and\s+benefits)?|"
    r"what\s+we\s+offer|compensation\s+and\s+benefits|employee\s+benefits)$",
    re.I,
)
_STRIP_PHYS = re.compile(
    r"^(?:physical\s+requirements|physical\s+demands|working\s+conditions|"
    r"ada\s+accommodation|reasonable\s+accommodation)\b",
    re.I,
)
_STRIP_PRIV = re.compile(
    r"^(?:privacy\s+notice|gdpr|ccpa|candidate\s+privacy)\b",
    re.I,
)
_KEEP_HDR = re.compile(
    r"^(?:requirements|qualifications|responsibilities|what\s+you.?ll\s+do|"
    r"must[- ]have|nice[- ]to[- ]have|tech(?:nical)?\s+skills|experience|"
    r"about\s+the\s+role|job\s+description)\b",
    re.I,
)
_SALARY_NEEDLE = re.compile(
    r"(?:base\s+salary|\$\d|\blpa\b|\bctc\b|compensation\s+range)",
    re.I,
)
def _section_kind(line: str) -> str | None:
    """Return strip kind, 'keep', or None if not a real section header."""
    s = line.strip()
    if not s or len(s) > 90:
        return None
    m = _HDR.match(s)
    if not m:
        return None
    body = m.group("body").strip().strip("*").strip().rstrip(":").strip()
    if _KEEP_HDR.match(body):
        return "keep"
    if _STRIP_EEO.match(body):
        return "eeo"
    if _STRIP_BENEFITS.match(body):
        return "benefits"
    if _STRIP_PHYS.match(body):
        return "phys"
    if _STRIP_PRIV.match(body):
        return "priv"
    if s.startswith("#") or (s.endswith(":") and len(s) <= 60):
        return "other"
    return None


def minify_jd(text: str) -> str:
    """Strip EEO/benefits/physical/privacy boilerplate via strict headers only."""
    raw = (text or "").strip()
    if len(raw) < 80:
        return raw
    lines = raw.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        kind = _section_kind(lines[i])
        if kind not in {"eeo", "benefits", "phys", "priv"}:
            out.append(lines[i])
            i += 1
            continue
        peek = [lines[j].strip() for j in range(i + 1, min(i + 4, n))]
        if kind == "benefits" and any(_SALARY_NEEDLE.search(p) for p in peek if p):
            out.append(lines[i])
            i += 1
            continue
        i += 1
        while i < n:
            nxt_kind = _section_kind(lines[i])
            if nxt_kind is not None:
                break
            i += 1
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    # Safety: never drop skill/requirements headers that existed in the source.
    had_keep = any(_section_kind(l) == "keep" for l in lines)
    keep_ok = any(_section_kind(l) == "keep" for l in cleaned.splitlines())
    if had_keep and not keep_ok:
        return raw
    if len(cleaned) < 0.15 * len(raw):
        return raw
    return cleaned or raw


def env_truthy(name: str, default: str = "1") -> bool:
    """True unless env is 0/false/no/off."""
    return (os.environ.get(name) or default).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def api_request_with_retry(
    fn: Callable[[], T],
    *,
    label: str = "api",
    attempts: int = 1,
) -> T:
    """Run a single API callable; optional micro-retries for transient blips.

    Provider hopping stays in llm_router.invoke_with_failover — this only
    wraps one provider invoke so call sites share one retry path.
    """
    last: BaseException | None = None
    for i in range(max(1, int(attempts))):
        try:
            return fn()
        except BaseException as exc:
            last = exc
            if i + 1 >= max(1, int(attempts)):
                break
            logger.debug("%s attempt %s failed: %s", label, i + 1, exc)
    assert last is not None
    raise last


def fit_reject(rule_broken: str, reason: str) -> dict[str, Any]:
    """Build a failed FitResult-shaped dict (fit_score=0, passed=False)."""
    return {
        "fit_score": 0,
        "reason": reason,
        "passed": False,
        "rule_broken": rule_broken,
        "matched_skills": [],
        "missing_must_haves": [],
        "recommended_track": "",
        "company_provided_salary": False,
        "salary_min": 0.0,
        "salary_max": 0.0,
        "salary_format": "ANNUAL",
        "includes_equity_or_bonus": False,
        "jd_salary_format": "ANNUAL",
    }
