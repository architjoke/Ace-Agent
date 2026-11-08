"""Small shared helpers — keep this file intentionally thin."""

from __future__ import annotations

import os
from typing import Any


def env_truthy(name: str, default: str = "1") -> bool:
    """True unless env is 0/false/no/off."""
    return (os.environ.get(name) or default).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def clip(text: str, max_chars: int, *, from_tail: bool = False) -> str:
    """Truncate string to max_chars from head or tail."""
    raw = text or ""
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    if from_tail:
        return raw[-max_chars:]
    return raw[:max_chars]


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
    }
