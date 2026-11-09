"""Small shared helpers — keep this file intentionally thin."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
