"""Shared apply outcome policy: retry vs terminal, suppress rules.

Playbooks and the browser pool must not invent their own success/retry semantics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# Outcomes returned by ``pipeline.apply_job`` / browser workers.
TERMINAL_OK = frozenset({"success"})
# manual_action: Manual_Action package already written (e.g. captcha after human wait).
TERMINAL_SKIP = frozenset({"already_applied", "job_closed", "manual_action"})
STOP_POOL = frozenset({"token_limit_reached"})
RETRYABLE = frozenset(
    {
        "timeout",
        "requires_human",
        "failed",
        "fallback",
    }
)

# Memory statuses that mean "do not try this URL again".
# MANUAL_ACTION: captcha/human package already produced — do not re-hammer the wall.
SUPPRESS_MEMORY_STATUSES = frozenset(
    {
        "SUCCESS",
        "ALREADY_APPLIED",
        "JOB_CLOSED",
        "MANUAL_ACTION",
    }
)

Disposition = Literal["done", "requeue", "stop_pool"]


def max_apply_attempts() -> int:
    raw = (os.environ.get("ACE_MAX_APPLY_ATTEMPTS") or "3").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


# Default constant for imports/tests; runtime uses max_apply_attempts().
MAX_APPLY_ATTEMPTS = 3


def decide(outcome: str, attempt: int, *, max_attempts: int | None = None) -> Disposition:
    """Map an apply outcome + attempt count to queue disposition.

    ``attempt`` is how many times this URL has already been tried *including*
    the attempt that just finished (1-based after first run).
    """
    out = (outcome or "").strip().lower() or "failed"
    limit = int(max_attempts) if max_attempts is not None else max_apply_attempts()
    n = max(0, int(attempt))

    if out in STOP_POOL:
        return "stop_pool"
    if out in TERMINAL_OK or out in TERMINAL_SKIP:
        return "done"
    if out in RETRYABLE:
        if n >= limit:
            return "done"
        return "requeue"
    # Unknown strings: treat like failed (retry until exhausted).
    if n >= limit:
        return "done"
    return "requeue"


def should_suppress_apply(job_url: str, db_path: Path | None = None) -> bool:
    """True when triage must skip this URL (terminal success/skip or retries exhausted)."""
    from src.core.memory import (
        count_failed_apply_attempts,
        has_terminal_apply_status,
    )

    url = (job_url or "").strip()
    if not url:
        return False
    if has_terminal_apply_status(url, db_path=db_path):
        return True
    # Queue attempt_count may exceed memory rows when requeues happen without a log.
    try:
        from src.core.queue import get_attempt_count

        q_attempts = get_attempt_count(url, db_path=db_path)
    except Exception:
        q_attempts = 0
    mem_attempts = count_failed_apply_attempts(url, db_path=db_path)
    attempts = max(int(q_attempts or 0), int(mem_attempts or 0))
    return attempts >= max_apply_attempts()
