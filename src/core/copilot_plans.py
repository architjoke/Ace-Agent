"""Server-side one-shot plans for destructive Copilot bulk ops.

The model never invents a trusted company list — it may only replay a plan_id
issued by a prior read tool in this process.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MarkThroughPlan:
    plan_id: str
    resolved: str
    companies: tuple[str, ...]
    created_at: float
    expires_at: float
    digest: str
    consumed: bool = False


_LOCK = threading.Lock()
_PLANS: dict[str, MarkThroughPlan] = {}
DEFAULT_TTL_SEC = 600


def _digest(resolved: str, companies: list[str] | tuple[str, ...]) -> str:
    payload = resolved.strip().lower() + "\0" + "\0".join(companies)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def create_mark_through_plan(
    resolved: str,
    companies: list[str],
    *,
    ttl_sec: int = DEFAULT_TTL_SEC,
) -> MarkThroughPlan:
    if not companies:
        raise ValueError("Cannot create empty mark-through plan.")
    now = time.time()
    plan = MarkThroughPlan(
        plan_id=f"mtp_{secrets.token_hex(8)}",
        resolved=resolved,
        companies=tuple(companies),
        created_at=now,
        expires_at=now + max(30, int(ttl_sec)),
        digest=_digest(resolved, companies),
        consumed=False,
    )
    with _LOCK:
        _purge_locked(now)
        _PLANS[plan.plan_id] = plan
    return plan


def peek_plan(plan_id: str) -> MarkThroughPlan | None:
    with _LOCK:
        _purge_locked(time.time())
        return _PLANS.get(plan_id)


def consume_mark_through_plan(plan_id: str) -> MarkThroughPlan:
    """Return plan and mark it consumed (one-shot). Raises ValueError on miss/expiry."""
    pid = (plan_id or "").strip()
    if not pid:
        raise ValueError("plan_id is required. Call preview_mark_through first.")
    now = time.time()
    with _LOCK:
        _purge_locked(now)
        plan = _PLANS.get(pid)
        if plan is None:
            raise ValueError(
                f"Unknown or expired plan_id {pid!r}. Call preview_mark_through again."
            )
        if plan.consumed:
            raise ValueError(
                f"plan_id {pid!r} was already used. Call preview_mark_through again."
            )
        if now > plan.expires_at:
            _PLANS.pop(pid, None)
            raise ValueError(
                f"plan_id {pid!r} expired. Call preview_mark_through again."
            )
        used = MarkThroughPlan(
            plan_id=plan.plan_id,
            resolved=plan.resolved,
            companies=plan.companies,
            created_at=plan.created_at,
            expires_at=plan.expires_at,
            digest=plan.digest,
            consumed=True,
        )
        _PLANS[pid] = used
        return used


def _purge_locked(now: float) -> None:
    dead = [k for k, p in _PLANS.items() if now > p.expires_at or p.consumed]
    for k in dead:
        # Keep consumed briefly? Drop immediately — one-shot.
        _PLANS.pop(k, None)


def clear_plans_for_tests() -> None:
    with _LOCK:
        _PLANS.clear()


def plan_public_view(plan: MarkThroughPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "resolved": plan.resolved,
        "count": len(plan.companies),
        "digest": plan.digest,
        "expires_in_sec": max(0, int(plan.expires_at - time.time())),
    }
