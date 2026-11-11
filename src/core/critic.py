"""Internal Critic — block unsafe/illogical mutate tool calls before HITL.

Deterministic checks are zero-latency. Optional cheap LLM semantic check
compares user intent vs proposed tool; rejections become ToolMessages so the
main agent self-corrects without bothering the user.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.copilot_plans import peek_plan

logger = logging.getLogger(__name__)

_PREVIEW_INTENT_RE = re.compile(
    r"\b("
    r"preview|show\s+me\s+what|what\s+would\s+you\s+(delete|mark|clean)|"
    r"dry\s*-?\s*run|just\s+(list|show)|don'?t\s+(delete|mark|clean)\s+yet|"
    r"before\s+(deleting|marking|cleaning)"
    r")\b",
    re.IGNORECASE,
)

_AMBIGUOUS_BULK_RE = re.compile(
    r"\b("
    r"get\s+rid\s+of\s+all|delete\s+all|clear\s+all|remove\s+all|"
    r"wipe\s+all|clean\s+out\s+all|purge\s+all"
    r")\b",
    re.IGNORECASE,
)

_DESTRUCTIVE_MUTATE = frozenset(
    {
        "mark_applications_through",
        "cleanup_manual_folder",
        "undo_bulk_user_marks",
    }
)


class CriticVerdict(BaseModel):
    proceed: bool
    reason: str = Field(default="", max_length=240)


def critique_tool_call(
    *,
    tool_name: str,
    args: dict[str, Any],
    user_text: str,
    risk: str,
    allow_llm: bool = True,
) -> CriticVerdict:
    """Return proceed=False to reject before HITL (inject as ToolMessage)."""
    if risk == "read":
        return CriticVerdict(proceed=True, reason="read_ok")

    det = _deterministic_critique(tool_name, args, user_text)
    if not det.proceed:
        return det

    if not allow_llm or not (user_text or "").strip():
        return CriticVerdict(proceed=True, reason="deterministic_ok")

    return _semantic_critique(tool_name, args, user_text, risk)


def _deterministic_critique(
    tool_name: str, args: dict[str, Any], user_text: str
) -> CriticVerdict:
    if tool_name == "mark_applications_through":
        pid = str(args.get("plan_id") or "").strip()
        plan = peek_plan(pid)
        if plan is None:
            return CriticVerdict(
                proceed=False,
                reason=(
                    "Critic blocked mark_applications_through: plan_id is missing, "
                    "expired, or was never issued. Call preview_mark_through first, "
                    "then retry with that plan_id."
                ),
            )
        if plan.consumed:
            return CriticVerdict(
                proceed=False,
                reason=(
                    "Critic blocked: plan_id already consumed. "
                    "Run preview_mark_through again."
                ),
            )

    # User asked for preview/show only — do not hit mutate tools yet.
    if tool_name in _DESTRUCTIVE_MUTATE and _PREVIEW_INTENT_RE.search(user_text or ""):
        return CriticVerdict(
            proceed=False,
            reason=(
                "Critic blocked: user asked for a preview/list only. "
                "Call preview_mark_through (or list_open_applications) first — "
                f"do not run {tool_name} yet."
            ),
        )

    # Ambiguous bulk destroy ("get rid of all the failed jobs") — force clarify/preview.
    if tool_name in _DESTRUCTIVE_MUTATE and _AMBIGUOUS_BULK_RE.search(user_text or ""):
        return CriticVerdict(
            proceed=False,
            reason=(
                "Critic blocked: bulk delete/clear is ambiguous. "
                "Ask what to clear (Manual_Action vs failed apps), then "
                "preview_mark_through or list_open_applications before any mutate."
            ),
        )

    return CriticVerdict(proceed=True, reason="deterministic_ok")


def _semantic_critique(
    tool_name: str,
    args: dict[str, Any],
    user_text: str,
    risk: str,
) -> CriticVerdict:
    prompt = (
        "You are Ace Copilot's safety critic. Decide if this tool call matches "
        "the user's request. Block if the user only wanted a preview/status, or "
        "if the tool is clearly the wrong action. Allow normal apply/mark/"
        "cleanup/config when the user clearly asked for it.\n\n"
        f"User said:\n{(user_text or '')[:600]}\n\n"
        f"Proposed tool: {tool_name} (risk={risk})\n"
        f"Args: {args!r}\n\n"
        "Return proceed=true|false and a short reason."
    )
    try:
        from src.core.llm_router import invoke_with_failover

        out = invoke_with_failover(
            "copilot",
            prompt,
            temperature=0,
            structured=CriticVerdict,
        )
        if isinstance(out, CriticVerdict):
            if not out.proceed and not out.reason:
                return CriticVerdict(
                    proceed=False,
                    reason="Critic blocked: proposed action mismatches user intent.",
                )
            if not out.proceed:
                return CriticVerdict(
                    proceed=False,
                    reason=f"Critic blocked: {out.reason}",
                )
            return out
        return CriticVerdict.model_validate(out)
    except Exception:
        # Fail open to HITL — human remains the backstop.
        logger.debug("Critic LLM unavailable; deferring to HITL", exc_info=True)
        return CriticVerdict(proceed=True, reason="critic_llm_skip")
