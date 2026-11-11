"""Append-only HITL audit trail for Ace Copilot approvals.

Writes structured JSON lines under output/audit_log.jsonl (path-jailed).
Never stores secrets from .env — only tool name, validated args, user text snippet.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import OUTPUT_DIR, ensure_runtime_dirs
from src.core.security import redact_secrets, validate_safe_path

logger = logging.getLogger(__name__)

AUDIT_LOG_NAME = "audit_log.jsonl"


def audit_log_path() -> Path:
    ensure_runtime_dirs()
    return validate_safe_path(OUTPUT_DIR / AUDIT_LOG_NAME)


def append_hitl_audit(
    *,
    call_id: str,
    tool_name: str,
    args: dict[str, Any],
    user_text: str,
    approved: bool,
    risk: str = "",
) -> None:
    """Append one HITL resolution event (approved or denied)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "hitl_resolve",
        "approved": bool(approved),
        "call_id": call_id,
        "tool_name": tool_name,
        "risk": risk,
        "user_text": redact_secrets((user_text or "")[:500]),
        "args": json.loads(redact_secrets(json.dumps(args or {}, default=str))),
    }
    try:
        path = audit_log_path()
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        logger.debug("HITL audit log write failed", exc_info=True)
