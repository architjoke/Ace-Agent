"""Durable HIT LIMIT + next-day resume / queue-drain awareness."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_quota_persists_and_clears_after_reset(tmp_path, monkeypatch):
    from src.core import llm_router as lr

    state = tmp_path / "quota_state.json"
    monkeypatch.setattr(lr, "_QUOTA_STATE_PATH", state)
    lr.clear_token_limit_reached()

    now = 1_700_000_000.0
    reset_at = now + 3600.0
    lr.mark_token_limit_reached(
        "openai: no credits remaining",
        blocks_apply=False,
        reset_at=reset_at,
    )
    assert state.is_file()
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["token_limit_reached"] is True
    assert payload["blocks_browser_applies"] is False
    assert payload["reset_at"] == reset_at

    # Simulate process restart — in-memory clear, then sync from disk.
    lr._TOKEN_LIMIT_REACHED = False
    lr._TOKEN_LIMIT_DETAIL = ""
    lr._BLOCKS_BROWSER_APPLIES = False
    lr._TOKEN_LIMIT_RESET_AT = 0.0
    assert lr.sync_token_limit_state(now=now + 10) is True
    assert lr.token_limit_reached() is True
    assert lr.blocks_browser_applies() is False

    # After reset_at — auto clear.
    assert lr.sync_token_limit_state(now=reset_at + 1) is False
    assert lr.token_limit_reached() is False
    assert not state.is_file()


def test_vision_burn_blocks_browser_applies(tmp_path, monkeypatch):
    from src.core import llm_router as lr

    monkeypatch.setattr(lr, "_QUOTA_STATE_PATH", tmp_path / "q.json")
    lr.clear_token_limit_reached()
    lr.mark_token_limit_reached("gemini capacity", blocks_apply=True, reset_at=time.time() + 99)
    assert lr.token_limit_reached() is True
    assert lr.blocks_browser_applies() is True
    lr.clear_token_limit_reached()


def test_next_midnight_reset_helper():
    from src.core.llm_router import _next_local_midnight_ts

    ts = _next_local_midnight_ts()
    assert ts > time.time()
    assert ts - time.time() < 48 * 3600
