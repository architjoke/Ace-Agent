"""Preference engine + Manual_Action path jail smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_hard_soft_and_prune(monkeypatch):
    from src.core import preferences as pref

    rules = [
        pref.PreferenceRule(
            id="h1",
            category="JOB_FILTERING",
            rule_text='Exclude jobs mentioning "whatsapp community"',
            enforcement_level="HARD",
            created="2026-01-01T00:00:00+00:00",
            affirmed="2026-01-01T00:00:00+00:00",
        ),
        pref.PreferenceRule(
            id="s1",
            category="CULTURE",
            rule_text="Prefer remote-friendly roles",
            enforcement_level="SOFT",
            created="2026-01-01T00:00:00+00:00",
            affirmed="2026-01-01T00:00:00+00:00",
        ),
    ]
    monkeypatch.setattr(pref, "load_preference_rules", lambda path=None: rules)

    reason, penalty, hits, cit = pref.apply_preference_gate(
        "Join our WhatsApp community for updates. Python intern."
    )
    assert reason and "preference_hard" in reason
    assert penalty == 0
    assert cit is False

    reason2, penalty2, hits2, cit2 = pref.apply_preference_gate(
        "Remote-friendly Python internship in India."
    )
    assert reason2 is None
    assert penalty2 >= 0
    assert cit2 is False


def test_cleanup_jail():
    from src.core.tools import _assert_under, _manual_action_root
    from src.core.config import PROJECT_ROOT

    root = _manual_action_root()
    root.mkdir(parents=True, exist_ok=True)
    ok = _assert_under(root / "Acme", root)
    assert ok.is_relative_to(root) or str(ok).startswith(str(root))
    try:
        _assert_under(PROJECT_ROOT / "src" / "core", root)
        raise AssertionError("expected path jail to raise")
    except PermissionError:
        pass


if __name__ == "__main__":
    # Minimal no-pytest runner
    from src.core.preferences import (
        apply_preference_gate,
        memorize_preference,
        prune_decayed_preferences,
    )
    from src.core.reflect import _ensure_lessons_file

    _ensure_lessons_file()
    memorize_preference(
        "JOB_FILTERING",
        'Exclude jobs mentioning "whatsapp community"',
        "HARD",
    )
    r, p, h, c = apply_preference_gate("Please join our WhatsApp community first")
    print("hard_gate", bool(r), p, c)
    print("prune", prune_decayed_preferences())
    print("ok")
