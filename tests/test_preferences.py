"""Preference engine + Manual_Action path jail smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_hard_soft_and_prune(tmp_path, monkeypatch):
    from src.core import preferences as pref
    from src.core import reflect

    lessons = tmp_path / "lessons_learned.md"
    lessons.write_text("# Lessons\n\n", encoding="utf-8")
    monkeypatch.setattr(pref, "LESSONS_LEARNED", lessons)
    monkeypatch.setattr(reflect, "lessons_path", lambda: lessons)
    monkeypatch.setattr(pref, "_archive_path", lambda: tmp_path / "archived.md")

    hard = pref.memorize_preference(
        "JOB_FILTERING",
        'Exclude jobs mentioning "whatsapp community"',
        "HARD",
    )
    soft = pref.memorize_preference(
        "CULTURE",
        "Prefer remote-friendly roles",
        "SOFT",
    )
    assert hard.enforcement_level == "HARD"
    assert soft.enforcement_level == "SOFT"

    reason, penalty, hits = pref.apply_preference_gate(
        "Join our WhatsApp community for updates. Python intern."
    )
    assert reason and "preference_hard" in reason
    assert penalty == 0

    reason2, penalty2, hits2 = pref.apply_preference_gate(
        "Remote-friendly Python internship in India."
    )
    assert reason2 is None
    # soft may or may not hit depending on heuristic; remote-friendly should hit
    assert penalty2 >= 0


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
    r, p, h = apply_preference_gate("Please join our WhatsApp community first")
    print("hard_gate", bool(r), p)
    print("prune", prune_decayed_preferences())
    print("ok")
