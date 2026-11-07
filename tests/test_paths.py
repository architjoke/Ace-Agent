"""Smoke tests: PROJECT_ROOT + SQLite path resolve from any CWD."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_project_root_and_memory_db():
    from src.core.config import MEMORY_DB, PROJECT_ROOT, ensure_runtime_dirs
    from src.core.memory import init_db
    from src.core.settings import get_settings, reload_settings

    ensure_runtime_dirs()
    assert PROJECT_ROOT == REPO
    assert MEMORY_DB == REPO / "data" / "ace_memory.db"
    assert MEMORY_DB.is_absolute()

    reload_settings()
    settings = get_settings()
    assert settings.memory_db.is_absolute()
    assert settings.memory_db == MEMORY_DB or settings.memory_db.name == "ace_memory.db"

    path = init_db()
    assert path.is_absolute()
    assert path.exists()


if __name__ == "__main__":
    test_project_root_and_memory_db()
    print("ok", REPO / "data" / "ace_memory.db")
