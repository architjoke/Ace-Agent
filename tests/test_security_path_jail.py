"""Path jail / sandbox unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_validate_allows_output_and_data():
    from src.core.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
    from src.core.security import validate_safe_path

    out = validate_safe_path(OUTPUT_DIR / "Manual_Action" / "Acme")
    assert out.is_relative_to(PROJECT_ROOT.resolve())
    data = validate_safe_path(DATA_DIR / "lessons_learned.md")
    assert "data" in data.parts


def test_validate_allows_config_yaml_with_flag():
    from src.core.config import CONFIG_YAML
    from src.core.security import validate_safe_path

    p = validate_safe_path(CONFIG_YAML, allow_config_yaml=True)
    assert p.name == "config.yaml"


def test_validate_blocks_traversal():
    from src.core.security import validate_safe_path

    with pytest.raises(PermissionError, match="Sandbox Violation"):
        validate_safe_path("../../../.ssh/id_rsa")


def test_validate_blocks_absolute_escape():
    from src.core.security import validate_safe_path

    # Windows system path
    with pytest.raises(PermissionError, match="Sandbox Violation"):
        validate_safe_path(r"C:\Windows\System32\drivers\etc\hosts")


def test_validate_blocks_empty():
    from src.core.security import validate_safe_path

    with pytest.raises(PermissionError, match="Sandbox Violation"):
        validate_safe_path("   ")


def test_gitignore_covers_profiles_and_logs():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/browser_profile/" in text
    assert "browser_profile_w" in text
    assert "*.log" in text
    assert "agent.log" in text
