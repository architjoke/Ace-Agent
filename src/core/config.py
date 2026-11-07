"""Project root and path helpers — safe from any CWD (Task Scheduler / scripts/)."""

from __future__ import annotations

from pathlib import Path

# src/core/config.py → parents: core → src → <repo root>
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
THEMES_DIR: Path = PROJECT_ROOT / "themes"
CONFIG_YAML: Path = PROJECT_ROOT / "config.yaml"
ENV_FILE: Path = PROJECT_ROOT / ".env"

# Canonical user artifacts under data/
MASTER_YAML: Path = DATA_DIR / "master.yaml"
MASTER_PROFILE_YAML: Path = DATA_DIR / "master_profile.yaml"
MEMORY_DB: Path = DATA_DIR / "ace_memory.db"
LESSONS_LEARNED: Path = DATA_DIR / "lessons_learned.md"
BROWSER_PROFILE: Path = DATA_DIR / "browser_profile"
WORKER_PID: Path = DATA_DIR / "ace_worker.pid"
WORKER_STATUS: Path = DATA_DIR / "browser_workers.json"


def ensure_runtime_dirs() -> None:
    """Create data/ and output/ if missing (first clone / fresh machine)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_master_yaml() -> Path:
    """Prefer data/master.yaml, then master_profile.yaml, then legacy root master.yaml."""
    for candidate in (
        MASTER_YAML,
        MASTER_PROFILE_YAML,
        PROJECT_ROOT / "master.yaml",
    ):
        if candidate.is_file():
            return candidate
    return MASTER_YAML
