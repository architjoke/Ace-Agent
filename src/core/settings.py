"""Load config.yaml + env into a single settings object for local OSS runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.core.config import (
    BROWSER_PROFILE,
    CONFIG_YAML,
    LESSONS_LEARNED,
    MASTER_YAML,
    MEMORY_DB,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_runtime_dirs,
    resolve_master_yaml,
)

# Back-compat alias used across the codebase
REPO_ROOT = PROJECT_ROOT
DEFAULT_CONFIG_PATH = CONFIG_YAML


def _as_path(value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass
class Settings:
    locations: list[str] = field(default_factory=lambda: ["India"])
    country_indeed: str = "India"
    max_results_per_track: int = 8
    sites: list[str] = field(default_factory=lambda: ["linkedin", "indeed", "glassdoor"])
    tracks: dict[str, str] = field(default_factory=dict)
    minimum_fit_score: int = 80
    require_paid: bool = True
    exclude_terms: list[str] = field(default_factory=list)
    master_yaml: Path = field(default_factory=resolve_master_yaml)
    lessons_learned: Path = field(default_factory=lambda: LESSONS_LEARNED)
    memory_db: Path = field(default_factory=lambda: MEMORY_DB)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    browser_user_data_dir: Path = field(default_factory=lambda: BROWSER_PROFILE)
    browser_profile_directory: str = "Default"
    browser_headless: bool = False
    browser_workers: int = 3
    tailor_workers: int = 4
    groq_concurrency: int = 2
    linkedin_max_per_hour: int = 10
    pipeline_time: str = "08:00"
    inbox_time: str = "19:00"
    retrain_time: str = "19:30"
    report_time: str = "20:00"
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    email_lookback_hours: int = 24
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    send_brief: bool = False
    email_address: str = ""
    email_app_password: str = ""
    google_api_key: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_location(self) -> str:
        return self.locations[0] if self.locations else "India"


def _dig(data: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_headless(raw: dict[str, Any]) -> bool:
    """Unify ACE_BROWSER_HEADLESS, browser.headless, and browser.headless_mode."""
    env = (os.environ.get("ACE_BROWSER_HEADLESS") or "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    browser = raw.get("browser") if isinstance(raw.get("browser"), dict) else {}
    if "headless_mode" in browser:
        return bool(browser.get("headless_mode"))
    return bool(browser.get("headless", False))


def load_settings(config_path: Path | None = None) -> Settings:
    ensure_runtime_dirs()
    if config_path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
    raw: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw = loaded

    tracks = _dig(raw, "search_preferences", "tracks", default={}) or {}
    if not isinstance(tracks, dict):
        tracks = {}

    default_master = resolve_master_yaml()
    settings = Settings(
        locations=list(
            _dig(raw, "search_preferences", "locations", default=["India"]) or ["India"]
        ),
        country_indeed=str(
            _dig(raw, "search_preferences", "country_indeed", default="India") or "India"
        ),
        max_results_per_track=int(
            _dig(raw, "search_preferences", "max_results_per_track", default=8) or 8
        ),
        sites=list(
            _dig(
                raw,
                "search_preferences",
                "sites",
                default=["linkedin", "indeed", "glassdoor"],
            )
            or ["linkedin", "indeed", "glassdoor"]
        ),
        tracks={str(k): str(v) for k, v in tracks.items()},
        minimum_fit_score=int(
            _dig(raw, "filters", "minimum_fit_score", default=80) or 80
        ),
        require_paid=bool(_dig(raw, "filters", "require_paid", default=True)),
        exclude_terms=list(_dig(raw, "filters", "exclude_terms", default=[]) or []),
        master_yaml=_as_path(
            _dig(raw, "paths", "master_yaml", default="data/master.yaml"),
            default_master,
        ),
        lessons_learned=_as_path(
            _dig(raw, "paths", "lessons_learned", default="data/lessons_learned.md"),
            LESSONS_LEARNED,
        ),
        memory_db=_as_path(
            _dig(raw, "paths", "memory_db", default="data/ace_memory.db"),
            MEMORY_DB,
        ),
        output_dir=_as_path(
            _dig(raw, "paths", "output_dir", default="output"),
            OUTPUT_DIR,
        ),
        browser_user_data_dir=_as_path(
            os.environ.get("BROWSER_USER_DATA_DIR")
            or _dig(raw, "browser", "user_data_dir", default="data/browser_profile"),
            BROWSER_PROFILE,
        ),
        browser_profile_directory=str(
            _dig(raw, "browser", "profile_directory", default="Default") or "Default"
        ),
        browser_headless=_resolve_headless(raw),
        browser_workers=max(
            1,
            min(
                8,
                int(
                    os.environ.get("ACE_BROWSER_WORKERS")
                    or _dig(raw, "browser", "workers", default=3)
                    or 3
                ),
            ),
        ),
        tailor_workers=max(
            1,
            min(
                8,
                int(
                    os.environ.get("ACE_TAILOR_WORKERS")
                    or _dig(raw, "pipeline", "tailor_workers", default=4)
                    or 4
                ),
            ),
        ),
        groq_concurrency=max(
            1,
            min(
                4,
                int(
                    os.environ.get("ACE_GROQ_CONCURRENCY")
                    or _dig(raw, "pipeline", "groq_concurrency", default=2)
                    or 2
                ),
            ),
        ),
        linkedin_max_per_hour=max(
            1,
            int(
                os.environ.get("ACE_LINKEDIN_MAX_PER_HOUR")
                or _dig(raw, "browser", "linkedin_max_per_hour", default=10)
                or 10
            ),
        ),
        pipeline_time=str(
            _dig(raw, "schedule", "pipeline_time", default="08:00") or "08:00"
        ),
        inbox_time=str(_dig(raw, "schedule", "inbox_time", default="19:00") or "19:00"),
        retrain_time=str(
            _dig(raw, "schedule", "retrain_time", default="19:30") or "19:30"
        ),
        report_time=str(
            _dig(raw, "schedule", "report_time", default="20:00") or "20:00"
        ),
        imap_host=str(
            _dig(raw, "email", "imap_host", default="imap.gmail.com") or "imap.gmail.com"
        ),
        imap_port=int(_dig(raw, "email", "imap_port", default=993) or 993),
        email_lookback_hours=int(
            _dig(raw, "email", "lookback_hours", default=24) or 24
        ),
        smtp_host=str(
            _dig(raw, "email", "smtp_host", default="smtp.gmail.com") or "smtp.gmail.com"
        ),
        smtp_port=int(_dig(raw, "email", "smtp_port", default=587) or 587),
        send_brief=bool(_dig(raw, "email", "send_brief", default=False)),
        email_address=(
            os.environ.get("EMAIL_ADDRESS") or os.environ.get("GMAIL_ADDRESS") or ""
        ).strip(),
        email_app_password=(
            os.environ.get("EMAIL_APP_PASSWORD")
            or os.environ.get("GMAIL_APP_PASSWORD")
            or ""
        ).strip(),
        google_api_key=(
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        ).strip(),
        raw=raw,
    )
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
