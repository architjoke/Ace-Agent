"""Strict Pydantic argument models for Ace Copilot tools (L0).

Hallucinated / malformed LLM args fail validation and are returned as
ToolMessage errors for self-correction — never crash the chat loop.
"""

from __future__ import annotations

from typing import Any, Literal, Type

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.core.copilot_capabilities import ALLOWED_PREF_CATEGORIES

# Config keys allowed via chat (kept here so schemas can validate without importing tools).
ALLOWED_CONFIG_KEYS = frozenset(
    {
        "filters.minimum_fit_score",
        "filters.require_paid",
        "filters.exclude_terms",
        "search_preferences.locations",
        "search_preferences.max_results_per_track",
        "search_preferences.country_indeed",
        "browser.workers",
        "browser.linkedin_max_per_hour",
        "browser.headless",
        "pipeline.tailor_workers",
        "pipeline.groq_concurrency",
    }
)


class EmptyArgs(BaseModel):
    """Tools that take no arguments."""


class FindCompaniesArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=120)


class ListOpenApplicationsArgs(BaseModel):
    limit: int = Field(default=40, ge=5, le=80)


class MarkApplicationSubmittedArgs(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=120)
    job_role: str = Field(default="", max_length=160)


class CleanupManualFolderArgs(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=120)


class PreviewMarkThroughArgs(BaseModel):
    until_company: str = Field(..., min_length=1, max_length=120)


class MarkApplicationsThroughArgs(BaseModel):
    plan_id: str = Field(..., min_length=8, max_length=64)
    also_cleanup: bool = True

    @field_validator("plan_id")
    @classmethod
    def _plan_prefix(cls, v: str) -> str:
        s = (v or "").strip()
        if not s.startswith("mtp_"):
            raise ValueError("plan_id must come from preview_mark_through (mtp_…)")
        return s


class UndoBulkUserMarksArgs(BaseModel):
    keep_company: str = Field(default="", max_length=120)


class MemorizeNewRuleArgs(BaseModel):
    category: str = Field(..., min_length=1, max_length=64)
    rule_text: str = Field(..., min_length=1, max_length=500)
    enforcement_level: Literal["HARD", "SOFT"]

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        cat = "".join(c if c.isalnum() or c == "_" else "_" for c in v.strip().upper())
        cat = "_".join(p for p in cat.split("_") if p)
        if cat not in ALLOWED_PREF_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(ALLOWED_PREF_CATEGORIES)}"
            )
        return cat

    @field_validator("enforcement_level", mode="before")
    @classmethod
    def _level(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().upper()
        return v


class UpdateConfigPreferenceArgs(BaseModel):
    target_key: str = Field(..., min_length=1, max_length=80)
    new_value: str = Field(..., min_length=1, max_length=500)

    @field_validator("target_key")
    @classmethod
    def _key(cls, v: str) -> str:
        key = (v or "").strip()
        if key not in ALLOWED_CONFIG_KEYS:
            raise ValueError(
                f"Key not allowlisted. Allowed: {', '.join(sorted(ALLOWED_CONFIG_KEYS))}"
            )
        return key


class GetAceStatusArgs(BaseModel):
    question: str = Field(default="", max_length=240)


class ResearchCompanyArgs(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=120)
    focus: str = Field(default="", max_length=160)


class GetApplicationFailureReasonArgs(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=120)


class ListLtmEpisodesArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)


class StartApplyWorkerArgs(BaseModel):
    cycle_sleep: int = Field(default=900, ge=60, le=7200)


TOOL_ARG_MODELS: dict[str, Type[BaseModel]] = {
    "list_copilot_capabilities": EmptyArgs,
    "get_ace_status": GetAceStatusArgs,
    "research_company": ResearchCompanyArgs,
    "get_application_failure_reason": GetApplicationFailureReasonArgs,
    "list_ltm_episodes": ListLtmEpisodesArgs,
    "read_active_boards": EmptyArgs,
    "find_companies": FindCompaniesArgs,
    "list_open_applications": ListOpenApplicationsArgs,
    "preview_mark_through": PreviewMarkThroughArgs,
    "mark_application_submitted": MarkApplicationSubmittedArgs,
    "mark_applications_through": MarkApplicationsThroughArgs,
    "cleanup_manual_folder": CleanupManualFolderArgs,
    "undo_bulk_user_marks": UndoBulkUserMarksArgs,
    "list_learned_preferences": EmptyArgs,
    "memorize_new_rule": MemorizeNewRuleArgs,
    "list_config_allowlist": EmptyArgs,
    "update_config_preference": UpdateConfigPreferenceArgs,
    "scan_recruiter_inbox": EmptyArgs,
    "write_daily_brief": EmptyArgs,
    "start_apply_worker": StartApplyWorkerArgs,
    "stop_apply_worker": EmptyArgs,
}


def validate_tool_args(tool_name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Return cleaned args dict or raise ValidationError / ValueError."""
    model = TOOL_ARG_MODELS.get(tool_name)
    if model is None:
        raise ValueError(
            f"ValidationError for {tool_name}: no argument schema registered."
        )
    raw = args if isinstance(args, dict) else {}
    return model.model_validate(raw).model_dump()


def format_validation_error(tool_name: str, exc: Exception) -> str:
    """Compact error string for ToolMessage self-correction."""
    if isinstance(exc, ValidationError):
        parts = [f"ValidationError for {tool_name}:"]
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
            parts.append(f"  - {loc}: {err.get('msg')}")
        parts.append("Fix the arguments and retry the same tool if appropriate.")
        return "\n".join(parts)
    return f"ValidationError for {tool_name}: {exc}"


def try_validate_tool_args(
    tool_name: str, args: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (cleaned_args, None) or (None, error_message)."""
    try:
        return validate_tool_args(tool_name, args), None
    except (ValidationError, ValueError) as exc:
        return None, format_validation_error(tool_name, exc)
