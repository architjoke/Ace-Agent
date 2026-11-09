"""Sandboxed Copilot tools — path-jailed, allowlisted mutations only."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool

from src.core.config import CONFIG_YAML, OUTPUT_DIR, PROJECT_ROOT, ensure_runtime_dirs
from src.core.memory import _connect, _norm_company, init_db
from src.core.preferences import memorize_preference
from src.core.security import validate_safe_path
from src.core.settings import reload_settings

logger = logging.getLogger(__name__)

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

# Read-only tools skip HITL confirmation in the copilot loop.
READ_ONLY_TOOLS = frozenset({"find_companies"})


def _safe_dir_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", (name or "").strip()).strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unknown"


def _assert_under(path: Path, root: Path) -> Path:
    """Resolve path and require it stays strictly under root (no escape)."""
    # Global project jail first, then tighten to the caller's subtree.
    path_r = validate_safe_path(path)
    root_r = root.resolve()
    try:
        path_r.relative_to(root_r)
    except ValueError as exc:
        raise PermissionError(
            f"Sandbox Violation: {path_r} is outside {root_r}"
        ) from exc
    return path_r


def _manual_action_root() -> Path:
    ensure_runtime_dirs()
    return (OUTPUT_DIR / "Manual_Action").resolve()


def _company_match_score(query: str, candidate: str) -> float:
    """Score how well a partial/typo'd query matches a real company name (0..1)."""
    needle = _norm_company(query)
    target = _norm_company(candidate)
    if not needle or not target:
        return 0.0
    if needle == target:
        return 1.0

    compact_n = needle.replace(" ", "")
    compact_t = target.replace(" ", "")
    score = 0.0
    if compact_n == compact_t:
        score = 0.98
    elif compact_t.startswith(compact_n) or compact_n.startswith(compact_t):
        # "midbrain" → "midbrains"
        ratio = min(len(compact_n), len(compact_t)) / max(len(compact_n), len(compact_t))
        score = max(score, 0.88 + 0.1 * ratio)
    elif compact_n in compact_t or compact_t in compact_n:
        ratio = min(len(compact_n), len(compact_t)) / max(len(compact_n), len(compact_t))
        score = max(score, 0.78 + 0.15 * ratio)

    seq = SequenceMatcher(None, compact_n, compact_t).ratio()
    score = max(score, seq)

    # Short queries need a stronger hit to avoid "ai" → everything
    if len(compact_n) < 4:
        score *= 0.85
    return round(min(score, 1.0), 4)


def _rank_company_matches(
    query: str,
    candidates: list[str],
    *,
    min_score: float = 0.62,
) -> list[tuple[float, str]]:
    ranked: list[tuple[float, str]] = []
    seen: set[str] = set()
    for name in candidates:
        key = name.strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        score = _company_match_score(query, key)
        if score >= min_score:
            ranked.append((score, key))
    ranked.sort(key=lambda x: (-x[0], x[1].lower()))
    return ranked


def _resolve_unique_company(
    query: str,
    candidates: list[str],
) -> tuple[str | None, list[tuple[float, str]]]:
    """Return (best_unique_name_or_None, ranked_matches)."""
    ranked = _rank_company_matches(query, candidates)
    if not ranked:
        return None, []
    best_score, best_name = ranked[0]
    if best_score < 0.72 and len(ranked) > 1:
        # Weak + contested → ask user
        return None, ranked
    if len(ranked) == 1:
        return best_name, ranked
    second = ranked[1][0]
    if best_score - second >= 0.08 or best_score >= 0.92:
        return best_name, ranked
    return None, ranked


def _recent_application_rows(limit: int = 300) -> list[Any]:
    init_db()
    with _connect() as conn:
        return list(
            conn.execute(
                "SELECT id, company, status, job_url, timestamp FROM applications "
                "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        )


@tool
def find_companies(query: str) -> str:
    """Fuzzy-search recent applications + Manual_Action folders by partial company name.

    STATUS / LOOKUP ONLY — not for form-fill behavior or config.yaml changes.
    Use when the user gives an incomplete / informal company name
    (e.g. 'midbrain', 'kaleris', 'smartbridge') before mark or cleanup.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("No company query provided.")

    rows = _recent_application_rows()
    companies = [str(r["company"]) for r in rows if r["company"]]
    ranked = _rank_company_matches(q, companies, min_score=0.55)

    # Also surface Manual_Action folder names that match
    allowed = _manual_action_root()
    folder_hits: list[str] = []
    if allowed.is_dir():
        folder_names = [p.name.replace("_", " ") for p in allowed.iterdir() if p.is_dir()]
        for score, name in _rank_company_matches(q, folder_names, min_score=0.55)[:5]:
            folder_hits.append(f"{name} (folder, score={score:.2f})")

    if not ranked and not folder_hits:
        return (
            f"No companies matched {q!r}. "
            "Try a longer fragment of the company name."
        )

    # Attach latest status per company
    status_by: dict[str, str] = {}
    for row in rows:
        c = str(row["company"])
        if c not in status_by:
            status_by[c] = str(row["status"])

    lines = [f"Matches for {q!r}:"]
    for score, name in ranked[:8]:
        st = status_by.get(name, "?")
        lines.append(f"  - {name} [{st}] score={score:.2f}")
    if folder_hits:
        lines.append("Manual_Action folders:")
        lines.extend(f"  - {h}" for h in folder_hits)
    unique, _ = _resolve_unique_company(q, companies)
    if unique:
        lines.append(f"Best unique resolve → {unique}")
    else:
        lines.append("Ambiguous — ask the user which company they mean.")
    return "\n".join(lines)


@tool
def mark_application_submitted(company_name: str) -> str:
    """Mark matching application row(s) as SUCCESS (user confirmed they submitted).

    STATUS ONLY — use when the user says they applied / submitted a company.
    Accepts partial names (e.g. 'MidBrains' → 'Midbrains Technologies').
    Do NOT use for form-fill instructions or config.yaml changes.
    """
    q = (company_name or "").strip()
    if not q:
        raise ValueError("No company name provided.")

    rows = _recent_application_rows()
    companies = [str(r["company"]) for r in rows if r["company"]]
    resolved, ranked = _resolve_unique_company(q, companies)
    if not resolved:
        if not ranked:
            return f"No application rows matched company={q!r}."
        opts = ", ".join(f"{n} ({s:.2f})" for s, n in ranked[:6])
        return f"Ambiguous company {q!r}. Candidates: {opts}. Ask which one."

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated = 0
    matched: list[str] = []
    target_norm = _norm_company(resolved)
    with _connect() as conn:
        for row in rows:
            if _norm_company(str(row["company"])) != target_norm:
                continue
            # Prefer pending / manual rows; still allow SUCCESS re-mark
            conn.execute(
                """
                UPDATE applications
                SET status='SUCCESS',
                    error_log='USER_MARKED_SUBMITTED',
                    timestamp=?
                WHERE id=?
                """,
                (ts, row["id"]),
            )
            updated += 1
            matched.append(f"{row['company']} [{row['status']}→SUCCESS]")
        conn.commit()
    if not updated:
        return f"Resolved {q!r} → {resolved}, but no DB rows to update."
    prefix = "" if q.lower() == resolved.lower() else f"Resolved {q!r} → {resolved}. "
    return prefix + f"Marked {updated} row(s) submitted: " + "; ".join(matched[:8])


@tool
def cleanup_manual_folder(company_name: str) -> str:
    """Delete one company folder under output/Manual_Action/ only (path-jailed).

    CLEANUP ONLY — after the user submitted / no longer needs the package.
    Accepts partial names; fuzzy-resolves to a single folder when unambiguous.
    Do NOT use for form-fill instructions or config.yaml changes.
    """
    allowed = _manual_action_root()
    allowed.mkdir(parents=True, exist_ok=True)
    q = (company_name or "").strip()
    if not q:
        raise ValueError("No company name provided.")

    dirs = [p for p in allowed.iterdir() if p.is_dir()]
    # Score against display-ish names (underscores → spaces)
    display = {p: p.name.replace("_", " ") for p in dirs}
    resolved, ranked = _resolve_unique_company(q, list(display.values()))

    target: Path | None = None
    if resolved:
        for path, label in display.items():
            if _norm_company(label) == _norm_company(resolved):
                target = path
                break

    if target is None:
        # Exact safe-name fallback
        safe = _safe_dir_name(q)
        exact = allowed / safe
        if exact.is_dir():
            target = exact
        elif ranked:
            # Ambiguous is a recoverable soft signal — ask the user
            opts = ", ".join(f"{n} ({s:.2f})" for s, n in ranked[:6])
            return f"Ambiguous Manual_Action folders for {q!r}: {opts}."
        else:
            raise FileNotFoundError(
                f"No Manual_Action folder found for {q!r}."
            )

    target = _assert_under(target, allowed)
    if target == allowed:
        raise PermissionError(
            "Path jail violation: cannot delete the Manual_Action root."
        )

    shutil.rmtree(target)
    logger.info("Deleted Manual_Action folder %s", target)
    rel = target.relative_to(PROJECT_ROOT)
    prefix = "" if _norm_company(q) == _norm_company(target.name.replace("_", " ")) else (
        f"Resolved {q!r} → {target.name}. "
    )
    return prefix + f"Deleted Manual_Action folder: {rel}"


@tool
def memorize_new_rule(
    category: str,
    rule_text: str,
    enforcement_level: str,
) -> str:
    """Permanently learn a behavioral preference into lessons_learned.md (ACE_PREF).
    enforcement_level must be HARD or SOFT.

    USE THIS FOR ALL BEHAVIORAL INSTRUCTIONS. If the user says "when you apply, write X",
    "when asked for notice period, put Y", "don't apply to Z", or any form/resume/apply
    behavior — save it here as a rule (NOT update_config_preference).

    Categories (prefer one of):
    - APPLY_BEHAVIOR — how to fill forms, notice period, phone/PIN wording, submit habits
    - JOB_FILTERING — what jobs to skip/prefer (WhatsApp, unpaid, citizenship, locations taste)
    - TAILORING — how to write or shape the resume/cover letter

    HARD = absolute dealbreaker (instant reject where applicable).
    SOFT = -15 fit penalty (serendipity). Re-affirming refreshes the forgetting curve.
    """
    level = (enforcement_level or "").strip().upper()
    if level not in {"HARD", "SOFT"}:
        raise ValueError(
            f"enforcement_level must be strictly 'HARD' or 'SOFT', got {enforcement_level!r}."
        )
    # memorize_preference also validates; let ValueError propagate for self-correction
    rule = memorize_preference(category, rule_text, level)
    return (
        f"Memorized {rule.enforcement_level} rule [{rule.category}] "
        f"(id={rule.id}, affirmed={rule.affirmed}): {rule.rule_text}"
    )


def _dig_set(data: dict, dotted: str, value: Any) -> Any:
    keys = dotted.split(".")
    cur: Any = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    old = cur.get(keys[-1])
    cur[keys[-1]] = value
    return old


@tool
def update_config_preference(target_key: str, new_value: str) -> str:
    """Safely update an allowlisted key in config.yaml (e.g. filters.minimum_fit_score,
    search_preferences.locations, browser.workers).

    USE ONLY FOR SYSTEM SETTINGS like search locations, minimum ATS/fit scores,
    worker counts, headless mode, or other allowlisted config.yaml knobs.
    NEVER use this for instructions on how to fill out web forms or resumes,
    notice periods, cover-letter wording, or "when you apply, write X" behavior —
    those belong in memorize_new_rule (APPLY_BEHAVIOR / JOB_FILTERING / TAILORING).
    """
    key = (target_key or "").strip()
    if key not in ALLOWED_CONFIG_KEYS:
        allowed = ", ".join(sorted(ALLOWED_CONFIG_KEYS))
        raise ValueError(
            f"Key {key!r} is not in the allowlist. Allowed keys: [{allowed}]"
        )

    path = validate_safe_path(CONFIG_YAML, allow_config_yaml=True)
    if not path.is_file():
        raise ValueError(f"config.yaml missing at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml root must be a mapping.")

    val_s = (new_value or "").strip()
    # Type coercion for known keys
    if key.endswith(
        (
            "minimum_fit_score",
            "max_results_per_track",
            "workers",
            "linkedin_max_per_hour",
            "tailor_workers",
            "groq_concurrency",
        )
    ):
        try:
            coerced: Any = int(val_s)
        except ValueError as exc:
            raise ValueError(
                f"Expected integer for {key}, got {new_value!r}"
            ) from exc
    elif key.endswith(("require_paid", "headless")):
        coerced = val_s.lower() in {"1", "true", "yes", "on"}
    elif key.endswith(("exclude_terms", "locations")):
        # Append mode if value doesn't start with REPLACE:
        existing = raw
        for part in key.split("."):
            existing = (existing or {}).get(part) if isinstance(existing, dict) else None
        items = [x.strip() for x in val_s.split("|") if x.strip()]
        if val_s.upper().startswith("REPLACE:"):
            coerced = [x.strip() for x in val_s[8:].split("|") if x.strip()]
        else:
            base = list(existing) if isinstance(existing, list) else []
            for item in items:
                if item not in base:
                    base.append(item)
            coerced = base
    else:
        coerced = val_s

    old = _dig_set(raw, key, coerced)
    path = validate_safe_path(path, allow_config_yaml=True)
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    reload_settings()
    return f"Updated {key}: {old!r} → {coerced!r}"


COPILOT_TOOLS = [
    find_companies,
    mark_application_submitted,
    cleanup_manual_folder,
    memorize_new_rule,
    update_config_preference,
]


def get_copilot_tools() -> list:
    return list(COPILOT_TOOLS)
