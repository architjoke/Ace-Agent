"""Sandboxed Copilot tools — path-jailed, allowlisted mutations only."""

from __future__ import annotations

import logging
import re
import shutil
from contextvars import ContextVar
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool

from src.core.config import CONFIG_YAML, OUTPUT_DIR, PROJECT_ROOT, ensure_runtime_dirs
from src.core.company_research import research_company_brief
from src.core.copilot_capabilities import (
    ALLOWED_PREF_CATEGORIES,
    CONFIG_VALUE_BOUNDS,
    ChatRole,
    format_capabilities_help,
    read_only_tools,
    tool_names_for_role,
    tool_risk_map,
)
from src.core.tool_schemas import ALLOWED_CONFIG_KEYS
from src.core.copilot_plans import (
    consume_mark_through_plan,
    create_mark_through_plan,
    plan_public_view,
)
from src.core.memory import (
    _connect,
    _norm_company,
    init_db,
    mark_company_role_submitted,
)
from src.core.preferences import memorize_preference
from src.core.security import validate_safe_path
from src.core.settings import reload_settings

logger = logging.getLogger(__name__)

# Derived from capability catalog — do not hand-edit (full shipped set for risk/HITL).
READ_ONLY_TOOLS = read_only_tools()
TOOL_RISK = tool_risk_map()

# RBAC: engine sets role for this turn so tools can self-gate.
_CHAT_ROLE: ContextVar[ChatRole] = ContextVar("ace_copilot_chat_role", default="user")

_BOARD_LABELS = {
    "linkedin": "LinkedIn",
    "naukri": "Naukri",
    "indeed": "Indeed",
    "glassdoor": "Glassdoor",
    "wellfound": "Wellfound",
    "weworkremotely": "We Work Remotely",
    "hiring_cafe": "Hiring Cafe",
    "himalayas": "Himalayas",
    "yc_waas": "YC",
    "yc": "YC",
}


def set_chat_role(role: ChatRole) -> None:
    _CHAT_ROLE.set("admin" if role == "admin" else "user")


def get_chat_role() -> ChatRole:
    return _CHAT_ROLE.get()


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
    try:
        from src.core.settings import get_settings

        root = Path(get_settings().output_dir).expanduser().resolve()
    except Exception:
        root = Path(OUTPUT_DIR).resolve()
    return (root / "Manual_Action").resolve()


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
                "SELECT id, company, status, job_url, timestamp, track, job_title "
                "FROM applications ORDER BY timestamp DESC LIMIT ?",
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


def _open_application_rows(limit: int = 120) -> list[dict[str, Any]]:
    """Recent non-success applications (newest first), de-duped by company."""
    rows = _recent_application_rows(limit=500)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    skip = {"SUCCESS", "ALREADY_APPLIED", "JOB_CLOSED"}
    for row in rows:
        status = str(row["status"] or "").upper()
        if status in skip:
            continue
        company = str(row["company"] or "").strip()
        if not company:
            continue
        key = _norm_company(company)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "company": company,
                "status": status,
                "job_title": str(row["job_title"] or "").strip(),
                "job_url": str(row["job_url"] or ""),
                "track": str(row["track"] or "") if "track" in row.keys() else "",
                "timestamp": str(row["timestamp"] or ""),
            }
        )
        if len(out) >= limit:
            break
    return out


def _manual_folders_newest_first() -> list[Path]:
    root = _manual_action_root()
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


@tool
def list_open_applications(limit: int = 40) -> str:
    """List open/pending applications + Manual_Action folders (read-only).

    Use when the user says they applied to several companies, asks what's left,
    or says "all until X" so you can see the queue before bulk-marking.
    """
    try:
        lim = max(5, min(int(limit or 40), 80))
    except (TypeError, ValueError):
        lim = 40
    opens = _open_application_rows(limit=lim)
    folders = _manual_folders_newest_first()
    lines = [f"Open applications (newest first, up to {lim}):"]
    if not opens:
        lines.append("  (none in DB)")
    else:
        for i, row in enumerate(opens, 1):
            role = row["job_title"] or "(role unknown)"
            lines.append(
                f"  {i}. {row['company']} · {role} [{row['status']}]"
            )
    lines.append(f"Manual_Action folders ({len(folders)}):")
    if not folders:
        lines.append("  (none)")
    else:
        for p in folders[:lim]:
            lines.append(f"  - {p.name.replace('_', ' ')}")
    return "\n".join(lines)


@tool
def mark_application_submitted(company_name: str, job_role: str = "") -> str:
    """Mark company(+role) as submitted so Ace will not reopen that profile for 30 days.

    STATUS ONLY — use when the user says they applied / submitted.
    Pass job_role when the company has multiple openings (e.g. 'SDE Intern').
    Accepts partial company names (e.g. 'MidBrains' → 'Midbrains Technologies').
    Creates a ledger row if none exist yet.
    Do NOT use for form-fill instructions or config.yaml changes.
    """
    q = (company_name or "").strip()
    if not q:
        raise ValueError("No company name provided.")
    role = (job_role or "").strip()

    rows = _recent_application_rows()
    companies = [str(r["company"]) for r in rows if r["company"]]
    resolved, ranked = _resolve_unique_company(q, companies)
    if not resolved and ranked:
        opts = ", ".join(f"{n} ({s:.2f})" for s, n in ranked[:6])
        return f"Ambiguous company {q!r}. Candidates: {opts}. Ask which one."
    company = resolved or q

    # Prefer an existing URL for this company when marking.
    job_url = ""
    track = ""
    target_norm = _norm_company(company)
    for row in rows:
        if _norm_company(str(row["company"])) != target_norm:
            continue
        job_url = str(row["job_url"] or "")
        track = str(row["track"] or "") if "track" in row.keys() else ""
        break

    summary = mark_company_role_submitted(
        company, role, job_url=job_url, track=track
    )
    prefix = "" if q.lower() == company.lower() else f"Resolved {q!r} → {company}. "
    return (
        prefix
        + f"Recorded submitted: {summary['company']} · {summary['job_title']} "
        f"({summary['updated']} row(s)). Ace will skip this company+role for 30 days."
    )


def _cleanup_manual_folder_impl(company_name: str) -> str:
    """Path-jailed Manual_Action folder delete (shared by tool + bulk plan)."""
    allowed = _manual_action_root()
    allowed.mkdir(parents=True, exist_ok=True)
    q = (company_name or "").strip()
    if not q:
        raise ValueError("No company name provided.")

    dirs = [p for p in allowed.iterdir() if p.is_dir()]
    display = {p: p.name.replace("_", " ") for p in dirs}
    resolved, ranked = _resolve_unique_company(q, list(display.values()))

    target: Path | None = None
    if resolved:
        for path, label in display.items():
            if _norm_company(label) == _norm_company(resolved):
                target = path
                break

    if target is None:
        safe = _safe_dir_name(q)
        exact = allowed / safe
        if exact.is_dir():
            target = exact
        elif ranked:
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
    # Absolute-prefix jail (blocks ../ escape even if resolve quirks).
    allowed_abs = str(allowed.resolve())
    target_abs = str(target.resolve())
    if not (target_abs == allowed_abs or target_abs.startswith(allowed_abs + "\\")
            or target_abs.startswith(allowed_abs + "/")):
        raise PermissionError(
            f"Path jail violation: {target_abs} escapes Manual_Action"
        )

    company_label = target.name.replace("_", " ")
    shutil.rmtree(target)
    logger.info("Deleted Manual_Action folder %s", target)
    try:
        from src.core.output_artifacts import release_company_build

        release_company_build(
            company_label, archive=True, reason="manual_folder_cleaned"
        )
    except Exception:
        logger.debug("build GC after manual cleanup failed", exc_info=True)
    rel = target.relative_to(PROJECT_ROOT)
    prefix = "" if _norm_company(q) == _norm_company(company_label) else (
        f"Resolved {q!r} → {target.name}. "
    )
    return prefix + f"Deleted Manual_Action folder: {rel}"


@tool
def cleanup_manual_folder(company_name: str) -> str:
    """Delete one company folder under output/Manual_Action/ only (path-jailed).

    CLEANUP ONLY — after the user submitted / no longer needs the package.
    Accepts partial names; fuzzy-resolves to a single folder when unambiguous.
    Do NOT use for form-fill instructions or config.yaml changes.
    """
    return _cleanup_manual_folder_impl(company_name)


def _cut_manual_folders_through(until_company: str) -> tuple[str | None, list[str], str]:
    """Oldest→newest Manual_Action walk, inclusive cut at until_company.

    Returns (resolved_name, company_labels, error_or_empty).
    Never returns every folder if the cut company is missing.
    """
    folders = list(reversed(_manual_folders_newest_first()))  # oldest first
    labels = [p.name.replace("_", " ") for p in folders]
    if not labels:
        return None, [], "No Manual_Action folders found."
    resolved, ranked = _resolve_unique_company(until_company, labels)
    if not resolved and ranked:
        opts = ", ".join(f"{n} ({s:.2f})" for s, n in ranked[:6])
        return None, [], f"Ambiguous until_company {until_company!r}. Candidates: {opts}."
    if not resolved:
        return (
            None,
            [],
            (
                f"{until_company!r} is not in Manual_Action. "
                "Refusing bulk wipe. Mark that company alone, or list_open_applications."
            ),
        )
    target_norm = _norm_company(resolved)
    batch: list[str] = []
    for label in labels:
        batch.append(label)
        if _norm_company(label) == target_norm:
            return resolved, batch, ""
    return (
        None,
        [],
        f"Internal error: resolved {resolved!r} but cut not found.",
    )


@tool
def preview_mark_through(until_company: str) -> str:
    """Preview which Manual_Action companies would be marked through until_company.

    ALWAYS call this before mark_applications_through. Issues a one-shot plan_id
    bound to the exact company list (oldest→newest inclusive). Does not mutate.
    """
    resolved, batch, err = _cut_manual_folders_through(until_company)
    if err:
        return err
    assert resolved is not None
    if len(batch) > 25:
        return (
            f"Preview too large: {len(batch)} companies (max 25). "
            "Pick a nearer cut company, or mark one-by-one."
        )
    plan = create_mark_through_plan(resolved, batch)
    meta = plan_public_view(plan)
    lines = [
        f"Preview: mark {len(batch)} Manual_Action package(s) through {resolved} "
        f"(oldest → newest, inclusive):",
    ]
    lines.extend(f"  {i}. {name}" for i, name in enumerate(batch, 1))
    lines.append(
        f"plan_id={plan.plan_id} digest={meta['digest']} "
        f"expires_in_sec={meta['expires_in_sec']}"
    )
    lines.append(
        "If this list looks right, call mark_applications_through("
        f"plan_id={plan.plan_id!r}, also_cleanup=true)."
    )
    lines.append("If wrong, do NOT call mark_applications_through — ask the user.")
    return "\n".join(lines)


@tool
def mark_applications_through(plan_id: str, also_cleanup: bool = True) -> str:
    """Execute a preview_mark_through plan_id (mark + optional folder cleanup). Cap 25."""
    try:
        plan = consume_mark_through_plan(plan_id)
    except ValueError as exc:
        return str(exc)
    resolved = plan.resolved
    batch = list(plan.companies)
    if len(batch) > 25:
        return f"Refusing: {len(batch)} companies exceeds max 25."

    marked: list[str] = []
    cleaned: list[str] = []

    for company in batch:
        summary = mark_company_role_submitted(company, "")
        marked.append(f"{summary['company']} · {summary['job_title']}")
        if also_cleanup:
            try:
                # cleanup_manual also archives/releases output/<Company>/ when safe.
                msg = _cleanup_manual_folder_impl(company)
                if "Deleted Manual_Action" in str(msg):
                    cleaned.append(company)
            except Exception:
                logger.debug("bulk cleanup miss for %s", company, exc_info=True)

    lines = [f"Marked {len(marked)} Manual_Action package(s) through {resolved}:"]
    lines.extend(f"  ✓ {m}" for m in marked)
    if also_cleanup:
        lines.append(f"Cleaned Manual_Action folders: {len(cleaned)}")
        lines.extend(f"  - {c}" for c in cleaned)
    lines.append("Ace will skip these company+role pairs for 30 days.")
    return "\n".join(lines)


@tool
def undo_bulk_user_marks(keep_company: str = "") -> str:
    """Undo the latest mistaken USER_MARKED_SUBMITTED bulk (restore MANUAL_ACTION).

    Use when mark_applications_through marked too many companies. Keeps keep_company
    as SUCCESS if provided (e.g. 'Eulogik'). Recreates Manual_Action stubs from
    leftover output/<Company> PDFs when possible.
    """
    keep_norm = _norm_company(keep_company) if keep_company else ""
    init_db()
    reverted: list[str] = []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, company, timestamp
            FROM applications
            WHERE error_log='USER_MARKED_SUBMITTED'
              AND status='SUCCESS'
            ORDER BY timestamp DESC
            LIMIT 200
            """
        ).fetchall()
        if not rows:
            return "Nothing to undo — no USER_MARKED_SUBMITTED SUCCESS rows found."
        # Undo only the latest burst (same minute: YYYY-MM-DDTHH:MM).
        latest = str(rows[0]["timestamp"] or "")[:16]
        for row in rows:
            company = str(row["company"] or "")
            stamp = str(row["timestamp"] or "")
            if latest and stamp[:16] != latest:
                continue
            if keep_norm and _norm_company(company) == keep_norm:
                continue
            from src.core.db_jail import jailed_update_application_status

            jailed_update_application_status(
                conn,
                int(row["id"]),
                status="MANUAL_ACTION",
                error_log="REVERTED_BULK_MARK_MISTAKE",
            )
            reverted.append(company)
        conn.commit()

    manual = _manual_action_root()
    manual.mkdir(parents=True, exist_ok=True)
    out = manual.parent  # settings-aware output root
    restored = 0
    for company in sorted(set(reverted)):
        safe = _safe_dir_name(company)
        dest = manual / safe
        dest.mkdir(parents=True, exist_ok=True)
        # Prefer live build; fall back to Submitted/ archive after GC.
        sources = [out / safe, out / "Submitted" / safe]
        copied: list[str] = []
        seen_names: set[str] = set()
        for src in sources:
            if not src.is_dir():
                continue
            for pat in ("*Resume*.pdf", "*CV*.pdf", "*.pdf", "tailored.yaml"):
                for f in src.glob(pat):
                    if not f.is_file() or f.name in seen_names:
                        continue
                    shutil.copy2(f, dest / f.name)
                    seen_names.add(f.name)
                    copied.append(f.name)
                    restored += 1
        (dest / "RESTORED.txt").write_text(
            f"Restored stub after bulk-mark undo.\ncompany={company}\ncopied={copied}\n",
            encoding="utf-8",
        )

    keep_msg = f" Kept {keep_company!r} as SUCCESS." if keep_company else ""
    return (
        f"Reverted {len(set(reverted))} companies back to MANUAL_ACTION.{keep_msg} "
        f"Restored Manual_Action stubs (files copied where available). "
        f"Original cover letters may be missing."
    )


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
    cat = re.sub(r"[^A-Z0-9_]", "_", (category or "").strip().upper())
    cat = re.sub(r"_+", "_", cat).strip("_")
    if cat not in ALLOWED_PREF_CATEGORIES:
        raise ValueError(
            f"category must be one of {sorted(ALLOWED_PREF_CATEGORIES)}, got {category!r}."
        )
    # Block obvious instruction-injection payloads into durable lessons.
    lowered = (rule_text or "").lower()
    for needle in (
        "ignore previous",
        "ignore all instructions",
        "system prompt",
        "jailbreak",
        "<untrusted_",
        "sudo ",
        "rm -rf",
    ):
        if needle in lowered:
            raise ValueError(
                "rule_text looks like an injection/jailbreak attempt — refused."
            )
    rule = memorize_preference(cat, rule_text, level)
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
        bounds = CONFIG_VALUE_BOUNDS.get(key)
        if bounds is not None:
            lo, hi = bounds
            if coerced < lo or coerced > hi:
                raise ValueError(
                    f"{key} must be between {lo} and {hi}, got {coerced}."
                )
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


@tool
def list_copilot_capabilities() -> str:
    """List what Ace chat can do now vs planned, with risk classes."""
    return format_capabilities_help(get_chat_role())


@tool
def get_ace_status(question: str = "") -> str:
    """Ace status dump, or answer a specific status question (manuals/failures/leftovers)."""
    from src.core.desk import answer, format_status

    q = (question or "").strip()
    return answer(q) if q else format_status()


@tool
def research_company(company_name: str, focus: str = "") -> str:
    """Job-focused company research: what they do, history, Ace apply history."""
    return research_company_brief(company_name, focus=focus)


@tool
def get_application_failure_reason(company_name: str) -> str:
    """Most recent failed / human-gated application for a company (error_log + status)."""
    init_db()
    needle = _norm_company(company_name)
    if not needle:
        return "company_name is empty."
    fail_statuses = (
        "FAILED",
        "FAILED_TIMEOUT_OR_STUCK",
        "FAILED_INTERRUPTED",
        "REQUIRES_HUMAN_INTERVENTION",
        "REQUIRES_HUMAN_CITIZENSHIP_REVIEW",
        "JOB_CLOSED",
        "MANUAL_ACTION",
    )
    # Short-lived connection — respect SQLite busy limits (timeout already in _connect).
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT company, job_title, status, error_log, timestamp, job_url
            FROM applications
            WHERE status IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT 80
            """.format(placeholders=",".join("?" * len(fail_statuses))),
            fail_statuses,
        ).fetchall()
    matches: list[Any] = []
    for row in rows:
        if needle in _norm_company(str(row["company"] or "")) or _norm_company(
            str(row["company"] or "")
        ).startswith(needle):
            matches.append(row)
        if len(matches) >= 5:
            break
    if not matches:
        # Fuzzy: SequenceMatcher against recent failures
        ranked: list[tuple[float, Any]] = []
        for row in rows:
            score = _company_match_score(company_name, str(row["company"] or ""))
            if score >= 0.72:
                ranked.append((score, row))
        ranked.sort(key=lambda x: -x[0])
        matches = [r for _, r in ranked[:3]]
    if not matches:
        return (
            f"No failed / human-gated applications found for {company_name!r} "
            "in ace_memory.db."
        )
    lines = [f"Failure lookup for {company_name!r} (most recent first):"]
    for row in matches[:3]:
        err = (row["error_log"] or "").strip() or "(no error_log)"
        if len(err) > 400:
            err = err[:399].rstrip() + "…"
        lines.append(
            f"- {row['company']} | {row['job_title'] or '—'} | "
            f"status={row['status']} | ts={row['timestamp']}\n"
            f"  error_log: {err}"
        )
    return "\n".join(lines)


@tool
def list_ltm_episodes(limit: int = 10) -> str:
    """Admin-only: recent long-term memory episodes from ace_chat_memory.db."""
    if get_chat_role() != "admin":
        return (
            "UNAUTHORIZED: list_ltm_episodes is admin-only. "
            "Use chat_role=admin for LTM diagnostics."
        )
    from src.core.chat_memory import get_memory_store

    lim = max(1, min(int(limit or 10), 50))
    store = get_memory_store()
    episodes = store.recent(limit=lim)
    if not episodes:
        return "No active LTM episodes in ace_chat_memory.db."
    lines = [f"Recent LTM episodes ({len(episodes)}, limit={lim}):"]
    for ep in episodes:
        lines.append(
            f"- [{ep.kind}] {ep.subject or '—'} conf={ep.confidence:.2f} "
            f"id={ep.id}\n  {ep.content[:220]}"
        )
    return "\n".join(lines)


@tool
def read_active_boards() -> str:
    """Parse config.yaml / settings for currently enabled job boards."""
    from src.core.settings import get_settings

    sites = [str(s).strip().lower() for s in (get_settings().sites or []) if str(s).strip()]
    if not sites:
        return "No job boards enabled in config.yaml search_preferences.sites."
    labels = [_BOARD_LABELS.get(s, s.replace("_", " ").title()) for s in sites]
    lines = [
        f"Enabled job boards ({len(labels)}):",
        "  " + ", ".join(labels),
        "Keys: " + ", ".join(sites),
    ]
    return "\n".join(lines)


@tool
def list_learned_preferences() -> str:
    """Show active HARD/SOFT ACE_PREF rules from lessons_learned.md."""
    from src.core.preferences import load_preference_rules

    rules = load_preference_rules()
    if not rules:
        return "No ACE_PREF rules stored yet."
    lines = [f"Active preferences ({len(rules)}):"]
    for r in rules:
        lines.append(
            f"  • [{r.level}] {r.category} id={r.id}: {r.rule_text}"
        )
    return "\n".join(lines)


@tool
def list_config_allowlist() -> str:
    """Show config.yaml keys chat is allowed to change."""
    lines = ["Allowlisted config.yaml keys:"]
    for key in sorted(ALLOWED_CONFIG_KEYS):
        bounds = CONFIG_VALUE_BOUNDS.get(key)
        extra = f" (int {bounds[0]}..{bounds[1]})" if bounds else ""
        lines.append(f"  • {key}{extra}")
    return "\n".join(lines)


@tool
def scan_recruiter_inbox() -> str:
    """Scan recruiter IMAP now and update application outcomes."""
    from src.core.inbox_scanner import process_inbox

    init_db()
    results = process_inbox() or []
    if not results:
        return "Inbox scan complete — no actionable recruiter outcomes found."
    lines = [f"Actionable outcomes: {len(results)}"]
    for row in results[:40]:
        lines.append(
            f"- {row.get('outcome')} @ {row.get('company_name')} "
            f"(db_matches={row.get('db_matches', 0)})"
        )
    return "\n".join(lines)


@tool
def write_daily_brief() -> str:
    """Generate today's Markdown daily brief under output/."""
    from src.core.report import generate_daily_brief

    init_db()
    path = generate_daily_brief()
    return f"Brief written → {path}"


@tool
def start_apply_worker(cycle_sleep: int = 900) -> str:
    """Start the background hunt→apply worker (same as `ace start`)."""
    from src.core.desk import worker_running

    alive, pid = worker_running()
    if alive:
        return f"Worker already running (pid {pid})."
    sleep = int(cycle_sleep or 900)
    if sleep < 60 or sleep > 7200:
        raise ValueError("cycle_sleep must be between 60 and 7200 seconds.")
    # Import late to avoid circular imports with agent CLI.
    import agent as ace_agent

    code = ace_agent._start_worker(None, sleep)
    if code != 0:
        return f"Failed to start worker (exit={code})."
    alive2, pid2 = worker_running()
    return (
        f"Started apply worker (pid {pid2}, cycle_sleep={sleep}s)."
        if alive2
        else "Start requested; worker pid not confirmed yet — check `get_ace_status`."
    )


@tool
def stop_apply_worker() -> str:
    """Stop the background apply worker (same as `ace stop`)."""
    from src.core.desk import worker_running

    alive, pid = worker_running()
    if not alive:
        return "Worker is already stopped."
    import agent as ace_agent

    code = ace_agent._stop_worker()
    return (
        f"Stopped apply worker (was pid {pid})."
        if code == 0
        else f"Stop attempted for pid {pid} (exit={code})."
    )


COPILOT_TOOLS = [
    list_copilot_capabilities,
    get_ace_status,
    research_company,
    get_application_failure_reason,
    list_ltm_episodes,
    read_active_boards,
    find_companies,
    list_open_applications,
    preview_mark_through,
    mark_application_submitted,
    mark_applications_through,
    undo_bulk_user_marks,
    cleanup_manual_folder,
    list_learned_preferences,
    memorize_new_rule,
    list_config_allowlist,
    update_config_preference,
    scan_recruiter_inbox,
    write_daily_brief,
    start_apply_worker,
    stop_apply_worker,
]


def get_copilot_tools(role: ChatRole = "user") -> list:
    """Return tools allowed for chat_role (USER_CAPABILITIES vs ADMIN_CAPABILITIES)."""
    allowed = tool_names_for_role(role)
    return [t for t in COPILOT_TOOLS if t.name in allowed]

