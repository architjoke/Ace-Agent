"""Hybrid NLP + LLM LangGraph tailor: extract → draft → rigid ATS score → compile."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from src.core.ats_playbook import track_brief
from src.core.ats_scanner import (
    calculate_ats_score,
    extract_jd_keywords,
    filter_supportable_keywords,
    resolve_ats_pass_score,
)
from src.core.llm_router import (
    DEFAULT_GROQ_FALLBACK_MODEL,
    invoke_with_failover,
    text_provider_name,
)
from src.core.config import OUTPUT_DIR as _DEFAULT_OUTPUT
from src.core.config import PROJECT_ROOT, THEMES_DIR, ensure_runtime_dirs, resolve_master_yaml
from src.core.llm_text import message_text
from src.core.prompts import DRAFT_RESUME
from src.core.reflect import load_lessons

logger = logging.getLogger(__name__)

_ARCHETYPE_LOCK = threading.Lock()
_ARCHETYPE_CACHE_FILE = _DEFAULT_OUTPUT / "archetype_pdf_cache.json"
_ALIAS_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:sde|swe|software\s*dev(?:eloper|elopment)?|software\s*engineer)s?\b", re.I), "software_engineer"),
    (re.compile(r"\b(?:back[\s-]?end|be)\b", re.I), "backend"),
    (re.compile(r"\b(?:front[\s-]?end|fe)\b", re.I), "frontend"),
    (re.compile(r"\bfull[\s-]?stack\b", re.I), "fullstack"),
    (re.compile(r"\b(?:machine\s+learning|ml)\b", re.I), "ml"),
    (re.compile(r"\bdata\s+scientists?\b", re.I), "data_scientist"),
    (re.compile(r"\b(?:devops|sre)\b", re.I), "devops"),
]


def normalize_role_title(title: str) -> str:
    """Normalize job title → archetype slug (title hash, not full JD)."""
    s = (title or "").lower().replace("&", " ").replace("/", " ").replace("|", " ")
    for pat, repl in _ALIAS_SUBS:
        s = pat.sub(repl, s)
    s = re.sub(r"[^a-z0-9_\s]+", " ", s)
    s = re.sub(r"\s+", "_", s.strip()).strip("_")
    return (s or "general")[:64]


def _archetype_key(track: str, title: str) -> str:
    return f"{(track or 'GENERAL').strip().upper()}|{normalize_role_title(title)}"


def _cache_ttl_sec() -> float:
    try:
        return max(1.0, float(os.environ.get("ACE_ARCHETYPE_CACHE_HOURS") or "48")) * 3600.0
    except ValueError:
        return 48.0 * 3600.0


def _load_archetype_cache() -> dict:
    if not _ARCHETYPE_CACHE_FILE.is_file():
        return {}
    try:
        return json.loads(_ARCHETYPE_CACHE_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_archetype_cache(data: dict) -> None:
    ensure_runtime_dirs()
    tmp = _ARCHETYPE_CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_ARCHETYPE_CACHE_FILE)


def lookup_archetype_pdf(track: str, title: str) -> tuple[str, str] | None:
    """Return (pdf, yaml) if a same-archetype PDF was built within TTL.

    Same-title archetypes share PDFs across companies for 48h — intentional.
    """
    if not (title or "").strip():
        return None
    key = _archetype_key(track, title)
    with _ARCHETYPE_LOCK:
        row = _load_archetype_cache().get(key)
    if not isinstance(row, dict):
        return None
    try:
        if time.time() - float(row.get("ts") or 0) > _cache_ttl_sec():
            return None
    except (TypeError, ValueError):
        return None
    pdf = str(row.get("pdf") or "").strip()
    if not pdf:
        return None
    try:
        from src.core.security import validate_safe_path

        pdf_path = validate_safe_path(pdf)
    except Exception:
        return None
    if not pdf_path.is_file():
        return None
    return str(pdf_path.resolve()), str(row.get("yaml") or "")


def store_archetype_pdf(track: str, title: str, pdf: str, yaml_text: str = "") -> None:
    if not (title or "").strip() or not pdf:
        return
    try:
        from src.core.security import validate_safe_path

        pdf_path = validate_safe_path(pdf)
    except Exception:
        return
    if not pdf_path.is_file():
        return
    key = _archetype_key(track, title)
    with _ARCHETYPE_LOCK:
        data = _load_archetype_cache()
        data[key] = {
            "pdf": str(pdf_path.resolve()),
            "yaml": yaml_text or "",
            "ts": time.time(),
        }
        try:
            _save_archetype_cache(data)
        except Exception:
            logger.debug("archetype cache write failed", exc_info=True)

# Fallback only — real gate is per-company / per-JD via resolve_ats_pass_score()
ATS_PASS_SCORE_DEFAULT = 85
REPO_ROOT = PROJECT_ROOT
OUTPUT_DIR = _DEFAULT_OUTPUT
STYLE_GUIDE_PATH = Path(__file__).resolve().parent / "resume_style_guide.md"
LAYOUT_PATH = Path(__file__).resolve().parent / "resume_layout.yaml"

_SCHOOL_MARKERS = (
    "class x",
    "class xii",
    "class 10",
    "class 12",
    "cbse",
    "matriculation",
    "intermediate",
    "high school",
    "secondary",
    "senior secondary",
)


def _load_style_guide() -> str:
    if STYLE_GUIDE_PATH.is_file():
        return STYLE_GUIDE_PATH.read_text(encoding="utf-8")
    logger.warning("Missing style guide at %s", STYLE_GUIDE_PATH)
    return (
        "Use XYZ bullets (Accomplished X as measured by Y, by doing Z). "
        "Strong action verbs only. No fluff. No invented metrics. "
        "ONE page max. College education only when tight."
    )


def _text_provider_is_groq() -> bool:
    try:
        return text_provider_name("draft") == "groq"
    except Exception:
        return False


def _clip_head(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n…[truncated]"


def _clip_tail(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "…[truncated earlier]\n" + text[-(max_chars - 24) :].lstrip()


# Groq free tier ≈ 8k TPM — keep draft prompts well under that ceiling.
_GROQ_LESSONS_CHARS = 900
_GROQ_PREV_DRAFT_CHARS = 2200
_GROQ_MASTER_CHARS = 5500

_PRESET_ORDER = ("airy", "balanced", "dense", "ultra_dense")

# Soft draft target / hard shrink caps — prefer weighty handmade bullets over stubs.
MAX_BULLET_CHARS = 175
MAX_SKILL_DETAIL_CHARS = 110
MAX_SUMMARY_LINE_CHARS = 320

# Canonical fresher section order (matches handmade Archit resumes).
_SECTION_ORDER = (
    "summary",
    "education",
    "experience",
    "projects",
    "skills",
    "hackathons_and_activities",
)
_JD_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "your",
        "you",
        "are",
        "our",
        "will",
        "have",
        "has",
        "been",
        "into",
        "about",
        "their",
        "they",
        "them",
        "who",
        "what",
        "when",
        "where",
        "which",
        "while",
        "using",
        "used",
        "work",
        "role",
        "team",
        "job",
        "experience",
        "required",
        "requirements",
        "preferred",
        "ability",
        "skills",
        "strong",
        "good",
        "etc",
    }
)


def _load_layout_file() -> dict:
    if not LAYOUT_PATH.is_file():
        return {}
    loaded = yaml.safe_load(LAYOUT_PATH.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _design_for_preset(preset: str = "balanced") -> dict:
    """Merge shared chrome + spacing preset. Never emit page margins."""
    layout = _load_layout_file()
    name = (preset or "balanced").strip().lower()
    presets = layout.get("presets") if isinstance(layout.get("presets"), dict) else {}
    body = presets.get(name) if isinstance(presets.get(name), dict) else None
    if body is None:
        body = presets.get("balanced") if isinstance(presets.get("balanced"), dict) else {}

    design: dict = {"theme": str(layout.get("theme") or "ace")}
    page = layout.get("page")
    if isinstance(page, dict):
        # Strip any accidental margin keys — Typst crash on RenderCV 2.3.
        safe_page = {
            k: v
            for k, v in page.items()
            if not str(k).endswith("_margin") and "margin" not in str(k).lower()
        }
        design["page"] = safe_page
    header = layout.get("header")
    if isinstance(header, dict):
        design["header"] = dict(header)
    # Preset may tighten header gaps (ultra_dense) without touching page margins.
    if isinstance(body.get("header"), dict):
        merged_header = dict(design.get("header") or {})
        merged_header.update(body["header"])
        design["header"] = merged_header
    for key in ("text", "section_titles", "entries", "highlights", "colors"):
        if isinstance(body.get(key), dict):
            design[key] = body[key]
    if not body and isinstance(layout.get("design"), dict):
        # Legacy single-block layout file
        return dict(layout["design"])
    return design


def _count_highlights(data: dict) -> int:
    sections = _cv_sections(data) if isinstance(data, dict) else {}
    total = 0
    for key in ("experience", "projects"):
        rows = sections.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("highlights"), list):
                total += len(row["highlights"])
    return total


def _select_design_preset(yaml_text: str, *, force: str | None = None) -> str:
    if force in _PRESET_ORDER:
        return force
    layout = _load_layout_file()
    vol = layout.get("volume") if isinstance(layout.get("volume"), dict) else {}
    airy_max = int(vol.get("airy_max_bullets") or 10)
    dense_min = int(vol.get("dense_min_bullets") or 18)
    try:
        data = yaml.safe_load(yaml_text) or {}
        bullets = _count_highlights(data if isinstance(data, dict) else {})
    except Exception:
        bullets = dense_min
    if bullets <= airy_max:
        return "airy"
    if bullets >= dense_min:
        return "dense"
    return "balanced"


def _next_denser_preset(preset: str) -> str | None:
    try:
        idx = _PRESET_ORDER.index(preset)
    except ValueError:
        return "dense"
    if idx >= len(_PRESET_ORDER) - 1:
        return None
    return _PRESET_ORDER[idx + 1]


def _bullet_density_issues(yaml_text: str) -> list[str]:
    """Return human-readable density failures (empty = OK)."""
    issues: list[str] = []
    try:
        data = yaml.safe_load(yaml_text) or {}
    except Exception:
        return ["YAML failed to parse for density check."]
    if not isinstance(data, dict):
        return ["Tailored YAML must be a mapping."]
    sections = _cv_sections(data)
    if not sections:
        return ["Missing cv.sections."]

    experience = sections.get("experience")
    if not isinstance(experience, list) or not experience:
        issues.append("Experience missing — need exactly 2 roles with 3–4 bullets each.")
    else:
        if len(experience) != 2:
            issues.append(
                f"Experience has {len(experience)} roles — need exactly 2."
            )
        for i, row in enumerate(experience[:2], start=1):
            if not isinstance(row, dict):
                issues.append(f"Experience role {i} is malformed.")
                continue
            highs = row.get("highlights")
            n = len(highs) if isinstance(highs, list) else 0
            # Prefer exactly 3; allow 4 only when page still has room (compile may trim).
            if n < 3 or n > 4:
                issues.append(
                    f"Experience role {i} has {n} bullets — need 3 XYZ bullets "
                    "(4th only if JD-critical)."
                )

    projects = sections.get("projects")
    if not isinstance(projects, list) or not projects:
        issues.append("Projects missing — need 2 projects with 3 XYZ bullets each.")
    else:
        if len(projects) < 2:
            issues.append(f"Projects has {len(projects)} entries — need 2.")
        for i, row in enumerate(projects[:2], start=1):
            if not isinstance(row, dict):
                issues.append(f"Project {i} is malformed.")
                continue
            highs = row.get("highlights")
            n = len(highs) if isinstance(highs, list) else 0
            if n < 3 or n > 4:
                issues.append(
                    f"Project {i} has {n} bullets — need 3 XYZ bullets "
                    "(4th only if JD-critical)."
                )
    return issues


_PROJECT_GITHUB_LINKS: dict[str, str] = {
    "earthquake impact prediction system": (
        "https://github.com/architjoke/earthquake-impact-app"
    ),
    "dcgan fashion item generator": "https://github.com/architjoke/DCGAN",
    "dcgan": "https://github.com/architjoke/DCGAN",
    "ai road inspection system": (
        "https://github.com/MSG-Technologies/AI-Road-Inspection"
    ),
    "offline road inspection": (
        "https://github.com/MSG-Technologies/AI-Road-Inspection"
    ),
    "road inspection": (
        "https://github.com/MSG-Technologies/AI-Road-Inspection"
    ),
    "pothole detection": (
        "https://github.com/MSG-Technologies/AI-Road-Inspection"
    ),
}

# Never ship these projects on tailored fresher resumes
_EXCLUDED_PROJECT_MARKERS = (
    "another u",
    "avatar-driven social",
    "avatar driven social",
)


def _resolve_project_url(name: str) -> tuple[str, str | None]:
    """Return (display_title, github_url_or_None)."""
    name = name.strip()
    link_match = re.match(r"^\[([^\]]+)\]\((https?://[^)]+)\)\s*$", name)
    if link_match:
        return link_match.group(1), link_match.group(2)
    key = re.sub(r"\s+", " ", name.lower())
    url = _PROJECT_GITHUB_LINKS.get(key)
    if not url:
        for alias, mapped in _PROJECT_GITHUB_LINKS.items():
            if alias in key:
                url = mapped
                break
    return name, url


def _normalize_project_links(data: dict) -> None:
    """GitHub on the right (location); tech tags in summary; no project years."""
    sections = _cv_sections(data)
    projects = sections.get("projects")
    if not isinstance(projects, list):
        return
    for project in projects:
        if not isinstance(project, dict):
            continue
        # Do not show project year beside repo titles
        project.pop("date", None)
        project.pop("start_date", None)
        project.pop("end_date", None)
        title, url = _resolve_project_url(str(project.get("name") or ""))
        summary = str(project.get("summary") or "").strip()
        if not url:
            m = re.search(r"https://github\.com/[^\s)\"']+", summary)
            if m:
                url = m.group(0)
        # Strip any existing GitHub chip from summary (moved to location/right column)
        summary = re.sub(
            r"\s*[·•|-]?\s*\[GitHub\]\([^)]+\)\s*",
            "",
            summary,
            flags=re.IGNORECASE,
        ).strip(" ·|-")
        summary = re.sub(
            r"\s*https://github\.com/[^\s]+",
            "",
            summary,
            flags=re.IGNORECASE,
        ).strip(" ·|-")
        project["name"] = title
        if summary:
            project["summary"] = summary
        elif "summary" in project:
            project.pop("summary", None)
        if url:
            project["location"] = f"[GitHub]({url})"
        elif str(project.get("location") or "").lower().startswith("[github]"):
            project.pop("location", None)


def _cv_sections(data: dict) -> dict:
    """Return cv.sections as a dict, normalizing list-shaped mistakes."""
    cv = data.get("cv")
    if not isinstance(cv, dict):
        return {}
    sections = cv.get("sections")
    if isinstance(sections, dict):
        return sections
    if isinstance(sections, list):
        merged: dict = {}
        for item in sections:
            if isinstance(item, dict):
                merged.update(item)
        cv["sections"] = merged
        data["cv"] = cv
        return merged
    return {}


def _reorder_sections(data: dict) -> None:
    """Force Summary → Education → Experience → Projects → Skills → Activities."""
    sections = _cv_sections(data)
    if not sections:
        return
    # Normalize alternate activity keys into hackathons_and_activities.
    for alt in ("activities", "extracurricular", "extracurricular_activities"):
        if alt in sections and "hackathons_and_activities" not in sections:
            sections["hackathons_and_activities"] = sections.pop(alt)
        elif alt in sections:
            sections.pop(alt, None)
    ordered: dict = {}
    for key in _SECTION_ORDER:
        if key in sections:
            ordered[key] = sections[key]
    for key, value in sections.items():
        if key not in ordered:
            ordered[key] = value
    cv = data.get("cv")
    if isinstance(cv, dict):
        cv["sections"] = ordered
        data["cv"] = cv


_HALLUCINATION_MARKERS = (
    r"\b\d+\s*%\b",
    r"\b\d+k\b",
    r"\breq/?day\b",
    r"\bmAP\b",
    r"\bF1\b",
    r"\blatency\b",
    r"\bHackathonX\b",
    r"\bGoogle Developer Student Club\b",
    r"\bGDSC\b",
    r"\bTrie\b",
    r"\b12k\b",
    r"\b500\+\s*daily\b",
)


def _strip_hallucinated_highlights(data: dict, master_yaml: str = "") -> None:
    """Drop bullets that invent metrics/orgs absent from the fact bank."""
    master_blob = (master_yaml or "").lower()
    sections = _cv_sections(data)
    for key in ("experience", "projects"):
        rows = sections.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("highlights"), list):
                continue
            kept: list[str] = []
            for h in row["highlights"]:
                text = str(h or "").strip()
                if not text:
                    continue
                lower = text.lower()
                suspicious = any(re.search(pat, text, re.I) for pat in _HALLUCINATION_MARKERS)
                if suspicious:
                    # Keep only if the same numeric/claim fragment exists in master.
                    nums = re.findall(r"\d+(?:\.\d+)?%?", text)
                    if nums and not all(n.lower() in master_blob for n in nums):
                        continue
                    if re.search(r"hackathonx|gdsc|google developer student", lower):
                        continue
                kept.append(text)
            if kept:
                row["highlights"] = kept
    # Fake activity sections from LLM drafts.
    acts = sections.get("hackathons_and_activities") or sections.get("activities")
    if isinstance(acts, list):
        clean_acts = []
        for a in acts:
            lower = str(a).lower()
            if re.search(r"hackathonx|gdsc|google developer student", lower):
                continue
            clean_acts.append(a)
        if "hackathons_and_activities" in sections or clean_acts:
            sections["hackathons_and_activities"] = clean_acts
            sections.pop("activities", None)
    cv = data.get("cv")
    if isinstance(cv, dict):
        cv["sections"] = sections
        data["cv"] = cv


def _drop_excluded_projects(data: dict) -> None:
    """Remove Another U (and similar) from tailored project lists."""
    sections = _cv_sections(data)
    projects = sections.get("projects")
    if not isinstance(projects, list):
        return
    kept: list = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        blob = " ".join(
            str(project.get(k) or "") for k in ("name", "summary")
        ).lower()
        if any(marker in blob for marker in _EXCLUDED_PROJECT_MARKERS):
            continue
        kept.append(project)
    sections["projects"] = kept
    cv = data.get("cv")
    if isinstance(cv, dict):
        cv["sections"] = sections
        data["cv"] = cv


def _normalize_header_contacts(data: dict) -> None:
    """Clickable LinkedIn/GitHub profile URLs; drop duplicate website=GitHub."""
    cv = data.get("cv")
    if not isinstance(cv, dict):
        return
    # website often duplicates GitHub and shows without a clear label
    website = str(cv.get("website") or "").lower()
    if "github.com" in website:
        cv.pop("website", None)

    networks = cv.get("social_networks")
    if not isinstance(networks, list):
        networks = []
    by_net: dict[str, dict] = {}
    for row in networks:
        if isinstance(row, dict) and row.get("network"):
            by_net[str(row["network"]).strip().lower()] = row

    # Ensure canonical usernames → RenderCV builds full https URLs
    linkedin = by_net.get("linkedin") or {
        "network": "LinkedIn",
        "username": "archit-gupta-41379b293",
    }
    linkedin["network"] = "LinkedIn"
    linkedin["username"] = str(
        linkedin.get("username") or "archit-gupta-41379b293"
    ).strip().lstrip("/")
    if linkedin["username"].startswith("in/"):
        linkedin["username"] = linkedin["username"][3:]

    github = by_net.get("github") or {"network": "GitHub", "username": "architjoke"}
    github["network"] = "GitHub"
    github["username"] = str(github.get("username") or "architjoke").strip().lstrip("/")

    cv["social_networks"] = [linkedin, github]
    data["cv"] = cv


def _unwrap_rendercv_root(data: dict) -> dict:
    """LLMs sometimes wrap the doc as ``RenderCV: {cv: …}`` — unwrap to top-level cv."""
    if not isinstance(data, dict):
        return data
    if "cv" in data:
        return data
    for key in ("RenderCV", "rendercv", "render_cv"):
        inner = data.get(key)
        if isinstance(inner, dict) and ("cv" in inner or "design" in inner):
            # Preserve sibling design/settings if nested oddly
            out = dict(inner)
            for k, v in data.items():
                if k == key:
                    continue
                if k not in out:
                    out[k] = v
            return out
    return data


def _apply_design(yaml_text: str, preset: str = "balanced") -> str:
    """Normalize content + apply ace theme spacing preset (no page margins)."""
    data = yaml.safe_load(yaml_text) or {}
    if not isinstance(data, dict):
        raise ValueError("Tailored YAML must be a mapping")
    data = _unwrap_rendercv_root(data)
    _drop_excluded_projects(data)
    _normalize_project_links(data)
    _normalize_header_contacts(data)
    _normalize_education_rows(data)
    _normalize_skills_rows(data)
    _reorder_sections(data)
    data["design"] = _design_for_preset(preset)
    settings = data.get("settings")
    if isinstance(settings, dict):
        settings.pop("date", None)
    elif "settings" in data:
        data.pop("settings", None)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)



def _extract_cgpa_only(text: str) -> str | None:
    """Keep only `CGPA: X.X/10` — strip coursework / leadership smuggled into the line."""
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.search(
        r"(?:CGPA|GPA)\s*[:\-]?\s*([\d]+(?:\.[\d]+)?)\s*(?:/\s*([\d]+(?:\.[\d]+)?))?",
        raw,
        re.I,
    )
    if not match:
        return None
    num = match.group(1)
    den = match.group(2) or "10"
    return f"CGPA: {num}/{den}"


def _truncate_text(text: str, max_chars: int) -> str:
    raw = str(text or "").strip()
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    cut = raw[: max_chars - 1].rstrip(" ,;:-")
    return cut + "…"


def _jd_tokens(jd: str) -> set[str]:
    tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{2,}", jd or "")
        if t.lower() not in _JD_STOPWORDS
    }
    return tokens


def _row_jd_score(row: dict, tokens: set[str]) -> int:
    if not tokens:
        return 0
    parts: list[str] = []
    for key in ("company", "position", "location", "name", "summary", "area", "degree"):
        if row.get(key):
            parts.append(str(row.get(key)))
    highs = row.get("highlights")
    if isinstance(highs, list):
        parts.extend(str(h) for h in highs)
    blob = " ".join(parts).lower()
    return sum(1 for tok in tokens if tok in blob)


def _rank_rows_by_jd(rows: list, jd: str | None) -> list:
    if not isinstance(rows, list) or not rows:
        return rows if isinstance(rows, list) else []
    tokens = _jd_tokens(jd or "")
    if not tokens:
        return list(rows)
    indexed = list(enumerate(rows))
    indexed.sort(
        key=lambda pair: (
            -_row_jd_score(pair[1], tokens) if isinstance(pair[1], dict) else 0,
            pair[0],
        )
    )
    return [row for _, row in indexed]


def _master_skills_fallback() -> list[dict]:
    """OneLineEntry skills from master.yaml when the LLM breaks the schema."""
    master_path = resolve_master_yaml()
    try:
        raw = yaml.safe_load(master_path.read_text(encoding="utf-8")) or {}
        rows = (
            ((raw.get("cv") or {}).get("sections") or {}).get("skills")
            if isinstance(raw, dict)
            else None
        )
    except Exception:
        rows = None
    out: list[dict] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("label") and row.get("details"):
                out.append(
                    {
                        "label": str(row["label"]).strip(),
                        "details": str(row["details"]).strip(),
                    }
                )
    return out[:5]


def _normalize_skills_rows(data: dict) -> None:
    """Force RenderCV OneLineEntry skills: {label, details} only."""
    sections = _cv_sections(data)
    skills = sections.get("skills")
    if isinstance(skills, dict):
        # LLM sometimes emits a mapping instead of a list of OneLineEntry rows.
        skills = [
            {
                "label": str(k),
                "details": (
                    ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                ),
            }
            for k, v in skills.items()
        ]
        sections["skills"] = skills
    if not isinstance(skills, list):
        skills = []
    cleaned: list[dict] = []
    for row in skills:
        if isinstance(row, str) and row.strip():
            # "Programming: Python, SQL" or bare skill line
            if ":" in row:
                left, right = row.split(":", 1)
                cleaned.append(
                    {"label": left.strip() or "Skills", "details": right.strip()}
                )
            else:
                cleaned.append({"label": "Skills", "details": row.strip()})
            continue
        if not isinstance(row, dict):
            continue
        # LLM shorthand: {Programming: "Python, SQL"} (no label/details keys)
        if (
            "label" not in row
            and "details" not in row
            and "items" not in row
            and "keywords" not in row
            and len(row) == 1
        ):
            key, val = next(iter(row.items()))
            details = (
                ", ".join(str(x) for x in val) if isinstance(val, list) else str(val)
            )
            cleaned.append({"label": str(key).strip(), "details": details.strip()})
            continue
        label = (
            row.get("label")
            or row.get("group")
            or row.get("name")
            or row.get("category")
            or ""
        )
        details = row.get("details")
        if details is None and isinstance(row.get("items"), list):
            details = ", ".join(
                str(x).replace("**", "").strip()
                for x in row["items"]
                if str(x).strip()
            )
        if details is None and isinstance(row.get("keywords"), list):
            details = ", ".join(str(x).strip() for x in row["keywords"] if str(x).strip())
        if not label and not details:
            continue
        cleaned.append(
            {
                "label": str(label or "Skills").strip(),
                "details": _truncate_text(
                    str(details or "").strip(), MAX_SKILL_DETAIL_CHARS * 2
                ),
            }
        )
    if not cleaned:
        cleaned = _master_skills_fallback()
        if cleaned:
            logger.warning("Skills schema broken — restored %s rows from master.yaml", len(cleaned))
    if cleaned:
        sections["skills"] = cleaned[:5]
        cv = data.get("cv")
        if isinstance(cv, dict):
            cv["sections"] = sections
            data["cv"] = cv


def _normalize_education_rows(data: dict) -> None:
    """College-only education; optional single CGPA highlight (no coursework)."""
    sections = _cv_sections(data)
    education = sections.get("education")
    if not isinstance(education, list):
        return

    kept: list = []
    for row in education:
        if not isinstance(row, dict):
            continue
        blob = " ".join(
            str(row.get(k) or "") for k in ("institution", "area", "degree")
        ).lower()
        if any(marker in blob for marker in _SCHOOL_MARKERS):
            continue
        clean = {
            k: row[k]
            for k in (
                "institution",
                "area",
                "degree",
                "start_date",
                "end_date",
                "location",
                "date",
            )
            if k in row and row[k] not in (None, "")
        }
        cgpa: str | None = None
        highs = row.get("highlights")
        if isinstance(highs, list):
            for h in highs:
                cgpa = _extract_cgpa_only(str(h or ""))
                if cgpa:
                    break
        if not cgpa:
            for key in ("gpa", "cgpa"):
                if row.get(key):
                    cgpa = _extract_cgpa_only(f"CGPA: {row.get(key)}")
                    break
        if cgpa:
            clean["highlights"] = [cgpa]
        kept.append(clean)

    if kept:
        sections["education"] = kept[:1]
        cv = data.get("cv")
        if isinstance(cv, dict):
            cv["sections"] = sections
            data["cv"] = cv


def _drop_precollege_education(yaml_text: str) -> str:
    """Keep university row only; strip education bullets except optional CGPA."""
    data = yaml.safe_load(yaml_text) or {}
    if not isinstance(data, dict):
        return yaml_text
    _normalize_education_rows(data)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _enforce_one_page_budget(
    yaml_text: str,
    *,
    preset: str | None = None,
    jd: str | None = None,
    master_yaml: str | None = None,
) -> str:
    """Deterministic fresher budget: JD-ranked 2 roles / 2 projects, clean education."""
    data = yaml.safe_load(yaml_text) or {}
    if not isinstance(data, dict):
        return yaml_text
    data = _unwrap_rendercv_root(data)
    _normalize_education_rows(data)
    _normalize_skills_rows(data)
    _drop_excluded_projects(data)
    # Only strip metrics when we have a fact bank to compare against — empty
    # master would delete every quantified bullet (false hallucination cull).
    if (master_yaml or "").strip():
        _strip_hallucinated_highlights(data, master_yaml)
    sections = _cv_sections(data)

    experience = sections.get("experience")
    if isinstance(experience, list) and experience:
        ranked = _rank_rows_by_jd(experience, jd)
        # Prefer real internships; drop soft campus filler when we already have 2.
        ranked = [
            r
            for r in ranked
            if not (
                isinstance(r, dict)
                and "university intern" in str(r.get("position") or "").lower()
                and len(ranked) > 2
            )
        ] or ranked
        sections["experience"] = ranked[:2]

    projects = sections.get("projects")
    if isinstance(projects, list) and projects:
        ranked = _rank_rows_by_jd(projects, jd)
        sections["projects"] = ranked[:2]

    for act_key in ("hackathons_and_activities", "extracurricular", "activities"):
        acts = sections.get(act_key)
        if isinstance(acts, list) and len(acts) > 3:
            sections[act_key] = acts[:3]

    for drop_key in ("certificates", "publications", "interests", "awards"):
        sections.pop(drop_key, None)

    cv = data.get("cv")
    if isinstance(cv, dict):
        cv["sections"] = sections
        data["cv"] = cv
    _reorder_sections(data)

    chosen = preset or _select_design_preset(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )
    return _apply_design(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        chosen,
    )


def _cap_entry_highlights(row: dict, *, max_bullets: int, max_chars: int) -> dict:
    out = dict(row)
    highs = out.get("highlights")
    if isinstance(highs, list):
        out["highlights"] = [
            _truncate_text(str(h), max_chars) for h in highs[:max_bullets]
        ]
    return out


def _hard_shrink_yaml(
    yaml_text: str,
    *,
    preset: str = "ultra_dense",
    jd: str | None = None,
    level: int = 3,
) -> str:
    """Progressive 1-page shrink — densify spacing first; keep weighty bullets."""
    text = _enforce_one_page_budget(yaml_text, preset=preset, jd=jd)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        return text
    sections = _cv_sections(data)

    # Level 1: trim to 4 bullets, keep activities.
    # Level 2: trim to 3 bullets, keep 2 activity lines.
    # Level 3: 3 bullets, 1 project only if still overflowing.
    exp_bullets = 4 if level <= 1 else 3
    proj_bullets = 4 if level <= 1 else 3
    proj_keep = 2 if level < 3 else 1
    act_keep = 3 if level <= 1 else (2 if level == 2 else 2)

    experience = sections.get("experience")
    if isinstance(experience, list) and experience:
        sections["experience"] = [
            _cap_entry_highlights(
                row, max_bullets=exp_bullets, max_chars=MAX_BULLET_CHARS
            )
            if isinstance(row, dict)
            else row
            for row in experience[:2]
        ]

    projects = sections.get("projects")
    if isinstance(projects, list) and projects:
        sections["projects"] = [
            _cap_entry_highlights(
                row, max_bullets=proj_bullets, max_chars=MAX_BULLET_CHARS
            )
            if isinstance(row, dict)
            else row
            for row in projects[:proj_keep]
        ]

    for act_key in ("hackathons_and_activities", "extracurricular", "activities"):
        acts = sections.get(act_key)
        if isinstance(acts, list) and acts:
            sections["hackathons_and_activities"] = [
                _truncate_text(str(a), MAX_BULLET_CHARS + 40) for a in acts[:act_keep]
            ]
            if act_key != "hackathons_and_activities":
                sections.pop(act_key, None)

    summary = sections.get("summary")
    if isinstance(summary, list):
        sections["summary"] = [
            _truncate_text(str(line), MAX_SUMMARY_LINE_CHARS) for line in summary[:1]
        ]
    elif isinstance(summary, str):
        sections["summary"] = _truncate_text(summary, MAX_SUMMARY_LINE_CHARS)

    skills = sections.get("skills")
    if isinstance(skills, list):
        capped_skills = []
        for row in skills[:5]:
            if isinstance(row, dict):
                item = dict(row)
                if item.get("details"):
                    item["details"] = _truncate_text(
                        str(item["details"]), MAX_SKILL_DETAIL_CHARS
                    )
                capped_skills.append(item)
            else:
                capped_skills.append(row)
        sections["skills"] = capped_skills

    _normalize_education_rows(data)
    cv = data.get("cv")
    if isinstance(cv, dict):
        cv["sections"] = sections
        data["cv"] = cv
    _reorder_sections(data)
    return _apply_design(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        preset,
    )


def assert_one_page_pdf(pdf_path: str | Path) -> int:
    """Raise if PDF missing or not exactly one page. Returns page count (1)."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF missing: {path}")
    pages = _pdf_page_count(path)
    if pages != 1:
        raise ValueError(
            f"Strict 1-page gate failed: {path.name} has {pages} page(s). "
            "Refuse to ship or upload multi-page resumes."
        )
    return pages


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        try:
            from PyPDF2 import PdfReader

            return len(PdfReader(str(pdf_path)).pages)
        except Exception as exc:
            logger.warning("Could not count PDF pages: %s", exc)
            return -1


# Cap LangGraph draft↔compile retries early for speed (override ACE_MAX_DRAFT_REVISIONS).
MAX_DRAFT_REVISIONS = max(1, int(os.environ.get("ACE_MAX_DRAFT_REVISIONS", "2")))
# Skip airy→balanced escalate — start dense by default.
_START_PRESET = (os.environ.get("ACE_TAILOR_START_PRESET") or "dense").strip().lower()
if _START_PRESET not in _PRESET_ORDER:
    _START_PRESET = "dense"


def _floor_preset(preset: str) -> str:
    name = (preset or _START_PRESET).strip().lower()
    try:
        if name not in _PRESET_ORDER or _PRESET_ORDER.index(name) < _PRESET_ORDER.index(
            _START_PRESET
        ):
            return _START_PRESET
    except ValueError:
        return _START_PRESET
    return name


class State(TypedDict):
    jd: str
    master_yaml: str
    company: str
    job_track: str
    required_keywords: list[str]
    unsupported_keywords: list[str]
    missing_keywords: list[str]
    current_tailored_yaml: str
    ats_score: int
    ats_pass_score: int
    ats_pass_reason: str
    critique_feedback: str
    compile_error: str
    yaml_path: str
    pdf_path: str
    revision_count: int
    density_ok: bool
    design_preset: str


def _draft_invoke(prompt: str, *, model: str | None = None):
    """Draft invoke with Groq→NVIDIA→Ollama failover."""
    return invoke_with_failover("draft", prompt, temperature=0, model=model)


def _quote_markdown_list_items(yaml_text: str) -> str:
    """Quote list scalars that contain **bold** so PyYAML does not treat * as aliases."""

    def _fix(match: re.Match[str]) -> str:
        indent, body = match.group(1), match.group(2).rstrip()
        if body.startswith(("'", '"', "|", ">")):
            return match.group(0)
        if "**" in body or body.startswith("*"):
            escaped = body.replace("'", "''")
            return f"{indent}- '{escaped}'"
        return match.group(0)

    return re.sub(r"^([ \t]*)-[ \t]+(.+)$", _fix, yaml_text, flags=re.M)


def _extract_yaml(text: str) -> str:
    """Pull RenderCV YAML out of LLM output — tolerant of fences/prose/hallucinations."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty text (no RenderCV YAML)")

    # Prefer the *last* fenced yaml/yml block (models often put prose in an earlier fence).
    fences = list(
        re.finditer(
            r"```(?:ya?ml)?[ \t]*\r?\n(.*?)```",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
    )
    yaml_text = ""
    if fences:
        # Prefer a fence that looks like RenderCV; else take the last fence.
        for fence in reversed(fences):
            candidate = fence.group(1).strip()
            if re.search(r"(?m)^(cv|RenderCV|ssot|design)\s*:", candidate):
                yaml_text = candidate
                break
        if not yaml_text:
            yaml_text = fences[-1].group(1).strip()
    else:
        # ~~~ fences or opening fence without a clean close (truncated Groq output)
        tilde = re.search(
            r"~~~(?:ya?ml)?[ \t]*\r?\n(.*?)~~~",
            raw,
            re.DOTALL | re.IGNORECASE,
        )
        if tilde:
            yaml_text = tilde.group(1).strip()
        else:
            yaml_text = re.sub(
                r"^```(?:ya?ml)?[ \t]*\r?\n?", "", raw, count=1, flags=re.I
            )
            yaml_text = re.sub(r"```\s*$", "", yaml_text).strip()

    # Unwrap accidental `RenderCV:` root if the model echoed the file wrapper.
    yaml_text = re.sub(r"(?m)^RenderCV:\s*\r?\n", "", yaml_text, count=1)

    # Drop leading prose before the first YAML key
    if not re.match(r"^(cv|ssot|design|settings|RenderCV)\s*:", yaml_text, re.I | re.M):
        m = re.search(r"(?m)^(cv|ssot|design|settings|RenderCV)\s*:", yaml_text)
        if m:
            yaml_text = yaml_text[m.start() :].strip()
    yaml_text = re.sub(r"(?mi)^COMPANY:.*(?:\r?\n)?", "", yaml_text, count=1).strip()
    # Strip common LLM chatter lines that break PyYAML
    yaml_text = re.sub(
        r"(?mi)^(here is|here's|sure[,!]?|updated resume|output:)\b.*(?:\r?\n)?",
        "",
        yaml_text,
    ).strip()
    yaml_text = _quote_markdown_list_items(yaml_text)
    if not yaml_text:
        raise ValueError("Model returned no RenderCV YAML")
    try:
        loaded = yaml.safe_load(yaml_text)
    except Exception as exc:
        # Last-chance: drop lines that look like prose (no ':' and no list marker)
        repaired_lines: list[str] = []
        for line in yaml_text.splitlines():
            stripped = line.strip()
            if not stripped:
                repaired_lines.append(line)
                continue
            if stripped.startswith(("#", "-", "cv:", "ssot:", "design:", "settings:")):
                repaired_lines.append(line)
                continue
            if ":" in stripped or stripped.startswith(("'", '"', "|", ">")):
                repaired_lines.append(line)
                continue
            # skip unkeyed prose
        repaired = "\n".join(repaired_lines).strip()
        try:
            loaded = yaml.safe_load(repaired)
            yaml_text = repaired
        except Exception:
            raise ValueError(f"Model returned invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Model YAML did not parse to a mapping (dict)")
    return yaml_text


def _safe_dir_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name).strip().strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unknown"


def _find_pdf(tailored_yaml: str, search_dirs: list[Path] | None = None) -> Path:
    loaded = yaml.safe_load(tailored_yaml) or {}
    cv_name = str((loaded.get("cv") or {}).get("name") or "CV")
    stem = f"{cv_name.replace(' ', '_')}_CV"
    candidates = search_dirs or [REPO_ROOT / "rendercv_output", OUTPUT_DIR]
    expected = candidates[0] / f"{stem}.pdf"
    for folder in candidates:
        direct = folder / f"{stem}.pdf"
        if direct.is_file():
            return direct
        pdfs = sorted(folder.rglob("*.pdf"), key=lambda path: path.stat().st_mtime)
        if pdfs:
            return pdfs[-1]
    return expected


def extract_node(state: State) -> dict:
    """Classical NLP extraction of JD keywords (no LLM) + company ATS bar."""
    raw = extract_jd_keywords(state["jd"], top_k=20)
    supportable, unsupported = filter_supportable_keywords(raw, state["master_yaml"])
    threshold = resolve_ats_pass_score(
        company=state.get("company") or "",
        supportable_keywords=supportable,
        unsupported_keywords=unsupported,
    )
    logger.info(
        "Supportable keywords=%s unsupported=%s ats_pass=%s (%s)",
        supportable,
        unsupported,
        threshold["pass_score"],
        threshold["reason"],
    )
    return {
        "required_keywords": supportable,
        "unsupported_keywords": unsupported,
        "missing_keywords": [],
        "ats_score": 0,
        "ats_pass_score": int(threshold["pass_score"]),
        "ats_pass_reason": str(threshold["reason"]),
        "critique_feedback": "",
    }


def _build_draft_prompt(state: State, *, compact: bool) -> str:
    track = state.get("job_track") or "GENERAL"
    required = state.get("required_keywords") or []
    missing = state.get("missing_keywords") or []
    unsupported = state.get("unsupported_keywords") or []
    compile_error = (state.get("compile_error") or "").strip()
    critique = (state.get("critique_feedback") or "").strip()
    lessons = load_lessons()
    prev_draft = state.get("current_tailored_yaml") or ""
    master = state.get("master_yaml") or ""
    jd = state.get("jd") or ""

    if compact:
        lessons = _clip_tail(lessons, _GROQ_LESSONS_CHARS)
        prev_draft = _clip_head(prev_draft, _GROQ_PREV_DRAFT_CHARS)
        master = _clip_head(master, _GROQ_MASTER_CHARS)
        logger.info(
            "Groq compact prompt: lessons=%s prev=%s master=%s",
            len(lessons),
            len(prev_draft),
            len(master),
        )

    lessons_block = f"## Lessons (follow)\n{lessons}\n" if lessons else ""
    missing_block = (
        f"## Missing ATS keywords (integrate organically)\n{missing}\n"
        if missing
        else ""
    )
    compile_block = (
        f"## Fix prior compile/page error\n{compile_error}\n" if compile_error else ""
    )
    critique_block = (
        f"## Prior critique / density feedback\n{critique}\n" if critique else ""
    )
    prev_block = (
        f"## Previous draft\n---\n{prev_draft}\n---\n" if prev_draft.strip() else ""
    )
    style_extra = (
        ""
        if compact
        else f"## Resume Style Guide\n---\n{_load_style_guide()}\n---\n"
    )

    from src.core.prompts import UNTRUSTED_INPUT_GUARDRAIL
    from src.core.security import wrap_untrusted

    return DRAFT_RESUME.format(
        guardrail=UNTRUSTED_INPUT_GUARDRAIL,
        pass_score=int(state.get("ats_pass_score") or ATS_PASS_SCORE_DEFAULT),
        company=state.get("company") or "Unknown",
        track=track,
        track_brief=track_brief(track),
        style_extra=style_extra,
        lessons_block=lessons_block,
        jd=wrap_untrusted("job_description", jd),
        master=master,
        required=required,
        unsupported=unsupported,
        missing_block=missing_block,
        compile_block=compile_block,
        critique_block=critique_block,
        prev_block=prev_block,
    )


def draft_node(state: State) -> dict:
    """LLM drafts dense RenderCV YAML (Groq-first) with XYZ + density gate."""
    compact = _text_provider_is_groq()
    prompt = _build_draft_prompt(state, compact=compact)

    try:
        response = _draft_invoke(prompt)
    except Exception as exc:
        err = str(exc).lower()
        # Extra compact retry before giving up (TPM / oversized fact bank).
        if compact and ("413" in err or "rate_limit" in err or "tokens" in err):
            logger.warning(
                "Draft still TPM-bound after failover — compact retry with %s",
                DEFAULT_GROQ_FALLBACK_MODEL,
            )
            state_retry = dict(state)
            state_retry["master_yaml"] = _clip_head(
                state.get("master_yaml") or "", 4500
            )
            state_retry["current_tailored_yaml"] = _clip_head(
                state.get("current_tailored_yaml") or "", 1500
            )
            prompt = _build_draft_prompt(state_retry, compact=True)
            response = _draft_invoke(prompt, model=DEFAULT_GROQ_FALLBACK_MODEL)
        else:
            raise

    raw_text = message_text(getattr(response, "content", response))
    try:
        tailored = _extract_yaml(raw_text)
    except ValueError as parse_exc:
        # One repair pass — common when gpt-oss truncates a fence.
        logger.warning(
            "Draft YAML parse failed (%s) — requesting repair; raw_head=%r",
            parse_exc,
            raw_text[:240],
        )
        repair = (
            "Your previous reply was not valid RenderCV YAML. "
            "Reply with ONLY a fenced ```yaml block (no prose). "
            "Must include cv.name and cv.sections with experience (2 roles × 3 "
            "highlights) and projects (2 × 3 highlights).\n\nBroken output:\n"
            + _clip_head(raw_text, 3500)
        )
        response = _draft_invoke(repair)
        tailored = _extract_yaml(message_text(getattr(response, "content", response)))
    tailored = _enforce_one_page_budget(
        tailored,
        jd=state.get("jd") or "",
        master_yaml=state.get("master_yaml") or "",
    )
    try:
        data = _unwrap_rendercv_root(yaml.safe_load(tailored) or {})
        if isinstance(data, dict):
            _strip_hallucinated_highlights(data, state.get("master_yaml") or "")
            _reorder_sections(data)
            tailored = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except Exception:
        pass
    preset = _floor_preset(_select_design_preset(tailored))
    tailored = _apply_design(tailored, preset)
    issues = _bullet_density_issues(tailored)
    density_ok = not issues
    feedback = (state.get("critique_feedback") or "").strip()
    if issues:
        density_msg = (
            "DENSITY FAIL — expand using ONLY fact-bank tools/metrics: "
            + "; ".join(issues)
        )
        feedback = f"{feedback}\n{density_msg}".strip()
        logger.warning(density_msg)

    return {
        "current_tailored_yaml": tailored,
        "compile_error": "",
        "revision_count": int(state.get("revision_count") or 0) + 1,
        "density_ok": density_ok,
        "design_preset": preset,
        "critique_feedback": feedback,
    }


def critique_node(state: State) -> dict:
    """Rigid classical ATS keyword score — no LLM judging."""
    required = state.get("required_keywords") or []
    pass_score = int(state.get("ats_pass_score") or ATS_PASS_SCORE_DEFAULT)
    result = calculate_ats_score(state.get("current_tailored_yaml") or "", required)
    score = int(result["score"])
    missing = list(result["missing_keywords"])
    matched = list(result.get("matched_keywords") or [])
    company = state.get("company") or "Unknown"
    feedback = (
        f"NLP ATS coverage={score}% (need ≥{pass_score}% for {company}; "
        f"reason={state.get('ats_pass_reason') or 'default'}). "
        f"Matched={matched}. "
        f"Rewrite bullets with XYZ formula to include missing exact phrases: {missing}"
    )
    logger.info(
        "Rigid ATS critique company=%s score=%s pass=%s missing=%s",
        company,
        score,
        pass_score,
        missing,
    )
    return {
        "ats_score": score,
        "missing_keywords": missing,
        "critique_feedback": feedback,
    }


def _ensure_ace_theme_beside_yaml(yaml_path: Path) -> None:
    """RenderCV loads custom themes from the YAML's parent folder."""
    import shutil

    src = THEMES_DIR / "ace"
    if not src.is_dir():
        logger.warning("Custom theme missing at %s", src)
        return
    dest = Path(yaml_path).resolve().parent / "ace"
    if dest.resolve() == src.resolve():
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _run_rendercv(yaml_path: Path, company_dir: Path) -> tuple[int, str]:
    yaml_path = Path(yaml_path).expanduser().resolve()
    company_dir = Path(company_dir).expanduser().resolve()
    _ensure_ace_theme_beside_yaml(yaml_path)
    relative_yaml = yaml_path.relative_to(REPO_ROOT).as_posix()
    relative_out = company_dir.relative_to(REPO_ROOT).as_posix()
    try:
        completed = subprocess.run(
            [
                "rendercv",
                "render",
                relative_yaml,
                "--output-folder-name",
                relative_out,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO_ROOT),
        )
    except OSError as exc:
        logger.exception("Failed to launch RenderCV")
        return 1, f"Failed to launch RenderCV: {exc}"
    err = (completed.stderr or completed.stdout or "").strip()
    return completed.returncode, err


def _output_root() -> Path:
    try:
        from src.core.settings import get_settings

        return Path(get_settings().output_dir)
    except Exception:
        return OUTPUT_DIR


def compile_node(state: State) -> dict:
    """design_tuner → RenderCV; escalate denser presets before LLM rewrite."""
    jd = state.get("jd") or ""
    master = state.get("master_yaml") or ""
    tailored = _enforce_one_page_budget(
        state["current_tailored_yaml"], jd=jd, master_yaml=master
    )
    preset = _floor_preset(
        state.get("design_preset") or _select_design_preset(tailored)
    )
    tailored = _apply_design(tailored, preset)

    company = _safe_dir_name(state.get("company") or "Unknown")
    company_dir = _output_root() / company
    yaml_path = company_dir / "tailored.yaml"
    company_dir.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(tailored, encoding="utf-8")

    code, err = _run_rendercv(yaml_path, company_dir)
    if code != 0:
        logger.warning("RenderCV failed — routing back to draft:\n%s", err[-2000:])
        return {
            "compile_error": err[-4000:] or "unknown RenderCV error",
            "yaml_path": str(yaml_path),
            "pdf_path": "",
            "current_tailored_yaml": tailored,
            "design_preset": preset,
            "critique_feedback": (
                (state.get("critique_feedback") or "")
                + "\nRenderCV rejected the YAML. Fix syntax/schema strictly."
            ).strip(),
        }

    pdf_path = _find_pdf(
        tailored,
        search_dirs=[company_dir, REPO_ROOT / "rendercv_output"],
    )
    if not pdf_path.is_file():
        return {
            "compile_error": f"RenderCV exited 0 but PDF missing under {company_dir}",
            "yaml_path": str(yaml_path),
            "pdf_path": "",
            "current_tailored_yaml": tailored,
            "design_preset": preset,
        }

    pages = _pdf_page_count(pdf_path)
    revisions = int(state.get("revision_count") or 0)

    # Page overflow: escalate denser presets without another LLM call.
    while pages > 1:
        denser = _next_denser_preset(preset)
        if denser is None:
            break
        logger.warning(
            "PDF is %s pages — escalating design preset %s → %s",
            pages,
            preset,
            denser,
        )
        preset = denser
        tailored = _drop_precollege_education(tailored)
        try:
            data = _unwrap_rendercv_root(yaml.safe_load(tailored) or {})
            sections = ((data.get("cv") or {}).get("sections") or {})
            if isinstance(sections, dict) and isinstance(data.get("cv"), dict):
                sections.pop("certificates", None)
                data["cv"]["sections"] = sections
                tailored = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        except Exception:
            pass
        tailored = _apply_design(tailored, preset)
        yaml_path.write_text(tailored, encoding="utf-8")
        code, err = _run_rendercv(yaml_path, company_dir)
        if code != 0:
            return {
                "compile_error": (
                    f"PDF was {pages} pages; dense re-render failed: {err[-3000:]}"
                ),
                "yaml_path": str(yaml_path),
                "pdf_path": "",
                "current_tailored_yaml": tailored,
                "design_preset": preset,
                "critique_feedback": (
                    (state.get("critique_feedback") or "")
                    + "\nDraft exceeded 1 page. Prefer 2×3 XYZ bullets; "
                    "shorten each bullet to ≤120 chars; CGPA-only education."
                ).strip(),
            }
        pdf_path = _find_pdf(
            tailored,
            search_dirs=[company_dir, REPO_ROOT / "rendercv_output"],
        )
        pages = _pdf_page_count(pdf_path) if pdf_path.is_file() else -1

    if pages > 1 and revisions < MAX_DRAFT_REVISIONS:
        return {
            "compile_error": (
                f"PDF still {pages} pages after ultra_dense preset. "
                "Prefer 2×3 bullets ≤120 chars; drop weakest project if needed."
            ),
            "yaml_path": str(yaml_path),
            "pdf_path": "",
            "current_tailored_yaml": tailored,
            "design_preset": preset,
            "critique_feedback": (
                (state.get("critique_feedback") or "")
                + "\nStill over 1 page. Keep 2 roles × 3 bullets and 2 projects "
                "× 3 bullets; shorten each bullet; CGPA-only education."
            ).strip(),
        }

    if pages > 1:
        logger.warning(
            "Page budget exhausted (revisions=%s); progressive hard-shrink",
            revisions,
        )
        for level, shrink_preset in (
            (1, "dense"),
            (2, "ultra_dense"),
            (3, "ultra_dense"),
        ):
            hard = _hard_shrink_yaml(
                tailored, preset=shrink_preset, jd=jd, level=level
            )
            yaml_path.write_text(hard, encoding="utf-8")
            code, err = _run_rendercv(yaml_path, company_dir)
            if code != 0:
                logger.warning(
                    "Hard-shrink level=%s RenderCV failed: %s", level, err[-500:]
                )
                continue
            pdf_path = _find_pdf(
                hard,
                search_dirs=[company_dir, REPO_ROOT / "rendercv_output"],
            )
            tailored = hard
            preset = shrink_preset
            pages = _pdf_page_count(pdf_path) if pdf_path.is_file() else pages
            if pages == 1:
                break
        if pages != 1:
            # Fail closed — never ship a multi-page fresher resume.
            if pdf_path.is_file() and pages > 1:
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
            return {
                "compile_error": (
                    f"PDF still {pages} pages after hard shrink. "
                    "Cut to 2 experience × 3 bullets and 1–2 projects × 3 bullets; "
                    "education with NO bullets except optional CGPA: X.X/10."
                ),
                "yaml_path": str(yaml_path),
                "pdf_path": "",
                "current_tailored_yaml": tailored,
                "design_preset": preset,
                "critique_feedback": (
                    (state.get("critique_feedback") or "")
                    + "\nONE-PAGE FAIL: CGPA-only education; 2 roles; "
                    "drop to 1 project if needed; ≤120-char bullets."
                ).strip(),
            }

    # Strict ship gate — never return a multi-page path.
    if pages != 1:
        return {
            "compile_error": f"Strict 1-page gate failed (pages={pages}).",
            "yaml_path": str(yaml_path),
            "pdf_path": "",
            "current_tailored_yaml": tailored,
            "design_preset": preset,
        }

    # Sparse 1-page: one optional expand pass if volume is still thin.
    prior_fb = state.get("critique_feedback") or ""
    if (
        pages == 1
        and preset == "airy"
        and revisions < MAX_DRAFT_REVISIONS
        and "SPARSE LAYOUT" not in prior_fb
    ):
        try:
            bullets = _count_highlights(yaml.safe_load(tailored) or {})
        except Exception:
            bullets = 99
        if bullets < 10:
            return {
                "compile_error": (
                    "Page fits but content looks sparse. Deepen XYZ bullets "
                    "with fact-bank stack detail (keep ≤120 chars)."
                ),
                "yaml_path": str(yaml_path),
                "pdf_path": "",
                "current_tailored_yaml": tailored,
                "design_preset": preset,
                "critique_feedback": (
                    prior_fb
                    + "\nSPARSE LAYOUT: expand to 3 XYZ bullets per entry using "
                    "fact-bank technical depth (no invented metrics)."
                ).strip(),
            }

    try:
        import json

        report = {
            "company": company,
            "job_track": state.get("job_track") or "",
            "ats_score": int(state.get("ats_score") or 0),
            "ats_pass_score": int(state.get("ats_pass_score") or ATS_PASS_SCORE_DEFAULT),
            "ats_pass_reason": state.get("ats_pass_reason") or "",
            "required_keywords": state.get("required_keywords") or [],
            "missing_keywords": state.get("missing_keywords") or [],
            "unsupported_keywords": state.get("unsupported_keywords") or [],
            "pdf_path": str(pdf_path.resolve()),
            "pages": pages,
            "design_preset": preset,
            "density_ok": bool(state.get("density_ok", True)),
        }
        (company_dir / "ats_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write ats_report.json: %s", exc)

    logger.info(
        "Compiled PDF → %s (pages=%s preset=%s ats=%s/%s)",
        pdf_path,
        pages,
        preset,
        state.get("ats_score"),
        state.get("ats_pass_score"),
    )
    return {
        "compile_error": "",
        "yaml_path": str(yaml_path),
        "pdf_path": str(pdf_path.resolve()),
        "current_tailored_yaml": tailored,
        "design_preset": preset,
    }


def route_after_critique(state: State) -> Literal["draft_node", "compile_node"]:
    pass_score = int(state.get("ats_pass_score") or ATS_PASS_SCORE_DEFAULT)
    revisions = int(state.get("revision_count") or 0)
    if not state.get("density_ok", True) and revisions < MAX_DRAFT_REVISIONS:
        return "draft_node"
    if int(state.get("ats_score") or 0) < pass_score and revisions < MAX_DRAFT_REVISIONS:
        return "draft_node"
    return "compile_node"


def route_after_compile(state: State) -> Literal["draft_node", "__end__"]:
    if (state.get("compile_error") or "").strip():
        if int(state.get("revision_count") or 0) < MAX_DRAFT_REVISIONS:
            return "draft_node"
        logger.warning(
            "Stopping compile retries after %s revisions (last error=%s)",
            state.get("revision_count"),
            (state.get("compile_error") or "")[:200],
        )
    return "__end__"


def build_tailor_graph():
    graph = StateGraph(State)
    graph.add_node("extract_node", extract_node)
    graph.add_node("draft_node", draft_node)
    graph.add_node("critique_node", critique_node)
    graph.add_node("compile_node", compile_node)

    graph.add_edge(START, "extract_node")
    graph.add_edge("extract_node", "draft_node")
    graph.add_edge("draft_node", "critique_node")
    graph.add_conditional_edges(
        "critique_node",
        route_after_critique,
        {
            "draft_node": "draft_node",
            "compile_node": "compile_node",
        },
    )
    graph.add_conditional_edges(
        "compile_node",
        route_after_compile,
        {
            "draft_node": "draft_node",
            "__end__": END,
        },
    )
    return graph.compile()


_TAILOR_GRAPH = None


def get_tailor_graph():
    global _TAILOR_GRAPH
    if _TAILOR_GRAPH is None:
        _TAILOR_GRAPH = build_tailor_graph()
    return _TAILOR_GRAPH


def tailor_with_graph(
    jd: str,
    master_yaml: str,
    *,
    company: str = "Unknown",
    job_track: str = "",
    job_title: str = "",
) -> tuple[str, str]:
    """Hybrid extract → draft → NLP critique (≥85) → compile.

    Returns ``(pdf_path, tailored_yaml_text)``.
    """
    hit = lookup_archetype_pdf(job_track, job_title)
    if hit is not None:
        logger.info(
            "Archetype PDF cache hit (%s|%s) — skip LLM/RenderCV",
            (job_track or "GENERAL").upper(),
            normalize_role_title(job_title),
        )
        return hit

    result = get_tailor_graph().invoke(
        {
            "jd": jd,
            "master_yaml": master_yaml,
            "company": company or "Unknown",
            "job_track": (job_track or "").strip().upper(),
            "required_keywords": [],
            "unsupported_keywords": [],
            "missing_keywords": [],
            "current_tailored_yaml": "",
            "ats_score": 0,
            "ats_pass_score": ATS_PASS_SCORE_DEFAULT,
            "ats_pass_reason": "pending_extract",
            "critique_feedback": "",
            "compile_error": "",
            "yaml_path": "",
            "pdf_path": "",
            "revision_count": 0,
            "density_ok": True,
            "design_preset": _START_PRESET,
        },
        {"recursion_limit": 40},
    )
    pdf_path = (result.get("pdf_path") or "").strip()
    if not pdf_path or not Path(pdf_path).is_file():
        raise FileNotFoundError(
            "Tailor graph finished without a PDF "
            f"(last ATS score={result.get('ats_score')}/"
            f"{result.get('ats_pass_score')}, "
            f"missing={result.get('missing_keywords')}, "
            f"compile_error={result.get('compile_error')!r})"
        )
    yaml_text = (result.get("current_tailored_yaml") or "").strip()
    resolved = str(Path(pdf_path).resolve())
    store_archetype_pdf(job_track, job_title, resolved, yaml_text)
    return resolved, yaml_text


def optimize_resume(
    jd_text: str,
    master_yaml_path: str | Path,
    *,
    company: str = "Unknown",
    job_track: str = "",
    job_title: str = "",
) -> tuple[str, str]:
    """Load master YAML then run the hybrid graph. Returns (pdf_path, yaml)."""
    master_yaml = Path(master_yaml_path).read_text(encoding="utf-8")
    return tailor_with_graph(
        jd_text,
        master_yaml,
        company=company,
        job_track=job_track,
        job_title=job_title,
    )
