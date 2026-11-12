"""Optional overrides from human takeovers (local — not the primary form brain).

Primary answers come from ``data/master.yaml`` via ``FormFillEngine``.
This file stores **kind/intent overrides** after a human teaches an unknown
field. Keep it gitignored; never commit personal answers.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.core.config import DATA_DIR

logger = logging.getLogger(__name__)

LEARNED_ANSWERS_PATH = DATA_DIR / "learned_form_answers.yaml"

# mtime-cached YAML parse — lookups hit every form field; avoid re-reading disk.
_cache_path: str | None = None
_cache_mtime_ns: int | None = None
_cache_data: dict[str, Any] | None = None

# Map common screening labels → canonical application_answers keys.
_LABEL_KEY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"notice\s*period", re.I), "notice_period_days"),
    (re.compile(r"current\s*(ctc|salary|compensation)", re.I), "current_ctc_inr"),
    (re.compile(r"expected\s*(ctc|salary|compensation)|salary\s*expectation", re.I), "expected_ctc_inr"),
    (re.compile(r"years?\s*(of\s*)?experience", re.I), "years_of_experience"),
    (re.compile(r"(additional\s*)?months?\s*(of\s*)?experience", re.I), "additional_months_of_experience"),
    (re.compile(r"linkedin(\s*url|\s*profile)?$", re.I), "linkedin"),
    (re.compile(r"portfolio|personal\s*website|online\s*portfolio", re.I), "portfolio_url"),
    (re.compile(r"github(\s*url|\s*profile)?$", re.I), "github"),
    (re.compile(r"phone|mobile|cell", re.I), "phone"),
    (re.compile(r"e-?mail", re.I), "email"),
    (re.compile(r"first\s*name", re.I), "first_name"),
    (re.compile(r"last\s*name|surname|family\s*name", re.I), "last_name"),
    (re.compile(r"full\s*name|^name$", re.I), "full_name"),
    (re.compile(r"city", re.I), "city"),
    (re.compile(r"\bstate\b|province|region", re.I), "state"),
    (re.compile(r"\bcountry\b", re.I), "country"),
    (re.compile(r"pin\s*code|postal|zip", re.I), "pincode"),
    (re.compile(r"street\s*address|home\s*address|mailing\s*address|address\s*line|^address$", re.I), "address_line"),
    (re.compile(r"graduation\s*year|year\s*of\s*graduation", re.I), "graduation_year"),
    (re.compile(r"(university|college|degree).*(start|from).*month|start\s*month", re.I), "university_start_month"),
    (re.compile(r"(university|college|degree).*(start|from).*year|start\s*year", re.I), "university_start_year"),
    (re.compile(r"(university|college|degree).*(end|to|grad).*month|end\s*month", re.I), "university_end_month"),
    (re.compile(r"(university|college|degree).*(end|to|grad).*year|end\s*year|graduat", re.I), "university_end_year"),
    (re.compile(r"currently\s*attend|still\s*(enrolled|studying)", re.I), "university_currently_attending"),
)

_SKIP_LABEL = re.compile(
    r"password|otp|captcha|upload|browse|submit|next|continue|review|cancel|close|dismiss",
    re.I,
)


def learned_path() -> Path:
    return LEARNED_ANSWERS_PATH


def normalize_question(label: str) -> str:
    text = (label or "").strip().lower()
    text = re.sub(r"[\s\*\:\?\.\,\;\!\(\)\[\]\{\}\"']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:160]


def infer_canonical_key(label: str) -> str | None:
    blob = (label or "").strip()
    if not blob:
        return None
    for pattern, key in _LABEL_KEY_RULES:
        if pattern.search(blob):
            return key
    return None


def _invalidate_cache() -> None:
    global _cache_path, _cache_mtime_ns, _cache_data
    _cache_path = None
    _cache_mtime_ns = None
    _cache_data = None


def _snapshot(data: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy payload so callers can mutate without corrupting the cache."""
    return {
        "version": data.get("version", 1),
        "entries": list(data.get("entries") or []),
    }


def _load_raw(path: Path | None = None) -> dict[str, Any]:
    global _cache_path, _cache_mtime_ns, _cache_data
    p = path or learned_path()
    try:
        key = str(p.resolve())
    except OSError:
        key = str(p)
    if not p.is_file():
        empty: dict[str, Any] = {"version": 1, "entries": []}
        _cache_path = key
        _cache_mtime_ns = None
        _cache_data = empty
        return _snapshot(empty)
    try:
        mtime_ns = p.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    if (
        _cache_path == key
        and _cache_mtime_ns == mtime_ns
        and _cache_data is not None
    ):
        return _snapshot(_cache_data)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.debug("learned_answers load failed", exc_info=True)
        data = {"entries": []}
    if not isinstance(data, dict):
        data = {"entries": []}
    if not isinstance(data.get("entries"), list):
        data["entries"] = []
    _cache_path = key
    _cache_mtime_ns = mtime_ns
    _cache_data = data
    return _snapshot(data)


def _save_raw(data: dict[str, Any], path: Path | None = None) -> None:
    p = path or learned_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": list(data.get("entries") or []),
    }
    p.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _invalidate_cache()


def list_entries(path: Path | None = None) -> list[dict[str, Any]]:
    data = _load_raw(path)
    out: list[dict[str, Any]] = []
    for item in data.get("entries") or []:
        if isinstance(item, dict) and item.get("answer") not in (None, ""):
            out.append(item)
    return out


def _should_auto_promote(path: Path | None, master_path: Path | None) -> bool:
    """Promote to default master only for the real learned file (not test temps)."""
    if master_path is not None:
        return True
    if path is None:
        return True
    try:
        return path.resolve() == learned_path().resolve()
    except OSError:
        return path == learned_path()


def remember_answer(
    question: str,
    answer: str,
    *,
    company: str = "",
    job_url: str = "",
    key: str | None = None,
    path: Path | None = None,
    master_path: Path | None = None,
    promote: bool = True,
) -> dict[str, Any] | None:
    """Upsert one human-taught answer. Returns the stored entry or None if skipped."""
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return None
    if _SKIP_LABEL.search(q):
        return None
    # Ignore tiny / control-looking values.
    if len(a) > 500:
        a = a[:500]
    norm = normalize_question(q)
    if not norm or len(norm) < 3:
        return None
    canon = (key or "").strip() or infer_canonical_key(q)
    kind = ""
    try:
        from src.core.form_rules import classify_label

        kind = str(classify_label(q).value)
    except Exception:
        kind = ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = _load_raw(path)
    entries: list[dict[str, Any]] = list(data.get("entries") or [])
    updated: dict[str, Any] | None = None
    for item in entries:
        if not isinstance(item, dict):
            continue
        if normalize_question(str(item.get("question_norm") or item.get("question") or "")) == norm:
            item["question"] = q[:200]
            item["question_norm"] = norm
            item["answer"] = a
            if canon:
                item["key"] = canon
            if kind:
                item["kind"] = kind
            if company:
                item["company"] = company
            if job_url:
                item["job_url"] = job_url
            item["updated_at"] = now
            item["hits"] = int(item.get("hits") or 0) + 1
            updated = item
            break
    if updated is None:
        updated = {
            "question": q[:200],
            "question_norm": norm,
            "answer": a,
            "key": canon,
            "kind": kind,
            "company": company or "",
            "job_url": job_url or "",
            "updated_at": now,
            "hits": 1,
        }
        entries.append(updated)
    data["entries"] = entries[-400:]  # hard cap
    _save_raw(data, path)
    logger.info(
        "Learned form answer [%s] %s → %s",
        canon or "freeform",
        norm[:60],
        a[:80],
    )
    if promote and updated and _should_auto_promote(path, master_path):
        try:
            from src.core.profile_promote import promote_entry_to_master

            promote_entry_to_master(updated, master_path=master_path)
        except Exception:
            logger.debug("promote after remember failed", exc_info=True)
    return updated


def remember_many(
    pairs: list[tuple[str, str]],
    *,
    company: str = "",
    job_url: str = "",
    path: Path | None = None,
    master_path: Path | None = None,
    promote: bool = True,
) -> int:
    """Store many label→value pairs. Returns count saved."""
    n = 0
    for question, answer in pairs:
        if remember_answer(
            question,
            answer,
            company=company,
            job_url=job_url,
            path=path,
            master_path=master_path,
            promote=promote,
        ):
            n += 1
    return n


def lookup_answer(question: str, path: Path | None = None) -> str | None:
    norm = normalize_question(question)
    if not norm:
        return None
    for item in list_entries(path):
        if normalize_question(str(item.get("question_norm") or item.get("question") or "")) == norm:
            return str(item.get("answer") or "").strip() or None
        # Soft containment match for long LinkedIn labels.
        other = normalize_question(str(item.get("question") or ""))
        if other and (other in norm or norm in other) and len(other) >= 8:
            return str(item.get("answer") or "").strip() or None
    return None


def lookup_by_kind(kind: str, path: Path | None = None) -> str | None:
    """Latest override stored under a form field kind (e.g. skills_tech)."""
    want = (kind or "").strip()
    if not want:
        return None
    best: str | None = None
    best_ts = ""
    for item in list_entries(path):
        if str(item.get("kind") or "").strip() != want:
            continue
        answer = str(item.get("answer") or "").strip()
        if not answer:
            continue
        ts = str(item.get("updated_at") or "")
        if ts >= best_ts:
            best_ts = ts
            best = answer
    return best


def as_overlay(path: Path | None = None) -> dict[str, str]:
    """Canonical-key overlay + freeform ``learned_q:*`` keys for prompts."""
    out: dict[str, str] = {}
    for item in list_entries(path):
        answer = str(item.get("answer") or "").strip()
        if not answer:
            continue
        key = str(item.get("key") or "").strip()
        if key:
            out[key] = answer
        q = str(item.get("question") or "").strip()
        norm = normalize_question(q)
        if norm:
            out[f"learned_q:{norm}"] = answer
    return out


def merge_into_answers(answers: dict[str, Any] | None, path: Path | None = None) -> dict[str, Any]:
    """Merge learned values into a profile without clobbering non-empty facts."""
    out: dict[str, Any] = dict(answers or {})
    overlay = as_overlay(path)
    for key, value in overlay.items():
        if not value:
            continue
        existing = out.get(key)
        if existing not in (None, ""):
            continue
        out[key] = value
    return out


def learned_qa_prompt_block(path: Path | None = None, *, limit: int = 40) -> str:
    entries = list_entries(path)
    if not entries:
        return ""
    # Prefer recently updated.
    entries = sorted(
        entries,
        key=lambda e: str(e.get("updated_at") or ""),
        reverse=True,
    )[:limit]
    lines = [
        "## LEARNED FROM PRIOR HUMAN FILLS (reuse these — do not ask human again)",
    ]
    for item in entries:
        q = str(item.get("question") or "").strip()
        a = str(item.get("answer") or "").strip()
        if not q or not a:
            continue
        key = str(item.get("key") or "").strip()
        suffix = f" [{key}]" if key else ""
        lines.append(f'- Q: "{q}" → A: {a}{suffix}')
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def append_lesson_from_learning(question: str, answer: str) -> None:
    """Best-effort one-line lesson so reflect/executor prompts also see it."""
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return
    try:
        from src.core.reflect import _ensure_lessons_file
        from src.core.security import validate_safe_path

        path = validate_safe_path(_ensure_lessons_file())
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        line = f"- Human-taught ({stamp}): when asked \"{q[:100]}\", answer: {a[:120]}\n"
        existing = path.read_text(encoding="utf-8")
        needle = f"when asked \"{q[:100]}\""
        if needle.lower() in existing.lower():
            return
        with path.open("a", encoding="utf-8") as fh:
            if not existing.endswith("\n"):
                fh.write("\n")
            fh.write(line)
    except Exception:
        logger.debug("append lesson from learning failed", exc_info=True)
