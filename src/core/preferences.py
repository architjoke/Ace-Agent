"""Dynamic Preference Engine — HARD dealbreakers vs SOFT fluid preferences.

Storage lives in data/lessons_learned.md as machine-parseable ACE_PREF markers.
SOFT rules older than 30 days without recent re-affirmation are archived by
``prune_decayed_preferences`` (daemon evening reflect).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.core.config import DATA_DIR
from src.core.reflect import _ensure_lessons_file
from src.core.security import validate_safe_path

logger = logging.getLogger(__name__)

ENFORCEMENT_LEVELS = frozenset({"HARD", "SOFT"})
SOFT_PENALTY_POINTS = 15
SOFT_TTL_DAYS = 30

_MARKER_RE = re.compile(
    r"<!--\s*ACE_PREF\s+"
    r'id="(?P<id>[^"]+)"\s+'
    r'level="(?P<level>HARD|SOFT)"\s+'
    r'category="(?P<category>[^"]+)"\s+'
    r'created="(?P<created>[^"]+)"\s+'
    r'affirmed="(?P<affirmed>[^"]+)"\s*'
    r"-->\s*\n"
    r"-\s*\*\*\[(?P=category)\|(?P=level)\]\*\*\s*(?P<text>.+?)(?=\n<!--\s*ACE_PREF|\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Looser line-oriented parse (more robust to minor formatting drift)
_LINE_MARKER_RE = re.compile(
    r'<!--\s*ACE_PREF\s+id="(?P<id>[^"]+)"\s+level="(?P<level>HARD|SOFT)"\s+'
    r'category="(?P<category>[^"]+)"\s+created="(?P<created>[^"]+)"\s+'
    r'affirmed="(?P<affirmed>[^"]+)"\s*-->',
    re.I,
)
_RULE_BODY_RE = re.compile(
    r"-\s*\*\*\[(?P<cat>[^\]|]+)\|(?P<lvl>HARD|SOFT)\]\*\*\s*(?P<text>.+)",
    re.I,
)

_STOP = frozenset(
    {
        "a", "an", "the", "to", "of", "and", "or", "for", "in", "on", "at", "by",
        "with", "from", "jobs", "job", "exclude", "avoid", "never", "do", "not",
        "dont", "don't", "apply", "mentioning", "requiring", "that", "this",
        "prefer", "please", "rule", "soft", "hard",
    }
)


@dataclass
class PreferenceRule:
    id: str
    category: str
    rule_text: str
    enforcement_level: str  # HARD | SOFT
    created: str  # ISO
    affirmed: str  # ISO

    @property
    def level(self) -> str:
        return self.enforcement_level.upper()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat((ts or "").strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def load_preference_rules(path: Path | None = None) -> list[PreferenceRule]:
    """Parse active ACE_PREF rules from lessons_learned.md."""
    p = path or _ensure_lessons_file()
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    # Drop archived section from active parse
    active = text.split("## Archived Soft Preferences")[0]
    rules: list[PreferenceRule] = []
    lines = active.splitlines()
    i = 0
    while i < len(lines):
        m = _LINE_MARKER_RE.search(lines[i])
        if not m:
            i += 1
            continue
        body = ""
        if i + 1 < len(lines):
            bm = _RULE_BODY_RE.match(lines[i + 1].strip())
            if bm:
                body = bm.group("text").strip()
                i += 2
            else:
                i += 1
                continue
        else:
            i += 1
            continue
        level = m.group("level").upper()
        if level not in ENFORCEMENT_LEVELS:
            continue
        rules.append(
            PreferenceRule(
                id=m.group("id"),
                category=m.group("category").strip().upper().replace(" ", "_"),
                rule_text=body,
                enforcement_level=level,
                created=m.group("created"),
                affirmed=m.group("affirmed"),
            )
        )
    return rules


def memorize_preference(
    category: str,
    rule_text: str,
    enforcement_level: str,
) -> PreferenceRule:
    """Append or re-affirm a preference rule. Returns the stored rule."""
    level = (enforcement_level or "").strip().upper()
    if level not in ENFORCEMENT_LEVELS:
        raise ValueError("enforcement_level must be HARD or SOFT")
    cat = re.sub(r"[^A-Z0-9_]", "_", (category or "GENERAL").strip().upper())
    cat = re.sub(r"_+", "_", cat).strip("_") or "GENERAL"
    text = (rule_text or "").strip()
    if not text:
        raise ValueError("rule_text is empty")
    if len(text) > 500:
        text = text[:500].rstrip() + "…"

    path = validate_safe_path(_ensure_lessons_file())
    existing = load_preference_rules(path)
    now = _now_iso()
    # Re-affirm duplicate (same category+text+level)
    for rule in existing:
        if (
            rule.category == cat
            and rule.level == level
            and rule.rule_text.lower() == text.lower()
        ):
            _rewrite_affirmed(path, rule.id, now)
            rule.affirmed = now
            logger.info("Re-affirmed preference %s (%s)", rule.id, level)
            return rule

    rid = uuid.uuid4().hex[:10]
    block = (
        f"\n<!-- ACE_PREF id=\"{rid}\" level=\"{level}\" category=\"{cat}\" "
        f"created=\"{now}\" affirmed=\"{now}\" -->\n"
        f"- **[{cat}|{level}]** {text}\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)
    logger.info("Memorized %s preference [%s] %s", level, cat, text[:80])
    return PreferenceRule(rid, cat, text, level, now, now)


def _rewrite_affirmed(path: Path, rule_id: str, affirmed_iso: str) -> None:
    path = validate_safe_path(path)
    raw = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(<!--\s*ACE_PREF\s+id="{re.escape(rule_id)}"[^>]*?\s)affirmed="[^"]+"',
        re.I,
    )
    updated, n = pattern.subn(rf'\1affirmed="{affirmed_iso}"', raw, count=1)
    if n:
        path.write_text(updated, encoding="utf-8")


def _archive_path() -> Path:
    return validate_safe_path(DATA_DIR / "lessons_archived.md")


def prune_decayed_preferences(ttl_days: int = SOFT_TTL_DAYS) -> str:
    """Archive SOFT rules whose last affirm is older than ``ttl_days``.

    HARD rules are never auto-pruned. Called from evening daemon reflect.
    """
    path = _ensure_lessons_file()
    rules = load_preference_rules(path)
    if not rules:
        return "No preference rules to prune."

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(ttl_days)))
    keep: list[PreferenceRule] = []
    archive: list[PreferenceRule] = []
    for rule in rules:
        if rule.level == "HARD":
            keep.append(rule)
            continue
        affirmed = _parse_iso(rule.affirmed) or _parse_iso(rule.created)
        if affirmed is None or affirmed < cutoff:
            archive.append(rule)
        else:
            keep.append(rule)

    if not archive:
        return f"No SOFT preferences older than {ttl_days}d — nothing pruned."

    # Rebuild active file: preserve non-ACE_PREF content, rewrite active rules
    original = path.read_text(encoding="utf-8")
    head = original.split("<!-- ACE_PREF", 1)[0].rstrip()
    # Strip any prior Archived Soft Preferences section from head
    head = re.split(r"\n## Archived Soft Preferences\b", head, maxsplit=1)[0].rstrip()

    parts = [head, ""]
    for rule in keep:
        parts.append(
            f'<!-- ACE_PREF id="{rule.id}" level="{rule.level}" '
            f'category="{rule.category}" created="{rule.created}" '
            f'affirmed="{rule.affirmed}" -->\n'
            f"- **[{rule.category}|{rule.level}]** {rule.rule_text}"
        )
    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")

    arch = _archive_path()
    arch.parent.mkdir(parents=True, exist_ok=True)
    stamp = _now_iso()
    lines = [f"\n## Soft prune {stamp}\n"]
    for rule in archive:
        lines.append(
            f"- [{rule.category}|SOFT] (id={rule.id}, affirmed={rule.affirmed}) "
            f"{rule.rule_text}"
        )
    with arch.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    msg = (
        f"Archived {len(archive)} SOFT preference(s) older than {ttl_days}d → {arch}"
    )
    logger.info(msg)
    return msg


def rule_matches_jd(rule_text: str, jd_text: str) -> bool:
    """Heuristic: does this preference apply to the JD?"""
    jd = (jd_text or "").lower()
    rule = (rule_text or "").lower()
    if not jd.strip() or not rule.strip():
        return False

    for q in re.findall(r"[\"']([^\"']{3,})[\"']", rule):
        if q.lower() in jd:
            return True

    for pat in (
        r"mentioning\s+(.+?)(?:\.|;|$)",
        r"requiring\s+(.+?)(?:\.|;|$)",
        r"about\s+(.+?)(?:\.|;|$)",
        r"involving\s+(.+?)(?:\.|;|$)",
        r"exclude\s+jobs?\s+(?:that\s+)?(?:mention|require|involve|with)\s+(.+?)(?:\.|;|$)",
    ):
        m = re.search(pat, rule)
        if m:
            needle = m.group(1).strip().strip("\"'")
            # Skip if this capture is only an "A" or "B" list of quotes (handled above)
            if '"' in needle or "'" in needle:
                continue
            if len(needle) >= 3 and needle in jd:
                return True

    # Token fallback ignores quoted phrases so "training fee" does not
    # fire on every JD that merely says "training".
    rule_unquoted = re.sub(r"[\"'][^\"']{3,}[\"']", " ", rule)
    tokens = [
        t
        for t in re.findall(r"[a-z0-9][a-z0-9+\-]{2,}", rule_unquoted)
        if t not in _STOP
    ]
    # Prefer distinctive multi-token phrases (whatsapp + community)
    if len(tokens) >= 2:
        hits = sum(1 for t in tokens if t in jd)
        if hits >= 2:
            return True
    for t in tokens:
        if len(t) >= 6 and t in jd:
            return True
    return False


def apply_preference_gate(jd_text: str) -> tuple[str | None, int, list[str]]:
    """Evaluate HARD/SOFT preferences against a JD.

    Returns ``(hard_reject_reason_or_None, soft_penalty_points, soft_hit_labels)``.
    """
    soft_penalty = 0
    soft_hits: list[str] = []
    for rule in load_preference_rules():
        if not rule_matches_jd(rule.rule_text, jd_text):
            continue
        if rule.level == "HARD":
            reason = (
                f"RULE BROKEN: preference_hard[{rule.category}]. {rule.rule_text}"
            )
            return reason, 0, []
        soft_penalty += SOFT_PENALTY_POINTS
        soft_hits.append(f"{rule.category}: {rule.rule_text[:120]}")
    return None, soft_penalty, soft_hits
