"""Promote stable learned form answers into local ``master.yaml``.

Personal facts stay gitignored. Shared playbooks/engine never get personal data.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.core.config import resolve_master_yaml

logger = logging.getLogger(__name__)

# Fixed identity / location / contact / screening keys safe to mirror into
# application_answers. Freeform employer prose is never promoted.
PROMOTE_KEYS: frozenset[str] = frozenset(
    {
        "full_name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "phone_national",
        "phone_display",
        "phone_e164",
        "phone_country_code",
        "city",
        "state",
        "country",
        "pincode",
        "address_line",
        "linkedin",
        "linkedin_url",
        "github",
        "github_url",
        "portfolio_url",
        "website_url",
        "notice_period_days",
        "current_ctc_inr",
        "expected_ctc_inr",
        "years_of_experience",
        "additional_months_of_experience",
        "university_start_month",
        "university_start_year",
        "university_end_month",
        "university_end_year",
        "university_currently_attending",
        "graduation_year",
    }
)

# Allowlisted keys may fill an *empty* master slot on first teach (hits >= 1).
# Overwriting a conflicting non-empty value requires hits >= 2.
_MIN_HITS_TO_OVERWRITE = 2


def _norm_answer(value: Any) -> str:
    return str(value or "").strip()


def should_promote_entry(entry: dict[str, Any] | None) -> bool:
    """True when the learned entry is eligible for master promotion."""
    if not isinstance(entry, dict):
        return False
    key = str(entry.get("key") or "").strip()
    answer = _norm_answer(entry.get("answer"))
    if not key or key not in PROMOTE_KEYS or not answer:
        return False
    hits = int(entry.get("hits") or 0)
    # Allowlist keys: promote on first hit when filling empty / confirming.
    # hits >= 2 also qualifies (explicit plan rule).
    return hits >= 1


def _load_master(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.debug("master.yaml load failed for promote: %s", path, exc_info=True)
        return {}
    return raw if isinstance(raw, dict) else {}


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def promote_entry_to_master(
    entry: dict[str, Any] | None,
    *,
    master_path: Path | None = None,
) -> bool:
    """Upsert one learned entry into ``application_answers`` when safe.

    Returns True when master.yaml was modified.
    """
    if not should_promote_entry(entry):
        return False
    assert entry is not None
    key = str(entry.get("key") or "").strip()
    answer = _norm_answer(entry.get("answer"))
    hits = int(entry.get("hits") or 0)

    path = master_path or resolve_master_yaml()
    if not path.is_file():
        logger.debug("promote skipped — master.yaml missing: %s", path)
        return False

    data = _load_master(path)
    answers = data.get("application_answers")
    if not isinstance(answers, dict):
        answers = {}
        data["application_answers"] = answers

    existing = _norm_answer(answers.get(key))
    if not existing:
        answers[key] = answer
    elif existing == answer:
        return False
    elif hits >= _MIN_HITS_TO_OVERWRITE:
        # Repeated confirmation of a different value — update master.
        answers[key] = answer
    else:
        logger.debug(
            "promote skipped conflict [%s]: master=%r learned=%r hits=%s",
            key,
            existing[:40],
            answer[:40],
            hits,
        )
        return False

    try:
        _atomic_write_yaml(path, data)
    except Exception:
        logger.warning("promote write failed: %s", path, exc_info=True)
        return False
    logger.info("Promoted form answer → master.yaml [%s] = %s", key, answer[:80])
    return True


def promote_learned_to_master(
    *,
    learned_path: Path | None = None,
    master_path: Path | None = None,
) -> int:
    """Catch-up: promote all eligible learned entries. Returns count written."""
    from src.core.learned_answers import list_entries

    n = 0
    for item in list_entries(learned_path):
        if promote_entry_to_master(item, master_path=master_path):
            n += 1
    return n
