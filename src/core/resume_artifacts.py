"""JD-fingerprint resume PDF/YAML artifact cache (SQLite).

Title alone never bypasses keyword fit — lookups require a full
``artifact_key`` from :func:`jd_signature`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
_LOCK = threading.Lock()
_ARTIFACTS_PATH = Path(__file__).resolve().parents[2] / "data" / "resume_artifacts.db"


def artifact_ttl_sec() -> float:
    try:
        return max(1.0, float(os.environ.get("ACE_ARCHETYPE_CACHE_HOURS") or "48")) * 3600.0
    except ValueError:
        return 48.0 * 3600.0


def _db_path(db_path: Path | None = None) -> Path:
    return Path(db_path) if db_path else _ARTIFACTS_PATH


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.Error:
        pass
    return conn


def init_resume_artifacts(db_path: Path | None = None) -> Path:
    path = _db_path(db_path)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_artifacts (
                artifact_key TEXT PRIMARY KEY,
                pdf_path TEXT NOT NULL,
                yaml_path_or_text_hash TEXT,
                supportable_json TEXT,
                track TEXT,
                title_norm TEXT,
                created_at REAL NOT NULL,
                ats_score REAL,
                schema_version INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_resume_artifacts_created "
            "ON resume_artifacts(created_at)"
        )
        conn.commit()
    return path


def jd_signature(
    jd: str,
    master_yaml: str,
    *,
    track: str = "",
    title: str = "",
    ats_pass_score: int | None = None,
) -> tuple[str, list[str]]:
    """Compute artifact fingerprint + supportable keywords (cheap path before lookup)."""
    from src.core.ats_scanner import (
        extract_jd_keywords,
        filter_supportable_keywords,
        resolve_ats_pass_score,
    )
    from src.core.utils import minify_jd

    # Local import avoids circular import with tailor at module load.
    from src.core.tailor import ATS_PASS_SCORE_DEFAULT, normalize_role_title

    raw = extract_jd_keywords(jd or "", top_k=20)
    supportable, unsupported = filter_supportable_keywords(raw, master_yaml or "")
    supportable_sorted = sorted(
        {str(s).strip().lower() for s in supportable if str(s).strip()}
    )
    if ats_pass_score is None:
        try:
            threshold = resolve_ats_pass_score(
                company="",
                supportable_keywords=supportable,
                unsupported_keywords=unsupported,
            )
            pass_score = int(threshold.get("pass_score") or ATS_PASS_SCORE_DEFAULT)
        except Exception:
            pass_score = int(ATS_PASS_SCORE_DEFAULT)
    else:
        pass_score = int(ats_pass_score)

    master_bytes = (master_yaml or "").encode("utf-8", errors="replace")
    master_hash = hashlib.sha256(master_bytes).hexdigest()
    title_norm = normalize_role_title(title or "")
    track_norm = (track or "GENERAL").strip().upper() or "GENERAL"

    h = hashlib.sha256()
    h.update(minify_jd(jd or "").encode("utf-8", errors="replace"))
    h.update(b"|")
    h.update(",".join(supportable_sorted).encode("utf-8", errors="replace"))
    h.update(b"|")
    h.update(master_hash.encode("ascii"))
    h.update(b"|")
    h.update(track_norm.encode("utf-8", errors="replace"))
    h.update(b"|")
    h.update(title_norm.encode("utf-8", errors="replace"))
    h.update(b"|")
    h.update(str(pass_score).encode("ascii"))
    h.update(b"|")
    h.update(str(CACHE_SCHEMA_VERSION).encode("ascii"))
    return h.hexdigest(), supportable_sorted


def lookup_artifact(
    artifact_key: str,
    *,
    db_path: Path | None = None,
    now: float | None = None,
) -> tuple[str, str] | None:
    """Return (pdf_path, yaml_text) on fresh TTL hit with PDF on disk."""
    key = (artifact_key or "").strip()
    if not key:
        return None
    init_resume_artifacts(db_path)
    clock = float(now if now is not None else time.time())
    ttl = artifact_ttl_sec()
    with _LOCK:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM resume_artifacts WHERE artifact_key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            try:
                created = float(row["created_at"] or 0.0)
            except (TypeError, ValueError):
                return None
            if clock - created > ttl:
                return None
            if int(row["schema_version"] or 0) != CACHE_SCHEMA_VERSION:
                return None
            pdf = str(row["pdf_path"] or "").strip()
            if not pdf:
                return None
            try:
                from src.core.security import validate_safe_path

                pdf_path = validate_safe_path(pdf)
            except Exception:
                pdf_path = Path(pdf)
            if not pdf_path.is_file():
                try:
                    conn.execute(
                        "DELETE FROM resume_artifacts WHERE artifact_key = ?",
                        (key,),
                    )
                    conn.commit()
                except Exception:
                    logger.debug("resume artifact cleanup failed", exc_info=True)
                return None
            yaml_blob = str(row["yaml_path_or_text_hash"] or "")
            return str(pdf_path.resolve()), yaml_blob


def store_artifact(
    artifact_key: str,
    *,
    pdf_path: str,
    yaml_text: str = "",
    supportable: list[str] | None = None,
    track: str = "",
    title: str = "",
    ats_score: float | None = None,
    db_path: Path | None = None,
    now: float | None = None,
) -> None:
    """Persist a successful tailor artifact under its JD fingerprint key."""
    key = (artifact_key or "").strip()
    if not key or not pdf_path:
        return
    try:
        from src.core.security import validate_safe_path

        resolved = validate_safe_path(pdf_path)
    except Exception:
        resolved = Path(pdf_path)
    if not resolved.is_file():
        return

    from src.core.tailor import normalize_role_title

    init_resume_artifacts(db_path)
    clock = float(now if now is not None else time.time())
    title_norm = normalize_role_title(title or "")
    track_norm = (track or "GENERAL").strip().upper() or "GENERAL"
    supportable_json = json.dumps(list(supportable or []), ensure_ascii=False)
    with _LOCK:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO resume_artifacts (
                    artifact_key, pdf_path, yaml_path_or_text_hash, supportable_json,
                    track, title_norm, created_at, ats_score, schema_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_key) DO UPDATE SET
                    pdf_path=excluded.pdf_path,
                    yaml_path_or_text_hash=excluded.yaml_path_or_text_hash,
                    supportable_json=excluded.supportable_json,
                    track=excluded.track,
                    title_norm=excluded.title_norm,
                    created_at=excluded.created_at,
                    ats_score=excluded.ats_score,
                    schema_version=excluded.schema_version
                """,
                (
                    key,
                    str(resolved.resolve()),
                    yaml_text or "",
                    supportable_json,
                    track_norm,
                    title_norm,
                    clock,
                    float(ats_score) if ats_score is not None else None,
                    CACHE_SCHEMA_VERSION,
                ),
            )
            conn.commit()
