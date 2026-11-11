"""Daemon hygiene: orphan reclaim, log rotation, path smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _seed_claimed(db: Path, *, url: str, error_log: str = "") -> int:
    from src.core.queue import _now, init_queue
    from src.core.memory import _connect

    init_queue(db)
    ts = _now()
    with _connect(db) as conn:
        conn.execute(
            """
            INSERT INTO apply_queue (
                company, job_url, track, jd_text, pdf_path, yaml_snapshot,
                status, worker_id, error_log, created_at, updated_at
            )
            VALUES (?, ?, 'sde', '', 'r.pdf', '', 'CLAIMED', 'w1', ?, ?, ?)
            """,
            ("Acme", url, error_log, ts, ts),
        )
        row = conn.execute(
            "SELECT id FROM apply_queue WHERE job_url = ?", (url,)
        ).fetchone()
        conn.commit()
    return int(row["id"])


def test_reclaim_orphaned_claims_retries_then_abandons(tmp_path, monkeypatch):
    from src.core import queue as q
    from src.core.memory import application_counts, init_db

    db = tmp_path / "mem.db"
    monkeypatch.setenv("ACE_CLAIM_RECLAIM_MAX", "2")
    init_db(db)

    url = "https://jobs.example.com/x"
    _seed_claimed(db, url=url)

    s1 = q.reclaim_orphaned_claims(db, max_reclaims=2)
    assert s1 == {"retried": 1, "abandoned": 0}
    with q._connect(db) as conn:
        row = conn.execute(
            "SELECT status, error_log FROM apply_queue WHERE job_url=?", (url,)
        ).fetchone()
    assert row["status"] == "READY"
    assert "reclaim=1" in (row["error_log"] or "")

    # Simulate crash again: mark CLAIMED with same error_log
    with q._connect(db) as conn:
        conn.execute(
            "UPDATE apply_queue SET status='CLAIMED', worker_id='w1' WHERE job_url=?",
            (url,),
        )
        conn.commit()
    s2 = q.reclaim_orphaned_claims(db, max_reclaims=2)
    assert s2["retried"] == 1

    with q._connect(db) as conn:
        row = conn.execute(
            "SELECT status, error_log FROM apply_queue WHERE job_url=?", (url,)
        ).fetchone()
        conn.execute(
            "UPDATE apply_queue SET status='CLAIMED', worker_id='w1' WHERE job_url=?",
            (url,),
        )
        conn.commit()
    assert "reclaim=2" in (row["error_log"] or "")

    s3 = q.reclaim_orphaned_claims(db, max_reclaims=2)
    assert s3 == {"retried": 0, "abandoned": 1}
    with q._connect(db) as conn:
        row = conn.execute(
            "SELECT status FROM apply_queue WHERE job_url=?", (url,)
        ).fetchone()
    assert row["status"] == "DONE"
    counts = application_counts(db)
    assert counts.get("failed_interrupted", 0) == 1


def test_telemetry_rotation(tmp_path, monkeypatch):
    from src.core import telemetry as tel
    from src.core.logutil import rotate_if_oversized

    path = tmp_path / "telemetry.jsonl"
    path.write_text("x" * 5000, encoding="utf-8")
    assert rotate_if_oversized(path, max_bytes=1000) is True
    assert not path.exists()
    assert (tmp_path / "telemetry.jsonl.1").is_file()

    monkeypatch.setattr(tel, "_TELEMETRY_FILE", path)
    monkeypatch.setattr(tel, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setenv("ACE_TELEMETRY_MAX_BYTES", "1000000")
    with tel.time_stage("playbook_execution_ms", kind="test"):
        pass
    assert path.is_file()
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert json.loads(line)["stage"] == "playbook_execution_ms"

def test_project_root_and_memory_db():
    from src.core.config import MEMORY_DB, PROJECT_ROOT, ensure_runtime_dirs
    from src.core.memory import init_db
    from src.core.settings import get_settings, reload_settings

    ensure_runtime_dirs()
    assert PROJECT_ROOT == REPO
    assert MEMORY_DB == REPO / "data" / "ace_memory.db"
    assert MEMORY_DB.is_absolute()

    reload_settings()
    settings = get_settings()
    assert settings.memory_db.is_absolute()
    assert settings.memory_db == MEMORY_DB or settings.memory_db.name == "ace_memory.db"

    path = init_db()
    assert path.is_absolute()
    assert path.exists()


if __name__ == "__main__":
    test_project_root_and_memory_db()
    print("ok", REPO / "data" / "ace_memory.db")
