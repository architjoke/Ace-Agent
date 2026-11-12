"""Cold-path: JD minify, archetype PDF cache, queue warm threshold."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_minify_jd_strips_eeo_keeps_requirements():
    from src.core.utils import minify_jd

    jd = """\
About the role
Build APIs in Python.

Requirements
- Python, FastAPI, PostgreSQL
- Distributed systems

Equal Employment Opportunity
We are an equal opportunity employer and do not discriminate.

Benefits
- Health insurance
- Gym membership

Physical Requirements
Must lift 50 pounds.
"""
    out = minify_jd(jd)
    assert "Python" in out and "FastAPI" in out
    assert "equal opportunity" not in out.lower()
    assert "Gym membership" not in out
    assert "lift 50" not in out
    assert "Requirements" in out


def test_minify_jd_safety_valve():
    from src.core.utils import minify_jd

    # Tiny body after aggressive headers → keep original if <40%
    jd = "EEO\n" + ("x\n" * 5) + "Equal Employment Opportunity\n" + ("boilerplate\n" * 40)
    # Mostly EEO boilerplate after first tiny line — still should not nuke skills if any
    skills = "Requirements\n- Rust\n- Go\n"
    mixed = skills + "\nEqual Employment Opportunity\n" + ("n/a\n" * 20)
    out = minify_jd(mixed)
    assert "Rust" in out


def test_minify_benefits_keeps_salary_block():
    from src.core.utils import minify_jd

    jd = """\
Role
Ship features.

Benefits
Base salary $120,000 – $150,000
Equity eligible
"""
    out = minify_jd(jd)
    assert "Base salary" in out or "$120" in out


def test_normalize_role_title_sde_backend():
    from src.core.tailor import normalize_role_title

    a = normalize_role_title("SDE II - Backend")
    assert "software_engineer" in a
    assert "backend" in a


def _stub_jd_keywords(monkeypatch):
    monkeypatch.setattr(
        "src.core.ats_scanner.extract_jd_keywords",
        lambda jd_text, top_k=20: ["Python", "FastAPI"][:top_k],
    )
    monkeypatch.setattr(
        "src.core.ats_scanner.filter_supportable_keywords",
        lambda raw, master: (list(raw), []),
    )
    monkeypatch.setattr(
        "src.core.ats_scanner.resolve_ats_pass_score",
        lambda **k: {"pass_score": 85, "reason": "test"},
    )


def test_artifact_cache_hit_skips_rendercv(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra
    from src.core import tailor as t

    _stub_jd_keywords(monkeypatch)
    pdf = tmp_path / "arch.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    db = tmp_path / "artifacts.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )
    jd = "Requirements\n- Python\n- FastAPI\nBuild APIs."
    master = "skills:\n  - Python\n  - FastAPI\n"
    key, supportable = ra.jd_signature(jd, master, track="SDE", title="SDE II - Backend")
    ra.store_artifact(
        key,
        pdf_path=str(pdf),
        yaml_text="cv: {}\n",
        supportable=supportable,
        track="SDE",
        title="SDE II - Backend",
        db_path=db,
    )

    monkeypatch.setattr(
        t,
        "get_tailor_graph",
        lambda: SimpleNamespace(
            invoke=lambda *a, **k: (_ for _ in ()).throw(AssertionError("graph"))
        ),
    )

    path, yaml_text = t.tailor_with_graph(
        jd,
        master,
        job_track="SDE",
        job_title="SDE II - Backend",
    )
    assert Path(path) == pdf.resolve()
    assert "cv:" in yaml_text


def test_artifact_cache_same_fingerprint_hits(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra
    from src.core import tailor as t

    _stub_jd_keywords(monkeypatch)
    db = tmp_path / "artifacts.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    pdf = tmp_path / "out.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )

    class _G:
        def invoke(self, *_a, **_k):
            return {
                "pdf_path": str(pdf),
                "current_tailored_yaml": "x: 1\n",
                "ats_score": 90,
                "ats_pass_score": 85,
                "missing_keywords": [],
                "compile_error": "",
            }

    monkeypatch.setattr(t, "get_tailor_graph", lambda: _G())
    jd = "Requirements\n- Python\nUnique Title XYZ role."
    master = "skills:\n  - Python\n"
    path, y = t.tailor_with_graph(
        jd, master, job_track="SDE", job_title="Unique Title XYZ"
    )
    assert Path(path) == pdf.resolve()
    assert y.startswith("x:")
    monkeypatch.setattr(
        t,
        "get_tailor_graph",
        lambda: (_ for _ in ()).throw(AssertionError("should cache")),
    )
    path2, _ = t.tailor_with_graph(
        jd, master, job_track="SDE", job_title="Unique Title XYZ"
    )
    assert Path(path2) == pdf.resolve()


def test_wait_queue_warm_threshold(monkeypatch):
    from src.core import pipeline as p

    state = {"ready": 0}
    monkeypatch.setattr(p, "queue_counts", lambda: {"READY": state["ready"]})

    async def _run():
        done = asyncio.Event()

        async def bump():
            await asyncio.sleep(0.05)
            state["ready"] = 5

        asyncio.create_task(bump())
        await asyncio.wait_for(
            p._wait_queue_warm(threshold=5, producers_done=done, poll_sec=0.02),
            timeout=2.0,
        )

    asyncio.run(_run())
    assert state["ready"] >= 5


def test_wait_queue_warm_on_producers_done(monkeypatch):
    from src.core import pipeline as p

    monkeypatch.setattr(p, "queue_counts", lambda: {"READY": 0})

    async def _run():
        done = asyncio.Event()

        async def finish():
            await asyncio.sleep(0.05)
            done.set()

        asyncio.create_task(finish())
        await asyncio.wait_for(
            p._wait_queue_warm(threshold=5, producers_done=done, poll_sec=0.02),
            timeout=2.0,
        )
        assert done.is_set()

    asyncio.run(_run())
