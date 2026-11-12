"""JD-fingerprint resume artifact cache — title alone must never hit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _master_python() -> str:
    return "skills:\n  - Python\n  - FastAPI\n  - PostgreSQL\n  - Kubernetes\n"


def _stub_keyword_extract(monkeypatch):
    """Avoid spaCy/torch in unit tests — keyword lists drive the fingerprint."""

    def _extract(jd_text: str, top_k: int = 20) -> list[str]:
        text = (jd_text or "").lower()
        tokens = []
        for word in (
            "python",
            "fastapi",
            "postgresql",
            "kubernetes",
            "docker",
            "terraform",
            "go",
        ):
            if word in text:
                tokens.append(word.title() if word != "postgresql" else "PostgreSQL")
        return tokens[:top_k]

    def _filter(raw, master_yaml):
        master = (master_yaml or "").lower()
        supportable = [k for k in raw if k.lower() in master or k.lower() in {"python", "fastapi", "postgresql", "kubernetes"}]
        unsupported = [k for k in raw if k not in supportable]
        return supportable, unsupported

    monkeypatch.setattr(
        "src.core.ats_scanner.extract_jd_keywords", _extract
    )
    monkeypatch.setattr(
        "src.core.ats_scanner.filter_supportable_keywords", _filter
    )
    monkeypatch.setattr(
        "src.core.ats_scanner.resolve_ats_pass_score",
        lambda **k: {"pass_score": 85, "reason": "test"},
    )


def test_same_title_different_jd_is_miss(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra

    _stub_keyword_extract(monkeypatch)
    db = tmp_path / "a.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )

    title = "Software Engineer Intern"
    master = _master_python()
    jd_a = (
        "Requirements\n- Python\n- FastAPI\n- PostgreSQL\n"
        "Build REST APIs for payments."
    )
    jd_b = (
        "Requirements\n- Kubernetes\n- Docker\n- Terraform\n"
        "Operate platform clusters."
    )
    key_a, supportable_a = ra.jd_signature(jd_a, master, track="SDE", title=title)
    key_b, _ = ra.jd_signature(jd_b, master, track="SDE", title=title)
    assert key_a != key_b

    ra.store_artifact(
        key_a,
        pdf_path=str(pdf),
        yaml_text="cv: a\n",
        supportable=supportable_a,
        track="SDE",
        title=title,
        db_path=db,
    )
    assert ra.lookup_artifact(key_a, db_path=db) is not None
    assert ra.lookup_artifact(key_b, db_path=db) is None


def test_same_jd_fingerprint_hits(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra

    _stub_keyword_extract(monkeypatch)
    db = tmp_path / "b.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )
    jd = "Requirements\n- Python\n- FastAPI\nShip features."
    master = _master_python()
    key, supportable = ra.jd_signature(jd, master, track="SDE", title="Backend Eng")
    ra.store_artifact(
        key,
        pdf_path=str(pdf),
        yaml_text="cv: hit\n",
        supportable=supportable,
        track="SDE",
        title="Backend Eng",
        db_path=db,
    )
    hit = ra.lookup_artifact(key, db_path=db)
    assert hit is not None
    assert Path(hit[0]) == pdf.resolve()
    assert "hit" in hit[1]


def test_master_yaml_change_misses(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra

    _stub_keyword_extract(monkeypatch)
    db = tmp_path / "c.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )
    jd = "Requirements\n- Python\nBuild things."
    master_a = "skills:\n  - Python\n"
    master_b = "skills:\n  - Python\n  - Go\n"
    key_a, supportable = ra.jd_signature(jd, master_a, track="SDE", title="SDE")
    key_b, _ = ra.jd_signature(jd, master_b, track="SDE", title="SDE")
    assert key_a != key_b
    ra.store_artifact(
        key_a,
        pdf_path=str(pdf),
        yaml_text="y",
        supportable=supportable,
        track="SDE",
        title="SDE",
        db_path=db,
    )
    assert ra.lookup_artifact(key_b, db_path=db) is None


def test_expired_ttl_and_missing_pdf(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra

    _stub_keyword_extract(monkeypatch)
    db = tmp_path / "d.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    monkeypatch.setenv("ACE_ARCHETYPE_CACHE_HOURS", "1")
    pdf = tmp_path / "gone.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )
    jd = "Requirements\n- Python\n"
    master = _master_python()
    key, supportable = ra.jd_signature(jd, master, track="AI", title="AI Eng")
    now = time.time()
    ra.store_artifact(
        key,
        pdf_path=str(pdf),
        yaml_text="old",
        supportable=supportable,
        track="AI",
        title="AI Eng",
        db_path=db,
        now=now - 7200,  # 2h ago, TTL=1h
    )
    assert ra.lookup_artifact(key, db_path=db, now=now) is None

    # Fresh store then delete PDF → miss + cleanup
    ra.store_artifact(
        key,
        pdf_path=str(pdf),
        yaml_text="fresh",
        supportable=supportable,
        track="AI",
        title="AI Eng",
        db_path=db,
        now=now,
    )
    pdf.unlink()
    assert ra.lookup_artifact(key, db_path=db, now=now) is None


def test_title_only_lookup_never_hits():
    from src.core.tailor import lookup_archetype_pdf, store_archetype_pdf

    store_archetype_pdf("SDE", "Software Engineer", "/tmp/nope.pdf", "x")
    assert lookup_archetype_pdf("SDE", "Software Engineer") is None
