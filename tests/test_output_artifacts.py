"""Output build lifecycle: archive on success, never touch open Manual_Action."""

from __future__ import annotations

from pathlib import Path


def test_release_skips_while_manual_open(tmp_path, monkeypatch):
    from src.core import output_artifacts as oa

    monkeypatch.setattr(oa, "output_root", lambda: tmp_path)
    monkeypatch.setenv("ACE_OUTPUT_GC_ON_SUCCESS", "1")
    build = tmp_path / "Acme"
    build.mkdir()
    (build / "tailored.yaml").write_text("cv: {}\n", encoding="utf-8")
    (build / "resume.pdf").write_bytes(b"%PDF")
    manual = tmp_path / "Manual_Action" / "Acme"
    manual.mkdir(parents=True)
    (manual / "note.txt").write_text("open", encoding="utf-8")

    assert oa.release_company_build("Acme", reason="test") is False
    assert build.is_dir()


def test_release_archives_and_removes_build(tmp_path, monkeypatch):
    from src.core import output_artifacts as oa

    monkeypatch.setattr(oa, "output_root", lambda: tmp_path)
    monkeypatch.setenv("ACE_OUTPUT_GC_ON_SUCCESS", "1")
    build = tmp_path / "Acme"
    build.mkdir()
    pdf = build / "Archit_Resume.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (build / "tailored.yaml").write_text("cv: {}\n", encoding="utf-8")
    (tmp_path / "Manual_Action").mkdir(exist_ok=True)

    assert oa.release_company_build("Acme", pdf_path=pdf, reason="test") is True
    assert not build.exists()
    archived = tmp_path / "Submitted" / "Acme" / "Archit_Resume.pdf"
    assert archived.is_file()


def test_prune_respects_ttl_and_manual(tmp_path, monkeypatch):
    import time

    from src.core import output_artifacts as oa

    monkeypatch.setattr(oa, "output_root", lambda: tmp_path)
    monkeypatch.setenv("ACE_OUTPUT_PRUNE_STALE", "1")
    monkeypatch.setenv("ACE_OUTPUT_BUILD_TTL_DAYS", "1")
    monkeypatch.setenv("ACE_OUTPUT_PRUNE_INTERVAL_HOURS", "0")

    stale = tmp_path / "OldCo"
    stale.mkdir()
    (stale / "x.txt").write_text("x", encoding="utf-8")
    old = time.time() - 3 * 86400
    os_utime = __import__("os").utime
    os_utime(stale, (old, old))

    keep_manual = tmp_path / "OpenCo"
    keep_manual.mkdir()
    (tmp_path / "Manual_Action" / "OpenCo").mkdir(parents=True)
    os_utime(keep_manual, (old, old))

    fresh = tmp_path / "NewCo"
    fresh.mkdir()

    (tmp_path / "Manual_Action").mkdir(exist_ok=True)
    stats = oa.prune_stale_company_builds(force=True)
    assert stats["removed"] >= 1
    assert not stale.exists()
    assert keep_manual.exists()
    assert fresh.exists()
