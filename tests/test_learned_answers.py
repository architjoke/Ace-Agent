from __future__ import annotations

from pathlib import Path

from src.core import learned_answers as la
from src.core import human_assist as ha


def test_remember_and_lookup(tmp_path: Path):
    path = tmp_path / "learned.yaml"
    entry = la.remember_answer(
        "What is your notice period in days?",
        "7",
        company="Lumenci",
        job_url="https://example.com/job",
        path=path,
    )
    assert entry is not None
    assert entry["key"] == "notice_period_days"
    assert la.lookup_answer("notice period in days", path=path) == "7"

    overlay = la.as_overlay(path)
    assert overlay["notice_period_days"] == "7"
    assert any(k.startswith("learned_q:") for k in overlay)


def test_merge_does_not_clobber(tmp_path: Path):
    path = tmp_path / "learned.yaml"
    la.remember_answer("Expected CTC", "700000", path=path)
    merged = la.merge_into_answers({"expected_ctc_inr": "650000"}, path=path)
    assert merged["expected_ctc_inr"] == "650000"
    merged2 = la.merge_into_answers({"expected_ctc_inr": ""}, path=path)
    assert merged2["expected_ctc_inr"] == "700000"


def test_diff_new_values_prefers_changes():
    before = [("Notice period", "")]
    after = [("Notice period", "7"), ("City", "Delhi")]
    # empty before value isn't in snapshot_filled (only non-empty), simulate:
    before = []
    got = ha._diff_new_values(before, after)
    assert ("Notice period", "7") in got
    assert ("City", "Delhi") in got


def test_diff_falls_back_to_after_when_unchanged():
    pairs = [("Are you willing to relocate?", "Yes")]
    assert ha._diff_new_values(pairs, pairs) == pairs


def test_prompt_block_includes_learned(tmp_path: Path):
    path = tmp_path / "learned.yaml"
    la.remember_answer(
        "Are you comfortable working in a fast-paced startup?",
        "Yes",
        path=path,
    )
    block = la.learned_qa_prompt_block(path)
    assert "LEARNED FROM PRIOR HUMAN FILLS" in block
    assert "fast-paced startup" in block
    assert "Yes" in block


def test_unknown_disabled_returns_false(monkeypatch):
    monkeypatch.setattr(ha, "unknown_answer_human_enabled", lambda: False)
    monkeypatch.setattr(ha, "capture_failure_shot", lambda *a, **k: None)
    assert ha.resolve_unknown_with_human(object(), company="X") is False


def test_learned_load_uses_mtime_cache(tmp_path: Path, monkeypatch):
    path = tmp_path / "learned.yaml"
    la.remember_answer("Phone number", "9999999999", path=path)
    la._invalidate_cache()
    reads = {"n": 0}
    real_read = Path.read_text

    def counting_read(self, *args, **kwargs):
        if self == path:
            reads["n"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    assert la.lookup_answer("Phone number", path=path) == "9999999999"
    assert la.lookup_answer("Phone number", path=path) == "9999999999"
    assert reads["n"] == 1
    # Mutating caller snapshot must not corrupt the cache.
    data = la._load_raw(path)
    data["entries"].clear()
    assert la.lookup_answer("Phone number", path=path) == "9999999999"


def _write_master(path: Path, answers: dict | None = None) -> Path:
    import yaml

    payload = {"application_answers": dict(answers or {}), "cv": {"name": "Test User"}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_promote_allowlisted_key_fills_empty_master(tmp_path: Path):
    import yaml
    from src.core import profile_promote as pp

    learned = tmp_path / "learned.yaml"
    master = _write_master(tmp_path / "master.yaml")
    la.remember_answer(
        "Pin code",
        "110001",
        path=learned,
        master_path=master,
    )
    data = yaml.safe_load(master.read_text(encoding="utf-8"))
    assert data["application_answers"]["pincode"] == "110001"
    assert pp.promote_learned_to_master(learned_path=learned, master_path=master) == 0


def test_promote_does_not_overwrite_conflict_on_first_hit(tmp_path: Path):
    import yaml

    learned = tmp_path / "learned.yaml"
    master = _write_master(tmp_path / "master.yaml", {"pincode": "110001"})
    la.remember_answer(
        "Pin code",
        "560001",
        path=learned,
        master_path=master,
    )
    data = yaml.safe_load(master.read_text(encoding="utf-8"))
    assert data["application_answers"]["pincode"] == "110001"


def test_promote_overwrites_conflict_after_repeated_hits(tmp_path: Path):
    import yaml

    learned = tmp_path / "learned.yaml"
    master = _write_master(tmp_path / "master.yaml", {"city": "Old City"})
    la.remember_answer("City", "Delhi", path=learned, master_path=master)
    assert yaml.safe_load(master.read_text(encoding="utf-8"))["application_answers"]["city"] == "Old City"
    la.remember_answer("City", "Delhi", path=learned, master_path=master)
    assert yaml.safe_load(master.read_text(encoding="utf-8"))["application_answers"]["city"] == "Delhi"


def test_promote_skips_freeform_without_key(tmp_path: Path):
    import yaml
    from src.core import profile_promote as pp

    learned = tmp_path / "learned.yaml"
    master = _write_master(tmp_path / "master.yaml")
    entry = la.remember_answer(
        "Why do you want to join our company culture?",
        "I love building products.",
        path=learned,
        master_path=master,
    )
    assert entry is not None
    assert not entry.get("key")
    data = yaml.safe_load(master.read_text(encoding="utf-8"))
    assert data["application_answers"] == {}
    assert pp.promote_entry_to_master(entry, master_path=master) is False


def test_promote_skips_non_allowlisted_key(tmp_path: Path):
    import yaml
    from src.core import profile_promote as pp

    master = _write_master(tmp_path / "master.yaml")
    entry = {
        "question": "Custom screening",
        "key": "why_this_role",
        "answer": "Because of the mission.",
        "hits": 5,
    }
    assert pp.promote_entry_to_master(entry, master_path=master) is False
    data = yaml.safe_load(master.read_text(encoding="utf-8"))
    assert "why_this_role" not in data["application_answers"]
