from __future__ import annotations

from pathlib import Path

from src.core import apply_policy as ap
from src.core import human_assist as ha
from src.core import memory as mem
from src.core import queue as q
from src.playbooks import _common as common


def test_decide_matrix():
    assert ap.decide("success", 1) == "done"
    assert ap.decide("already_applied", 1) == "done"
    assert ap.decide("job_closed", 9) == "done"
    assert ap.decide("manual_action", 1) == "done"
    assert ap.decide("token_limit_reached", 1) == "stop_pool"
    assert ap.decide("timeout", 1, max_attempts=3) == "requeue"
    assert ap.decide("failed", 2, max_attempts=3) == "requeue"
    assert ap.decide("fallback", 3, max_attempts=3) == "done"
    assert ap.decide("requires_human", 3, max_attempts=3) == "done"
    assert ap.decide("weird", 1, max_attempts=3) == "requeue"


def test_ensure_captcha_cleared_raises_for_manual_action(monkeypatch):
    from src.core import human_assist as ha
    from src.core.executor import ApplicationBlockedError

    monkeypatch.setattr(ha, "page_shows_captcha", lambda *_a, **_k: True)
    monkeypatch.setattr(ha, "captcha_human_enabled", lambda: True)
    monkeypatch.setattr(ha, "resolve_captcha_with_human", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "src.playbooks._common.cloudflare_turnstile_present", lambda *_a, **_k: False
    )
    try:
        ha.ensure_captcha_cleared(object(), company="Co", site="test")
        assert False, "expected ApplicationBlockedError"
    except ApplicationBlockedError as exc:
        assert "Manual_Action" in str(exc)


def test_suppress_success_not_failed(tmp_path: Path, monkeypatch):
    db = tmp_path / "mem.db"
    monkeypatch.setenv("ACE_MAX_APPLY_ATTEMPTS", "3")
    mem.init_db(db)
    url = "https://example.com/jobs/1"
    assert not mem.should_suppress_apply(url, db_path=db)

    mem.log_application("Co", url, "t", "FAILED", error_log="timeout", db_path=db)
    assert not mem.should_suppress_apply(url, db_path=db)

    mem.log_application("Co", url, "t", "SUCCESS", db_path=db)
    assert mem.should_suppress_apply(url, db_path=db)


def test_suppress_on_retry_exhausted_via_queue(tmp_path: Path, monkeypatch):
    db = tmp_path / "mem.db"
    monkeypatch.setenv("ACE_MAX_APPLY_ATTEMPTS", "2")
    mem.init_db(db)
    q.init_queue(db)
    url = "https://example.com/jobs/retry"
    qid = q.enqueue_apply(
        company="Co",
        job_url=url,
        track="t",
        jd_text="jd",
        pdf_path=str(tmp_path / "r.pdf"),
        db_path=db,
    )
    (tmp_path / "r.pdf").write_bytes(b"%PDF")
    assert qid > 0
    # Simulate two finished attempts stored on the queue row.
    with mem._connect(db) as conn:
        conn.execute(
            "UPDATE apply_queue SET attempt_count=2 WHERE job_url=?",
            (url,),
        )
        conn.commit()
    assert mem.should_suppress_apply(url, db_path=db)


def test_queue_bump_attempt(tmp_path: Path):
    db = tmp_path / "q.db"
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    qid = q.enqueue_apply(
        company="Co",
        job_url="https://example.com/jobs/a",
        track="t",
        jd_text="jd",
        pdf_path=str(pdf),
        db_path=db,
    )
    claimed = q.claim_next("w1", db_path=db)
    assert claimed and int(claimed["id"]) == qid
    n1 = q.bump_attempt_on_claim(qid, db_path=db)
    assert n1 == 1
    assert q.requeue_claimed(qid, error_log="retry;failed", db_path=db)
    claimed2 = q.claim_next("w1", db_path=db)
    assert claimed2
    n2 = q.bump_attempt_on_claim(qid, db_path=db)
    assert n2 == 2


def test_confirm_requires_markers():
    class _Page:
        url = "https://jobs.example.com/apply"
        def inner_text(self, *_a, **_k):
            return "Please complete the form"
        def title(self):
            return "Apply"
    assert common.confirm_application_success(_Page()) is False
    assert common.submission_result(_Page(), submitted_click=True) is False

    class _Ok:
        url = "https://jobs.example.com/thank-you"
        def inner_text(self, *_a, **_k):
            return "Thank you for applying to our team"
        def title(self):
            return "Thanks"
    assert common.confirm_application_success(_Ok()) is True


def test_assist_and_confirm_needs_success(monkeypatch):
    calls = {"n": 0}

    def fake_assist(_reason):
        calls["n"] += 1
        return True

    class _Page:
        url = "https://example.com/form"
        def inner_text(self, *_a, **_k):
            return "still on form"
        def title(self):
            return "Apply"

    assert common.assist_and_confirm(_Page(), fake_assist, "stuck") is False
    assert calls["n"] == 1


def test_try_cdp_assist_learns_after_unblock(monkeypatch):
    learned = {"n": 0}

    def fake_assist(_reason):
        return True

    monkeypatch.setattr(
        "src.core.human_assist.snapshot_filled_fields", lambda *a, **k: []
    )

    def fake_learn(*_a, **_k):
        learned["n"] += 1
        return 1

    monkeypatch.setattr("src.core.human_assist.learn_from_page", fake_learn)
    assert common.try_cdp_assist(fake_assist, "stuck", page=object()) is True
    assert learned["n"] == 1


def test_settle_returns_early_on_success(monkeypatch):
    class _Page:
        url = "https://example.com/thank-you"
        def inner_text(self, *_a, **_k):
            return "Thank you for applying"
        def title(self):
            return "Thanks"
        def evaluate(self, *_a, **_k):
            return "complete"
        def wait_for_timeout(self, *_a, **_k):
            raise AssertionError("should not burn full wait when success is ready")

    assert common.settle(_Page(), timeout_ms=2000) is True


def test_unknown_gate_false_without_progress(monkeypatch):
    monkeypatch.setattr(ha, "unknown_answer_human_enabled", lambda: True)
    monkeypatch.setattr(ha, "capture_failure_shot", lambda *a, **k: None)
    monkeypatch.setattr(ha, "snapshot_filled_fields", lambda *a, **k: [])
    monkeypatch.setattr(
        ha, "wait_for_operator", lambda **k: True  # dismissed, but no learn
    )
    monkeypatch.setattr(ha, "learn_from_page", lambda *a, **k: 0)
    monkeypatch.setattr(ha, "form_looks_submitted", lambda *a, **k: False)

    assert ha.resolve_unknown_with_human(object(), company="X", timeout_s=5) is False


def test_wait_for_operator_respects_budget(monkeypatch, tmp_path: Path):
    import threading

    monkeypatch.setattr(ha, "failure_shots_dir", lambda: tmp_path)
    monkeypatch.setattr(ha, "_toast_windows", lambda *a, **k: None)
    hold = threading.Event()

    def hang_box(*_a, **_k):
        hold.wait(timeout=30)

    monkeypatch.setattr(ha, "_message_box_blocking", hang_box)
    clock = {"t": 1_000.0}
    monkeypatch.setattr(ha.time, "time", lambda: clock["t"])

    def advance(s):
        clock["t"] += max(float(s), 0.1) + 0.5

    monkeypatch.setattr(ha.time, "sleep", advance)
    try:
        ok = ha.wait_for_operator(
            title="t",
            message="m",
            budget_s=1.0,
            page=None,
            flag_name="_t.flag",
        )
        assert ok is False
    finally:
        hold.set()
