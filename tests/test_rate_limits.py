"""Ban armor pacing + per-host dispatch limits."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def test_pinned_locale_timezone_defaults():
    from src.core.browser_identity import pinned_locale, pinned_timezone_id

    assert pinned_locale()
    assert "/" in pinned_timezone_id() or pinned_timezone_id() == "UTC"


def test_apply_hours_window(monkeypatch):
    from src.core import ban_armor as ba

    monkeypatch.setenv("ACE_BROWSER_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("ACE_APPLY_HOURS", "9-17")
    tz = ZoneInfo("Asia/Kolkata")
    noon = datetime(2026, 3, 10, 12, 0, tzinfo=tz)
    night = datetime(2026, 3, 10, 23, 0, tzinfo=tz)
    assert ba.in_apply_hours(now=noon) is True
    assert ba.in_apply_hours(now=night) is False
    assert ba.seconds_until_apply_hours(now=night) > 0


def test_captcha_circuit_trips_and_cools():
    from src.core.ban_armor import CaptchaCircuitBreaker

    clock = {"t": 1000.0}
    br = CaptchaCircuitBreaker(
        threshold=3, window_sec=600, cooldown_sec=120, clock=lambda: clock["t"]
    )
    assert br.ready_in("linkedin.com") == 0.0
    assert br.record_block("linkedin.com") is False
    assert br.record_block("linkedin.com") is False
    assert br.record_block("linkedin.com") is True
    assert br.ready_in("linkedin.com") == 120.0
    clock["t"] += 121
    assert br.ready_in("linkedin.com") == 0.0
    assert br.is_cf_block_message("cloudflare/turnstile wall")
    assert not br.is_cf_block_message("timeout waiting for selector")

import sys
from pathlib import Path


def test_host_bucket_keys():
    from src.core.host_limiter import host_bucket_key

    assert host_bucket_key("https://www.linkedin.com/jobs/view/1") == "linkedin.com"
    assert host_bucket_key("https://company.myworkdayjobs.com/en-US/job") == "myworkdayjobs.com"
    assert host_bucket_key("https://jobs.lever.co/acme") == "lever.co"
    assert host_bucket_key("https://boards.greenhouse.io/acme") == "greenhouse.io"


def test_ready_in_and_mark_dispatched():
    from src.core.host_limiter import HostRateLimiter

    clock = {"t": 1000.0}

    def now() -> float:
        return clock["t"]

    lim = HostRateLimiter(clock=now)
    url = "https://www.linkedin.com/jobs/view/1"
    assert lim.ready_in(url) == 0.0
    lim.mark_dispatched(url)
    wait = lim.ready_in(url)
    assert 75.0 <= wait <= 105.0  # 90 ± 15
    clock["t"] += wait + 0.1
    assert lim.ready_in(url) == 0.0


def test_claim_interleaves_ready_host(tmp_path):
    from src.core.host_limiter import HostRateLimiter
    from src.core.queue import WAIT_SENTINEL_KEY, claim_next, enqueue_apply, init_queue

    db = tmp_path / "q.db"
    init_queue(db)
    enqueue_apply(
        company="LI",
        job_url="https://www.linkedin.com/jobs/view/1",
        track="SDE",
        jd_text="x",
        pdf_path=str(tmp_path / "a.pdf"),
        db_path=db,
    )
    enqueue_apply(
        company="LV",
        job_url="https://jobs.lever.co/acme/1",
        track="SDE",
        jd_text="y",
        pdf_path=str(tmp_path / "b.pdf"),
        db_path=db,
    )

    clock = {"t": 0.0}
    lim = HostRateLimiter(clock=lambda: clock["t"])
    lim.mark_dispatched("https://www.linkedin.com/jobs/view/1")

    item = claim_next("w1", db_path=db, host_limiter=lim)
    assert item is not None
    assert WAIT_SENTINEL_KEY not in item or "id" in item
    assert "lever.co" in str(item.get("job_url"))


def test_claim_wait_sentinel_when_all_cooling(tmp_path):
    from src.core.host_limiter import HostRateLimiter
    from src.core.memory import _connect
    from src.core.queue import WAIT_SENTINEL_KEY, claim_next, enqueue_apply, init_queue

    db = tmp_path / "q2.db"
    init_queue(db)
    enqueue_apply(
        company="LI",
        job_url="https://www.linkedin.com/jobs/view/2",
        track="SDE",
        jd_text="x",
        pdf_path=str(tmp_path / "a.pdf"),
        db_path=db,
    )

    clock = {"t": 0.0}
    lim = HostRateLimiter(clock=lambda: clock["t"])
    lim.mark_dispatched("https://www.linkedin.com/jobs/view/2")

    item = claim_next("w1", db_path=db, host_limiter=lim)
    assert item is not None
    assert WAIT_SENTINEL_KEY in item
    assert "id" not in item

    # No CLAIMED rows left behind
    with _connect(db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM apply_queue WHERE status='CLAIMED'"
        ).fetchone()["n"]
    assert int(n) == 0


def test_host_limits_env_override(monkeypatch):
    from src.core import host_limiter as hl

    monkeypatch.setenv("ACE_HOST_LIMITS", "linkedin.com=200:10")
    hl.reload_host_limits_for_tests()
    assert hl.HOST_LIMITS["linkedin.com"] == (200.0, 10.0)
    monkeypatch.delenv("ACE_HOST_LIMITS", raising=False)
    hl.reload_host_limits_for_tests()


def test_requeue_claimed(tmp_path):
    from src.core.memory import _connect
    from src.core.queue import claim_next, enqueue_apply, init_queue, requeue_claimed

    db = tmp_path / "q.db"
    init_queue(db)
    enqueue_apply(
        company="Acme",
        job_url="https://www.linkedin.com/jobs/view/9",
        track="SDE",
        jd_text="x",
        pdf_path=str(tmp_path / "a.pdf"),
        db_path=db,
    )
    item = claim_next("w1", db_path=db)
    assert item and "id" in item
    assert requeue_claimed(int(item["id"]), db_path=db) is True
    with _connect(db) as conn:
        row = conn.execute(
            "SELECT status FROM apply_queue WHERE id=?", (int(item["id"]),)
        ).fetchone()
    assert row["status"] == "READY"


def test_apply_queue_has_host_key_and_schedule(tmp_path):
    from src.core.memory import _connect
    from src.core.queue import enqueue_apply, init_queue

    db = tmp_path / "s.db"
    init_queue(db)
    enqueue_apply(
        company="LI",
        job_url="https://www.linkedin.com/jobs/view/schema",
        track="SDE",
        jd_text="x",
        pdf_path=str(tmp_path / "a.pdf"),
        db_path=db,
    )
    with _connect(db) as conn:
        cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(apply_queue)").fetchall()
        }
        tables = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        row = conn.execute(
            "SELECT host_key FROM apply_queue WHERE job_url LIKE '%schema'"
        ).fetchone()
    assert "host_key" in cols
    assert "attempt_count" in cols
    assert "host_schedule" in tables
    assert row["host_key"] == "linkedin.com"


def test_claim_skips_cooling_prefix_beyond_peek(tmp_path, monkeypatch):
    """Eligible host after many cooling LinkedIn rows must still be claimable."""
    from src.core.queue import WAIT_SENTINEL_KEY, claim_next, enqueue_apply, init_queue, reserve_host

    monkeypatch.setenv("ACE_CLAIM_SCAN_LIMIT", "500")
    db = tmp_path / "peek.db"
    init_queue(db)
    now = 1_000_000.0
    for i in range(30):
        enqueue_apply(
            company=f"LI{i}",
            job_url=f"https://www.linkedin.com/jobs/view/{i}",
            track="SDE",
            jd_text="x",
            pdf_path=str(tmp_path / f"a{i}.pdf"),
            db_path=db,
        )
    enqueue_apply(
        company="LV",
        job_url="https://jobs.lever.co/acme/beyond",
        track="SDE",
        jd_text="y",
        pdf_path=str(tmp_path / "lever.pdf"),
        db_path=db,
    )
    # Cool LinkedIn for a long window; Lever remains free.
    reserve_host("linkedin.com", db_path=db, now=now)
    item = claim_next("w1", db_path=db, now=now)
    assert item is not None
    assert WAIT_SENTINEL_KEY not in item
    assert "id" in item
    assert "lever.co" in str(item.get("job_url"))


def test_dual_workers_cannot_both_claim_same_host(tmp_path):
    from src.core.memory import _connect
    from src.core.queue import WAIT_SENTINEL_KEY, claim_next, enqueue_apply, init_queue

    db = tmp_path / "dual.db"
    init_queue(db)
    now = 2_000_000.0
    enqueue_apply(
        company="LI1",
        job_url="https://www.linkedin.com/jobs/view/dual1",
        track="SDE",
        jd_text="x",
        pdf_path=str(tmp_path / "a.pdf"),
        db_path=db,
    )
    enqueue_apply(
        company="LI2",
        job_url="https://www.linkedin.com/jobs/view/dual2",
        track="SDE",
        jd_text="y",
        pdf_path=str(tmp_path / "b.pdf"),
        db_path=db,
    )
    first = claim_next("w1", db_path=db, now=now)
    assert first and "id" in first
    second = claim_next("w2", db_path=db, now=now)
    # Host schedule reserved — second LinkedIn must wait (not both claimed).
    assert second is not None
    assert WAIT_SENTINEL_KEY in second
    with _connect(db) as conn:
        claimed = conn.execute(
            "SELECT COUNT(*) AS n FROM apply_queue WHERE status='CLAIMED'"
        ).fetchone()["n"]
        ready = conn.execute(
            "SELECT COUNT(*) AS n FROM apply_queue WHERE status='READY'"
        ).fetchone()["n"]
    assert int(claimed) == 1
    assert int(ready) == 1


def test_host_schedule_survives_restart(tmp_path):
    from src.core.memory import _connect
    from src.core.queue import claim_next, enqueue_apply, init_queue, reserve_host

    db = tmp_path / "persist.db"
    init_queue(db)
    now = 3_000_000.0
    enqueue_apply(
        company="LI",
        job_url="https://www.linkedin.com/jobs/view/persist",
        track="SDE",
        jd_text="x",
        pdf_path=str(tmp_path / "a.pdf"),
        db_path=db,
    )
    reserve_host("linkedin.com", db_path=db, now=now)
    with _connect(db) as conn:
        row = conn.execute(
            "SELECT next_eligible_at FROM host_schedule WHERE host_key='linkedin.com'"
        ).fetchone()
    assert float(row["next_eligible_at"]) > now
    # Re-init simulates process restart — schedule row must remain.
    init_queue(db)
    item = claim_next("w1", db_path=db, now=now)
    from src.core.queue import WAIT_SENTINEL_KEY

    assert item is not None and WAIT_SENTINEL_KEY in item
