"""Per-host rate limiter + claim_next domain interleaving."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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


def test_apply_queue_schema_unchanged(tmp_path):
    """Guard: no new columns from this performance pass."""
    from src.core.memory import _connect
    from src.core.queue import init_queue

    db = tmp_path / "s.db"
    init_queue(db)
    with _connect(db) as conn:
        cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(apply_queue)").fetchall()
        }
    expected = {
        "id",
        "company",
        "job_url",
        "track",
        "jd_text",
        "pdf_path",
        "yaml_snapshot",
        "status",
        "worker_id",
        "error_log",
        "created_at",
        "updated_at",
    }
    assert cols == expected
