"""Process-isolated discovery deadlines — hung child must not block parent."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_hung_scraper_killed_sibling_still_returns():
    from src.core.discovery_runners import run_parallel_deadlines, run_with_deadline

    started = time.monotonic()
    hang = run_with_deadline("_hang", {}, timeout_s=1.5, label="hang")
    elapsed = time.monotonic() - started
    assert hang.timed_out is True
    assert hang.records == []
    assert elapsed < 8.0  # must not hang on context exit

    ok = run_with_deadline("_ok", {"label": "sib"}, timeout_s=10.0, label="ok")
    assert ok.ok is True
    assert ok.timed_out is False
    assert any("example.com/sib" in str(r.get("job_url")) for r in ok.records)

    # Parallel: hang + ok under overall budget → partial merge
    results = run_parallel_deadlines(
        [
            ("_hang", {}, 1.5, "hang"),
            ("_ok", {"label": "alive"}, 10.0, "alive"),
        ],
        overall_timeout_s=12.0,
    )
    by_label = {r.label: r for r in results}
    assert by_label["hang"].timed_out is True
    assert by_label["alive"].ok is True
    assert by_label["alive"].records


def test_run_with_deadline_success_path():
    from src.core.discovery_runners import run_with_deadline

    result = run_with_deadline("_ok", {"label": "solo"}, timeout_s=10.0)
    assert result.ok and not result.timed_out
    assert result.records[0]["title"] == "solo"
