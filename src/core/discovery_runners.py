"""Process-isolated discovery workers with hard killable deadlines.

ThreadPoolExecutor + ``fut.result(timeout=…)`` only stops *waiting*; hung
scraper threads keep running and can block ``shutdown``. This module runs each
source in a spawn-safe child process and ``terminate``/``kill`` on deadline.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeadlineResult:
    ok: bool
    records: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    timed_out: bool = False
    label: str = ""


def track_timeout_s() -> float:
    raw = (os.environ.get("ACE_DISCOVERY_TRACK_TIMEOUT_S") or "120").strip()
    try:
        return max(15.0, float(raw))
    except ValueError:
        return 120.0


def overall_timeout_s() -> float:
    raw = (os.environ.get("ACE_DISCOVERY_OVERALL_TIMEOUT_S") or "300").strip()
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 300.0


def extra_timeout_s() -> float:
    raw = (os.environ.get("ACE_EXTRA_DISCOVERY_TIMEOUT_S") or "90").strip()
    try:
        return max(20.0, float(raw))
    except ValueError:
        return 90.0


def jobdrop_site_timeout_s() -> float:
    raw = (os.environ.get("ACE_JOBDROP_SITE_TIMEOUT_S") or "35").strip()
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 35.0


def _df_to_records(df: Any) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    try:
        return df.to_dict(orient="records")
    except Exception:
        return []


def run_jobspy_track(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Picklable JobSpy track worker (imported lazily inside the child)."""
    from src.core.discovery import fetch_jobs

    df = fetch_jobs(
        keywords=str(payload.get("keywords") or ""),
        location=str(payload.get("location") or "India"),
        max_results=int(payload.get("max_results") or 10),
        country_indeed=str(payload.get("country_indeed") or "India"),
        track=payload.get("track"),
        sites=payload.get("sites"),
    )
    return _df_to_records(df)


def run_jobdrop_site(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Picklable single Jobdrop site scrape."""
    from src.core.discovery_extra import fetch_jobdrop_jobs

    df = fetch_jobdrop_jobs(
        str(payload.get("keywords") or ""),
        location=str(payload.get("location") or "India"),
        max_results=int(payload.get("max_results") or 10),
        track=payload.get("track"),
        sites=[str(payload.get("site") or "")],
        country_indeed=str(payload.get("country_indeed") or "India"),
        per_site_timeout_s=float(payload.get("per_site_timeout_s") or 35.0),
        isolate_sites=False,
    )
    return _df_to_records(df)


def run_himalayas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from src.core.discovery_extra import fetch_himalayas_jobs

    df = fetch_himalayas_jobs(
        str(payload.get("keywords") or ""),
        max_results=int(payload.get("max_results") or 10),
        track=payload.get("track"),
    )
    return _df_to_records(df)


def run_yc_waas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    from src.core.discovery_extra import fetch_yc_waas_jobs

    df = fetch_yc_waas_jobs(
        str(payload.get("keywords") or ""),
        max_results=int(payload.get("max_results") or 10),
        track=payload.get("track"),
    )
    return _df_to_records(df)


def run_extra_track(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One track's Jobdrop + Himalayas + YC (used when isolating whole tracks)."""
    from src.core.discovery_extra import (
        fetch_himalayas_jobs,
        fetch_jobdrop_jobs,
        fetch_yc_waas_jobs,
        jobdrop_sites_for_hunt,
    )

    keywords = str(payload.get("keywords") or "")
    track = payload.get("track")
    location = str(payload.get("location") or "India")
    max_results = int(payload.get("max_results") or 8)
    country_indeed = str(payload.get("country_indeed") or "India")
    site_timeout = float(payload.get("per_site_timeout_s") or jobdrop_site_timeout_s())
    out: list[dict[str, Any]] = []
    try:
        out.extend(
            _df_to_records(
                fetch_jobdrop_jobs(
                    keywords,
                    location=location,
                    max_results=max_results,
                    track=track,
                    country_indeed=country_indeed,
                    sites=jobdrop_sites_for_hunt(),
                    per_site_timeout_s=site_timeout,
                    isolate_sites=False,
                )
            )
        )
    except Exception:
        pass
    try:
        out.extend(
            _df_to_records(
                fetch_himalayas_jobs(
                    keywords,
                    max_results=max(3, max_results // 2),
                    track=track,
                )
            )
        )
    except Exception:
        pass
    try:
        out.extend(
            _df_to_records(
                fetch_yc_waas_jobs(
                    keywords,
                    max_results=max(3, max_results // 2),
                    track=track,
                )
            )
        )
    except Exception:
        pass
    return out


def _hang_forever(_payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Test-only infinite worker (must be killable by the parent)."""
    while True:
        time.sleep(60)


def _ok_sample(payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Test-only successful worker."""
    label = (payload or {}).get("label") or "ok"
    return [{"job_url": f"https://example.com/{label}", "title": str(label)}]


WORKERS: dict[str, Any] = {
    "jobspy_track": run_jobspy_track,
    "jobdrop_site": run_jobdrop_site,
    "himalayas": run_himalayas,
    "yc_waas": run_yc_waas,
    "extra_track": run_extra_track,
    "_hang": _hang_forever,
    "_ok": _ok_sample,
}


def _child_entry(
    fn_name: str,
    payload: dict[str, Any] | None,
    conn: Any,
) -> None:
    """Spawn entry: run named worker and send records over a Pipe."""
    try:
        fn = WORKERS.get(fn_name)
        if fn is None:
            conn.send(
                {
                    "ok": False,
                    "records": [],
                    "error": f"unknown worker {fn_name!r}",
                }
            )
            return
        records = fn(payload or {})
        conn.send({"ok": True, "records": records or [], "error": ""})
    except Exception as exc:
        conn.send(
            {
                "ok": False,
                "records": [],
                "error": f"{exc}\n{traceback.format_exc()}",
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _force_kill(proc: Any) -> None:
    if not proc.is_alive():
        return
    try:
        proc.terminate()
    except Exception:
        pass
    proc.join(timeout=2.0)
    if proc.is_alive():
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(timeout=1.0)


def run_with_deadline(
    fn_name: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = 120.0,
    label: str = "",
) -> DeadlineResult:
    """Run a named picklable worker in a child process; kill on deadline."""
    tag = label or fn_name
    timeout = max(1.0, float(timeout_s))
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_entry,
        args=(fn_name, payload, child_conn),
        daemon=True,
    )
    proc.start()
    try:
        child_conn.close()
    except Exception:
        pass

    deadline = time.monotonic() + timeout
    result_payload: dict[str, Any] | None = None
    timed_out = False

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            if parent_conn.poll(min(0.5, remaining)):
                result_payload = parent_conn.recv()
                break
        except (EOFError, OSError, BrokenPipeError):
            break
        if not proc.is_alive():
            try:
                if parent_conn.poll(0.2):
                    result_payload = parent_conn.recv()
            except (EOFError, OSError, BrokenPipeError):
                pass
            break

    if timed_out or (result_payload is None and proc.is_alive()):
        logger.warning(
            "Discovery worker %s timed out after %.0fs — killing process",
            tag,
            timeout,
        )
        _force_kill(proc)
        try:
            parent_conn.close()
        except Exception:
            pass
        return DeadlineResult(
            ok=False,
            records=[],
            error="timeout",
            timed_out=True,
            label=tag,
        )

    _force_kill(proc)  # noop if already exited
    try:
        parent_conn.close()
    except Exception:
        pass

    if not isinstance(result_payload, dict):
        return DeadlineResult(
            ok=False,
            records=[],
            error="no result from child",
            timed_out=False,
            label=tag,
        )

    records = result_payload.get("records") or []
    if not isinstance(records, list):
        records = []
    ok = bool(result_payload.get("ok"))
    err = str(result_payload.get("error") or "")
    return DeadlineResult(
        ok=ok,
        records=records,
        error=err,
        timed_out=False,
        label=tag,
    )


def run_parallel_deadlines(
    jobs: list[tuple[str, dict[str, Any] | None, float, str]],
    *,
    overall_timeout_s: float | None = None,
) -> list[DeadlineResult]:
    """Start several workers; enforce per-job and optional overall deadlines.

    Each job is ``(fn_name, payload, timeout_s, label)``. Collects partial
    successes; hung children are killed and never contribute late results.
    """
    if not jobs:
        return []

    wall = float(
        overall_timeout_s
        if overall_timeout_s is not None
        else globals()["overall_timeout_s"]()
    )
    wall_deadline = time.monotonic() + max(1.0, wall)
    ctx = mp.get_context("spawn")

    slots: list[dict[str, Any]] = []
    for fn_name, payload, timeout_s, label in jobs:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_entry,
            args=(fn_name, payload, child_conn),
            daemon=True,
        )
        proc.start()
        try:
            child_conn.close()
        except Exception:
            pass
        per = max(1.0, float(timeout_s))
        slots.append(
            {
                "fn_name": fn_name,
                "label": label or fn_name,
                "proc": proc,
                "conn": parent_conn,
                "deadline": time.monotonic() + per,
                "done": False,
                "result": None,
            }
        )

    results: list[DeadlineResult] = []
    while any(not s["done"] for s in slots):
        now = time.monotonic()
        if now >= wall_deadline:
            for s in slots:
                if s["done"]:
                    continue
                logger.warning(
                    "Discovery overall deadline — killing %s",
                    s["label"],
                )
                _force_kill(s["proc"])
                try:
                    s["conn"].close()
                except Exception:
                    pass
                s["done"] = True
                results.append(
                    DeadlineResult(
                        ok=False,
                        records=[],
                        error="overall_timeout",
                        timed_out=True,
                        label=str(s["label"]),
                    )
                )
            break

        for s in slots:
            if s["done"]:
                continue
            proc = s["proc"]
            conn = s["conn"]
            label = str(s["label"])
            if now >= float(s["deadline"]):
                logger.warning(
                    "Discovery worker %s timed out — killing process",
                    label,
                )
                _force_kill(proc)
                try:
                    conn.close()
                except Exception:
                    pass
                s["done"] = True
                results.append(
                    DeadlineResult(
                        ok=False,
                        records=[],
                        error="timeout",
                        timed_out=True,
                        label=label,
                    )
                )
                continue
            try:
                if conn.poll(0.05):
                    payload = conn.recv()
                    _force_kill(proc)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    s["done"] = True
                    records = (payload or {}).get("records") or []
                    if not isinstance(records, list):
                        records = []
                    results.append(
                        DeadlineResult(
                            ok=bool((payload or {}).get("ok")),
                            records=records,
                            error=str((payload or {}).get("error") or ""),
                            timed_out=False,
                            label=label,
                        )
                    )
                    continue
            except (EOFError, OSError, BrokenPipeError):
                _force_kill(proc)
                try:
                    conn.close()
                except Exception:
                    pass
                s["done"] = True
                results.append(
                    DeadlineResult(
                        ok=False,
                        records=[],
                        error="pipe closed",
                        timed_out=False,
                        label=label,
                    )
                )
                continue
            if not proc.is_alive():
                try:
                    payload = conn.recv() if conn.poll(0.2) else None
                except (EOFError, OSError, BrokenPipeError):
                    payload = None
                try:
                    conn.close()
                except Exception:
                    pass
                s["done"] = True
                if isinstance(payload, dict):
                    records = payload.get("records") or []
                    if not isinstance(records, list):
                        records = []
                    results.append(
                        DeadlineResult(
                            ok=bool(payload.get("ok")),
                            records=records,
                            error=str(payload.get("error") or ""),
                            timed_out=False,
                            label=label,
                        )
                    )
                else:
                    results.append(
                        DeadlineResult(
                            ok=False,
                            records=[],
                            error="exited without result",
                            timed_out=False,
                            label=label,
                        )
                    )
        time.sleep(0.05)

    return results


def records_to_frame(records: list[dict[str, Any]]):
    """Build a DataFrame from worker records (empty → empty frame)."""
    import pandas as pd

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
