"""Lightweight stage timing → output/telemetry.jsonl (no SQLite schema).

Post-run p50/p95 example::

    python -c "import json,statistics as s; from pathlib import Path
    rows=[json.loads(l) for l in Path('output/telemetry.jsonl').read_text().splitlines() if l.strip()]
    for stage in sorted({r['stage'] for r in rows}):
        xs=sorted(r['ms'] for r in rows if r['stage']==stage and r.get('ok'))
        if not xs: continue
        def pct(p):
            i=min(len(xs)-1, max(0, int(round((p/100)*(len(xs)-1)))))
            return xs[i]
        print(stage, 'n=',len(xs),'p50=',pct(50),'p90=',pct(90),'p95=',pct(95))"
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.core.config import OUTPUT_DIR, ensure_runtime_dirs

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TELEMETRY_FILE = OUTPUT_DIR / "telemetry.jsonl"


def telemetry_path() -> Path:
    return _TELEMETRY_FILE


def _emit(stage: str, *, ms: float, ok: bool, **tags: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "stage": stage,
        "ms": round(float(ms), 3),
        "ok": bool(ok),
    }
    for k, v in tags.items():
        if v is None:
            continue
        record[k] = v
    try:
        from src.core.security import redact_secrets

        line = redact_secrets(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        ensure_runtime_dirs()
        from src.core.logutil import env_max_bytes, rotate_if_oversized

        with _LOCK:
            rotate_if_oversized(
                _TELEMETRY_FILE,
                max_bytes=env_max_bytes("ACE_TELEMETRY_MAX_BYTES"),
            )
            with _TELEMETRY_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        logger.debug("telemetry emit failed", exc_info=True)
    logger.debug("telemetry %s ms=%.1f ok=%s", stage, ms, ok)


@contextmanager
def time_stage(stage: str, **tags: Any) -> Iterator[None]:
    """Time a stage and append one JSONL sample."""
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        _emit(
            stage,
            ms=(time.perf_counter() - t0) * 1000.0,
            ok=False,
            error=type(exc).__name__,
            **tags,
        )
        raise
    else:
        _emit(
            stage,
            ms=(time.perf_counter() - t0) * 1000.0,
            ok=True,
            **tags,
        )
