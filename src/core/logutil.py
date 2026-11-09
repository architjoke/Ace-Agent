"""Size-bounded append helpers for long-running daemon logs / telemetry."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 20_000_000


def env_max_bytes(name: str, default: int = _DEFAULT_MAX_BYTES) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return max(1_000_000, int(raw))
    except ValueError:
        return int(default)


def rotate_if_oversized(path: Path | str, *, max_bytes: int | None = None) -> bool:
    """If ``path`` exceeds max_bytes, move it to ``path.name.1`` (single backup).

    Returns True when a rotation happened.
    """
    p = Path(path)
    limit = int(max_bytes) if max_bytes is not None else _DEFAULT_MAX_BYTES
    try:
        if not p.is_file():
            return False
        if p.stat().st_size < limit:
            return False
    except OSError:
        return False

    backup = p.with_name(p.name + ".1")
    try:
        if backup.exists():
            backup.unlink()
        shutil.move(str(p), str(backup))
        logger.info("Rotated oversized log %s → %s (limit=%s)", p, backup, limit)
        return True
    except OSError:
        logger.warning("Failed rotating %s", p, exc_info=True)
        return False
