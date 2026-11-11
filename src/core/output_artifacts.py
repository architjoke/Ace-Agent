"""Output artifact lifecycle — DB is source of truth; disk is a scratch cache.

Industry workflow:
  prepare → output/<Company>/ (build)
  human gate → Manual_Action/<Company>/ (keep until mark)
  SUCCESS / user-mark → archive PDF under Submitted/, remove build dir
  periodic prune → drop stale builds not tied to open Manual_Action

Hot path: one company rmtree (same cost as existing post-SUCCESS cleanup).
Prune: top-level mtime scan only, throttled via stamp file.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path

from src.core.config import OUTPUT_DIR, ensure_runtime_dirs

logger = logging.getLogger(__name__)

RESERVED_OUTPUT_DIRS = frozenset(
    {
        "Manual_Action",
        "Submitted",
        "Failure_Shots",
        "runs",
    }
)

_GC_STAMP = ".output_gc_stamp"
_SAFE_NAME_RE = re.compile(r'[<>:"/\\|?*]')


def _env_truthy(name: str, default: str = "1") -> bool:
    raw = (os.environ.get(name) if name in os.environ else default) or default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ttl_days() -> int:
    raw = (os.environ.get("ACE_OUTPUT_BUILD_TTL_DAYS") or "14").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 14


def _prune_interval_sec() -> float:
    raw = (os.environ.get("ACE_OUTPUT_PRUNE_INTERVAL_HOURS") or "12").strip()
    try:
        return max(1.0, float(raw)) * 3600.0
    except ValueError:
        return 12.0 * 3600.0


def safe_dir_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("", (name or "").strip()).strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unknown"


def output_root() -> Path:
    try:
        from src.core.settings import get_settings

        return Path(get_settings().output_dir).expanduser().resolve()
    except Exception:
        return OUTPUT_DIR.resolve()


def company_build_dir(company: str) -> Path:
    return output_root() / safe_dir_name(company)


def manual_action_dir(company: str | None = None) -> Path:
    root = output_root() / "Manual_Action"
    if company is None:
        return root
    return root / safe_dir_name(company)


def gc_on_success_enabled() -> bool:
    return _env_truthy("ACE_OUTPUT_GC_ON_SUCCESS", "1")


def release_company_build(
    company: str,
    *,
    pdf_path: Path | str | None = None,
    archive: bool = True,
    reason: str = "",
) -> bool:
    """Archive resume PDF under Submitted/ and remove output/<Company>/ build dir.

    Never touches Manual_Action/. Best-effort; never raises to callers.
    Returns True if the build dir was removed (or already absent).
    """
    if not gc_on_success_enabled():
        return False
    name = safe_dir_name(company)
    if not name or name in RESERVED_OUTPUT_DIRS:
        return False

    out = output_root()
    company_dir = out / name
    manual = out / "Manual_Action" / name
    # Open human package still needs the build as source for retries/undo copies.
    if manual.is_dir():
        logger.debug(
            "Skip build GC for %s — Manual_Action package still open (%s)",
            name,
            reason or "n/a",
        )
        return False

    try:
        ensure_runtime_dirs()
        submitted_dir = out / "Submitted" / name
        if archive:
            submitted_dir.mkdir(parents=True, exist_ok=True)
            src_pdf = Path(pdf_path) if pdf_path else None
            archived = False
            if src_pdf and src_pdf.is_file():
                try:
                    # Only archive if under output/ (path jail soft check).
                    src_pdf.resolve().relative_to(out)
                except ValueError:
                    src_pdf = None
            if src_pdf and src_pdf.is_file():
                dest = submitted_dir / src_pdf.name
                shutil.copy2(src_pdf, dest)
                archived = True
            elif company_dir.is_dir():
                for pdf in company_dir.glob("*.pdf"):
                    if pdf.is_file():
                        shutil.copy2(pdf, submitted_dir / pdf.name)
                        archived = True
                        break
            if archived:
                logger.info(
                    "Archived submitted resume → %s (%s)",
                    submitted_dir,
                    reason or "success",
                )

        if company_dir.is_dir():
            # Same-company parallel applies share one build dir — don't nuke peers.
            try:
                from src.core.queue import company_has_peer_apply

                # Current job is still CLAIMED during SUCCESS cleanup; peers ⇒ count>1.
                if company_has_peer_apply(company):
                    logger.debug(
                        "Skip build rmtree for %s — peer READY/CLAIMED (%s)",
                        name,
                        reason or "n/a",
                    )
                    return False
            except Exception:
                logger.debug("peer-queue check failed", exc_info=True)
            shutil.rmtree(company_dir, ignore_errors=True)
            if not company_dir.exists():
                logger.info(
                    "Released build folder %s (%s)",
                    company_dir,
                    reason or "success",
                )
                return True
            return False
        return True
    except OSError:
        logger.exception("Build GC failed for %s", name)
        return False


def _dir_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def prune_stale_company_builds(*, force: bool = False) -> dict[str, int]:
    """Drop top-level company build dirs older than TTL if not in Manual_Action.

    Cheap: lists output/ children only (no deep walk). Throttled by stamp file.
    """
    if not _env_truthy("ACE_OUTPUT_PRUNE_STALE", "1"):
        return {"removed": 0, "skipped": 0, "throttled": 1}

    out = output_root()
    stamp = out / _GC_STAMP
    now = time.time()
    if not force and stamp.is_file():
        try:
            last = float(stamp.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            last = 0.0
        if (now - last) < _prune_interval_sec():
            return {"removed": 0, "skipped": 0, "throttled": 1}

    ttl = _ttl_days() * 86400.0
    cutoff = now - ttl
    manual_root = out / "Manual_Action"
    open_manual: set[str] = set()
    if manual_root.is_dir():
        try:
            open_manual = {p.name for p in manual_root.iterdir() if p.is_dir()}
        except OSError:
            open_manual = set()

    removed = 0
    skipped = 0
    try:
        children = list(out.iterdir())
    except OSError:
        return {"removed": 0, "skipped": 0, "throttled": 0}

    for child in children:
        if not child.is_dir():
            skipped += 1
            continue
        name = child.name
        if name in RESERVED_OUTPUT_DIRS or name.startswith("."):
            skipped += 1
            continue
        if name in open_manual:
            skipped += 1
            continue
        if _dir_mtime(child) > cutoff:
            skipped += 1
            continue
        try:
            shutil.rmtree(child, ignore_errors=True)
            if not child.exists():
                removed += 1
                logger.info("Pruned stale build folder %s (ttl=%sd)", child, _ttl_days())
        except OSError:
            logger.debug("prune failed for %s", child, exc_info=True)
            skipped += 1

    try:
        stamp.write_text(str(now), encoding="utf-8")
    except OSError:
        pass

    return {"removed": removed, "skipped": skipped, "throttled": 0}
