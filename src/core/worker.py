"""Continuous hunt → tailor → apply worker (no company cap)."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from src.core.config import ENV_FILE, PROJECT_ROOT

REPO_ROOT = PROJECT_ROOT
load_dotenv(ENV_FILE)

from src.core.desk import LOG_FILE, clear_pid, write_pid
from src.core.pipeline import run_parallel

logger = logging.getLogger("ace_worker")


async def _cycle_loop(
    *,
    max_results_per_track: int | None,
    cycle_sleep_sec: int,
) -> None:
    """One long-lived event loop for all hunt cycles (avoids 'Event loop is closed')."""
    cycle = 0
    while True:
        cycle += 1
        logger.info("=== Ace worker cycle %s (uncapped) ===", cycle)
        try:
            code = await run_parallel(
                max_results_per_track=max_results_per_track,
                target_companies=None,
            )
            logger.info("Cycle %s finished exit=%s", cycle, code)
        except Exception:
            logger.exception(
                "Cycle %s crashed — structured recovery; will retry", cycle
            )
        finally:
            try:
                from src.core.browser_warmup import blank_all_slots

                blanked = blank_all_slots()
                if blanked:
                    logger.info(
                        "Cycle %s blanked %s warm browser slot(s) for reuse",
                        cycle,
                        blanked,
                    )
            except Exception:
                logger.debug("blank_all_slots between cycles failed", exc_info=True)
        sleep_for = max(60, int(cycle_sleep_sec))
        logger.info("Sleeping %ss before next hunt…", sleep_for)
        await asyncio.sleep(sleep_for)


def run_forever(
    *,
    max_results_per_track: int | None = None,
    cycle_sleep_sec: int = 900,
) -> None:
    """Apply to as many fit jobs as possible; re-hunt after each batch."""
    import atexit
    import signal

    from src.core.browser_warmup import close_all_slots
    from src.core.logutil import env_max_bytes, rotate_if_oversized
    from src.core.queue import reclaim_orphaned_claims
    from src.core.security import install_redacting_logging

    install_redacting_logging(force=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    rotate_if_oversized(LOG_FILE, max_bytes=env_max_bytes("ACE_LOG_MAX_BYTES"))
    write_pid(os.getpid())
    atexit.register(close_all_slots)

    def _sigterm(_signum, _frame):
        try:
            close_all_slots()
        except Exception:
            pass
        raise SystemExit(0)

    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except Exception:
            logger.debug("SIGTERM handler not installed", exc_info=True)
    try:
        stats = reclaim_orphaned_claims()
        logger.info("Startup orphan reclaim: %s", stats)
    except Exception:
        logger.exception("Startup orphan reclaim failed — continuing")
    try:
        asyncio.run(
            _cycle_loop(
                max_results_per_track=max_results_per_track,
                cycle_sleep_sec=cycle_sleep_sec,
            )
        )
    finally:
        try:
            close_all_slots()
        except Exception:
            logger.debug("close_all_slots on worker exit failed", exc_info=True)
        clear_pid()


if __name__ == "__main__":
    from src.core.security import install_redacting_logging

    install_redacting_logging(force=True)
    run_forever()
