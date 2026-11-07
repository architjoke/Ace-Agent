"""Continuous hunt → tailor → apply worker (no company cap)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

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
            logger.exception("Cycle %s crashed — will retry", cycle)
        sleep_for = max(60, int(cycle_sleep_sec))
        logger.info("Sleeping %ss before next hunt…", sleep_for)
        await asyncio.sleep(sleep_for)


def run_forever(
    *,
    max_results_per_track: int | None = None,
    cycle_sleep_sec: int = 900,
) -> None:
    """Apply to as many fit jobs as possible; re-hunt after each batch."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    write_pid(os.getpid())
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(
            _cycle_loop(
                max_results_per_track=max_results_per_track,
                cycle_sleep_sec=cycle_sleep_sec,
            )
        )
    finally:
        clear_pid()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_forever()
