"""Background scheduler: pipeline → inbox → rejection retrain → daily brief.

Usage (from repo root):
  python scripts/daemon.py
  ace daemon
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import schedule
from dotenv import load_dotenv

from src.core.config import ENV_FILE

load_dotenv(ENV_FILE)

from src.core.inbox_scanner import process_inbox
from src.core.reflect import analyze_rejections_and_retrain, daily_checkout_and_train
from src.core.report import generate_daily_brief
from src.core.settings import reload_settings
from src.main import run_daily_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ace_daemon")


def _morning_job() -> None:
    logger.info("Pipeline starting")
    try:
        code = run_daily_pipeline()
        logger.info("Pipeline finished (exit=%s)", code)
    except Exception:
        logger.exception("Pipeline crashed")


def _inbox_job() -> None:
    logger.info("Inbox scan starting")
    try:
        results = process_inbox()
        logger.info("Inbox scan finished (%s actionable)", len(results))
    except Exception:
        logger.exception("Inbox scan crashed")


def _retrain_job() -> None:
    logger.info("Retrain starting")
    try:
        apply_rules = daily_checkout_and_train()
        logger.info("Apply reflection: %s", str(apply_rules)[:300])
        rejection_rules = analyze_rejections_and_retrain()
        logger.info("Rejection retrain: %s", str(rejection_rules)[:300])
    except Exception:
        logger.exception("Retrain crashed")


def _report_job() -> None:
    logger.info("Daily brief starting")
    try:
        path = generate_daily_brief()
        logger.info("Daily brief ready → %s", path)
    except Exception:
        logger.exception("Daily brief crashed")


def start_daemon() -> None:
    from src.core.security import install_redacting_logging

    install_redacting_logging(force=True)
    settings = reload_settings()
    schedule.clear()
    schedule.every().day.at(settings.pipeline_time).do(_morning_job)
    schedule.every().day.at(settings.inbox_time).do(_inbox_job)
    schedule.every().day.at(settings.retrain_time).do(_retrain_job)
    schedule.every().day.at(settings.report_time).do(_report_job)
    logger.info(
        "Daemon armed (local time) — pipeline %s | inbox %s | retrain %s | report %s. "
        "Ctrl+C to stop. Keep Chromium profile logged into LinkedIn/Gmail/ATS.",
        settings.pipeline_time,
        settings.inbox_time,
        settings.retrain_time,
        settings.report_time,
    )
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Daemon interrupted — shutting down cleanly")
        raise


if __name__ == "__main__":
    try:
        start_daemon()
    except KeyboardInterrupt:
        print("\nDaemon stopped.")
        sys.exit(0)
