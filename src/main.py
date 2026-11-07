"""Ace Agent orchestrator: hunt → triage → LangGraph tailor → browser-use apply → fallback.

Default path is the parallel pipeline (multi-browser workers + apply queue).
``process_job`` remains for one-off / legacy serial calls.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import ENV_FILE, PROJECT_ROOT

REPO_ROOT = PROJECT_ROOT
load_dotenv(ENV_FILE)

from src.core.discovery import pick_apply_url
from src.core.memory import has_applied
from src.core.pipeline import apply_job, run_parallel, tailor_job, triage_job
from src.core.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ace_agent")


async def process_job(job, master_yaml: Path, master_text: str, min_fit: int) -> str:
    """Serial triage → tailor → apply (legacy one-at-a-time path)."""
    triage = await asyncio.to_thread(triage_job, job, master_text, min_fit)
    if triage is None:
        job_url = pick_apply_url(job)
        if not job_url or not str(job.get("description") or "").strip():
            return "failed"
        if has_applied(job_url):
            return "skipped"
        return "dropped"

    tailored = await asyncio.to_thread(tailor_job, triage, master_yaml)
    if tailored is None:
        return "failed"

    return await apply_job(
        company=tailored["company"],
        job_url=tailored["job_url"],
        track=tailored["track"],
        jd_text=tailored["description"],
        pdf_path=tailored["pdf_path"],
        yaml_snapshot=tailored.get("yaml_snapshot") or "",
        master_yaml=master_yaml,
        cleanup_linkedin_resume=("linkedin.com" in tailored["job_url"].lower()),
        job_url_direct=str(tailored.get("job_url_direct") or ""),
    )


async def run(
    location: str | None = None,
    max_results_per_track: int | None = None,
    master_yaml: Path | None = None,
    target_companies: int | None = None,
) -> int:
    """Parallel pipeline entrypoint."""
    return await run_parallel(
        location=location,
        max_results_per_track=max_results_per_track,
        master_yaml=master_yaml,
        target_companies=target_companies,
    )


def run_daily_pipeline(
    location: str | None = None,
    max_results_per_track: int | None = None,
    master_yaml: Path | None = None,
    target_companies: int | None = None,
) -> int:
    """Sync entrypoint for the daemon / scheduler."""
    return asyncio.run(
        run(
            location=location,
            max_results_per_track=max_results_per_track,
            master_yaml=master_yaml,
            target_companies=target_companies,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Track-aware Ace Agent: parallel hunt → fit triage → LangGraph tailor → "
            "multi-browser apply → manual fallback."
        ),
    )
    parser.add_argument(
        "--location",
        default=None,
        help=f"Search location (default from config: {settings.primary_location!r})",
    )
    parser.add_argument(
        "--max-results-per-track",
        type=int,
        default=None,
        help="Max listings per track (default from config)",
    )
    parser.add_argument(
        "--master-yaml",
        type=Path,
        default=None,
        help=f"Path to RenderCV master YAML (default: {settings.master_yaml})",
    )
    parser.add_argument(
        "--target-companies",
        type=int,
        default=None,
        help="Stop after N queued apply packages",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_daily_pipeline(
        location=args.location,
        max_results_per_track=args.max_results_per_track,
        master_yaml=args.master_yaml,
        target_companies=args.target_companies,
    )


if __name__ == "__main__":
    sys.exit(main())
