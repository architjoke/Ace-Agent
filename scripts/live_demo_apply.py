"""Live demo: tailor + auto-submit a few career jobs (headed browser).

Usage (from repo root):
  python scripts/live_demo_apply.py
  python scripts/live_demo_apply.py --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.core.config import ENV_FILE  # noqa: E402

load_dotenv(ENV_FILE)

from src.core.discovery import pick_apply_url
from src.core.memory import has_applied, init_db
from src.core.settings import get_settings
from src.main import process_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ace_demo")


def _score_row(row: pd.Series) -> int:
    blob = f"{row.get('title') or ''} {str(row.get('description') or '')[:800]}".lower()
    keys = (
        "python",
        "fastapi",
        "flask",
        "javascript",
        "typescript",
        "node",
        "backend",
        "full stack",
        "full-stack",
        "software",
        "intern",
        "react",
    )
    return sum(1 for k in keys if k in blob)


def _pick_jobs(limit: int) -> pd.DataFrame:
    csv_path = REPO / "output" / "jobs.csv"
    if not csv_path.is_file():
        raise SystemExit("No output/jobs.csv — run discovery once first (ace once).")
    df = pd.read_csv(csv_path)
    scored: list[tuple[int, dict]] = []
    for _, row in df.iterrows():
        title = str(row.get("title") or "").lower()
        site = str(row.get("site") or "").lower()
        if not any(k in title for k in ("intern", "junior", "graduate", "trainee")):
            continue
        url = pick_apply_url(row)
        # Prefer LinkedIn listing URL for Easy Apply demos
        listing = str(row.get("job_url") or "").strip()
        if "linkedin.com" in listing.lower():
            url = listing
        if not url or has_applied(url):
            continue
        if "glassdoor" in url.lower():
            continue
        item = row.to_dict()
        item["job_url"] = url
        # Boost LinkedIn so Easy Apply is visible in the headed window
        boost = 5 if site == "linkedin" or "linkedin.com" in url.lower() else 0
        company = str(row.get("company") or "").lower()
        # Prefer backend / SDE-shaped internships that match the fact bank
        for needle, pts in (
            ("digiplus", 8),
            ("nextloop", 8),
            ("rempact", 7),
            ("vahan", 6),
            ("geminisoft", 6),
            ("viionn", 9),
            ("qubehealth", 5),
            ("kaleris", 5),
        ):
            if needle in company:
                boost += pts
        scored.append((_score_row(row) + boost, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    rows = [item for _, item in scored[:limit]]
    return pd.DataFrame(rows)


async def main(limit: int, min_fit: int) -> int:
    init_db()
    settings = get_settings()
    master = settings.master_yaml
    master_text = master.read_text(encoding="utf-8")
    jobs = _pick_jobs(limit)
    if jobs.empty:
        logger.error("No unused applyable jobs found in jobs.csv")
        return 1
    print("\nDemo will tailor + auto-submit these (watch the Chromium window):\n")
    for _, j in jobs.iterrows():
        print(f"  • {j.get('company')} — {j.get('title')}")
        print(f"    {j.get('job_url')}\n")
    counts = {"success": 0, "fallback": 0, "dropped": 0, "failed": 0, "skipped": 0}
    for _, job in jobs.iterrows():
        outcome = await process_job(job, master, master_text, min_fit)
        counts[outcome] = counts.get(outcome, 0) + 1
        print(f"→ {job.get('company')}: {outcome}")
    print("\nDemo done:", counts)
    print("ace status  # to refresh counts")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--min-fit",
        type=int,
        default=55,
        help="Lower fit gate for demo so browser apply is reachable",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.limit, args.min_fit)))
