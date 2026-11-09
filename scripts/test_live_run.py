"""Isolated critical-path dry-runs.

# debug only — not part of the production ace CLI / daemon path.

Usage (from repo root):
  python scripts/test_live_run.py
  python scripts/test_live_run.py --full
  python scripts/test_live_run.py --url "https://..."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/test_live_run.py` from any CWD
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

from src.core.config import ENV_FILE, OUTPUT_DIR, PROJECT_ROOT, resolve_master_yaml

load_dotenv(ENV_FILE)

from src.core.executor import execute_application
from src.core.fallback import generate_manual_package
from src.core.tailor import tailor_with_graph

REPO_ROOT = PROJECT_ROOT
DEFAULT_MASTER = resolve_master_yaml()
DEFAULT_STATIC_PDF = OUTPUT_DIR / "HackerRankOnePage" / "Archit_Gupta_CV.pdf"

DEFAULT_TEST_URL = "https://job-boards.greenhouse.io/hackerrank/jobs/8229735"
DEFAULT_COMPANY = "HackerRank"
DEFAULT_JD = """
Software Development Engineer Intern — HackerRank.

We are looking for an entry-level / internship Software Development Engineer
comfortable with Python or JavaScript/TypeScript, data structures and algorithms,
REST APIs, and collaborative software engineering. Build backend services, write
tests, and ship product features with mentorship. Paid internship. Location:
remote-friendly / India-eligible engineering internship track.
""".strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ace_dry_run")


async def browser_only_run(
    *,
    test_url: str,
    company: str,
    pdf_path: Path,
    master_yaml: Path,
) -> int:
    """Isolate the vision agent: upload a known PDF, stop at READY_FOR_REVIEW."""
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        logger.error("Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running.")
        return 1

    pdf_path = pdf_path.expanduser().resolve()
    master_yaml = master_yaml.expanduser().resolve()
    if not pdf_path.is_file():
        logger.error("Static PDF missing: %s", pdf_path)
        return 1
    if not master_yaml.is_file():
        logger.error("Master YAML missing: %s", master_yaml)
        return 1
    try:
        from src.core.tailor import assert_one_page_pdf

        assert_one_page_pdf(pdf_path)
    except Exception as exc:
        logger.error("1-page gate failed for %s: %s", pdf_path, exc)
        return 1

    print("[*] BROWSER-ONLY isolation (no tailor)")
    print(f"    URL:  {test_url}")
    print(f"    PDF:  {pdf_path}")
    print(f"    Co:   {company}")
    print("    Keep hands off the mouse. Expect READY_FOR_REVIEW (no final Submit).")
    try:
        success = await execute_application(test_url, pdf_path, master_yaml)
        if success:
            print(
                f"[✓] Staged application for {company}. "
                "READY_FOR_REVIEW — waiting for human submit."
            )
            return 0
        raise RuntimeError("execute_application returned False")
    except Exception as exc:
        print(f"[!] Autonomous apply failed: {exc}")
        print("[!] Triggering Fallback Package...")
        package = generate_manual_package(
            company=company,
            job_url=test_url,
            jd=DEFAULT_JD,
            tailored_pdf_path=pdf_path,
            master_yaml_path=master_yaml,
        )
        print(f"[✓] Fallback package generated at: {package}")
        return 1


async def dry_run(
    *,
    test_url: str,
    test_jd: str,
    company: str,
    master_yaml: Path,
) -> int:
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        logger.error("Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running.")
        return 1

    master_yaml = master_yaml.expanduser().resolve()
    if not master_yaml.is_file():
        logger.error("Master YAML missing: %s", master_yaml)
        return 1

    master_text = master_yaml.read_text(encoding="utf-8")

    print("[*] 1. Testing LangGraph ATS tailor (track-aware)...")
    pdf_path, _yaml = tailor_with_graph(
        test_jd,
        master_text,
        company=company,
        job_track="SDE",
    )
    print(f"[✓] PDF generated at: {pdf_path}")

    print("[*] 2. Testing Browser-Use Execution...")
    print("    Keep hands off the mouse. Expect READY_FOR_REVIEW (no final Submit).")
    try:
        success = await execute_application(test_url, pdf_path, master_yaml)
        if success:
            print(
                f"[✓] Agent successfully staged application for {company}. "
                "READY_FOR_REVIEW — waiting for human submit."
            )
            return 0
        raise RuntimeError("execute_application returned False")
    except Exception as exc:
        print(f"[!] Autonomous apply failed: {exc}")
        print("[!] Triggering Fallback Package...")
        package = generate_manual_package(
            company=company,
            job_url=test_url,
            jd=test_jd,
            tailored_pdf_path=pdf_path,
            master_yaml_path=master_yaml,
        )
        print(f"[✓] Fallback package generated at: {package}")
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ace Agent live dry-run (single job).")
    parser.add_argument("--url", default=DEFAULT_TEST_URL, help="Greenhouse/Lever job URL")
    parser.add_argument("--company", default=DEFAULT_COMPANY)
    parser.add_argument("--jd", default=DEFAULT_JD, help="Job description text")
    parser.add_argument("--jd-file", type=Path, default=None, help="Optional JD text file")
    parser.add_argument("--master-yaml", type=Path, default=DEFAULT_MASTER)
    parser.add_argument(
        "--browser-only",
        action="store_true",
        default=True,
        help="Skip tailor; use --pdf to isolate browser-use",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run LangGraph tailor before browser apply",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_STATIC_PDF,
        help="Static PDF for browser-only dry-run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.full:
        jd = args.jd
        if args.jd_file:
            jd = Path(args.jd_file).read_text(encoding="utf-8")
        return asyncio.run(
            dry_run(
                test_url=args.url,
                test_jd=jd,
                company=args.company,
                master_yaml=args.master_yaml,
            )
        )
    return asyncio.run(
        browser_only_run(
            test_url=args.url,
            company=args.company,
            pdf_path=args.pdf,
            master_yaml=args.master_yaml,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
