"""Ace Job Agent — single entrypoint.

Usage:
  ace start              # background: hunt → tailor → apply (as many as possible)
  ace stop               # stop background worker
  ace status             # counts + worker health
  ace manual             # list Manual_Action packages
  ace ask <question>     # ask the agent about applies / manuals / progress
  ace login              # bootstrap Chromium logins once
  ace daemon             # scheduled 08:00 / evening jobs
  ace once               # one hunt batch now (foreground)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ace_cli")


def _start_worker(max_results: int | None, cycle_sleep: int) -> int:
    from src.core.desk import LOG_FILE, worker_running

    alive, pid = worker_running()
    if alive:
        print(f"Worker already running (pid {pid}). Log: {LOG_FILE}")
        return 0

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Clone login profile → w1..wn and clear Chromium singleton locks.
    try:
        from src.core.browser_pool import clear_singleton_locks, clone_profiles_from_master
        from src.core.settings import get_settings

        settings = get_settings()
        clear_singleton_locks(settings.browser_user_data_dir)
        clone_profiles_from_master(settings.browser_workers)
    except Exception as exc:
        logger.warning("Could not prepare browser worker profiles: %s", exc)
        profile = REPO / "data" / "browser_profile"
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            try:
                (profile / name).unlink(missing_ok=True)
            except TypeError:
                p = profile / name
                if p.exists():
                    p.unlink()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    args = [
        sys.executable,
        "-c",
        (
            "from src.core.worker import run_forever; "
            f"run_forever(max_results_per_track={max_results!r}, "
            f"cycle_sleep_sec={int(cycle_sleep)})"
        ),
    ]
    # Background worker: no flashing console windows on Windows.
    creationflags = 0
    startupinfo = None
    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE

    log_f = open(LOG_FILE, "a", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        cwd=str(REPO),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        startupinfo=startupinfo,
        close_fds=True,
    )
    from src.core.desk import write_pid

    write_pid(proc.pid)
    print(f"Ace worker started (pid {proc.pid})")
    print(f"Log → {LOG_FILE}")
    print("Ask anytime:  ace status  |  ace ask how many applied  |  ace manual")
    return 0


def _stop_worker() -> int:
    from src.core.desk import clear_pid, worker_running

    alive, pid = worker_running()
    if not alive or pid is None:
        clear_pid()
        print("Worker was not running.")
        return 0
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.kill(pid, 15)
    except Exception as exc:
        print(f"Could not stop pid {pid}: {exc}")
        return 1
    clear_pid()
    print(f"Stopped worker pid {pid}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Natural: `ace ask how many applied` without quotes
    if argv and argv[0].lower() == "ask" and len(argv) > 1:
        from src.core.desk import answer

        print(answer(" ".join(argv[1:])))
        return 0

    parser = argparse.ArgumentParser(
        prog="ace",
        description="Ace Job Agent — hunt, tailor, apply, and answer status questions.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_start = sub.add_parser("start", help="Background apply loop (uncapped)")
    p_start.add_argument("--max-results-per-track", type=int, default=None)
    p_start.add_argument(
        "--cycle-sleep",
        type=int,
        default=900,
        help="Seconds between hunt cycles (default 900)",
    )

    sub.add_parser("stop", help="Stop background worker")
    sub.add_parser("status", help="Show apply counts + manual queue")
    sub.add_parser("manual", help="List Manual_Action packages")
    p_ask = sub.add_parser("ask", help="Ask a status question")
    p_ask.add_argument("question", nargs="+")

    p_once = sub.add_parser("once", help="One hunt→apply batch in foreground")
    p_once.add_argument("--max-results-per-track", type=int, default=None)
    p_once.add_argument("--target-companies", type=int, default=None)

    sub.add_parser("daemon", help="24/7 schedule (08:00 pipeline + evening jobs)")
    p_login = sub.add_parser("login", help="Open Chromium to log into Gmail/LinkedIn/ATS")
    p_login.add_argument("--verify", action="store_true")

    sub.add_parser("brief", help="Write today's Markdown brief")
    sub.add_parser("inbox", help="Scan recruiter inbox now")
    sub.add_parser(
        "chat",
        help="HITL Self-Training Copilot (mark submitted, clean Manual_Action, memorize rules)",
    )

    # Legacy flags still work when no subcommand
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--bootstrap-login", action="store_true")
    parser.add_argument("--verify-login", action="store_true")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--scan-inbox", action="store_true")
    parser.add_argument("--target-companies", type=int, default=None)
    parser.add_argument("--max-results-per-track", type=int, default=None)
    parser.add_argument("--dry-run", type=str, default=None)
    parser.add_argument("--company", default="DryRunCo")
    parser.add_argument("--jd", default="")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--reflect", action="store_true")

    args = parser.parse_args(argv)

    cmd = (args.cmd or "").lower()

    if not argv:
        from src.core.desk import format_status

        print(parser.format_help())
        print(format_status())
        print("\nStart applying:  ace start")
        return 0

    if cmd == "start":
        return _start_worker(args.max_results_per_track, args.cycle_sleep)

    if cmd == "stop":
        return _stop_worker()

    if cmd == "status":
        from src.core.desk import format_status

        print(format_status())
        return 0

    if cmd == "manual":
        from src.core.desk import format_manual_list

        print(format_manual_list())
        return 0

    if cmd == "ask":
        from src.core.desk import answer

        print(answer(" ".join(args.question)))
        return 0

    if cmd == "once" or args.run_once:
        from src.main import run_daily_pipeline

        return run_daily_pipeline(
            max_results_per_track=args.max_results_per_track,
            target_companies=args.target_companies,
        )

    if cmd == "daemon" or args.daemon:
        from scripts.daemon import start_daemon

        start_daemon()
        return 0

    if cmd == "login" or args.bootstrap_login:
        from scripts.bootstrap_login import run_bootstrap

        verify = bool(getattr(args, "verify", False) or args.verify_login)
        return run_bootstrap(verify=verify, wait_enter=True)

    if cmd == "brief" or args.brief:
        from src.core.memory import init_db
        from src.core.report import generate_daily_brief

        init_db()
        path = generate_daily_brief()
        print(f"Brief → {path}")
        return 0

    if cmd == "chat":
        from src.core.copilot import start_chat_loop

        return start_chat_loop()

    if cmd == "inbox" or args.scan_inbox:
        from src.core.inbox_scanner import process_inbox
        from src.core.memory import init_db

        init_db()
        results = process_inbox()
        print(f"Actionable outcomes: {len(results)}")
        for row in results:
            print(
                f"- {row.get('outcome')} @ {row.get('company_name')} "
                f"(db_matches={row.get('db_matches', 0)})"
            )
        return 0

    if args.dry_run:
        from src.core.settings import get_settings
        from scripts.test_live_run import dry_run

        settings = get_settings()
        jd = (args.jd or "").strip() or (
            "Software Engineer Intern. Python, JavaScript/TypeScript, DSA, REST APIs. "
            "Paid internship. India / remote-friendly."
        )
        return asyncio.run(
            dry_run(
                test_url=args.dry_run,
                test_jd=jd,
                company=args.company,
                master_yaml=settings.master_yaml,
            )
        )

    if args.retrain or args.reflect:
        from src.core.memory import init_db
        from src.core.reflect import analyze_rejections_and_retrain, daily_checkout_and_train

        init_db()
        if args.reflect:
            from src.core.inbox_scanner import process_inbox

            results = process_inbox()
            print(f"Inbox actionable outcomes: {len(results)}")
        print(daily_checkout_and_train())
        print(analyze_rejections_and_retrain())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
