"""Status desk — answer questions about applies, manuals, and agent health."""

from __future__ import annotations

import re
from pathlib import Path

from src.core.config import OUTPUT_DIR, PROJECT_ROOT, WORKER_PID
from src.core.memory import application_counts, init_db, list_applications, stats_since
from src.core.settings import get_settings

REPO_ROOT = PROJECT_ROOT
PID_FILE = WORKER_PID

try:
    LOG_FILE = Path(get_settings().output_dir) / "ace_worker.log"
except Exception:
    LOG_FILE = OUTPUT_DIR / "ace_worker.log"


def worker_running() -> tuple[bool, int | None]:
    """Return (alive, pid) for the background apply worker."""
    if not PID_FILE.is_file():
        return False, None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return False, None
    try:
        # Windows: OpenProcess check via os.kill(pid, 0) is unreliable; use ctypes-free probe.
        import psutil  # optional

        alive = psutil.pid_exists(pid)
        return alive, pid if alive else None
    except ImportError:
        pass
    # Fallback: tasklist
    try:
        import subprocess

        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        alive = str(pid) in out and "No tasks" not in out
        return alive, pid if alive else None
    except Exception:
        return False, pid


def manual_packages() -> list[dict]:
    """Filesystem Manual_Action packages + DB rows."""
    settings = get_settings()
    root = settings.output_dir / "Manual_Action"
    packages: list[dict] = []
    if root.is_dir():
        for folder in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not folder.is_dir():
                continue
            guide = folder / "application_guide.md"
            pdf = next(folder.glob("*.pdf"), None)
            packages.append(
                {
                    "company": folder.name.replace("_", " "),
                    "path": str(folder),
                    "has_guide": guide.is_file(),
                    "pdf": str(pdf) if pdf else None,
                }
            )
    return packages


def format_status() -> str:
    init_db()
    counts = application_counts()
    today = stats_since(hours=24)
    alive, pid = worker_running()
    manuals = manual_packages()
    settings = get_settings()
    from src.core.browser_pool import read_worker_status
    from src.core.queue import init_queue, queue_counts

    init_queue()
    q = queue_counts()
    wstatus = read_worker_status().get("workers") or {}
    lines = [
        "Ace Agent status",
        "────────────────",
        f"Worker: {'RUNNING (pid ' + str(pid) + ')' if alive else 'stopped'}",
        (
            f"Parallel: browsers={settings.browser_workers} "
            f"tailor={settings.tailor_workers} "
            f"groq={settings.groq_concurrency}"
        ),
        (
            f"Apply queue: READY={q.get('READY', 0)} "
            f"CLAIMED={q.get('CLAIMED', 0)} DONE={q.get('DONE', 0)}"
        ),
    ]
    if wstatus:
        busy = []
        idle = 0
        for wid, meta in sorted(wstatus.items()):
            st = str((meta or {}).get("status") or "?")
            company = str((meta or {}).get("company") or "")
            if st == "busy":
                busy.append(f"{wid}:{company or '…'}")
            elif st == "idle":
                idle += 1
        lines.append(
            "Browser workers: "
            + (f"{idle} idle" if idle else "0 idle")
            + (f"; busy [{', '.join(busy)}]" if busy else "")
        )
    lines.extend(
        [
            f"Unique companies touched: {counts['unique_companies']}",
            f"Auto-submitted (SUCCESS): {counts['staged_ready']}",
            f"Manual Action (couldn't submit): {counts['manual_action']} (folders: {len(manuals)})",
            f"Already applied (skipped): {counts.get('already_applied', 0)}",
            f"Timeout/stuck:            {counts.get('failed_timeout_or_stuck', 0)}",
            f"Interrupted (reclaim):    {counts.get('failed_interrupted', 0)}",
            f"Needs human (soft-fail):  {counts.get('requires_human_intervention', 0)}",
            f"Job closed/missing:       {counts.get('job_closed', 0)}",
            f"Failed:                   {counts['failed']}",
            f"Total attempted:          {counts['attempted']}",
            f"Awaiting recruiter reply: {counts['pending_outcomes']}",
            "",
            f"Last 24h applied/staged:  {today['applied_today']}",
            f"Last 24h invites:         {today['interview_invites_today']}",
            f"Last 24h rejections:      {today['rejections_today']}",
        ]
    )
    if manuals:
        lines.append("")
        lines.append("Manual applies waiting for you:")
        for row in manuals[:15]:
            lines.append(f"  • {row['company']}  →  {row['path']}")
        if len(manuals) > 15:
            lines.append(f"  … +{len(manuals) - 15} more (run: ace ask manual list)")
    return "\n".join(lines)


def format_manual_list() -> str:
    packages = manual_packages()
    db_rows = list_applications(status="MANUAL_ACTION", limit=50)
    if not packages and not db_rows:
        return "No manual-apply packages yet."
    lines = ["Manual apply queue", "──────────────────"]
    for row in packages:
        flag = "guide✓" if row["has_guide"] else "no-guide"
        lines.append(f"• {row['company']}  [{flag}]")
        lines.append(f"  {row['path']}")
        if row.get("pdf"):
            lines.append(f"  resume: {row['pdf']}")
    if db_rows:
        lines.append("")
        lines.append("From memory DB:")
        for r in db_rows[:30]:
            lines.append(
                f"• {r['company']} | {r.get('track') or '?'} | {r.get('timestamp', '')[:19]}"
            )
            lines.append(f"  {r.get('job_url', '')}")
    return "\n".join(lines)


def format_companies(limit: int = 40) -> str:
    rows = list_applications(limit=limit)
    if not rows:
        return "No applications logged yet."
    lines = ["Recent companies", "────────────────"]
    for r in rows:
        lines.append(
            f"• {r['company']}  [{r['status']}]  outcome={r.get('outcome_status', 'PENDING')}"
        )
    return "\n".join(lines)


def answer(question: str) -> str:
    """Rule-based Q&A over agent memory (no LLM needed)."""
    q = (question or "").strip().lower()
    if not q:
        return format_status()

    if any(w in q for w in ("manual", "fallback", "need me", "to submit", "package")):
        if any(w in q for w in ("how many", "count", "number")):
            n = len(manual_packages())
            return f"{n} manual-apply package(s) waiting under output/Manual_Action."
        return format_manual_list()

    if any(w in q for w in ("left", "remaining", "todo", "pending for me")):
        manuals = manual_packages()
        counts = application_counts()
        return (
            f"Your queue: {len(manuals)} manual submit(s) left.\n"
            f"Agent staged READY_FOR_REVIEW: {counts['staged_ready']}.\n"
            f"Still waiting on recruiter replies: {counts['pending_outcomes']}.\n\n"
            + (format_manual_list() if manuals else "No manual folders right now.")
        )

    if any(w in q for w in ("how many", "count", "applied", "status", "progress", "summary")):
        return format_status()

    if any(w in q for w in ("list", "companies", "which", "show")):
        if "fail" in q:
            rows = list_applications(status="FAILED", limit=40)
            if not rows:
                return "No failed applications."
            return "Failed applies:\n" + "\n".join(
                f"• {r['company']} — {r.get('error_log', '')[:120]}" for r in rows
            )
        if "success" in q or "staged" in q or "ready" in q:
            rows = list_applications(status="SUCCESS", limit=40)
            if not rows:
                return "No staged READY_FOR_REVIEW applications yet."
            return "Staged (READY_FOR_REVIEW):\n" + "\n".join(
                f"• {r['company']} — {r.get('job_url', '')}" for r in rows
            )
        return format_companies()

    if any(w in q for w in ("running", "alive", "worker", "daemon")):
        alive, pid = worker_running()
        return (
            f"Background worker is RUNNING (pid {pid}). Log: {LOG_FILE}"
            if alive
            else "Background worker is stopped. Start with: ace start"
        )

    if "help" in q or q in {"?", "hi", "hello"}:
        return (
            "Ask me things like:\n"
            "  ace ask how many applied\n"
            "  ace ask list manual applies\n"
            "  ace ask what is left for me\n"
            "  ace ask is the worker running\n"
            "  ace status\n"
            "  ace start          # hunt→tailor→apply loop (background)\n"
            "  ace stop\n"
        )

    # fuzzy fallback
    if re.search(r"\b(apply|application|job|company)\b", q):
        return format_status()
    return (
        "Not sure — try: ace status | ace ask manual list | ace ask how many applied\n\n"
        + format_status()
    )


def write_pid(pid: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")


def clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except TypeError:
        if PID_FILE.exists():
            PID_FILE.unlink()
