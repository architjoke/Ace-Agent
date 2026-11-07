"""Daily analytics brief (Markdown + optional SMTP)."""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

from src.core.memory import stats_since
from src.core.reflect import load_lessons
from src.core.settings import get_settings

logger = logging.getLogger(__name__)


def generate_daily_brief(*, hours: int = 24) -> Path:
    """Write output/Daily_Brief_YYYY-MM-DD.md and return its path."""
    settings = get_settings()
    stats = stats_since(hours=hours)
    lessons = load_lessons()
    # Keep brief readable: last ~40 lines of lessons
    lesson_lines = lessons.splitlines()
    lesson_tail = "\n".join(lesson_lines[-40:]) if lesson_lines else "_No lessons yet._"

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""# Ace Agent Daily Brief — {day}

Generated: {stamp}

## Pipeline (last {hours}h)

| Metric | Count |
|---|---|
| Applied / staged | {stats["applied_today"]} |
| Rejections processed | {stats["rejections_today"]} |
| Interview invites | {stats["interview_invites_today"]} |
| Assessments | {stats["assessments_today"]} |
| Still pending outcome | {stats["pending_outcomes"]} |

## Latest self-training rules (tail)

{lesson_tail}

## Next actions for you

1. Open any `output/Manual_Action_*` folders and submit if needed.
2. Confirm interview / assessment emails in your inbox.
3. Leave the daemon running so tomorrow's 08:00 run reads updated lessons.
"""
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"Daily_Brief_{day}.md"
    path.write_text(body, encoding="utf-8")
    logger.info("Daily brief written → %s", path)

    if settings.send_brief:
        try:
            email_daily_brief_to_user(body)
        except Exception:
            logger.exception("Failed to email daily brief")

    return path


def email_daily_brief_to_user(markdown_body: str | None = None) -> None:
    """Send the brief via SMTP using EMAIL_ADDRESS + EMAIL_APP_PASSWORD."""
    settings = get_settings()
    address = settings.email_address
    password = settings.email_app_password
    if not address or not password:
        raise RuntimeError("EMAIL_ADDRESS / EMAIL_APP_PASSWORD required to send brief")

    if markdown_body is None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = settings.output_dir / f"Daily_Brief_{day}.md"
        if not path.is_file():
            path = generate_daily_brief()
        markdown_body = path.read_text(encoding="utf-8")

    msg = MIMEText(markdown_body, "plain", "utf-8")
    msg["Subject"] = f"Ace Agent Daily Brief — {datetime.now(timezone.utc).date()}"
    msg["From"] = address
    msg["To"] = address

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(address, password)
        server.sendmail(address, [address], msg.as_string())
    logger.info("Daily brief emailed to %s", address)
