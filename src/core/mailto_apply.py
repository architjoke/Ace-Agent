"""Deterministic mailto: apply — humanized short note + resume PDF via SMTP."""

from __future__ import annotations

import logging
import re
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.core.pitch import pitch_from_snapshot_or_generate

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_mailto(url: str) -> tuple[str, str, str]:
    """Return (to, subject, body_hint) from a mailto: URL."""
    raw = (url or "").strip()
    if not raw.lower().startswith("mailto:"):
        return "", "", ""
    parsed = urlparse(raw)
    to = unquote(parsed.path or "").strip()
    # Some mailto forms use mailto:addr?query with path empty and addr in netloc-ish
    if not to and parsed.netloc:
        to = unquote(parsed.netloc).strip()
    qs = parse_qs(parsed.query or "")
    subject = unquote((qs.get("subject") or [""])[0] or "").strip()
    body = unquote((qs.get("body") or [""])[0] or "").strip()
    # Strip display-name wrappers
    m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", to)
    if m:
        to = m.group(0)
    return to, subject, body


def is_mailto_url(url: str) -> bool:
    return (url or "").strip().lower().startswith("mailto:")


def build_mailto_message(
    *,
    company: str,
    job_title: str,
    jd_text: str,
    yaml_snapshot: str,
    to_hint: str = "",
    subject_hint: str = "",
) -> tuple[str, str, str]:
    """Return (to, subject, body) ready to send."""
    to = (to_hint or "").strip()
    if to and not _EMAIL_RE.match(to):
        m = re.search(r"[\w.+-]+@[\w.-]+\.\w+", to)
        to = m.group(0) if m else ""
    role = (job_title or "Internship").strip()
    co = (company or "your team").strip()
    subject = subject_hint.strip() or f"Application — {role} at {co}"
    pitch = pitch_from_snapshot_or_generate(
        yaml_snapshot,
        company=co,
        job_title=role,
        jd_text=jd_text,
    )
    body = (
        f"{pitch}\n\n"
        f"I've attached my one-page resume (PDF).\n\n"
        f"Best regards,\n"
        f"Archit Gupta\n"
        f"architgupta147@gmail.com · +91 8572855980\n"
        f"https://www.linkedin.com/in/archit-gupta-41379b293"
    )
    return to, subject, body


def send_mailto_application(
    *,
    mailto_url: str,
    company: str,
    job_title: str,
    jd_text: str,
    pdf_path: Path | str,
    yaml_snapshot: str = "",
) -> bool:
    """Send application email with resume attached. Raises on hard config/send errors."""
    from src.core.settings import get_settings

    settings = get_settings()
    address = (settings.email_address or "").strip()
    password = (settings.email_app_password or "").strip()
    if not address or not password:
        raise RuntimeError(
            "mailto apply needs EMAIL_ADDRESS + EMAIL_APP_PASSWORD in .env"
        )

    to_hint, subject_hint, _ = parse_mailto(mailto_url)
    to, subject, body = build_mailto_message(
        company=company,
        job_title=job_title,
        jd_text=jd_text,
        yaml_snapshot=yaml_snapshot,
        to_hint=to_hint,
        subject_hint=subject_hint,
    )
    if not to or not _EMAIL_RE.match(to):
        raise ValueError(f"mailto apply: no valid recipient in {mailto_url!r}")

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"Resume PDF missing: {pdf}")

    msg = MIMEMultipart()
    msg["From"] = address
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    with pdf.open("rb") as fh:
        part = MIMEApplication(fh.read(), Name=pdf.name)
    part["Content-Disposition"] = f'attachment; filename="{pdf.name}"'
    msg.attach(part)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=45) as server:
        server.starttls()
        server.login(address, password)
        server.sendmail(address, [to], msg.as_string())
    logger.info("Mailto apply sent to %s for %s", to, company)
    return True
