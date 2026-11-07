"""Local IMAP inbox scanner → classify recruiter outcomes via text LLM."""

from __future__ import annotations

import email
import imaplib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import Message
from typing import Literal

from pydantic import BaseModel, Field

from src.core.llm_router import invoke_with_failover
from src.core.memory import update_application_outcome
from src.core.settings import get_settings

logger = logging.getLogger(__name__)

_RECRUITER_HINTS = re.compile(
    r"(application|interview|assessment|hackerrank|codility|greenhouse|lever|"
    r"workday|ashby|recruit|career|candidacy|unfortunately|not moving forward|"
    r"next steps|offer|position|role at)",
    re.I,
)
_NOISE_HINTS = re.compile(
    r"(newsletter|unsubscribe|promo|deal|receipt|invoice|otp|verification code|"
    r"security alert|linkedin job alerts digest)",
    re.I,
)

# Actionable assessment / scheduling hosts (allowlist — not bare https)
_ACTION_HOSTS = re.compile(
    r"https?://[^\s<>\"']*(?:"
    r"calendly\.com|"
    r"hackerrank\.com|"
    r"codesignal\.com|"
    r"mettl\.com|"
    r"hirevue\.com|"
    r"greenhouse\.io/|"
    r"lever\.co/"
    r")[^\s<>\"']*",
    re.I,
)
_TRAIL = re.compile(r"[\]\)\.,;:]+$")
_DEADLINE = re.compile(
    r"(?:deadline|complete by|schedule by|expires?(?:\s+on)?|by)\s*[:\-]?\s*"
    r"((?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})|"
    r"(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?))",
    re.I,
)
_HOST_PRIORITY = (
    "calendly.com",
    "hirevue.com",
    "hackerrank.com",
    "codesignal.com",
    "mettl.com",
    "greenhouse.io",
    "lever.co",
)


class EmailOutcome(BaseModel):
    company_name: str = Field(..., description="Employer / ATS company if identifiable")
    outcome: Literal[
        "REJECTED", "INTERVIEW_INVITE", "ASSESSMENT", "UNCLEAR", "SELECTED"
    ] = Field(...)
    reason: str = Field(..., description="One-line why this label was chosen")


@dataclass
class ScannedEmail:
    subject: str
    body: str
    from_addr: str
    date: str


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return str(make_header(decode_header(value.decode("utf-8", errors="replace"))))
        except Exception:
            return value.decode("utf-8", errors="replace")
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


def _body_of(msg: Message) -> str:
    if msg.is_multipart():
        chunks: list[str] = []
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            if ctype in ("text/plain", "text/html"):
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
                if ctype == "text/html":
                    text = re.sub(r"<[^>]+>", " ", text)
                chunks.append(text)
        return "\n".join(chunks)
    payload = msg.get_payload(decode=True) or b""
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def fetch_recent_emails(
    email_address: str | None = None,
    app_password: str | None = None,
    *,
    lookback_hours: int | None = None,
    limit: int = 40,
) -> list[ScannedEmail]:
    """Fetch recent INBOX messages via IMAP (Gmail App Password)."""
    settings = get_settings()
    address = (email_address or settings.email_address).strip()
    password = (app_password or settings.email_app_password).strip()
    hours = lookback_hours if lookback_hours is not None else settings.email_lookback_hours

    if not address or not password:
        raise RuntimeError(
            "EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set in .env for inbox scanning."
        )

    since = (datetime.now(timezone.utc) - timedelta(hours=max(hours, 1))).strftime(
        "%d-%b-%Y"
    )
    client = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
    try:
        client.login(address, password)
        client.select("INBOX")
        status, data = client.search(None, f'(SINCE "{since}")')
        if status != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()
        ids = ids[-limit:]
        out: list[ScannedEmail] = []
        for msg_id in reversed(ids):
            status, payload = client.fetch(msg_id, "(RFC822)")
            if status != "OK" or not payload or not payload[0]:
                continue
            raw = payload[0][1]
            if not isinstance(raw, (bytes, bytearray)):
                continue
            msg = email.message_from_bytes(raw)
            out.append(
                ScannedEmail(
                    subject=_decode(msg.get("Subject")),
                    body=_body_of(msg)[:8000],
                    from_addr=_decode(msg.get("From")),
                    date=_decode(msg.get("Date")),
                )
            )
        return out
    finally:
        try:
            client.logout()
        except Exception:
            pass


def _looks_like_recruiter(subject: str, body: str, from_addr: str) -> bool:
    blob = f"{subject}\n{from_addr}\n{body[:1500]}"
    if _NOISE_HINTS.search(blob) and not _RECRUITER_HINTS.search(blob):
        return False
    return bool(_RECRUITER_HINTS.search(blob))


def classify_email_outcome(email_subject: str, email_body: str) -> EmailOutcome:
    """Structured classification of a recruiter-ish email."""
    prompt = f"""Classify this email about a job application.

Return outcome as one of:
- REJECTED — not selected / not moving forward
- INTERVIEW_INVITE — phone/video/onsite interview ask
- ASSESSMENT — OA / take-home / Hackerrank-style test
- SELECTED — offer or clear positive selection
- UNCLEAR — cannot tell

Subject: {email_subject}

Body:
{email_body[:6000]}
"""
    return invoke_with_failover(
        "inbox",
        prompt,
        temperature=0,
        structured=EmailOutcome,
    )


def extract_action_links(body: str) -> tuple[str | None, str | None]:
    """Pull best assessment/schedule URL + optional deadline from email body."""
    text = body or ""
    found: list[str] = []
    seen: set[str] = set()
    for match in _ACTION_HOSTS.findall(text):
        url = _TRAIL.sub("", match.strip())
        if url and url not in seen:
            seen.add(url)
            found.append(url)

    def _rank(url: str) -> int:
        lower = url.lower()
        for i, host in enumerate(_HOST_PRIORITY):
            if host in lower:
                return i
        return len(_HOST_PRIORITY)

    found.sort(key=_rank)
    action_url = found[0] if found else None

    deadline: str | None = None
    m = _DEADLINE.search(text)
    if m:
        raw = m.group(1).strip()[:64]
        deadline = _normalize_deadline(raw) or raw

    return action_url, deadline


def _normalize_deadline(raw: str) -> str | None:
    """Best-effort ISO-ish normalization; None if unparseable."""
    text = (raw or "").strip()
    if not text:
        return None
    # Already ISO-ish
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text.replace(" ", "T", 1) if " " in text and "T" not in text else text
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def process_inbox(
    *,
    email_address: str | None = None,
    app_password: str | None = None,
    lookback_hours: int | None = None,
    update_db: bool = True,
) -> list[dict]:
    """Fetch → filter → classify → extract action links → optionally update SQLite."""
    messages = fetch_recent_emails(
        email_address,
        app_password,
        lookback_hours=lookback_hours,
    )
    actionable: list[dict] = []
    for msg in messages:
        if not _looks_like_recruiter(msg.subject, msg.body, msg.from_addr):
            continue
        try:
            result = classify_email_outcome(msg.subject, msg.body)
        except Exception:
            logger.exception("Failed to classify: %s", msg.subject[:120])
            continue

        action_url, action_deadline = None, None
        if result.outcome in {"INTERVIEW_INVITE", "ASSESSMENT", "SELECTED"}:
            action_url, action_deadline = extract_action_links(msg.body)

        record = {
            "company_name": result.company_name,
            "outcome": result.outcome,
            "reason": result.reason,
            "subject": msg.subject,
            "from_addr": msg.from_addr,
            "action_url": action_url,
            "action_deadline": action_deadline,
        }
        if update_db and result.outcome != "UNCLEAR":
            matched = update_application_outcome(
                result.company_name,
                result.outcome,
                reason=result.reason,
                action_url=action_url or "",
                action_deadline=action_deadline or "",
            )
            record["db_matches"] = matched
            logger.info(
                "Inbox outcome %s @ %s (matched=%s, action_url=%s)",
                result.outcome,
                result.company_name,
                matched,
                action_url or "-",
            )
        actionable.append(record)

    logger.info("Inbox scan produced %s actionable outcomes", len(actionable))
    return actionable
