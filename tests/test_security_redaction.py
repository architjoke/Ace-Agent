"""Log / secret redaction unit tests."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_redact_openai_groq_nvidia_keys():
    from src.core.security import REDACTED_SECRET, redact_secrets

    blob = (
        "openai sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX1234567890abcd "
        "groq gsk_abcdefghijklmnopqrstuvwxyz123456 "
        "nvidia nvapi-abcdefghijklmnopqrstuvwxyz12"
    )
    out = redact_secrets(blob)
    assert "sk-proj-" not in out
    assert "gsk_" not in out
    assert "nvapi-" not in out
    assert out.count(REDACTED_SECRET) >= 3


def test_redact_email_and_phone():
    from src.core.security import REDACTED_PII, redact_secrets

    out = redact_secrets("Contact me at alice@example.com or +91 98765 43210")
    assert "alice@example.com" not in out
    assert "+91" not in out or REDACTED_PII in out
    assert REDACTED_PII in out


def test_redact_env_secret_values(monkeypatch):
    from src.core.security import REDACTED_SECRET, redact_secrets

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-UNIQUEENVVALUE1234567890")
    out = redact_secrets("using sk-test-UNIQUEENVVALUE1234567890 now")
    assert "UNIQUEENVVALUE" not in out
    assert REDACTED_SECRET in out


def test_redacting_formatter_masks_message():
    from src.core.security import RedactingFormatter, REDACTED_SECRET

    fmt = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="key=gsk_abcdefghijklmnopqrstuvwxyz123456",
        args=(),
        exc_info=None,
    )
    rendered = fmt.format(record)
    assert "gsk_" not in rendered
    assert REDACTED_SECRET in rendered
