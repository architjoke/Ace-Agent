"""Zero-Trust controls: log redaction, untrusted-input firewall, path jail."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from src.core.config import (
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    THEMES_DIR,
)

logger = logging.getLogger(__name__)

REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_PII = "[REDACTED_PII]"
REDACTED_PROXY = "[REDACTED_PROXY]://"

# ── Phase 1: secret / PII redaction ─────────────────────────────────

_SECRET_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"), REDACTED_SECRET),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), REDACTED_SECRET),
    (re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}\b"), REDACTED_SECRET),
    (re.compile(r"\bcohere_[A-Za-z0-9]{20,}\b"), REDACTED_SECRET),
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"), REDACTED_SECRET),
    (re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}\b"), REDACTED_SECRET),
    (
        re.compile(r"(?i)\b(?:https?|socks5?):\/\/[^\/\s:]+:[^\/\s@]+@"),
        REDACTED_PROXY,
    ),
    # Gmail-style 16-char app passwords (often spaced 4x4)
    (re.compile(r"\b(?:[a-z]{4}\s+){3}[a-z]{4}\b"), REDACTED_SECRET),
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        REDACTED_PII,
    ),
    # Phone-ish (E.164 / spaced IN mobiles)
    (re.compile(r"(?<!\w)(?:\+?\d[\d\-\s()]{8,}\d)(?!\w)"), REDACTED_PII),
]
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|api[_-]?key|token|password|secret)\s*[:=]\s*\S+"
)

_ENV_SECRET_KEY_RE = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PROXY|PASS)", re.I
)

_PII_CACHE: list[str] | None = None
_LOGGING_INSTALLED = False


def _env_secret_values() -> list[str]:
    out: list[str] = []
    for k, v in os.environ.items():
        if not v or len(v) < 8:
            continue
        if not _ENV_SECRET_KEY_RE.search(k):
            continue
        out.append(v)
        # Also scrub space-stripped / quote-stripped variants
        compact = v.strip().strip('"').strip("'")
        if compact and compact != v and len(compact) >= 8:
            out.append(compact)
        nospace = re.sub(r"\s+", "", compact)
        if nospace and nospace != compact and len(nospace) >= 8:
            out.append(nospace)
    # Longest first so overlapping values redact fully
    out.sort(key=len, reverse=True)
    return out


def _pii_substrings() -> list[str]:
    global _PII_CACHE
    if _PII_CACHE is not None:
        return _PII_CACHE
    found: list[str] = []
    try:
        import yaml

        from src.core.config import resolve_master_yaml

        path = resolve_master_yaml()
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                answers = data.get("application_answers")
                cv = data.get("cv")
                blobs: list[Any] = []
                if isinstance(answers, dict):
                    blobs.append(answers)
                if isinstance(cv, dict):
                    blobs.append(cv)
                keys = (
                    "phone",
                    "phone_display",
                    "phone_e164",
                    "phone_national",
                    "email",
                    "address_line",
                    "full_location",
                    "pincode",
                    "postal_code",
                    "city",
                    "state",
                )
                for blob in blobs:
                    for key in keys:
                        val = str(blob.get(key) or "").strip()
                        if len(val) >= 3:
                            found.append(val)
    except Exception:
        pass
    found.sort(key=len, reverse=True)
    _PII_CACHE = found
    return found


def redact_secrets(text: str | None) -> str:
    """Mask secrets/PII in an arbitrary string (safe for logs)."""
    if text is None:
        return ""
    out = str(text)
    if not out:
        return out
    for pattern, repl in _SECRET_RES:
        out = pattern.sub(repl, out)
    out = _AUTH_HEADER_RE.sub(
        lambda m: f"{m.group(1)}={REDACTED_SECRET}", out
    )
    for secret in _env_secret_values():
        if secret and secret in out:
            out = out.replace(secret, REDACTED_SECRET)
    for pii in _pii_substrings():
        if pii and pii in out:
            out = out.replace(pii, REDACTED_PII)
    return out


class RedactingFormatter(logging.Formatter):
    """Logging formatter that redacts secrets/PII before emit."""

    def format(self, record: logging.LogRecord) -> str:
        # Copy so we don't mutate the live record for other handlers forever
        try:
            original_msg = record.msg
            original_args = record.args
            msg = record.getMessage()
            record.msg = redact_secrets(msg)
            record.args = ()
            if record.exc_text:
                record.exc_text = redact_secrets(record.exc_text)
            rendered = super().format(record)
            record.msg = original_msg
            record.args = original_args
            return redact_secrets(rendered)
        except Exception:
            try:
                return redact_secrets(super().format(record))
            except Exception:
                return REDACTED_SECRET


def install_redacting_logging(
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s %(message)s",
    datefmt: str = "%H:%M:%S",
    force: bool = True,
) -> None:
    """Idempotent: attach RedactingFormatter to root handlers."""
    global _LOGGING_INSTALLED
    formatter = RedactingFormatter(fmt=fmt, datefmt=datefmt)
    root = logging.getLogger()
    if force or not root.handlers:
        # Rebuild root handlers with redaction
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(level)
    else:
        for h in root.handlers:
            h.setFormatter(formatter)
    for name in ("ace_agent", "ace_worker", "ace_cli"):
        logging.getLogger(name).setLevel(level)
    _LOGGING_INSTALLED = True


# ── Phase 2: untrusted input firewall ───────────────────────────────

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff\u2060]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def sanitize_untrusted_input(text: str | None, *, max_chars: int = 12_000) -> str:
    """Neutralize control chars / delimiter breakout before prompt injection."""
    raw = "" if text is None else str(text)
    raw = _CONTROL_RE.sub("", raw)
    raw = _ZERO_WIDTH_RE.sub("", raw)
    raw = _HTML_COMMENT_RE.sub("", raw)
    # Prevent closing our XML wrappers from inside the payload
    raw = raw.replace("</untrusted_", "‹/untrusted_")
    raw = raw.replace("<untrusted_", "‹untrusted_")
    # Escape remaining angle brackets so models see data, not markup
    raw = raw.replace("<", "‹").replace(">", "›")
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = raw.strip()
    if len(raw) > max_chars:
        raw = raw[:max_chars].rstrip() + "\n…[TRUNCATED_UNTRUSTED]"
    return raw


def wrap_untrusted(kind: str, body: str | None, *, max_chars: int = 12_000) -> str:
    """Wrap sanitized untrusted text in XML delimiters."""
    tag_kind = re.sub(r"[^a-z0-9_]", "_", (kind or "payload").strip().lower()) or "payload"
    tag = f"untrusted_{tag_kind}"
    safe = sanitize_untrusted_input(body, max_chars=max_chars)
    return f"<{tag}>\n{safe}\n</{tag}>"


# ── Phase 3: path jail ──────────────────────────────────────────────

def validate_safe_path(
    target_path: str | Path,
    *,
    must_exist: bool = False,
    allow_config_yaml: bool = False,
) -> Path:
    """Resolve path and assert it stays under an allowlisted project subtree.

    Raises PermissionError("Sandbox Violation: ...") on escape attempts.
    """
    if target_path is None or not str(target_path).strip():
        raise PermissionError("Sandbox Violation: empty path")
    if "\x00" in str(target_path):
        raise PermissionError("Sandbox Violation: NUL in path")

    root = PROJECT_ROOT.resolve()
    raw = Path(target_path).expanduser()
    try:
        resolved = raw.resolve(strict=must_exist)
    except FileNotFoundError:
        parent = raw.parent if raw.parent != raw else Path(".")
        try:
            resolved = parent.resolve() / raw.name
        except Exception as exc:
            raise PermissionError(
                f"Sandbox Violation: cannot resolve {target_path!r}"
            ) from exc

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Sandbox Violation: {resolved} escapes project root {root}"
        ) from exc

    if allow_config_yaml:
        cfg = (PROJECT_ROOT / "config.yaml").resolve()
        if resolved == cfg:
            return resolved

    for allowed in (DATA_DIR, OUTPUT_DIR, PROJECT_ROOT / "src", PROJECT_ROOT / "resumes", THEMES_DIR):
        allowed_r = allowed.resolve()
        try:
            resolved.relative_to(allowed_r)
            return resolved
        except ValueError:
            continue

    raise PermissionError(
        f"Sandbox Violation: {resolved} is outside allowlisted roots"
    )
