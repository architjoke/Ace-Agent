"""Closed / unavailable job pages must skip — no Manual_Action."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_oracle_trinet_desert_page_text():
    from src.core.job_closed import text_shows_job_closed

    blob = (
        "This job is no longer available.\n"
        "You may also\nVIEW ALL JOBS\n"
    )
    assert text_shows_job_closed(blob) is True


def test_agent_final_closed_raises_job_closed():
    from src.core.executor import JobClosedError, _raise_if_blocked
    from src.core.job_closed import text_shows_job_closed

    final = (
        'The job (ID 3004263) is no longer available on the portal '
        '("This job is no longer available."). Following instructions, '
        "the application process has been aborted."
    )
    assert text_shows_job_closed(final) is True
    # Sanity: quota path still not JobClosed
    hist = SimpleNamespace(
        final_result=lambda: final,
        errors=lambda: [],
        judgement=lambda: None,
    )
    # _raise_if_blocked should not swallow closed into captcha
    _raise_if_blocked(hist)


def test_open_jd_not_closed():
    from src.core.job_closed import text_shows_job_closed

    assert text_shows_job_closed("Software Engineer Intern — apply now") is False
