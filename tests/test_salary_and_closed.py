"""Salary annualization + closed-job detector unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_default_expected_ctc_band():
    from src.core.salary import default_expected_ctc, resolve_expected_ctc

    assert "6,00,000" in default_expected_ctc()
    assert "7,00,000" in default_expected_ctc()
    assert "per annum" in default_expected_ctc()
    assert resolve_expected_ctc(answers={}) == default_expected_ctc()


def test_lpa_from_jd_prefers_overlap_with_default_band():
    from src.core.salary import annual_from_text

    text = annual_from_text("Compensation: 5-8 LPA")
    assert text is not None
    assert "6,00,000" in text
    assert "7,00,000" in text
    assert "per annum" in text


def test_monthly_inr_converted_to_annual():
    from src.core.salary import annual_from_text

    text = annual_from_text("Stipend Rs 50,000 per month")
    assert text is not None
    assert "per annum" in text
    # 50k * 12 = 6,00,000
    assert "6,00,000" in text


def test_inject_salary_overrides_soft_norms():
    from src.core.salary import inject_salary_answers

    out = inject_salary_answers(
        {
            "expected_ctc": "As per company norms",
            "salary_expectation": "negotiable",
        }
    )
    # Scenario A: no company range → SalaryCalculator median 6.5 LPA
    assert "6,50,000" in out["expected_ctc"]
    assert out["salary_expectation"] == out["expected_ctc"]


def test_page_shows_job_closed_does_not_exist():
    from src.core.job_closed import page_shows_job_closed

    class _Node:
        def inner_text(self, timeout=0):
            return "This job does not exist. It may have been removed."

    class _Loc:
        @property
        def first(self):
            return _Node()

    class _Page:
        def title(self):
            return "Job not found"

        def locator(self, sel):
            return _Loc()

        def content(self):
            return ""

    assert page_shows_job_closed(_Page()) is True


def test_page_shows_job_closed_ignores_normal_jd():
    from src.core.job_closed import page_shows_job_closed

    class _Node:
        def inner_text(self, timeout=0):
            return (
                "We are hiring a Software Engineer. Responsibilities include building APIs. "
                * 40
            )

    class _Loc:
        @property
        def first(self):
            return _Node()

    class _Page:
        def title(self):
            return "Software Engineer - Acme"

        def locator(self, sel):
            return _Loc()

        def content(self):
            return ""

    assert page_shows_job_closed(_Page()) is False


def test_job_closed_status_valid(tmp_path):
    from src.core.memory import VALID_STATUSES, init_db, log_application

    assert "JOB_CLOSED" in VALID_STATUSES
    db = tmp_path / "m.db"
    init_db(db)
    log_application(
        "GoneCo",
        "http://127.0.0.1/missing",
        "swe",
        "JOB_CLOSED",
        error_log="does not exist",
        db_path=db,
    )


def test_cohere_in_triage_chain_not_draft(monkeypatch):
    from src.core import llm_router as lr

    keys = {
        "GROQ_API_KEY": "g",
        "NVIDIA_API_KEY": "n",
        "COHERE_API_KEY": "c",
        "OPENAI_API_KEY": "o",
    }

    def fake_env(*names: str) -> str:
        for name in names:
            if keys.get(name):
                return keys[name]
        return ""

    monkeypatch.setattr(lr, "_env", fake_env)
    monkeypatch.setattr(lr, "_ollama_reachable", lambda: False)
    monkeypatch.setattr(lr, "_allow_gemini_text", lambda: False)
    monkeypatch.setattr(lr, "_load_dotenv", lambda: None)
    monkeypatch.setattr(lr, "_yaml_llm_config", lambda: {})
    lr.text_provider_name.cache_clear()

    triage = lr.providers_for_role("triage")
    draft = lr.providers_for_role("draft")
    assert triage[0] == "cohere"
    assert "cohere" not in draft
    assert draft[0] == "openai"
