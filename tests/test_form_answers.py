"""Constrained form fields must not receive narrative ranges."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_notice_range_becomes_single_day():
    from src.core.form_answers import sanitize_application_answers

    out = sanitize_application_answers(
        {
            "notice_period": "7 to 14 days",
            "notice_period_days": "7-14",
            "years_of_experience": "0-1 (internships)",
            "expected_ctc": "₹6,50,000 per annum",
            "current_ctc": "Not applicable",
            "portfolio_url": "https://github.com/architjoke",
        }
    )
    assert out["notice_period_days"] == "7"
    assert out["notice_period_days_numeric"] == "7"
    assert out["years_of_experience"] == "0"
    assert out["expected_ctc"] == "650000"
    assert out["expected_ctc_inr"] == "650000"
    assert out["current_ctc"] == "0"
    assert out["portfolio_url"] == ""


def test_answers_prompt_separates_narrative():
    from src.core.form_answers import answers_prompt_block

    block = answers_prompt_block(
        {
            "notice_period": "7 to 14 days",
            "notice_period_days": "7-14",
            "expected_ctc_display": "₹6,50,000 per annum",
            "expected_ctc_inr": "650000",
        }
    )
    assert "Constrained / numeric" in block
    assert "Narrative context ONLY" in block
    assert "notice_period_days: 7" in block
    assert "7-14" not in block.split("Narrative")[0]
