"""Salary calculator + annualization + job-closed detector."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_scenario_a_no_company_range_uses_median():
    from src.core.salary_engine import SalaryCalculator

    d = SalaryCalculator().compute(None, field_format="ANNUAL")
    assert d.annual_inr == 650_000
    assert d.display == "6,50,000"
    assert "default_median" in d.rationale


def test_scenario_b_p60_inside_band():
    from src.core.salary_engine import SalaryCalculator

    # 600k–800k → p60 = 600k + 0.6*200k = 720k
    d = SalaryCalculator().compute(
        {
            "company_provided_salary": True,
            "salary_min": 600_000,
            "salary_max": 800_000,
            "salary_format": "ANNUAL",
            "includes_equity_or_bonus": False,
        },
        field_format="ANNUAL",
    )
    assert d.annual_inr == 720_000
    assert "p60" in d.rationale


def test_scenario_c_base_with_equity_applies_share():
    from src.core.salary_engine import SalaryCalculator

    d = SalaryCalculator().compute(
        {
            "company_provided_salary": True,
            "salary_min": 600_000,
            "salary_max": 800_000,
            "salary_format": "ANNUAL",
            "includes_equity_or_bonus": True,
        },
        field_format="ANNUAL",
        field_kind="BASE",
    )
    # 720k * 0.75 = 540k
    assert d.annual_inr == 540_000
    assert "base_x" in d.rationale


def test_company_max_below_floor_takes_company_max():
    from src.core.salary_engine import SalaryCalculator

    d = SalaryCalculator().compute(
        {
            "company_provided_salary": True,
            "salary_min": 200_000,
            "salary_max": 250_000,
            "salary_format": "ANNUAL",
        }
    )
    assert d.annual_inr == 250_000
    assert "below_floor" in d.rationale


def test_p60_bumped_to_floor_when_inside_band():
    from src.core.salary_engine import SalaryCalculator

    # lo=100k hi=400k → p60=280k < floor 300k ≤ hi → bump to 300k
    d = SalaryCalculator().compute(
        {
            "company_provided_salary": True,
            "salary_min": 100_000,
            "salary_max": 400_000,
            "salary_format": "ANNUAL",
        }
    )
    assert d.annual_inr == 300_000
    assert "bump_to_floor" in d.rationale


def test_lpa_heuristic_and_monthly_annualize():
    from src.core.salary_engine import normalize_compensation

    lakhs = normalize_compensation(
        {
            "company_provided_salary": True,
            "salary_min": 6,
            "salary_max": 8,
            "salary_format": "ANNUAL",
        }
    )
    assert lakhs["salary_min"] == 600_000
    assert lakhs["salary_max"] == 800_000

    monthly = normalize_compensation(
        {
            "company_provided_salary": True,
            "salary_min": 50_000,
            "salary_max": 50_000,
            "salary_format": "MONTHLY",
        }
    )
    assert monthly["salary_min"] == 600_000
    assert monthly["salary_format"] == "ANNUAL"
    assert monthly["jd_salary_format"] == "MONTHLY"


def test_normalize_idempotent_no_double_annualize():
    from src.core.salary_engine import normalize_compensation

    once = normalize_compensation(
        {
            "company_provided_salary": True,
            "salary_min": 25_000,
            "salary_max": 25_000,
            "salary_format": "MONTHLY",
        }
    )
    twice = normalize_compensation(once)
    assert once["salary_min"] == twice["salary_min"] == 300_000
    assert twice["jd_salary_format"] == "MONTHLY"


@pytest.mark.parametrize(
    "junk",
    [
        None,
        {},
        {"company_provided_salary": "yes", "salary_min": "abc", "salary_max": None},
        {"company_provided_salary": True, "salary_min": float("nan"), "salary_max": float("inf")},
        {"company_provided_salary": True, "salary_min": "₹6,00,000", "salary_max": "8 LPA"},
        {"company_provided_salary": 1, "salary_min": True, "salary_max": False},
        {"company_provided_salary": True, "salary_min": "null", "salary_max": ""},
    ],
)
def test_calculator_never_raises_on_junk(junk):
    from src.core.salary_engine import SalaryCalculator

    d = SalaryCalculator().compute(junk)  # type: ignore[arg-type]
    assert isinstance(d.annual_inr, int)
    assert d.annual_inr >= 0
    assert isinstance(d.display, str)
    assert d.amount >= 0
    assert not math.isnan(d.amount)


def test_format_monthly_and_lakhs():
    from src.core.salary_engine import SalaryCalculator

    calc = SalaryCalculator()
    monthly = calc.compute(
        {"company_provided_salary": False},
        field_format="MONTHLY",
    )
    assert monthly.amount == round(650_000 / 12)
    assert monthly.display == str(round(650_000 / 12))

    lakhs = calc.format_amount(650_000, "LAKHS")
    assert lakhs[0] == 6.5
    assert lakhs[1] == "6.5"


def test_classify_salary_field():
    from src.core.salary_engine import classify_salary_field

    assert classify_salary_field("Expected CTC (LPA)")[0] == "LAKHS"
    assert classify_salary_field("Monthly stipend")[0] == "MONTHLY"
    assert classify_salary_field("Base salary")[1] == "BASE"
    assert classify_salary_field("Total CTC package")[1] == "CTC"


def test_embed_extract_roundtrip():
    from src.core.salary_engine import (
        COMP_MARKER,
        embed_compensation,
        extract_compensation,
        strip_compensation_marker,
    )

    body = "cv:\n  name: Test\n"
    snap = embed_compensation(
        body,
        {
            "company_provided_salary": True,
            "salary_min": 6,
            "salary_max": 7,
            "salary_format": "ANNUAL",
            "includes_equity_or_bonus": False,
        },
    )
    assert snap.startswith(COMP_MARKER)
    comp = extract_compensation(snap)
    assert comp["company_provided_salary"] is True
    assert comp["salary_min"] == 600_000
    assert comp["salary_max"] == 700_000
    assert strip_compensation_marker(snap) == body


def test_inject_uses_calculator_median():
    from src.core.salary import inject_salary_answers

    out = inject_salary_answers(
        {
            "expected_ctc": "As per company norms",
            "salary_expectation": "negotiable",
        }
    )
    assert out["expected_ctc"] == "650000"
    assert out["expected_ctc_inr"] == "650000"
    assert "6,50,000" in out["expected_ctc_display"]
    assert out["salary_annual_inr"] == 650_000
    assert out["salary_expectation"] == out["expected_ctc"]


def test_inject_from_yaml_snapshot_company_range():
    from src.core.salary import inject_salary_answers
    from src.core.salary_engine import embed_compensation

    snap = embed_compensation(
        "x: 1\n",
        {
            "company_provided_salary": True,
            "salary_min": 600_000,
            "salary_max": 800_000,
            "salary_format": "ANNUAL",
        },
    )
    out = inject_salary_answers({}, yaml_snapshot=snap)
    assert out["salary_annual_inr"] == 720_000
    assert out["expected_ctc"] == "720000"
    assert "7,20,000" in out["expected_ctc_display"]

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
