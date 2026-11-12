"""FormFillEngine — general kinds resolve from profile, not pasted employer QAs."""

from __future__ import annotations


def test_classify_common_labels():
    from src.core.form_rules import FieldKind, classify_label

    assert classify_label("Email address*") == FieldKind.IDENTITY_EMAIL
    assert classify_label("Please enter your expected ctc in INR") == FieldKind.COMP_EXPECTED_CTC
    assert classify_label("Please enter your notice period in days") == FieldKind.AVAIL_NOTICE_DAYS
    assert (
        classify_label("Which AI/LLM technologies or tools have you worked with?")
        == FieldKind.SKILLS_TECH
    )
    assert (
        classify_label("Are you available to commit to a full-time internship for 2 months?")
        == FieldKind.YES_NO_ELIGIBILITY
    )


def test_skills_tech_from_profile_intersect_jd():
    from src.core.form_engine import resolve_field, resolve_skills_tech

    profile = {
        "skills_list": ["Python", "TensorFlow", "PyTorch", "Excel", "LangChain"],
    }
    jd = "We need Python and LangChain for LLM prototypes"
    text = resolve_skills_tech(profile, jd)
    assert "Python" in text
    assert "LangChain" in text
    # Excel should drop when JD does not mention it (intersection preferred).
    assert "Excel" not in text or "Python" in text

    result = resolve_field(
        "Which AI/LLM technologies or tools have you worked with?",
        profile,
        jd_text=jd,
    )
    assert result.value
    assert result.needs_human is False
    assert "Python" in (result.value or "")


def test_yes_no_from_eligibility_prefs():
    from src.core.form_engine import resolve_field

    profile = {
        "open_to_internship_months": "6",
        "can_commit_hours_per_day": "8",
        "skills_list": ["Python"],
    }
    r1 = resolve_field(
        "Are you available to commit to a full-time internship for 2 months?",
        profile,
    )
    assert r1.value == "Yes"
    r2 = resolve_field(
        "Are you available to work 8 hours a day, 5 days a week during the internship?",
        profile,
    )
    assert r2.value == "Yes"


def test_identity_from_profile_keys():
    from src.core.form_engine import resolve_field

    profile = {
        "email": "you@example.com",
        "phone": "9999999999",
        "city": "Example City",
        "expected_ctc_inr": "650000",
        "notice_period_days": "7",
        "portfolio_url": "",
    }
    assert resolve_field("Email address*", profile).value == "you@example.com"
    assert resolve_field("Mobile phone number*", profile).value == "9999999999"
    assert resolve_field("Enter city or location", profile).value == "Example City"
    assert resolve_field("Please enter your expected ctc in INR", profile).value == "650000"
    assert resolve_field("Please enter your notice period in days", profile).value == "7"
    # Empty portfolio is valid — not human.
    port = resolve_field("Portfolio URL", profile)
    assert port.value == ""
    assert port.needs_human is False


def test_unknown_needs_human():
    from src.core.form_engine import resolve_field

    r = resolve_field("What is your favourite colour of stapler?", {})
    assert r.needs_human is True
    assert r.value is None


def test_linkedin_playbook_has_no_personal_tech_paste():
    from pathlib import Path

    text = Path("src/playbooks/linkedin.py").read_text(encoding="utf-8")
    assert "TensorFlow" not in text
    assert "LangChain" not in text
    assert "lumenci" not in text.lower()
    # Engine is the fill path.
    assert "FormFillEngine" in text or "_fill_via_engine" in text


def test_token_limit_flag_and_message():
    from src.core.llm_router import (
        clear_token_limit_reached,
        mark_token_limit_reached,
        token_limit_detail,
        token_limit_reached,
    )

    clear_token_limit_reached()
    assert token_limit_reached() is False
    mark_token_limit_reached("no credits remaining")
    assert token_limit_reached() is True
    assert "TOKEN LIMIT" in token_limit_detail().upper() or "credit" in token_limit_detail().lower()
    clear_token_limit_reached()
    assert token_limit_reached() is False


def test_yes_no_visa_resolves_no_not_default_yes():
    from src.core.form_engine import resolve_field, resolve_yes_no

    profile = {"require_visa_sponsorship_india": "No"}
    assert resolve_yes_no("Will you require visa sponsorship?", profile) == "No"
    r = resolve_field("Do you require visa sponsorship to work in India?", profile)
    assert r.value == "No"
    assert r.confidence >= 0.7


def test_yes_no_taught_override():
    from src.core.form_engine import resolve_yes_no

    assert resolve_yes_no("Random eligibility?", {}, taught="No") == "No"
    assert resolve_yes_no("Random eligibility?", {}, taught="yes") == "Yes"


def test_fill_yes_no_skips_unlabeled(monkeypatch):
    """Unlabeled Yes controls must never be auto-clicked."""
    from src.core.form_engine import FormFillEngine

    clicks: list[str] = []

    class _Loc:
        def is_visible(self, timeout=0):
            return True

        def evaluate(self, *_a, **_k):
            return ""  # no surrounding question label

        def click(self, **_k):
            clicks.append("Yes")

    class _EmptyModal:
        def count(self):
            return 0

        @property
        def first(self):
            return self

    class _Page:
        def locator(self, *_a, **_k):
            return _EmptyModal()

        def get_by_text(self, *_a, **_k):
            class _All:
                def all(self):
                    return [_Loc()]

            return _All()

        def evaluate(self, *_a, **_k):
            return []

    engine = FormFillEngine({"open_to_internship_months": "6"})
    stats = engine.fill_visible(_Page())
    assert clicks == []
    assert stats["filled"] == 0


def test_fill_yes_no_clicks_matching_value():
    from src.core.form_engine import FormFillEngine

    clicks: list[str] = []

    class _Loc:
        def __init__(self, word: str):
            self.word = word

        def is_visible(self, timeout=0):
            return True

        def evaluate(self, *_a, **_k):
            return "Will you require visa sponsorship?"

        def click(self, **_k):
            clicks.append(self.word)

    class _Text:
        def __init__(self, word: str):
            self.word = word

        def all(self):
            return [_Loc(self.word)]

    class _EmptyModal:
        def count(self):
            return 0

        @property
        def first(self):
            return self

    class _Page:
        def locator(self, *_a, **_k):
            return _EmptyModal()

        def get_by_text(self, pattern, *_a, **_k):
            src = getattr(pattern, "pattern", str(pattern))
            if "Yes" in src:
                return _Text("Yes")
            return _Text("No")

        def evaluate(self, *_a, **_k):
            return []

    engine = FormFillEngine({"require_visa_sponsorship_india": "No"})
    stats = engine.fill_visible(_Page())
    assert "No" in clicks
    assert "Yes" not in clicks
    assert stats["filled"] >= 1
