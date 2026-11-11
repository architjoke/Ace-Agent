"""Playbook helpers: boards, pitch, mailto, captcha screenshots."""

from __future__ import annotations

from src.core.discovery import is_redirect_board
from src.core.mailto_apply import build_mailto_message, parse_mailto
from src.core.pitch import embed_pitch, extract_pitch, generate_pitch
from src.playbooks import classify_ats
from src.playbooks._common import (
    cloudflare_turnstile_present,
    resolve_pitch_short,
)
from src.playbooks.naukri import preflight_dismiss_naukri


def test_classify_seven_boards_and_ats():
    assert classify_ats("https://weworkremotely.com/remote-jobs/foo") == "aggregator"
    assert classify_ats("https://himalayas.app/companies/acme/jobs/eng") == "aggregator"
    assert classify_ats("https://hiring.cafe/jobs/1") == "aggregator"
    assert classify_ats("https://www.naukri.com/job-listings-python-intern") == "naukri"
    assert classify_ats("https://wellfound.com/jobs/123") == "wellfound"
    assert classify_ats("https://www.workatastartup.com/jobs/12345") == "yc_waas"
    assert classify_ats("https://www.glassdoor.co.in/job-listing/js.htm") == "glassdoor"
    assert classify_ats("mailto:jobs@acme.com?subject=Intern") == "mailto"
    assert classify_ats("https://boards.greenhouse.io/acme/jobs/1") == "greenhouse"


def test_redirect_board_hosts():
    assert is_redirect_board("https://weworkremotely.com/x")
    assert is_redirect_board("https://himalayas.app/companies/a/jobs/b")
    assert not is_redirect_board("https://www.linkedin.com/jobs/view/1")


def test_pitch_embed_stacks_under_compensation():
    snap = "ACE_COMPENSATION_JSON:{\"annual_ctc_inr\":600000}\ncv:\n  name: Archit\n"
    out = embed_pitch(snap, "Short human pitch about Python and AI.")
    assert out.startswith("ACE_COMPENSATION_JSON:")
    assert "# ace_pitch:" in out
    assert "Short human pitch" in extract_pitch(out)


def test_pitch_fallback_without_llm(monkeypatch):
    import src.core.pitch as pitch_mod

    def boom(*_a, **_k):
        raise RuntimeError("no llm")

    monkeypatch.setattr(
        "src.core.llm_router.invoke_with_failover", boom, raising=False
    )
    # Force import path used inside generate_pitch
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.core.llm_router",
        type("M", (), {"invoke_with_failover": staticmethod(boom)})(),
    )
    text = generate_pitch(company="Acme", job_title="AI Intern", jd_text="Python")
    assert "Archit" in text or "Acme" in text
    assert len(text) > 40


def test_mailto_parse_and_build():
    to, subject, body_hint = parse_mailto(
        "mailto:careers@startup.io?subject=SDE%20Intern&body=Hi"
    )
    assert to == "careers@startup.io"
    assert "SDE" in subject
    to2, subject2, body = build_mailto_message(
        company="Startup",
        job_title="SDE Intern",
        jd_text="Python Node",
        yaml_snapshot=embed_pitch("", "I build Python backends and AI features."),
        to_hint=to,
        subject_hint=subject,
    )
    assert to2 == "careers@startup.io"
    assert "Startup" in subject2 or "SDE" in subject2
    assert "Archit" in body
    assert "resume" in body.lower()


def test_resolve_pitch_prefers_embedded_marker():
    snap = embed_pitch("", "Embedded pitch about Node and AI for Acme Labs.")
    text = resolve_pitch_short(
        yaml_snapshot=snap, company="Acme", job_title="Intern", jd_text=""
    )
    assert "Embedded pitch" in text


def test_cloudflare_helper_false_on_clean_stub():
    class _FakeLoc:
        def count(self):
            return 0

        def is_visible(self, timeout=0):
            return False

        def first(self):
            return self

    class _FakePage:
        url = "https://wellfound.com/jobs/1"

        def locator(self, *_a, **_k):
            return _FakeLoc()

        def inner_text(self, *_a, **_k):
            return "Apply to this role at a startup"

    assert cloudflare_turnstile_present(_FakePage()) is False


def test_cloudflare_helper_true_on_turnstile_copy():
    class _FakeLoc:
        def count(self):
            return 0

        def is_visible(self, timeout=0):
            return False

        def first(self):
            return self

    class _FakePage:
        url = "https://wellfound.com/jobs/1"

        def locator(self, *_a, **_k):
            return _FakeLoc()

        def inner_text(self, *_a, **_k):
            return "One more step before you proceed… cf-turnstile"

    assert cloudflare_turnstile_present(_FakePage()) is True


def test_naukri_preflight_helper_exported():
    assert callable(preflight_dismiss_naukri)

import sys
from pathlib import Path
from unittest.mock import MagicMock


def test_page_shows_captcha_from_body():
    from src.core import human_assist as ha

    page = MagicMock()
    page.inner_text.return_value = "Please verify you are human to continue"
    page.title.return_value = "Apply"
    page.locator.return_value.first.count.return_value = 0
    assert ha.page_shows_captcha(page) is True


def test_page_shows_captcha_false_on_clean_form():
    from src.core import human_assist as ha

    page = MagicMock()
    page.inner_text.return_value = "Email First Name Submit Application"
    page.title.return_value = "Careers"
    page.locator.return_value.first.count.return_value = 0
    assert ha.page_shows_captcha(page) is False


def test_capture_failure_shot_writes_png(tmp_path, monkeypatch):
    from src.core import human_assist as ha

    monkeypatch.setattr(ha, "_output_root", lambda: tmp_path)
    page = MagicMock()
    page.is_closed.return_value = False
    page.url = "https://example.com/apply"
    page.screenshot = MagicMock()

    path = ha.capture_failure_shot(
        page, company="Acme Labs", reason="playbook_fallback", job_url="https://x"
    )
    assert path is not None
    assert path.suffix == ".png"
    assert path.parent.name == "Acme_Labs"
    assert path.with_suffix(".txt").is_file()
    page.screenshot.assert_called_once()


def test_resolve_captcha_disabled_returns_false(monkeypatch):
    from src.core import human_assist as ha

    monkeypatch.setattr(ha, "captcha_human_enabled", lambda: False)
    monkeypatch.setattr(ha, "capture_failure_shot", lambda *a, **k: None)
    page = MagicMock()
    assert ha.resolve_captcha_with_human(page, company="X") is False
