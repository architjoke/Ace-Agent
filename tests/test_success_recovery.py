"""Auto-apply SUCCESS recovery: force upload, Indeed classify, stuck/captcha, vision cap."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_classify_indeed():
    from src.playbooks import classify_ats

    assert classify_ats("https://in.indeed.com/viewjob?jk=abc") == "indeed"
    assert classify_ats("https://smartapply.indeed.com/beta/indeedapply/form") == "indeed"
    assert classify_ats("https://www.linkedin.com/jobs/view/1") == "linkedin"


def test_block_internshala_and_deprioritize_indeed(monkeypatch):
    from src.core import discovery as d

    assert d.is_blocked_board("https://internshala.com/internship/detail/foo")
    assert not d.is_applyable_url("https://internshala.com/internship/detail/foo")
    monkeypatch.delenv("ACE_ALLOW_INDEED", raising=False)
    assert d.is_indeed_soft_url("https://in.indeed.com/viewjob?jk=x")
    assert not d.is_applyable_url("https://in.indeed.com/viewjob?jk=x")
    monkeypatch.setenv("ACE_ALLOW_INDEED", "1")
    assert d.is_applyable_url("https://in.indeed.com/viewjob?jk=x")
    # Prefer Greenhouse direct over Indeed listing
    url = d.pick_apply_url(
        {
            "job_url": "https://in.indeed.com/viewjob?jk=x",
            "job_url_direct": "https://boards.greenhouse.io/acme/jobs/1",
        }
    )
    assert "greenhouse" in url
    assert d.apply_url_priority("https://boards.greenhouse.io/x") < d.apply_url_priority(
        "https://www.linkedin.com/jobs/view/1"
    )
    # Hiring Cafe listing loses to company career / ATS direct
    hc = d.pick_apply_url(
        {
            "job_url": "https://hiring.cafe/job/acme-backend",
            "job_url_direct": "https://jobs.ashbyhq.com/acme/abcd",
        }
    )
    assert "ashbyhq" in hc
    assert d.apply_url_priority("https://jobs.lever.co/acme/1") < d.apply_url_priority(
        "https://hiring.cafe/job/x"
    )
    assert d.is_career_ats_url("https://paypal.eightfold.ai/careers/job/1")
    assert d.is_career_ats_url("https://selah.zohorecruit.in/jobs/Careers/1")
    assert d.is_career_ats_url("https://careers.quinite.co/jobs/x")
    assert d.apply_url_priority(
        "https://faveohelpdesk.zohorecruit.in/jobs/Careers/1"
    ) < d.apply_url_priority("https://www.linkedin.com/jobs/view/1")


def test_camoufox_cache_enables_hiring_cafe(monkeypatch):
    from src.core import discovery_extra as de

    monkeypatch.delenv("ACE_JOBDROP_SITES", raising=False)
    monkeypatch.delenv("ACE_JOBDROP_ALLOW_CAMOUFOX", raising=False)
    assert de._camoufox_cached() is True
    sites = de.jobdrop_sites_for_hunt()
    assert sites[0] == "hiring_cafe"
    assert "wellfound" in sites
    assert "weworkremotely" in sites


def test_already_applied_application_submitted_banner():
    from src.core.already_applied import page_shows_already_applied

    page = MagicMock()

    def _by_text(pat):
        text = pat.pattern if hasattr(pat, "pattern") else str(pat)
        hit = MagicMock()
        hit.first = hit
        # Only the application-submitted locator is visible
        visible = "application" in text.lower() and "submitted" in text.lower()
        hit.is_visible.return_value = visible
        hit.count.return_value = 1 if visible else 0
        return hit

    page.get_by_text.side_effect = _by_text
    page.get_by_role.return_value = MagicMock(
        first=MagicMock(is_visible=MagicMock(return_value=False), count=MagicMock(return_value=0))
    )
    page.locator.return_value = MagicMock(
        first=MagicMock(is_visible=MagicMock(return_value=False), count=MagicMock(return_value=0))
    )
    assert page_shows_already_applied(page) is True


def test_indeed_in_ats_kinds_and_heavy():
    from src.core import executor as ex

    assert "indeed" in ex._ATS_KINDS
    assert ex._is_heavy_ats("https://in.indeed.com/viewjob?jk=x")
    assert ex._playbook_runner("indeed") is not None


def test_force_upload_clicks_replace_then_sets_files(tmp_path):
    from src.playbooks._common import force_upload_resume

    pdf = tmp_path / "Archit_Gupta_Resume.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    file_inp = MagicMock()
    file_inp.set_input_files = MagicMock()

    inputs = MagicMock()
    inputs.count.return_value = 1
    inputs.nth.return_value = file_inp

    upload_btn = MagicMock()
    upload_btn.count.return_value = 1
    upload_btn.is_visible.return_value = True
    upload_btn.inner_text.return_value = "Upload resume"
    upload_btn.click = MagicMock()
    upload_btn.first = upload_btn
    upload_btn.nth.return_value = upload_btn

    def _locator(sel: str):
        if "file" in sel:
            return inputs
        # attached-resume / wait markers — invisible; stem + evaluate confirm
        miss = MagicMock()
        miss.count.return_value = 0
        miss.is_visible.return_value = False
        miss.first = miss
        miss.or_ = MagicMock(return_value=miss)
        miss.wait_for = MagicMock(side_effect=TimeoutError("not visible"))
        return miss

    page = MagicMock()
    page.locator.side_effect = _locator
    page.get_by_role.return_value = upload_btn
    page.get_by_text.return_value = MagicMock(count=lambda: 0, first=MagicMock(count=lambda: 0, is_visible=lambda **k: False))
    page.inner_text.return_value = "Archit_Gupta_Resume.pdf attached"
    page.wait_for_timeout = MagicMock()
    page.evaluate = MagicMock(return_value=True)

    assert force_upload_resume(page, pdf) is True
    file_inp.set_input_files.assert_called()
    assert page.evaluate.called  # React input/change dispatch +/or files.length


def test_force_upload_falls_back_to_files_length(tmp_path):
    from src.playbooks._common import force_upload_resume

    pdf = tmp_path / "Archit_Gupta_Resume.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    file_inp = MagicMock()
    file_inp.set_input_files = MagicMock()
    inputs = MagicMock()
    inputs.count.return_value = 1
    inputs.nth.return_value = file_inp

    def _locator(sel: str):
        if "file" in sel:
            return inputs
        miss = MagicMock()
        miss.count.return_value = 0
        miss.is_visible.return_value = False
        miss.first = miss
        miss.or_ = MagicMock(return_value=miss)
        miss.wait_for = MagicMock(side_effect=TimeoutError("x"))
        return miss

    page = MagicMock()
    page.locator.side_effect = _locator
    page.get_by_role.return_value = MagicMock(
        count=lambda: 0, first=MagicMock(count=lambda: 0, is_visible=lambda **k: False)
    )
    page.get_by_text.return_value = MagicMock(
        count=lambda: 0, first=MagicMock(count=lambda: 0, is_visible=lambda **k: False)
    )
    page.inner_text.return_value = "Apply to Company"  # no stem
    page.wait_for_timeout = MagicMock()
    # dispatch returns int; files.length probe returns True
    page.evaluate = MagicMock(side_effect=[1, True, 1, True])

    assert force_upload_resume(page, pdf) is True


def test_linkedin_budget_and_advance_order():
    from src.playbooks import linkedin as li

    assert li._MAX_STEPS == 15
    assert li._OVERALL_BUDGET_S >= 90

    clicks: list[str] = []

    def _btn(name):
        pat = name.pattern if hasattr(name, "pattern") else str(name)
        kind = "submit" if "Submit" in pat else "review" if "Review" in pat else "next"
        btn = MagicMock()
        btn.count.return_value = 1
        btn.is_visible.return_value = True

        def _click(**k):
            clicks.append(kind)

        btn.click.side_effect = _click
        btn.first = btn
        return btn

    page = MagicMock()
    page.get_by_role.side_effect = lambda role, name: _btn(name)
    page.wait_for_timeout = MagicMock()
    assert li._advance(page) == "submit"
    assert clicks == ["submit"]


def test_linkedin_error_banner_visible():
    from src.playbooks import linkedin as li

    err = MagicMock()
    err.count.return_value = 1
    err.is_visible.return_value = True
    err.inner_text.return_value = "Please fix the following errors"
    err.first = err

    page = MagicMock()
    page.locator.return_value = err
    page.get_by_text.return_value = MagicMock(
        first=MagicMock(count=lambda: 0, is_visible=lambda **k: False)
    )
    assert li._form_error_banner(page) is True


def test_parse_rendercv_pdf_ignores_version_banner(tmp_path):
    from src.core import tailor as t

    pdf = tmp_path / "Archit_Gupta_CV.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkey_stdout = (
        "A new version of RenderCV is available. Run: pip install --upgrade rendercv\n"
        f"✔ PDF generated: {pdf}\n"
    )
    paths = t._parse_rendercv_pdf_paths(monkey_stdout, "")
    assert paths == [pdf.resolve()]
    scrubbed = t._scrub_rendercv_output(monkey_stdout)
    assert "new version" not in scrubbed.lower()
    assert str(pdf) in scrubbed or pdf.name in scrubbed


def test_gemini_capacity_error_detection():
    from src.core.executor import _is_gemini_capacity_error

    class ServiceUnavailable(Exception):
        pass

    class TooManyRequests(Exception):
        pass

    assert _is_gemini_capacity_error(ServiceUnavailable("503 unavailable"))
    assert _is_gemini_capacity_error(TooManyRequests("429 rate limit"))
    assert not _is_gemini_capacity_error(ValueError("bad selector"))


def test_run_agent_retries_then_yields_api_error(monkeypatch):
    from src.core import executor as ex

    class ServiceUnavailable(Exception):
        pass

    calls = {"n": 0}

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        async def run(self, max_steps=0):
            calls["n"] += 1
            raise ServiceUnavailable("503 unavailable high demand")

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(ex, "Agent", FakeAgent)
    monkeypatch.setattr(ex, "_build_llm", lambda *a, **k: object())
    monkeypatch.setattr(ex, "_api_key", lambda: "x")
    monkeypatch.setattr(ex, "_browser_profile", lambda *a, **k: object())
    monkeypatch.setattr(ex, "_detach_agent", _noop)
    monkeypatch.setattr(ex, "_force_close_agent", _noop)
    monkeypatch.setattr(ex, "_vision_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(ex.asyncio, "sleep", _noop)

    async def _run():
        with pytest.raises(ex.ApplicationTimeoutError, match="requires_human_api_error"):
            await ex._run_agent("task", [], max_steps=1, timeout_seconds=5, teardown=False)
        assert calls["n"] == 3

    asyncio.run(_run())


def test_raise_if_blocked_ignores_lesson_echo():
    from src.core.executor import ApplicationBlockedError, _raise_if_blocked

    class _H:
        def final_result(self):
            return (
                "On CAPTCHA / Cloudflare / bot walls, abort immediately and leave MANUAL_ACTION. "
                "I filled the form and will continue."
            )

        def errors(self):
            return []

        def judgement(self):
            return None

    # Lesson echo alone must not block
    _raise_if_blocked(_H())


def test_raise_if_blocked_quota_is_timeout():
    from src.core.executor import ApplicationTimeoutError, _raise_if_blocked

    class _H:
        def final_result(self):
            return "503 unavailable. this model is currently experiencing high demand"

        def errors(self):
            return []

        def judgement(self):
            return None

    with pytest.raises(ApplicationTimeoutError, match="quota|capacity"):
        _raise_if_blocked(_H())


def test_raise_if_blocked_403_is_blocked_not_quota():
    from src.core.executor import ApplicationBlockedError, _raise_if_blocked

    class _H:
        def final_result(self):
            return (
                "Failed to access the Fusionpact Keka careers page due to a "
                "403 Forbidden Access error. Also saw 503 unavailable noise."
            )

        def errors(self):
            return []

        def judgement(self):
            return None

    with pytest.raises(ApplicationBlockedError, match="403"):
        _raise_if_blocked(_H())


def test_linkedin_easy_apply_miss_with_submitted_raises_already():
    from src.core.executor import AlreadyAppliedError
    from src.playbooks import linkedin as li

    page = MagicMock()
    # No Easy Apply clickable
    miss = MagicMock()
    miss.count.return_value = 0
    miss.is_visible.return_value = False
    miss.first = miss
    page.get_by_role.return_value = miss
    page.locator.return_value = miss
    page.frames = []
    page.main_frame = page

    # Application submitted visible
    def _by_text(pat):
        text = pat.pattern if hasattr(pat, "pattern") else str(pat)
        hit = MagicMock()
        hit.first = hit
        hit.count.return_value = 1
        hit.is_visible.return_value = "application" in text.lower() and "submitted" in text.lower()
        return hit

    page.get_by_text.side_effect = _by_text
    page.inner_text.return_value = "Application submitted"

    with pytest.raises(AlreadyAppliedError, match="Application submitted|Applied"):
        li._click_easy_apply(page)


def test_raise_if_blocked_real_amazon_captcha():
    from src.core.executor import ApplicationBlockedError, _raise_if_blocked

    class _H:
        def final_result(self):
            return "Encountered Amazon captcha wall requiring image selection"

        def errors(self):
            return []

        def judgement(self):
            return None

    with pytest.raises(ApplicationBlockedError):
        _raise_if_blocked(_H())


def test_stuck_exempts_preparing_review():
    from src.core.executor import ApplicationTimeoutError, _StuckStateTracker

    tracker = _StuckStateTracker({"agent": None})
    state = SimpleNamespace(
        url="https://smartapply.indeed.com/beta/indeedapply/form/review-module",
        page_content="Preparing review 100%",
        selector_map={1: object(), 2: object()},
    )
    # Same FP repeatedly while preparing — must NOT raise
    for i in range(6):
        tracker(state, None, i)
    assert tracker._stuck_events == 0


def test_goto_dom_skips_thread_error_reload(monkeypatch):
    from src.playbooks import _common as common

    page = MagicMock()
    page.url = "https://example.com/other"

    def boom_goto(*a, **k):
        raise RuntimeError("Cannot switch to a different thread")

    page.goto.side_effect = boom_goto
    page.reload = MagicMock(side_effect=AssertionError("must not reload"))
    with pytest.raises(TimeoutError, match="different thread"):
        common.goto_dom(page, "https://example.com/job/1")
    page.reload.assert_not_called()


def test_vision_semaphore_serializes():
    from src.core import executor as ex

    async def _run():
        s1 = ex._vision_semaphore()
        s2 = ex._vision_semaphore()
        assert s1 is s2
        assert s1._value >= 0

    asyncio.run(_run())
