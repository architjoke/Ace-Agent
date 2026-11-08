"""Localhost mock ATS playbooks — no Groq / no live sites."""

from __future__ import annotations

import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MOCKS = Path(__file__).resolve().parent / "mocks"
MASTER = MOCKS / "master.yaml"
PDF = MOCKS / "_blank.pdf"


@pytest.fixture(scope="module")
def mock_server():
    handler = partial(SimpleHTTPRequestHandler, directory=str(MOCKS))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def profile_dir(tmp_path):
    d = tmp_path / "chromium_profile"
    d.mkdir()
    return d


def _stub_plan_semantic_actions(elements, *, answers, resume_path):
    """Deterministic filler — never calls Groq."""
    actions: list[dict] = []
    for e in elements:
        ace_id = e.get("id")
        if ace_id is None:
            continue
        label = str(
            e.get("label") or e.get("name") or e.get("placeholder") or e.get("text") or ""
        ).lower()
        typ = str(e.get("type") or "").lower()
        if typ == "file" or "resume" in label or "cv" in label:
            actions.append({"action": "upload", "id": int(ace_id)})
            continue
        text = ""
        if "phone" in label or "mobile" in label:
            text = str(answers.get("phone") or "")
        elif "email" in label:
            text = str(answers.get("email") or "")
        elif "first" in label:
            text = str(answers.get("first_name") or "")
        elif "last" in label:
            text = str(answers.get("last_name") or "")
        elif "city" in label or "location" in label:
            text = str(answers.get("location") or answers.get("city") or "")
        elif "name" in label:
            text = str(answers.get("full_name") or "")
        elif "linkedin" in label:
            text = str(answers.get("linkedin") or "")
        if text:
            actions.append({"action": "fill", "id": int(ace_id), "text": text})
    return actions


@pytest.fixture(autouse=True)
def stub_semantic_planner(monkeypatch):
    monkeypatch.setattr(
        "src.core.semantic_dom.plan_semantic_actions",
        _stub_plan_semantic_actions,
    )
    # Localhost mocks must not go through a sticky proxy.
    monkeypatch.setattr("src.core.network.playwright_proxy_dict", lambda: None)
    monkeypatch.setattr("src.core.network.browser_proxy_settings", lambda: None)


def _run_playbook(fn, *args, **kwargs):
    """Close CDP handoff sessions if a playbook raises PlaybookHandoff."""
    from src.playbooks.session import PlaybookHandoff

    try:
        return fn(*args, **kwargs)
    except PlaybookHandoff as handoff:
        try:
            handoff.session.close()
        except Exception:
            pass
        raise


def test_mock_assets_exist():
    assert MASTER.is_file()
    assert PDF.is_file()
    assert (MOCKS / "greenhouse_apply.html").is_file()
    assert (MOCKS / "lever_apply.html").is_file()
    assert (MOCKS / "linkedin_easy_apply.html").is_file()


def test_greenhouse_playbook_localhost(mock_server, profile_dir):
    from src.playbooks.greenhouse import run_greenhouse

    url = f"{mock_server}/greenhouse_apply.html"
    assert url.startswith("http://127.0.0.1:")
    ok = _run_playbook(
        run_greenhouse,
        url,
        PDF,
        MASTER,
        user_data_dir=profile_dir,
        headless=True,
    )
    assert ok is True


def test_lever_playbook_localhost(mock_server, profile_dir):
    from src.playbooks.lever import run_lever

    url = f"{mock_server}/lever_apply.html"
    assert url.startswith("http://127.0.0.1:")
    ok = _run_playbook(
        run_lever,
        url,
        PDF,
        MASTER,
        user_data_dir=profile_dir,
        headless=True,
    )
    assert ok is True


def test_linkedin_playbook_localhost(mock_server, profile_dir):
    from src.playbooks.linkedin import run_linkedin

    url = f"{mock_server}/linkedin_easy_apply.html"
    assert url.startswith("http://127.0.0.1:")
    ok = _run_playbook(
        run_linkedin,
        url,
        PDF,
        MASTER,
        user_data_dir=profile_dir,
        headless=True,
    )
    assert ok is True


def test_requires_human_status_accepted(tmp_path):
    from src.core.memory import VALID_STATUSES, init_db, log_application

    assert "REQUIRES_HUMAN_INTERVENTION" in VALID_STATUSES
    db = tmp_path / "t.db"
    init_db(db)
    log_application(
        "MockCo",
        "http://127.0.0.1/job",
        "swe",
        "REQUIRES_HUMAN_INTERVENTION",
        error_log="handoff soft-fail",
        db_path=db,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
