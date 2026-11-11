"""Critic, HITL freeze, audit log, execution jails."""

from __future__ import annotations

import json


def test_critic_plan_id_and_preview():
    from src.core import tools as t
    from src.core.copilot_plans import clear_plans_for_tests, peek_plan
    from src.core.critic import critique_tool_call

    clear_plans_for_tests()
    bad = critique_tool_call(
        tool_name="mark_applications_through",
        args={"plan_id": "mtp_deadbeefdeadbeef", "also_cleanup": True},
        user_text="yes mark those",
        risk="mutate_fs",
        allow_llm=False,
    )
    assert bad.proceed is False and "preview_mark_through" in bad.reason

    preview = critique_tool_call(
        tool_name="cleanup_manual_folder",
        args={"company_name": "Shure"},
        user_text="show me what you'd delete for Shure",
        risk="mutate_fs",
        allow_llm=False,
    )
    assert preview.proceed is False
    assert "preview" in preview.reason.lower() or "list" in preview.reason.lower()

    phrase = "Get rid of all the failed jobs."
    for tool, args in (
        (
            "mark_applications_through",
            {"plan_id": "mtp_deadbeefdeadbeef", "also_cleanup": True},
        ),
        ("cleanup_manual_folder", {"company_name": "FailedJobs"}),
        ("undo_bulk_user_marks", {"keep_company": ""}),
    ):
        v = critique_tool_call(
            tool_name=tool, args=args, user_text=phrase, risk="mutate_fs", allow_llm=False
        )
        assert v.proceed is False, tool


def test_critic_allows_real_plan(tmp_path, monkeypatch):
    from src.core import tools as t
    from src.core.copilot_plans import clear_plans_for_tests, peek_plan
    from src.core.critic import critique_tool_call

    clear_plans_for_tests()
    monkeypatch.setattr(t, "_manual_action_root", lambda: tmp_path)
    (tmp_path / "Eulogik").mkdir()
    preview = t.preview_mark_through.invoke({"until_company": "eulogik"})
    plan_id = [tok for tok in preview.split() if tok.startswith("plan_id=")][0].split(
        "=", 1
    )[1]
    assert peek_plan(plan_id) is not None
    v = critique_tool_call(
        tool_name="mark_applications_through",
        args={"plan_id": plan_id, "also_cleanup": True},
        user_text="yes mark those",
        risk="mutate_fs",
        allow_llm=False,
    )
    assert v.proceed is True


def test_db_jail():
    import sqlite3

    from src.core.db_jail import jailed_update_application_status

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE applications (id INTEGER PRIMARY KEY, status TEXT, "
        "error_log TEXT, timestamp TEXT, job_title TEXT)"
    )
    conn.execute("INSERT INTO applications VALUES (1,'MANUAL_ACTION','','','')")
    try:
        jailed_update_application_status(
            conn, 1, status="DROP_ME", error_log="USER_MARKED_SUBMITTED"
        )
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass

    conn.execute("DELETE FROM applications")
    conn.execute("INSERT INTO applications VALUES (1,'MANUAL_ACTION','','','x')")
    jailed_update_application_status(
        conn, 1, status="SUCCESS", error_log="USER_MARKED_SUBMITTED", job_title="Intern"
    )
    row = conn.execute("SELECT status, error_log FROM applications").fetchone()
    assert row[0] == "SUCCESS" and row[1] == "USER_MARKED_SUBMITTED"


def test_path_jail_blocks_escape(tmp_path, monkeypatch):
    from src.core import tools as t

    manual = tmp_path / "Manual_Action"
    manual.mkdir()
    (manual / "SafeCo").mkdir()
    outside = tmp_path / "Outside"
    outside.mkdir()
    monkeypatch.setattr(t, "_manual_action_root", lambda: manual)
    try:
        t._cleanup_manual_folder_impl("../Outside")
    except (PermissionError, FileNotFoundError, ValueError):
        pass
    assert outside.exists()


def test_resolve_approval_alias_and_state(monkeypatch):
    from concurrent.futures import Future

    from src.core.copilot import CopilotEngine, EngineState, PendingToolCall

    monkeypatch.setattr(
        "src.core.copilot._bind_chain",
        lambda role="user": (["fake"], [object()], [object()]),
    )
    monkeypatch.setattr("src.core.copilot._tool_map", lambda role="user": {})
    eng = CopilotEngine()
    eng._awaiting = PendingToolCall(
        call_id="c1", tool_name="cleanup_manual_folder", args={"company_name": "X"}
    )
    eng.state = EngineState.AWAITING_APPROVAL
    eng._approval_future = Future()
    events = eng.resolve_approval("c1", False)
    assert any(e.kind == "aborted" for e in events)
    assert eng.state == EngineState.IDLE and eng._awaiting is None


def test_hitl_audit_appends_jsonl(tmp_path, monkeypatch):
    from src.core import audit_log as al

    path = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(al, "audit_log_path", lambda: path)
    al.append_hitl_audit(
        call_id="c1",
        tool_name="cleanup_manual_folder",
        args={"company_name": "Shure"},
        user_text="delete manual for Shure",
        approved=True,
        risk="mutate_fs",
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["approved"] is True
    assert rows[0]["tool_name"] == "cleanup_manual_folder"
