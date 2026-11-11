"""Company research brief (offline-safe units + optional live)."""

from __future__ import annotations


def test_research_requires_name():
    from src.core.company_research import research_company_brief

    try:
        research_company_brief("  ")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_research_uses_ace_memory(monkeypatch):
    from src.core import company_research as cr

    monkeypatch.setattr(cr, "_wiki_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(cr, "_ddg_instant", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cr,
        "_ace_history",
        lambda company: {
            "resolved_company": "FlytBase",
            "ambiguous": None,
            "applications": [
                {
                    "company": "FlytBase",
                    "status": "MANUAL_ACTION",
                    "job_title": "Intern",
                    "job_url": "https://example.com/job",
                    "outcome_status": "PENDING",
                    "timestamp": "2026-11-11T10:00:00",
                }
            ],
            "manual_action": True,
            "manual_folder": "FlytBase",
        },
    )
    out = cr.research_company_brief("flytbase", focus="interview")
    assert "FlytBase" in out
    assert "Ace memory" in out
    assert "Intern" in out
    assert "interview" in out.lower()


def test_best_match_unique():
    from src.core.company_research import _best_match

    hit, amb = _best_match("eulog", ["Eulogik", "Infosys"])
    assert hit == "Eulogik"
    assert amb == []
