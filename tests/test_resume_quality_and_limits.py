"""Resume quality gates + HIT LIMIT stop behavior."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_phone_locale_avoids_india_trunk_zero():
    from src.core.form_answers import sanitize_application_answers
    from src.core.tailor import _apply_design
    import yaml

    out = _apply_design(
        "cv:\n  name: T\n  phone: '+91 8572855980'\n  sections:\n    summary: [x]\n",
        "balanced",
    )
    data = yaml.safe_load(out)
    assert data["locale"]["phone_number_format"] == "international"
    assert data["cv"]["phone"] == "+91 8572855980"
    assert not str(data["cv"]["phone"]).lstrip("+").startswith("0")

    ans = sanitize_application_answers(
        {"phone_national": "08572855980", "phone_country_code": "+91"}
    )
    assert ans["phone_national"] == "8572855980"
    assert "0857" not in ans["phone_display"]
    assert ans["phone_display"].startswith("+91 ")


def test_phone_respects_non_india_calling_code():
    from src.core.form_answers import normalize_phone_fields, sanitize_application_answers

    ans = sanitize_application_answers(
        {"phone_national": "5551234567", "phone_country_code": "+1"}
    )
    assert ans["phone_national"] == "5551234567"
    assert ans["country_calling_code"] == "1"
    assert ans["phone_display"] == "+1 5551234567"
    assert ans["phone_e164"] == "+15551234567"
    assert "+91" not in ans["phone_display"]

    cv_phone = normalize_phone_fields(
        {"phone": "+1 5551234567", "phone_country_code": "+1"}
    )
    assert cv_phone["phone"] == "+1 5551234567"


def test_stopish_drops_junk_pronouns(monkeypatch):
    from src.core import ats_scanner as ats

    # Avoid spaCy/torch on Windows CI hosts — exercise stop-list via regex path.
    monkeypatch.setattr(ats, "_spacy_nouns", lambda _t: [])
    jd = (
        "You will build and add features. You are a great fit. "
        "Required: Python, FastAPI, PostgreSQL, REST API."
    )
    kws = ats.extract_jd_keywords(jd, top_k=20)
    lowered = {k.lower() for k in kws}
    assert "you" not in lowered
    assert "build" not in lowered
    assert "add" not in lowered
    assert "are" not in lowered
    assert any("python" in k for k in lowered) or "python" in lowered
    assert any("fastapi" in k or "rest" in k for k in lowered)


def test_supportable_uses_word_boundaries():
    from src.core.ats_scanner import filter_supportable_keywords

    master = "average latency improved with Cursor and Claude tooling"
    supportable, unsupported = filter_supportable_keywords(
        ["rag", "cursor", "claude"], master
    )
    assert "rag" in unsupported
    assert "cursor" in supportable
    assert "claude" in supportable


def test_extract_known_tokens_use_word_boundaries(monkeypatch):
    from src.core import ats_scanner as ats

    monkeypatch.setattr(ats, "_spacy_nouns", lambda _t: [])
    # "average" must not yield "rag"; "capital" must not yield bare "api".
    kws = {
        k.lower()
        for k in ats.extract_jd_keywords(
            "On average capital markets use Python.", top_k=20
        )
    }
    assert "rag" not in kws
    assert "api" not in kws


def test_normalize_project_github_location_consistent():
    from src.core.tailor import _normalize_project_links

    data = {
        "cv": {
            "sections": {
                "projects": [
                    {
                        "name": "Road Inspection System",
                        "location": "[GitHub](https://github.com/MSG-Technologies/AI-Road-Inspection)",
                        "highlights": ["a", "b", "c"],
                    },
                    {
                        "name": "Earthquake Early Warning",
                        "location": "https://github.com/architjoke/earthquake-warning",
                        "highlights": ["a", "b", "c"],
                    },
                ]
            }
        }
    }
    _normalize_project_links(data)
    locs = [p["location"] for p in data["cv"]["sections"]["projects"]]
    assert all(loc.startswith("[GitHub](https://github.com/") for loc in locs)
    assert all(loc.endswith(")") for loc in locs)


def test_density_flags_missing_education_and_raw_github():
    from src.core.tailor import _bullet_density_issues

    yaml_text = """
cv:
  name: Test
  sections:
    experience:
      - company: A
        position: Intern
        highlights: ["one", "two", "three"]
      - company: B
        position: Intern
        highlights: ["one", "two", "three"]
    projects:
      - name: Proj1
        location: https://github.com/x/y
        highlights: ["one", "two", "three"]
      - name: Proj2
        location: "[GitHub](https://github.com/x/z)"
        highlights: ["one", "two", "three"]
    skills:
      - label: Lang
        details: Python
"""
    issues = _bullet_density_issues(yaml_text)
    assert any("Education missing" in i for i in issues)
    assert any("GitHub location" in i for i in issues)
    assert any("Summary missing" in i for i in issues)
    assert any("Activities missing" in i for i in issues)


def test_fact_bank_contracts_fill_sparse_draft():
    from src.core.tailor import _bullet_density_issues, _enforce_one_page_budget

    master = """
cv:
  name: Archit Gupta
  sections:
    summary:
      - "CSE student building AI products with Python and Node.js."
    education:
      - institution: Jaypee University
        area: Computer Science Engineering
        degree: B.Tech
        start_date: "2024-08"
        end_date: "2027-05"
        highlights: ["CGPA: 8.8/10"]
    experience:
      - company: MSG Technologies
        position: AI Engineer Intern
        start_date: "2026-08"
        end_date: present
        location: Remote
        highlights:
          - "Building a client SaaS US CMA learning platform with Node.js REST APIs end-to-end."
          - "Integrating AI capabilities into product workflows with TypeScript services."
          - "Designing database schemas for high-throughput API sessions and experiments."
      - company: Infosys Springboard
        position: AI Intern (Virtual)
        start_date: "2025-09"
        end_date: "2025-11"
        highlights:
          - "Developed ImpactSense ML pipeline with Scikit-learn for earthquake impact."
          - "Implemented Random Forest and XGBoost models with careful evaluation loops."
          - "Executed data preprocessing and feature engineering on seismic datasets."
    projects:
      - name: AI Road Inspection System
        summary: "Python, Flask, YOLOv8"
        highlights:
          - "Built Flask road inspection platform with PostgreSQL and GIS map workflows."
          - "Trained YOLOv8 detection for potholes and cracks with background analysis."
          - "Implemented OpenCV verification workflows for before/after repair checks."
      - name: Earthquake Impact Prediction System
        summary: "Python, Scikit-learn"
        highlights:
          - "Architected predictive ML pipeline with Scikit-learn on seismic features."
          - "Evaluated XGBoost and Random Forest ensembles for reliability gains."
          - "Implemented feature engineering protocols on unstructured seismic data."
    skills:
      - label: Programming Languages
        details: Python, JavaScript, TypeScript, SQL
      - label: Backend & Web Engineering
        details: Node.js, REST APIs, Flask, PostgreSQL
      - label: AI & Machine Learning
        details: Scikit-learn, TensorFlow, PyTorch, Cursor, Claude
      - label: DevOps & Tooling
        details: Git, GitHub, Docker, Cursor
    hackathons_and_activities:
      - "Hackathons: RIDEhack and Amazon ML Challenge."
      - "Campus Leadership: NEP Saarthi Campus Ambassador."
"""
    sparse = """
cv:
  name: Archit Gupta
  sections:
    experience:
      - company: MSG Technologies
        position: Wrong Title
        start_date: "2020-01"
        end_date: "2020-06"
        highlights:
          - "Built REST APIs in Node.js for SaaS sessions."
      - company: Infosys Springboard
        position: AI Intern (Virtual)
        highlights:
          - "Developed ImpactSense ML pipeline with Scikit-learn for earthquake impact."
          - "Implemented Random Forest and XGBoost models with careful evaluation loops."
          - "Executed data preprocessing and feature engineering on seismic datasets."
    projects:
      - name: AI Road Inspection System
        location: "[GitHub](https://github.com/MSG-Technologies/AI-Road-Inspection)"
        highlights:
          - "Built Flask road inspection platform with PostgreSQL and GIS map workflows."
          - "Trained YOLOv8 detection for potholes and cracks with background analysis."
          - "Implemented OpenCV verification workflows for before/after repair checks."
      - name: Earthquake Impact Prediction System
        location: "[GitHub](https://github.com/architjoke/earthquake-impact-app)"
        highlights:
          - "Architected predictive ML pipeline with Scikit-learn on seismic features."
          - "Evaluated XGBoost and Random Forest ensembles for reliability gains."
          - "Implemented feature engineering protocols on unstructured seismic data."
"""
    jd = "Product Intern building AI LLM prototypes with RAG, APIs, and Cursor."
    out = _enforce_one_page_budget(
        sparse,
        jd=jd,
        master_yaml=master,
        keywords=["rest api", "node.js", "cursor", "scikit-learn"],
    )
    assert "summary:" in out.lower() or "CSE student" in out
    assert "Jaypee University" in out
    assert "hackathons_and_activities:" in out
    assert "AI Engineer Intern" in out  # anchored title
    assert "2026-08" in out  # anchored dates
    assert "**" in out  # JD bold injected
    assert "Cursor" in out or "cursor" in out.lower()
    issues = [
        i
        for i in _bullet_density_issues(out)
        if "chars" not in i  # length is soft critique; structure must pass
    ]
    assert issues == [], issues


def test_density_uses_master_budget_not_draft_inference():
    from src.core.page_budget import resolve_page_budget
    from src.core.tailor import _bullet_density_issues

    master_exp = [
        {"company": "A", "position": "SDE Intern", "highlights": ["x"] * 3},
        {"company": "B", "position": "AI Intern", "highlights": ["x"] * 3},
        {"company": "C", "position": "Backend Intern", "highlights": ["x"] * 3},
    ]
    budget = resolve_page_budget(
        experience=master_exp,
        projects=[{"name": "P1"}, {"name": "P2"}],
    )
    assert budget.exp_keep == 3
    # Draft only kept 2 roles — must fail against master budget.
    bullet = (
        "Built production APIs with FastAPI and PostgreSQL for high-throughput "
        "sessions across client SaaS workloads."
    )
    draft = f"""
cv:
  name: T
  sections:
    summary: ["Dense summary paragraph for the role covering ML and backend."]
    education:
      - institution: Uni
        degree: B.Tech
    experience:
      - company: A
        position: Intern
        highlights: ["{bullet}", "{bullet}", "{bullet}"]
      - company: B
        position: Intern
        highlights: ["{bullet}", "{bullet}", "{bullet}"]
    projects:
      - name: P1
        location: "[GitHub](https://github.com/org/p1)"
        highlights: ["{bullet}", "{bullet}", "{bullet}"]
      - name: P2
        location: "[GitHub](https://github.com/org/p2)"
        highlights: ["{bullet}", "{bullet}", "{bullet}"]
    skills:
      - label: Lang
        details: Python
      - label: Backend
        details: FastAPI
      - label: Data
        details: SQL
    hackathons_and_activities:
      - "Hackathon win with team delivery."
"""
    issues = _bullet_density_issues(draft, budget=budget)
    assert any(
        "need 3" in i.lower() or "3 (" in i.lower() or "3 roles" in i.lower()
        for i in issues
    ), issues


def test_degraded_fallback_not_cached(tmp_path, monkeypatch):
    from src.core import resume_artifacts as ra
    from src.core import tailor as t

    db = tmp_path / "artifacts.db"
    monkeypatch.setattr(ra, "_ARTIFACTS_PATH", db)
    monkeypatch.setattr(
        "src.core.security.validate_safe_path",
        lambda p, **k: Path(p),
    )
    fallback = tmp_path / "base.pdf"
    fallback.write_bytes(b"%PDF")
    monkeypatch.setattr(t, "_default_base_resume_pdf", lambda: fallback)

    class _G:
        def invoke(self, *_a, **_k):
            return {
                "pdf_path": "",
                "current_tailored_yaml": "cv:\n  name: T\n",
                "ats_score": 90,
                "ats_pass_score": 85,
                "missing_keywords": [],
                "compile_error": "boom",
                "density_ok": True,
            }

    monkeypatch.setattr(t, "get_tailor_graph", lambda: _G())
    monkeypatch.setattr(
        "src.core.resume_artifacts.jd_signature",
        lambda *a, **k: ("deadbeef", ["python"]),
    )
    raised = False
    try:
        t.tailor_with_graph("Python role", "skills:\n  - Python\n", company="Acme")
    except FileNotFoundError as exc:
        raised = True
        assert "TAILOR_DEGRADED" in str(exc)
    assert raised
    assert ra.lookup_artifact("deadbeef", db_path=db) is None


def test_page_budget_scales_with_three_strong_roles():
    from src.core.page_budget import apply_budget_to_sections, resolve_page_budget
    from src.core.tailor import _rank_rows_by_jd

    experience = [
        {
            "company": "A",
            "position": "SDE Intern",
            "highlights": ["a1 " * 20, "a2 " * 20, "a3 " * 20, "a4 " * 20],
        },
        {
            "company": "B",
            "position": "AI Intern",
            "highlights": ["b1 " * 20, "b2 " * 20, "b3 " * 20],
        },
        {
            "company": "C",
            "position": "Backend Intern",
            "highlights": ["c1 " * 20, "c2 " * 20, "c3 " * 20],
        },
        {
            "company": "Campus",
            "position": "University Intern",
            "highlights": ["soft filler only"],
        },
    ]
    projects = [
        {"name": "P1", "highlights": ["p1a " * 20, "p1b " * 20, "p1c " * 20]},
        {"name": "P2", "highlights": ["p2a " * 20, "p2b " * 20, "p2c " * 20]},
    ]
    budget = resolve_page_budget(experience=experience, projects=projects)
    assert budget.exp_keep == 3
    assert budget.proj_keep == 2
    assert budget.reason == "three_strong_roles"

    sections = {"experience": experience, "projects": projects}
    apply_budget_to_sections(
        sections, budget, rank_fn=_rank_rows_by_jd, jd="backend API Node.js"
    )
    assert len(sections["experience"]) == 3
    assert all(
        "university intern" not in str(r.get("position") or "").lower()
        for r in sections["experience"]
    )
    assert len(sections["projects"]) == 2
    assert all(len(r["highlights"]) <= 3 for r in sections["experience"])


def test_page_budget_four_plus_keeps_top_three_one_project():
    from src.core.page_budget import resolve_page_budget

    experience = [
        {"company": f"Co{i}", "position": "Engineer Intern", "highlights": ["x"] * 3}
        for i in range(5)
    ]
    budget = resolve_page_budget(experience=experience, projects=[{"name": "P"}] * 3)
    assert budget.exp_keep == 3
    assert budget.proj_keep == 1
    assert budget.reason == "four_plus_strong_roles_top3"


def test_hard_burn_marks_hit_limit_immediately():
    from src.core.llm_router import (
        _mark_provider_cooldown,
        clear_token_limit_reached,
        token_limit_reached,
    )

    clear_token_limit_reached()
    _mark_provider_cooldown(
        "openai",
        RuntimeError(
            "Error code: 429 - You have no credits remaining. Add credits to continue"
        ),
    )
    assert token_limit_reached() is True
    clear_token_limit_reached()


def test_soft_tpm_stops_after_repeats():
    from src.core import llm_router as lr

    lr.clear_token_limit_reached()
    exc = RuntimeError(
        "Error code: 429 - rate_limit_exceeded tokens per minute (TPM): Limit 8000"
    )
    for _ in range(lr._SOFT_RATE_STOP_AFTER - 1):
        lr._mark_provider_cooldown("groq", exc)
        assert lr.token_limit_reached() is False
    lr._mark_provider_cooldown("groq", exc)
    assert lr.token_limit_reached() is True
    assert "rate limit" in lr.token_limit_detail().lower()
    lr.clear_token_limit_reached()


def test_invoke_raises_when_limit_already_hit(monkeypatch):
    from src.core import llm_router as lr

    lr.clear_token_limit_reached()
    lr.mark_token_limit_reached("test burn")
    try:
        raised = False
        try:
            lr.invoke_with_failover("draft", "hello")
        except RuntimeError as exc:
            raised = True
            assert "TOKEN_LIMIT_REACHED" in str(exc) or "HIT LIMIT" in str(exc)
        assert raised
    finally:
        lr.clear_token_limit_reached()
