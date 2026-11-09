"""Multi-model role → provider preference matrix."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _patch_router(monkeypatch, *, keys: dict[str, str], ollama: bool = False):
    from src.core import llm_router as lr

    def fake_env(*names: str) -> str:
        for name in names:
            if keys.get(name):
                return keys[name]
        return ""

    monkeypatch.setattr(lr, "_env", fake_env)
    monkeypatch.setattr(lr, "_ollama_reachable", lambda: ollama)
    monkeypatch.setattr(lr, "_allow_gemini_text", lambda: False)
    monkeypatch.setattr(lr, "_load_dotenv", lambda: None)
    monkeypatch.setattr(lr, "_yaml_llm_config", lambda: {})
    lr.text_provider_name.cache_clear()
    return lr


def test_triage_cohere_primary(monkeypatch):
    lr = _patch_router(
        monkeypatch,
        keys={
            "GROQ_API_KEY": "g",
            "NVIDIA_API_KEY": "n",
            "COHERE_API_KEY": "c",
            "OPENAI_API_KEY": "o",
        },
    )
    chain = lr.providers_for_role("triage")
    assert chain[0] == "cohere"
    assert "groq" not in chain
    assert "openai" not in chain


def test_draft_openai_primary_no_cohere_or_groq(monkeypatch):
    lr = _patch_router(
        monkeypatch,
        keys={
            "GROQ_API_KEY": "g",
            "NVIDIA_API_KEY": "n",
            "COHERE_API_KEY": "c",
            "OPENAI_API_KEY": "o",
        },
    )
    chain = lr.providers_for_role("draft")
    assert chain[0] == "openai"
    assert "cohere" not in chain
    assert "groq" not in chain


def test_semantic_groq_then_nvidia(monkeypatch):
    lr = _patch_router(
        monkeypatch,
        keys={"GROQ_API_KEY": "g", "NVIDIA_API_KEY": "n", "OPENAI_API_KEY": "o"},
    )
    assert lr.providers_for_role("semantic") == ["groq", "nvidia"]


def test_default_openai_head(monkeypatch):
    lr = _patch_router(
        monkeypatch,
        keys={"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o", "NVIDIA_API_KEY": "n"},
    )
    assert lr.providers_for_role("default")[0] == "openai"


def test_reflect_skips_groq(monkeypatch):
    lr = _patch_router(
        monkeypatch,
        keys={"GROQ_API_KEY": "g", "NVIDIA_API_KEY": "n", "COHERE_API_KEY": "c"},
        ollama=True,
    )
    chain = lr.providers_for_role("reflect")
    assert chain[0] == "ollama"
    assert "groq" not in chain
