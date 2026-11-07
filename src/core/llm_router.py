"""Four-tier hybrid LLM routing — Gemini browser-only; cloud + night local text.

Default free stack:
  draft / triage / fallback  → Groq → NVIDIA → Ollama
  reflect / inbox            → Ollama → Groq → NVIDIA
  browser-use (executor)     → Gemini vision/DOM only (never this module)

Gemini text is OFF by default (ACE_ALLOW_GEMINI_TEXT=0) to protect apply quota.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

from src.core.config import CONFIG_YAML, ENV_FILE, PROJECT_ROOT

REPO_ROOT = PROJECT_ROOT
CONFIG_PATH = CONFIG_YAML

# Groq free catalog: llama-3.3-70b-versatile is retired. Prefer gpt-oss-120b.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
DEFAULT_NVIDIA_MODEL = "mistralai/mistral-small-3.1-24b-instruct-2503"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_GEMINI_TEXT_MODEL = "gemini-3.5-flash-lite"
# High-throughput default for free-tier browser-use; override GEMINI_BROWSER_MODEL=gemini-3.8-flash when capacity allows.
DEFAULT_GEMINI_BROWSER_MODEL = "gemini-3.5-flash-lite"

TextRole = Literal["triage", "draft", "reflect", "inbox", "fallback", "default"]
ProviderName = Literal["groq", "nvidia", "ollama", "gemini"]

_VALID_PROVIDERS = frozenset({"groq", "nvidia", "ollama", "gemini"})

# Preferred provider order per role when ACE_*_PROVIDER / config.yaml unset.
_ROLE_PREFERENCE: dict[str, tuple[str, ...]] = {
    "triage": ("groq", "nvidia", "ollama"),
    "draft": ("groq", "nvidia", "ollama"),
    "fallback": ("groq", "nvidia", "ollama"),
    "reflect": ("ollama", "groq", "nvidia"),
    "inbox": ("ollama", "groq", "nvidia"),
    "default": ("groq", "nvidia", "ollama"),
}

_ROLE_ENV: dict[str, str] = {
    "triage": "ACE_TRIAGE_PROVIDER",
    "draft": "ACE_DRAFT_PROVIDER",
    "reflect": "ACE_REFLECT_PROVIDER",
    "inbox": "ACE_INBOX_PROVIDER",
    "fallback": "ACE_FALLBACK_PROVIDER",
    "default": "ACE_TEXT_PROVIDER",
}

_YAML_ROLE_KEYS = {
    "triage": "triage",
    "draft": "draft",
    "reflect": "reflect",
    "inbox": "inbox",
    "fallback": "fallback",
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)
    except Exception:
        pass


def _env(*names: str) -> str:
    for name in names:
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


@lru_cache(maxsize=1)
def _yaml_llm_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        block = raw.get("llm") if isinstance(raw, dict) else None
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _allow_gemini_text() -> bool:
    flag = _env("ACE_ALLOW_GEMINI_TEXT").lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    yaml_flag = _yaml_llm_config().get("allow_gemini_text")
    return bool(yaml_flag)


def _ollama_reachable(base_url: str | None = None) -> bool:
    url = (base_url or _env("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    try:
        import urllib.request

        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _provider_available(provider: str) -> bool:
    if provider == "groq":
        return bool(_env("GROQ_API_KEY"))
    if provider == "nvidia":
        return bool(_env("NVIDIA_API_KEY"))
    if provider == "ollama":
        return _ollama_reachable()
    if provider == "gemini":
        return _allow_gemini_text() and bool(_env("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    return False


def _role_preference(role_key: str) -> tuple[str, ...]:
    base = list(_ROLE_PREFERENCE.get(role_key, _ROLE_PREFERENCE["default"]))
    yaml_cfg = _yaml_llm_config()
    yaml_role = str(yaml_cfg.get(_YAML_ROLE_KEYS.get(role_key, ""), "") or "").lower()
    if yaml_role in _VALID_PROVIDERS and yaml_role != "gemini":
        # Pin preferred head from config.yaml; keep remaining failover order unique.
        rest = [p for p in base if p != yaml_role]
        base = [yaml_role, *rest]
    if _allow_gemini_text() and "gemini" not in base:
        base.append("gemini")
    return tuple(base)


def providers_for_role(role: TextRole | str = "default") -> list[str]:
    """Ordered available providers for a role (for failover)."""
    _load_dotenv()
    role_key = (role or "default").strip().lower()
    if role_key not in _ROLE_PREFERENCE:
        role_key = "default"

    # ACE_TEXT_PROVIDER prefers a head provider but keeps the failover chain
    # (unlike the old exclusive pin that killed Groq→NVIDIA→Ollama).
    global_force = _env("ACE_TEXT_PROVIDER").lower()
    role_force = _env(_ROLE_ENV.get(role_key, "ACE_TEXT_PROVIDER")).lower()
    if not role_force and global_force in _VALID_PROVIDERS:
        role_force = global_force
    # role_force handled below (prefer head, then rest of chain)
    ordered: list[str] = []
    if role_force in _VALID_PROVIDERS:
        if role_force == "ollama" and not _ollama_reachable():
            logger.warning(
                "%s=ollama but daemon unreachable; using failover chain",
                _ROLE_ENV.get(role_key, "ACE_TEXT_PROVIDER"),
            )
        elif role_force == "gemini" and not _allow_gemini_text():
            logger.warning("Gemini text disabled (ACE_ALLOW_GEMINI_TEXT=0); ignoring pin")
        elif _provider_available(role_force):
            ordered.append(role_force)
        else:
            logger.warning(
                "%s=%s unavailable; using failover chain",
                _ROLE_ENV.get(role_key, "ACE_TEXT_PROVIDER"),
                role_force,
            )

    for candidate in _role_preference(role_key):
        if candidate not in ordered and _provider_available(candidate):
            ordered.append(candidate)
    return ordered


@lru_cache(maxsize=8)
def text_provider_name(role: TextRole | str = "default") -> str:
    """Primary text-provider id for a workload role."""
    chain = providers_for_role(role)
    if chain:
        return chain[0]
    if _allow_gemini_text() and _env("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        return "gemini"
    raise RuntimeError(
        f"No text LLM available for role={role}. "
        "Set GROQ_API_KEY and/or NVIDIA_API_KEY, or run Ollama "
        "(ACE_REFLECT_PROVIDER=ollama), or ACE_ALLOW_GEMINI_TEXT=1 as last resort."
    )


def _build_groq(*, temperature: float, model: str | None = None) -> Any:
    from langchain_groq import ChatGroq

    key = _env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    resolved = model or _env("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    max_tokens = int(_env("GROQ_MAX_TOKENS") or "8192")
    timeout = float(_env("ACE_GROQ_TIMEOUT") or "120")
    kwargs: dict[str, Any] = {
        "model": resolved,
        "api_key": key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": 1,
    }
    if "gpt-oss" in resolved:
        kwargs["reasoning_effort"] = _env("GROQ_REASONING_EFFORT") or "low"
    logger.info(
        "Text LLM → Groq model=%s max_tokens=%s effort=%s timeout=%ss",
        resolved,
        max_tokens,
        kwargs.get("reasoning_effort"),
        timeout,
    )
    return ChatGroq(**kwargs)


def _build_nvidia(*, temperature: float, model: str | None = None) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    key = _env("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY missing")
    resolved = model or _env("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
    timeout = float(_env("ACE_NVIDIA_TIMEOUT") or "120")
    logger.info("Text LLM → NVIDIA model=%s timeout=%ss", resolved, timeout)
    try:
        return ChatNVIDIA(
            model=resolved,
            api_key=key,
            temperature=temperature,
            timeout=timeout,
            max_retries=1,
        )
    except TypeError:
        # Older langchain_nvidia_ai_endpoints may not accept timeout=
        return ChatNVIDIA(model=resolved, api_key=key, temperature=temperature)


def _build_ollama(*, temperature: float) -> Any:
    from langchain_ollama import ChatOllama

    model = _env("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    base_url = _env("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
    logger.info("Text LLM → Ollama model=%s base=%s", model, base_url)
    # Cap hang time so day-path triage/draft cannot stall for minutes on local LLM.
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=300,
    )


def _build_gemini(*, temperature: float) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = _env("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY missing")
    model = _env("GEMINI_TEXT_MODEL", "GEMINI_MODEL") or DEFAULT_GEMINI_TEXT_MODEL
    logger.warning(
        "Text LLM → Gemini model=%s (ACE_ALLOW_GEMINI_TEXT=1 — burns browser quota)",
        model,
    )
    return ChatGoogleGenerativeAI(model=model, api_key=key, temperature=temperature)


def _build_provider(
    provider: str,
    *,
    temperature: float,
    model: str | None = None,
) -> Any:
    if provider == "groq":
        return _build_groq(temperature=temperature, model=model)
    if provider == "nvidia":
        return _build_nvidia(temperature=temperature, model=model)
    if provider == "ollama":
        return _build_ollama(temperature=temperature)
    if provider == "gemini":
        return _build_gemini(temperature=temperature)
    raise RuntimeError(f"Unknown provider: {provider}")


def get_text_llm(
    *,
    temperature: float = 0,
    role: TextRole | str = "default",
    model: str | None = None,
    provider: str | None = None,
) -> Any:
    """LangChain chat model for text-only nodes (NOT browser-use)."""
    _load_dotenv()
    resolved_provider = (provider or text_provider_name(role)).lower()
    logger.info("Text role=%s → provider=%s", role, resolved_provider)
    return _build_provider(resolved_provider, temperature=temperature, model=model)


def _is_failover_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "429",
        "413",
        "410",
        "404",
        "rate_limit",
        "rate limit",
        "ratelimit",
        "too many requests",
        "resource_exhausted",
        "quota",
        "timeout",
        "timed out",
        "timeouterror",
        "connection",
        "unavailable",
        "503",
        "502",
        "504",
        "overloaded",
        "tokens per minute",
        "tpm",
        "payload too large",
        "end of life",
        "no longer available",
        "httpx",
        "httpcore",
        "readerror",
        "connecterror",
        "remoteprotocolerror",
    )
    if any(n in msg for n in needles):
        return True
    if any(n in name for n in ("timeout", "connection", "ratelimit", "apistat", "http")):
        return True
    # httpx / openai-style status_code attribute
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if status is not None and int(status) in {408, 409, 429, 500, 502, 503, 504}:
            return True
    except (TypeError, ValueError):
        pass
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            code = int(getattr(response, "status_code", 0) or 0)
            if code in {408, 409, 429, 500, 502, 503, 504}:
                return True
        except (TypeError, ValueError):
            pass
    return False


def invoke_with_failover(
    role: TextRole | str,
    prompt: Any,
    *,
    temperature: float = 0,
    structured: Any | None = None,
    model: str | None = None,
) -> Any:
    """Invoke prompt across the role's provider chain on 429/413/timeouts.

    structured: optional Pydantic model for with_structured_output(...).
    """
    _load_dotenv()
    chain = providers_for_role(role)
    if not chain:
        raise RuntimeError(f"No providers available for role={role}")

    errors: list[str] = []
    for idx, provider in enumerate(chain):
        try:
            llm = _build_provider(provider, temperature=temperature, model=model)
            target = llm.with_structured_output(structured) if structured else llm
            logger.info(
                "invoke_with_failover role=%s provider=%s attempt=%s",
                role,
                provider,
                idx + 1,
            )
            return target.invoke(prompt)
        except Exception as exc:
            if provider == "groq" and (
                "413" in str(exc) or "tokens per minute" in str(exc).lower()
            ):
                # One same-provider TPM escape on smaller model.
                try:
                    logger.warning(
                        "Groq TPM/size error — retry once with %s",
                        DEFAULT_GROQ_FALLBACK_MODEL,
                    )
                    llm = _build_groq(
                        temperature=temperature,
                        model=DEFAULT_GROQ_FALLBACK_MODEL,
                    )
                    target = (
                        llm.with_structured_output(structured) if structured else llm
                    )
                    return target.invoke(prompt)
                except Exception as retry_exc:
                    exc = retry_exc

            errors.append(f"{provider}: {exc}")
            if _is_failover_error(exc) and idx < len(chain) - 1:
                nxt = chain[idx + 1]
                logger.warning(
                    "Text role=%s provider=%s failed; failover→%s (%s)",
                    role,
                    provider,
                    nxt,
                    exc,
                )
                continue
            if idx < len(chain) - 1:
                logger.warning(
                    "Non-failover error on %s but trying next provider: %s",
                    provider,
                    exc,
                )
                continue
            raise RuntimeError(
                f"All providers failed for role={role}. " + " | ".join(errors)
            ) from exc

    raise RuntimeError(f"All providers failed for role={role}. " + " | ".join(errors))


def gemini_browser_model_name() -> str:
    """Model id for browser-use ChatGoogle (vision/DOM)."""
    _load_dotenv()
    return _env("GEMINI_BROWSER_MODEL", "GEMINI_MODEL") or DEFAULT_GEMINI_BROWSER_MODEL
