"""Specialized multi-model text router — Gemini is browser vision only.

  triage (fit)     → Cohere → NVIDIA → Ollama
  draft / fallback → OpenAI → Cohere → NVIDIA → Ollama
  semantic (DOM)   → Groq → NVIDIA
  reflect / inbox  → Ollama → NVIDIA → Cohere
  default / copilot → Groq → Cohere → NVIDIA → Ollama → OpenAI

Gemini text only if ACE_ALLOW_GEMINI_TEXT=1. Groq stays off fit/draft to save RPM for DOM.
Cohere is the cloud escape hatch when OpenAI is out of credits / rate-limited.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

from src.core.config import CONFIG_YAML, ENV_FILE, PROJECT_ROOT
from src.core.utils import api_request_with_retry

REPO_ROOT = PROJECT_ROOT
CONFIG_PATH = CONFIG_YAML

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
# Catalog free endpoint for mistral-small-3.1 is deprecated (Extra data / parse flakes).
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
DEFAULT_COHERE_MODEL = "command-r-08-2024"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_GEMINI_TEXT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_BROWSER_MODEL = "gemini-3.5-flash-lite"

TextRole = Literal[
    "triage",
    "draft",
    "reflect",
    "inbox",
    "fallback",
    "default",
    "semantic",
    "copilot",
]
ProviderName = Literal["groq", "nvidia", "ollama", "gemini", "cohere", "openai"]

_VALID_PROVIDERS = frozenset(
    {"groq", "nvidia", "ollama", "gemini", "cohere", "openai"}
)

_ROLE_PREFERENCE: dict[str, tuple[str, ...]] = {
    "triage": ("cohere", "nvidia", "ollama"),
    "draft": ("openai", "cohere", "nvidia", "ollama"),
    "fallback": ("openai", "cohere", "nvidia", "ollama"),
    "semantic": ("groq", "nvidia"),
    "reflect": ("ollama", "nvidia", "cohere"),
    "inbox": ("ollama", "nvidia", "cohere"),
    # Copilot chat: prefer live free/paid keys; OpenAI last (often credit-empty).
    "default": ("groq", "cohere", "nvidia", "ollama", "openai"),
    "copilot": ("groq", "cohere", "nvidia", "ollama", "openai"),
}

_ROLE_ENV: dict[str, str] = {
    "triage": "ACE_TRIAGE_PROVIDER",
    "draft": "ACE_DRAFT_PROVIDER",
    "reflect": "ACE_REFLECT_PROVIDER",
    "inbox": "ACE_INBOX_PROVIDER",
    "fallback": "ACE_FALLBACK_PROVIDER",
    "semantic": "ACE_SEMANTIC_PROVIDER",
    "default": "ACE_TEXT_PROVIDER",
    "copilot": "ACE_COPILOT_PROVIDER",
}

_YAML_ROLE_KEYS = {
    "triage": "triage",
    "draft": "draft",
    "reflect": "reflect",
    "inbox": "inbox",
    "fallback": "fallback",
    "semantic": "semantic",
    "copilot": "copilot",
}

_FAILOVER_NEEDLES = (
    "429", "413", "410", "404", "rate_limit", "rate limit", "ratelimit",
    "too many requests", "resource_exhausted", "quota", "timeout", "timed out",
    "timeouterror", "connection", "unavailable", "503", "502", "504", "overloaded",
    "tokens per minute", "tpm", "payload too large", "end of life", "no longer available",
    "httpx", "httpcore", "readerror", "connecterror", "remoteprotocolerror",
    "insufficient_quota", "credit",
    "extra data", "jsondecode", "json decode", "expecting value", "deprecated",
)
_FAILOVER_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# Soft cooldown so burned TPD / empty OpenAI credits don't burn seconds every draft.
_PROVIDER_COOLDOWN_UNTIL: dict[str, float] = {}
_HARD_BURN_NEEDLES = (
    "tokens per day",
    "tpd",
    "no credits remaining",
    "insufficient_quota",
    "billing",
    "you exceeded your current quota",
)
# Set when every provider in a role chain is hard-burned / cooled — batch should pause.
_TOKEN_LIMIT_REACHED = False
_TOKEN_LIMIT_DETAIL = ""


def token_limit_reached() -> bool:
    return bool(_TOKEN_LIMIT_REACHED)


def token_limit_detail() -> str:
    return _TOKEN_LIMIT_DETAIL or "API token/quota limit reached"


def mark_token_limit_reached(detail: str = "") -> None:
    global _TOKEN_LIMIT_REACHED, _TOKEN_LIMIT_DETAIL
    _TOKEN_LIMIT_REACHED = True
    _TOKEN_LIMIT_DETAIL = (detail or "API token/quota limit reached").strip()[:400]
    logger.error(
        "TOKEN LIMIT REACHED — pausing applies until quotas reset (%s)",
        _TOKEN_LIMIT_DETAIL,
    )


def clear_token_limit_reached() -> None:
    global _TOKEN_LIMIT_REACHED, _TOKEN_LIMIT_DETAIL
    _TOKEN_LIMIT_REACHED = False
    _TOKEN_LIMIT_DETAIL = ""


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
    return bool(_yaml_llm_config().get("allow_gemini_text"))


def _ollama_reachable(base_url: str | None = None) -> bool:
    url = (base_url or _env("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    try:
        import urllib.request

        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def _provider_available(provider: str) -> bool:
    checks: dict[str, Callable[[], bool]] = {
        "groq": lambda: bool(_env("GROQ_API_KEY")),
        "nvidia": lambda: bool(_env("NVIDIA_API_KEY")),
        "cohere": lambda: bool(_env("COHERE_API_KEY")),
        "openai": lambda: bool(_env("OPENAI_API_KEY")),
        "ollama": _ollama_reachable,
        "gemini": lambda: _allow_gemini_text()
        and bool(_env("GOOGLE_API_KEY", "GEMINI_API_KEY")),
    }
    fn = checks.get(provider)
    return bool(fn and fn())


def _role_preference(role_key: str) -> tuple[str, ...]:
    base = list(_ROLE_PREFERENCE.get(role_key, _ROLE_PREFERENCE["default"]))
    yaml_role = str(
        _yaml_llm_config().get(_YAML_ROLE_KEYS.get(role_key, ""), "") or ""
    ).lower()
    if yaml_role in _VALID_PROVIDERS and yaml_role != "gemini":
        base = [yaml_role, *[p for p in base if p != yaml_role]]
    if _allow_gemini_text() and "gemini" not in base:
        base.append("gemini")
    return tuple(base)


def providers_for_role(role: TextRole | str = "default") -> list[str]:
    """Ordered available providers for a role (for failover)."""
    _load_dotenv()
    role_key = (role or "default").strip().lower()
    if role_key not in _ROLE_PREFERENCE:
        role_key = "default"

    global_force = _env("ACE_TEXT_PROVIDER").lower()
    role_force = _env(_ROLE_ENV.get(role_key, "ACE_TEXT_PROVIDER")).lower()
    if not role_force and global_force in _VALID_PROVIDERS:
        role_force = global_force

    ordered: list[str] = []
    if role_force in _VALID_PROVIDERS:
        env_name = _ROLE_ENV.get(role_key, "ACE_TEXT_PROVIDER")
        if role_force == "ollama" and not _ollama_reachable():
            logger.warning("%s=ollama but daemon unreachable; using failover chain", env_name)
        elif role_force == "gemini" and not _allow_gemini_text():
            logger.warning("Gemini text disabled (ACE_ALLOW_GEMINI_TEXT=0); ignoring pin")
        elif _provider_available(role_force):
            ordered.append(role_force)
        else:
            logger.warning("%s=%s unavailable; using failover chain", env_name, role_force)

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
        "Set OPENAI_API_KEY / COHERE_API_KEY / GROQ_API_KEY / NVIDIA_API_KEY, "
        "or run Ollama, or ACE_ALLOW_GEMINI_TEXT=1 as last resort."
    )


def _build_groq(*, temperature: float, model: str | None = None) -> Any:
    from langchain_groq import ChatGroq

    key = _env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    resolved = model or _env("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    kwargs: dict[str, Any] = {
        "model": resolved,
        "api_key": key,
        "temperature": temperature,
        "max_tokens": int(_env("GROQ_MAX_TOKENS") or "8192"),
        "timeout": float(_env("ACE_GROQ_TIMEOUT") or "120"),
        "max_retries": 1,
    }
    if "gpt-oss" in resolved:
        kwargs["reasoning_effort"] = _env("GROQ_REASONING_EFFORT") or "low"
    logger.info("Text LLM → Groq model=%s", resolved)
    return ChatGroq(**kwargs)


def _build_nvidia(*, temperature: float, model: str | None = None) -> Any:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    key = _env("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY missing")
    resolved = model or _env("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
    logger.info("Text LLM → NVIDIA model=%s", resolved)
    try:
        return ChatNVIDIA(
            model=resolved,
            api_key=key,
            temperature=temperature,
            timeout=float(_env("ACE_NVIDIA_TIMEOUT") or "120"),
            max_retries=1,
        )
    except TypeError:
        return ChatNVIDIA(model=resolved, api_key=key, temperature=temperature)


def _build_openai(*, temperature: float, model: str | None = None) -> Any:
    from langchain_openai import ChatOpenAI

    key = _env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    resolved = model or _env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    logger.info("Text LLM → OpenAI model=%s", resolved)
    return ChatOpenAI(
        model=resolved,
        api_key=key,
        temperature=temperature,
        timeout=float(_env("ACE_OPENAI_TIMEOUT") or "120"),
        max_retries=1,
    )


def _build_cohere(*, temperature: float, model: str | None = None) -> Any:
    from langchain_cohere import ChatCohere

    key = _env("COHERE_API_KEY")
    if not key:
        raise RuntimeError("COHERE_API_KEY missing")
    resolved = model or _env("COHERE_MODEL") or DEFAULT_COHERE_MODEL
    logger.info("Text LLM → Cohere model=%s", resolved)
    return ChatCohere(model=resolved, cohere_api_key=key, temperature=temperature)


def _build_ollama(*, temperature: float, model: str | None = None) -> Any:
    from langchain_ollama import ChatOllama

    resolved = model or _env("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    base_url = _env("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
    # Keep short — hung local inference was starving the apply queue (bake13).
    timeout = float(_env("ACE_OLLAMA_TIMEOUT") or "60")
    logger.info("Text LLM → Ollama model=%s timeout=%.0fs", resolved, timeout)
    return ChatOllama(
        model=resolved, base_url=base_url, temperature=temperature, timeout=timeout
    )


def _build_gemini(*, temperature: float, model: str | None = None) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = _env("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY missing")
    resolved = model or _env("GEMINI_TEXT_MODEL", "GEMINI_MODEL") or DEFAULT_GEMINI_TEXT_MODEL
    logger.warning("Text LLM → Gemini model=%s (burns browser quota)", resolved)
    return ChatGoogleGenerativeAI(model=resolved, api_key=key, temperature=temperature)


_BUILDERS: dict[str, Callable[..., Any]] = {
    "groq": _build_groq,
    "nvidia": _build_nvidia,
    "openai": _build_openai,
    "cohere": _build_cohere,
    "ollama": _build_ollama,
    "gemini": _build_gemini,
}


def _build_provider(
    provider: str,
    *,
    temperature: float,
    model: str | None = None,
) -> Any:
    builder = _BUILDERS.get(provider)
    if not builder:
        raise RuntimeError(f"Unknown provider: {provider}")
    return builder(temperature=temperature, model=model)


def _is_failover_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if any(n in msg for n in _FAILOVER_NEEDLES):
        return True
    if any(n in name for n in ("timeout", "connection", "ratelimit", "apistat", "http")):
        return True
    for attr in ("status_code", "status"):
        try:
            code = getattr(exc, attr, None)
            if code is not None and int(code) in _FAILOVER_STATUS:
                return True
        except (TypeError, ValueError):
            pass
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            if int(getattr(response, "status_code", 0) or 0) in _FAILOVER_STATUS:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _cooldown_remaining(provider: str) -> float:
    import time

    until = float(_PROVIDER_COOLDOWN_UNTIL.get(provider) or 0)
    return max(0.0, until - time.time())


def _mark_provider_cooldown(provider: str, exc: BaseException) -> None:
    """Park hard-burned providers briefly so draft doesn't retry them every call."""
    import re
    import time

    msg = str(exc).lower()
    if not any(n in msg for n in _HARD_BURN_NEEDLES):
        return
    # Prefer upstream "retry in XmYs" when present; else 10 minutes.
    secs = 600.0
    m = re.search(r"try again in\s+(\d+)m([\d.]+)s", msg)
    if m:
        secs = int(m.group(1)) * 60 + float(m.group(2))
    else:
        m2 = re.search(r"try again in\s+([\d.]+)\s*s", msg)
        if m2:
            secs = float(m2.group(1))
    secs = min(max(secs, 30.0), 3600.0)
    _PROVIDER_COOLDOWN_UNTIL[provider] = time.time() + secs
    logger.warning(
        "Provider %s cooling down for %.0fs after burn (%s)",
        provider,
        secs,
        type(exc).__name__,
    )


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

    def _call(provider: str, override_model: str | None = None) -> Any:
        llm = _build_provider(
            provider, temperature=temperature, model=override_model or model
        )
        target = llm.with_structured_output(structured) if structured else llm
        return target.invoke(prompt)

    errors: list[str] = []
    for idx, provider in enumerate(chain):
        cool = _cooldown_remaining(provider)
        if cool > 0:
            errors.append(f"{provider}: cooling {cool:.0f}s")
            if idx >= len(chain) - 1:
                raise RuntimeError(
                    f"All providers failed for role={role}. " + " | ".join(errors)
                )
            logger.info(
                "Skipping cooled provider %s (%.0fs left) → %s",
                provider,
                cool,
                chain[idx + 1],
            )
            continue
        try:
            logger.info(
                "invoke_with_failover role=%s provider=%s attempt=%s",
                role,
                provider,
                idx + 1,
            )
            return api_request_with_retry(
                lambda p=provider: _call(p),
                label=f"{role}/{provider}",
            )
        except Exception as exc:
            # Groq TPM/size: one same-provider escape on smaller model.
            if provider == "groq" and (
                "413" in str(exc) or "tokens per minute" in str(exc).lower()
            ):
                try:
                    logger.warning(
                        "Groq TPM/size — retry once with %s", DEFAULT_GROQ_FALLBACK_MODEL
                    )
                    return api_request_with_retry(
                        lambda: _call("groq", DEFAULT_GROQ_FALLBACK_MODEL),
                        label=f"{role}/groq-fallback",
                    )
                except Exception as retry_exc:
                    exc = retry_exc

            _mark_provider_cooldown(provider, exc)
            errors.append(f"{provider}: {exc}")
            if idx >= len(chain) - 1:
                joined = " | ".join(errors)
                hard = any(
                    any(n in e.lower() for n in _HARD_BURN_NEEDLES) for e in errors
                ) or all(_cooldown_remaining(p) > 0 for p in chain)
                if hard:
                    mark_token_limit_reached(joined)
                    raise RuntimeError(
                        f"TOKEN_LIMIT_REACHED: All providers failed for role={role}. {joined}"
                    ) from exc
                raise RuntimeError(
                    f"All providers failed for role={role}. " + joined
                ) from exc
            nxt = chain[idx + 1]
            kind = "failover" if _is_failover_error(exc) else "non-failover"
            logger.warning(
                "Text role=%s provider=%s %s→%s (%s)",
                role,
                provider,
                kind,
                nxt,
                exc,
            )

    joined = " | ".join(errors)
    if any(any(n in e.lower() for n in _HARD_BURN_NEEDLES) for e in errors):
        mark_token_limit_reached(joined)
        raise RuntimeError(
            f"TOKEN_LIMIT_REACHED: All providers failed for role={role}. {joined}"
        )
    raise RuntimeError(f"All providers failed for role={role}. {joined}")


def gemini_browser_model_name() -> str:
    """Model id for browser-use ChatGoogle (vision/DOM)."""
    _load_dotenv()
    return _env("GEMINI_BROWSER_MODEL", "GEMINI_MODEL") or DEFAULT_GEMINI_BROWSER_MODEL


def message_text(content: Any) -> str:
    """Flatten LangChain / Gemini content-block lists into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                btype = str(block.get("type") or "").lower()
                if text and btype in {"", "text", "output_text", "answer"}:
                    parts.append(str(text))
                elif text and btype not in {"reasoning", "thinking"}:
                    parts.append(str(text))
                elif text and not parts:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    text = getattr(content, "text", None)
    if text:
        return str(text)
    return str(content)
