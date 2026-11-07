"""Human-in-the-loop Self-Training Copilot (firewalled tool use)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.llm_router import _build_provider, _is_failover_error, providers_for_role
from src.core.tools import READ_ONLY_TOOLS, get_copilot_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8

_RETRY_IN_RE = re.compile(
    r"try again in\s+([0-9]+m)?\s*([0-9]+(?:\.[0-9]+)?s)?",
    re.IGNORECASE,
)


def _is_token_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "429",
        "rate_limit",
        "rate limit",
        "ratelimit",
        "too many requests",
        "tokens per day",
        "tokens per minute",
        "tpd",
        "tpm",
        "quota",
        "resource_exhausted",
        "token limit",
        "daily token",
    )
    return any(n in msg for n in needles)


def _retry_hint(exc: BaseException) -> str:
    match = _RETRY_IN_RE.search(str(exc))
    if not match:
        return ""
    parts = [p for p in match.groups() if p]
    if not parts:
        return ""
    return " Try again in about " + " ".join(parts) + "."


def _guess_provider(exc: BaseException) -> str:
    msg = str(exc)
    lower = msg.lower()
    explicit = re.search(r"last was (\w+)", lower)
    if explicit:
        return explicit.group(1)
    for name in ("groq", "nvidia", "ollama", "gemini"):
        if re.search(rf"\b{name}\b", lower) or f"{name}.com" in lower:
            return name
    # Groq often returns model ids like openai/gpt-oss-120b with TPD wording
    if "tokens per day" in lower or "service tier" in lower:
        return "groq"
    return "LLM"


def _format_model_failure(exc: BaseException, *, partial_tools: bool = False) -> str:
    """Plain-language Ace> message so the user knows why chat stopped."""
    if _is_token_limit_error(exc):
        retry = _retry_hint(exc)
        provider = _guess_provider(exc)
        head = f"Failed: {provider} hit its token/rate limit.{retry}"
        if partial_tools:
            return (
                f"Ace> {head}\n"
                "Ace> Lookup/tools above may have run, but mark/cleanup did NOT finish.\n"
                "Ace> Wait for the limit to reset (or use another provider), then repeat "
                "your request (e.g. i applied midbrain)."
            )
        return (
            f"Ace> {head}\n"
            "Ace> Nothing was changed. Wait for the limit to reset, then try again."
        )
    if partial_tools:
        return (
            f"Ace> Follow-up model error after tools: {exc}\n"
            "Ace> Some tools may have run; mark/cleanup may be incomplete — check "
            "status or repeat the request."
        )
    return f"Ace> Model error: {exc}"

SYSTEM_PROMPT = """You are the Ace Job Agent Copilot.

Your job is to listen to the user, update application states, clean up manual folders,
and permanently learn new behavioral rules via tools.

You may ONLY use these tools:
- find_companies (fuzzy lookup — use when the company is unclear or Ambiguous)
- mark_application_submitted
- cleanup_manual_folder
- memorize_new_rule (ALWAYS use this when the user teaches a standing preference)
- update_config_preference (allowlisted config.yaml keys only)

Company name understanding:
- Users often give incomplete names ("midbrain", "MidBrains", "smartbridge").
- mark_application_submitted and cleanup_manual_folder fuzzy-resolve partial names
  internally — you MAY pass the user's fragment directly.
- When the user says they applied / submitted / finished a company:
  In ONE turn call BOTH mark_application_submitted AND cleanup_manual_folder
  with that company fragment (unless they only asked to mark or only to clean).
  Do NOT call find_companies first for clear apply/submit intents — that wastes a
  model round and can stall on rate limits.
- Call find_companies only if the name is unclear, or a prior tool returned Ambiguous.
- If find_companies (or mark/cleanup) says Ambiguous, ask the user which candidate.

Dynamic Preference Engine:
- When memorizing, set enforcement_level to HARD for absolute dealbreakers
  (e.g. WhatsApp community, unpaid, must relocate abroad).
- Set enforcement_level to SOFT for fluid tastes (e.g. prefer remote, avoid tiny startups).
  SOFT rules apply a score penalty and may still allow an exceptional match through.

STRICTLY FORBIDDEN:
- Deleting source code, wiping the SQLite database, or resetting the agent's core memory.
- If the user asks to "reset everything", "wipe memory", "delete all rules", or
  "forget your training", you MUST refuse in plain language and call NO tools.
- Never invent a shell/SQL/file tool. You do not have them.

Self-correction:
- If a tool returns "Tool execution failed: …", fix the arguments or choose a different
  allowed tool. Do not invent non-allowlisted config keys or tools that do not exist.

Be concise. After tools run, summarize what changed.
"""


def _tool_map() -> dict[str, Any]:
    return {t.name: t for t in get_copilot_tools()}


def _bind_chain() -> list[Any]:
    """Build bind_tools LLMs for every available provider (failover order)."""
    tools = get_copilot_tools()
    chain = providers_for_role("default")
    if not chain:
        raise RuntimeError("No text LLM available for copilot (set GROQ_API_KEY).")
    bound: list[Any] = []
    for provider in chain:
        try:
            llm = _build_provider(provider, temperature=0)
            bound.append(llm.bind_tools(tools))
        except Exception as exc:
            logger.warning("Copilot LLM bind failed on %s: %s", provider, exc)
    if not bound:
        raise RuntimeError("Could not bind copilot tools on any provider.")
    return bound


def _invoke_ai(llms: list[Any], messages: list[Any]) -> AIMessage:
    """Invoke across provider chain; failover on 429/timeouts/etc."""
    errors: list[str] = []
    chain = providers_for_role("default")
    for idx, llm in enumerate(llms):
        provider = chain[idx] if idx < len(chain) else f"provider-{idx + 1}"
        try:
            return llm.invoke(messages)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            if idx < len(llms) - 1:
                if _is_token_limit_error(exc):
                    retry = _retry_hint(exc)
                    print(
                        f"Ace> {provider} hit token/rate limit.{retry} "
                        f"Trying next provider…"
                    )
                elif _is_failover_error(exc):
                    print(f"Ace> {provider} failed ({type(exc).__name__}); trying next…")
                logger.warning(
                    "Copilot provider %s failed; failover (%s)",
                    provider,
                    exc,
                )
                continue
            # Exhausted chain — raise a clean, user-facing error
            if _is_token_limit_error(exc):
                raise RuntimeError(
                    f"All providers failed; last was {provider} token/rate limit. {exc}"
                ) from exc
            raise
    raise RuntimeError("Copilot invoke failed: " + " | ".join(errors))


def _confirm(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name in READ_ONLY_TOOLS:
        print(f"\n[lookup] {tool_name} {json.dumps(args, ensure_ascii=False)}")
        return True
    print(
        "\n[WARNING] The agent wants to execute "
        f"{tool_name} with arguments:\n{json.dumps(args, indent=2, ensure_ascii=False)}"
    )
    raw = input("Proceed? (Y/n): ").strip().lower()
    return raw in {"", "y", "yes"}


def _run_tool_calls(
    tool_calls: list[Any],
    tool_map: dict[str, Any],
    messages: list[Any],
) -> None:
    """Execute tool_calls with HITL; feed failures back as ToolMessage for self-correction."""
    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        call_id = (
            call.get("id")
            if isinstance(call, dict)
            else getattr(call, "id", name or "tool")
        )
        tool_name = str(name or "")
        args = dict(args or {})

        # Unknown tool — do not prompt HITL; warn and let the LLM self-correct
        if tool_name not in tool_map:
            print(f"[WARNING] Agent attempted to call unknown tool: {tool_name}")
            messages.append(
                ToolMessage(
                    content=(
                        f"Tool execution failed: unknown tool {tool_name!r}. "
                        "Please try a different approach or tool."
                    ),
                    tool_call_id=str(call_id),
                )
            )
            continue

        if not _confirm(tool_name, args):
            print("[INFO] Tool execution aborted by user.")
            messages.append(
                ToolMessage(
                    content=(
                        f"User aborted tool {tool_name}. "
                        "Do not retry the same call unless the user asks again."
                    ),
                    tool_call_id=str(call_id),
                )
            )
            continue

        fn = tool_map[tool_name]
        try:
            result = fn.invoke(args)
        except Exception as exc:
            # Log full traceback for ops; never dump stack to the terminal
            logger.exception("Copilot tool %s failed", tool_name)
            print(f"[TOOL ERROR] {exc}")
            error_msg = (
                f"Tool execution failed: {exc}. "
                "Please try a different approach or tool."
            )
            messages.append(
                ToolMessage(content=error_msg, tool_call_id=str(call_id))
            )
            continue

        print(f"[tool] {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=str(call_id)))


def start_chat_loop() -> int:
    """Interactive HITL copilot REPL. Returns process exit code."""
    print("Ace Copilot — type a message (quit / exit to leave).")
    print("Mutating tools pause for Y/n; find_companies auto-runs (read-only).\n")
    try:
        llms = _bind_chain()
        tool_map = _tool_map()
    except Exception as exc:
        print(f"Copilot unavailable: {exc}")
        return 1

    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            return 0

        messages.append(HumanMessage(content=user))
        if len(messages) > 41:
            messages = [messages[0]] + messages[-40:]

        try:
            ai = _invoke_ai(llms, messages)
        except Exception as exc:
            print(_format_model_failure(exc, partial_tools=False))
            messages.pop()
            continue

        # Agentic tool loop — keep going until the model stops calling tools
        for _round in range(MAX_TOOL_ROUNDS):
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                text = ai.content if isinstance(ai.content, str) else str(ai.content)
                print(f"Ace> {text}")
                break

            _run_tool_calls(tool_calls, tool_map, messages)

            try:
                ai = _invoke_ai(llms, messages)
            except Exception as exc:
                print(_format_model_failure(exc, partial_tools=True))
                break
        else:
            print("Ace> Stopped after too many tool rounds — ask again if needed.")

    return 0
