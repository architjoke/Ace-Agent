"""Human-in-the-loop Self-Training Copilot (firewalled tool use).

UI-ready: CopilotEngine never calls input() — CLI / future FastAPI / React
approve PendingToolCall events via approve(call_id, approved).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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
- memorize_new_rule (ALWAYS use this for behavioral / form / filter / tailor instructions)
- update_config_preference (allowlisted config.yaml SYSTEM settings only)

Tool routing (critical):
- config.yaml = search locations, fit scores, workers, headless — NOT form text.
- lessons_learned / memorize_new_rule = "when you apply…", notice period, phone/PIN
  wording, don't-apply filters, resume/cover instructions.
- If unsure between config vs behavior → memorize_new_rule.

Few-shot examples:
Example 1: User says "Only look in Pune."
  → Action: update_config_preference(target_key="search_preferences.locations", new_value="Pune")

Example 2: User says "When asked for notice period, put 2 weeks."
  → Action: memorize_new_rule(category="APPLY_BEHAVIOR",
       rule_text="When asked for notice period, always input 1-14 days",
       enforcement_level="HARD")

Example 3: User says "Never apply to unpaid roles."
  → Action: memorize_new_rule(category="JOB_FILTERING",
       rule_text="Never apply to unpaid roles.",
       enforcement_level="HARD")

Example 4: User says "i applied midbrain"
  → Action: mark_application_submitted(company_name="midbrain") AND
            cleanup_manual_folder(company_name="midbrain") in one turn

Company name understanding:
- Partial names OK ("midbrain"); tools fuzzy-resolve. Prefer mark+cleanup together
  on clear apply/submit intents. find_companies only if unclear/Ambiguous.
- If find_companies (or mark/cleanup) says Ambiguous, ask the user which candidate.

Dynamic Preference Engine:
- HARD = dealbreaker; SOFT = -15 fit penalty.
- When memorizing, set enforcement_level to HARD for absolute dealbreakers
  (e.g. WhatsApp community, unpaid, must relocate abroad).
- Set enforcement_level to SOFT for fluid tastes (e.g. prefer remote, avoid tiny startups).

STRICTLY FORBIDDEN:
- Deleting source code, wiping the SQLite database, or resetting the agent's core memory.
- If the user asks to "reset everything", "wipe memory", "delete all rules", or
  "forget your training", you MUST refuse in plain language and call NO tools.
- Never invent a shell/SQL/file tool. You do not have them.
- Wipe DB / delete source / invent shell-SQL tools / "reset all memory".

Self-correction:
- If a tool returns "Tool execution failed: …", fix the arguments or choose a different
  allowed tool. Do not invent non-allowlisted config keys or tools that do not exist.

Be concise. After tools run, summarize what changed.
"""


@dataclass(frozen=True)
class PendingToolCall:
    call_id: str
    tool_name: str
    args: dict[str, Any]
    read_only: bool = False


@dataclass
class CopilotEvent:
    kind: str  # assistant | pending_tool | tool_result | tool_error | aborted | info
    text: str = ""
    pending: PendingToolCall | None = None


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


def _parse_tool_call(call: Any) -> tuple[str, dict[str, Any], str]:
    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
    call_id = (
        call.get("id")
        if isinstance(call, dict)
        else getattr(call, "id", name or "tool")
    )
    return str(name or ""), dict(args or {}), str(call_id)


class CopilotEngine:
    """HITL-free core: submit() / approve() for CLI or future GUI/API."""

    def __init__(self) -> None:
        self.llms = _bind_chain()
        self.tool_map = _tool_map()
        self.messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
        self._remaining_calls: list[Any] = []
        self._awaiting: PendingToolCall | None = None
        self._rounds_used = 0
        self._tools_ran_this_turn = False

    def _invoke_ai(self) -> tuple[AIMessage, list[CopilotEvent]]:
        """Invoke across provider chain; failover notices become info events."""
        notices: list[CopilotEvent] = []
        errors: list[str] = []
        chain = providers_for_role("default")
        for idx, llm in enumerate(self.llms):
            provider = chain[idx] if idx < len(chain) else f"provider-{idx + 1}"
            try:
                return llm.invoke(self.messages), notices
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                if idx < len(self.llms) - 1:
                    if _is_token_limit_error(exc):
                        retry = _retry_hint(exc)
                        notices.append(
                            CopilotEvent(
                                kind="info",
                                text=(
                                    f"Ace> {provider} hit token/rate limit.{retry} "
                                    f"Trying next provider…"
                                ),
                            )
                        )
                    elif _is_failover_error(exc):
                        notices.append(
                            CopilotEvent(
                                kind="info",
                                text=(
                                    f"Ace> {provider} failed ({type(exc).__name__}); "
                                    f"trying next…"
                                ),
                            )
                        )
                    logger.warning(
                        "Copilot provider %s failed; failover (%s)",
                        provider,
                        exc,
                    )
                    continue
                if _is_token_limit_error(exc):
                    raise RuntimeError(
                        f"All providers failed; last was {provider} token/rate limit. {exc}"
                    ) from exc
                raise
        raise RuntimeError("Copilot invoke failed: " + " | ".join(errors))

    def submit(self, user_text: str) -> list[CopilotEvent]:
        """Start a user turn. May end with pending_tool awaiting approve()."""
        if self._awaiting is not None:
            return [
                CopilotEvent(
                    kind="info",
                    text=(
                        "Ace> A tool is still awaiting approval "
                        f"({self._awaiting.tool_name}). Approve or deny it first."
                    ),
                )
            ]

        text = (user_text or "").strip()
        if not text:
            return []

        self._remaining_calls = []
        self._rounds_used = 0
        self._tools_ran_this_turn = False
        self.messages.append(HumanMessage(content=text))
        if len(self.messages) > 41:
            self.messages = [self.messages[0]] + self.messages[-40:]

        try:
            ai, notices = self._invoke_ai()
        except Exception as exc:
            self.messages.pop()
            return notices + [
                CopilotEvent(
                    kind="assistant",
                    text=_format_model_failure(exc, partial_tools=False),
                )
            ]

        return notices + self._handle_ai(ai)

    def approve(self, call_id: str, approved: bool) -> list[CopilotEvent]:
        """Resolve a PendingToolCall, then continue the tool / LLM loop."""
        if self._awaiting is None:
            return [
                CopilotEvent(
                    kind="info",
                    text="Ace> No tool is awaiting approval.",
                )
            ]
        if self._awaiting.call_id != call_id:
            return [
                CopilotEvent(
                    kind="info",
                    text=(
                        f"Ace> Expected approval for call_id={self._awaiting.call_id!r}, "
                        f"got {call_id!r}."
                    ),
                )
            ]

        pending = self._awaiting
        self._awaiting = None
        if self._remaining_calls:
            self._remaining_calls.pop(0)

        events: list[CopilotEvent] = []
        if not approved:
            events.append(
                CopilotEvent(
                    kind="aborted",
                    text="[INFO] Tool execution aborted by user.",
                )
            )
            self.messages.append(
                ToolMessage(
                    content=(
                        f"User aborted tool {pending.tool_name}. "
                        "Do not retry the same call unless the user asks again."
                    ),
                    tool_call_id=pending.call_id,
                )
            )
        else:
            events.extend(self._execute_tool(pending.tool_name, pending.args, pending.call_id))

        events.extend(self._drain_remaining())
        return events

    def _handle_ai(self, ai: AIMessage) -> list[CopilotEvent]:
        self.messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            text = ai.content if isinstance(ai.content, str) else str(ai.content)
            return [CopilotEvent(kind="assistant", text=f"Ace> {text}")]

        self._rounds_used += 1
        if self._rounds_used > MAX_TOOL_ROUNDS:
            # Satisfy tool_call_id contract so history stays valid
            for call in tool_calls:
                _name, _args, call_id = _parse_tool_call(call)
                self.messages.append(
                    ToolMessage(
                        content="Stopped: too many tool rounds. Ask the user again.",
                        tool_call_id=call_id,
                    )
                )
            return [
                CopilotEvent(
                    kind="info",
                    text="Ace> Stopped after too many tool rounds — ask again if needed.",
                )
            ]

        self._remaining_calls = list(tool_calls)
        return self._drain_remaining()

    def _drain_remaining(self) -> list[CopilotEvent]:
        events: list[CopilotEvent] = []
        while self._remaining_calls:
            call = self._remaining_calls[0]
            tool_name, args, call_id = _parse_tool_call(call)

            if tool_name not in self.tool_map:
                self._remaining_calls.pop(0)
                events.append(
                    CopilotEvent(
                        kind="tool_error",
                        text=f"[WARNING] Agent attempted to call unknown tool: {tool_name}",
                    )
                )
                self.messages.append(
                    ToolMessage(
                        content=(
                            f"Tool execution failed: unknown tool {tool_name!r}. "
                            "Please try a different approach or tool."
                        ),
                        tool_call_id=call_id,
                    )
                )
                continue

            read_only = tool_name in READ_ONLY_TOOLS
            if read_only:
                self._remaining_calls.pop(0)
                events.append(
                    CopilotEvent(
                        kind="info",
                        text=(
                            f"[lookup] {tool_name} "
                            f"{json.dumps(args, ensure_ascii=False)}"
                        ),
                    )
                )
                events.extend(self._execute_tool(tool_name, args, call_id))
                continue

            # Mutating — pause for HITL / GUI approve
            pending = PendingToolCall(
                call_id=call_id,
                tool_name=tool_name,
                args=args,
                read_only=False,
            )
            self._awaiting = pending
            events.append(
                CopilotEvent(kind="pending_tool", pending=pending)
            )
            return events

        # All tools in this AIMessage done → follow-up model turn
        try:
            ai, notices = self._invoke_ai()
            events.extend(notices)
        except Exception as exc:
            events.append(
                CopilotEvent(
                    kind="assistant",
                    text=_format_model_failure(
                        exc, partial_tools=self._tools_ran_this_turn
                    ),
                )
            )
            return events

        events.extend(self._handle_ai(ai))
        return events

    def _execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        call_id: str,
    ) -> list[CopilotEvent]:
        self._tools_ran_this_turn = True
        fn = self.tool_map[tool_name]
        try:
            result = fn.invoke(args)
        except Exception as exc:
            logger.exception("Copilot tool %s failed", tool_name)
            error_msg = (
                f"Tool execution failed: {exc}. "
                "Please try a different approach or tool."
            )
            self.messages.append(
                ToolMessage(content=error_msg, tool_call_id=call_id)
            )
            return [
                CopilotEvent(kind="tool_error", text=f"[TOOL ERROR] {exc}")
            ]

        self.messages.append(
            ToolMessage(content=str(result), tool_call_id=call_id)
        )
        return [
            CopilotEvent(kind="tool_result", text=f"[tool] {result}")
        ]


def _print_event(ev: CopilotEvent) -> None:
    if ev.kind == "pending_tool" and ev.pending is not None:
        p = ev.pending
        print(
            "\n[WARNING] The agent wants to execute "
            f"{p.tool_name} with arguments:\n"
            f"{json.dumps(p.args, indent=2, ensure_ascii=False)}"
        )
        return
    if ev.text:
        print(ev.text)


def start_chat_loop() -> int:
    """CLI adapter over CopilotEngine. Returns process exit code."""
    print("Ace Copilot — type a message (quit / exit to leave).")
    print("Mutating tools pause for Y/n; find_companies auto-runs (read-only).\n")
    try:
        engine = CopilotEngine()
    except Exception as exc:
        print(f"Copilot unavailable: {exc}")
        return 1

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

        queue = engine.submit(user)
        while queue:
            advanced = False
            for ev in queue:
                _print_event(ev)
                if ev.kind == "pending_tool" and ev.pending is not None:
                    raw = input("Proceed? (Y/n): ").strip().lower()
                    ok = raw in {"", "y", "yes"}
                    queue = engine.approve(ev.pending.call_id, ok)
                    advanced = True
                    break
            if not advanced:
                break

    return 0
