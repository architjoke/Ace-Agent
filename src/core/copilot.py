"""Human-in-the-loop Self-Training Copilot (firewalled tool use).

UI-ready: CopilotEngine never calls input() — CLI / future FastAPI / React
approve PendingToolCall events via approve(call_id, approved).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.chat_memory import get_memory_store
from src.core.copilot_capabilities import (
    SESSION_MUTATION_BUDGET,
    ChatRole,
    build_copilot_system_prompt,
    tool_names_for_role,
)
from src.core.critic import _AMBIGUOUS_BULK_RE, critique_tool_call
from src.core.llm_router import (
    _build_provider,
    _cooldown_remaining,
    _is_failover_error,
    _mark_provider_cooldown,
    providers_for_role,
)
from src.core.memory_extract import schedule_memory_extract
from src.core.security import redact_secrets, wrap_data_block, wrap_untrusted
from src.core.semantic_router import (
    UNAUTHORIZED_REFUSAL,
    UNSAFE_REFUSAL,
    classify_route,
)
from src.core.tool_schemas import try_validate_tool_args
from src.core.tools import TOOL_RISK, get_copilot_tools, set_chat_role

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
_RETRIEVED_MARKER = "<!--ace_retrieved_memory-->"
_KNOWLEDGE_MARKER = "<!--ace_knowledge_prefetch-->"
STM_MAX_MESSAGES = 41  # system (+ optional retrieved) + ~40 turns


class EngineState(str, Enum):
    IDLE = "idle"
    TURN = "turn"
    AWAITING_APPROVAL = "awaiting_approval"

_RETRY_IN_RE = re.compile(
    r"try again in\s+([0-9]+m)?\s*([0-9]+(?:\.[0-9]+)?s)?",
    re.IGNORECASE,
)
_TOKEN_LIMIT_NEEDLES = (
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
    "no credits remaining",
    "insufficient_quota",
)

# Natural-language shortcuts so common intents don't depend on a flaky model.
_BULK_THROUGH_RE = re.compile(
    r"(?:i\s+)?(?:applied|submitted|done(?:\s+with)?|finished)"
    r".{0,60}?(?:all|everything|every(?:thing)?|the\s+all)"
    r".{0,40}?(?:till|until|upto|up\s+to|through|thru)\s+"
    r"['\"]?([A-Za-z0-9][\w .&'-]{1,60})",
    re.IGNORECASE,
)
_SINGLE_APPLIED_RE = re.compile(
    r"^(?:i\s+)?(?:applied|submitted|done(?:\s+with)?)\s+"
    r"(?!all\b|everything\b|every\b)"
    r"['\"]?([A-Za-z0-9][\w .&'-]{1,60})['\"]?\s*$",
    re.IGNORECASE,
)


def _is_token_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(n in msg for n in _TOKEN_LIMIT_NEEDLES)


def _retry_hint(exc: BaseException) -> str:
    match = _RETRY_IN_RE.search(str(exc))
    if not match:
        return ""
    parts = [p for p in match.groups() if p]
    return (" Try again in about " + " ".join(parts) + ".") if parts else ""


def _guess_provider(exc: BaseException) -> str:
    lower = str(exc).lower()
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
        head = (
            f"Failed: {_guess_provider(exc)} hit its token/rate limit."
            f"{_retry_hint(exc)}"
        )
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


@dataclass(frozen=True)
class PendingToolCall:
    call_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass
class CopilotEvent:
    kind: str  # assistant | pending_tool | tool_result | tool_error | aborted | info
    text: str = ""
    pending: PendingToolCall | None = None


def _tool_map(role: ChatRole = "user") -> dict[str, Any]:
    return {t.name: t for t in get_copilot_tools(role)}


def _bind_chain(
    role: ChatRole = "user",
) -> tuple[list[str], list[Any], list[Any]]:
    """Build tool-bound + plain LLMs for every available provider (role-scoped tools)."""
    tools = get_copilot_tools(role)
    chain = providers_for_role("copilot") or providers_for_role("default")
    if not chain:
        raise RuntimeError(
            "No text LLM available for copilot "
            "(set GROQ_API_KEY / COHERE_API_KEY / NVIDIA_API_KEY)."
        )
    names: list[str] = []
    bound: list[Any] = []
    plain: list[Any] = []
    for provider in chain:
        try:
            llm = _build_provider(provider, temperature=0)
            names.append(provider)
            plain.append(llm)
            bound.append(llm.bind_tools(tools))
        except Exception as exc:
            logger.warning("Copilot LLM bind failed on %s: %s", provider, exc)
    if not bound:
        raise RuntimeError("Could not bind copilot tools on any provider.")
    return names, bound, plain


_HELP_RE = re.compile(
    r"^(?:help|\?|what can you do|capabilities|commands)[\s.!?]*$",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"^(?:status|how (?:are|is) (?:we|ace|it) doing|is (?:the )?worker running)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"^(?:(?:what is|what's|whats|tell me about|who is|research|look up)\s+)"
    r"(?:the\s+)?(?:company\s+)?"
    r"['\"]?([A-Za-z0-9][\w .&'-]{1,80})['\"]?"
    r"(?:\s+for\s+(?:my\s+)?(?:job|interview|apply|application|hr))?"
    r"[\s.!?]*$",
    re.IGNORECASE,
)
_HR_PREP_RE = re.compile(
    r"(?:prepar(?:e|ing)|prep)\s+for\s+(?:an?\s+)?(?:hr|technical|hiring)?\s*"
    r"(?:round|interview)?\s*(?:at|for|with)?\s+"
    r"['\"]?([A-Za-z0-9][\w .&'-]{1,80})['\"]?",
    re.IGNORECASE,
)
_KNOW_ABOUT_RE = re.compile(
    r"what\s+should\s+i\s+know\s+about\s+"
    r"['\"]?([A-Za-z0-9][\w .&'-]{1,80})['\"]?",
    re.IGNORECASE,
)
_RESEARCH_BLOCKLIST = frozenset(
    {
        "status",
        "help",
        "you",
        "ace",
        "left",
        "pending",
        "next",
        "this",
        "that",
        "up",
        "wrong",
        "new",
        "my",
        "the",
        "an",
        "a",
    }
)


def _extract_research_company(text: str) -> tuple[str, str] | None:
    """Return (company, focus) for knowledge prefetch, if detectable."""
    raw = (text or "").strip()
    if not raw:
        return None
    for cre, focus in (
        (_HR_PREP_RE, "HR / interview prep — talking points for company questions"),
        (_KNOW_ABOUT_RE, "interview prep"),
        (_RESEARCH_RE, "job-seeker overview"),
    ):
        m = cre.search(raw) if cre is _HR_PREP_RE or cre is _KNOW_ABOUT_RE else cre.match(raw)
        if not m:
            continue
        company = m.group(1).strip(" .,\"'")
        first = company.split()[0].lower() if company else ""
        if company and first not in _RESEARCH_BLOCKLIST and " to apply" not in company.lower():
            if "hr" in raw.lower() or "interview" in raw.lower():
                focus = "HR / interview prep — talking points for company questions"
            return company, focus
    return None


def _format_hr_research_answer(user_text: str, brief: str) -> str:
    """Turn a research brief into a chat-ready HR-prep answer (no LLM required)."""
    hr = bool(re.search(r"\b(hr|interview|prepar)\b", user_text or "", re.I))
    head = (
        "Here’s company research you can use for an HR round "
        "(facts only — don’t invent culture/funding):\n\n"
        if hr
        else "Company research:\n\n"
    )
    return head + (brief or "").strip()


def _synthetic_tool_calls(user_text: str) -> list[dict[str, Any]] | None:
    """Map clear phrases to tools without an LLM round-trip."""
    text = (user_text or "").strip()
    if not text:
        return None
    if _HELP_RE.match(text):
        return [
            {
                "name": "list_copilot_capabilities",
                "args": {},
                "id": "fast_capabilities",
            }
        ]
    if _STATUS_RE.match(text):
        return [
            {
                "name": "get_ace_status",
                "args": {},
                "id": "fast_status",
            }
        ]
    research = _RESEARCH_RE.match(text)
    if research:
        company = research.group(1).strip(" .,\"'")
        first = company.split()[0].lower() if company else ""
        # Avoid "what's left…", "what is next…", etc.
        if company and first not in _RESEARCH_BLOCKLIST and " to apply" not in company.lower():
            return [
                {
                    "name": "research_company",
                    "args": {"company_name": company},
                    "id": "fast_research",
                }
            ]
    bulk = _BULK_THROUGH_RE.search(text)
    if bulk:
        company = bulk.group(1).strip(" .,\"'")
        if company:
            # Preview only — never auto-wipe. User confirms after seeing the list.
            return [
                {
                    "name": "preview_mark_through",
                    "args": {"until_company": company},
                    "id": "fast_preview_through",
                }
            ]
    single = _SINGLE_APPLIED_RE.match(text)
    if single:
        company = single.group(1).strip(" .,\"'")
        # Ignore status-y questions that look like commands.
        if company and company.lower() not in {"status", "help", "what", "how"}:
            return [
                {
                    "name": "mark_application_submitted",
                    "args": {"company_name": company},
                    "id": "fast_mark_one",
                },
                {
                    "name": "cleanup_manual_folder",
                    "args": {"company_name": company},
                    "id": "fast_cleanup_one",
                },
            ]
    return None


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

    def __init__(
        self,
        *,
        session_id: str = "",
        chat_role: Literal["user", "admin"] = "user",
    ) -> None:
        role: ChatRole = "admin" if chat_role == "admin" else "user"
        self.chat_role: ChatRole = role
        self._allowed_tools = tool_names_for_role(role)
        set_chat_role(role)
        self.provider_names, self.llms, self.llms_plain = _bind_chain(role)
        self.tool_map = _tool_map(role)
        self.session_id = session_id or f"sess_{id(self):x}"
        self.messages: list[Any] = [
            SystemMessage(content=build_copilot_system_prompt(role))
        ]
        self._remaining_calls: list[Any] = []
        self._mutation_counts: dict[str, int] = {k: 0 for k in SESSION_MUTATION_BUDGET}
        self._awaiting: PendingToolCall | None = None
        self._rounds_used = 0
        self._tools_ran_this_turn = False
        self._failover_noticed = False
        self._pending_extract_user: str | None = None
        self._last_user_text: str = ""
        self._turn_events: list[CopilotEvent] = []
        self._allow_tools = True
        self.state = EngineState.IDLE
        self._approval_future: Future[bool] | None = None

    def _invoke_ai(self, *, allow_tools: bool | None = None) -> tuple[AIMessage, list[CopilotEvent]]:
        """Invoke across provider chain; skip cooled providers quietly."""
        use_tools = self._allow_tools if allow_tools is None else allow_tools
        chain = self.llms if use_tools else self.llms_plain
        notices: list[CopilotEvent] = []
        errors: list[str] = []
        for idx, llm in enumerate(chain):
            provider = (
                self.provider_names[idx]
                if idx < len(self.provider_names)
                else f"provider-{idx + 1}"
            )
            cool = _cooldown_remaining(provider)
            if cool > 0:
                errors.append(f"{provider}: cooling {cool:.0f}s")
                logger.info(
                    "Copilot skipping cooled provider %s (%.0fs left)",
                    provider,
                    cool,
                )
                continue
            try:
                return llm.invoke(self.messages), notices
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                if _is_token_limit_error(exc):
                    _mark_provider_cooldown(provider, exc)
                if idx < len(chain) - 1:
                    if not self._failover_noticed and (
                        _is_token_limit_error(exc) or _is_failover_error(exc)
                    ):
                        self._failover_noticed = True
                        notices.append(
                            CopilotEvent(
                                kind="info",
                                text=(
                                    f"Ace> Switching chat model ({provider} unavailable)…"
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

        self.state = EngineState.TURN
        self._last_user_text = text
        set_chat_role(self.chat_role)

        # L4: route before RAG / tool-bound LLM (RBAC refuses admin telemetry for users)
        decision = classify_route(text, chat_role=self.chat_role)
        if decision.route == "UNSAFE":
            self._pending_extract_user = None
            self.state = EngineState.IDLE
            return [CopilotEvent(kind="assistant", text=UNSAFE_REFUSAL)]
        if decision.route == "UNAUTHORIZED":
            self._pending_extract_user = None
            self.state = EngineState.IDLE
            return [CopilotEvent(kind="assistant", text=UNAUTHORIZED_REFUSAL)]

        if decision.route == "STATUS":
            return self._finalize_turn(self._short_circuit_status(text))

        # Ambiguous bulk destroy — clarify without LLM / HITL.
        if _AMBIGUOUS_BULK_RE.search(text) and decision.route != "KNOWLEDGE":
            msg = (
                "Ace> That sounds like a bulk clear, but I need a precise target.\n"
                "  • Manual_Action packages → say e.g. "
                "'preview mark through Infosys' then confirm the plan.\n"
                "  • Failed apps list only → ask 'list failures' / 'what's left'.\n"
                "I won't delete or bulk-mark until you preview a cut company."
            )
            self._pending_extract_user = text
            self.messages.append(
                HumanMessage(content=wrap_untrusted("user_chat", text, max_chars=4_000))
            )
            self.messages.append(AIMessage(content=msg))
            self._trim_stm()
            return self._finalize_turn([CopilotEvent(kind="assistant", text=msg)])

        self._remaining_calls = []
        self._rounds_used = 0
        self._tools_ran_this_turn = False
        self._failover_noticed = False
        self._pending_extract_user = text
        self._turn_events = []
        self._allow_tools = decision.route != "KNOWLEDGE"
        self._refresh_retrieved_memory(text)

        # KNOWLEDGE with company research: answer from prefetch (works even if LLM is down).
        if decision.route == "KNOWLEDGE":
            brief = self._prefetch_knowledge(text)
            if brief:
                answer = _format_hr_research_answer(text, brief)
                self.messages.append(
                    HumanMessage(
                        content=wrap_untrusted("user_chat", text, max_chars=4_000)
                    )
                )
                self.messages.append(AIMessage(content=answer))
                self._trim_stm()
                return self._finalize_turn(
                    [CopilotEvent(kind="assistant", text=f"Ace> {answer}")]
                )

        self.messages.append(
            HumanMessage(content=wrap_untrusted("user_chat", text, max_chars=4_000))
        )
        self._trim_stm()

        # Deterministic intent path (ACTION / META shortcuts) — skip on KNOWLEDGE
        if self._allow_tools:
            fast = _synthetic_tool_calls(text)
            if fast:
                # Drop any synthetic calls outside this role's allowlist.
                fast = [c for c in fast if c.get("name") in self._allowed_tools]
                if fast:
                    ai = AIMessage(content="", tool_calls=fast)
                    return self._finalize_turn(self._handle_ai(ai))

        notices: list[CopilotEvent] = []
        try:
            ai, notices = self._invoke_ai()
        except Exception as exc:
            self.messages.pop()
            self._pending_extract_user = None
            return notices + [
                CopilotEvent(
                    kind="assistant",
                    text=_format_model_failure(exc, partial_tools=False),
                )
            ]

        return self._finalize_turn(notices + self._handle_ai(ai))

    def _short_circuit_status(self, text: str) -> list[CopilotEvent]:
        """STATUS route: prefer read tools / desk (no tool-bound LLM)."""
        self._pending_extract_user = text
        self._turn_events = []
        question = ""
        low = text.lower().strip()
        if low not in {"status", "progress"} and not low.startswith("status "):
            question = text
        try:
            body = self._status_body_for(text, question)
        except Exception as exc:
            body = f"Status lookup failed: {exc}"
        msg = f"Ace> {body}"
        self.messages.append(
            HumanMessage(content=wrap_untrusted("user_chat", text, max_chars=4_000))
        )
        self.messages.append(AIMessage(content=body))
        self._trim_stm()
        return [CopilotEvent(kind="assistant", text=msg)]

    def _status_body_for(self, text: str, question: str) -> str:
        """Map STATUS phrasing onto read tools when clear; else desk answer."""
        low = (text or "").lower()
        # Active boards
        if re.search(
            r"(?:which|what|show).{0,40}(?:job\s+)?boards?|"
            r"active\s+(?:job\s+)?boards?|enabled\s+(?:job\s+)?(?:boards?|sites?)",
            low,
        ):
            from src.core.tools import read_active_boards

            return read_active_boards.invoke({})

        # Failure reason for a company
        company = ""
        for cre in (
            re.compile(
                r"why did\s+['\"]?([A-Za-z0-9][\w .&'-]{1,60}?)['\"]?\s+fail",
                re.I,
            ),
            re.compile(
                r"what (?:error|failure).{0,24}(?:for|at|with)\s+"
                r"['\"]?([A-Za-z0-9][\w .&'-]{1,60})['\"]?",
                re.I,
            ),
            re.compile(
                r"failure(?: reason)?(?: for)?\s+"
                r"['\"]?([A-Za-z0-9][\w .&'-]{1,60})['\"]?",
                re.I,
            ),
        ):
            m = cre.search(text or "")
            if m:
                company = m.group(1).strip(" .,\"'?")
                break
        if (
            company
            and company.lower() not in {"status", "it", "this", "that"}
            and "get_application_failure_reason" in self._allowed_tools
        ):
            from src.core.tools import get_application_failure_reason

            return get_application_failure_reason.invoke({"company_name": company})

        # Admin LTM listing
        if self.chat_role == "admin" and re.search(
            r"\b(?:ltm|long[- ]?term\s+memory)\b", low
        ):
            from src.core.tools import list_ltm_episodes

            return list_ltm_episodes.invoke({"limit": 10})

        from src.core.desk import answer, format_status

        return format_status() if not question else answer(question)

    def _prefetch_knowledge(self, text: str) -> str | None:
        """Server-side company research for KNOWLEDGE. Returns brief text or None."""
        self.messages = [
            m
            for m in self.messages
            if not (
                isinstance(m, SystemMessage)
                and _KNOWLEDGE_MARKER in str(getattr(m, "content", ""))
            )
        ]
        hit = _extract_research_company(text)
        if not hit:
            return None
        company, focus = hit
        try:
            from src.core.company_research import research_company_brief

            brief = research_company_brief(company, focus=focus)
        except Exception as exc:
            logger.debug("knowledge prefetch failed: %s", exc)
            return None
        if not (brief or "").strip():
            return None
        block = wrap_data_block("knowledge_prefetch", brief, max_chars=3_500)
        self.messages.insert(
            1,
            SystemMessage(content=f"{_KNOWLEDGE_MARKER}\n{block}"),
        )
        return brief

    def resolve_approval(self, call_id: str, approved: bool) -> list[CopilotEvent]:
        """Unlock a frozen HITL wait (CLI/GUI). Alias of approve()."""
        return self.approve(call_id, approved)

    def approve(self, call_id: str, approved: bool) -> list[CopilotEvent]:
        """Resolve a PendingToolCall, then continue the tool / LLM loop.

        While ``state == AWAITING_APPROVAL`` the engine is frozen: no LLM tokens
        are generated until this method unlocks the turn (zero busy-wait).
        """
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
        self.state = EngineState.TURN
        if self._approval_future is not None and not self._approval_future.done():
            self._approval_future.set_result(bool(approved))
        self._approval_future = None
        if self._remaining_calls:
            self._remaining_calls.pop(0)

        risk = TOOL_RISK.get(pending.tool_name, "mutate_fs")
        try:
            from src.core.audit_log import append_hitl_audit

            append_hitl_audit(
                call_id=pending.call_id,
                tool_name=pending.tool_name,
                args=pending.args,
                user_text=self._last_user_text,
                approved=bool(approved),
                risk=risk,
            )
        except Exception:
            logger.debug("HITL audit failed", exc_info=True)

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
            if risk in self._mutation_counts:
                self._mutation_counts[risk] = self._mutation_counts.get(risk, 0) + 1
            events.extend(self._execute_tool(pending.tool_name, pending.args, pending.call_id))

        return self._finalize_turn(events + self._drain_remaining())

    def _refresh_retrieved_memory(self, user_text: str) -> None:
        """RAG: inject top long-term episodes as DATA (not policy)."""
        self.messages = [
            m
            for m in self.messages
            if not (
                isinstance(m, SystemMessage)
                and _RETRIEVED_MARKER in str(getattr(m, "content", ""))
            )
        ]
        try:
            store = get_memory_store()
            episodes = store.retrieve(user_text, limit=5)
            block = store.format_for_prompt(episodes)
        except Exception:
            logger.debug("LTM retrieve failed", exc_info=True)
            return
        if not block:
            return
        # Keep immutable persona at [0]; retrieved memory next.
        self.messages.insert(
            1,
            SystemMessage(content=f"{_RETRIEVED_MARKER}\n{block}"),
        )

    def _trim_stm(self) -> None:
        if len(self.messages) <= STM_MAX_MESSAGES:
            return
        head = [self.messages[0]]
        for m in self.messages[1:3]:
            if not isinstance(m, SystemMessage):
                break
            content = str(m.content)
            if _RETRIEVED_MARKER in content or _KNOWLEDGE_MARKER in content:
                head.append(m)
            else:
                break
        tail_n = STM_MAX_MESSAGES - len(head)
        self.messages = head + self.messages[-tail_n:]

    def _finalize_turn(self, events: list[CopilotEvent]) -> list[CopilotEvent]:
        """When idle (no HITL pending), schedule background memory extract."""
        self._turn_events.extend(events)
        if self._awaiting is not None:
            self.state = EngineState.AWAITING_APPROVAL
            return events
        user = self._pending_extract_user
        self._pending_extract_user = None
        self.state = EngineState.IDLE
        if not user:
            return events
        parts: list[str] = []
        for ev in self._turn_events:
            if ev.kind in {"assistant", "tool_result"} and ev.text:
                parts.append(ev.text)
        self._turn_events = []
        schedule_memory_extract(
            user,
            "\n".join(parts)[:4_000],
            session_id=self.session_id,
        )
        return events

    def _handle_ai(self, ai: AIMessage) -> list[CopilotEvent]:
        self.messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if tool_calls and not self._allow_tools:
            # KNOWLEDGE route: never execute tools even if the model emits them.
            text = ai.content if isinstance(ai.content, str) else str(ai.content)
            if not (text or "").strip():
                text = (
                    "I answered from research/memory only (tools disabled for this "
                    "knowledge question)."
                )
            return [CopilotEvent(kind="assistant", text=f"Ace> {text}")]
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

            if tool_name not in self.tool_map or tool_name not in self._allowed_tools:
                self._remaining_calls.pop(0)
                deny = (
                    f"UNAUTHORIZED for role={self.chat_role}: tool {tool_name!r} "
                    "is not in this session's capability allowlist."
                    if tool_name in TOOL_RISK
                    else (
                        f"Tool execution failed: unknown tool {tool_name!r}. "
                        "Please try a different approach or tool."
                    )
                )
                events.append(
                    CopilotEvent(
                        kind="tool_error",
                        text=f"[WARNING] {deny}",
                    )
                )
                self.messages.append(
                    ToolMessage(content=deny, tool_call_id=call_id)
                )
                continue

            cleaned, verr = try_validate_tool_args(tool_name, args)
            if verr is not None:
                self._remaining_calls.pop(0)
                self.messages.append(ToolMessage(content=verr, tool_call_id=call_id))
                events.append(
                    CopilotEvent(kind="tool_error", text=f"[TOOL ERROR] {verr}")
                )
                continue
            assert cleaned is not None
            args = cleaned

            risk = TOOL_RISK.get(tool_name, "mutate_fs")
            if risk == "read":
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

            # Internal Critic — block before bothering the user with HITL.
            verdict = critique_tool_call(
                tool_name=tool_name,
                args=args,
                user_text=self._last_user_text,
                risk=risk,
            )
            if not verdict.proceed:
                self._remaining_calls.pop(0)
                msg = verdict.reason or "Critic blocked this tool call."
                self.messages.append(ToolMessage(content=msg, tool_call_id=call_id))
                events.append(
                    CopilotEvent(kind="tool_error", text=f"[critic] {msg}")
                )
                continue

            budget = SESSION_MUTATION_BUDGET.get(risk)
            used = self._mutation_counts.get(risk, 0)
            if budget is not None and used >= budget:
                self._remaining_calls.pop(0)
                msg = (
                    f"Refused: session {risk} budget exhausted "
                    f"({used}/{budget}). Restart chat to reset."
                )
                self.messages.append(ToolMessage(content=msg, tool_call_id=call_id))
                events.append(CopilotEvent(kind="tool_error", text=f"[TOOL ERROR] {msg}"))
                continue

            # Freeze: yield pending_tool and wait for resolve_approval / approve.
            pending = PendingToolCall(
                call_id=call_id,
                tool_name=tool_name,
                args=args,
            )
            self._awaiting = pending
            self.state = EngineState.AWAITING_APPROVAL
            self._approval_future = Future()
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
        cleaned, verr = try_validate_tool_args(tool_name, args)
        if verr is not None:
            self.messages.append(ToolMessage(content=verr, tool_call_id=call_id))
            return [CopilotEvent(kind="tool_error", text=f"[TOOL ERROR] {verr}")]
        assert cleaned is not None
        fn = self.tool_map[tool_name]
        try:
            result = fn.invoke(cleaned)
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

        safe = redact_secrets(str(result))
        self.messages.append(
            ToolMessage(content=safe, tool_call_id=call_id)
        )
        return [
            CopilotEvent(kind="tool_result", text=f"[tool] {safe}")
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


def start_chat_loop(*, chat_role: Literal["user", "admin"] = "user") -> int:
    """CLI adapter over CopilotEngine. Returns process exit code."""
    role: ChatRole = "admin" if chat_role == "admin" else "user"
    print(f"Ace Copilot (role={role}) — type a message (quit / exit to leave).")
    if role == "user":
        print(
            "Status · manuals · research · failure reasons · boards · mark submitted.\n"
            "Lookups auto-run; mutations y/N. Bulk = preview → plan_id.\n"
            "Admin ops (worker/config/LTM/audit) require chat_role=admin.\n"
        )
    else:
        print(
            "Admin session: user tools + worker · config · inbox · brief · LTM.\n"
            "Lookups auto-run; mutations y/N. Bulk = preview → plan_id.\n"
        )
    try:
        engine = CopilotEngine(chat_role=role)
        providers = ", ".join(engine.provider_names) or "none"
        print(f"Chat models: {providers}\n")
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
                    raw = input("Proceed? (y/N): ").strip().lower()
                    ok = raw in {"y", "yes"}  # default DENY
                    queue = engine.resolve_approval(ev.pending.call_id, ok)
                    advanced = True
                    break
            if not advanced:
                break

    return 0
