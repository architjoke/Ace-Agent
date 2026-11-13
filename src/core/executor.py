"""Vision/DOM application executor powered by browser-use + Gemini."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

import yaml
from browser_use import Agent, ChatGoogle
from browser_use.browser.profile import BrowserProfile

from src.core.prompts import (
    BROWSER_SYSTEM_EXTENSION,
    TASK_AUTO,
    TASK_HANDOFF,
    TASK_LINKEDIN_CLEANUP,
    TASK_REVIEW,
)
from src.core.reflect import load_lessons
from src.core.settings import get_settings
from src.core.utils import env_truthy

logger = logging.getLogger(__name__)

from src.core.llm_router import gemini_browser_model_name

MODEL = gemini_browser_model_name()
# Easy Apply defaults (Workday/ATS get a modestly higher budget via URL heuristics).
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("ACE_BROWSER_USE_TIMEOUT_SECONDS", "180"))
DEFAULT_MAX_STEPS = int(os.environ.get("ACE_BROWSER_USE_MAX_STEPS", "10"))
WORKDAY_TIMEOUT_SECONDS = int(os.environ.get("ACE_WORKDAY_TIMEOUT_SECONDS", "300"))
WORKDAY_MAX_STEPS = int(os.environ.get("ACE_WORKDAY_MAX_STEPS", "20"))
# Multi-page LinkedIn Easy Apply (often 5–7 screens) needs a larger CDP budget.
LINKEDIN_MAX_STEPS = int(os.environ.get("ACE_LINKEDIN_MAX_STEPS", "18"))
LINKEDIN_TIMEOUT_SECONDS = int(os.environ.get("ACE_LINKEDIN_TIMEOUT_SECONDS", "240"))
SUBMITTED_TOKEN = "APPLICATION_SUBMITTED"
READY_TOKEN = "READY_FOR_REVIEW"  # legacy / dry-run only
CLEANUP_TOKEN = "LINKEDIN_RESUME_DELETED"

# Serialize Gemini vision applies to protect free-tier RPM (default 1).
_VISION_SEM: asyncio.Semaphore | None = None
_VISION_SEM_LOOP: object | None = None


def _vision_semaphore() -> asyncio.Semaphore:
    """One semaphore per running event loop; size from ACE_VISION_CONCURRENCY (default 1)."""
    global _VISION_SEM, _VISION_SEM_LOOP
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        n = max(1, int(os.environ.get("ACE_VISION_CONCURRENCY", "1") or "1"))
    except ValueError:
        n = 1
    if _VISION_SEM is None or _VISION_SEM_LOOP is not loop:
        _VISION_SEM = asyncio.Semaphore(n)
        _VISION_SEM_LOOP = loop
    return _VISION_SEM


def _is_heavy_ats(job_url: str) -> bool:
    u = (job_url or "").lower()
    return any(
        m in u
        for m in (
            "workday",
            "myworkdayjobs",
            "oraclecloud",
            "successfactors",
            "icims.com",
            "taleo",
            "indeed.com",
            "smartapply.indeed",
        )
    )


def _step_budget(job_url: str) -> tuple[int, int]:
    """Return (max_steps, timeout_seconds) for this apply URL."""
    url = (job_url or "").lower()
    if "linkedin.com" in url:
        return LINKEDIN_MAX_STEPS, LINKEDIN_TIMEOUT_SECONDS
    if _is_heavy_ats(job_url):
        return WORKDAY_MAX_STEPS, WORKDAY_TIMEOUT_SECONDS
    return DEFAULT_MAX_STEPS, DEFAULT_TIMEOUT_SECONDS

# Filename shown on LinkedIn / ATS (no company, no timestamp).
# Override with ACE_APPLY_RESUME_FILENAME=Your_Name_Resume.pdf
APPLY_RESUME_FILENAME = (
    os.environ.get("ACE_APPLY_RESUME_FILENAME") or "Ace_Agent_Resume.pdf"
).strip() or "Ace_Agent_Resume.pdf"

# Start cleanup on Jobs; agent then clicks Preferences → Resumes and applications.
LINKEDIN_RESUMES_URL = "https://www.linkedin.com/jobs/"

# User's permanent LinkedIn resumes — NEVER delete these (substring match).
# Set ACE_LINKEDIN_KEEP_RESUMES=Name1|Name2 (empty default = delete only the agent upload name).
_DEFAULT_KEEP_RESUMES: tuple[str, ...] = ()

# Page/wall markers only — ignore prompt/lesson echoes of the word "captcha".
_BLOCK_MARKERS = (
    "amzn-captcha",
    "amazon captcha",
    "hcaptcha",
    "cloudflare",
    "blocked by bot",
    "access denied",
    "verify you are human",
    "confirm you are human",
    "unusual traffic",
    "let's confirm you are human",
    "solve this puzzle",
    "reached_captcha\": true",
    "reached_captcha': true",
    "reached_captcha: true",
    # Hard site walls — not Gemini capacity.
    "403 forbidden",
    "http error 403",
    "forbidden access",
    "access forbidden",
    "persistent recaptcha",
    "bot verification wall",
    "captcha wall",
)
_LOADING_STUCK_EXEMPT = (
    "preparing review",
    "preparing your",
    "uploading",
    "please wait",
    "loading application",
    "submitting application",
    "almost done",
)
# Soft model/API failures must not become ApplicationBlocked(captcha).
_QUOTA_MARKERS = (
    "resource_exhausted",
    "429",
    "503 unavailable",
    "high demand",
    "rate limit",
    "quota",
)

class ApplicationBlockedError(RuntimeError):
    """Raised when the browser agent is permanently blocked (CAPTCHA / hard wall)."""


class ApplicationTimeoutError(TimeoutError):
    """Raised when the browser agent exceeds the allowed wall-clock timeout."""


class AlreadyAppliedError(RuntimeError):
    """Raised when DB or DOM indicates this job was already applied to."""


class JobClosedError(RuntimeError):
    """Raised when the job posting is closed / missing (does not exist)."""


class RequiresHumanInterventionError(RuntimeError):
    """Playbook + CDP handoff failed — soft-fail without Manual_Action package."""


class TokenLimitReachedError(RuntimeError):
    """LLM / vision API quota exhausted — pause applies until quotas reset."""


class WarmBrowserUnavailable(RuntimeError):
    """Warm Chromium ensure failed — do not attach vision to a stale CDP port."""


class ExternalPortalLoadError(RuntimeError):
    """Company / external ATS page failed to load (DNS, blank, connection).

    Distinct from Gemini vision quota — do not route these into CDP fallback.
    """


_PORTAL_LOAD_MARKERS = (
    "err_name_not_resolved",
    "err_connection_refused",
    "err_connection_timed_out",
    "err_address_unreachable",
    "err_internet_disconnected",
    "err_connection_reset",
    "err_timed_out",
    "err_failed",
    "net::err_",
    "name_not_resolved",
    "dns_probe",
    "this site can't be reached",
    "this site can’t be reached",
    "failed to load",
    "external application portal failed",
    "external application page failed",
    "external company application site that failed",
    "blank page",
    "dns/",
    "failed_external_load",
    "failed_portal_dns",
)


def is_portal_load_failure(exc_or_text: BaseException | str) -> bool:
    """True when the failure is a dead external portal, not vision capacity."""
    blob = (
        str(exc_or_text)
        if not isinstance(exc_or_text, BaseException)
        else f"{type(exc_or_text).__name__}: {exc_or_text}"
    ).lower()
    return any(m in blob for m in _PORTAL_LOAD_MARKERS)


def _is_cdp_attach_error(exc: BaseException) -> bool:
    """True when browser-use failed to attach to a (usually dead) CDP endpoint."""
    name = type(exc).__name__
    if name in {"ConnectError", "ConnectTimeout", "ReadTimeout"}:
        return True
    msg = str(exc).lower()
    if "root cdp client not initialized" in msg:
        return True
    if "all connection attempts failed" in msg:
        return True
    if "cdp" in msg and ("connect" in msg or "not initialized" in msg):
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return _is_cdp_attach_error(cause)
    return False


def _auto_submit_enabled() -> bool:
    return env_truthy("ACE_AUTO_SUBMIT", "1")


def _keep_resume_names() -> list[str]:
    raw = (os.environ.get("ACE_LINKEDIN_KEEP_RESUMES") or "").strip()
    if raw:
        return [x.strip() for x in raw.split("|") if x.strip()]
    return list(_DEFAULT_KEEP_RESUMES)


def _application_answers_block(
    master_path: Path,
    *,
    jd_text: str | None = None,
    yaml_snapshot: str | None = None,
) -> str:
    """Extract application_answers from master.yaml for the browser prompt."""
    try:
        data = yaml.safe_load(master_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not parse application_answers from %s: %s", master_path, exc)
        return ""
    if not isinstance(data, dict):
        return ""
    answers = data.get("application_answers")
    if not isinstance(answers, dict) or not answers:
        return ""
    from src.core.form_answers import answers_prompt_block
    from src.core.salary import inject_salary_answers

    answers = inject_salary_answers(
        answers, jd_text=jd_text, yaml_snapshot=yaml_snapshot
    )
    try:
        from src.core.learned_answers import merge_into_answers

        answers = merge_into_answers(answers)
    except Exception:
        pass
    return answers_prompt_block(answers)


def unique_apply_pdf(pdf_path: Path | str, company: str = "") -> Path:
    """Copy tailored PDF to the configured upload name (ACE_APPLY_RESUME_FILENAME).

    Company/time are tracked in SQLite / output folders — never in the filename
    LinkedIn or ATS sees. Freshness comes from always clicking Upload + upload_file.
    """
    del company  # reserved for callers; filename stays constant
    src = Path(pdf_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Tailored PDF not found: {src}")
    dest = src.parent / APPLY_RESUME_FILENAME
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    else:
        # Already the right name; touch content by rewrite so file is current
        shutil.copy2(src, dest)
    logger.info("Apply PDF ready as %s", dest.name)
    return dest


def _api_key() -> str:
    key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY."
        )
    return key


def _build_llm(api_key: str) -> ChatGoogle:
    return ChatGoogle(model=MODEL, api_key=api_key, temperature=0)


def _browser_profile(user_data_dir: Path | str | None = None) -> BrowserProfile:
    settings = get_settings()
    profile_dir = Path(user_data_dir) if user_data_dir else Path(settings.browser_user_data_dir)
    profile_dir = profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        lock = profile_dir / name
        try:
            lock.unlink(missing_ok=True)
        except TypeError:
            if lock.exists():
                try:
                    lock.unlink()
                except OSError:
                    pass
        except OSError:
            pass
    # keep_alive default ON — warm reuse; set ACE_BROWSER_KEEP_ALIVE=0 to tear down.
    # Google/LinkedIn cookies still persist via user_data_dir on disk.
    keep = env_truthy("ACE_BROWSER_KEEP_ALIVE", "1")
    from src.core.browser_identity import (
        pinned_device_scale_factor,
        pinned_locale,
        pinned_timezone_id,
        pinned_user_agent,
        pinned_viewport,
    )
    from src.core.network import browser_proxy_settings
    from src.playbooks._common import CHROMIUM_STEALTH_ARGS

    proxy = browser_proxy_settings()
    if proxy is not None:
        logger.info("BrowserProfile sticky proxy enabled (PROXY_URL / network.proxies)")
    vp = pinned_viewport()
    profile_kwargs: dict = {
        "user_data_dir": str(profile_dir),
        "profile_directory": settings.browser_profile_directory,
        "headless": settings.browser_headless,
        "keep_alive": keep,
        "proxy": proxy,
        "args": list(CHROMIUM_STEALTH_ARGS),
        "user_agent": pinned_user_agent(user_data_dir=profile_dir),
        "window_size": vp,
        "device_scale_factor": pinned_device_scale_factor(),
        # Leave allowed_domains unset so accounts.google.com OAuth popups work.
        "cross_origin_iframes": True,
        # Zero-Trust: block downloads + strip media/geo permissions.
        "accept_downloads": False,
        "permissions": [],
        "auto_download_pdfs": False,
        # Fluidity: take control on DOM-ready; do not wait on tracking pixels.
        "minimum_wait_page_load_time": 0.1,
        "wait_for_network_idle_page_load_time": 0.15,
        "wait_between_actions": 0.05,
    }
    # Ban armor: pin locale/TZ when BrowserProfile accepts them.
    try:
        return BrowserProfile(
            **profile_kwargs,
            locale=pinned_locale(),
            timezone_id=pinned_timezone_id(),
        )
    except TypeError:
        return BrowserProfile(**profile_kwargs)


def _browser_profile_cdp(cdp_url: str) -> BrowserProfile:
    """Attach browser-use to an already-running playbook Chromium via CDP."""
    from src.core.browser_identity import pinned_device_scale_factor, pinned_viewport

    vp = pinned_viewport()
    return BrowserProfile(
        cdp_url=cdp_url,
        is_local=True,
        keep_alive=True,
        headless=get_settings().browser_headless,
        window_size=vp,
        device_scale_factor=pinned_device_scale_factor(),
        cross_origin_iframes=True,
        accept_downloads=False,
        permissions=[],
        auto_download_pdfs=False,
        minimum_wait_page_load_time=0.1,
        wait_for_network_idle_page_load_time=0.15,
        wait_between_actions=0.05,
    )


def _history_blob(history) -> str:
    parts: list[str] = []
    final = history.final_result() if history is not None else None
    if final:
        parts.append(str(final))
    if history is not None:
        for err in history.errors():
            if err:
                parts.append(str(err))
        for attr in (
            "model_thoughts",
            "model_outputs",
            "action_results",
            "extracted_content",
        ):
            getter = getattr(history, attr, None)
            if callable(getter):
                try:
                    val = getter()
                    if val:
                        parts.append(str(val))
                except Exception:
                    pass
        judgement = history.judgement() if hasattr(history, "judgement") else None
        if judgement:
            parts.append(str(judgement))
    return "\n".join(parts)


def _raise_if_blocked(history) -> None:
    blob = _history_blob(history).lower()
    # Strip false negatives and instructional CAPTCHA wording from prompts/lessons.
    cleaned = (
        blob.replace('"reached_captcha": false', " ")
        .replace("'reached_captcha': false", " ")
        .replace("reached_captcha: false", " ")
        .replace('"reached_captcha":false', " ")
    )
    for noise in (
        "on captcha / cloudflare / hard bot walls",
        "on captcha / cloudflare / bot walls",
        "abort immediately and leave manual_action",
        "captchas are auto-solved",
        "treat as blocked",
    ):
        cleaned = cleaned.replace(noise, " ")
    # Site walls first — never mislabel 403/captcha as Gemini capacity.
    for marker in _BLOCK_MARKERS:
        if marker in cleaned:
            raise ApplicationBlockedError(
                f"Browser agent permanently blocked ({marker!r}). Detail: {blob[:1500]}"
            )
    # Bare "recaptcha" / "captcha" only if paired with wall/page language (not lessons).
    if ("recaptcha" in cleaned or " captcha" in cleaned or cleaned.startswith("captcha")) and any(
        w in cleaned
        for w in (
            "security check",
            "bot detection",
            "image selection",
            "puzzle",
            "human",
            "wall",
            "verification",
            "blocked",
        )
    ):
        raise ApplicationBlockedError(
            f"Browser agent permanently blocked ('captcha'). Detail: {blob[:1500]}"
        )
    # Dead company portals / DNS — before quota language (agents often mention both).
    if is_portal_load_failure(cleaned) or is_portal_load_failure(blob):
        raise ExternalPortalLoadError(
            f"FAILED_EXTERNAL_LOAD: company portal failed to load. Detail: {blob[:800]}"
        )
    # Gemini quota / capacity → timeout-style retry path, not captcha block.
    if any(m in cleaned for m in _QUOTA_MARKERS):
        raise ApplicationTimeoutError(
            f"Vision model quota/capacity exhausted mid-apply. Detail: {blob[:800]}"
        )


def _resume_upload_verified(blob: str, pdf_stem: str) -> bool:
    """Require a fresh disk upload — not clicking a previously listed LinkedIn resume."""
    lower = blob.lower()
    stem = (pdf_stem or APPLY_RESUME_FILENAME.rsplit(".", 1)[0]).lower()
    name_ok = stem in lower or APPLY_RESUME_FILENAME.lower().replace(".pdf", "") in lower
    if not name_ok:
        return False
    uploaded = "successfully uploaded file" in lower or (
        "uploaded file" in lower and stem in lower
    )
    # Reject library reuse without a new upload_file this run
    reused = (
        "already selected" in lower
        or "green indicator" in lower
        or "previously uploaded" in lower
        or "selected an existing" in lower
    ) and "successfully uploaded file" not in lower
    if reused:
        logger.warning(
            "Resume looks reused from LinkedIn library without fresh upload (stem=%s)",
            pdf_stem,
        )
        return False
    return bool(uploaded)


def _submit_confirmed(blob: str) -> bool:
    lower = blob.lower()
    markers = (
        "application submitted",
        "application was sent",
        "application has been submitted",
        "successfully submitted",
        "thank you for applying",
        "thanks for applying",
        "we received your application",
        "your application was sent",
        "submission confirmed",
    )
    return any(m in lower for m in markers)


def _verified_auto_submit(history, pdf_stem: str) -> bool:
    if history is None:
        return False
    blob = _history_blob(history)
    token_ok = SUBMITTED_TOKEN in blob or (
        history.is_done() and _submit_confirmed(blob)
    )
    if not token_ok:
        return False
    if not _resume_upload_verified(blob, pdf_stem):
        logger.warning(
            "Rejecting false SUCCESS: fresh resume upload not verified (stem=%s)",
            pdf_stem,
        )
        return False
    if not _submit_confirmed(blob) and SUBMITTED_TOKEN not in blob:
        return False
    return True


def _reached_token(history, token: str, *, pdf_stem: str = "") -> bool:
    if history is None:
        return False
    if token == SUBMITTED_TOKEN:
        return _verified_auto_submit(history, pdf_stem)
    if token == CLEANUP_TOKEN:
        # Only the agent's final_result — task text also contains the token string.
        try:
            final = str(history.final_result() or "")
        except Exception:
            final = ""
        if CLEANUP_TOKEN not in final:
            return False
        low = final.lower()
        # Reject "could not delete" / failure narrations that echo the token.
        if any(
            n in low
            for n in (
                "could not",
                "wasn't deleted",
                "was not deleted",
                "failed",
                "not found/deleted",
                "unable to",
            )
        ):
            return False
        return True
    blob = _history_blob(history)
    if token in blob:
        if pdf_stem and not _resume_upload_verified(blob, pdf_stem):
            return False
        return True
    lower = blob.lower()
    markers = ("ready_for_review", "awaiting human", "form complete", "paused")
    return history.is_done() and any(m in lower for m in markers)


STUCK_OBSERVATION = (
    "[SYSTEM] Your last actions had no effect. A modal may be blocking the screen, "
    "or the element is unclickable. Re-evaluate the screen and attempt an alternate "
    "route, or look for a Close/X button."
)


class _StuckStateTracker:
    """Detect unchanged URL+DOM fingerprints across consecutive agent steps."""

    def __init__(self, agent_ref: dict) -> None:
        self._agent_ref = agent_ref
        self._last_fp: str | None = None
        self._same_streak = 0
        self._stuck_events = 0

    def _loading_exempt(self, browser_state) -> bool:
        """True when Soft Apply / upload spinners should not trip stuck detection."""
        chunks: list[str] = []
        for attr in ("url", "current_page_url", "page_url", "title"):
            val = getattr(browser_state, attr, None)
            if val:
                chunks.append(str(val))
        for attr in ("page_content", "content", "extracted_content", "text"):
            val = getattr(browser_state, attr, None)
            if val:
                chunks.append(str(val)[:1200])
                break
        blob = " ".join(chunks).lower()
        return any(m in blob for m in _LOADING_STUCK_EXEMPT)

    def _fingerprint(self, browser_state) -> str:
        url = ""
        for attr in ("url", "current_page_url", "page_url"):
            val = getattr(browser_state, attr, None)
            if val:
                url = str(val)
                break
        if not url:
            try:
                url = str(getattr(browser_state, "tabs", "") or "")[:200]
            except Exception:
                url = ""
        n_els = 0
        for attr in ("selector_map", "interacted_element", "elements", "dom_items"):
            val = getattr(browser_state, attr, None)
            if val is None:
                continue
            try:
                n_els = len(val)
                break
            except TypeError:
                continue
        # Fallback: hash a short string representation of interactive summary
        extra = ""
        summary = getattr(browser_state, "element_tree", None) or getattr(
            browser_state, "dom_state", None
        )
        if summary is not None:
            extra = str(type(summary).__name__) + str(hash(str(summary)[:800]) % 10_000_000)
        return f"{url}|{n_els}|{extra}"

    def _inject_observation(self, agent: Agent | None) -> bool:
        if agent is None:
            return False
        # Try common browser-use injection points
        for target_name in ("message_manager", "state"):
            target = getattr(agent, target_name, None)
            if target is None:
                continue
            for meth_name in (
                "add_message",
                "add_state_message",
                "add_new_task",
                "add_context",
            ):
                meth = getattr(target, meth_name, None)
                if not callable(meth):
                    continue
                try:
                    meth(STUCK_OBSERVATION)
                    return True
                except TypeError:
                    try:
                        meth(role="system", content=STUCK_OBSERVATION)
                        return True
                    except Exception:
                        continue
                except Exception:
                    continue
        # Last resort: append to task string for next reasoning turn visibility
        try:
            if hasattr(agent, "task") and isinstance(agent.task, str):
                if STUCK_OBSERVATION not in agent.task:
                    agent.task = f"{agent.task}\n\n{STUCK_OBSERVATION}"
                    return True
        except Exception:
            pass
        return False

    def __call__(self, browser_state, model_output, step_number: int) -> None:
        if self._loading_exempt(browser_state):
            # Soft Apply "Preparing review" / upload spinners — reset streak, no abort.
            self._same_streak = 0
            self._last_fp = self._fingerprint(browser_state)
            return

        fp = self._fingerprint(browser_state)
        if self._last_fp is not None and fp == self._last_fp:
            self._same_streak += 1
        else:
            self._same_streak = 0
        self._last_fp = fp

        # Unchanged for 2 consecutive steps after an action → stuck
        if self._same_streak < 2:
            return

        self._stuck_events += 1
        agent = self._agent_ref.get("agent")
        injected = self._inject_observation(agent)
        logger.warning(
            "Stuck-state detected (streak=%s events=%s step=%s injected=%s fp=%s)",
            self._same_streak,
            self._stuck_events,
            step_number,
            injected,
            fp[:160],
        )
        # After 2 stuck detections (≈4 identical fingerprints), fail fast
        if self._stuck_events >= 2:
            raise ApplicationTimeoutError(
                "stuck_state: URL/DOM fingerprint unchanged across repeated steps"
            )


async def _detach_agent(agent: Agent | None) -> None:
    """Stop the browser-use agent client without assuming Chromium is gone."""
    if agent is None:
        return
    for attr in ("close", "stop"):
        fn = getattr(agent, attr, None)
        if callable(fn):
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.debug("agent.%s detach failed", attr, exc_info=True)


async def _force_close_agent(agent: Agent | None, user_data_dir: Path | str | None) -> None:
    """Best-effort Chromium teardown so the next job starts clean."""
    await _detach_agent(agent)
    if agent is not None:
        browser = getattr(agent, "browser", None) or getattr(agent, "browser_session", None)
        if browser is not None:
            for attr in ("close", "kill", "stop"):
                fn = getattr(browser, attr, None)
                if callable(fn):
                    try:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.debug("browser.%s teardown failed", attr, exc_info=True)
    # Clear Chromium singleton locks so the next launch is not blocked.
    try:
        profile = Path(user_data_dir) if user_data_dir else Path(get_settings().browser_user_data_dir)
        profile = profile.expanduser().resolve()
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            lock = profile / name
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        logger.debug("profile lock clear failed", exc_info=True)


def _is_gemini_capacity_error(exc: BaseException) -> bool:
    """True for free-tier Gemini 503/429 / ServiceUnavailable / TooManyRequests."""
    # Never treat dead company portals / DNS as vision capacity.
    if isinstance(exc, ExternalPortalLoadError) or is_portal_load_failure(exc):
        return False
    try:
        from google.api_core import exceptions as gexc

        if isinstance(
            exc,
            (
                gexc.ServiceUnavailable,
                gexc.TooManyRequests,
                gexc.ResourceExhausted,
            ),
        ):
            return True
    except Exception:
        pass
    name = type(exc).__name__
    if name in {"ServiceUnavailable", "TooManyRequests", "ResourceExhausted"}:
        return True
    blob = f"{name}: {exc}".lower()
    if any(m in blob for m in _QUOTA_MARKERS):
        return True
    return "service unavailable" in blob or "too many requests" in blob


def _build_browser_agent(
    task: str,
    file_paths: list[str],
    *,
    profile: BrowserProfile,
    stuck: "_StuckStateTracker",
) -> Agent:
    return Agent(
        task=task,
        llm=_build_llm(_api_key()),
        browser_profile=profile,
        use_vision=True,
        available_file_paths=file_paths,
        max_failures=8,
        max_actions_per_step=12,
        extend_system_message=BROWSER_SYSTEM_EXTENSION,
        register_new_step_callback=stuck,
    )


async def _run_agent(
    task: str,
    file_paths: list[str],
    *,
    max_steps: int,
    timeout_seconds: int | None = None,
    user_data_dir: Path | str | None = None,
    browser_profile: BrowserProfile | None = None,
    teardown: bool = True,
) -> object:
    agent_ref: dict = {"agent": None}
    stuck = _StuckStateTracker(agent_ref)
    profile = browser_profile or _browser_profile(user_data_dir)
    agent: Agent | None = _build_browser_agent(
        task, file_paths, profile=profile, stuck=stuck
    )
    agent_ref["agent"] = agent
    wall = int(timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS)
    sem = _vision_semaphore()
    await sem.acquire()
    last_capacity: BaseException | None = None
    try:
        # Free-tier Gemini 503/429: 3 attempts with exponential backoff, then yield.
        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    agent.run(max_steps=max_steps),
                    timeout=wall,
                )
            except ApplicationTimeoutError:
                raise
            except TimeoutError as exc:
                # asyncio.wait_for wall clock — not a capacity retry.
                raise ApplicationTimeoutError(
                    f"browser-use agent timed out after {wall}s"
                ) from exc
            except Exception as exc:
                if not _is_gemini_capacity_error(exc):
                    raise
                last_capacity = exc
                delay = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "Gemini capacity error (%s) attempt %s/3 — backoff %ss",
                    type(exc).__name__,
                    attempt + 1,
                    delay,
                )
                if attempt >= 2:
                    break
                await asyncio.sleep(delay)
                # Detach failed client; rebuild agent for a clean retry.
                await _detach_agent(agent)
                agent = _build_browser_agent(
                    task, file_paths, profile=profile, stuck=stuck
                )
                agent_ref["agent"] = agent
        # Exhausted retries — pause applies; do not keep burning the batch.
        detail = f"{type(last_capacity).__name__}: {last_capacity}" if last_capacity else "unknown"
        from src.core.llm_router import mark_token_limit_reached

        # Vision capacity burn — stop browser applies (not just hunt/tailor).
        mark_token_limit_reached(detail[:400], blocks_apply=True)
        raise TokenLimitReachedError(
            f"TOKEN LIMIT REACHED — pausing applies until quotas reset "
            f"(Gemini capacity exhausted after 3 attempts: {detail[:400]})"
        ) from last_capacity
    finally:
        try:
            sem.release()
        except Exception:
            pass
        if teardown:
            await _force_close_agent(agent, user_data_dir)
        else:
            # CDP handoff/assist: detach agent client; PlaybookSession still owns Chromium.
            await _detach_agent(agent)


async def _linkedin_cleanup_safe(
    pdf_path: Path | str,
    *,
    job_url: str,
    user_data_dir: Path | str | None = None,
    reason: str = "",
) -> None:
    """Best-effort delete of ACE_APPLY_RESUME_FILENAME after a LinkedIn apply."""
    if "linkedin.com" not in (job_url or "").lower():
        return
    try:
        ok = await cleanup_linkedin_uploaded_resume(
            pdf_path, user_data_dir=user_data_dir
        )
        logger.info(
            "LinkedIn resume cleanup %s (%s)",
            "ok" if ok else "unconfirmed",
            reason or "post-apply",
        )
    except Exception:
        logger.exception(
            "LinkedIn resume cleanup failed (%s)", reason or "post-apply"
        )


async def cleanup_linkedin_uploaded_resume(
    pdf_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
) -> bool:
    """Delete agent upload via Jobs → Preferences → Resumes and applications."""
    if pdf_path:
        Path(pdf_path).expanduser().resolve()
    keep = _keep_resume_names()
    keep_list = "\n".join(f"  - {name}" for name in keep)
    filename = APPLY_RESUME_FILENAME
    task = TASK_LINKEDIN_CLEANUP.format(
        resumes_url=LINKEDIN_RESUMES_URL,
        pdf_filename=filename,
        keep_list=keep_list,
        cleanup_token=CLEANUP_TOKEN,
    )
    logger.info(
        "LinkedIn resume cleanup starting for %s (protecting %s permanent resumes)",
        filename,
        len(keep),
    )
    try:
        history = await _run_agent(
            task,
            [],
            max_steps=24,
            timeout_seconds=300,
            user_data_dir=user_data_dir,
        )
    except TimeoutError:
        logger.warning("LinkedIn resume cleanup timed out for %s", filename)
        return False
    except Exception:
        logger.exception("LinkedIn resume cleanup crashed for %s", filename)
        return False

    ok = _reached_token(history, CLEANUP_TOKEN, pdf_stem=Path(filename).stem)
    if ok:
        logger.info("LinkedIn resume deleted (or already gone): %s", filename)
    else:
        logger.warning(
            "Could not confirm LinkedIn resume deletion for %s — check Jobs→Preferences→Resumes",
            filename,
        )
    return ok


def _loop_usable(loop) -> bool:
    """True if ``loop`` can accept run_coroutine_threadsafe from a sync thread."""
    if loop is None:
        return False
    try:
        if getattr(loop, "is_closed", lambda: False)():
            return False
        return bool(loop.is_running())
    except Exception:
        return False


def _try_click_submit_application(page) -> bool:
    """Click LinkedIn Easy Apply Submit (or Review→Submit). Returns True on success UI."""
    if page is None:
        return False
    import re

    from src.playbooks._common import success_text

    submit_re = re.compile(r"^Submit( application)?$", re.I)
    review_re = re.compile(r"^Review$", re.I)
    for pattern in (submit_re, review_re):
        try:
            btn = page.get_by_role("button", name=pattern).first
            if btn.count() == 0 or not btn.is_visible(timeout=800):
                continue
            try:
                btn.click(timeout=5000)
            except Exception:
                btn.click(timeout=5000, force=True)
            page.wait_for_timeout(450)
            if pattern is review_re:
                # After Review, Submit should appear — click it too.
                try:
                    sub = page.get_by_role("button", name=submit_re).first
                    if sub.count() and sub.is_visible(timeout=600):
                        try:
                            sub.click(timeout=3000)
                        except Exception:
                            sub.click(timeout=3000, force=True)
                        page.wait_for_timeout(450)
                except Exception:
                    pass
            break
        except Exception:
            continue
    try:
        body = page.inner_text("body", timeout=900) or ""
    except Exception:
        body = ""
    lower = body.lower()
    if success_text(body):
        return True
    return any(
        m in lower
        for m in (
            "application sent",
            "your application was sent",
            "application submitted",
        )
    )


async def _try_finish_easy_apply_submit(
    user_data_dir: Path | str | None,
    *,
    job_url: str,
    pdf_path: Path | str | None = None,
) -> bool:
    """Last-chance Submit on warm LinkedIn modal; else human popup + learn."""
    if not user_data_dir or "linkedin.com" not in (job_url or "").lower():
        return False

    def _sync() -> bool:
        try:
            from src.core.browser_warmup import get_or_create_slot
            from src.core.human_assist import (
                form_looks_submitted,
                resolve_unknown_with_human,
            )

            slot = get_or_create_slot(user_data_dir)
            page = getattr(slot, "page", None)
            if page is None:
                return False
            if _try_click_submit_application(page):
                return True
            if form_looks_submitted(page):
                return True
            helped = resolve_unknown_with_human(
                page,
                company="",
                job_url=job_url,
                reason="Reached Review/final page — click Submit application, then OK",
            )
            if not helped:
                return False
            if form_looks_submitted(page):
                return True
            return _try_click_submit_application(page)
        except Exception:
            logger.debug("easy-apply submit salvage failed", exc_info=True)
            return False

    return bool(await asyncio.to_thread(_sync))


def make_cdp_assist(loop, session_factory, *, job_url: str = ""):
    """Sync bridge: playbook thread → asyncio CDP assist on the worker's loop.

    Playwright runs in ``asyncio.to_thread``; Gemini assist must be scheduled onto
    the *same* worker event loop captured before the thread started — never
    ``asyncio.run`` / a new loop inside the sync playbook.
    """
    url = (job_url or "").lower()
    # LinkedIn Easy Apply gets a longer assist budget (dialog intercept / multi-page).
    assist_steps = 6 if "linkedin.com" in url else 2
    assist_timeout = 120.0 if "linkedin.com" in url else 60.0
    fut_wait = 150 if "linkedin.com" in url else 90

    def cdp_assist(reason: str) -> bool:
        if not _loop_usable(loop):
            logger.debug("cdp_assist skipped — worker loop not usable")
            return False
        session = session_factory()
        try:
            fut = asyncio.run_coroutine_threadsafe(
                _execute_cdp_assist(
                    session,
                    reason,
                    max_steps=assist_steps,
                    timeout_sec=assist_timeout,
                ),
                loop,
            )
        except Exception:
            logger.debug("cdp_assist schedule failed", exc_info=True)
            return False
        try:
            return bool(fut.result(timeout=fut_wait))
        except TimeoutError:
            fut.cancel()
            logger.debug("cdp_assist timed out — cancelled")
            return False
        except Exception:
            logger.debug("cdp_assist future failed", exc_info=True)
            return False

    return cdp_assist


def _pin_running_loop(expected=None):
    """Bind CDP work to the active worker asyncio loop (never spawn a new one)."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RequiresHumanInterventionError(
            "CDP requires active worker event loop"
        ) from exc
    if expected is not None and running is not expected:
        raise RequiresHumanInterventionError(
            "CDP loop mismatch: running loop is not the worker loop"
        )
    return running


_ATS_KINDS = frozenset(
    {
        "greenhouse",
        "lever",
        "ashby",
        "workday",
        "linkedin",
        "indeed",
        "naukri",
        "wellfound",
        "yc_waas",
        "glassdoor",
        "aggregator",
        "google_forms",
        "career_site",
    }
)

# PlaybookHandoff reasons that must NOT burn Gemini CDP (dead ends).
_NO_CDP_HANDOFF_MARKERS = (
    "easy apply not clickable",
    "skip_no_easy_apply",
    "skip_budget_no_modal",
    "skip_blocked_board",
    "unsupported company ats host",
    "failed_external_load",
    "failed_portal_dns",
)


def _handoff_should_skip_cdp(reason: str) -> bool:
    blob = (reason or "").lower()
    return any(m in blob for m in _NO_CDP_HANDOFF_MARKERS)


def _playbook_runner(kind: str):
    if kind == "greenhouse":
        from src.playbooks.greenhouse import run_greenhouse as run
    elif kind in {"lever", "ashby"}:
        from src.playbooks.lever import run_lever as run
    elif kind == "workday":
        from src.playbooks.workday import run_workday as run
    elif kind == "linkedin":
        from src.playbooks.linkedin import run_linkedin as run
    elif kind == "indeed":
        from src.playbooks.indeed import run_indeed as run
    elif kind == "naukri":
        from src.playbooks.naukri import run_naukri as run
    elif kind == "wellfound":
        from src.playbooks.wellfound import run_wellfound as run
    elif kind == "yc_waas":
        from src.playbooks.yc_waas import run_yc_waas as run
    elif kind == "glassdoor":
        from src.playbooks.glassdoor import run_glassdoor as run
    elif kind == "aggregator":
        from src.playbooks.aggregator import run_aggregator as run
    elif kind == "google_forms":
        from src.playbooks.google_forms import run_google_forms as run
    elif kind == "career_site":
        from src.playbooks.career_site import run_career_site as run
    else:
        return None
    return run


async def _warm_ats_apply(
    job_url: str,
    pdf_path: Path,
    master_path: Path,
    *,
    user_data_dir: Path | str,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
    company: str = "",
    job_title: str = "",
) -> bool:
    """One owner thread: ensure slot → probe → resolve ATS → playbook (greenlet-safe)."""
    from src.core.browser_warmup import get_or_create_slot
    from src.core.telemetry import time_stage
    from src.playbooks.session import PlaybookHandoff

    try:
        assist_loop = asyncio.get_running_loop()
    except RuntimeError:
        assist_loop = None

    slot = get_or_create_slot(user_data_dir)

    def _sync() -> bool:
        from src.core.browser_warmup import (
            browser_alive,
            get_or_create_slot as _get_slot,
            invalidate_slot,
        )
        from src.core.job_closed import probe_on_page
        from src.playbooks import PlaybookFallback, resolve_ats_kind
        from src.playbooks.session import PlaybookSession

        from src.core.semantic_dom import try_semantic_apply

        keep = env_truthy("ACE_BROWSER_KEEP_ALIVE", "1")
        # Re-resolve inside owner thread so invalidate+recreate stays local.
        local_slot = _get_slot(user_data_dir)
        handoff_pending = False
        leave_for_vision = False
        try:
            try:
                ctx, page = local_slot.ensure()
            except Exception:
                # Stay on this owner thread — do not invalidate/shutdown the
                # executor mid-flight (would spawn a new thread → greenlet miss).
                logger.warning(
                    "warm slot ensure failed — relaunching Chromium on owner thread",
                    exc_info=True,
                )
                try:
                    local_slot._close_browser_only()
                    ctx, page = local_slot.ensure()
                except Exception as exc:
                    from src.playbooks.session import clear_stale_devtools_port

                    clear_stale_devtools_port(user_data_dir)
                    raise WarmBrowserUnavailable(
                        f"Chromium ensure failed for {str(user_data_dir)[:120]}: {exc}"
                    ) from exc

            probe = probe_on_page(page, job_url)
            if probe == "closed":
                raise JobClosedError(f"Job closed or page does not exist at {job_url[:160]}")
            if probe == "already_applied":
                raise AlreadyAppliedError(f"DOM already-applied signal at {job_url[:160]}")

            # Early hard-wall (iCIMS Amazon captcha etc.) — ask human before skip.
            try:
                body = (page.inner_text("body", timeout=1200) or "").lower()
                title = (page.title() or "").lower()
            except Exception:
                body, title = "", ""
            wall_blob = f"{title}\n{body}"
            if any(
                n in wall_blob
                for n in (
                    "amzn-captcha",
                    "amazon captcha",
                    "let's confirm you are human",
                    "solve this puzzle",
                )
            ):
                from src.core.human_assist import (
                    resolve_captcha_with_human,
                    shot_on_fail,
                )

                if not resolve_captcha_with_human(
                    page, company=company or "", job_url=job_url
                ):
                    shot_on_fail(
                        page,
                        company=company or "",
                        reason="hard_captcha_wall",
                        job_url=job_url,
                    )
                    raise ApplicationBlockedError(
                        "Hard bot/captcha wall — human did not clear; skipping"
                    )

            kind = resolve_ats_kind(job_url, page=page, user_data_dir=user_data_dir)

            def _semantic() -> bool:
                return try_semantic_apply(
                    job_url, pdf_path, master_path, user_data_dir=user_data_dir, page=page
                )

            if kind not in _ATS_KINDS:
                if _semantic():
                    return True
                leave_for_vision = keep
                return False

            run = _playbook_runner(kind)
            assert run is not None
            logger.info("ATS router → %s playbook for %s (warm)", kind, job_url)
            local: dict = {
                "user_data_dir": user_data_dir,
                "yaml_snapshot": yaml_snapshot,
                "context": ctx,
                "page": page,
                "jd_text": jd_text or "",
                "company": company or "",
                "job_title": job_title or "",
            }
            if kind in {"lever", "ashby"}:
                local["ats"] = kind
            if assist_loop is not None:

                def _session_factory():
                    return PlaybookSession.from_live(
                        local_slot.context,
                        local_slot.page,
                        user_data_dir,
                        playwright=local_slot.playwright,
                    )

                local["cdp_assist"] = make_cdp_assist(
                    assist_loop, _session_factory, job_url=job_url
                )

            try:
                with time_stage("playbook_execution_ms", kind=kind, warm=True):
                    ok = run(job_url, pdf_path, master_path, **local)
            except PlaybookFallback as pb:
                from src.core.human_assist import (
                    page_shows_captcha,
                    resolve_captcha_with_human,
                    shot_on_fail,
                )

                shot_on_fail(
                    page,
                    company=company or "",
                    reason=f"playbook_fallback_{kind}",
                    job_url=job_url,
                )
                # Captcha mid-playbook: human pause then one retry.
                if page_shows_captcha(page) and resolve_captcha_with_human(
                    page, company=company or "", job_url=job_url
                ):
                    try:
                        with time_stage(
                            "playbook_execution_ms", kind=kind, warm=True, retry=1
                        ):
                            ok = run(job_url, pdf_path, master_path, **local)
                    except PlaybookFallback as pb2:
                        shot_on_fail(
                            page,
                            company=company or "",
                            reason=f"playbook_retry_fail_{kind}",
                            job_url=job_url,
                        )
                        logger.warning("Playbook retry after captcha still failed: %s", pb2)
                        ok = False
                else:
                    logger.warning("PlaybookFallback (%s): %s", kind, pb)
                    ok = False
            if ok:
                return True
            from src.core.human_assist import shot_on_fail as _shot

            _shot(
                page,
                company=company or "",
                reason=f"playbook_miss_{kind}",
                job_url=job_url,
            )
            if _semantic():
                return True
            leave_for_vision = keep
            return False
        except PlaybookHandoff as handoff:
            handoff_pending = True
            if getattr(handoff.session, "playwright", None) is None:
                handoff.session.playwright = local_slot.playwright
            raise
        except (
            AlreadyAppliedError,
            JobClosedError,
            ApplicationBlockedError,
            ExternalPortalLoadError,
        ):
            raise
        finally:
            if not handoff_pending and not leave_for_vision:
                if not keep:
                    invalidate_slot(user_data_dir)
                elif local_slot.context is not None and browser_alive(local_slot.context):
                    local_slot.reset()
                else:
                    invalidate_slot(user_data_dir)

    # Pin Playwright to this slot's single owner thread (greenlet-safe).
    # After invalidate+recreate inside _sync, the registry slot may be new — always
    # re-resolve executor from the live registry entry.
    live = get_or_create_slot(user_data_dir)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(live.executor(), _sync)


async def _run_ats_playbook(
    kind: str,
    job_url: str,
    pdf_path: Path,
    master_path: Path,
    *,
    user_data_dir: Path | str | None = None,
    yaml_snapshot: str | None = None,
) -> bool:
    """Dispatch playbook (owned cold launch when no warm slot / no user_data_dir)."""
    from src.core.telemetry import time_stage

    run = _playbook_runner(kind)
    if run is None:
        return False
    logger.info("ATS router → %s playbook for %s", kind, job_url)
    kwargs: dict = {"user_data_dir": user_data_dir, "yaml_snapshot": yaml_snapshot}
    if kind in {"lever", "ashby"}:
        kwargs["ats"] = kind
    with time_stage("playbook_execution_ms", kind=kind, warm=False):
        return await asyncio.to_thread(run, job_url, pdf_path, master_path, **kwargs)


async def _execute_cdp_assist(
    session,
    reason: str,
    *,
    max_steps: int = 2,
    timeout_sec: float = 60,
) -> bool:
    """Short Gemini CDP assist — does NOT close the playbook session."""
    from src.core.telemetry import time_stage

    _pin_running_loop()  # refuse assist off the worker loop
    cdp_url = getattr(session, "cdp_url", None)
    from src.playbooks.session import cdp_endpoint_alive, read_live_cdp_url

    if cdp_url and not await asyncio.to_thread(cdp_endpoint_alive, cdp_url):
        cdp_url = None
        session.cdp_url = None
    if not cdp_url:
        cdp_url = await asyncio.to_thread(
            read_live_cdp_url, session.user_data_dir, wait_s=3.0
        )
        session.cdp_url = cdp_url
    if not cdp_url:
        logger.warning("CDP assist missing live DevTools port (%s)", reason)
        return False

    assist_task = (
        f"{TASK_HANDOFF.format(pdf_abs_path='', pdf_stem='', submitted_token=SUBMITTED_TOKEN)}"
        f"\n\nASSIST MODE: fix ONLY the stuck control described in reason. "
        f"Do not submit. Do not navigate away. Stop after the field is filled.\n"
        f"Assist reason: {reason}"
    )
    with time_stage("cdp_handoff_duration_ms", mode="assist", reason=str(reason)[:120]):
        try:
            await _run_agent(
                assist_task,
                [],
                max_steps=max_steps,
                timeout_seconds=timeout_sec,
                user_data_dir=session.user_data_dir,
                browser_profile=_browser_profile_cdp(cdp_url),
                teardown=False,
            )
        except TokenLimitReachedError as exc:
            logger.warning(
                "CDP assist yielding after TOKEN LIMIT REACHED: %s", exc
            )
            return False
        except RequiresHumanInterventionError:
            return False
        except ApplicationTimeoutError as exc:
            if "requires_human_api_error" in str(exc) or "TOKEN LIMIT" in str(exc):
                logger.warning(
                    "CDP assist yielding after Gemini API capacity / token limit: %s",
                    exc,
                )
            return False
        except RuntimeError as exc:
            if "different loop" in str(exc).lower() or "attached to a different" in str(
                exc
            ).lower():
                logger.warning("CDP assist loop mismatch: %s", exc)
                return False
            logger.debug("CDP assist failed", exc_info=True)
            return False
        except Exception:
            logger.debug("CDP assist failed", exc_info=True)
            return False
    return True


async def _execute_cdp_handoff(
    session,
    reason: str,
    job_url: str,
    pdf_path: Path,
    master_path: Path,
    *,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
    cleanup_linkedin_resume: bool = False,
) -> bool:
    """Inject Gemini into the open playbook Chromium via CDP; always close session."""
    from src.core.browser_warmup import invalidate_slot
    from src.core.telemetry import time_stage

    _pin_running_loop()
    pdf_abs = os.path.abspath(str(pdf_path))
    pdf_stem = pdf_path.stem
    from src.playbooks.session import cdp_endpoint_alive, read_live_cdp_url

    cdp_url = getattr(session, "cdp_url", None)
    if cdp_url and not await asyncio.to_thread(cdp_endpoint_alive, cdp_url):
        cdp_url = None
        session.cdp_url = None
    if not cdp_url:
        cdp_url = await asyncio.to_thread(
            read_live_cdp_url, session.user_data_dir, wait_s=3.0
        )
        session.cdp_url = cdp_url

    succeeded = False
    try:
        if not cdp_url:
            logger.warning(
                "CDP handoff missing live DevTools port (%s) — soft-fail", reason
            )
            raise RequiresHumanInterventionError(
                f"No live CDP endpoint for handoff: {reason}"
            )

        # Prefer deterministic disk upload before Gemini so library resumes are not reused.
        page = getattr(session, "page", None)
        if page is not None and Path(pdf_path).is_file():
            try:
                from src.playbooks._common import force_upload_resume

                uploaded = await asyncio.to_thread(force_upload_resume, page, pdf_path)
                logger.info(
                    "CDP handoff pre-upload %s for %s",
                    "ok" if uploaded else "miss",
                    Path(pdf_path).name,
                )
            except Exception:
                logger.debug("CDP handoff pre-upload failed", exc_info=True)

        form_answers = _application_answers_block(
            master_path, jd_text=jd_text, yaml_snapshot=yaml_snapshot
        )
        max_steps, timeout_sec = _step_budget(job_url)
        task = TASK_HANDOFF.format(
            pdf_abs_path=pdf_abs,
            pdf_stem=pdf_stem,
            submitted_token=SUBMITTED_TOKEN,
        )
        form_block = f"\n\n{form_answers}\n" if form_answers else ""
        task = f"{task}{form_block}\n\nHandoff reason: {reason}"

        logger.info(
            "CDP Gemini handoff for %s (cdp=%s, reason=%s)",
            job_url[:160],
            cdp_url,
            reason,
        )
        with time_stage(
            "cdp_handoff_duration_ms", mode="takeover", reason=str(reason)[:120]
        ):
            history = await _run_agent(
                task,
                [pdf_abs],
                max_steps=max_steps,
                timeout_seconds=timeout_sec,
                user_data_dir=session.user_data_dir,
                browser_profile=_browser_profile_cdp(cdp_url),
                teardown=False,
            )
        hist_blob = _history_blob(history) if history is not None else ""
        if _reached_token(history, SUBMITTED_TOKEN, pdf_stem=pdf_stem):
            logger.info("CDP handoff verified %s for %s", SUBMITTED_TOKEN, job_url[:160])
            succeeded = True
            if cleanup_linkedin_resume:
                await _linkedin_cleanup_safe(
                    pdf_path,
                    job_url=job_url,
                    user_data_dir=getattr(session, "user_data_dir", None),
                    reason="cdp_handoff_success",
                )
            return True
        # Playbook / pre-upload already attached the disk PDF — trust LinkedIn
        # confirmation copy even when Gemini history has no upload_file event.
        if SUBMITTED_TOKEN in hist_blob and _submit_confirmed(hist_blob):
            logger.info(
                "CDP handoff accepting LinkedIn submit confirm without CDP upload event for %s",
                job_url[:160],
            )
            succeeded = True
            if cleanup_linkedin_resume:
                await _linkedin_cleanup_safe(
                    pdf_path,
                    job_url=job_url,
                    user_data_dir=getattr(session, "user_data_dir", None),
                    reason="cdp_handoff_submit_confirm",
                )
            return True
        # Gemini stuck on unknown fields — popup so human can finish + Ace learns.
        page = getattr(session, "page", None)
        if page is not None:
            from src.core.human_assist import (
                form_looks_submitted,
                resolve_unknown_with_human,
            )

            helped = await asyncio.to_thread(
                resolve_unknown_with_human,
                page,
                company="",
                job_url=job_url,
                reason=f"CDP handoff incomplete: {reason}",
            )
            page_ok = False
            if helped:
                try:
                    page_ok = bool(await asyncio.to_thread(form_looks_submitted, page))
                except Exception:
                    page_ok = False
            if helped and (
                page_ok
                or _submit_confirmed(hist_blob)
                or SUBMITTED_TOKEN in hist_blob
            ):
                logger.info(
                    "Human finished apply after CDP handoff for %s", job_url[:160]
                )
                succeeded = True
                if cleanup_linkedin_resume:
                    await _linkedin_cleanup_safe(
                        pdf_path,
                        job_url=job_url,
                        user_data_dir=getattr(session, "user_data_dir", None),
                        reason="human_unknown_success",
                    )
                return True
        raise RequiresHumanInterventionError(
            f"CDP handoff did not verify submit after: {reason}"
        )
    except RequiresHumanInterventionError:
        raise
    except TokenLimitReachedError:
        raise
    except ApplicationTimeoutError as exc:
        # Clean yield — warm Chromium stays owned by session teardown below;
        # worker moves to the next job without crashing the slot.
        if "requires_human_api_error" in str(exc) or "TOKEN LIMIT" in str(exc):
            from src.core.llm_router import mark_token_limit_reached

            mark_token_limit_reached(str(exc), blocks_apply=True)
            raise TokenLimitReachedError(
                f"TOKEN LIMIT REACHED — pausing applies until quotas reset ({exc})"
            ) from exc
        raise RequiresHumanInterventionError(
            f"CDP handoff timed out: {exc}"
        ) from exc
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "different loop" in msg or "attached to a different" in msg:
            raise RequiresHumanInterventionError(
                f"CDP handoff loop mismatch: {exc}"
            ) from exc
        raise RequiresHumanInterventionError(
            f"CDP handoff failed ({type(exc).__name__}: {exc})"
        ) from exc
    except Exception as exc:
        raise RequiresHumanInterventionError(
            f"CDP handoff failed ({type(exc).__name__}: {exc})"
        ) from exc
    finally:
        if not succeeded:
            try:
                await asyncio.to_thread(session.discard_modal)
            except Exception:
                logger.debug("modal discard failed", exc_info=True)
        # Non-destructive: disconnect Gemini path via page reset; keep Chromium/slot.
        try:
            await asyncio.to_thread(session.detach)
        except Exception:
            logger.debug("playbook session detach failed", exc_info=True)
        if not env_truthy("ACE_BROWSER_KEEP_ALIVE", "1"):
            try:
                await asyncio.to_thread(session.close)
            except Exception:
                logger.debug("playbook session close failed", exc_info=True)
            try:
                invalidate_slot(getattr(session, "user_data_dir", None))
            except Exception:
                logger.debug("slot invalidate after cold handoff failed", exc_info=True)


async def execute_application(
    job_url: str,
    tailored_pdf_path: str | Path,
    master_yaml_path: str | Path,
    *,
    auto_submit: bool | None = None,
    cleanup_linkedin_resume: bool | None = None,
    user_data_dir: Path | str | None = None,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
    company: str = "",
    job_title: str = "",
) -> bool:
    """Apply via ATS playbook when possible, else browser-use Gemini vision.

    Greenhouse / Lever / Ashby / Workday / LinkedIn DOM playbooks run first.
    On PlaybookHandoff the same Chromium stays open and Gemini attaches via CDP.
    Cold Gemini (user_data_dir launch) runs when there is no live handoff session.
    """
    pdf_path = Path(tailored_pdf_path).expanduser().resolve()
    master_path = Path(master_yaml_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Tailored PDF not found: {pdf_path}")
    if not master_path.is_file():
        raise FileNotFoundError(f"Master YAML not found: {master_path}")

    job_url = (job_url or "").strip()
    if not job_url:
        raise ValueError("job_url is required")

    # Mailto applies need no Chromium — SMTP + attached resume.
    if job_url.lower().startswith("mailto:"):
        from src.core.mailto_apply import send_mailto_application

        return bool(
            send_mailto_application(
                mailto_url=job_url,
                company=company or "",
                job_title=job_title or "",
                jd_text=jd_text or "",
                pdf_path=pdf_path,
                yaml_snapshot=yaml_snapshot or "",
            )
        )

    do_submit = _auto_submit_enabled() if auto_submit is None else bool(auto_submit)
    do_cleanup = (
        cleanup_linkedin_resume
        if cleanup_linkedin_resume is not None
        else ("linkedin.com" in job_url.lower())
    )

    from src.core.telemetry import time_stage
    from src.playbooks import PlaybookFallback, resolve_ats_kind
    from src.playbooks.session import PlaybookHandoff

    prefer_cdp = bool(user_data_dir) and env_truthy("ACE_BROWSER_KEEP_ALIVE", "1")

    with time_stage("total_apply_duration_ms", url=(job_url or "")[:160]):
        # Warm path: one Chromium does probe → ATS resolve → playbook (pool/daemon).
        if do_submit and user_data_dir:
            try:
                ok = await _warm_ats_apply(
                    job_url,
                    pdf_path,
                    master_path,
                    user_data_dir=user_data_dir,
                    yaml_snapshot=yaml_snapshot,
                    jd_text=jd_text,
                    company=company,
                    job_title=job_title,
                )
                if ok:
                    logger.info("Warm ATS playbook succeeded for %s", job_url)
                    if do_cleanup:
                        await _linkedin_cleanup_safe(
                            pdf_path,
                            job_url=job_url,
                            user_data_dir=user_data_dir,
                            reason="warm_success",
                        )
                    return True
            except AlreadyAppliedError:
                raise
            except JobClosedError:
                raise
            except ApplicationBlockedError:
                raise
            except ExternalPortalLoadError:
                # Dead company portal / DNS — never burn Gemini CDP.
                raise
            except PlaybookHandoff as handoff:
                if is_portal_load_failure(handoff.reason) or _handoff_should_skip_cdp(
                    handoff.reason
                ):
                    raise ExternalPortalLoadError(
                        f"FAILED_EXTERNAL_LOAD: {handoff.reason}"
                    ) from handoff
                logger.warning(
                    "ATS playbook stuck → CDP Gemini (reason=%s)",
                    handoff.reason,
                )
                return await _execute_cdp_handoff(
                    handoff.session,
                    handoff.reason,
                    job_url,
                    pdf_path,
                    master_path,
                    yaml_snapshot=yaml_snapshot,
                    jd_text=jd_text,
                    cleanup_linkedin_resume=do_cleanup,
                )
            except WarmBrowserUnavailable:
                # No live Chromium — skip zombie CDP attach; cold-launch same profile.
                prefer_cdp = False
                logger.exception(
                    "Warm ATS Chromium unavailable — cold profile launch (no CDP attach)"
                )
            except (TimeoutError, PlaybookFallback) as exc:
                if is_portal_load_failure(exc):
                    raise ExternalPortalLoadError(
                        f"FAILED_EXTERNAL_LOAD: {exc}"
                    ) from exc
                logger.warning("Warm playbook miss → fallback (%s)", exc)
            except (
                TokenLimitReachedError,
                RequiresHumanInterventionError,
            ):
                raise
            except Exception:
                logger.exception("Warm ATS path crashed — falling back")

        # Cold / no-profile: probe + classify then optional owned playbook (scripts).
        elif do_submit:
            from src.core.job_closed import probe_job_page

            probe = await asyncio.to_thread(
                probe_job_page, job_url, user_data_dir=user_data_dir
            )
            if probe == "closed":
                raise JobClosedError(
                    f"Job closed or page does not exist at {job_url[:160]}"
                )
            if probe == "already_applied":
                raise AlreadyAppliedError(
                    f"DOM already-applied signal at {job_url[:160]}"
                )
            kind = await asyncio.to_thread(
                resolve_ats_kind, job_url, user_data_dir=user_data_dir
            )
            if kind in _ATS_KINDS:
                try:
                    ok = await _run_ats_playbook(
                        kind,
                        job_url,
                        pdf_path,
                        master_path,
                        user_data_dir=user_data_dir,
                        yaml_snapshot=yaml_snapshot,
                    )
                    if ok:
                        if do_cleanup:
                            await _linkedin_cleanup_safe(
                                pdf_path,
                                job_url=job_url,
                                user_data_dir=user_data_dir,
                                reason="cold_playbook_success",
                            )
                        return True
                except PlaybookHandoff as handoff:
                    if is_portal_load_failure(handoff.reason) or _handoff_should_skip_cdp(
                        handoff.reason
                    ):
                        raise ExternalPortalLoadError(
                            f"FAILED_EXTERNAL_LOAD: {handoff.reason}"
                        ) from handoff
                    return await _execute_cdp_handoff(
                        handoff.session,
                        handoff.reason,
                        job_url,
                        pdf_path,
                        master_path,
                        yaml_snapshot=yaml_snapshot,
                        jd_text=jd_text,
                        cleanup_linkedin_resume=do_cleanup,
                    )
                except (
                    AlreadyAppliedError,
                    JobClosedError,
                    ApplicationBlockedError,
                    ExternalPortalLoadError,
                    TokenLimitReachedError,
                    RequiresHumanInterventionError,
                ):
                    raise
                except Exception:
                    logger.warning("Cold playbook path failed — continuing", exc_info=True)

        # Cold semantic only when no warm profile and quotas are still alive.
        if do_submit and not user_data_dir:
            from src.core.llm_router import token_limit_detail, token_limit_reached

            if token_limit_reached():
                raise TokenLimitReachedError(
                    f"TOKEN LIMIT REACHED — pausing applies until quotas reset "
                    f"({token_limit_detail()})"
                )
            try:
                from src.core.semantic_dom import try_semantic_apply

                semantic_ok = await asyncio.to_thread(
                    try_semantic_apply,
                    job_url,
                    pdf_path,
                    master_path,
                    user_data_dir=user_data_dir,
                )
                if semantic_ok:
                    logger.info("Semantic DOM apply succeeded for %s", job_url[:160])
                    if do_cleanup:
                        await _linkedin_cleanup_safe(
                            pdf_path,
                            job_url=job_url,
                            user_data_dir=user_data_dir,
                            reason="cold_semantic_success",
                        )
                    return True
            except Exception:
                logger.exception("Semantic DOM apply crashed — continuing to Gemini")

        from src.core.llm_router import token_limit_detail, token_limit_reached

        if token_limit_reached():
            raise TokenLimitReachedError(
                f"TOKEN LIMIT REACHED — pausing applies until quotas reset "
                f"({token_limit_detail()})"
            )

        return await _execute_with_browser_use(
            job_url,
            pdf_path,
            master_path,
            auto_submit=do_submit,
            cleanup_linkedin_resume=do_cleanup,
            user_data_dir=user_data_dir,
            yaml_snapshot=yaml_snapshot,
            jd_text=jd_text,
            prefer_cdp=prefer_cdp,
        )


async def _execute_with_browser_use(
    job_url: str,
    pdf_path: Path,
    master_path: Path,
    *,
    auto_submit: bool,
    cleanup_linkedin_resume: bool,
    user_data_dir: Path | str | None = None,
    yaml_snapshot: str | None = None,
    jd_text: str | None = None,
    prefer_cdp: bool = False,
) -> bool:
    """Drive browser-use through the apply form (Gemini vision/DOM)."""
    pdf_abs = os.path.abspath(str(pdf_path))
    pdf_stem = pdf_path.stem
    pdf_filename = pdf_path.name
    do_submit = bool(auto_submit)
    do_cleanup = bool(cleanup_linkedin_resume)

    master_preview = master_path.read_text(encoding="utf-8")
    form_answers = _application_answers_block(
        master_path, jd_text=jd_text, yaml_snapshot=yaml_snapshot
    )
    lessons = load_lessons()
    max_steps, timeout_sec = _step_budget(job_url)

    if do_submit:
        task = TASK_AUTO.format(
            job_url=job_url,
            pdf_abs_path=pdf_abs,
            pdf_stem=pdf_stem,
            pdf_filename=pdf_filename,
            submitted_token=SUBMITTED_TOKEN,
        )
        win_token = SUBMITTED_TOKEN
    else:
        task = TASK_REVIEW.format(
            job_url=job_url,
            pdf_abs_path=pdf_abs,
            pdf_filename=pdf_filename,
            ready_token=READY_TOKEN,
        )
        win_token = READY_TOKEN

    lessons_block = (
        f"\n\n## Lessons Learned (self-training — follow strictly)\n"
        f"---\n{lessons}\n---\n"
        if lessons
        else ""
    )
    lessons_block = lessons_block[:2500] + (
        "\n…[truncated]\n" if len(lessons_block) > 2500 else ""
    )
    form_block = f"\n\n{form_answers}\n" if form_answers else ""
    master_clip = master_preview
    if len(master_clip) > 12000:
        master_clip = master_clip[:12000] + "\n…[truncated master.yaml]\n"
    task = (
        f"{task}{form_block}{lessons_block}\n\n"
        f"--- BEGIN master.yaml contents ---\n{master_clip}\n"
        f"--- END master.yaml contents ---"
    )

    cdp_profile = None
    if prefer_cdp and user_data_dir:
        from src.playbooks.session import read_live_cdp_url

        cdp_url = await asyncio.to_thread(read_live_cdp_url, user_data_dir, wait_s=3.0)
        if cdp_url:
            cdp_profile = _browser_profile_cdp(cdp_url)
            logger.info("Gemini vision attaching via warm CDP %s", cdp_url)
        else:
            logger.warning(
                "Warm CDP missing/dead — cold Gemini launch for %s", job_url[:160]
            )

    logger.info(
        "browser-use agent starting for %s (auto_submit=%s, steps=%s, timeout=%ss, pdf=%s, profile=%s, cdp=%s)",
        job_url,
        do_submit,
        max_steps,
        timeout_sec,
        pdf_abs,
        user_data_dir or "(master)",
        bool(cdp_profile),
    )

    history = None
    used_cdp = bool(cdp_profile)
    try:
        try:
            history = await _run_agent(
                task,
                [pdf_abs],
                max_steps=max_steps,
                timeout_seconds=timeout_sec,
                user_data_dir=user_data_dir,
                browser_profile=cdp_profile,
                teardown=not bool(cdp_profile),
            )
        except Exception as exc:
            if not cdp_profile or not _is_cdp_attach_error(exc):
                raise
            logger.warning(
                "CDP attach failed (%s) — cold profile retry for %s",
                type(exc).__name__,
                job_url[:160],
            )
            try:
                from src.playbooks.session import clear_stale_devtools_port

                await asyncio.to_thread(clear_stale_devtools_port, user_data_dir)
            except Exception:
                logger.debug("clear stale CDP after attach fail", exc_info=True)
            used_cdp = False
            history = await _run_agent(
                task,
                [pdf_abs],
                max_steps=max_steps,
                timeout_seconds=timeout_sec,
                user_data_dir=user_data_dir,
                browser_profile=None,
                teardown=True,
            )

        if _reached_token(history, win_token, pdf_stem=pdf_stem):
            logger.info("Verified %s for %s (resume=%s)", win_token, job_url, pdf_stem)
            if do_submit and do_cleanup:
                await _linkedin_cleanup_safe(
                    pdf_path,
                    job_url=job_url,
                    user_data_dir=user_data_dir,
                    reason="vision_success",
                )
            return True

        # Lumenci-class miss: agent reaches Review then burns budget / 503 before Submit.
        # Must run before slot reset in finally.
        if do_submit and await _try_finish_easy_apply_submit(
            user_data_dir, job_url=job_url, pdf_path=pdf_path
        ):
            logger.info(
                "Salvaged LinkedIn Submit after vision miss for %s", job_url[:160]
            )
            if do_cleanup:
                await _linkedin_cleanup_safe(
                    pdf_path,
                    job_url=job_url,
                    user_data_dir=user_data_dir,
                    reason="review_submit_salvage",
                )
            return True

        _raise_if_blocked(history)

        blob = _history_blob(history)
        if do_submit and SUBMITTED_TOKEN in blob and not _resume_upload_verified(
            blob, pdf_stem
        ):
            raise RuntimeError(
                "Agent claimed submit but fresh resume upload was NOT verified. "
                "Treat as failed apply — do not count SUCCESS."
            )

        success = history.is_successful() if history is not None else None
        final = history.final_result() if history is not None else None
        # Oracle/TriNet desert page etc. — skip quietly, never Manual_Action.
        try:
            from src.core.job_closed import text_shows_job_closed

            if text_shows_job_closed(f"{final or ''}\n{blob}"):
                raise JobClosedError(
                    f"Job closed/unavailable (agent reported): {str(final or blob)[:240]}"
                )
        except JobClosedError:
            raise
        except Exception:
            logger.debug("job-closed text check failed", exc_info=True)
        raise RuntimeError(
            f"browser-use agent finished without verified {win_token}. "
            f"is_done={history.is_done() if history else None}, "
            f"is_successful={success}, final={final!r}"
        )
    except TimeoutError as exc:
        if do_submit and await _try_finish_easy_apply_submit(
            user_data_dir, job_url=job_url, pdf_path=pdf_path
        ):
            logger.info(
                "Salvaged LinkedIn Submit after vision timeout for %s", job_url[:160]
            )
            if do_cleanup:
                await _linkedin_cleanup_safe(
                    pdf_path,
                    job_url=job_url,
                    user_data_dir=user_data_dir,
                    reason="timeout_submit_salvage",
                )
            return True
        raise ApplicationTimeoutError(
            f"browser-use agent timed out after {timeout_sec}s on {job_url}"
        ) from exc
    except ApplicationTimeoutError:
        if do_submit and await _try_finish_easy_apply_submit(
            user_data_dir, job_url=job_url, pdf_path=pdf_path
        ):
            logger.info(
                "Salvaged LinkedIn Submit after capacity/timeout for %s",
                job_url[:160],
            )
            if do_cleanup:
                await _linkedin_cleanup_safe(
                    pdf_path,
                    job_url=job_url,
                    user_data_dir=user_data_dir,
                    reason="capacity_submit_salvage",
                )
            return True
        raise
    finally:
        if used_cdp and user_data_dir and env_truthy("ACE_BROWSER_KEEP_ALIVE", "1"):
            try:
                from src.core.browser_warmup import get_or_create_slot

                await asyncio.to_thread(get_or_create_slot(user_data_dir).reset)
            except Exception:
                logger.debug("slot reset after vision CDP failed", exc_info=True)
