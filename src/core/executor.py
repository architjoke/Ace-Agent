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
SUBMITTED_TOKEN = "APPLICATION_SUBMITTED"
READY_TOKEN = "READY_FOR_REVIEW"  # legacy / dry-run only
CLEANUP_TOKEN = "LINKEDIN_RESUME_DELETED"


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
        )
    )


def _step_budget(job_url: str) -> tuple[int, int]:
    """Return (max_steps, timeout_seconds) for this apply URL."""
    if _is_heavy_ats(job_url):
        return WORKDAY_MAX_STEPS, WORKDAY_TIMEOUT_SECONDS
    return DEFAULT_MAX_STEPS, DEFAULT_TIMEOUT_SECONDS

# Filename shown on LinkedIn / ATS (no company, no timestamp).
# Override with ACE_APPLY_RESUME_FILENAME=Your_Name_Resume.pdf
APPLY_RESUME_FILENAME = (
    os.environ.get("ACE_APPLY_RESUME_FILENAME") or "Ace_Agent_Resume.pdf"
).strip() or "Ace_Agent_Resume.pdf"

# LinkedIn Job application settings → Resumes library
LINKEDIN_RESUMES_URL = "https://www.linkedin.com/mypreferences/d/job-application-settings"

# User's permanent LinkedIn resumes — NEVER delete these (substring match).
# Set ACE_LINKEDIN_KEEP_RESUMES=Name1|Name2 (empty default = delete only the agent upload name).
_DEFAULT_KEEP_RESUMES: tuple[str, ...] = ()

_BLOCK_MARKERS = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "cloudflare",
    "blocked by bot",
    "access denied",
    "verify you are human",
    "unusual traffic",
)

class ApplicationBlockedError(RuntimeError):
    """Raised when the browser agent is permanently blocked (CAPTCHA / hard wall)."""


class ApplicationTimeoutError(TimeoutError):
    """Raised when the browser agent exceeds the allowed wall-clock timeout."""


class AlreadyAppliedError(RuntimeError):
    """Raised when DB or DOM indicates this job was already applied to."""


def _auto_submit_enabled() -> bool:
    return env_truthy("ACE_AUTO_SUBMIT", "1")


def _keep_resume_names() -> list[str]:
    raw = (os.environ.get("ACE_LINKEDIN_KEEP_RESUMES") or "").strip()
    if raw:
        return [x.strip() for x in raw.split("|") if x.strip()]
    return list(_DEFAULT_KEEP_RESUMES)


def _application_answers_block(master_path: Path) -> str:
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
    lines = ["## APPLICATION FORM ANSWERS (authoritative — use these on the form)"]
    for key, value in answers.items():
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


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
    # keep_alive default OFF — tear down Chromium after each apply (anti-zombie).
    # Google/LinkedIn cookies still persist via user_data_dir on disk.
    keep = env_truthy("ACE_BROWSER_KEEP_ALIVE", "0")
    from src.core.network import browser_proxy_settings
    from src.playbooks._common import CHROMIUM_STEALTH_ARGS, pick_chrome_user_agent

    proxy = browser_proxy_settings()
    if proxy is not None:
        logger.info("BrowserProfile sticky proxy enabled (PROXY_URL / network.proxies)")
    return BrowserProfile(
        user_data_dir=str(profile_dir),
        profile_directory=settings.browser_profile_directory,
        headless=settings.browser_headless,
        keep_alive=keep,
        proxy=proxy,
        args=list(CHROMIUM_STEALTH_ARGS),
        user_agent=pick_chrome_user_agent(),
        # Leave allowed_domains unset so accounts.google.com OAuth popups work.
        cross_origin_iframes=True,
        # Fluidity: take control on DOM-ready; do not wait on tracking pixels.
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
    cleaned = (
        blob.replace('"reached_captcha": false', " ")
        .replace("'reached_captcha': false", " ")
        .replace("reached_captcha: false", " ")
        .replace('"reached_captcha":false', " ")
    )
    for marker in _BLOCK_MARKERS:
        if marker in cleaned:
            raise ApplicationBlockedError(
                f"Browser agent permanently blocked ({marker!r}). Detail: {blob[:1500]}"
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
        blob = _history_blob(history)
        return CLEANUP_TOKEN in blob or (
            history.is_done()
            and ("deleted" in blob.lower() or "removed" in blob.lower())
            and (pdf_stem.lower() in blob.lower() if pdf_stem else True)
        )
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
        mm = getattr(agent, "message_manager", None) or getattr(agent, "state", None)
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


async def _force_close_agent(agent: Agent | None, user_data_dir: Path | str | None) -> None:
    """Best-effort Chromium teardown so the next job starts clean."""
    if agent is not None:
        for attr in ("close", "stop"):
            fn = getattr(agent, attr, None)
            if callable(fn):
                try:
                    result = fn()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.debug("agent.%s teardown failed", attr, exc_info=True)
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


async def _run_agent(
    task: str,
    file_paths: list[str],
    *,
    max_steps: int,
    timeout_seconds: int | None = None,
    user_data_dir: Path | str | None = None,
) -> object:
    api_key = _api_key()
    llm = _build_llm(api_key)
    agent_ref: dict = {"agent": None}
    stuck = _StuckStateTracker(agent_ref)
    agent: Agent | None = Agent(
        task=task,
        llm=llm,
        browser_profile=_browser_profile(user_data_dir),
        use_vision=True,
        available_file_paths=file_paths,
        max_failures=8,
        max_actions_per_step=12,
        extend_system_message=BROWSER_SYSTEM_EXTENSION,
        register_new_step_callback=stuck,
    )
    agent_ref["agent"] = agent
    wall = int(timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS)
    try:
        return await asyncio.wait_for(
            agent.run(max_steps=max_steps),
            timeout=wall,
        )
    except ApplicationTimeoutError:
        raise
    except TimeoutError as exc:
        raise ApplicationTimeoutError(
            f"browser-use agent timed out after {wall}s"
        ) from exc
    finally:
        await _force_close_agent(agent, user_data_dir)


async def cleanup_linkedin_uploaded_resume(
    pdf_path: Path | str,
    *,
    user_data_dir: Path | str | None = None,
) -> bool:
    """Delete the agent upload PDF from LinkedIn Jobs → Preferences → Resumes."""
    if pdf_path:
        pdf = Path(pdf_path).expanduser().resolve()
    else:
        pdf = Path(get_settings().output_dir) / APPLY_RESUME_FILENAME
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
            max_steps=12,
            timeout_seconds=180,
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


async def _run_ats_playbook(
    kind: str,
    job_url: str,
    pdf_path: Path,
    master_path: Path,
    *,
    user_data_dir: Path | str | None = None,
) -> bool:
    """Dispatch to the DOM playbook for a classified ATS kind."""
    logger.info("ATS router → %s playbook for %s", kind, job_url)
    if kind == "greenhouse":
        from src.playbooks.greenhouse import run_greenhouse

        return await asyncio.to_thread(
            run_greenhouse,
            job_url,
            pdf_path,
            master_path,
            user_data_dir=user_data_dir,
        )
    if kind in {"lever", "ashby"}:
        from src.playbooks.lever import run_lever

        return await asyncio.to_thread(
            run_lever,
            job_url,
            pdf_path,
            master_path,
            user_data_dir=user_data_dir,
            ats=kind,
        )
    if kind == "workday":
        from src.playbooks.workday import run_workday

        return await asyncio.to_thread(
            run_workday,
            job_url,
            pdf_path,
            master_path,
            user_data_dir=user_data_dir,
        )
    return False


async def execute_application(
    job_url: str,
    tailored_pdf_path: str | Path,
    master_yaml_path: str | Path,
    *,
    auto_submit: bool | None = None,
    cleanup_linkedin_resume: bool | None = None,
    user_data_dir: Path | str | None = None,
) -> bool:
    """Apply via ATS playbook when possible, else browser-use Gemini vision.

    Greenhouse / Lever / Ashby / Workday DOM playbooks run first (no vision
    tokens). On TimeoutError / PlaybookFallback the Chromium profile is closed
    and Gemini takes over with the same user_data_dir cookies.
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

    do_submit = _auto_submit_enabled() if auto_submit is None else bool(auto_submit)
    do_cleanup = (
        cleanup_linkedin_resume
        if cleanup_linkedin_resume is not None
        else ("linkedin.com" in job_url.lower())
    )

    # Phase 1B: DOM fast-fail before playbook / Gemini token burn.
    from src.core.already_applied import probe_already_applied

    already = await asyncio.to_thread(
        probe_already_applied, job_url, user_data_dir=user_data_dir
    )
    if already:
        raise AlreadyAppliedError(f"DOM already-applied signal at {job_url[:160]}")

    from src.playbooks import PlaybookFallback, classify_ats

    kind = classify_ats(job_url)
    # Known ATS: deterministic playbooks first (fast). Semantic/Gemini only as fallback.
    if do_submit and kind in {"greenhouse", "lever", "ashby", "workday"}:
        try:
            ok = await _run_ats_playbook(
                kind, job_url, pdf_path, master_path, user_data_dir=user_data_dir
            )
            if ok:
                logger.info("%s playbook succeeded for %s", kind, job_url)
                return True
        except (TimeoutError, PlaybookFallback) as exc:
            logger.warning(
                "ATS playbook stuck → semantic/Gemini handoff (kind=%s, reason=%s)",
                kind,
                exc,
            )
            try:
                from src.core.proxy_session import ProxySessionManager

                mgr = ProxySessionManager()
                if mgr.available() and mgr.rotate():
                    ok_retry = await _run_ats_playbook(
                        kind, job_url, pdf_path, master_path, user_data_dir=user_data_dir
                    )
                    if ok_retry:
                        logger.info(
                            "%s playbook succeeded after proxy rotate for %s",
                            kind,
                            job_url,
                        )
                        return True
            except Exception:
                logger.debug("Playbook proxy-rotate retry skipped", exc_info=True)
        except Exception:
            logger.exception(
                "ATS playbook crashed (kind=%s) — falling back",
                kind,
            )

    # Groq semantic DOM for unknown ATS / after playbook miss — before Gemini vision.
    if do_submit:
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
                return True
        except Exception:
            logger.exception("Semantic DOM apply crashed — continuing to Gemini")

    return await _execute_with_browser_use(
        job_url,
        pdf_path,
        master_path,
        auto_submit=do_submit,
        cleanup_linkedin_resume=do_cleanup,
        user_data_dir=user_data_dir,
    )


async def _execute_with_browser_use(
    job_url: str,
    pdf_path: Path,
    master_path: Path,
    *,
    auto_submit: bool,
    cleanup_linkedin_resume: bool,
    user_data_dir: Path | str | None = None,
) -> bool:
    """Drive browser-use through the apply form (Gemini vision/DOM)."""
    pdf_abs = os.path.abspath(str(pdf_path))
    pdf_stem = pdf_path.stem
    pdf_filename = pdf_path.name
    do_submit = bool(auto_submit)
    do_cleanup = bool(cleanup_linkedin_resume)

    master_preview = master_path.read_text(encoding="utf-8")
    form_answers = _application_answers_block(master_path)
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

    logger.info(
        "browser-use agent starting for %s (auto_submit=%s, steps=%s, timeout=%ss, pdf=%s, profile=%s)",
        job_url,
        do_submit,
        max_steps,
        timeout_sec,
        pdf_abs,
        user_data_dir or "(master)",
    )

    try:
        history = await _run_agent(
            task,
            [pdf_abs],
            max_steps=max_steps,
            timeout_seconds=timeout_sec,
            user_data_dir=user_data_dir,
        )
    except TimeoutError as exc:
        raise ApplicationTimeoutError(
            f"browser-use agent timed out after {timeout_sec}s on {job_url}"
        ) from exc

    if _reached_token(history, win_token, pdf_stem=pdf_stem):
        logger.info("Verified %s for %s (resume=%s)", win_token, job_url, pdf_stem)
        if do_submit and do_cleanup and "linkedin.com" in job_url.lower():
            try:
                await cleanup_linkedin_uploaded_resume(
                    pdf_path, user_data_dir=user_data_dir
                )
            except Exception:
                logger.exception(
                    "Post-apply LinkedIn resume cleanup failed for %s", pdf_path.name
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
    raise RuntimeError(
        f"browser-use agent finished without verified {win_token}. "
        f"is_done={history.is_done() if history else None}, "
        f"is_successful={success}, final={final!r}"
    )
