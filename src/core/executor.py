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

from src.core.reflect import load_lessons
from src.core.settings import get_settings

logger = logging.getLogger(__name__)

from src.core.llm_router import gemini_browser_model_name

MODEL = gemini_browser_model_name()
# Easy Apply defaults (Workday/ATS get a higher budget via URL heuristics).
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("ACE_BROWSER_USE_TIMEOUT_SECONDS", "420"))
DEFAULT_MAX_STEPS = int(os.environ.get("ACE_BROWSER_USE_MAX_STEPS", "50"))
WORKDAY_TIMEOUT_SECONDS = int(os.environ.get("ACE_WORKDAY_TIMEOUT_SECONDS", "900"))
WORKDAY_MAX_STEPS = int(os.environ.get("ACE_WORKDAY_MAX_STEPS", "120"))
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

_TASK_TEMPLATE_AUTO = """Navigate to {job_url}.

You are AUTONOMOUSLY APPLYING for this job.

## Resume — ALWAYS upload from disk (never reuse LinkedIn library)
Absolute path (ONLY this file):
{pdf_abs_path}
On-screen name must show: {pdf_stem}

### LinkedIn Easy Apply — mandatory resume steps
1) Click Easy Apply.
2) On the Resume step you MUST:
   - Do NOT select any already-listed resume — even one with the same display name.
   - Always click **Upload resume** / **Upload file** / **Replace** (fresh file from disk).
   - Use upload_file with path = {pdf_abs_path}.
   - Confirm the attached file shows as "{pdf_filename}" / "{pdf_stem}".
3) Only then click Next through remaining questions.
4) On Review, confirm that same file is attached, then **Submit application**.
5) Wait for confirmation ("Application submitted" / "Your application was sent").

### Other ATS (Indeed / Greenhouse / Lever / Workday / Keka / careers pages)
Always upload {pdf_abs_path} via the file input — never reuse a previously saved CV.

## Form filling (mandatory)
Use the APPLICATION FORM ANSWERS block below for every matching field.
- Phone: use phone_country_code + phone_national / phone_display (never invent another country).
- Notice / availability / join date: use notice_period / earliest_join_date (never claim same-day join unless forced — then pick within that window).
- Address / pincode / ZIP / postal: use the listed city, state, country, pincode fields.
- Do not invent street/house numbers if not listed — use city/state/PIN/country.
- Never fabricate salary history, foreign citizenship, or security-clearance eligibility.

## HARD RULES
- ALWAYS upload from the disk path above — never click a previously listed LinkedIn resume.
- NEVER click Next/Submit until "{pdf_stem}" is visibly attached from this upload.
- On CAPTCHA / Cloudflare / login wall you cannot clear: stop and describe the blocker.
- SCAM ABORT: If the form/JD asks to join a WhatsApp group/community/channel, or is a
  "training and placement" internship that asks for any fee/amount/deposit — STOP.
  Do NOT submit. Describe the scam signal and leave (treat as blocked).
- After real confirmation UI, output exactly: {submitted_token}
"""

_TASK_TEMPLATE_REVIEW = """Navigate to {job_url}.

ALWAYS upload resume from disk (never select an old LinkedIn resume):
{pdf_abs_path}
On-screen name should be: {pdf_filename}

## Form filling (mandatory)
Use the APPLICATION FORM ANSWERS block below for phone, notice/join window,
address/pincode, and other listed fields. Do not invent unsupported facts.

DO NOT click final Submit. When filled + OUR file is attached, output: {ready_token}
"""

_TASK_TEMPLATE_LINKEDIN_CLEANUP = """Go to LinkedIn Job application settings / Resumes library:
{resumes_url}

Goal: DELETE the agent-uploaded resume named exactly:
{pdf_filename}

Steps:
1) Open Resumes (Jobs → Preferences → Job application settings → Resumes).
2) Find "{pdf_filename}" in the uploaded list (exact agent upload name).
3) Delete / Remove ONLY that file.
4) NEVER delete these permanent resumes (user keep-list):
{keep_list}
5) When "{pdf_filename}" is gone (or already absent), output exactly: {cleanup_token}

Abort on CAPTCHA. Do not touch any other resume.
"""


class ApplicationBlockedError(RuntimeError):
    """Raised when the browser agent is permanently blocked (CAPTCHA / hard wall)."""


class ApplicationTimeoutError(TimeoutError):
    """Raised when the browser agent exceeds the allowed wall-clock timeout."""


def _auto_submit_enabled() -> bool:
    raw = (os.environ.get("ACE_AUTO_SUBMIT") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


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
    # keep_alive=True: reuse one Chromium window across applies (no open/close flash).
    keep = (os.environ.get("ACE_BROWSER_KEEP_ALIVE") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return BrowserProfile(
        user_data_dir=str(profile_dir),
        profile_directory=settings.browser_profile_directory,
        headless=settings.browser_headless,
        keep_alive=keep,
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
    agent = Agent(
        task=task,
        llm=llm,
        browser_profile=_browser_profile(user_data_dir),
        use_vision=True,
        available_file_paths=file_paths,
        max_failures=8,
    )
    wall = int(timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS)
    return await asyncio.wait_for(
        agent.run(max_steps=max_steps),
        timeout=wall,
    )


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
    task = _TASK_TEMPLATE_LINKEDIN_CLEANUP.format(
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
            max_steps=40,
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


async def execute_application(
    job_url: str,
    tailored_pdf_path: str | Path,
    master_yaml_path: str | Path,
    *,
    auto_submit: bool | None = None,
    cleanup_linkedin_resume: bool | None = None,
    user_data_dir: Path | str | None = None,
) -> bool:
    """Drive browser-use through the apply form.

    Upload filename comes from ACE_APPLY_RESUME_FILENAME (via unique_apply_pdf).
    Per-apply LinkedIn cleanup is optional — prefer batch cleanup at cycle end.
    """
    pdf_path = Path(tailored_pdf_path).expanduser().resolve()
    master_path = Path(master_yaml_path).expanduser().resolve()
    pdf_abs = os.path.abspath(str(pdf_path))
    pdf_stem = pdf_path.stem
    pdf_filename = pdf_path.name

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Tailored PDF not found: {pdf_path}")
    if not master_path.is_file():
        raise FileNotFoundError(f"Master YAML not found: {master_path}")

    job_url = (job_url or "").strip()
    if not job_url:
        raise ValueError("job_url is required")

    do_submit = _auto_submit_enabled() if auto_submit is None else bool(auto_submit)
    # One-offs: cleanup LinkedIn after success. Parallel pool passes False and
    # runs batch cleanup once per cycle instead.
    do_cleanup = (
        cleanup_linkedin_resume
        if cleanup_linkedin_resume is not None
        else ("linkedin.com" in job_url.lower())
    )
    master_preview = master_path.read_text(encoding="utf-8")
    form_answers = _application_answers_block(master_path)
    lessons = load_lessons()
    max_steps, timeout_sec = _step_budget(job_url)

    if do_submit:
        task = _TASK_TEMPLATE_AUTO.format(
            job_url=job_url,
            pdf_abs_path=pdf_abs,
            pdf_stem=pdf_stem,
            pdf_filename=pdf_filename,
            submitted_token=SUBMITTED_TOKEN,
        )
        win_token = SUBMITTED_TOKEN
    else:
        task = _TASK_TEMPLATE_REVIEW.format(
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
    # Prefer form answers + trim huge master dump so phone/PIN/notice stay visible
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
