"""Staged parallel pipeline: triage → tailor → apply-queue → browser workers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.core.browser_pool import (
    run_browser_pool,
    worker_profile_dir,
)
from src.core.discovery import (
    fetch_jobs_by_tracks,
    is_aggregator_listing,
    is_applyable_url,
    pick_apply_url,
)
from src.core.evaluate import evaluate_fit
from src.core.executor import (
    AlreadyAppliedError,
    ApplicationBlockedError,
    ApplicationTimeoutError,
    APPLY_RESUME_FILENAME,
    ExternalPortalLoadError,
    JobClosedError,
    RequiresHumanInterventionError,
    SUBMITTED_TOKEN,
    cleanup_linkedin_uploaded_resume,
    execute_application,
    unique_apply_pdf,
)
from src.core.fallback import generate_manual_package
from src.core.memory import (
    has_applied,
    has_recent_company_role_apply,
    has_successful_apply,
    init_db,
    log_application,
)
from src.core.queue import enqueue_apply, init_queue, queue_counts
from src.core.settings import reload_settings
from src.core.config import PROJECT_ROOT
from src.core.tailor import optimize_resume

logger = logging.getLogger("ace_agent")

REPO_ROOT = PROJECT_ROOT


def _cleanup_tailored_artifacts(company: str, pdf_path: Path | str | None = None) -> None:
    """After verified submit: archive PDF under Submitted/, release build folder."""
    from src.core.output_artifacts import release_company_build

    release_company_build(
        company, pdf_path=pdf_path, archive=True, reason="auto_success"
    )


def triage_job(
    job: Any,
    master_text: str,
    min_fit: int,
) -> dict[str, Any] | None:
    """Fit triage. Returns enriched job dict or None if dropped/skipped/failed."""
    from src.core.utils import minify_jd

    company = str(job.get("company") or "Unknown").strip() or "Unknown"
    title = str(job.get("title") or "").strip()
    track = str(job.get("track") or "").strip()
    job_url = pick_apply_url(job)
    description = minify_jd(str(job.get("description") or "").strip())

    logger.info(
        "Triage [%s] %s @ %s → %s",
        track or "?",
        title or "(untitled)",
        company,
        job_url[:140],
    )

    if not job_url or not description:
        logger.warning("Skipping %s — missing job_url or description", company)
        return None

    from src.core.discovery import (
        indeed_applies_allowed,
        is_blocked_board,
        is_indeed_soft_url,
    )

    if is_blocked_board(job_url) or is_blocked_board(str(job.get("job_url") or "")):
        logger.info("Skipping %s — blocked board (Internshala/scam) %s", company, job_url[:120])
        return None

    # LinkedIn wrappers that advertise Internshala in the JD — drop before tailor.
    desc_l = description.lower()
    if "internshala.com" in desc_l or "internshala." in desc_l:
        logger.info(
            "Skipping %s — JD funnels to Internshala (blocked board)",
            company,
        )
        return None

    if is_indeed_soft_url(job_url) and not indeed_applies_allowed():
        logger.info(
            "Skipping %s — Indeed Soft Apply off (prefer career ATS); set ACE_ALLOW_INDEED=1 to enable",
            company,
        )
        return None

    if has_applied(job_url):
        logger.info("Skipping %s — already in memory (%s)", company, job_url[:120])
        return None

    if has_recent_company_role_apply(company, title):
        logger.info(
            "Skipping %s — recent submitted company+role (%s)",
            company,
            (title or "(any role)")[:120],
        )
        return None

    try:
        fit = evaluate_fit(description, master_text, min_score=min_fit)
    except Exception as exc:
        logger.exception("Fit triage crashed for %s — continuing", company)
        log_application(
            company,
            job_url,
            track,
            "FAILED",
            error_log=f"fit_triage_crash: {type(exc).__name__}: {exc}",
            jd_text=description,
            job_title=title,
        )
        return None

    if not fit["passed"]:
        logger.info(
            "Dropping %s (fit=%s < %s, rule=%s): %s",
            company,
            fit["fit_score"],
            min_fit,
            fit.get("rule_broken") or "skill_fit",
            fit["reason"][:300],
        )
        return None

    logger.info("Fit passed (%s): %s", fit["fit_score"], fit["reason"][:240])
    job_track = track or fit.get("recommended_track") or ""
    citizenship_review = bool(fit.get("citizenship_review"))
    if citizenship_review:
        logger.info("citizenship_review=1 for %s — soft-gated into queue", company)
    return {
        "company": company,
        "title": title,
        "track": job_track,
        "job_url": job_url,
        "description": description,
        "job_url_direct": str(job.get("job_url_direct") or ""),
        "raw": job,
        "company_provided_salary": bool(fit.get("company_provided_salary")),
        "salary_min": float(fit.get("salary_min") or 0),
        "salary_max": float(fit.get("salary_max") or 0),
        "salary_format": str(fit.get("salary_format") or "ANNUAL"),
        "includes_equity_or_bonus": bool(fit.get("includes_equity_or_bonus")),
        "jd_salary_format": str(fit.get("jd_salary_format") or "ANNUAL"),
        "citizenship_review": citizenship_review,
    }


def tailor_job(
    triage: dict[str, Any],
    master_yaml: Path,
) -> dict[str, Any] | None:
    """Build tailored PDF. Returns dict with pdf_path + yaml_snapshot or None."""
    company = triage["company"]
    job_url = triage["job_url"]
    job_track = triage["track"]
    description = triage["description"]
    yaml_snapshot = ""
    try:
        pdf_path, yaml_snapshot = optimize_resume(
            description,
            master_yaml,
            company=company,
            job_track=job_track,
            job_title=str(triage.get("title") or ""),
        )
        from src.core.salary_engine import compensation_from_fit, embed_compensation

        yaml_snapshot = embed_compensation(
            yaml_snapshot, compensation_from_fit(triage)
        )
        try:
            from src.core.pitch import embed_pitch, generate_pitch

            pitch = generate_pitch(
                company=company,
                job_title=str(triage.get("title") or ""),
                jd_text=description,
            )
            yaml_snapshot = embed_pitch(yaml_snapshot, pitch)
        except Exception:
            logger.debug("Pitch embed skipped for %s", company, exc_info=True)
        logger.info("Tailored PDF ready (%s): %s", job_track or "GENERAL", pdf_path)
        apply_pdf = unique_apply_pdf(pdf_path, company)
        return {
            **triage,
            "pdf_path": str(Path(apply_pdf).resolve()),
            "yaml_snapshot": yaml_snapshot,
        }
    except Exception as exc:
        logger.exception("Tailoring failed for %s — continuing", company)
        log_application(
            company,
            job_url,
            job_track,
            "FAILED",
            error_log=f"tailor_failed: {type(exc).__name__}: {exc}",
            jd_text=description,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=str(triage.get("title") or ""),
        )
        return None


async def apply_job(
    *,
    company: str,
    job_url: str,
    track: str,
    jd_text: str,
    pdf_path: str,
    yaml_snapshot: str,
    master_yaml: Path,
    user_data_dir: Path | str | None = None,
    cleanup_linkedin_resume: bool = False,
    job_url_direct: str = "",
    job_title: str = "",
) -> str:
    """Browser-only apply. Returns success|fallback|failed|already_applied|timeout|requires_human|job_closed."""
    skip_browser = is_aggregator_listing(job_url) and not is_applyable_url(
        job_url_direct or ""
    )
    title = (job_title or "").strip()

    # Phase 1A: DB fast-fail before Chromium.
    if has_successful_apply(job_url):
        reason = "db_success_url"
    elif has_recent_company_role_apply(company, title):
        reason = "db_success_company_role_30d"
    else:
        reason = ""
    if reason:
        logger.info("Skipping %s — already SUCCESS in memory (%s)", company, reason)
        log_application(
            company,
            job_url,
            track,
            "ALREADY_APPLIED",
            error_log=reason,
            jd_text=jd_text,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=title,
        )
        return "already_applied"

    try:
        if skip_browser:
            raise ApplicationBlockedError(
                f"No applyable career URL (aggregator listing only): {job_url[:120]}"
            )
        ok = await execute_application(
            job_url,
            pdf_path,
            master_yaml,
            cleanup_linkedin_resume=cleanup_linkedin_resume,
            user_data_dir=user_data_dir,
            yaml_snapshot=yaml_snapshot,
            jd_text=jd_text,
            company=company,
            job_title=title,
        )
        if ok:
            logger.info(
                "[✓] Auto-submitted application for %s (%s)",
                company,
                SUBMITTED_TOKEN,
            )
            log_application(
                company,
                job_url,
                track,
                "SUCCESS",
                error_log=SUBMITTED_TOKEN,
                jd_text=jd_text,
                tailored_yaml_snapshot=yaml_snapshot,
                job_title=title,
            )
            _cleanup_tailored_artifacts(company, pdf_path)
            return "success"
        raise RuntimeError("execute_application returned False")
    except AlreadyAppliedError as exc:
        logger.info("[skip] Already applied %s — %s", company, exc)
        log_application(
            company,
            job_url,
            track,
            "ALREADY_APPLIED",
            error_log=str(exc),
            jd_text=jd_text,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=title,
        )
        return "already_applied"
    except JobClosedError as exc:
        logger.info("[skip] Job closed/missing %s — %s", company, exc)
        log_application(
            company,
            job_url,
            track,
            "JOB_CLOSED",
            error_log=str(exc),
            jd_text=jd_text,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=title,
        )
        return "job_closed"
    except RequiresHumanInterventionError as exc:
        logger.warning(
            "[human] %s needs intervention — skipping Manual_Action (%s)",
            company,
            exc,
        )
        human_status = (
            "REQUIRES_HUMAN_CITIZENSHIP_REVIEW"
            if "ace:citizenship_review=1" in (yaml_snapshot or "")
            else "REQUIRES_HUMAN_INTERVENTION"
        )
        log_application(
            company,
            job_url,
            track,
            human_status,
            error_log=f"{type(exc).__name__}: {exc}",
            jd_text=jd_text,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=title,
        )
        return "requires_human"
    except ExternalPortalLoadError as exc:
        logger.warning(
            "[portal] %s external portal/DNS failed — skip without CDP (%s)",
            company,
            exc,
        )
        log_application(
            company,
            job_url,
            track,
            "FAILED_TIMEOUT_OR_STUCK",
            error_log=f"FAILED_EXTERNAL_LOAD: {exc}",
            jd_text=jd_text,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=title,
        )
        return "timeout"
    except ApplicationTimeoutError as exc:
        logger.warning(
            "[timeout] %s stuck/timeout — skipping Manual_Action (%s)",
            company,
            exc,
        )
        log_application(
            company,
            job_url,
            track,
            "FAILED_TIMEOUT_OR_STUCK",
            error_log=f"{type(exc).__name__}: {exc}",
            jd_text=jd_text,
            tailored_yaml_snapshot=yaml_snapshot,
            job_title=title,
        )
        return "timeout"
    except (ApplicationBlockedError, Exception) as exc:
        logger.warning(
            "[!] Could not auto-submit %s — Manual_Action package. (%s: %s)",
            company,
            type(exc).__name__,
            exc,
        )
        # apply_job swallows; record CF/captcha here so the pool circuit can trip.
        try:
            from src.core.ban_armor import get_captcha_breaker
            from src.core.host_limiter import host_bucket_key

            err_l = f"{type(exc).__name__} {exc}".lower()
            breaker = get_captcha_breaker()
            if breaker.is_cf_block_message(err_l):
                breaker.record_block(
                    host_bucket_key(job_url),
                    reason=str(exc)[:160],
                )
        except Exception:
            logger.debug("captcha circuit record failed", exc_info=True)
        try:
            package_dir = generate_manual_package(
                company=company,
                job_url=job_url,
                jd=jd_text,
                tailored_pdf_path=pdf_path,
                master_yaml_path=master_yaml,
            )
            logger.info("Fallback package ready for %s → %s", company, package_dir)
            log_application(
                company,
                job_url,
                track,
                "MANUAL_ACTION",
                error_log=f"{type(exc).__name__}: {exc}",
                jd_text=jd_text,
                tailored_yaml_snapshot=yaml_snapshot,
                job_title=title,
            )
            return "fallback"
        except Exception as fallback_exc:
            logger.exception(
                "Fallback package generation also failed for %s — continuing",
                company,
            )
            log_application(
                company,
                job_url,
                track,
                "FAILED",
                error_log=(
                    f"apply={type(exc).__name__}: {exc}; "
                    f"fallback={type(fallback_exc).__name__}: {fallback_exc}"
                ),
                jd_text=jd_text,
                tailored_yaml_snapshot=yaml_snapshot,
                job_title=title,
            )
            return "failed"


async def _produce_one(
    job: Any,
    master_yaml: Path,
    master_text: str,
    min_fit: int,
    produce_sem: asyncio.Semaphore,
    groq_sem: asyncio.Semaphore,
    counts: dict[str, int],
    target: int,
    target_lock: asyncio.Lock,
    stop_produce: asyncio.Event,
) -> None:
    if stop_produce.is_set():
        return
    async with produce_sem:
        if stop_produce.is_set():
            return
        triage = await asyncio.to_thread(triage_job, job, master_text, min_fit)
        if triage is None:
            counts["dropped"] = counts.get("dropped", 0) + 1
            return
        async with groq_sem:
            tailored = await asyncio.to_thread(tailor_job, triage, master_yaml)
        if tailored is None:
            counts["failed"] = counts.get("failed", 0) + 1
            return
        snap = tailored.get("yaml_snapshot") or ""
        if tailored.get("citizenship_review") and "ace:citizenship_review=1" not in snap:
            snap = f"# ace:citizenship_review=1\n{snap}"
        enqueue_apply(
            company=tailored["company"],
            job_url=tailored["job_url"],
            track=tailored["track"],
            jd_text=tailored["description"],
            pdf_path=tailored["pdf_path"],
            yaml_snapshot=snap,
            job_title=str(tailored.get("title") or ""),
        )
        counts["queued"] = counts.get("queued", 0) + 1
        logger.info(
            "Queued apply for %s (queue READY≈%s)",
            tailored["company"],
            queue_counts().get("READY", 0),
        )
        if target > 0:
            async with target_lock:
                if counts.get("queued", 0) >= target:
                    stop_produce.set()


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return max(1, int((os.environ.get(name) or str(default)).strip()))
    except ValueError:
        return default


async def _wait_queue_warm(
    *,
    threshold: int,
    producers_done: asyncio.Event,
    poll_sec: float = 0.25,
) -> None:
    """Block until READY≥threshold or producers finished (avoids eternal wait)."""
    while True:
        if queue_counts().get("READY", 0) >= threshold:
            logger.info("Queue warm — READY≥%s; starting browser pool", threshold)
            return
        if producers_done.is_set():
            logger.info(
                "Producers done before warm threshold (%s); starting browser pool",
                threshold,
            )
            return
        await asyncio.sleep(poll_sec)


async def run_parallel(
    location: str | None = None,
    max_results_per_track: int | None = None,
    master_yaml: Path | None = None,
    target_companies: int | None = None,
) -> int:
    """Discover → parallel triage/tailor → N browser workers → batch LinkedIn cleanup."""
    settings = reload_settings()
    init_db()
    init_queue()

    master_yaml = Path(master_yaml or settings.master_yaml).expanduser().resolve()
    if not master_yaml.is_file():
        logger.error(
            "Master resume YAML not found at %s. Create it before running.",
            master_yaml,
        )
        return 1

    master_text = master_yaml.read_text(encoding="utf-8")
    loc = location or settings.primary_location
    per_track = (
        max_results_per_track
        if max_results_per_track is not None
        else settings.max_results_per_track
    )
    tracks = settings.tracks or None
    min_fit = settings.minimum_fit_score
    target = int(target_companies) if target_companies and target_companies > 0 else 0
    produce_n = _env_int("ACE_PRODUCER_CONCURRENCY", 15)
    # Default 2: start applies while more drafts finish (avoids Ollama starve gate).
    warm_n = _env_int("ACE_QUEUE_WARM_THRESHOLD", 2)
    browser_workers = max(1, int(settings.browser_workers))
    groq_slots = max(1, int(settings.groq_concurrency))

    logger.info(
        "Parallel hunt near %s — %s/track fit≥%s | produce=%s groq=%s browsers=%s warm≥%s%s",
        loc,
        per_track,
        min_fit,
        produce_n,
        groq_slots,
        browser_workers,
        warm_n,
        f", stop after {target} queued" if target else "",
    )

    try:
        jobs = await asyncio.to_thread(
            fetch_jobs_by_tracks,
            location=loc,
            max_results_per_track=per_track,
            country_indeed=settings.country_indeed,
            tracks=tracks,
        )
    except Exception:
        logger.exception("Job discovery failed")
        return 1

    if jobs.empty:
        logger.warning("No jobs found matching the track filters")
        return 0

    logger.info("Discovered %s unique jobs — parallel triage/tailor → apply queue", len(jobs))

    counts: dict[str, int] = {
        "success": 0,
        "fallback": 0,
        "dropped": 0,
        "failed": 0,
        "skipped": 0,
        "queued": 0,
    }
    produce_sem = asyncio.Semaphore(produce_n)
    groq_sem = asyncio.Semaphore(groq_slots)
    stop_produce = asyncio.Event()
    producers_done = asyncio.Event()
    target_lock = asyncio.Lock()

    async def _apply_wrapper(**kwargs: Any) -> str:
        outcome = await apply_job(**kwargs)
        counts[outcome] = counts.get(outcome, 0) + 1
        return outcome

    async def _start_pool_when_warm() -> dict:
        await _wait_queue_warm(threshold=warm_n, producers_done=producers_done)
        return await run_browser_pool(
            master_yaml=master_yaml,
            apply_fn=_apply_wrapper,
            n_workers=browser_workers,
            producers_done=producers_done,
        )

    pool_task = asyncio.create_task(_start_pool_when_warm(), name="browser-pool")

    produce_tasks = [
        asyncio.create_task(
            _produce_one(
                job,
                master_yaml,
                master_text,
                min_fit,
                produce_sem,
                groq_sem,
                counts,
                target,
                target_lock,
                stop_produce,
            )
        )
        for _, job in jobs.iterrows()
    ]
    produce_results = await asyncio.gather(*produce_tasks, return_exceptions=True)
    for result in produce_results:
        if isinstance(result, Exception):
            logger.error("Producer task crashed: %s", result, exc_info=result)
            counts["failed"] = counts.get("failed", 0) + 1
    producers_done.set()
    logger.info(
        "Producers done — queued=%s dropped=%s failed=%s; draining apply queue…",
        counts.get("queued", 0),
        counts.get("dropped", 0),
        counts.get("failed", 0),
    )

    pool_result = await pool_task
    if isinstance(pool_result, dict):
        logger.info("Browser pool finished: %s", pool_result)

    # Batch LinkedIn resume cleanup once per cycle (not after every apply).
    if counts.get("success", 0) > 0:
        logger.info("Batch LinkedIn resume cleanup (%s)…", APPLY_RESUME_FILENAME)
        try:
            from src.core.settings import get_settings as _gs

            cleanup_profile = worker_profile_dir("w1")
            out_dir = Path(_gs().output_dir)
            sample = out_dir / APPLY_RESUME_FILENAME
            pdf_hint = sample if sample.is_file() else out_dir / APPLY_RESUME_FILENAME
            await cleanup_linkedin_uploaded_resume(
                pdf_hint,
                user_data_dir=cleanup_profile,
            )
        except Exception:
            logger.exception("Batch LinkedIn resume cleanup failed")

    logger.info(
        "Done — success=%s fallback=%s dropped=%s failed=%s queued=%s total_jobs=%s",
        counts.get("success", 0),
        counts.get("fallback", 0),
        counts.get("dropped", 0),
        counts.get("failed", 0),
        counts.get("queued", 0),
        len(jobs),
    )
    return 0
