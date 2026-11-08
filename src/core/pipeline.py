"""Staged parallel pipeline: triage → tailor → apply-queue → browser workers."""

from __future__ import annotations

import asyncio
import logging
import shutil
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
    SUBMITTED_TOKEN,
    cleanup_linkedin_uploaded_resume,
    execute_application,
    unique_apply_pdf,
)
from src.core.fallback import generate_manual_package
from src.core.memory import (
    has_applied,
    has_successful_apply,
    has_successful_company_apply,
    init_db,
    log_application,
)
from src.core.queue import enqueue_apply, init_queue, queue_counts
from src.core.settings import get_settings, reload_settings
from src.core.config import OUTPUT_DIR, PROJECT_ROOT
from src.core.tailor import _safe_dir_name, optimize_resume

logger = logging.getLogger("ace_agent")

REPO_ROOT = PROJECT_ROOT


def _cleanup_tailored_artifacts(company: str, pdf_path: Path | str | None = None) -> None:
    """After verified submit: archive under output/Submitted/, remove work folder."""
    from src.core.settings import get_settings as _gs

    out = Path(_gs().output_dir)
    name = _safe_dir_name(company)
    company_dir = out / name
    submitted_dir = out / "Submitted" / name
    try:
        submitted_dir.mkdir(parents=True, exist_ok=True)
        src_pdf = Path(pdf_path) if pdf_path else None
        if src_pdf and src_pdf.is_file():
            dest = submitted_dir / src_pdf.name
            shutil.copy2(src_pdf, dest)
            logger.info("Archived submitted resume → %s", dest)
        elif company_dir.is_dir():
            for pdf in company_dir.glob("*.pdf"):
                shutil.copy2(pdf, submitted_dir / pdf.name)
                logger.info("Archived submitted resume → %s", submitted_dir / pdf.name)
                break
        if company_dir.is_dir():
            shutil.rmtree(company_dir)
            logger.info("Removed work folder after submit → %s", company_dir)
    except OSError:
        logger.exception("Could not archive/cleanup tailored package for %s", company)


def triage_job(
    job: Any,
    master_text: str,
    min_fit: int,
) -> dict[str, Any] | None:
    """Fit triage. Returns enriched job dict or None if dropped/skipped/failed."""
    company = str(job.get("company") or "Unknown").strip() or "Unknown"
    title = str(job.get("title") or "").strip()
    track = str(job.get("track") or "").strip()
    job_url = pick_apply_url(job)
    description = str(job.get("description") or "").strip()

    print(company)
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

    if has_applied(job_url):
        logger.info("Skipping %s — already in memory (%s)", company, job_url[:120])
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
    return {
        "company": company,
        "title": title,
        "track": job_track,
        "job_url": job_url,
        "description": description,
        "job_url_direct": str(job.get("job_url_direct") or ""),
        "raw": job,
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
        )
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
) -> str:
    """Browser-only apply. Returns success|fallback|failed|already_applied|timeout."""
    skip_browser = is_aggregator_listing(job_url) and not is_applyable_url(
        job_url_direct or ""
    )

    # Phase 1A: DB fast-fail before Chromium.
    if has_successful_apply(job_url):
        reason = "db_success_url"
    elif has_successful_company_apply(company):
        reason = "db_success_company_30d"
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
        )
        return "already_applied"
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
        )
        return "timeout"
    except (ApplicationBlockedError, Exception) as exc:
        logger.warning(
            "[!] Could not auto-submit %s — Manual_Action package. (%s: %s)",
            company,
            type(exc).__name__,
            exc,
        )
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
            )
            return "failed"


async def _produce_one(
    job: Any,
    master_yaml: Path,
    master_text: str,
    min_fit: int,
    tailor_sem: asyncio.Semaphore,
    groq_sem: asyncio.Semaphore,
    counts: dict[str, int],
    target: int,
    target_lock: asyncio.Lock,
    stop_produce: asyncio.Event,
) -> None:
    if stop_produce.is_set():
        return
    async with tailor_sem:
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
        enqueue_apply(
            company=tailored["company"],
            job_url=tailored["job_url"],
            track=tailored["track"],
            jd_text=tailored["description"],
            pdf_path=tailored["pdf_path"],
            yaml_snapshot=tailored.get("yaml_snapshot") or "",
        )
        counts["queued"] = counts.get("queued", 0) + 1
        logger.info(
            "Queued apply for %s (queue READY≈%s)",
            tailored["company"],
            queue_counts().get("READY", 0),
        )
        if target > 0:
            async with target_lock:
                appliedish = counts.get("queued", 0)
                if appliedish >= target:
                    stop_produce.set()


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
    tailor_workers = max(1, int(settings.tailor_workers))
    browser_workers = max(1, int(settings.browser_workers))
    groq_slots = max(1, int(settings.groq_concurrency))

    logger.info(
        "Parallel hunt near %s — %s/track fit≥%s | tailor=%s groq=%s browsers=%s%s",
        loc,
        per_track,
        min_fit,
        tailor_workers,
        groq_slots,
        browser_workers,
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
    tailor_sem = asyncio.Semaphore(tailor_workers)
    groq_sem = asyncio.Semaphore(groq_slots)
    stop_produce = asyncio.Event()
    producers_done = asyncio.Event()
    target_lock = asyncio.Lock()

    async def _apply_wrapper(**kwargs: Any) -> str:
        outcome = await apply_job(**kwargs)
        counts[outcome] = counts.get(outcome, 0) + 1
        return outcome

    pool_task = asyncio.create_task(
        run_browser_pool(
            master_yaml=master_yaml,
            apply_fn=_apply_wrapper,
            n_workers=browser_workers,
            producers_done=producers_done,
        ),
        name="browser-pool",
    )

    produce_tasks = [
        asyncio.create_task(
            _produce_one(
                job,
                master_yaml,
                master_text,
                min_fit,
                tailor_sem,
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
