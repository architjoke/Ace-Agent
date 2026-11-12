"""Track-aware job sourcing with python-jobspy (India-first)."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from jobspy import scrape_jobs

logger = logging.getLogger(__name__)

def _output_dir() -> Path:
    try:
        from src.core.settings import get_settings

        return Path(get_settings().output_dir)
    except Exception:
        from src.core.config import OUTPUT_DIR as _OUT

        return _OUT


from src.core.config import OUTPUT_DIR  # noqa: E402 — fallback constant

OUTPUT_CSV = OUTPUT_DIR / "jobs.csv"  # recomputed at write time via _jobs_csv()


def _jobs_csv() -> Path:
    return _output_dir() / "jobs.csv"

TRACK_QUERIES: dict[str, str] = {
    "SDE": (
        "Software Development Engineer OR Software Engineer Intern OR "
        "Backend Developer Intern OR Full Stack Developer Intern"
    ),
    "AI": (
        "AI Engineer Intern OR AI Software Engineer OR Applied AI Engineer OR "
        "Generative AI Engineer Intern"
    ),
    "ML": (
        "Machine Learning Engineer Intern OR ML Engineer Intern OR "
        "Deep Learning Engineer Intern OR Computer Vision Engineer Intern"
    ),
}

DEFAULT_LOCATION = "India"
DEFAULT_COUNTRY_INDEED = "India"
DEFAULT_SITES = ["linkedin", "naukri"]

# Aggregator listing pages that usually cannot host a real application form.
# Glassdoor stays discovery-blocked (Cloudflare); external job_url_direct still OK.
_AGGREGATOR_HOSTS = (
    "glassdoor.",
    "glassdoor.co",
    "indeed.glassdoor",
)

# Scam / never-apply boards (hard drop even if JobSpy returns them).
_BLOCKED_BOARD_HOSTS = (
    "internshala.com",
    "internshala.",
)

# Crowded Soft Apply surfaces — skip unless ACE_ALLOW_INDEED=1.
_INDEED_SOFT_HOSTS = (
    "indeed.com",
    "indeed.co",
    "smartapply.indeed",
)

# Redirect boards — listing may not host the form; prefer job_url_direct / bridge.
_REDIRECT_BOARD_HOSTS = (
    "weworkremotely.com",
    "himalayas.app",
    "hiring.cafe",
    "hiringcafe.com",
)

# Prefer company ATS / career boards over aggregator Easy Apply.
# Redirect boards (Hiring Cafe, WWR, …) stay out of this list so job_url_direct
# Greenhouse/Lever/Ashby always outranks the aggregator listing itself.
_CAREER_ATS_HOST_HINTS = (
    "greenhouse.io",
    "boards.greenhouse",
    "lever.co",
    "jobs.lever",
    "ashbyhq.com",
    "myworkdayjobs.com",
    "workday.com",
    "jobs.ashby",
    "smartrecruiters.com",
    "jobvite.com",
    "icims.com",
    "bamboohr.com",
    # Bridged via career_site playbook (Eightfold / Zoho / careers.*).
    "eightfold.ai",
    "zohorecruit.",
    "zoho.com",
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def is_blocked_board(url: str) -> bool:
    """True for scam/never-apply hosts (Internshala, …)."""
    h = _host(url)
    blob = f"{h} {(url or '').lower()}"
    return any(marker in blob for marker in _BLOCKED_BOARD_HOSTS)


def is_indeed_soft_url(url: str) -> bool:
    """Indeed Soft / SmartApply listing — crowded + captcha-prone."""
    h = _host(url)
    return any(marker in h for marker in _INDEED_SOFT_HOSTS)


def indeed_applies_allowed() -> bool:
    """Opt-in Indeed Soft Apply (default off — prefer career-site directs)."""
    import os

    from src.core.utils import env_truthy

    return env_truthy("ACE_ALLOW_INDEED", "0")


def is_aggregator_listing(url: str) -> bool:
    """Glassdoor-class walls — skip browser unless a direct ATS URL exists."""
    h = _host(url)
    return any(marker in h for marker in _AGGREGATOR_HOSTS)


def is_redirect_board(url: str) -> bool:
    """Boards that usually click-through to Greenhouse/Lever/mailto (bridge playbook)."""
    h = _host(url)
    return any(marker in h for marker in _REDIRECT_BOARD_HOSTS)


def is_career_ats_url(url: str) -> bool:
    """Company career / ATS hosts that usually autofill from an uploaded resume."""
    h = _host(url)
    if any(marker in h for marker in _CAREER_ATS_HOST_HINTS):
        return True
    # Custom career microsites (often Zoho/Eightfold wrappers).
    return h.startswith("careers.") and "linkedin.com" not in h


def apply_url_priority(url: str) -> int:
    """Lower = better. Career ATS first; Indeed Soft last; blocked = worst."""
    u = (url or "").strip()
    if not u or is_blocked_board(u):
        return 100
    if is_career_ats_url(u):
        return 0
    if is_indeed_soft_url(u):
        return 80
    # Hiring Cafe / WWR / Himalayas — bridge to ATS; worse than direct career hosts.
    if is_redirect_board(u):
        return 30
    h = _host(u)
    if "linkedin.com" in h:
        return 40
    if "naukri.com" in h:
        return 35
    return 20


def is_applyable_url(url: str) -> bool:
    """True when the URL can plausibly open an apply / Easy Apply flow."""
    u = (url or "").strip()
    if not re.match(r"https?://", u, re.I):
        return False
    if is_blocked_board(u):
        return False
    if is_aggregator_listing(u):
        return False
    if is_indeed_soft_url(u) and not indeed_applies_allowed():
        return False
    return True


def pick_apply_url(row: dict | pd.Series) -> str:
    """Prefer company/ATS direct apply URL; else LinkedIn listing (Indeed last)."""
    direct = str(row.get("job_url_direct") or "").strip()
    listing = str(row.get("job_url") or "").strip()
    candidates = [u for u in (direct, listing) if u]
    # Prefer career ATS / direct over crowded Soft Apply listings.
    candidates.sort(key=apply_url_priority)
    for url in candidates:
        if is_applyable_url(url):
            return url
    # Last resort: listing even if aggregator (caller may Manual_Action)
    return listing or direct


def _normalize_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    """Attach apply_url; drop blocked / glassdoor-only / Indeed-soft (unless allowed)."""
    if jobs.empty:
        return jobs
    apply_urls: list[str] = []
    keep: list[bool] = []
    for _, row in jobs.iterrows():
        apply = pick_apply_url(row)
        apply_urls.append(apply)
        listing = str(row.get("job_url") or "")
        direct = str(row.get("job_url_direct") or "")
        if is_blocked_board(apply) or is_blocked_board(listing) or is_blocked_board(direct):
            keep.append(False)
            continue
        # Drop pure Glassdoor walls with no company apply link.
        if is_aggregator_listing(listing) and not is_applyable_url(direct):
            keep.append(False)
        elif is_indeed_soft_url(apply) and not indeed_applies_allowed():
            keep.append(False)
        else:
            keep.append(bool(apply) and not is_blocked_board(apply))
    out = jobs.loc[keep].copy()
    out["apply_url"] = [u for u, k in zip(apply_urls, keep) if k]
    # Prefer browsing the apply_url for the rest of the pipeline.
    out["job_url"] = out["apply_url"]
    if not out.empty:
        out["_prio"] = out["job_url"].map(apply_url_priority)
        out = out.sort_values("_prio", kind="stable").drop(columns=["_prio"])
    return out.reset_index(drop=True)


def _sites_from_settings() -> list[str]:
    try:
        from src.core.settings import get_settings

        sites = [str(s).strip().lower() for s in (get_settings().sites or []) if s]
        # Never scrape Glassdoor for autonomous apply — Cloudflare walls.
        # Prefer Naukri via JobSpy; WWR/Wellfound/HiringCafe via jobdrop (extra).
        sites = [
            s
            for s in sites
            if s
            and s
            not in {
                "glassdoor",
                "weworkremotely",
                "wellfound",
                "hiring_cafe",
                "himalayas",
                "yc_waas",
            }
        ]
        return sites or list(DEFAULT_SITES)
    except Exception:
        return list(DEFAULT_SITES)


def fetch_jobs(
    keywords: str,
    location: str = DEFAULT_LOCATION,
    max_results: int = 10,
    *,
    country_indeed: str = DEFAULT_COUNTRY_INDEED,
    track: str | None = None,
    sites: list[str] | None = None,
) -> pd.DataFrame:
    """Scrape career boards for one search term (LinkedIn / Indeed by default)."""
    site_names = sites or _sites_from_settings()
    logger.info(
        "Scraping track=%s sites=%s keywords=%r location=%r max=%s",
        track or "custom",
        site_names,
        keywords,
        location,
        max_results,
    )
    from src.core.network import discovery_proxies_arg

    proxy_list = discovery_proxies_arg()
    scrape_kwargs: dict = {
        "site_name": site_names,
        "search_term": keywords,
        "location": location,
        "results_wanted": max_results,
        "country_indeed": country_indeed,
        "fetch_description": True,
    }
    if proxy_list:
        scrape_kwargs["proxies"] = proxy_list
        logger.info("JobSpy using %s proxy endpoint(s)", len(proxy_list))
    jobs = scrape_jobs(**scrape_kwargs)

    if jobs is None or jobs.empty:
        return pd.DataFrame()

    has_url = jobs["job_url"].fillna("").astype(str).str.match(r"https?://", na=False)
    has_description = jobs["description"].fillna("").astype(str).str.strip().ne("")
    filtered = jobs.loc[has_url & has_description].copy()
    if track:
        filtered["track"] = track
    filtered = _normalize_frame(filtered)
    return filtered.reset_index(drop=True)


def fetch_jobs_by_tracks(
    location: str = DEFAULT_LOCATION,
    max_results_per_track: int = 10,
    *,
    country_indeed: str = DEFAULT_COUNTRY_INDEED,
    tracks: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run JobSpy per track in killable processes; merge partial results."""
    from src.core.discovery_runners import (
        extra_timeout_s,
        overall_timeout_s,
        records_to_frame,
        run_parallel_deadlines,
        track_timeout_s,
    )

    queries = tracks or TRACK_QUERIES
    sites = _sites_from_settings()
    frames: list[pd.DataFrame] = []
    hunt_started = time.monotonic()
    wall = overall_timeout_s()
    per_track = track_timeout_s()

    jobs = [
        (
            "jobspy_track",
            {
                "keywords": query,
                "location": location,
                "max_results": max_results_per_track,
                "country_indeed": country_indeed,
                "track": track_name,
                "sites": sites,
            },
            per_track,
            f"jobspy:{track_name}",
        )
        for track_name, query in queries.items()
    ]
    for result in run_parallel_deadlines(jobs, overall_timeout_s=wall):
        if result.timed_out:
            logger.warning(
                "Track %s timed out — continuing with partial discovery",
                result.label,
            )
            continue
        if not result.ok:
            if result.error:
                logger.warning(
                    "Track %s scrape failed: %s",
                    result.label,
                    (result.error or "")[:240],
                )
            continue
        frame = records_to_frame(result.records)
        if frame is not None and not frame.empty:
            frames.append(frame)
            logger.info(
                "Track %s returned %s applyable jobs",
                result.label,
                len(frame),
            )

    # Jobdrop + Himalayas / YC — process-isolated; never block apply on hang.
    remaining = wall - (time.monotonic() - hunt_started)
    if remaining > 5.0:
        try:
            from src.core.discovery_extra import fetch_extra_jobs_by_track

            extra_budget = min(extra_timeout_s(), remaining)
            extra = fetch_extra_jobs_by_track(
                queries,
                location=location,
                max_results_per_track=max(4, max_results_per_track // 2),
                country_indeed=country_indeed,
                overall_deadline_s=extra_budget,
            )
            if extra is not None and not extra.empty:
                extra = _normalize_frame(extra)
                frames.append(extra)
                logger.info("Extra discovery sources returned %s jobs", len(extra))
        except Exception:
            logger.exception("Extra discovery sources failed — continuing with JobSpy")
    else:
        logger.warning(
            "Overall discovery budget exhausted — skipping extra sources"
        )

    if not frames:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(frames, ignore_index=True)
        if "job_url" in combined.columns:
            combined = combined.drop_duplicates(subset=["job_url"], keep="first")
        combined = combined.reset_index(drop=True)

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "jobs.csv"
    combined.to_csv(csv_path, index=False)
    logger.info("Combined discovery wrote %s jobs → %s", len(combined), csv_path)
    return combined
