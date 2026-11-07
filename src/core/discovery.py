"""Track-aware job sourcing with python-jobspy (India-first)."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_SITES = ["linkedin", "indeed"]

# Aggregator listing pages that usually cannot host a real application form.
_AGGREGATOR_HOSTS = (
    "glassdoor.",
    "glassdoor.co",
    "indeed.glassdoor",
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def is_aggregator_listing(url: str) -> bool:
    h = _host(url)
    return any(marker in h for marker in _AGGREGATOR_HOSTS)


def is_applyable_url(url: str) -> bool:
    """True when the URL can plausibly open an apply / Easy Apply flow."""
    u = (url or "").strip()
    if not re.match(r"https?://", u, re.I):
        return False
    if is_aggregator_listing(u):
        return False
    return True


def pick_apply_url(row: dict | pd.Series) -> str:
    """Prefer company/ATS direct apply URL; else LinkedIn/Indeed listing."""
    direct = str(row.get("job_url_direct") or "").strip()
    listing = str(row.get("job_url") or "").strip()
    if is_applyable_url(direct):
        return direct
    if is_applyable_url(listing):
        return listing
    # Last resort: listing even if aggregator (caller may Manual_Action)
    return listing or direct


def _normalize_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    """Attach apply_url; drop glassdoor-only rows with no usable direct link."""
    if jobs.empty:
        return jobs
    apply_urls: list[str] = []
    keep: list[bool] = []
    for _, row in jobs.iterrows():
        apply = pick_apply_url(row)
        apply_urls.append(apply)
        listing = str(row.get("job_url") or "")
        direct = str(row.get("job_url_direct") or "")
        # Drop pure Glassdoor walls with no company apply link.
        if is_aggregator_listing(listing) and not is_applyable_url(direct):
            keep.append(False)
        else:
            keep.append(bool(apply))
    out = jobs.loc[keep].copy()
    out["apply_url"] = [u for u, k in zip(apply_urls, keep) if k]
    # Prefer browsing the apply_url for the rest of the pipeline.
    out["job_url"] = out["apply_url"]
    return out.reset_index(drop=True)


def _sites_from_settings() -> list[str]:
    try:
        from src.core.settings import get_settings

        sites = [str(s).strip().lower() for s in (get_settings().sites or []) if s]
        # Never scrape Glassdoor for autonomous apply — Cloudflare walls.
        sites = [s for s in sites if s and s != "glassdoor"]
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
    """Run JobSpy per track in parallel, prefer direct apply URLs, write jobs.csv."""
    queries = tracks or TRACK_QUERIES
    sites = _sites_from_settings()
    frames: list[pd.DataFrame] = []

    def _one(track_name: str, query: str) -> pd.DataFrame:
        return fetch_jobs(
            keywords=query,
            location=location,
            max_results=max_results_per_track,
            country_indeed=country_indeed,
            track=track_name,
            sites=sites,
        )

    max_workers = max(1, min(4, len(queries)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_one, track_name, query): track_name
            for track_name, query in queries.items()
        }
        for fut in as_completed(futures):
            track_name = futures[fut]
            try:
                frame = fut.result()
            except Exception:
                logger.exception("Track %s scrape failed — continuing", track_name)
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
                logger.info(
                    "Track %s returned %s applyable jobs", track_name, len(frame)
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
