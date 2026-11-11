"""Extra discovery sources: Jobdrop boards + custom Himalayas / YC scrapers."""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

import pandas as pd

logger = logging.getLogger(__name__)

# Jobdrop site keys (not covered well / at all by python-jobspy for our 7).
JOBDROP_SITES = ("weworkremotely", "wellfound", "hiring_cafe")
# These Jobdrop scrapers pull Camoufox (~1.3GB) on first use and can stall the
# entire hunt→apply cycle. Keep them opt-in until the binary is cached.
_JOBDROP_CAMOUFOX_SITES = frozenset({"wellfound", "hiring_cafe"})


def _env_truthy(name: str, default: str = "0") -> bool:
    return (os.environ.get(name, default) or default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _camoufox_cached() -> bool:
    """True when a local Camoufox install already exists (no download needed).

    Prefer official pkgman path (INSTALL_DIR / camoufox_path) — Windows layouts
    live under AppData\\Local\\camoufox\\camoufox\\Cache, not a camoufox*.exe glob.
    """
    from pathlib import Path

    try:
        from camoufox.pkgman import INSTALL_DIR, camoufox_path

        try:
            path = camoufox_path(download_if_missing=False)
            if path and Path(path).is_file():
                return True
        except Exception:
            pass
        install = Path(INSTALL_DIR)
        if install.is_dir():
            # version.json + any firefox/camoufox binary under Cache
            has_ver = (install / "version.json").is_file()
            exes = list(install.rglob("*.exe")) + list(install.rglob("camoufox*"))
            if has_ver and any(p.is_file() for p in exes):
                return True
    except Exception:
        pass
    try:
        home = Path.home()
        candidates = [
            home / ".cache" / "camoufox",
            home / "AppData" / "Local" / "camoufox",
            home / "AppData" / "Local" / "camoufox" / "camoufox",
            home / "AppData" / "Local" / "camoufox" / "camoufox" / "Cache",
        ]
        for base in candidates:
            if not base.is_dir():
                continue
            if (base / "version.json").is_file() and any(base.rglob("*.exe")):
                return True
            if any(base.rglob("camoufox*.exe")):
                return True
    except Exception:
        return False
    return False


def jobdrop_sites_for_hunt() -> list[str]:
    """Sites safe for the live hunt loop (never block cycle on Camoufox download).

    Prefer Hiring Cafe when Camoufox is ready — company career directs autofill
    from tailored resumes better than crowded Indeed Soft Apply.
    """
    raw = (os.environ.get("ACE_JOBDROP_SITES") or "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip() in JOBDROP_SITES]
    allow_cf = _env_truthy("ACE_JOBDROP_ALLOW_CAMOUFOX", "0") or _camoufox_cached()
    if allow_cf:
        # Hiring Cafe first — surfaces company career pages with resume autofill.
        ordered = ["hiring_cafe", "weworkremotely", "wellfound"]
        return [s for s in ordered if s in JOBDROP_SITES]
    logger.info(
        "Jobdrop: using weworkremotely only "
        "(set ACE_JOBDROP_ALLOW_CAMOUFOX=1 after Camoufox is installed for Hiring Cafe / Wellfound)"
    )
    return ["weworkremotely"]

_HIMALAYAS_SEARCH = "https://himalayas.app/jobs?q={q}"
_YC_JOBS = "https://www.workatastartup.com/jobs"


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def _std_frame(rows: list[dict[str, Any]], *, track: str | None) -> pd.DataFrame:
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    if track:
        df["track"] = track
    return df


def fetch_jobdrop_jobs(
    keywords: str,
    *,
    location: str = "India",
    max_results: int = 10,
    track: str | None = None,
    sites: list[str] | None = None,
    country_indeed: str = "India",
    per_site_timeout_s: float = 35.0,
) -> pd.DataFrame:
    """Scrape Jobdrop boards one site at a time (timeout-isolated, no Camoufox stall)."""
    try:
        from jobdrop import scrape_jobs
    except ImportError:
        logger.warning("jobdrop not installed — skip WWR/Wellfound/HiringCafe discovery")
        return _empty()

    wanted = sites or jobdrop_sites_for_hunt()
    site_names = [s for s in wanted if s in JOBDROP_SITES]
    if not _env_truthy("ACE_JOBDROP_ALLOW_CAMOUFOX", "0") and not _camoufox_cached():
        skipped = [s for s in site_names if s in _JOBDROP_CAMOUFOX_SITES]
        if skipped:
            logger.info("Skipping Jobdrop Camoufox sites until cached: %s", skipped)
        site_names = [s for s in site_names if s not in _JOBDROP_CAMOUFOX_SITES]
    if not site_names:
        return _empty()
    try:
        from src.core.network import discovery_proxies_arg

        proxies = discovery_proxies_arg()
    except Exception:
        proxies = None

    frames: list[pd.DataFrame] = []

    def _one(site: str) -> pd.DataFrame:
        kwargs: dict[str, Any] = {
            "site_name": [site],
            "search_term": keywords,
            "location": location,
            "results_wanted": max_results,
            "country_indeed": country_indeed,
            "is_remote": True,
            "verbose": 0,
        }
        if proxies:
            kwargs["proxies"] = proxies
        jobs = scrape_jobs(**kwargs)
        if jobs is None or getattr(jobs, "empty", True):
            return _empty()
        if track:
            jobs = jobs.copy()
            jobs["track"] = track
        return jobs.reset_index(drop=True)

    # Isolate each site so one hung scraper cannot block the hunt cycle.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for site in site_names:
            fut = pool.submit(_one, site)
            try:
                frame = fut.result(timeout=max(10.0, float(per_site_timeout_s)))
            except FuturesTimeout:
                logger.warning(
                    "Jobdrop site=%s timed out after %.0fs — skipping",
                    site,
                    per_site_timeout_s,
                )
                continue
            except Exception:
                logger.exception("jobdrop scrape failed site=%s", site)
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
                logger.info(
                    "Jobdrop site=%s returned %s jobs track=%s",
                    site,
                    len(frame),
                    track or "custom",
                )

    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    if "job_url" in out.columns:
        out = out.drop_duplicates(subset=["job_url"], keep="first")
    return out.reset_index(drop=True)


def _http_get(url: str, *, timeout: float = 20.0) -> str:
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_himalayas_jobs(
    keywords: str,
    *,
    max_results: int = 10,
    track: str | None = None,
) -> pd.DataFrame:
    """Lightweight Himalayas.app HTML discovery (no JobSpy/Jobdrop support)."""
    q = quote_plus((keywords or "software engineer intern")[:120])
    url = _HIMALAYAS_SEARCH.format(q=q)
    try:
        html = _http_get(url)
    except Exception:
        logger.warning("Himalayas fetch failed for %s", url[:120], exc_info=True)
        return _empty()

    # Prefer absolute job links
    hrefs = re.findall(
        r'href="(https://himalayas\.app/companies/[^"/]+/jobs/[^"?#]+)"',
        html,
        flags=re.I,
    )
    if not hrefs:
        hrefs = [
            urljoin("https://himalayas.app", path)
            for path in re.findall(
                r'href="(/companies/[^"/]+/jobs/[^"?#]+)"', html, flags=re.I
            )
        ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for href in hrefs:
        if href in seen:
            continue
        seen.add(href)
        company = ""
        m = re.search(r"/companies/([^/]+)/jobs/", href)
        if m:
            company = m.group(1).replace("-", " ").title()
        title_m = re.search(r"/jobs/([^/?#]+)", href)
        title = (title_m.group(1).replace("-", " ").title() if title_m else "Role")
        rows.append(
            {
                "site": "himalayas",
                "title": title,
                "company": company or "Unknown",
                "location": "Remote",
                "job_url": href,
                "job_url_direct": "",
                "description": f"{title} at {company}. Keywords: {keywords}",
            }
        )
        if len(rows) >= max_results:
            break
    logger.info("Himalayas custom scrape → %s jobs", len(rows))
    return _std_frame(rows, track=track)


def fetch_yc_waas_jobs(
    keywords: str,
    *,
    max_results: int = 10,
    track: str | None = None,
) -> pd.DataFrame:
    """Custom Work at a Startup (YC) discovery via public jobs HTML."""
    # Public listing; auth not required to read cards (apply still needs login).
    url = _YC_JOBS
    try:
        html = _http_get(url)
    except Exception:
        logger.warning("YC WaAS fetch failed", exc_info=True)
        return _empty()

    hrefs = re.findall(
        r'href="(https://www\.workatastartup\.com/jobs/\d+)"', html, flags=re.I
    )
    if not hrefs:
        hrefs = [
            urljoin("https://www.workatastartup.com", path)
            for path in re.findall(r'href="(/jobs/\d+)"', html, flags=re.I)
        ]

    needles = [
        w.lower()
        for w in re.findall(r"[A-Za-z]{3,}", keywords or "")
        if w.lower() not in {"and", "the", "for", "with", "intern", "or"}
    ][:8]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for href in hrefs:
        if href in seen:
            continue
        seen.add(href)
        # Soft keyword filter on surrounding snippet
        idx = html.lower().find(href.lower().split("/")[-1])
        snippet = html[max(0, idx - 200) : idx + 400].lower() if idx >= 0 else ""
        if needles and not any(n in snippet or n in href.lower() for n in needles):
            # Keep some unfiltered if nothing matches yet
            if len(rows) < 2:
                pass
            else:
                continue
        job_id = urlparse(href).path.rstrip("/").split("/")[-1]
        rows.append(
            {
                "site": "yc_waas",
                "title": f"YC Startup Role {job_id}",
                "company": "YC Startup",
                "location": "Remote / US / Flexible",
                "job_url": href,
                "job_url_direct": href,
                "description": (
                    f"Y Combinator Work at a Startup listing {job_id}. "
                    f"Search keywords: {keywords}"
                ),
            }
        )
        if len(rows) >= max_results:
            break
    logger.info("YC WaAS custom scrape → %s jobs", len(rows))
    return _std_frame(rows, track=track)


def fetch_extra_jobs_by_track(
    queries: dict[str, str],
    *,
    location: str = "India",
    max_results_per_track: int = 8,
    country_indeed: str = "India",
) -> pd.DataFrame:
    """Parallel Jobdrop + Himalayas + YC discovery across tracks."""
    frames: list[pd.DataFrame] = []

    def _one(track: str, query: str) -> list[pd.DataFrame]:
        out: list[pd.DataFrame] = []
        try:
            out.append(
                fetch_jobdrop_jobs(
                    query,
                    location=location,
                    max_results=max_results_per_track,
                    track=track,
                    country_indeed=country_indeed,
                    sites=jobdrop_sites_for_hunt(),
                    per_site_timeout_s=float(
                        os.environ.get("ACE_JOBDROP_SITE_TIMEOUT_S", "35") or 35
                    ),
                )
            )
        except Exception:
            logger.exception("Jobdrop track %s failed", track)
        try:
            out.append(
                fetch_himalayas_jobs(
                    query, max_results=max(3, max_results_per_track // 2), track=track
                )
            )
        except Exception:
            logger.exception("Himalayas track %s failed", track)
        try:
            out.append(
                fetch_yc_waas_jobs(
                    query, max_results=max(3, max_results_per_track // 2), track=track
                )
            )
        except Exception:
            logger.exception("YC track %s failed", track)
        return out

    max_workers = max(1, min(3, len(queries)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, t, q): t for t, q in queries.items()}
        for fut in as_completed(futs):
            try:
                for frame in fut.result():
                    if frame is not None and not frame.empty:
                        frames.append(frame)
            except Exception:
                logger.exception("Extra discovery track failed")

    if not frames:
        return _empty()
    combined = pd.concat(frames, ignore_index=True)
    if "job_url" in combined.columns:
        combined = combined.drop_duplicates(subset=["job_url"], keep="first")
    return combined.reset_index(drop=True)
