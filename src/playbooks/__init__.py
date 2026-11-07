"""Deterministic ATS playbooks (DOM) with Gemini vision fallback."""

from __future__ import annotations

from urllib.parse import urlparse


class PlaybookFallback(Exception):
    """Raised when a DOM playbook cannot finish — hand off to Gemini."""


def classify_ats(job_url: str) -> str:
    """Return ats kind: greenhouse | lever | ashby | workday | other."""
    host = (urlparse(job_url or "").hostname or "").lower()
    path = (urlparse(job_url or "").path or "").lower()
    blob = f"{host}{path}"
    if "boards.greenhouse.io" in host or host.endswith("greenhouse.io"):
        return "greenhouse"
    if "greenhouse.io" in blob and "/jobs/" in path:
        return "greenhouse"
    if "ashbyhq.com" in host or "jobs.ashbyhq.com" in host:
        return "ashby"
    if "jobs.lever.co" in host or host.endswith("lever.co"):
        return "lever"
    if "myworkdayjobs.com" in host or "workday.com" in host and "/job/" in path:
        return "workday"
    if host.startswith("wd") and "myworkdayjobs.com" in host:
        return "workday"
    return "other"
