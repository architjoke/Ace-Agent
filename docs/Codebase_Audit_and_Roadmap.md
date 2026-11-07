# Ace Job Agent — Codebase Audit & Roadmap

See historical hardening notes from the 2026-11-07 production audit.

## Highlights already shipped
- Parallel JobSpy + tailor pool + cloned Chromium workers
- Groq → NVIDIA → Ollama failover with timeouts / 429 detection
- Repo-anchored `PROJECT_ROOT` (`src/core/config.py`) for Task Scheduler safety
- SQLite WAL + apply queue; batch LinkedIn resume cleanup

## Suggested next features
1. Proxy / residential IP rotation for parallel browsers  
2. Local HTML dashboard over SQLite + worker status  
3. Richer IMAP extraction (assessment / calendar links)  
4. Deterministic ATS playbooks (Greenhouse / Lever / Easy Apply) with vision fallback  
