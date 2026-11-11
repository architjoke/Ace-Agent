"""Job-focused company research for Ace Copilot (no browser required).

Sources (read-only, network):
  - Wikipedia summary API
  - DuckDuckGo Instant Answer
  - Ace application memory (prior applies / roles / status)

Returns structured text the chat model can explain to the user.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (compatible; AceAgent/1.0; +https://github.com/ace-agent; "
    "job-research)"
)
_MAX_SNIPPET = 900


def _http_json(url: str, *, timeout: float = 12.0) -> Any | None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        logger.debug("research fetch failed %s: %s", url[:80], exc)
        return None


def _clean(text: str, limit: int = _MAX_SNIPPET) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) > limit:
        return t[: limit - 1].rstrip() + "…"
    return t


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _best_match(query: str, candidates: list[str]) -> tuple[str | None, list[str]]:
    needle = _norm(query)
    if not needle or not candidates:
        return None, []
    scored: list[tuple[float, str]] = []
    for c in candidates:
        t = _norm(c)
        if not t:
            continue
        if needle == t:
            return c, []
        score = SequenceMatcher(None, needle, t).ratio()
        if needle in t or t in needle:
            score = max(score, 0.85)
        if score >= 0.72:
            scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    if not scored:
        return None, []
    if len(scored) == 1 or scored[0][0] >= 0.95:
        return scored[0][1], []
    if scored[0][0] - scored[1][0] >= 0.08:
        return scored[0][1], []
    return None, [n for _, n in scored[:5]]


def _wiki_summary(company: str) -> dict[str, str] | None:
    title = urllib.parse.quote(company.strip().replace(" ", "_"))
    data = _http_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
    if not isinstance(data, dict):
        return None
    if data.get("type") == "disambiguation":
        return {
            "source": "wikipedia",
            "note": "Ambiguous Wikipedia title — refine the company name.",
            "url": str(
                (data.get("content_urls") or {})
                .get("desktop", {})
                .get("page")
                or ""
            ),
        }
    extract = _clean(str(data.get("extract") or ""))
    if not extract:
        return None
    desktop = (data.get("content_urls") or {}).get("desktop") or {}
    return {
        "source": "wikipedia",
        "title": str(data.get("title") or company),
        "summary": extract,
        "url": str(desktop.get("page") or ""),
    }


def _ddg_instant(company: str) -> dict[str, str] | None:
    q = urllib.parse.quote(f"{company.strip()} company")
    data = _http_json(
        f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
    )
    if not isinstance(data, dict):
        return None
    abstract = _clean(str(data.get("AbstractText") or ""))
    heading = str(data.get("Heading") or company)
    url = str(data.get("AbstractURL") or "")
    related: list[str] = []
    for item in (data.get("RelatedTopics") or [])[:4]:
        if isinstance(item, dict) and item.get("Text"):
            related.append(_clean(str(item["Text"]), 220))
        elif isinstance(item, dict) and isinstance(item.get("Topics"), list):
            for sub in item["Topics"][:2]:
                if isinstance(sub, dict) and sub.get("Text"):
                    related.append(_clean(str(sub["Text"]), 220))
    if not abstract and not related:
        return None
    out: dict[str, str] = {"source": "duckduckgo", "title": heading}
    if abstract:
        out["summary"] = abstract
    if url:
        out["url"] = url
    if related:
        out["related"] = " | ".join(related[:3])
    return out


def _ace_history(company: str) -> dict[str, Any]:
    from src.core.desk import manual_packages
    from src.core.memory import _connect, init_db

    init_db()
    needle = (company or "").strip()
    with _connect() as conn:
        all_rows = conn.execute(
            """
            SELECT company, status, job_title, job_url, track, outcome_status, timestamp
            FROM applications
            ORDER BY timestamp DESC
            LIMIT 400
            """
        ).fetchall()
    names = [str(r["company"]) for r in all_rows if r["company"]]
    resolved, ranked = _best_match(needle, names)
    target = resolved or needle
    target_n = _norm(target)
    rows: list[dict[str, Any]] = []
    for r in all_rows:
        if _norm(str(r["company"])) != target_n:
            continue
        rows.append(dict(r))
        if len(rows) >= 8:
            break

    manuals = [p["company"] for p in manual_packages()]
    folder_hit, _ = _best_match(target, manuals)

    return {
        "resolved_company": target if rows or folder_hit else needle,
        "ambiguous": ranked if not resolved else None,
        "applications": rows,
        "manual_action": bool(folder_hit),
        "manual_folder": folder_hit,
    }


def research_company_brief(company: str, focus: str = "") -> str:
    """Compile a job-seeker brief for ``company``."""
    name = (company or "").strip()
    if not name:
        raise ValueError("company is required.")
    focus_s = (focus or "").strip()

    ace = _ace_history(name)
    if ace.get("ambiguous") and not ace.get("applications"):
        opts = ", ".join(ace["ambiguous"])
        return (
            f"Ambiguous company {name!r} in Ace memory. Candidates: {opts}. "
            "Ask which one, or research with the exact legal name."
        )

    display = str(ace.get("resolved_company") or name)
    wiki = _wiki_summary(display) or _wiki_summary(name)
    ddg = _ddg_instant(display) or _ddg_instant(name)

    lines = [
        f"Company research: {display}",
        (
            f"Focus hint: {focus_s}"
            if focus_s
            else "Focus hint: job-seeker overview (what they do, history, hiring relevance)"
        ),
        "",
    ]

    if wiki and wiki.get("summary"):
        lines.append(f"[Wikipedia] {wiki.get('title') or display}")
        lines.append(wiki["summary"])
        if wiki.get("url"):
            lines.append(f"  → {wiki['url']}")
        lines.append("")
    elif wiki and wiki.get("note"):
        lines.append(f"[Wikipedia] {wiki['note']}")
        lines.append("")

    if ddg and ddg.get("summary"):
        lines.append(f"[Web] {ddg.get('title') or display}")
        lines.append(ddg["summary"])
        if ddg.get("url"):
            lines.append(f"  → {ddg['url']}")
        lines.append("")
    elif ddg and ddg.get("related"):
        lines.append(f"[Web related] {ddg['related']}")
        lines.append("")

    apps = ace.get("applications") or []
    if apps or ace.get("manual_action"):
        lines.append("[Ace memory]")
        if ace.get("manual_action"):
            lines.append(f"  Manual_Action package present: {ace.get('manual_folder')}")
        for row in apps:
            role = row.get("job_title") or "(role unknown)"
            lines.append(
                f"  • {row.get('status')} · {role} · "
                f"outcome={row.get('outcome_status') or 'PENDING'} · "
                f"{(row.get('timestamp') or '')[:19]}"
            )
            if row.get("job_url"):
                lines.append(f"    {row['job_url']}")
        lines.append("")

    if not (wiki and wiki.get("summary")) and not (ddg and ddg.get("summary")) and not apps:
        lines.append(
            "No public summary found. Ask for the official website / LinkedIn "
            "company page, or try a more precise legal name."
        )
    else:
        lines.append(
            "Use for interview prep / apply decisions. Prefer facts above; "
            "do not invent funding or culture claims missing from sources."
        )

    if re.search(r"\b(hr|interview)\b", focus_s, re.I):
        lines.append("")
        lines.append("[HR round — answer angles]")
        lines.append(
            "  • What they do: summarize the Wikipedia/Web blurb in 2–3 plain sentences."
        )
        lines.append(
            "  • Why you: tie your stack/role (from Ace memory above if present) "
            "to their products/services — no invented experience."
        )
        lines.append(
            "  • Why them: cite HQ/industry from sources (e.g. Bengaluru IT services); "
            "avoid fake culture or funding claims."
        )
        lines.append(
            "  • Smart questions: recent products/services from the summary; "
            "team’s tech stack; internship/mentorship structure."
        )
        if apps:
            lines.append(
                "  • Consistency: you already touched this company in Ace — "
                "mention the role you applied for if asked about interest."
            )

    return "\n".join(lines).strip()
