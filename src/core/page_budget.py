"""Dynamic 1-page content budget from fact-bank volume + JD ranking.

Invariant: slot counts scale with how many *strong* experience rows exist in
the profile. Adding a third internship/job automatically changes keep/eliminate
rules — never hardcode “always 2 roles” for one person.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Soft campus / non-engineering filler — drop first when stronger roles exist.
_SOFT_ROLE_MARKERS = (
    "university intern",
    "campus ambassador",
    "student volunteer",
    "club secretary",
)


@dataclass(frozen=True)
class PageBudget:
    """How many entries/bullets fit a truthful 1-page ace resume."""

    exp_keep: int
    exp_bullets: int
    proj_keep: int
    proj_bullets: int
    act_keep: int
    strong_roles: int
    reason: str

    def prompt_lines(self) -> str:
        return (
            f"- Experience: keep the {self.exp_keep} JD-best role(s); "
            f"EACH has {self.exp_bullets} XYZ bullets (120–175 chars) from the fact bank only.\n"
            f"- Projects: keep the {self.proj_keep} JD-best project(s); "
            f"EACH has {self.proj_bullets} weighty technical bullets from the fact bank.\n"
            f"- hackathons_and_activities: up to {self.act_keep} compact fact-bank bullets.\n"
            f"(Budget reason: {self.reason}; strong_roles={self.strong_roles})"
        )


def _is_soft_role(row: dict) -> bool:
    blob = " ".join(
        str(row.get(k) or "") for k in ("position", "company", "summary")
    ).lower()
    return any(m in blob for m in _SOFT_ROLE_MARKERS)


def _highlight_count(row: dict) -> int:
    highs = row.get("highlights")
    if not isinstance(highs, list):
        return 0
    return sum(1 for h in highs if str(h or "").strip())


def count_strong_roles(experience: list | None) -> tuple[int, list[dict]]:
    """Return (strong_count, strong_rows) — soft campus filler excluded when possible."""
    if not isinstance(experience, list):
        return 0, []
    rows = [r for r in experience if isinstance(r, dict)]
    strong = [r for r in rows if not _is_soft_role(r)]
    if strong:
        return len(strong), strong
    return len(rows), rows


def resolve_page_budget(
    *,
    experience: list | None = None,
    projects: list | None = None,
) -> PageBudget:
    """Map profile volume → 1-page keep/eliminate budget.

    Rough highlight budget on ace (~1 page): summary+edu+skills+acts + ≤16–18
    experience/project bullets. More strong roles → more exp slots, fewer projects.
    """
    n_strong, _ = count_strong_roles(experience)
    n_proj = (
        sum(1 for p in projects if isinstance(p, dict))
        if isinstance(projects, list)
        else 0
    )

    if n_strong <= 0:
        return PageBudget(
            exp_keep=0,
            exp_bullets=3,
            proj_keep=min(2, max(n_proj, 0)) or 2,
            proj_bullets=4,
            act_keep=3,
            strong_roles=0,
            reason="no_experience_project_heavy",
        )
    if n_strong == 1:
        return PageBudget(
            exp_keep=1,
            exp_bullets=4,
            proj_keep=2,
            proj_bullets=4,
            act_keep=3,
            strong_roles=1,
            reason="single_strong_role",
        )
    if n_strong == 2:
        return PageBudget(
            exp_keep=2,
            exp_bullets=4,
            proj_keep=2,
            proj_bullets=3,
            act_keep=3,
            strong_roles=2,
            reason="two_strong_roles",
        )
    if n_strong == 3:
        return PageBudget(
            exp_keep=3,
            exp_bullets=3,
            proj_keep=2 if n_proj >= 2 else 1,
            proj_bullets=3,
            act_keep=2,
            strong_roles=3,
            reason="three_strong_roles",
        )
    # 4+ strong roles: top-3 JD-ranked jobs, one flagship project
    return PageBudget(
        exp_keep=3,
        exp_bullets=3,
        proj_keep=1,
        proj_bullets=3,
        act_keep=2,
        strong_roles=n_strong,
        reason="four_plus_strong_roles_top3",
    )


def apply_budget_to_sections(
    sections: dict,
    budget: PageBudget,
    *,
    rank_fn,
    jd: str | None,
) -> None:
    """JD-rank then slice experience/projects/activities to the budget."""
    experience = sections.get("experience")
    if isinstance(experience, list) and experience:
        ranked = rank_fn(experience, jd)
        # Drop soft filler when we already have enough strong rows.
        strong_n, _ = count_strong_roles(ranked)
        if strong_n >= budget.exp_keep:
            ranked = [
                r
                for r in ranked
                if not (isinstance(r, dict) and _is_soft_role(r))
            ] or ranked
        kept = ranked[: budget.exp_keep]
        for row in kept:
            if not isinstance(row, dict):
                continue
            highs = row.get("highlights")
            if isinstance(highs, list):
                row["highlights"] = [
                    str(h).strip() for h in highs if str(h).strip()
                ][: budget.exp_bullets]
        sections["experience"] = kept

    projects = sections.get("projects")
    if isinstance(projects, list) and projects:
        ranked = rank_fn(projects, jd)
        kept = ranked[: budget.proj_keep]
        for row in kept:
            if not isinstance(row, dict):
                continue
            highs = row.get("highlights")
            if isinstance(highs, list):
                row["highlights"] = [
                    str(h).strip() for h in highs if str(h).strip()
                ][: budget.proj_bullets]
        sections["projects"] = kept

    for act_key in ("hackathons_and_activities", "extracurricular", "activities"):
        acts = sections.get(act_key)
        if isinstance(acts, list) and acts:
            trimmed = [str(a).strip() for a in acts if str(a).strip()][
                : budget.act_keep
            ]
            sections["hackathons_and_activities"] = trimmed
            if act_key != "hackathons_and_activities":
                sections.pop(act_key, None)


def bullet_length_issues(
    yaml_sections: dict,
    *,
    min_chars: int = 120,
    max_chars: int = 175,
) -> list[str]:
    """Flag stubs / runaway bullets (feedback for LLM; content gap if master-short)."""
    issues: list[str] = []
    for key in ("experience", "projects"):
        rows = yaml_sections.get(key)
        if not isinstance(rows, list):
            continue
        for i, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            highs = row.get("highlights")
            if not isinstance(highs, list):
                continue
            for j, h in enumerate(highs, start=1):
                text = str(h or "").strip()
                # Ignore markdown bold markers for length
                plain = re.sub(r"\*+", "", text)
                n = len(plain)
                if n and n < min_chars:
                    issues.append(
                        f"{key}[{i}] bullet {j} is {n} chars — deepen with fact-bank "
                        f"stack/outcome to ≥{min_chars} (no invented metrics)."
                    )
                elif n > max_chars + 40:
                    issues.append(
                        f"{key}[{i}] bullet {j} is {n} chars — tighten to ≤{max_chars}."
                    )
    return issues
