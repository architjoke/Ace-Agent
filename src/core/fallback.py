"""Manual-action fallback packages when autonomous apply cannot finish."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.config import OUTPUT_DIR, PROJECT_ROOT
from src.core.llm_router import invoke_with_failover

logger = logging.getLogger(__name__)

REPO_ROOT = PROJECT_ROOT


def _manual_root() -> Path:
    try:
        from src.core.settings import get_settings

        return Path(get_settings().output_dir) / "Manual_Action"
    except Exception:
        return OUTPUT_DIR / "Manual_Action"


def _output_root() -> Path:
    try:
        from src.core.settings import get_settings

        return Path(get_settings().output_dir)
    except Exception:
        return OUTPUT_DIR


class ManualPackageContent(BaseModel):
    cover_letter: str = Field(
        ...,
        description=(
            "A highly targeted cover letter of about 150 words in a professional "
            "engineering tone. No markdown fences."
        ),
    )
    # No min_length/max_length — Cohere structured-output rejects array minItems/maxItems.
    top_keywords: list[str] = Field(
        ...,
        description="Exactly five high-signal keywords/phrases extracted from the JD.",
    )
    apply_instructions: str = Field(
        ...,
        description="Concise step-by-step instructions for manually completing the application.",
    )


def _safe_dir_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "", (name or "").strip()).strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unknown"


def _resolve_tailored_yaml(company: str, tailored_pdf_path: Path, master_yaml_path: Path) -> str:
    """Prefer the company tailored YAML; fall back to master fact bank."""
    candidates = [
        tailored_pdf_path.parent / "tailored.yaml",
        tailored_pdf_path.with_name("tailored.yaml"),
        _output_root() / _safe_dir_name(company) / "tailored.yaml",
        master_yaml_path,
    ]
    for path in candidates:
        if path.is_file():
            logger.info("Using YAML context from %s", path)
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No tailored/master YAML found for {company}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def generate_manual_package(
    company: str,
    job_url: str,
    jd: str,
    tailored_pdf_path: str | Path,
    master_yaml_path: str | Path,
) -> str:
    """Build a manual-apply packet under output/Manual_Action/{company}/.

    Generates a ~150-word cover letter and application guide via the text LLM,
    copies the tailored PDF into the folder, and returns the folder path.
    """
    company_name = (company or "Unknown").strip() or "Unknown"
    pdf_path = Path(tailored_pdf_path).expanduser().resolve()
    master_path = Path(master_yaml_path).expanduser().resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Tailored PDF not found: {pdf_path}")
    if not master_path.is_file():
        raise FileNotFoundError(f"Master YAML not found: {master_path}")

    from src.core.tailor import assert_one_page_pdf

    assert_one_page_pdf(pdf_path)

    tailored_yaml = _resolve_tailored_yaml(company_name, pdf_path, master_path)

    prompt = f"""You are a technical recruiter ghostwriting for a CS/AI engineering candidate.

Using the job description and the tailored RenderCV YAML, produce:
1. cover_letter — ~150 words, professional engineering tone, specific to this role.
   Do not invent employers, degrees, or metrics absent from the YAML.
2. top_keywords — exactly 5 high-signal keywords/phrases from the JD the candidate should emphasize.
3. apply_instructions — short manual apply checklist for this posting.

## Company
{company_name}

## Job URL
{job_url}

## Job Description
---
{jd}
---

## Tailored RenderCV YAML
---
{tailored_yaml}
---
"""

    content = invoke_with_failover(
        "fallback",
        prompt,
        temperature=0.3,
        structured=ManualPackageContent,
    )
    if isinstance(content, dict):
        cover_letter = str(content.get("cover_letter") or "").strip()
        keywords = [str(k).strip() for k in (content.get("top_keywords") or []) if str(k).strip()]
        instructions = str(content.get("apply_instructions") or "").strip()
    else:
        cover_letter = str(content.cover_letter).strip()
        keywords = [str(k).strip() for k in content.top_keywords if str(k).strip()]
        instructions = str(content.apply_instructions).strip()

    if len(keywords) < 5:
        keywords = (keywords + ["N/A"] * 5)[:5]
    else:
        keywords = keywords[:5]

    out_dir = _manual_root() / _safe_dir_name(company_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_pdf = out_dir / pdf_path.name
    if pdf_path.resolve() != dest_pdf.resolve():
        shutil.copy2(pdf_path, dest_pdf)

    cover_path = out_dir / "cover_letter.md"
    cover_path.write_text(
        f"# Cover Letter — {company_name}\n\n{cover_letter}\n",
        encoding="utf-8",
    )

    keyword_lines = "\n".join(f"{i}. {kw}" for i, kw in enumerate(keywords, start=1))
    guide = f"""# Application Guide — {company_name}

## Direct Job Link
{job_url}

## Top 5 Keywords From the JD
{keyword_lines}

## Manual Apply Instructions
{instructions}

## Artifacts in This Folder
- Resume PDF: `{dest_pdf.name}`
- Cover letter: `cover_letter.md`

Upload the tailored resume, paste or attach the cover letter if requested, and emphasize the keywords above in any free-text questionnaire boxes.
"""
    (out_dir / "application_guide.md").write_text(guide, encoding="utf-8")

    logger.info("Manual package ready at %s", out_dir)
    return str(out_dir.resolve())
