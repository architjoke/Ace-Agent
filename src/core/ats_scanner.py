"""Classical NLP ATS simulator: TF-IDF extraction + rigid keyword scoring.

Uses spaCy when available; falls back to sklearn + regex if spaCy/torch is broken
on the host (common Windows DLL issues). Scoring never uses an LLM.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

_STOPISH = {
    "experience",
    "experiences",
    "role",
    "roles",
    "team",
    "teams",
    "work",
    "working",
    "ability",
    "strong",
    "good",
    "excellent",
    "preferred",
    "required",
    "requirement",
    "requirements",
    "qualification",
    "qualifications",
    "candidate",
    "candidates",
    "opportunity",
    "company",
    "including",
    "using",
    "etc",
    "year",
    "years",
    "intern",
    "internship",
    "job",
    "position",
    "responsible",
    "responsibilities",
    # High-frequency TF-IDF junk that must never gate ATS
    "and",
    "or",
    "the",
    "with",
    "for",
    "from",
    "into",
    "that",
    "this",
    "your",
    "our",
    "ace",
    "basics",
    "software",
}

_TECH_PHRASES = [
    "rest api",
    "restful api",
    "rest apis",
    "machine learning",
    "deep learning",
    "computer vision",
    "data preprocessing",
    "feature engineering",
    "model deployment",
    "model training",
    "object oriented",
    "object-oriented",
    "data structures",
    "operating systems",
    "computer networks",
    "software engineer",
    "software development",
    "full stack",
    "full-stack",
    "node.js",
    "react.js",
    "scikit-learn",
    "random forest",
    "decision trees",
    "generative ai",
    "large language",
    "natural language",
    "ci/cd",
    "unit testing",
    "system design",
]


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    except Exception as exc:  # noqa: BLE001 — host torch/spacy often broken on Windows
        logger.warning("spaCy unavailable (%s); using TF-IDF/regex fallback", exc)
        return None


def _normalize(text: str) -> str:
    text = (text or "").lower().replace("\u2019", "'")
    text = re.sub(r"[`*_#>]+", " ", text)
    text = re.sub(r"[^a-z0-9+.#/\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _phrase_hits(jd_norm: str) -> list[str]:
    return [phrase for phrase in _TECH_PHRASES if phrase in jd_norm]


def _tfidf_terms(jd_text: str, top_k: int = 20) -> list[str]:
    doc = _normalize(jd_text)
    if not doc:
        return []
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        token_pattern=r"(?u)\b[a-z][a-z0-9+.#/\-]{1,}\b",
    )
    try:
        matrix = vectorizer.fit_transform([doc])
    except ValueError:
        return []
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)
    out: list[str] = []
    for term, score in ranked:
        if score <= 0:
            continue
        tokens = term.split()
        if any(tok in _STOPISH for tok in tokens):
            continue
        if len(term) < 2:
            continue
        out.append(term)
        if len(out) >= top_k:
            break
    return out


def _spacy_nouns(jd_text: str, limit: int = 25) -> list[str]:
    nlp = _nlp()
    if nlp is None:
        return []
    doc = nlp(jd_text)
    counts: dict[str, int] = {}
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        lemma = token.lemma_.lower().strip()
        if len(lemma) < 3 or lemma in _STOPISH:
            continue
        if not re.match(r"^[a-z][a-z0-9+.#/\-]*$", lemma):
            continue
        counts[lemma] = counts.get(lemma, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def _regex_tech_tokens(jd_text: str, limit: int = 25) -> list[str]:
    """Fallback noun-ish extractor when spaCy cannot load."""
    norm = _normalize(jd_text)
    counts: dict[str, int] = {}
    for tok in re.findall(r"\b[a-z][a-z0-9+.#/\-]{2,}\b", norm):
        if tok in _STOPISH:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, count in ranked if count >= 1][:limit]


def extract_jd_keywords(jd_text: str, top_k: int = 20) -> list[str]:
    """Extract top technical keywords/phrases from a JD via spaCy/TF-IDF/regex."""
    jd_norm = _normalize(jd_text or "")
    if not jd_norm:
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        key = _normalize(term)
        if not key or key in seen or key in _STOPISH:
            return
        seen.add(key)
        ordered.append(key)

    for phrase in _phrase_hits(jd_norm):
        _add(phrase)
    nouns = _spacy_nouns(jd_text) or _regex_tech_tokens(jd_text)
    for noun in nouns:
        _add(noun)
    for term in _tfidf_terms(jd_text, top_k=top_k * 2):
        _add(term)

    keywords = ordered[:top_k]
    logger.info("Extracted %s JD keywords: %s", len(keywords), keywords)
    return keywords


def filter_supportable_keywords(
    required_keywords: list[str],
    master_yaml_text: str,
) -> tuple[list[str], list[str]]:
    """Split keywords into supportable vs unsupported given the fact bank."""
    hay = _normalize(master_yaml_text or "")
    supportable: list[str] = []
    unsupported: list[str] = []
    for kw in required_keywords:
        key = _normalize(kw)
        if not key:
            continue
        if key in hay or all(tok in hay for tok in key.split() if len(tok) > 2):
            supportable.append(key)
        else:
            unsupported.append(key)
    return supportable, unsupported


def _keyword_found(resume_norm: str, keyword: str) -> bool:
    key = _normalize(keyword)
    if not key:
        return False
    if key in resume_norm:
        return True
    if " " not in key:
        return (
            re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", resume_norm)
            is not None
        )
    return False


def calculate_ats_score(
    drafted_yaml_text: str,
    required_keywords: list[str],
) -> dict:
    """Strict keyword-coverage ATS score (pure Python / NLP, no LLM)."""
    required = [_normalize(k) for k in (required_keywords or []) if _normalize(k)]
    deduped: list[str] = []
    seen: set[str] = set()
    for kw in required:
        if kw not in seen:
            seen.add(kw)
            deduped.append(kw)
    required = deduped

    if not required:
        return {"score": 100, "missing_keywords": [], "matched_keywords": []}

    resume_norm = _normalize(drafted_yaml_text or "")
    matched: list[str] = []
    missing: list[str] = []
    for kw in required:
        if _keyword_found(resume_norm, kw):
            matched.append(kw)
        else:
            missing.append(kw)

    score = int(round(100.0 * len(matched) / len(required)))
    logger.info(
        "ATS keyword score=%s matched=%s/%s missing=%s",
        score,
        len(matched),
        len(required),
        missing,
    )
    return {
        "score": score,
        "missing_keywords": missing,
        "matched_keywords": matched,
    }


def _ats_config() -> dict:
    try:
        from src.core.settings import get_settings

        raw = get_settings().raw or {}
        cfg = raw.get("ats") if isinstance(raw, dict) else None
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def resolve_ats_pass_score(
    *,
    company: str = "",
    supportable_keywords: list[str] | None = None,
    unsupported_keywords: list[str] | None = None,
) -> dict:
    """Company + JD hardness → pass threshold for this tailor run.

    Priority:
    1) Explicit company override in config.yaml ``ats.company_pass_scores``
    2) Default + JD hardness adjustment (supportable / unsupported density)
    3) Clamped to ``ats.min_pass_score`` … ``ats.max_pass_score``
    """
    cfg = _ats_config()
    default = int(cfg.get("default_pass_score") or 85)
    lo = int(cfg.get("min_pass_score") or 75)
    hi = int(cfg.get("max_pass_score") or 95)
    hardness = cfg.get("jd_hardness") if isinstance(cfg.get("jd_hardness"), dict) else {}
    company_map = (
        cfg.get("company_pass_scores")
        if isinstance(cfg.get("company_pass_scores"), dict)
        else {}
    )

    supportable = list(supportable_keywords or [])
    unsupported = list(unsupported_keywords or [])
    company_name = (company or "").strip()
    reason = "default"

    score = default
    # Company override (substring, case-insensitive)
    if company_name and company_map:
        lowered = company_name.lower()
        for key, value in company_map.items():
            key_s = str(key or "").strip()
            if not key_s:
                continue
            if key_s.lower() in lowered or lowered in key_s.lower():
                score = int(value)
                reason = f"company_override:{key_s}"
                break

    if reason == "default":
        n_sup = len(supportable)
        n_unsup = len(unsupported)
        total = max(n_sup + n_unsup, 1)
        raise_by = int(hardness.get("dense_supportable_raise") or 5)
        lower_sparse = int(hardness.get("sparse_supportable_lower") or 5)
        lower_unsup = int(hardness.get("unsupported_heavy_lower") or 3)

        if n_sup >= 12:
            score = default + raise_by
            reason = f"jd_dense_supportable({n_sup})"
        elif n_sup <= 5:
            score = default - lower_sparse
            reason = f"jd_sparse_supportable({n_sup})"
        else:
            reason = f"jd_default_supportable({n_sup})"

        # Heavy unsupported share → don't demand an unreachable bar
        if (n_unsup / total) >= 0.4:
            score -= lower_unsup
            reason += f"+unsupported_heavy({n_unsup}/{total})"

    score = max(lo, min(hi, int(score)))
    logger.info(
        "ATS pass threshold=%s company=%r reason=%s supportable=%s unsupported=%s",
        score,
        company_name,
        reason,
        len(supportable),
        len(unsupported),
    )
    return {
        "pass_score": score,
        "reason": reason,
        "default_pass_score": default,
        "company": company_name,
        "supportable_count": len(supportable),
        "unsupported_count": len(unsupported),
    }
