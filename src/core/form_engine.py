"""FormFillEngine — resolve field kinds from personal profile + JD (no playbook QAs).

Playbooks pass visible labels / page widgets; this module returns values from
``data/master.yaml``-derived profile dicts. Shared and person-agnostic code only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.form_rules import FieldKind, classify_label, kind_to_profile_keys

logger = logging.getLogger(__name__)

_WIDGET_TIMEOUT_MS = 2_000
_VISIBLE_MS = 120
_GUIDE_PATH = Path(__file__).resolve().parent / "form_fill_guide.md"

_YES_TRUTHY = frozenset({"yes", "true", "1", "y"})
_SKILL_SPLIT = re.compile(r"[,;/|•·\n]+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.]{1,24}")

_YN_MIN_CONFIDENCE = 0.7

# One JS pass scrapes modal fields — avoids N× Playwright round-trips.
_SCRAPE_FIELDS_JS = """() => {
  const roots = [
    ...document.querySelectorAll(
      "dialog[open], .jobs-easy-apply-modal, [data-test-modal-id='easy-apply-modal'], div[role='dialog']"
    ),
  ];
  const root = roots.find(Boolean) || document;
  const vis = (el) => {
    if (!el) return false;
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect();
    return r.width > 2 && r.height > 2;
  };
  const labelFor = (el) => {
    const aria = (el.getAttribute('aria-label') || '').trim();
    if (aria) return aria;
    const id = el.id;
    if (id) {
      try {
        const lab = document.querySelector('label[for="' + CSS.escape(id) + '"]');
        if (lab) return (lab.innerText || lab.textContent || '').trim();
      } catch (e) {}
    }
    const wrap = el.closest('div,fieldset,li,section,label') || el.parentElement;
    return ((wrap && (wrap.innerText || '')) || '').trim().slice(0, 200);
  };
  const out = [];
  const nodes = root.querySelectorAll(
    "textarea, input[type='text'], input:not([type]), input[type='search'], " +
    "input[type='tel'], input[type='email'], select, " +
    "input[type='radio'], input[type='checkbox']"
  );
  for (const el of nodes) {
    if (!vis(el) || out.length >= 40) continue;
    const tag = (el.tagName || '').toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    let value = '';
    let checked = false;
    try {
      if (type === 'radio' || type === 'checkbox') {
        checked = !!el.checked;
        value = (el.value || el.getAttribute('aria-label') || '').trim();
      } else {
        value = (el.value || '').trim();
      }
    } catch (e) {}
    out.push({
      tag: type === 'radio' || type === 'checkbox' ? type : tag,
      name: el.name || '',
      id: el.id || '',
      label: labelFor(el),
      value,
      checked,
    });
  }
  return out;
}"""


@dataclass
class ResolveResult:
    kind: FieldKind
    value: str | None
    confidence: float
    needs_human: bool = False


def load_form_fill_guide() -> str:
    """Shared fill rules for CDP / prompts (no personal facts)."""
    try:
        return _GUIDE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _first_nonempty(profile: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        val = str(profile.get(key) or "").strip()
        if val:
            return val
    return ""


def _profile_skill_tokens(profile: dict[str, Any]) -> list[str]:
    """Collect skill / tool names from profile facts (skills list or narrative)."""
    raw_bits: list[str] = []
    for key in (
        "skills_list",
        "skills",
        "tech_stack",
        "ai_llm_technologies",
        "tools",
        "languages",
    ):
        val = profile.get(key)
        if isinstance(val, list):
            raw_bits.extend(str(x) for x in val)
        elif val:
            raw_bits.append(str(val))
    tokens: list[str] = []
    seen: set[str] = set()
    for bit in raw_bits:
        for part in _SKILL_SPLIT.split(bit):
            t = part.strip()
            if len(t) < 2:
                continue
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            tokens.append(t)
    return tokens


def _jd_tokens(jd_text: str | None) -> set[str]:
    if not jd_text:
        return set()
    # Cap JD parse cost on huge descriptions.
    blob = (jd_text or "")[:8000]
    return {m.group(0).lower() for m in _TOKEN.finditer(blob)}


def resolve_skills_tech(
    profile: dict[str, Any],
    jd_text: str | None = None,
    *,
    kind_override: str | None = None,
) -> str:
    """Profile skills ∩ JD keywords; fallback to top profile skills."""
    if kind_override:
        return kind_override
    skills = _profile_skill_tokens(profile)
    if not skills:
        return ""
    jd = _jd_tokens(jd_text)
    if jd:
        hit = [s for s in skills if any(tok in jd for tok in _TOKEN.findall(s.lower()))]
        if hit:
            return ", ".join(hit[:12])
    return ", ".join(skills[:10])


def resolve_yes_no(
    label: str,
    profile: dict[str, Any],
    *,
    taught: str | None = None,
) -> str | None:
    """Map eligibility-style questions to profile prefs."""
    if taught:
        low = taught.lower()
        if low in {"yes", "no"}:
            return taught[:1].upper() + taught[1:].lower() if len(taught) > 1 else taught
        if low in _YES_TRUTHY:
            return "Yes"
        if low in {"no", "false", "0", "n"}:
            return "No"

    blob = (label or "").lower()
    if "visa" in blob or "sponsor" in blob:
        raw = str(profile.get("require_visa_sponsorship_india") or "").strip()
        if raw:
            return "No" if raw.lower().startswith("n") else "Yes"
    if "relocat" in blob:
        raw = str(profile.get("willing_to_relocate_india") or "").strip()
        if raw:
            return "Yes" if raw.lower().startswith("y") else "No"
    if "8 hour" in blob or "hours a day" in blob or "hours per day" in blob:
        hours = profile.get("can_commit_hours_per_day")
        if hours is not None and str(hours).strip():
            try:
                return "Yes" if float(hours) >= 8 else "No"
            except ValueError:
                pass
        return "Yes"
    if "internship" in blob or "intern" in blob or "2-month" in blob or "2 month" in blob:
        months = profile.get("open_to_internship_months")
        if months is not None and str(months).strip():
            try:
                return "Yes" if float(months) >= 2 else "No"
            except ValueError:
                pass
        return "Yes"
    if (
        "prompt engineering" in blob
        or "ai/llm" in blob
        or "built any" in blob
        or "worked on" in blob
    ):
        return "Yes"
    return None


def resolve_field(
    label: str,
    profile: dict[str, Any],
    *,
    jd_text: str | None = None,
    kind_overrides: dict[str, str] | None = None,
    label_override: str | None = None,
) -> ResolveResult:
    """Classify label and resolve a value from the personal profile (+ JD for skills)."""
    kind = classify_label(label)
    overrides = kind_overrides or {}

    if label_override:
        return ResolveResult(kind=kind, value=label_override, confidence=0.7)

    by_kind = overrides.get(kind.value)
    if by_kind:
        return ResolveResult(kind=kind, value=by_kind, confidence=0.95)

    if kind == FieldKind.UPLOAD_RESUME:
        return ResolveResult(kind=kind, value=None, confidence=1.0)

    if kind == FieldKind.SKILLS_TECH:
        text = resolve_skills_tech(
            profile, jd_text, kind_override=overrides.get(FieldKind.SKILLS_TECH.value)
        )
        if text:
            return ResolveResult(kind=kind, value=text, confidence=0.9)
        return ResolveResult(kind=kind, value=None, confidence=0.2, needs_human=True)

    if kind == FieldKind.YES_NO_ELIGIBILITY:
        yn = resolve_yes_no(label, profile)
        if yn:
            return ResolveResult(kind=kind, value=yn, confidence=0.85)
        return ResolveResult(kind=kind, value=None, confidence=0.2, needs_human=True)

    if kind == FieldKind.IDENTITY_PORTFOLIO:
        val = _first_nonempty(profile, kind_to_profile_keys(kind))
        # Empty is intentional — never invent GitHub/LinkedIn as a portfolio URL.
        return ResolveResult(kind=kind, value=val if val else "", confidence=1.0)

    if kind == FieldKind.UNKNOWN:
        try:
            from src.core.learned_answers import lookup_answer

            taught = lookup_answer(label)
            if taught:
                return ResolveResult(
                    kind=FieldKind.UNKNOWN, value=taught, confidence=0.7
                )
        except Exception:
            pass
        return ResolveResult(kind=kind, value=None, confidence=0.0, needs_human=True)

    keys = kind_to_profile_keys(kind)
    val = _first_nonempty(profile, keys)
    if val:
        return ResolveResult(kind=kind, value=val, confidence=0.95)
    return ResolveResult(kind=kind, value=None, confidence=0.1, needs_human=True)


def enrich_profile_skills(
    profile: dict[str, Any], master_data: dict[str, Any] | None
) -> dict[str, Any]:
    """Pull skill details from RenderCV ``cv.sections.skills`` into profile dict."""
    out = dict(profile or {})
    if out.get("skills_list"):
        return out
    data = master_data or {}
    cv = data.get("cv") if isinstance(data.get("cv"), dict) else {}
    sections = cv.get("sections") if isinstance(cv.get("sections"), dict) else {}
    skills = sections.get("skills")
    bits: list[str] = []
    if isinstance(skills, list):
        for row in skills:
            if isinstance(row, dict):
                details = str(row.get("details") or "").strip()
                if details:
                    bits.append(details)
            elif row:
                bits.append(str(row))
    if bits:
        out["skills_list"] = bits
    return out


def _load_kind_overrides() -> dict[str, str]:
    try:
        from src.core.learned_answers import list_entries

        best: dict[str, tuple[str, str]] = {}
        for item in list_entries():
            kind = str(item.get("kind") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not kind or not answer:
                continue
            ts = str(item.get("updated_at") or "")
            prev = best.get(kind)
            if prev is None or ts >= prev[0]:
                best[kind] = (ts, answer)
        return {k: v[1] for k, v in best.items()}
    except Exception:
        return {}


class FormFillEngine:
    """Deterministic form filler driven by profile facts + general kinds."""

    def __init__(
        self,
        profile: dict[str, Any],
        *,
        jd_text: str | None = None,
        master_data: dict[str, Any] | None = None,
    ) -> None:
        self.profile = enrich_profile_skills(profile, master_data)
        self.jd_text = jd_text or ""
        self._kind_overrides = _load_kind_overrides()
        self._resolve_cache: dict[str, ResolveResult] = {}
        self._skills_cache: str | None = None

    def resolve(self, label: str) -> ResolveResult:
        key = (label or "").strip().lower()[:160]
        hit = self._resolve_cache.get(key)
        if hit is not None:
            return hit
        # Lazy skills string once.
        if self._skills_cache is None:
            self._skills_cache = resolve_skills_tech(
                self.profile,
                self.jd_text,
                kind_override=self._kind_overrides.get(FieldKind.SKILLS_TECH.value),
            )
        result = resolve_field(
            label,
            self.profile,
            jd_text=self.jd_text,
            kind_overrides=self._kind_overrides,
        )
        # Inject cached skills without re-intersecting.
        if result.kind == FieldKind.SKILLS_TECH and self._skills_cache:
            result = ResolveResult(
                kind=result.kind,
                value=self._skills_cache,
                confidence=0.9,
                needs_human=False,
            )
        self._resolve_cache[key] = result
        return result

    def _fill_yes_no_controls(
        self, page: Any, *, timeout_ms: int, unknown: list[str]
    ) -> int:
        """Click Yes or No only when resolve matches; never default unlabeled → Yes."""
        filled = 0
        try:
            modal = page.locator(
                "dialog[open], .jobs-easy-apply-modal, "
                "[data-test-modal-id='easy-apply-modal'], div[role='dialog']"
            ).first
            root = modal if modal.count() else page
        except Exception:
            root = page

        for word in ("Yes", "No"):
            try:
                locs = root.get_by_text(re.compile(rf"^{word}$", re.I)).all()[:12]
            except Exception:
                locs = []
            for loc in locs:
                try:
                    if not loc.is_visible(timeout=_VISIBLE_MS):
                        continue
                    label = ""
                    try:
                        label = (
                            loc.evaluate(
                                """el => {
                                  const wrap = el.closest('fieldset,div,li,section,form') || el.parentElement;
                                  return ((wrap && (wrap.innerText || '')) || '').trim().slice(0, 180);
                                }"""
                            )
                            or ""
                        )
                    except Exception:
                        label = ""
                    if not label or len(label.strip()) < 3:
                        # No surrounding question — never guess Yes/No.
                        continue
                    result = self.resolve(label)
                    want = (result.value or "").strip().lower()
                    if result.needs_human and not want:
                        unknown.append(label[:120])
                        continue
                    if result.kind not in (
                        FieldKind.YES_NO_ELIGIBILITY,
                        FieldKind.UNKNOWN,
                    ):
                        continue
                    if result.confidence < _YN_MIN_CONFIDENCE and not (
                        result.kind == FieldKind.UNKNOWN and want in {"yes", "no"}
                    ):
                        if result.needs_human:
                            unknown.append(label[:120])
                        continue
                    if want not in {"yes", "no"}:
                        continue
                    if want != word.lower():
                        continue
                    loc.click(timeout=timeout_ms, force=True)
                    filled += 1
                except Exception:
                    continue
        return filled

    def _fill_choice_input(
        self,
        page: Any,
        meta: dict[str, Any],
        result: ResolveResult,
        *,
        timeout_ms: int,
    ) -> bool:
        """Check/click radio or checkbox when resolved value matches the option."""
        tag = str(meta.get("tag") or "").lower()
        if tag not in {"radio", "checkbox"}:
            return False
        want = (result.value or "").strip().lower()
        if not want:
            return False
        option = str(meta.get("value") or "").strip().lower()
        label = str(meta.get("label") or "").strip().lower()
        checked = bool(meta.get("checked"))
        # Match option value, or Yes/No token in label/option.
        matches = False
        if option and (option == want or want in option or option in want):
            matches = True
        elif want in {"yes", "no"} and (
            option == want
            or option.startswith(want)
            or re.search(rf"\b{re.escape(want)}\b", label)
        ):
            matches = True
        elif want in {"true", "1", "y"} and tag == "checkbox":
            matches = True
            want = "yes"
        if not matches:
            return False
        if tag == "checkbox" and want in {"no", "false", "0", "n"}:
            if checked:
                # Uncheck if we resolved No — rare but safe.
                pass
            else:
                return False
        if checked and tag == "radio":
            return False

        loc = None
        fid = str(meta.get("id") or "")
        fname = str(meta.get("name") or "")
        try:
            if fid:
                cand = page.locator(f"#{fid}").first
                if cand.count() and cand.is_visible(timeout=_VISIBLE_MS):
                    loc = cand
            elif fname:
                # Prefer the radio whose value matches.
                opt = str(meta.get("value") or "")
                if opt:
                    cand = page.locator(
                        f"input[type='{tag}'][name={fname!r}][value={opt!r}]"
                    ).first
                else:
                    cand = page.locator(f"input[type='{tag}'][name={fname!r}]").first
                if cand.count() and cand.is_visible(timeout=_VISIBLE_MS):
                    loc = cand
        except Exception:
            loc = None
        if loc is None:
            return False
        try:
            if tag == "checkbox":
                if want in {"no", "false", "0", "n"}:
                    if checked:
                        loc.uncheck(timeout=timeout_ms, force=True)
                        return True
                    return False
                if not checked:
                    loc.check(timeout=timeout_ms, force=True)
                    return True
                return False
            loc.check(timeout=timeout_ms, force=True)
            return True
        except Exception:
            try:
                loc.click(timeout=timeout_ms, force=True)
                return True
            except Exception:
                return False

    def fill_visible(self, page: Any, *, timeout_ms: int = _WIDGET_TIMEOUT_MS) -> dict[str, Any]:
        """Fill visible modal fields. One JS scrape + targeted fills."""
        filled = 0
        unknown: list[str] = []

        try:
            filled += self._fill_yes_no_controls(
                page, timeout_ms=timeout_ms, unknown=unknown
            )
        except Exception:
            logger.debug("yes-no control pass failed", exc_info=True)

        scraped: list[dict[str, Any]] = []
        try:
            scraped = page.evaluate(_SCRAPE_FIELDS_JS) or []
        except Exception:
            scraped = []

        for meta in scraped:
            try:
                label = str(meta.get("label") or "").strip()
                if not label:
                    continue
                current = str(meta.get("value") or "").strip()
                tag = str(meta.get("tag") or "").lower()
                result = self.resolve(label)
                if result.needs_human and not result.value:
                    unknown.append(label[:120])
                    continue
                if result.value is None or result.kind == FieldKind.UPLOAD_RESUME:
                    continue

                if tag in {"radio", "checkbox"}:
                    if self._fill_choice_input(
                        page, meta, result, timeout_ms=timeout_ms
                    ):
                        filled += 1
                    continue

                # Skip already-correct text; still overwrite portfolio clears / mismatches.
                if (
                    current
                    and tag != "select"
                    and current == str(result.value).strip()
                ):
                    continue
                if (
                    current
                    and tag != "select"
                    and result.kind != FieldKind.IDENTITY_PORTFOLIO
                    and str(result.value).strip() == ""
                ):
                    continue

                # Prefer stable locator: get_by_label, else name/id.
                loc = None
                try:
                    cand = page.get_by_label(re.compile(re.escape(label[:80]), re.I)).first
                    if cand.count() and cand.is_visible(timeout=_VISIBLE_MS):
                        loc = cand
                except Exception:
                    loc = None
                if loc is None:
                    fid = str(meta.get("id") or "")
                    fname = str(meta.get("name") or "")
                    try:
                        if fid:
                            cand = page.locator(f"#{fid}").first
                            if cand.count() and cand.is_visible(timeout=_VISIBLE_MS):
                                loc = cand
                        elif fname:
                            cand = page.locator(f"[name={fname!r}]").first
                            if cand.count() and cand.is_visible(timeout=_VISIBLE_MS):
                                loc = cand
                    except Exception:
                        loc = None
                if loc is None:
                    continue

                if tag == "select":
                    for cand_label in (
                        result.value,
                        f"{result.value} year",
                        f"{result.value} years",
                        f"{result.value} months",
                        f"{result.value} month",
                    ):
                        try:
                            loc.select_option(label=cand_label, timeout=timeout_ms)
                            filled += 1
                            break
                        except Exception:
                            continue
                else:
                    loc.fill(str(result.value), timeout=timeout_ms)
                    filled += 1
            except Exception:
                continue

        return {"filled": filled, "unknown_labels": unknown}


def fill_application_form(
    page: Any,
    profile: dict[str, Any],
    *,
    jd_text: str | None = None,
    master_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Public entry used by ATS playbooks."""
    return FormFillEngine(profile, jd_text=jd_text, master_data=master_data).fill_visible(
        page
    )
