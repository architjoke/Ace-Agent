"""Industry ATS + role-matching playbook encoded for Gemini nodes.

Synthesized from RenderCV ATS guidance, Indeed/Jobscan keyword practice, and
common open-source tailor patterns (mirror JD language, evidence-backed skills,
single-column parseable structure via RenderCV).
"""

ATS_MATCHING_DOCTRINE = """
## Ace-level matching doctrine
You are optimizing for TWO screens in sequence:
1) ATS / keyword parsers (literal string + section parsing)
2) Human / LLM recruiters (evidence, impact, clarity)

The ATS pass bar is **company- and JD-specific** (config + keyword hardness).
A pretty 1-page resume that misses this posting's supportable keywords FAILS —
redraft until coverage clears that company's threshold.

CRITICAL — every application is a NEW rewrite for THAT JD:
- Do NOT reuse a previous company's summary/bullets verbatim.
- Change which projects/roles you emphasize AND how you describe them
  (same facts, different angle/vocabulary matching the posting).
- SDE JD → APIs, backend, shipping; AI JD → model integration in product;
  ML/CV JD → training, evaluation, YOLO/OpenCV/pipelines.
- Layout/theme stays locked; content must move with the JD.

Truth first: never invent employers, titles, dates, degrees, tools, or metrics.
If a JD keyword is not supported by the fact bank, omit it — do not stuff.

## How to extract JD must-haves
On a second mental pass of the JD, collect:
- Exact role title phrases (e.g., "Software Development Engineer Intern")
- Hard skills / tools / languages / frameworks written literally
- Methods & domains that repeat (REST APIs, DSA, model training, etc.)
- Soft requirements only when the candidate has evidence

Prefer the posting's exact spelling ("Node.js" vs "Node", "Scikit-learn" vs "sklearn").

## Where keywords earn score (descending credibility)
1. Recent experience / internship bullets with measurable outcomes
2. Professional summary (role target + 2–3 must-have terms)
3. Skills section as an index of terms also proven in bullets/projects
4. Project bullets that prove the same stack

A skill listed only in Skills with no evidence elsewhere is weak (orphan keyword).

## Bullet craft (industry standard)
- Start with strong action verbs (Built, Designed, Implemented, Optimized, Deployed)
- Mirror JD verbs when truthful ("managed" if JD says managed)
- Prefer impact: what + how + result (numbers only if present in fact bank)
- Keep bullets concise; no first-person; no fluff

## Structural ATS rules (RenderCV / parseability)
- Keep valid RenderCV YAML; theme monopoly is custom ``ace`` only (no stock themes)
- Standard section titles: summary, education, experience, projects, skills
- Contact in body fields (name/email/phone/social_networks), never invented headers
- Consistent date formats already used in the master YAML
- No tables, multi-column hacks, icons, or images in content
"""

# role_family registry seed — content emphasis only; chrome stays ace presets.
# Future multi-persona: map onboarding “studies + target roles” → these keys.
ROLE_FAMILIES = ("SDE", "AI", "ML", "PRODUCT", "GENERAL")

TRACK_GUIDANCE = {
    "SDE": (
        "Prioritize DSA, OOP, backend/API engineering, Node.js/TypeScript/Python, "
        "system fundamentals, collaborative Git workflows, and product-feature shipping."
    ),
    "AI": (
        "Prioritize applied AI in products: model integration, Python, TensorFlow/PyTorch, "
        "REST/API deployment of ML features, SaaS/backend bridging research to product."
    ),
    "ML": (
        "Prioritize ML pipelines: Scikit-learn/XGBoost/TensorFlow, feature engineering, "
        "training/evaluation, OpenCV/YOLO when relevant, and measurable model quality work."
    ),
    "PRODUCT": (
        "Prioritize AI product experimentation, prototypes, APIs, evaluation/testing, "
        "tooling (Cursor/Claude when factual), documentation, and cross-functional delivery."
    ),
}


def track_brief(job_track: str | None) -> str:
    key = (job_track or "").strip().upper()
    if key in TRACK_GUIDANCE:
        return f"Target track: {key}. {TRACK_GUIDANCE[key]}"
    return (
        "Target track: GENERAL engineering. Prefer the strongest overlap between "
        "the JD and the candidate's SDE/AI/ML evidence."
    )
