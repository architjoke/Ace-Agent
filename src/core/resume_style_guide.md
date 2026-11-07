# Ace Resume Style Guide (Content & Phrasing Only)

Distilled from the `resumes/` reference set (elite ATS templates + selected tech/ML resumes)
and Archit's hand-made SDE/AI/ML 1-pagers.  
**Visual layout is owned by RenderCV custom theme `ace` (labeled contacts + fresher spacing).**  
**Truth constraint:** only use employers, tools, dates, and metrics that exist in `master.yaml`.  
`master.yaml` is a **fact bank**, never dump it wholesale into the PDF.

---

## 0. Hard Fresher Layout Budget (1 page max)

1. Output must fit **exactly one A4 page**. Prefer dense, scannable content — **no sparse whitespace**.
2. **Education:** college/university only (B.Tech). At most one highlight: `CGPA: X.X/10`.  
   No coursework, no campus leadership under Education (those belong in Skills / Activities).  
   Drop Class X / Class XII.
3. **Section order (exact):** Summary → Education → Experience → Projects → Skills → Hackathons & Activities.
4. **Experience:** exactly **2 JD-best roles** (prefer MSG + Infosys); each role has **3–4** weighty XYZ bullets  
   rewritten only from the fact bank. Never invent % / latency / req/day metrics.  
   Drop weak campus filler roles when two stronger roles exist.
5. **Projects:** exactly **2** JD-best projects (prefer Road Inspection + Earthquake / DCGAN); each has **3–4**  
   impressive technical bullets from the fact bank. Never include Another U.  
   Put GitHub in `location` as `[GitHub](url)` — **no project year**.  
   Hard rule: final PDF must be **exactly one A4 page**.
6. **Summary:** **1 dense paragraph** (handmade tone): ML + data workflow + backend SaaS. Stay truthful.
7. **Skills:** 4–5 labeled groups from the fact bank; JD-overlapping skills first.
8. **Hackathons & Activities:** 2–3 compact bullets only from facts (RIDEhack, LNMIIT, IIT Madras, Amazon ML, NEP Saarthi, workshops, Infosys certs, Top Ranker). No invented clubs/hackathons.
9. No “Last updated”, no “Page X of Y”, photos, skill bars, or multi-column tricks.  
   Design chrome (`ace` theme + spacing presets) is applied in code — do not invent a theme.
10. **XYZ law:** Accomplished X as measured by Y (only if in fact bank), by doing Z.  
    If no metric exists, write a concrete Z-rich technical bullet (stack, APIs, data, method) — never fluff.

### Bold for impact (RenderCV Markdown)
- In bullets, bold **2–4 high-signal tokens** only: tech stack, metrics, outcomes.  
  Example: `Built **REST APIs** in **Node.js**, cutting response overhead for SaaS sessions.`
- Do **not** bold whole sentences or every word.

---

## 1. Core Writing Principles

1. **Lead with impact, not duties.** Every strong bullet answers: what changed because you did the work?
2. **Prefer the XYZ formula:**  
   `Accomplished [X] as measured by [Y], by doing [Z].`  
   Example pattern: `Improved [outcome] by [metric], by [method/tech].`
3. **Mirror the job description’s exact technical vocabulary** when truthful (e.g., “Data Preprocessing”, “REST API”, “Model Deployment”).
4. **One idea per bullet.** Dense but scannable; typically 1–2 lines.
5. **No fluff / no soft claims without evidence.** Avoid “passionate”, “hardworking”, “responsible for”, “helped with”, “various”, “etc.”
6. **Active voice only.** Start with a strong past-tense (or strong present for current role) action verb.
7. **Keyword evidence pairing.** If a skill appears in Skills, it should also appear in an Experience/Project bullet.

---

## 2. Bullet Point Structure (Strict)

Use this order inside each bullet when possible:

1. **Action verb** (elite / specific)
2. **Technical object** (system, model, API, pipeline, feature)
3. **Method / stack** (tools used)
4. **Outcome / metric** (only if present in fact bank)

### Preferred patterns
- `Built/Designed/Implemented [system] using [stack] to [business/technical outcome].`
- `Optimized [component] by [technique], improving [metric] by [N% / Nx].`
- `Developed [pipeline/feature] that [enabled capability], processing/supporting [scale].`
- `Collaborated with [team] to ship [feature], resulting in [measurable result].`

### Weak patterns to rewrite
- `Worked on ...`
- `Responsible for ...`
- `Helped the team ...`
- `Familiar with ...`
- Duties lists with no outcome

---

## 3. Powerful Action Verbs Observed in References

Prefer these (and similar concrete verbs):

**Build / create:** Built, Designed, Developed, Implemented, Architected, Engineered, Created, Prototyped  
**Improve / scale:** Optimized, Accelerated, Reduced, Improved, Streamlined, Refactored, Scaled  
**Analyze / ML:** Analyzed, Trained, Evaluated, Fine-tuned, Deployed, Preprocessed, Feature-engineered  
**Lead / ship:** Led, Owned, Delivered, Shipped, Launched, Coordinated, Mentored  
**Integrate / automate:** Integrated, Automated, Orchestrated, Instrumented, Migrated

Avoid weak openers: Helped, Assisted (unless unavoidable), Worked on, Participated in (without ownership).

---

## 4. Metrics & Quantification Rules

From high-signal “SELECTED” tech resumes in the reference set:

- Prefer **hard numbers**: %, x-speedup, latency, accuracy, users, files processed, contests rank, CGPA.
- Place metrics **near the end** of the bullet after the method.
- If the fact bank has no metric, write a crisp capability bullet **without inventing numbers**.
- Competitive programming / rankings belong in Achievements, not forced into every experience bullet.
- Project bullets should name **stack + what it does + one proof point** (accuracy, users, speedup, scale) when available.

---

## 5. Section Content Strategy (ATS + Recruiter)

### Summary
- **3 dense lines** (hand-made resume tone) that fill header space.
- Lead with CSE + ML workflow keywords for AI/ML tracks; lead with backend/SaaS for SDE.
- Always keep one truthful bridge to the current SaaS/Node.js internship when space allows.
- No autobiography; no soft personality adjectives (“passionate”, “eager”, “seeking”).

### Skills
- Group by category labels (`Programming Languages`, `AI & Machine Learning`, `Backend & Web`, etc.).
- Put **JD-overlapping skills first**.
- Use exact JD spellings (`Scikit-learn`, `Node.js`, `REST APIs`) when supported.
- Do not list tools with zero evidence elsewhere.

### Experience / Projects
- Experience: **2 roles × 3–4 bullets** (XYZ, dense).
- Projects: **2–3 entries × 3–4 bullets** most aligned to the track/JD.
- End project `summary` with tech tags when useful (`Python, FastAPI, MySQL`).
- Bullets should sound like shipped work, not coursework regurgitation.

### Education
- Degree, university, dates only — **zero bullet points** under Education.
- CGPA may appear as a single optional highlight (`CGPA: 8.8/10`) or be omitted; never coursework/leadership bullets here.
- Prefer omitting Class X/XII for fresher 1-pagers.

### Projects (role-effective wording)
- Rewrite project bullets for the **target JD/role**, not a frozen sample or generic dump.
- Same project can read differently per posting (e.g. Road Inspection → Flask/APIs for SDE,
  YOLOv8/OpenCV for ML/CV) — facts stay true; emphasis follows the JD.
- SDE: APIs, backend, full-stack, reliability, shipping.
- AI/ML: models, preprocessing, evaluation, pipelines, accuracy/impact when truthful.
- Lead with verb + technical object + stack + **impact** (capability or metric from fact bank).

---

## 6. Tone

- Professional engineering tone: precise, confident, terse.
- Recruiter-readable in 6–10 seconds per section.
- Prefer specificity over buzzwords (“Celery task queue”, “YOLO detection pipeline”) when fact bank supports it.
- Never copy names, employers, or metrics from reference resumes — only their **phrasing craft**.

---

## 7. Hard Avoid List

- Passive voice (“was responsible”, “duties included”)
- First person (“I developed”)
- Objective statements (“Seeking a challenging role…”)
- Keyword stuffing without evidence
- Invented metrics, tools, titles, or employers
- Multi-column content tricks, skill bars, photos, icons-as-text replacements (RenderCV handles structure; keep YAML clean)

---

## 8. Self-Check Before Finalizing a Draft

For each bullet, verify:
1. Starts with a strong verb?
2. Contains a concrete technical object?
3. Uses at least one JD-relevant keyword **truthfully**?
4. Has a metric **only if** present in `master.yaml`?
5. Would a classical ATS string-match find the important phrases?

Also verify the **company/JD ATS gate**: coverage of supportable JD keywords must meet
that posting’s pass score (FAANG overrides / JD hardness in `config.yaml` → `ats`).
Pretty layout without clearing the gate is a failed draft — rewrite until it passes.

If any answer is no, rewrite the bullet before emitting YAML.
