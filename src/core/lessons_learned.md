# Lessons Learned (agent self-training)

Operating rules appended by `core/reflect.py` after each evening checkout.
Tailor and executor inject this file into their prompts every run.
Seed copied into `data/lessons_learned.md` on first use.

## Baseline rules

1. Never invent employers, titles, metrics, or skills absent from master.yaml.
2. Always upload the configured apply PDF from disk via Upload resume — never select a previously listed LinkedIn resume.
3. On CAPTCHA / Cloudflare / bot walls, abort immediately and leave MANUAL_ACTION.
4. Prefer exact JD keyword phrasing only when truthfully supportable from the fact bank.
5. Prefer company career / ATS / LinkedIn Easy Apply URLs — never waste steps on Glassdoor listing walls.
6. Do not claim APPLICATION_SUBMITTED unless fresh upload_file success + confirmation UI.
7. After LinkedIn Easy Apply success, delete only the agent-uploaded resume; never delete the user's permanent keep-list resumes.
8. LinkedIn CTC boxes: plain digits only (current_ctc_inr=0, expected_ctc_inr≈650000) — no ₹/commas/"per annum".
9. Never put GitHub into portfolio/website fields; leave blank if portfolio_url is empty.
10. Experience dropdowns: years_of_experience=0, additional_months_of_experience=6 unless forced otherwise.
11. Notice period in days: single integer notice_period_days=7 — never type "7-14".
12. Education dates: University Aug 2024–May 2027 (currently attending). Month/year dropdowns → August/2024, May/2027.
13. LinkedIn Easy Apply is often 5–7 pages — keep going through education + additional questions to Submit; never stop at Review.
14. When a LinkedIn dialog intercepts clicks, force-click inside the Easy Apply dialog.
15. Unknown screening answer only: human toast/popup → learn into data/learned_form_answers.yaml. Never popup for resume upload.
16. Do not call semantic LLM for LinkedIn fills when Groq/NVIDIA are rate-limited — use master.yaml + learned answers deterministically.
17. Lumenci Product Intern pattern (reuse for similar Easy Apply): contact → resume → CTC/exp/notice/LinkedIn → education dates → Yes screening + tech list → Review → Submit.
