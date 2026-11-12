# General form-fill guide (shared — no personal facts)

Playbooks navigate ATS UI. This guide + `form_rules` / `FormFillEngine` decide
**what kind of field** is showing and **how** to fill it from the user's local
`data/master.yaml`. Never put personal answers in playbooks or this file.

## Fill order (Easy Apply / ATS)

1. Upload resume from disk (configured apply filename) — never reuse library resumes.
2. Identity: name, email, phone, city, LinkedIn URL.
3. Compensation: current / expected CTC as plain digits when the box shows Example numbers.
4. Availability: notice period in **days** as a single integer.
5. Experience: years + additional months as dropdown-friendly integers.
6. Education: school dates (month/year) + currently attending when true.
7. Screening yes/no: map to eligibility prefs in the profile (internship length, hours/day, etc.).
8. Skills / technologies: derive from profile skills ∩ job description tools — never invent stack.
9. Portfolio / website: only if profile has a real portfolio URL; never paste GitHub there.
10. Review → Submit. Confirm success UI before claiming submitted.

## Field kinds (what the form is asking)

| Kind | Means | Source |
|------|--------|--------|
| identity_* | Contact / name | profile application_answers / cv |
| compensation_* | Pay numbers | current_ctc_inr / expected_ctc_inr |
| availability_* | Notice / join | notice_period_days |
| experience_* | YoE / months | years_of_experience, additional_months_of_experience |
| education_* | School dates / attending | university_* keys |
| yes_no_eligibility | Commitment / fit booleans | eligibility prefs in profile |
| skills_tech | Tools / stack free text | cv skills + JD keywords |
| freeform | Open prose | only matching narrative profile keys |
| upload_resume | File input | disk PDF path from playbook |
| unknown | Cannot classify or resolve | human popup once → learn kind/intent |

## Formatting rules

- CTC Example boxes: digits only (no ₹, commas, "per annum").
- Notice "in days": single integer (e.g. 7), never "7-14".
- Portfolio empty → leave blank; never GitHub.
- Mechanical stuck (upload / click / timeout): Ace retries — never human popup.
- Unknown screening content only: human toast → store as kind/intent override, not employer prose as the primary brain.

## Multi-user

Each person keeps personal facts in local `data/master.yaml` (gitignored).
This guide and the playbooks are shared. New user = new master.yaml only.

## Learn → promote (local only)

1. Unknown screening field → human toast → answers stored in `data/learned_form_answers.yaml` (gitignored).
2. Fixed keys (pin, city, phone, email, URLs, notice/CTC integers, school dates, …) are **promoted** into local `master.yaml` → `application_answers` when the slot is empty, or after the same new value is confirmed twice (never wipe a conflicting profile fact on a single bad teach).
3. Next Easy Apply / ATS fill reads master first via FormFillEngine — no personal answers in playbooks or git.
