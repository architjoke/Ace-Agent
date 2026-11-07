# Lessons Learned (agent self-training)

Operating rules appended by `core/reflect.py` after each evening checkout.
Tailor and executor inject this file into their prompts every run.

## Baseline rules

1. Never invent employers, titles, metrics, or skills absent from master.yaml.
2. Always upload the configured apply PDF from disk via Upload resume — never select a previously listed LinkedIn resume.
3. On CAPTCHA / Cloudflare / bot walls, abort immediately and leave MANUAL_ACTION.
4. Prefer exact JD keyword phrasing only when truthfully supportable from the fact bank.
5. Prefer company career / ATS / LinkedIn Easy Apply URLs — never waste steps on Glassdoor listing walls.
6. Do not claim APPLICATION_SUBMITTED unless fresh upload_file success + confirmation UI.
7. After LinkedIn Easy Apply success, delete only the agent-uploaded resume; never delete the user's permanent keep-list resumes.
