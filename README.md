# Ace Job Agent

Open-source autonomous job agent: **hunt → fit → tailor → browser auto-submit → Manual_Action fallback → inbox → brief**.

Bring your own fact bank (`data/master.yaml`), API keys, and LinkedIn/Gmail login. The agent personalizes a 1-page resume per JD and applies on LinkedIn / Indeed / company ATS.

One CLI: **`ace`** (`python agent.py`). On **Windows PowerShell**, use `.\ace` (cwd commands need `.\`).

---

## Quick start

```bat
git clone <this-repo>
cd Ace_Agent_Monorepo
pip install -r requirements.txt
playwright install chromium
copy .env.example .env
copy data\master.yaml.example data\master.yaml
```

1. Edit **`.env`** — `GOOGLE_API_KEY`, `GROQ_API_KEY`, optional `NVIDIA_API_KEY`, email, and:
   - `ACE_APPLY_RESUME_FILENAME=Your_Name_Resume.pdf`
   - `ACE_LINKEDIN_KEEP_RESUMES=…` (permanent LinkedIn library names to never delete)
2. Edit **`data/master.yaml`** — only truthful skills, roles, and projects (RenderCV schema).
3. Tweak **`config.yaml`** — locations, tracks, fit score (optional).
4. Log in once:

```bat
ace login --verify
```

5. Run:

```bat
.\ace start
.\ace status
.\ace stop
```

Equivalent: `python agent.py start` (always works).

---

## Layout

```
agent.py                 # public CLI
config.yaml / .env       # preferences + secrets (env is gitignored)
src/core/                # discovery, tailor, executor, pipeline, …
scripts/                 # daemon, dry-run, Task Scheduler bats
data/                    # your master.yaml, SQLite, browser profiles (gitignored)
output/                  # generated PDFs / Manual_Action / logs (gitignored)
resumes/                 # optional personal PDFs (gitignored)
tests/
```

---

## Commands

| Command | What it does |
|---|---|
| `ace start` / `ace stop` | Background uncapped apply loop |
| `ace status` / `ace manual` / `ace ask …` | Progress & Manual_Action queue |
| `ace once` | One hunt batch (foreground) |
| `ace login --verify` | Chromium login bootstrap |
| `ace daemon` | Schedule: morning apply + evening inbox/brief |
| `ace inbox` / `ace brief` | Inbox scan / daily Markdown report |
| `ace chat` | HITL Copilot: mark submitted, clean Manual_Action, memorize HARD/SOFT rules |

```bat
python scripts\test_live_run.py
python scripts\live_demo_apply.py --limit 2
```

---

## Privacy / what is NOT pushed

`.gitignore` excludes:

- `.env`, API keys, email passwords  
- `data/master.yaml`, SQLite DB, Chromium profiles (cookies/sessions)  
- `output/**` (tailored PDFs, Manual_Action, logs)  
- `resumes/**` (personal files)  

Commit only code + examples (`*.example`, `.env.example`, empty `output/` / `data/` placeholders).

---

## Notes

- Always **Upload** a fresh PDF from disk; never re-select an old LinkedIn library resume.
- LinkedIn Preferences cleanup deletes **only** `ACE_APPLY_RESUME_FILENAME` (batch per cycle).
- Soft LinkedIn rate limit: ~10 Easy Applies/hour (`ACE_LINKEDIN_MAX_PER_HOUR`).
- Manual_Action = CAPTCHA / login wall / no online apply — packages land in `output/Manual_Action/`.
- Task Scheduler: set working directory to the repo root (`scripts\run_once.bat`, `scripts\reflect.bat`).
