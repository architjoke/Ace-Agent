"""Interactive Chromium login bootstrap for Ace Agent.

Opens the shared persistent profile (data/browser_profile) so you can sign into
Gmail → LinkedIn → Indeed → Naukri → Wellfound → YC WaAS → Glassdoor →
We Work Remotely → Himalayas → Hiring Cafe.
Apply runs reuse those cookies via warm worker clones.

Usage:
  python scripts/bootstrap_login.py
  python scripts/bootstrap_login.py --verify
  python agent.py login
  python agent.py login --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ace_bootstrap_login")

LOGIN_URLS: list[tuple[str, str]] = [
    ("Gmail", "https://mail.google.com/"),
    ("LinkedIn", "https://www.linkedin.com/feed/"),
    ("Indeed", "https://www.indeed.com/"),
    ("Naukri", "https://www.naukri.com/mnjuser/homepage"),
    ("Wellfound", "https://wellfound.com/"),
    ("YC_WaAS", "https://www.workatastartup.com/"),
    ("Glassdoor", "https://www.glassdoor.co.in/index.htm"),
    # Redirect boards — login optional for many listings, but sessions help
    # saved jobs / fewer bot walls when following Apply → ATS.
    ("WeWorkRemotely", "https://weworkremotely.com/"),
    ("Himalayas", "https://himalayas.app/"),
    ("HiringCafe", "https://hiring.cafe/"),
]

# Human checklist printed during bootstrap (profile readiness).
PROFILE_CHECKLIST = """
Complete these before Ace can apply smoothly on native boards:

  Naukri
    • Logged in at naukri.com/mnjuser
    • Resume uploaded on profile
    • Headline / skills roughly current (playbook will NOT rewrite profile)

  Wellfound
    • Logged in
    • Profile complete (experience + skills + resume)
    • No “complete your profile” banner on a job page

  YC Work at a Startup  ← most common blocker
    • Logged in at workatastartup.com
    • Profile wizard finished (personal → share)
    • Job search status is NOT “I'm not looking / HIDE MY PROFILE”
    • Set: “I'm actively looking” OR “I'm open to new opportunities”
    • Profile visible so Apply is enabled

  Glassdoor (optional / last-priority)
    • Logged in (Easy Apply needs session)
    • Expect Cloudflare occasionally — Ace prefers external ATS links

  We Work Remotely / Himalayas / Hiring Cafe
    • Sign in if the site offers an account (saved jobs + warmer Apply redirects)
    • Apply itself usually leaves to Greenhouse/Lever/mailto — cookies still help
    • No profile “not looking” gate on these boards

  Mailto applies
    • EMAIL_ADDRESS + EMAIL_APP_PASSWORD set in .env (Gmail app password)
"""

_LOCK_NAMES = ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile")


def _profile_locked(profile_dir: Path) -> bool:
    for name in _LOCK_NAMES:
        lock = profile_dir / name
        if lock.exists():
            return True
    return False


def _ace_chrome_running(profile_dir: Path) -> bool:
    needle = str(profile_dir).lower().replace("/", "\\")
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if "chrome" not in name and "chromium" not in name:
                    continue
                cmd = " ".join(proc.info.get("cmdline") or []).lower()
                if needle in cmd.replace("/", "\\"):
                    return True
            except (psutil.Error, TypeError):
                continue
        return False
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import subprocess

            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                    "Select-Object -ExpandProperty CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            return needle in (completed.stdout or "").lower().replace("/", "\\")
        except Exception:
            return False
    return False


def _login_hints(url: str, title: str = "", body: str = "") -> bool:
    blob = f"{url} {title} {body[:500]}".lower()
    if any(
        marker in blob
        for marker in (
            "login",
            "signin",
            "sign-in",
            "authwall",
            "checkpoint",
            "accounts.google.com/servicelogin",
            "challenge",
        )
    ):
        return False
    if any(
        marker in blob
        for marker in (
            "mail.google.com/mail",
            "linkedin.com/feed",
            "indeed.com",
            "naukri.com/mnjuser",
            "wellfound.com",
            "workatastartup.com",
            "glassdoor.",
            "weworkremotely.com",
            "himalayas.app",
            "hiring.cafe",
        )
    ):
        return True
    return False


def _waas_status_hint(body: str) -> str | None:
    low = (body or "").lower()
    if "hide my profile" in low or "i'm not looking" in low or "not looking" in low:
        if "actively looking" not in low:
            return "likely_not_looking_or_hidden"
    if "finish profile" in low or "complete your profile" in low:
        return "profile_incomplete"
    return None


def run_bootstrap(*, verify: bool = False, wait_enter: bool = True) -> int:
    from playwright.sync_api import sync_playwright

    from src.core.settings import get_settings

    settings = get_settings()
    profile_dir = Path(settings.browser_user_data_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    if _ace_chrome_running(profile_dir):
        logger.error(
            "Chrome is already using %s. Close apply/daemon browsers, then retry.",
            profile_dir,
        )
        return 1

    for name in _LOCK_NAMES:
        lock = profile_dir / name
        if lock.exists() and not _ace_chrome_running(profile_dir):
            try:
                lock.unlink()
                logger.info("Removed stale profile lock: %s", name)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", name, exc)

    print("=" * 60)
    print("Ace Agent — browser identity bootstrap")
    print(f"Profile: {profile_dir}")
    print("Log into each tab (passwords + 2FA). Do NOT close the window yet.")
    print(
        "Order: Gmail → LinkedIn → Indeed → Naukri → Wellfound → YC WaAS → "
        "Glassdoor → We Work Remotely → Himalayas → Hiring Cafe"
    )
    print(PROFILE_CHECKLIST)
    print("=" * 60)

    status: dict[str, object] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": str(profile_dir),
        "sites": {},
        "action_needed": [],
    }

    with sync_playwright() as p:
        launch_kwargs = dict(
            user_data_dir=str(profile_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )
        try:
            context = p.chromium.launch_persistent_context(
                channel="chrome", **launch_kwargs
            )
        except Exception:
            context = p.chromium.launch_persistent_context(**launch_kwargs)
        try:
            first = True
            for label, url in LOGIN_URLS:
                page = context.new_page() if not first else (
                    context.pages[0] if context.pages else context.new_page()
                )
                first = False
                print(f"\n[*] Opening {label}: {url}")
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                except Exception as exc:
                    logger.warning("%s navigation issue: %s", label, exc)
                time.sleep(1.2)

            if wait_enter:
                print(
                    "\nComplete every login in the Chromium window "
                    "(Gmail first helps Google-SSO ATS sites)."
                )
                print(
                    "On YC WaAS: open Profile / Application and set status to "
                    "Actively looking (or Open to opportunities)."
                )
                input("Press ENTER here when finished logging in… ")

            if verify:
                print("\n[*] Verifying session heuristics…")
                sites: dict[str, dict] = {}
                action_needed: list[str] = []
                for i, (label, url) in enumerate(LOGIN_URLS):
                    page = context.pages[i] if i < len(context.pages) else context.new_page()
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                        time.sleep(1.5)
                        body = ""
                        try:
                            body = page.inner_text("body", timeout=3000) or ""
                        except Exception:
                            pass
                        ok = _login_hints(page.url, page.title(), body)
                        entry: dict = {
                            "url": page.url,
                            "title": page.title(),
                            "likely_logged_in": ok,
                        }
                        if label == "YC_WaAS":
                            hint = _waas_status_hint(body)
                            entry["waas_status"] = hint
                            if hint:
                                action_needed.append(f"YC_WaAS:{hint}")
                                print(f"    [FIX] YC WaAS status → {hint}")
                        if label == "Wellfound" and (
                            "complete your profile" in body.lower()
                            or "sign in" in body.lower()
                        ):
                            action_needed.append("Wellfound:profile_or_login")
                            print("    [FIX] Wellfound profile/login needs attention")
                        sites[label] = entry
                        mark = "OK" if ok else "CHECK"
                        print(f"    [{mark}] {label}: {page.url}")
                    except Exception as exc:
                        sites[label] = {"error": str(exc), "likely_logged_in": False}
                        action_needed.append(f"{label}:error")
                        print(f"    [FAIL] {label}: {exc}")
                status["sites"] = sites
                status["action_needed"] = action_needed
            else:
                status["sites"] = {
                    label: {"opened": url, "likely_logged_in": None}
                    for label, url in LOGIN_URLS
                }

            status_path = profile_dir / ".login_status.json"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"\n[✓] Wrote {status_path}")
            needed = status.get("action_needed") or []
            if needed:
                print("\nAction needed before native applies:")
                for item in needed:
                    print(f"  • {item}")
        finally:
            context.close()

    try:
        from src.core.browser_pool import clone_profiles_from_master
        from src.core.settings import get_settings as _gs

        n = _gs().browser_workers
        paths = clone_profiles_from_master(n, force=True)
        print(f"[✓] Cloned login profile → {len(paths)} worker profile(s)")
        for p in paths:
            print(f"    • {p}")
    except Exception as exc:
        logger.warning("Could not clone worker profiles: %s", exc)

    print(
        "\n[✓] Bootstrap complete. Cookies flushed to the shared profile.\n"
        "    Next: ace start   (or: ace --dry-run <url>)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Ace Chromium logins.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After login, re-open each site and write login heuristics",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for ENTER (for scripted open-only checks)",
    )
    args = parser.parse_args(argv)
    try:
        return run_bootstrap(verify=args.verify, wait_enter=not args.no_wait)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as exc:
        logger.exception("Bootstrap failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
