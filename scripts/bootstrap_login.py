"""Interactive Chromium login bootstrap for Ace Agent.

Opens the shared persistent profile (data/browser_profile) so you can sign into
Gmail → LinkedIn → Indeed → Naukri → Wellfound. Apply runs reuse those cookies.

Usage:
  python scripts/bootstrap_login.py
  python scripts/bootstrap_login.py --verify
  python agent.py --bootstrap-login
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
]

_LOCK_NAMES = ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile")


def _profile_locked(profile_dir: Path) -> bool:
    for name in _LOCK_NAMES:
        lock = profile_dir / name
        if lock.exists():
            # Stale locks are common after a crash; warn but only hard-fail if
            # Chrome processes still hold the Ace profile.
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
    # Windows fallback without psutil
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


def _login_hints(url: str, title: str = "") -> bool:
    blob = f"{url} {title}".lower()
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
        )
    ):
        return True
    return False


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
    print("Order: Gmail → LinkedIn → Indeed → Naukri → Wellfound")
    print("=" * 60)

    status: dict[str, object] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": str(profile_dir),
        "sites": {},
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
                input("Press ENTER here when finished logging in… ")

            if verify:
                print("\n[*] Verifying session heuristics…")
                sites: dict[str, dict] = {}
                for i, (label, url) in enumerate(LOGIN_URLS):
                    page = context.pages[i] if i < len(context.pages) else context.new_page()
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                        time.sleep(1.5)
                        ok = _login_hints(page.url, page.title())
                        sites[label] = {
                            "url": page.url,
                            "title": page.title(),
                            "likely_logged_in": ok,
                        }
                        mark = "OK" if ok else "CHECK"
                        print(f"    [{mark}] {label}: {page.url}")
                    except Exception as exc:
                        sites[label] = {"error": str(exc), "likely_logged_in": False}
                        print(f"    [FAIL] {label}: {exc}")
                status["sites"] = sites
            else:
                status["sites"] = {
                    label: {"opened": url, "likely_logged_in": None}
                    for label, url in LOGIN_URLS
                }

            status_path = profile_dir / ".login_status.json"
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"\n[✓] Wrote {status_path}")
        finally:
            context.close()

    # Clone master → browser_profile_w1…wn for parallel applies.
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
