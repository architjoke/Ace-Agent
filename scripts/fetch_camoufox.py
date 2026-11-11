"""Fetch + verify Camoufox binary so Jobdrop Hiring Cafe / Wellfound unlock."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_ZIP_NAME_RE = re.compile(
    r"camoufox-(?P<version>[^-]+)-(?P<release>.+)-win\.[^.]+\.zip$",
    re.I,
)


def run_fetch(*, force: bool = False, from_zip: Path | None = None) -> int:
    """Download Camoufox if missing (or always when force=True). Returns exit code."""
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from src.core.discovery_extra import _camoufox_cached

    cached = _camoufox_cached()
    if cached and not force and from_zip is None:
        print("Camoufox already cached — skipping download.")
        _print_status()
        return 0

    if from_zip is not None:
        code = _install_from_zip(Path(from_zip))
        if code != 0:
            return code
    else:
        print("Fetching Camoufox binaries (one-time ~1.3GB)…")
        completed = subprocess.run(
            [sys.executable, "-m", "camoufox", "fetch"],
            cwd=str(repo),
            check=False,
        )
        if completed.returncode != 0:
            print(
                f"camoufox fetch failed (exit {completed.returncode})",
                file=sys.stderr,
            )
            return completed.returncode or 1

    if not _camoufox_cached():
        print(
            "Fetch finished but cache probe still False — check INSTALL_DIR",
            file=sys.stderr,
        )
        return 2
    _print_status()
    return 0


def _install_from_zip(zip_path: Path) -> int:
    """Install a pre-downloaded Camoufox release zip into INSTALL_DIR."""
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.is_file():
        print(f"zip not found: {zip_path}", file=sys.stderr)
        return 1
    match = _ZIP_NAME_RE.search(zip_path.name)
    if not match:
        print(
            f"unexpected zip name (want camoufox-<ver>-<release>-win.*.zip): {zip_path.name}",
            file=sys.stderr,
        )
        return 1
    try:
        import orjson
        from camoufox.pkgman import INSTALL_DIR
    except Exception as exc:
        print(f"camoufox pkgman unavailable: {exc}", file=sys.stderr)
        return 1

    install = Path(INSTALL_DIR)
    if install.exists():
        shutil.rmtree(install)
    install.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} → {install}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(install)
    (install / "version.json").write_bytes(
        orjson.dumps(
            {"version": match.group("version"), "release": match.group("release")}
        )
    )
    print("Wrote version.json")
    return 0


def _print_status() -> None:
    from src.core.discovery_extra import _camoufox_cached, jobdrop_sites_for_hunt

    try:
        from camoufox.pkgman import INSTALL_DIR, camoufox_path

        print(f"INSTALL_DIR: {INSTALL_DIR}")
        try:
            print(f"executable: {camoufox_path(download_if_missing=False)}")
        except Exception as exc:
            print(f"executable: (unavailable: {exc})")
    except Exception as exc:
        print(f"pkgman unavailable: {exc}")
    print(f"cached: {_camoufox_cached()}")
    print(f"jobdrop sites: {jobdrop_sites_for_hunt()}")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    force = "--force" in argv
    if "--status" in argv:
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        _print_status()
        from src.core.discovery_extra import _camoufox_cached

        return 0 if _camoufox_cached() else 1
    from_zip: Path | None = None
    if "--from-zip" in argv:
        i = argv.index("--from-zip")
        if i + 1 >= len(argv):
            print("--from-zip requires a path", file=sys.stderr)
            return 2
        from_zip = Path(argv[i + 1])
    return run_fetch(force=force, from_zip=from_zip)


if __name__ == "__main__":
    raise SystemExit(main())