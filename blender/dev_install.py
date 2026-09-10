#!/usr/bin/env python3
"""Link this repo into Blender's add-ons folder so edits are live.

Installing the zip copies the code, so every change needs a reinstall. A
symlink means the file you edit IS the file Blender runs - combine it with
the add-on's Reload button and the loop is: save, click Reload, re-run.

    python3 blender/dev_install.py                 # auto-detect Blender
    python3 blender/dev_install.py --version 4.5
    python3 blender/dev_install.py --scripts ~/blender-scripts
    python3 blender/dev_install.py --remove

On Windows, symlinks need either Developer Mode or an elevated prompt; the
script falls back to a directory junction, which works unprivileged.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "cbu_blender"
NAME = "cbu_blender"


def candidate_roots() -> list[Path]:
    home = Path.home()
    sysname = platform.system()
    if sysname == "Darwin":
        base = home / "Library/Application Support/Blender"
    elif sysname == "Windows":
        base = Path(os.environ.get("APPDATA", home)) / "Blender Foundation/Blender"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "blender"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        if d.is_dir() and d.name[0].isdigit():
            out.append(d)
    return out


def addons_dir(version: str | None, scripts: str | None) -> Path:
    if scripts:
        return Path(scripts).expanduser().resolve() / "addons"
    roots = candidate_roots()
    if not roots:
        sys.exit("could not find a Blender config folder - pass --scripts "
                 "<dir> and add it under Preferences > File Paths > Scripts")
    if version:
        for r in roots:
            if r.name == version:
                return r / "scripts/addons"
        sys.exit(f"no Blender {version} found. Available: "
                 f"{', '.join(r.name for r in roots)}")
    return roots[0] / "scripts/addons"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="Blender version folder, e.g. 4.5")
    ap.add_argument("--scripts", help="custom scripts folder instead of the default")
    ap.add_argument("--remove", action="store_true", help="unlink and exit")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of linking (no live editing)")
    a = ap.parse_args()

    dest_dir = addons_dir(a.version, a.scripts)
    dest = dest_dir / NAME

    if a.remove:
        if dest.is_symlink():
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest)
        print(f"removed {dest}")
        return 0

    if not SRC.is_dir():
        sys.exit(f"source not found: {SRC}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)

    if a.copy:
        shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns("__pycache__"))
        how = "copied"
    else:
        try:
            dest.symlink_to(SRC, target_is_directory=True)
            how = "linked"
        except OSError:
            if platform.system() == "Windows":
                subprocess.check_call(["cmd", "/c", "mklink", "/J",
                                       str(dest), str(SRC)])
                how = "junctioned"
            else:
                raise

    print(f"{how}  {dest}  ->  {SRC}")
    print()
    print("Next:")
    print("  1. Blender: Preferences > Add-ons > enable 'Cardboard Unfolder'")
    print("  2. First run: click Install Dependencies, restart once")
    print("  3. After that, edit code and press Reload Add-on in the")
    print("     Cardboard > Debug panel. No restart, scene preserved.")
    print()
    print("Launch Blender from a terminal so you can see print() and")
    print("tracebacks - on macOS and Linux there is no system console window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
