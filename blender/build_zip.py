#!/usr/bin/env python3
"""Build cbu_blender.zip for `Install from Disk`.

Prefer `dev_install.py` while developing - a symlink plus the add-on's Reload
button avoids rebuilding entirely. Use this when you want a self-contained zip
to install normally, or to hand to someone else.

    python blender/build_zip.py
    python blender/build_zip.py -o ~/Desktop
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "cbu_blender"
CORE = HERE.parent / "cardboard_unfolder"
VENDOR = SRC / "vendor" / "cardboard_unfolder"


def sync_vendor() -> list[str]:
    """The zip has to carry its own copy of the core, so refresh it first.
    Stale vendored code is the easiest way to ship a zip that behaves
    differently from your working tree."""
    changed = []
    VENDOR.mkdir(parents=True, exist_ok=True)
    for f in sorted(CORE.glob("*.py")):
        dst = VENDOR / f.name
        data = f.read_bytes()
        if not dst.exists() or dst.read_bytes() != data:
            dst.write_bytes(data)
            changed.append(f.name)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE),
                    help="directory to write cbu_blender.zip into")
    a = ap.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"add-on source not found: {SRC}")

    changed = sync_vendor()
    if changed:
        print(f"synced vendored core: {', '.join(changed)}")

    out = Path(a.out).expanduser().resolve() / "cbu_blender.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SRC.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            z.write(f, Path("cbu_blender") / f.relative_to(SRC))
            n += 1

    print(f"wrote {out}  ({n} files, {out.stat().st_size // 1024} KB)")
    print("Blender: Preferences > Add-ons > Install from Disk > pick that zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
