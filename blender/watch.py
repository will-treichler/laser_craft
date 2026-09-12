#!/usr/bin/env python3
"""Re-run the test suites whenever a source file changes.

The fastest debug loop is the one that never opens Blender's GUI. Core logic
bugs - unfolding, fold geometry, packing, export - are all reachable
headlessly, and this turns them around in seconds. Save the GUI for things
that are genuinely about the GUI.

    python3 blender/watch.py            # core tests only (fast)
    python3 blender/watch.py --addon    # also the headless Blender suite
    python3 blender/watch.py --once     # single pass, for CI
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCH = [ROOT / "cardboard_unfolder", ROOT / "tests",
         ROOT / "blender" / "cbu_blender", ROOT / "blender" / "test_addon.py"]


def snapshot() -> dict:
    out = {}
    for p in WATCH:
        if p.is_file():
            out[p] = p.stat().st_mtime
        elif p.is_dir():
            for f in p.rglob("*.py"):
                if "__pycache__" not in str(f):
                    out[f] = f.stat().st_mtime
    return out


def sync_vendor() -> None:
    """The add-on vendors the core, so keep the copy fresh while iterating."""
    src = ROOT / "cardboard_unfolder"
    dst = ROOT / "blender" / "cbu_blender" / "vendor" / "cardboard_unfolder"
    if not src.is_dir() or not dst.is_dir():
        return
    for f in src.glob("*.py"):
        t = dst / f.name
        # Content, not mtime. The vendored tree is a pure copy, so "the copy is
        # newer" is never a reason to keep it - and something else touching it
        # is exactly when it needs overwriting. test_dev.py edits the vendored
        # material.py to prove hot reload works; a killed run leaves that edit
        # in place with a fresh mtime, and an mtime sync then refuses to undo it
        # forever. The files are a few KB each.
        data = f.read_bytes()
        if not t.exists() or t.read_bytes() != data:
            t.write_bytes(data)


def run(addon: bool) -> bool:
    sync_vendor()
    ok = True
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q",
                        "--no-header", "-x"], cwd=ROOT)
    ok &= r.returncode == 0
    if addon and ok:
        r = subprocess.run([sys.executable, "blender/test_addon.py"], cwd=ROOT)
        ok &= r.returncode == 0
    print(f"\n{'PASS' if ok else 'FAIL'}  ({time.time() - t0:.1f}s)  "
          f"{time.strftime('%H:%M:%S')}\n" + "-" * 60)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addon", action="store_true",
                    help="also run the headless Blender suite (slower)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    if a.once:
        return 0 if run(a.addon) else 1

    print(f"watching {len(snapshot())} file(s) - Ctrl-C to stop")
    run(a.addon)
    prev = snapshot()
    try:
        while True:
            time.sleep(0.7)
            cur = snapshot()
            if cur != prev:
                changed = [p.name for p in set(cur) ^ set(prev)]
                changed += [p.name for p in set(cur) & set(prev)
                            if cur[p] != prev[p]]
                print(f"\nchanged: {', '.join(sorted(set(changed))[:6])}")
                run(a.addon)
                prev = snapshot()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
