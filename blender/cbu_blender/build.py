"""Which code is this Blender actually running?

A running Blender keeps the modules it imported, so a commit, a vendor sync
or a half-finished Reload can leave the panel drawing one version while the
files on disk say another - and nothing on screen said which. Twice that was
mistaken for a real bug: 6 mm fingers after the 3 mm fix had landed, and
lopsided joints after their fix, when Reload was skipping fingers.py.

So the Debug panel carries a stamp: the git commit the files came from, a
fingerprint of the add-on's own source AS LOADED, and when it was loaded.
The fingerprint is the honest part - it covers uncommitted edits and a
vendored copy that git knows nothing about - and comparing it with the files
now on disk is what says "press Reload".

Computed when this module is imported, which `register()` and every Reload
do, so it describes the code that came in with it.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent


def _sources() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def fingerprint() -> str:
    """Eight hex digits over every .py in the add-on, vendored core included.
    Paths go in as well as bytes, so a renamed file changes it too."""
    h = hashlib.sha1()
    for p in _sources():
        h.update(p.relative_to(PKG).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def _git() -> str:
    """Short commit, plus "+dirty" when the add-on's files differ from it.
    Empty outside a git checkout (an installed zip), and never slow enough
    to matter: this runs at import, so it gives up after two seconds."""
    try:
        def run(*args):
            return subprocess.run(["git", "-C", str(PKG), *args],
                                  capture_output=True, text=True, timeout=2,
                                  check=True).stdout.strip()
        sha = run("rev-parse", "--short", "HEAD")
        dirty = run("status", "--porcelain", "--", ".")
        return sha + ("+dirty" if dirty else "")
    except Exception:                                       # noqa: BLE001
        return ""


CODE = fingerprint()
GIT = _git()
LOADED_AT = time.time()

_disk = {"t": 0.0, "code": CODE}


def disk_code(max_age: float = 2.0) -> str:
    """The fingerprint of the files on disk NOW. Cached for `max_age`
    seconds, because a panel redraws continuously and hashing every source
    file on each redraw is work nobody asked for."""
    now = time.monotonic()
    if now - _disk["t"] > max_age:
        try:
            _disk["code"] = fingerprint()
        except OSError:
            pass
        _disk["t"] = now
    return _disk["code"]


def stale(max_age: float = 2.0) -> bool:
    """True when the files on disk are not the code that is running."""
    return disk_code(max_age) != CODE


def stamp() -> str:
    return "  ".join(stamp_lines())


def stamp_lines() -> list[str]:
    """The stamp as short lines, for the sidebar: on one line it truncated
    to "ba61aee+d... loaded 18:17:22" at a normal panel width."""
    loaded = time.strftime("%H:%M:%S", time.localtime(LOADED_AT))
    return [f"Build {GIT or 'no git'}", f"Code {CODE}  {loaded}"]
