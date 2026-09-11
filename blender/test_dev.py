"""Verify the debug tooling itself: hot reload really picks up edited code,
repeated reloads don't leak handlers or classes, and the dump/self-test run.

Run:  python3 blender/test_dev.py
"""
import sys
import time
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Start hermetic: a previous run that edited a file and restored it within
# the same second can leave a .pyc that Python still considers valid.
for _d in (ROOT / "cbu_blender").rglob("__pycache__"):
    for _f in _d.glob("*.pyc"):
        _f.unlink(missing_ok=True)

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


print("blender", bpy.app.version_string)

import cbu_blender
from cbu_blender import dev, ops, overlay, log as cbulog

cbu_blender.register()
bpy.ops.mesh.primitive_cube_add(size=0.1)
bpy.context.scene.cbu.trials = 30

# ------------------------------------------------------------- self-test
check(bpy.ops.cbu.selftest() == {"FINISHED"}, "self-test operator passes")

# ------------------------------------------------------- reload picks up edits
target = ROOT / "cbu_blender" / "vendor" / "cardboard_unfolder" / "material.py"
# Bytes, not text: read_text/write_text round-trip through os.linesep, so on
# Windows the restore in the finally block would rewrite every LF as CRLF and
# leave the vendored file permanently dirty in git.
original = target.read_bytes()
try:
    from cbu_blender.vendor.cardboard_unfolder.material import Material
    before = Material().crush
    check(abs(before - 0.75) < 1e-9, f"crush default starts at 0.75 (got {before})")

    target.write_bytes(original.replace(b"crush: float = 0.75",
                                        b"crush: float = 0.61"))
    # without a reload the old value must still be live
    from cbu_blender.vendor.cardboard_unfolder.material import Material as M2
    check(abs(M2().crush - 0.75) < 1e-9,
          "edit on disk is NOT live before reload (as expected)")

    ok, msg = dev.reload_all()
    check(ok, f"reload_all succeeded: {msg}")

    import importlib
    mod = sys.modules["cbu_blender.vendor.cardboard_unfolder.material"]
    check(abs(mod.Material().crush - 0.61) < 1e-9,
          f"reload picked up the edited value (got {mod.Material().crush})")

    # the addon must still work after reloading
    check(hasattr(bpy.types.Scene, "cbu"), "settings re-registered after reload")
    check(bpy.ops.cbu.unfold() == {"FINISHED"}, "unfold still works after reload")
finally:
    target.write_bytes(original)
    dev.reload_all()

mod = sys.modules["cbu_blender.vendor.cardboard_unfolder.material"]
check(abs(mod.Material().crush - 0.75) < 1e-9, "restored value after reload")

# ------------------------------------------------- repeated reloads are safe
t0 = time.time()
for i in range(6):
    ok, msg = dev.reload_all()
    if not ok:
        check(False, f"reload #{i + 1} failed: {msg}")
        break
else:
    check(True, f"6 consecutive reloads clean ({time.time() - t0:.1f}s)")

handlers = [h for h in bpy.app.handlers.load_post
            if getattr(h, "__name__", "") == "_on_load"]
check(len(handlers) == 1,
      f"load_post handler registered exactly once after 8 reloads "
      f"(got {len(handlers)})")
check(hasattr(bpy.types.Scene, "cbu"), "settings survive repeated reloads")
check(bpy.ops.cbu.unfold() == {"FINISHED"}, "unfold works after repeated reloads")

# ------------------------------------------------------ unregister is clean
cbu_blender.unregister()
check(not hasattr(bpy.types.Scene, "cbu"), "unregister removes settings")
check(not [h for h in bpy.app.handlers.load_post
           if getattr(h, "__name__", "") == "_on_load"],
      "unregister removes the load_post handler")
cbu_blender.register()

# -------------------------------------------------------------- state dump
bpy.ops.cbu.unfold()
d = dev.state_dict(bpy.context)
for key in ("blender", "python", "settings", "deps", "object",
            "last_run", "timings_ms"):
    check(key in d, f"state dump has '{key}'")
check(d["timings_ms"] and all(v >= 0 for v in d["timings_ms"].values()),
      f"timings recorded: {d['timings_ms']}")
check(d["last_run"]["islands"], "state dump lists islands")
import json
try:
    json.dumps(d)
    check(True, "state dump is JSON-serialisable")
except TypeError as exc:
    check(False, f"state dump not serialisable: {exc}")

check(bpy.ops.cbu.dump_state() == {"FINISHED"}, "dump_state operator runs")

# --------------------------------------------------------------- logging
cbulog.set_level("DEBUG")
check(cbulog.get().level == 10, "log level applies")
p = Path("/tmp/cbu_test.log")
cbulog.set_file(str(p))
cbulog.get().warning("hello from the test")
cbulog.set_file(None)
check(p.exists() and "hello from the test" in p.read_text(),
      "file logging captures output")
cbulog.set_level("WARNING")

# ------------------------------------------------- load_post clears caches
bpy.ops.cbu.unfold()
check(bool(ops._LAST), "results cached before file load")
bpy.ops.wm.read_homefile(use_empty=True)
check(not ops._LAST, "load_post cleared stale results (prevents crashes)")

cbu_blender.unregister()

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("all dev-tooling checks passed")
