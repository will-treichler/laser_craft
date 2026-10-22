"""Developer tools: hot reload, state dump, self-test, debugpy attach.

Hot reload is the one that matters. Without it every code change costs a
Blender restart plus rebuilding your scene. With it the loop is: edit file,
press one button, run the operator again.

The subtlety is that ``importlib.reload`` on the top package does NOT reload
its submodules, and this addon has a whole vendored core underneath. So the
submodules are reloaded explicitly, in dependency order, with the addon
unregistered first - otherwise Blender is left holding classes that no longer
exist, which is how you get a crash rather than an error.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

import bpy

from . import log

# Dependency order. Leaves first: reloading a module rebinds the names it
# imported at module level, so anything it depends on must already be fresh.
# Dependency order: a module is reloaded after everything it imports from, or
# its `from .x import y` binds the stale y. Every module that is ever loaded
# must be here - fingers and cost were not, so Reload kept running the old
# finger code under a freshly reloaded layout (test_dev: "every loaded add-on
# module is reloaded").
CORE_MODULES = [
    "vendor",
    "vendor.cardboard_unfolder.material",
    "vendor.cardboard_unfolder.strokefont",
    "vendor.cardboard_unfolder.mesh",
    "vendor.cardboard_unfolder.unfold",
    "vendor.cardboard_unfolder.cost",
    "vendor.cardboard_unfolder.folds",
    "vendor.cardboard_unfolder.tabs",
    "vendor.cardboard_unfolder.fingers",
    "vendor.cardboard_unfolder.layout",
    "vendor.cardboard_unfolder.export",
    "vendor.cardboard_unfolder.gcode",
    "vendor.cardboard_unfolder",
]
# build first: it fingerprints the files as they are loaded, so it should
# describe the code the modules below are about to pick up.
ADDON_MODULES = ["build", "log", "deps", "props", "bridge", "overlay", "ops",
                 "ui", "dev"]


def _purge_bytecode(pkg_dir: Path) -> int:
    """Delete __pycache__ under the add-on before reloading.

    Python decides a .pyc is still valid from the source's mtime **in whole
    seconds** plus its size. Edit a file, save, reload, edit again within the
    same second without changing the length - flipping 0.75 to 0.61, say -
    and the cached bytecode still looks valid, so the reload silently runs
    the OLD code. That failure mode is brutal to diagnose ("I changed it and
    nothing happened"), and clearing the cache costs milliseconds.
    """
    n = 0
    for d in pkg_dir.rglob("__pycache__"):
        for f in d.glob("*.pyc"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


def reload_all() -> tuple[bool, str]:
    """Unregister, reload every submodule in order, re-register."""
    pkg = __package__
    root = sys.modules.get(pkg)
    n = 0
    try:
        root.unregister()
    except Exception:                                       # noqa: BLE001
        log.get().warning("unregister raised during reload:\n%s",
                          traceback.format_exc())

    pkg_dir = Path(root.__file__).parent
    purged = _purge_bytecode(pkg_dir)
    importlib.invalidate_caches()
    log.get().debug("purged %d cached .pyc file(s)", purged)

    try:
        for rel in CORE_MODULES + ADDON_MODULES:
            name = f"{pkg}.{rel}"
            mod = sys.modules.get(name)
            if mod is not None:
                importlib.reload(mod)
                n += 1
        importlib.reload(root)
    except Exception as exc:                                # noqa: BLE001
        traceback.print_exc()
        # try to get back to a usable state rather than leaving Blender
        # with nothing registered
        try:
            sys.modules[pkg].register()
        except Exception:                                   # noqa: BLE001
            pass
        return False, f"{type(exc).__name__}: {exc}"

    try:
        sys.modules[pkg].register()
    except Exception as exc:                                # noqa: BLE001
        traceback.print_exc()
        return False, f"re-register failed: {exc}"
    return True, f"reloaded {n} module(s)"


def _deferred_reload():
    """Run the reload off the operator's stack. Timer callbacks return None to
    unregister themselves, so this fires exactly once."""
    ok, msg = reload_all()
    lg = log.get()
    (lg.info if ok else lg.error)("reload add-on: %s", msg)
    return None


class CBU_OT_reload(bpy.types.Operator):
    bl_idname = "cbu.reload"
    bl_label = "Reload Add-on"
    bl_description = ("Re-import every add-on and core module from disk, "
                      "without restarting Blender. Your scene is untouched")

    def execute(self, context):
        # reload_all() unregisters every class in the add-on - including this
        # operator's own. Anything touching `self` after that dereferences a
        # freed RNA type and segfaults Blender; self.report() did exactly
        # that, and because report() also tags the UI for redraw it ran the
        # freshly re-registered draw handlers at the same moment. Defer the
        # whole reload so the operator has finished before any of it happens.
        bpy.app.timers.register(_deferred_reload, first_interval=0.0)
        return {"FINISHED"}


class CBU_OT_debugpy(bpy.types.Operator):
    bl_idname = "cbu.debugpy"
    bl_label = "Start Debug Server"
    bl_description = ("Listen for a VS Code / debugpy attach on the given "
                      "port, so you can set breakpoints in the add-on")

    _started = False

    def execute(self, context):
        port = context.scene.cbu.debug_port
        if CBU_OT_debugpy._started:
            self.report({"INFO"}, f"debug server already listening on {port}")
            return {"FINISHED"}
        try:
            import debugpy
        except ImportError:
            self.report({"ERROR"},
                        "debugpy not installed - use Install Dependencies, "
                        "or run: <blender python> -m pip install debugpy")
            return {"CANCELLED"}
        try:
            debugpy.listen(("127.0.0.1", port))
        except Exception as exc:                            # noqa: BLE001
            self.report({"ERROR"}, f"debugpy.listen failed: {exc}")
            return {"CANCELLED"}
        CBU_OT_debugpy._started = True
        log.get().warning("debugpy listening on 127.0.0.1:%d - attach now", port)
        self.report({"INFO"}, f"listening on 127.0.0.1:{port} - attach from VS Code")
        return {"FINISHED"}


def state_dict(context) -> dict:
    """Everything worth pasting into a bug report."""
    from . import ops, build
    s = context.scene.cbu
    obj = context.active_object
    d = {
        "build": {"git": build.GIT, "code": build.CODE,
                  "code_on_disk": build.disk_code(0.0),
                  "loaded_at": build.LOADED_AT},
        "blender": bpy.app.version_string,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "settings": {k: _jsonable(getattr(s, k)) for k in s.__annotations__},
        "deps": {},
    }
    for m in ("shapely", "networkx", "ezdxf", "numpy", "debugpy"):
        try:
            d["deps"][m] = getattr(importlib.import_module(m), "__version__", "?")
        except ImportError:
            d["deps"][m] = None
    if obj is not None and obj.type == "MESH":
        me = obj.data
        d["object"] = {
            "name": obj.name,
            "verts": len(me.vertices), "edges": len(me.edges),
            "faces": len(me.polygons),
            "ngons": sum(1 for p in me.polygons if len(p.vertices) > 4),
            "tris": sum(1 for p in me.polygons if len(p.vertices) == 3),
            "uv_layers": [l.name for l in me.uv_layers],
            "attributes": [a.name for a in me.attributes if a.name.startswith("cbu")],
            "scale": list(obj.scale),
        }
        cached = ops._LAST.get(obj.name)
        if cached:
            pm, islands, parts, sheets, rep = cached
            d["last_run"] = {
                "report": {k: _jsonable(v) for k, v in rep.items()},
                "sheets": sheets,
                "core_faces": len(pm.faces),
                "islands": [{"id": i.id, "faces": len(i.faces),
                             "folds": len(i.folds), "cuts": len(i.cuts),
                             "area_mm2": round(i.area(), 1),
                             "size_mm": [round(v, 1) for v in i.size()]}
                            for i in islands],
                "parts": [{"id": p.island_id, "sheet": p.sheet,
                           "size_mm": [round(v, 1) for v in p.size],
                           "warnings": p.warnings} for p in parts],
            }
        d["timings_ms"] = {k: round(v * 1000, 1)
                           for k, v in ops.LAST_TIMINGS.items()}
    return d


def _jsonable(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return [_jsonable(x) for x in v]
    except TypeError:
        return str(v)


class CBU_OT_dump_state(bpy.types.Operator):
    bl_idname = "cbu.dump_state"
    bl_label = "Copy Debug Info"
    bl_description = ("Copy a JSON snapshot - versions, settings, mesh stats, "
                      "last run and timings - to the clipboard")

    to_file: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        try:
            data = json.dumps(state_dict(context), indent=2)
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"dump failed: {exc}")
            return {"CANCELLED"}
        context.window_manager.clipboard = data
        print(data)
        if self.to_file:
            p = Path(bpy.path.abspath(context.scene.cbu.out_dir)) / "cbu_debug.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)
            self.report({"INFO"}, f"written to {p}")
        else:
            self.report({"INFO"}, "debug info copied to clipboard (and printed)")
        return {"FINISHED"}


class CBU_OT_selftest(bpy.types.Operator):
    bl_idname = "cbu.selftest"
    bl_label = "Run Self-Test"
    bl_description = ("Unfold a known cube in memory and verify the "
                      "invariants. Fast sanity check after a reload")

    def execute(self, context):
        import numpy as np
        from .vendor.cardboard_unfolder.mesh import PolyMesh
        from .vendor.cardboard_unfolder import unfold as U_pkg
        import importlib
        U = importlib.import_module(
            __package__ + ".vendor.cardboard_unfolder.unfold")
        L = importlib.import_module(
            __package__ + ".vendor.cardboard_unfolder.layout")
        from .props import material_from

        fails = []
        try:
            v = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],
                          [0, 0, 100], [100, 0, 100], [100, 100, 100],
                          [0, 100, 100]], dtype=float)
            f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
            pm = PolyMesh.from_rings(v, f, merge=False)
            if len(pm.faces) != 6:
                fails.append(f"expected 6 faces, got {len(pm.faces)}")
            islands, rep = U.unfold(pm, trials=40)
            if rep["isometry_error"] > 1e-12:
                fails.append(f"isometry {rep['isometry_error']:.1e}")
            flat = sum(i.area() for i in islands)
            if abs(flat - 6 * 100 * 100) > 1e-6:
                fails.append(f"area {flat:.3f} != 60000")
            if rep["unresolved"]:
                fails.append(f"{rep['unresolved']} unresolved overlap(s)")
            parts = L.build_parts(islands, material_from(context.scene.cbu))
            if not parts or parts[0].cut.is_empty:
                fails.append("no part geometry produced")
        except Exception as exc:                            # noqa: BLE001
            traceback.print_exc()
            fails.append(f"{type(exc).__name__}: {exc}")

        if fails:
            for f_ in fails:
                self.report({"ERROR"}, f_)
            log.get().error("self-test FAILED: %s", "; ".join(fails))
            return {"CANCELLED"}
        self.report({"INFO"}, "self-test passed (unfold, isometry, area, parts)")
        log.get().warning("self-test passed")
        return {"FINISHED"}


classes = (CBU_OT_reload, CBU_OT_debugpy, CBU_OT_dump_state, CBU_OT_selftest)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
