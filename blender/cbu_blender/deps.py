"""Dependency handling.

The core needs shapely and networkx (and ezdxf for DXF). Blender ships numpy
but none of those, so the addon checks and offers a one-click install into
Blender's own Python. The alternative - bundling platform wheels via the 4.2+
extension manifest - is better for distribution but needs a wheel set per
platform, which is not something a single zip can carry cleanly.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

REQUIRED = {"shapely": "shapely>=2.0", "networkx": "networkx>=3.0"}
OPTIONAL = {"ezdxf": "ezdxf>=1.1", "debugpy": "debugpy>=1.8"}


def missing(include_optional: bool = False) -> list[str]:
    want = dict(REQUIRED)
    if include_optional:
        want.update(OPTIONAL)
    out = []
    for mod in want:
        try:
            importlib.import_module(mod)
        except ImportError:
            out.append(mod)
    return out


def _install(specs) -> tuple[bool, str]:
    exe = sys.executable
    try:
        subprocess.run([exe, "-m", "ensurepip", "--upgrade"],
                       capture_output=True, timeout=180)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        r = subprocess.run([exe, "-m", "pip", "install", "--upgrade", *specs],
                           capture_output=True, text=True, timeout=900)
    except Exception as exc:                            # noqa: BLE001
        return False, str(exc)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-500:]
    for mod in list(REQUIRED) + list(OPTIONAL):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    importlib.invalidate_caches()
    return True, "installed"


try:
    import bpy

    class CBU_OT_install_deps(bpy.types.Operator):
        bl_idname = "cbu.install_deps"
        bl_label = "Install Dependencies"
        bl_description = ("pip install shapely, networkx and ezdxf into "
                          "Blender's Python. Needs network access")

        def execute(self, context):
            specs = list(REQUIRED.values()) + list(OPTIONAL.values())
            ok, msg = _install(specs)
            if not ok:
                self.report({"ERROR"}, f"install failed: {msg}")
                return {"CANCELLED"}
            left = missing()
            if left:
                self.report({"WARNING"},
                            f"installed, but {', '.join(left)} still not "
                            f"importable - restart Blender")
            else:
                self.report({"INFO"}, "dependencies ready")
            return {"FINISHED"}

    classes = (CBU_OT_install_deps,)

    def register():
        for c in classes:
            bpy.utils.register_class(c)

    def unregister():
        for c in reversed(classes):
            bpy.utils.unregister_class(c)

except ImportError:                                     # importable outside Blender
    def register():
        pass

    def unregister():
        pass
