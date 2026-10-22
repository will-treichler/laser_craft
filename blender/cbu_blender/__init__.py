bl_info = {
    "name": "Cardboard Unfolder",
    "author": "M1/M4 prototype",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Cardboard",
    "description": "Unfold low-poly meshes into laser-cuttable corrugated cardboard",
    "category": "Import-Export",
}

import importlib
import sys

import bpy
from bpy.app.handlers import persistent

from . import build, log, deps, props, bridge, overlay, ops, ui, dev

_MODULES = (log, deps, props, bridge, overlay, ops, ui, dev)


@persistent
def _on_load(_dummy):
    """Drop cached draw batches and results when a new file is opened.

    Batches hold references to objects from the previous file; drawing them
    afterwards is a classic addon crash."""
    overlay.clear()
    ops._LAST.clear()
    ops._GROUP.clear()


@persistent
def _on_depsgraph(scene, depsgraph):
    """Drop the island colours once the mesh they describe actually changes.

    The batches hold vertex positions copied at unfold time, so after an edit
    they paint islands over geometry that has moved. The cached unfold goes
    with them - `ops._LAST` is what Export writes, and exporting it after an
    edit would cut a net that no longer matches the object. Same pairing as
    `_on_load` below.

    The depsgraph reports an edge-attribute write as a geometry update too, so
    marking a seam would trip a naive check and wipe the very colours you are
    marking against. `overlay.geometry_changed` compares counts and vertex
    positions instead, which marking does not touch.
    """
    # Catch the cut outline up with any UV edit. This is the safe moment to
    # touch bmesh: the transform has finished. A timer doing the same thing
    # fired mid-drag and crashed Blender.
    overlay.refresh_live_cuts()
    overlay.refresh_marks()

    tracked = overlay.tracked_names() | set(ops._LAST)
    if not tracked:
        return
    if not any(u.is_updated_geometry for u in depsgraph.updates):
        return
    for name in tracked:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            overlay.clear(name)
            ops._LAST.pop(name, None)
        elif overlay.geometry_changed(obj):
            overlay.clear(name)
            ops._LAST.pop(name, None)


def _reload():
    for m in _MODULES:
        importlib.reload(m)


def register():
    deps.register()
    props.register()
    ops.register()
    dev.register()
    ui.register()
    overlay.register()
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)
    if _on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)


def unregister():
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    overlay.unregister()
    ui.unregister()
    dev.unregister()
    ops.unregister()
    props.unregister()
    deps.unregister()
