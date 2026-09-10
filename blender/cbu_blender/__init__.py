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

from . import log, deps, props, bridge, overlay, ops, ui, dev

_MODULES = (log, deps, props, bridge, overlay, ops, ui, dev)


@persistent
def _on_load(_dummy):
    """Drop cached draw batches and results when a new file is opened.

    Batches hold references to objects from the previous file; drawing them
    afterwards is a classic addon crash."""
    overlay.clear()
    ops._LAST.clear()


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


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    overlay.unregister()
    ui.unregister()
    dev.unregister()
    ops.unregister()
    props.unregister()
    deps.unregister()
