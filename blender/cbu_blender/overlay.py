"""Viewport overlay: islands tinted by index, forced marks drawn on top.

Batches are rebuilt only when an unfold happens, never per frame. Note that
gpu.state.line_width_set is a no-op on the Metal and Vulkan backends, so
anything that must be thick uses the POLYLINE shaders instead of plain LINES.
"""
from __future__ import annotations

import colorsys

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

_handle = None
_batches = {}          # object name -> dict of batches
_TRI = None
_LINE = None


def available() -> bool:
    """Drawing is impossible in background mode, and a GPU backend can fail
    at runtime. The overlay is cosmetic, so never let it break an unfold."""
    return not bpy.app.background


def _shaders():
    """Returns (tri, line) or (None, None) if the GPU module can't serve us."""
    global _TRI, _LINE
    if not available():
        return None, None
    try:
        if _TRI is None:
            _TRI = gpu.shader.from_builtin("FLAT_COLOR")
        if _LINE is None:
            try:
                _LINE = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
            except Exception:                               # older builds
                _LINE = gpu.shader.from_builtin("FLAT_COLOR")
    except Exception:                                       # noqa: BLE001
        return None, None
    return _TRI, _LINE


def island_color(i: int, alpha: float = 0.55):
    h = (i * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.52, 0.98)
    return (r, g, b, alpha)


def _fan(ring):
    """Triangulate a convex-ish ring for display only."""
    return [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]


def rebuild(obj, pm, islands, alpha: float = 0.55) -> None:
    """Cache draw data for one object. Coordinates go back to object space so
    the overlay follows the object if it is moved."""
    tri_s, line_s = _shaders()
    if tri_s is None:
        _batches.pop(obj.name, None)
        return

    verts, cols = [], []
    cut_pts, cut_cols = [], []

    island_of = {}
    for isl in islands:
        for f in isl.faces:
            island_of[f] = isl.id

    for f, ring in enumerate(pm.faces):
        col = island_color(island_of.get(f, 0), alpha)
        co = [obj.data.vertices[v].co for v in ring]
        for tri in _fan(co):
            for p in tri:
                verts.append(p[:])
                cols.append(col)

    # forced marks, straight off the mesh attribute
    attr = obj.data.attributes.get("cbu_constraint")
    if attr is not None:
        vals = np.zeros(len(obj.data.edges), dtype=np.int32)
        attr.data.foreach_get("value", vals)
        for i, v in enumerate(vals):
            if v == 0:
                continue
            a, b = obj.data.edges[i].vertices
            c = (0.85, 0.18, 0.10, 1.0) if v == 1 else (0.12, 0.45, 0.95, 1.0)
            cut_pts += [obj.data.vertices[a].co[:], obj.data.vertices[b].co[:]]
            cut_cols += [c, c]

    _batches[obj.name] = {
        "tris": batch_for_shader(tri_s, "TRIS", {"pos": verts, "color": cols})
        if verts else None,
        "lines": batch_for_shader(line_s, "LINES",
                                  {"pos": cut_pts, "color": cut_cols})
        if cut_pts else None,
    }
    _tag_redraw()


def clear(name: str | None = None) -> None:
    if name is None:
        _batches.clear()
    else:
        _batches.pop(name, None)
    _tag_redraw()


def _tag_redraw():
    for w in getattr(bpy.context, "window_manager", None).windows if \
            getattr(bpy.context, "window_manager", None) else []:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                a.tag_redraw()


def _draw():
    ctx = bpy.context
    s = getattr(ctx.scene, "cbu", None)
    if s is None or not s.show_overlay or not _batches:
        return
    tri_s, line_s = _shaders()
    if tri_s is None:
        return
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        for name, b in _batches.items():
            obj = bpy.data.objects.get(name)
            if obj is None or not obj.visible_get():
                continue
            with gpu.matrix.push_pop():
                gpu.matrix.multiply_matrix(obj.matrix_world)
                if b["tris"] is not None:
                    tri_s.bind()
                    b["tris"].draw(tri_s)
                if b["lines"] is not None:
                    line_s.bind()
                    try:
                        line_s.uniform_float(
                            "viewportSize", gpu.state.viewport_get()[2:])
                        line_s.uniform_float("lineWidth", 3.0)
                    except Exception:
                        pass
                    b["lines"].draw(line_s)
    finally:
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("NONE")


def register():
    global _handle
    if _handle is None and available():
        _handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
    _batches.clear()
