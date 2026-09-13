"""Viewport overlay: islands tinted by index, forced marks drawn on top.

Batches are rebuilt only when an unfold happens, never per frame. Note that
gpu.state.line_width_set is a no-op on the Metal and Vulkan backends, so
anything that must be thick uses the POLYLINE shaders instead of plain LINES.
"""
from __future__ import annotations

import colorsys
import hashlib
import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

NET_UV = "CBU_Net"     # kept in step with bridge.UV_LAYER

_handle = None
_uv_handle = None
_batches = {}          # object name -> dict of batches
_net_bounds = {}       # object name -> (x0, y0, x1, y1) of the net, in mm
_net_cuts = {}         # object name -> per-island outlines, island space
_live_rings = {}       # object name -> resolved rings, sampled off the draw
_live_labels = {}      # object name -> resolved engraving, same treatment
_fingerprints = {}     # object name -> geometry fingerprint at unfold time
_bed_cache = {}        # draw key -> batch, so the grid is not rebuilt per frame
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


# The island tint is drawn on the *same* planes as the mesh faces, so without
# a nudge the two are exactly coplanar and the depth test decides per pixel -
# classic z-fighting, which reads as the object's own polygons punching through
# the colour. GPU state has no polygon-offset knob exposed to Python, so the
# triangles are lifted off their own plane at build time instead. A fraction of
# the object's local bounding box, so it works whatever the model is scaled to.
LIFT = 0.0015


def face_normal(co):
    """Unit normal of a ring, by Newell's method.

    Newell rather than a cross product of the first three corners: an n-gon
    whose first corners are nearly collinear gives a cross product that is
    numerically junk, and Newell averages over the whole ring instead.
    Degenerate rings come back as a zero vector, which lifts nothing.
    """
    a = np.asarray(co, dtype=float)
    b = np.roll(a, -1, axis=0)
    n = np.array([
        np.sum((a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2])),
        np.sum((a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0])),
        np.sum((a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])),
    ])
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else np.zeros(3)


def lift(co, amount: float):
    """Push a face ring off its own plane, so the tint clears the mesh."""
    return np.asarray(co, dtype=float) + face_normal(co) * amount


def lift_amount(obj) -> float:
    """How far to lift, in the object's own space.

    bound_box is local and excludes the object's scale, which is right: the
    overlay is drawn under matrix_world, so an offset derived here scales with
    the object exactly as the geometry does.
    """
    bb = np.asarray(obj.bound_box, dtype=float)
    return LIFT * float(np.linalg.norm(bb.max(axis=0) - bb.min(axis=0)))


def _fan(ring):
    """Triangulate a convex-ish ring for display only."""
    return [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]


def geometry_fingerprint(mesh) -> tuple:
    """Element counts plus vertex positions.

    Deliberately ignores attributes. Marking a seam writes an edge attribute
    and nothing else, yet the depsgraph still reports it as a geometry update
    - and clearing on that would wipe the island colours you are marking
    against, which is the one moment you want them.
    """
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    if n:
        mesh.vertices.foreach_get("co", co)
    return (n, len(mesh.edges), len(mesh.polygons),
            hashlib.blake2b(co.tobytes(), digest_size=16).digest())


def remember_geometry(obj) -> None:
    _fingerprints[obj.name] = geometry_fingerprint(obj.data)


def geometry_changed(obj) -> bool:
    """True only if we took a fingerprint and the mesh no longer matches it."""
    old = _fingerprints.get(obj.name)
    if old is None:
        return False
    return old != geometry_fingerprint(obj.data)


def tracked_names() -> set:
    return set(_batches) | set(_fingerprints)


def rebuild(obj, pm, islands, alpha: float = 0.55) -> None:
    """Cache draw data for one object. Coordinates go back to object space so
    the overlay follows the object if it is moved."""
    remember_geometry(obj)
    tri_s, line_s = _shaders()
    if tri_s is None:
        _batches.pop(obj.name, None)
        return

    verts, ids = [], []
    cut_pts, cut_cols = [], []

    island_of = {}
    for isl in islands:
        for f in isl.faces:
            island_of[f] = isl.id

    eps = lift_amount(obj)
    for f, ring in enumerate(pm.faces):
        i = island_of.get(f, 0)
        co = lift([obj.data.vertices[v].co[:] for v in ring], eps)
        for tri in _fan(co):
            for p in tri:
                verts.append(p[:])
                # The island index, not the colour. Opacity is a view setting
                # and has to be changeable without re-running the unfold, so
                # the colour is resolved in _tris - see set_alpha.
                ids.append(i)

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
        "pos": verts,
        "ids": ids,
        "alpha": alpha,
        "tris": _tris(verts, ids, alpha),
        "lines": batch_for_shader(line_s, "LINES",
                                  {"pos": cut_pts, "color": cut_cols})
        if cut_pts else None,
    }
    _tag_redraw()


def island_colors(ids, alpha: float):
    """Resolve one colour per vertex from cached island indices.

    Pure, and deliberately separate from the batch: there is no GPU in
    background mode, so `_batches` is always empty under
    `blender --background` and the colour rule can only be tested here.
    """
    lut = {i: island_color(i, alpha) for i in set(ids)}
    return [lut[i] for i in ids]


def _tris(verts, ids, alpha: float):
    """Colour the cached triangles at a given opacity."""
    tri_s, _ = _shaders()
    if tri_s is None or not verts:
        return None
    return batch_for_shader(tri_s, "TRIS",
                            {"pos": verts, "color": island_colors(ids, alpha)})


def set_alpha(alpha: float) -> None:
    """Re-colour every cached annotation, without re-unfolding.

    The alpha used to be baked into the vertex colours when the batch was
    built and never read again, so the Opacity slider did nothing at all until
    the next unfold - and the Unfold operator did not even pass the setting,
    so it was permanently 0.55. The island index is cached instead of the
    colour, which makes this a re-colour of geometry already in hand: no
    bmesh, no mesh read, nothing that could be unsafe to touch from a property
    update.
    """
    for b in _batches.values():
        if b.get("alpha") == alpha:
            continue
        b["alpha"] = alpha
        b["tris"] = _tris(b.get("pos") or [], b.get("ids") or [], alpha)
    _tag_redraw()


def set_net_cuts(name: str, groups) -> None:
    """Remember the real cut outlines so the 2D view can show what is cut.

    The UV layer only ever holds the islands. Glue tabs are generated in the
    core and exist solely in the exported parts, so a net drawn from UVs alone
    is about 20% smaller than the thing the laser follows - and anyone
    arranging islands by eye is optimising the wrong shape.

    Stored per island, in ISLAND space, with two reference loops each. The
    outline then follows the island when it is dragged in the UV editor
    instead of staying where the last unfold left it: a stale outline is worse
    than none, because it invites arranging against a shape that has moved.
    """
    if groups:
        _net_cuts[name] = groups
    else:
        _net_cuts.pop(name, None)
    _live_rings.pop(name, None)
    _live_labels.pop(name, None)
    _bed_cache.clear()
    _tag_redraw()


def _live_uv(obj, loop_idx):
    """Current UV of specific loops. Object Mode only - None in Edit Mode.

    There is no safe way to read live Edit Mode UVs from an automatic hook.
    The values live in the BMesh - the mesh datablock is empty there, and so is
    the depsgraph's evaluated copy, both checked - and reading it crashed
    Blender from every context tried:

      * a GPU draw handler: frees customdata under the draw, BLI_mempool_free
        under BM_data_layer_add under cb_region_draw
      * a timer: fires in the middle of a modal transform
      * depsgraph_update_post: access violation inside tbbmalloc

    So the outline is not sampled while in Edit Mode at all. It keeps the
    positions from the last unfold, nest, or Object Mode refresh and catches up
    the moment Edit Mode is left. Three crashes is enough evidence that this
    particular overlay cannot be made live safely, and a cosmetic overlay is
    never worth a crash.
    """
    if obj.mode == "EDIT":
        return None
    me = obj.data
    uv = me.uv_layers.get(NET_UV)
    if uv is None or not len(uv.data):
        return None
    return np.asarray([tuple(uv.data[int(i)].uv) for i in loop_idx],
                      dtype=np.float64)


def refresh_live_cuts(*_args):
    """Resolve the cut outlines against the current layout, into the cache.

    Called when parts are published and from depsgraph_update_post. It is a
    no-op in Edit Mode by design - _live_uv refuses to reach for the BMesh
    there, because doing so crashed Blender from every context available.
    """
    try:
        ctx = bpy.context
        scene = getattr(ctx, "scene", None)
        s = getattr(scene, "cbu", None) if scene else None
        if s is None or not _net_cuts:
            return
        from . import bridge
        sc = bridge.uv_scale(s.bed_x, s.bed_y)
        for name in list(_net_cuts):
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != "MESH":
                _live_rings.pop(name, None)
                _live_labels.pop(name, None)
                continue
            rings = live_cut_rings(obj, sc)
            if rings is None:
                continue                  # Edit Mode: keep what we last had
            before = _live_rings.get(name)
            _live_rings[name] = rings
            _live_labels[name] = live_label_strokes(obj, sc) or []
            if before is None or len(before) != len(rings) or any(
                    not np.array_equal(a, b) for a, b in zip(before, rings)):
                _tag_redraw()
    except Exception:                                   # noqa: BLE001
        pass                    # a cosmetic overlay must never break the session


def _resolve(obj, scale: float, key: str):
    """Island-space polylines placed where the islands are *now*, in UV units.

    Two reference points per island are enough to recover the rigid motion,
    and cheap enough to redo on every redraw - a full fit over every loop
    would not be. The cut outline and the engraving travel together and move
    together, so they share this.
    """
    groups = _net_cuts.get(obj.name)
    if not groups:
        return None
    loop_idx = []
    for g in groups:
        loop_idx.extend(g["refs"])
    cur = _live_uv(obj, loop_idx)
    if cur is None:
        return None

    out = []
    for k, g in enumerate(groups):
        p0, p1 = g["ref_pts"]                      # island space
        q0, q1 = cur[2 * k] / scale, cur[2 * k + 1] / scale
        u, v = p1 - p0, q1 - q0
        lu = float(np.hypot(u[0], u[1]))
        lv = float(np.hypot(v[0], v[1]))
        if lu < 1e-9 or lv < 1e-9:
            continue
        c = float(u @ v) / (lu * lv)
        s = float(u[0] * v[1] - u[1] * v[0]) / (lu * lv)
        R = np.array([[c, -s], [s, c]])
        for poly in g.get(key) or ():
            out.append((poly - p0) @ R.T + q0)
    return out


def live_cut_rings(obj, scale: float):
    """Closed cut outlines where the islands are now."""
    return _resolve(obj, scale, "rings")


def live_label_strokes(obj, scale: float):
    """Engraved strokes where the islands are now. Open polylines, not rings -
    closing them would draw a chord across every letter."""
    return _resolve(obj, scale, "labels")


def set_net_bounds(name: str, bounds) -> None:
    """Remember the net's extent so the bed reference knows how far to tile."""
    if bounds is None:
        _net_bounds.pop(name, None)
    else:
        _net_bounds[name] = tuple(bounds)
    _bed_cache.clear()
    _tag_redraw()


def clear(name: str | None = None) -> None:
    if name is None:
        _batches.clear()
        _net_bounds.clear()
        _net_cuts.clear()
        _live_rings.clear()
        _live_labels.clear()
        _fingerprints.clear()
    else:
        _batches.pop(name, None)
        _net_bounds.pop(name, None)
        _net_cuts.pop(name, None)
        _live_rings.pop(name, None)
        _live_labels.pop(name, None)
        _fingerprints.pop(name, None)
    _bed_cache.clear()
    _tag_redraw()


def _tag_redraw():
    for w in getattr(bpy.context, "window_manager", None).windows if \
            getattr(bpy.context, "window_manager", None) else []:
        for a in w.screen.areas:
            if a.type in {"VIEW_3D", "IMAGE_EDITOR"}:
                a.tag_redraw()


def _draw():
    ctx = bpy.context
    s = getattr(ctx.scene, "cbu", None)
    if s is None or not s.show_overlay or not _batches:
        return
    tri_s, line_s = _shaders()
    if tri_s is None:
        return
    # The fill is what makes the mesh hard to read - it hides the selection
    # highlight and the shading. The forced marks are the opposite: they are
    # the reason to have the overlay up while marking seams. So they are
    # switched separately, and turning the colour off never costs you the marks.
    fill = s.overlay_fill
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")
    try:
        for name, b in _batches.items():
            obj = bpy.data.objects.get(name)
            if obj is None or not obj.visible_get():
                continue
            with gpu.matrix.push_pop():
                gpu.matrix.multiply_matrix(obj.matrix_world)
                if b["tris"] is not None and fill:
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


# ------------------------------------------------- laser bed, in the UV editor

SHEET_RGBA = (0.95, 0.62, 0.15, 0.85)      # sheet outline
CUT_RGBA = (0.20, 0.85, 0.45, 0.95)        # what the laser actually follows
LABEL_RGBA = (0.55, 0.78, 1.00, 0.95)      # engraved seam / fold / part marks
AXIS_RGBA = (0.95, 0.62, 0.15, 0.70)       # the mm ruler down the bed edges
MAX_TICKS = 400                            # a runaway must not stall drawing
USABLE_RGBA = (0.95, 0.62, 0.15, 0.32)     # usable area inside the margins
MAX_TILES = 400                            # a runaway net must not stall drawing


def bed_grid(bounds, bed_w: float, bed_h: float):
    """The rectangle the sheet grid must cover, anchored at the sheet origin.

    Never the net's own bounding box. The net starts at the margin, so tiling
    from its min corner shifts every sheet - and its inner margin line - by
    exactly one margin: parts sitting correctly at x=8 then appear 8 mm outside
    a line drawn at 16, and a perfectly good layout reads as ignoring the
    margin. Measured on a pikachu net: the grid spanned x[8.1, 308.1] instead
    of x[0, 300].
    """
    return (0.0, 0.0, max(bounds[2], bed_w), max(bounds[3], bed_h))


def bed_segments(bounds, bed_w, bed_h, margin):
    """Line segments for a grid of sheets covering the net: (verts, colours).

    The net UV layer is written in millimetres, so a sheet is drawn at its
    true size and the islands can be read against it directly - that is the
    whole point of keeping the net in mm rather than normalising it. Kept free
    of GPU calls so it can be tested without a drawing context.
    """
    if bed_w <= 0.0 or bed_h <= 0.0:
        return [], []
    x0, y0, x1, y1 = bounds
    cols = max(1, min(MAX_TILES, int(np.ceil((x1 - x0) / bed_w))))
    rows = max(1, min(MAX_TILES, int(np.ceil((y1 - y0) / bed_h))))
    if cols * rows > MAX_TILES:
        rows = max(1, MAX_TILES // cols)

    verts, cols_rgba = [], []

    def rect(ax, ay, bx, by, rgba):
        # Three components, not two. The builtin FLAT_COLOR / POLYLINE_FLAT_COLOR
        # shaders declare `pos` as vec3 even when the draw is 2D, and handing
        # batch_for_shader 2-tuples is a vertex-buffer format mismatch that
        # takes Blender down with it rather than raising. z is just 0.
        corners = [(ax, ay, 0.0), (bx, ay, 0.0), (bx, by, 0.0), (ax, by, 0.0)]
        for i in range(4):
            verts.append(corners[i])
            verts.append(corners[(i + 1) % 4])
            cols_rgba.append(rgba)
            cols_rgba.append(rgba)

    for r in range(rows):
        for c in range(cols):
            ox, oy = x0 + c * bed_w, y0 + r * bed_h
            rect(ox, oy, ox + bed_w, oy + bed_h, SHEET_RGBA)
            if margin > 0.0 and bed_w > 2 * margin and bed_h > 2 * margin:
                rect(ox + margin, oy + margin,
                     ox + bed_w - margin, oy + bed_h - margin, USABLE_RGBA)

    return verts, cols_rgba


def cut_segments(rings, scale: float):
    """Closed outlines as line segments, in UV units. (verts, colours)."""
    verts, cols = [], []
    for ring in rings:
        n = len(ring)
        if n < 2:
            continue
        for i in range(n):
            a = ring[i] * scale
            b = ring[(i + 1) % n] * scale
            verts.append((float(a[0]), float(a[1]), 0.0))
            verts.append((float(b[0]), float(b[1]), 0.0))
            cols.append(CUT_RGBA)
            cols.append(CUT_RGBA)
    return verts, cols


def stroke_segments(strokes, scale: float):
    """Open polylines as line segments, in UV units. (verts, colours).

    Deliberately not cut_segments: that wraps the last point back to the first,
    which on a letter draws a chord straight through it.
    """
    verts, cols = [], []
    for st in strokes:
        for i in range(len(st) - 1):
            a, b = st[i] * scale, st[i + 1] * scale
            verts.append((float(a[0]), float(a[1]), 0.0))
            verts.append((float(b[0]), float(b[1]), 0.0))
            cols.append(LABEL_RGBA)
            cols.append(LABEL_RGBA)
    return verts, cols


def _label_batch(obj, scale: float):
    """Same contract as _cut_batch: read the cache, never sample live UVs."""
    strokes = _live_labels.get(obj.name)
    line_s = _shaders()[1]
    if not strokes or line_s is None:
        return None
    key = ("labels", obj.name, scale,
           float(np.round(np.concatenate(
               [st.ravel() for st in strokes]).sum(), 4)))
    hit = _bed_cache.get(key)
    if hit is None:
        verts, cols = stroke_segments(strokes, scale)
        hit = (batch_for_shader(line_s, "LINES", {"pos": verts, "color": cols})
               if verts else None)
        for k in [k for k in _bed_cache if isinstance(k, tuple)
                  and k and k[0] == "labels"]:
            del _bed_cache[k]
        _bed_cache[key] = hit
    return hit


def _cut_batch(obj, scale: float):
    """Rebuilt whenever the islands have actually moved, not every frame.

    Reads the cache the timer fills. Sampling here instead is what crashed
    Blender: bmesh.from_edit_mesh inside a draw handler frees customdata
    blocks out from under the draw.
    """
    rings = _live_rings.get(obj.name)
    line_s = _shaders()[1]
    if not rings or line_s is None:
        return None
    key = ("cuts", obj.name, scale,
           float(np.round(np.concatenate([r.ravel() for r in rings]).sum(), 4)))
    hit = _bed_cache.get(key)
    if hit is None:
        verts, cols = cut_segments(rings, scale)
        hit = (batch_for_shader(line_s, "LINES", {"pos": verts, "color": cols})
               if verts else None)
        for k in [k for k in _bed_cache if isinstance(k, tuple)
                  and k and k[0] == "cuts"]:
            del _bed_cache[k]
        _bed_cache[key] = hit
    return hit


def nice_step(span_mm: float, target: int = 6) -> float:
    """A 1 / 2 / 5 x 10^k step giving roughly `target` labelled ticks.

    Round numbers, or the ruler is no easier to read than the thing it exists
    to explain: 50 mm on a 300 mm bed, not 42.857.
    """
    if span_mm <= 0:
        return 1.0
    raw = span_mm / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def axis_segments(grid, scale: float, label_mm: float = 6.0):
    """A millimetre ruler along the bottom and left edges. (verts, colours).

    The UV editor measures in UV units, and the factor is the bed rounded UP to
    a power of two - 512 mm for a 300 mm bed, 1024 for a 600 mm one. A length
    read off the editor and multiplied by the wrong remembered constant is out
    by exactly two, and that reads as a bug in whatever was measured rather
    than in the measuring. A ruler drawn in the view needs no remembering.

    Ticks sit OUTSIDE the sheet, below and left of the origin, so nothing is
    ever drawn over the parts.
    """
    from .vendor.cardboard_unfolder import strokefont

    x0, y0, x1, y1 = grid
    w_mm, h_mm = (x1 - x0) / scale, (y1 - y0) / scale
    if w_mm <= 0 or h_mm <= 0:
        return [], []
    step = nice_step(max(w_mm, h_mm))
    minor = step / 5.0
    major_len, minor_len = 5.0 * scale, 2.5 * scale
    gap = 2.0 * scale
    verts, cols = [], []

    def seg(ax, ay, bx, by):
        verts.append((float(ax), float(ay), 0.0))
        verts.append((float(bx), float(by), 0.0))
        cols.append(AXIS_RGBA)
        cols.append(AXIS_RGBA)

    def text(s, at):
        for st in strokefont.text_strokes(s, label_mm * scale, at,
                                          anchor="center"):
            for i in range(len(st) - 1):
                seg(st[i][0], st[i][1], st[i + 1][0], st[i + 1][1])

    for axis in (0, 1):
        span = w_mm if axis == 0 else h_mm
        n = int(round(span / minor)) + 1
        if n > MAX_TICKS:
            continue
        for i in range(n):
            mm = i * minor
            big = abs(mm / step - round(mm / step)) < 1e-9
            ln = major_len if big else minor_len
            if axis == 0:
                u = x0 + mm * scale
                seg(u, y0, u, y0 - ln)
                if big:
                    text(str(int(round(mm))),
                         (u, y0 - ln - gap - label_mm * scale / 2.0))
            else:
                u = y0 + mm * scale
                seg(x0, u, x0 - ln, u)
                if big:
                    text(str(int(round(mm))),
                         (x0 - ln - gap - label_mm * scale * 1.6, u))

    # Say the unit once, at the corner, so nobody has to infer it.
    text("MM", (x0 - major_len - gap - label_mm * scale * 1.6,
                y0 - major_len - gap - label_mm * scale / 2.0))
    return verts, cols


def _bed_batch(bounds, bed_w, bed_h, margin, scale=None):
    line_s = _shaders()[1]
    if line_s is None:
        return None
    verts, cols_rgba = bed_segments(bounds, bed_w, bed_h, margin)
    if scale:
        av, ac = axis_segments(bounds, scale)
        verts = verts + av
        cols_rgba = cols_rgba + ac
    if not verts:
        return None
    return batch_for_shader(line_s, "LINES",
                            {"pos": verts, "color": cols_rgba})


def _draw_bed():
    ctx = bpy.context
    s = getattr(ctx.scene, "cbu", None)
    # Each layer on its own switch. Returning early on show_bed meant turning
    # the bed off also hid the cut outline and the engraving, which is not what
    # the checkbox says.
    if s is None or not (s.show_bed or s.show_cut or s.show_net_labels):
        return
    space = getattr(ctx, "space_data", None)
    if space is None or space.type != "IMAGE_EDITOR" or space.mode != "UV":
        return
    obj = ctx.active_object
    if obj is None or obj.type != "MESH":
        return
    # only when the net itself is on screen, never over somebody's texture UVs
    uvl = obj.data.uv_layers.active
    if uvl is None or uvl.name != NET_UV:
        return
    bounds = _net_bounds.get(obj.name)
    if bounds is None:
        return

    # bed_segments works in whatever units it is handed; the net is written
    # in scaled UV units, so the sheet has to be scaled to match.
    from . import bridge
    sc = bridge.uv_scale(s.bed_x, s.bed_y)
    batch = None
    if s.show_bed:
        grid = bed_grid(bounds, s.bed_x * sc, s.bed_y * sc)
        key = (obj.name, grid, s.bed_x, s.bed_y, s.margin, sc,
               s.show_bed_ruler)
        if key not in _bed_cache:
            # Keep the cut and label batches: they are keyed by their own
            # tag and rebuilding them on every bed change is wasted work.
            for k in [k for k in _bed_cache
                      if not (isinstance(k, tuple) and k
                              and k[0] in ("cuts", "labels"))]:
                del _bed_cache[k]
            _bed_cache[key] = _bed_batch(grid, s.bed_x * sc, s.bed_y * sc,
                                         s.margin * sc,
                                         sc if s.show_bed_ruler else None)
        batch = _bed_cache[key]

    line_s = _shaders()[1]
    if line_s is None:
        return
    gpu.state.blend_set("ALPHA")
    try:
        line_s.bind()
        try:
            line_s.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
            line_s.uniform_float("lineWidth", 1.5)
        except Exception:                               # noqa: BLE001
            pass                                        # plain FLAT_COLOR fallback
        if batch is not None:
            batch.draw(line_s)

        if s.show_cut:
            cuts = _cut_batch(obj, sc)
            if cuts is not None:
                cuts.draw(line_s)
        if s.show_net_labels:
            labels = _label_batch(obj, sc)
            if labels is not None:
                labels.draw(line_s)
    finally:
        gpu.state.blend_set("NONE")


def register():
    global _handle, _uv_handle
    if not available():
        return
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_VIEW")
    if _uv_handle is None:
        # POST_VIEW in the image editor draws in UV coordinates, which for the
        # net layer are millimetres - so the bed rectangle needs no transform.
        _uv_handle = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_bed, (), "WINDOW", "POST_VIEW")


def unregister():
    global _handle, _uv_handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
    if _uv_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_uv_handle, "WINDOW")
        _uv_handle = None
    _batches.clear()
    _net_bounds.clear()
    _net_cuts.clear()
    _live_rings.clear()
    _live_labels.clear()
    _fingerprints.clear()
    _bed_cache.clear()
