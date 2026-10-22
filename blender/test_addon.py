"""Headless Blender test for the addon. Run:  python3 blender/test_addon.py

Uses the `bpy` PyPI module, so this exercises the real Blender API - operator
registration, bmesh round trips, attributes, UV layers and pack_islands - not
a mock.
"""
import math
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def fresh(name="Cube", primitive="cube", **kw):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    getattr(bpy.ops.mesh, f"primitive_{primitive}_add")(**kw)
    obj = bpy.context.active_object
    obj.name = name
    return obj


print("blender", bpy.app.version_string)

# ---------------------------------------------------------------- register
import cbu_blender
from cbu_blender import bridge, dev, ops, overlay, props, ui


def pytest_approx(v, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - v) <= tol
    return _A()

cbu_blender.register()
check(hasattr(bpy.types.Scene, "cbu"), "settings registered on Scene")
for idname in ("cbu.unfold", "cbu.mark", "cbu.export", "cbu.pack_uv",
               "cbu.clear_marks", "cbu.select_marked",
               "cbu.lock", "cbu.clear_locks", "cbu.select_locked",
               "cbu.optimise"):
    mod, fn = idname.split(".")
    check(hasattr(getattr(bpy.ops, mod), fn), f"operator {idname} registered")

# ------------------------------------------------------------------ cube
obj = fresh("Cube", "cube", size=0.11)          # 110 mm at 1000 mm/unit
s = bpy.context.scene.cbu
s.target_height = 0.0
s.mm_per_unit = 1000.0
s.trials = 60

pm = bridge.polymesh_from_object(obj, merge_coplanar=True, mm_per_unit=1000.0)
check(len(pm.faces) == 6, f"cube merges to 6 faces (got {len(pm.faces)})")
check(abs(pm.extents.max() - 110.0) < 1e-6,
      f"cube is 110 mm across (got {pm.extents.max():.3f})")

res = bpy.ops.cbu.unfold()
check(res == {"FINISHED"}, f"unfold on a cube returns FINISHED (got {res})")
check(s.last_islands == 1, f"cube unfolds to 1 island (got {s.last_islands})")
check(s.last_sheets == 1, f"cube fits 1 sheet (got {s.last_sheets})")

# attributes written
check(bridge.FACE_ATTR in obj.data.attributes, "island attribute created")
vals = np.zeros(len(obj.data.polygons), dtype=np.int32)
obj.data.attributes[bridge.FACE_ATTR].data.foreach_get("value", vals)
check(set(vals.tolist()) == {0}, f"all faces tagged island 0 (got {set(vals.tolist())})")

# UV net written, and isometric
check(bridge.UV_LAYER in obj.data.uv_layers, "net UV layer created")
uv = obj.data.uv_layers[bridge.UV_LAYER]
arr = np.zeros(len(obj.data.loops) * 2)
uv.data.foreach_get("uv", arr)
arr = arr.reshape(-1, 2)
# The net is not normalised per model - it is millimetres scaled by a single
# power of two derived from the bed, so the same object at the same bed
# settings always lands at the same size and two models stay comparable.
# Normalising each net to 0-1 would destroy that.
_sc = bridge.uv_scale(bpy.context.scene.cbu.bed_x, bpy.context.scene.cbu.bed_y)
_mm = arr.max() / _sc
check(_mm > 100.0, f"UVs carry real size, not a normalised 0-1 fit ({_mm:.0f} mm)")
check(abs(_sc - bridge.uv_scale(bpy.context.scene.cbu.bed_x,
                                bpy.context.scene.cbu.bed_y)) < 1e-18,
      "the scale depends only on the bed, not on the model")


def uv_edge_lengths(mesh, layer):
    out = []
    lay = mesh.uv_layers[layer]
    for poly in mesh.polygons:
        li = list(poly.loop_indices)
        for i in range(len(li)):
            a = np.array(lay.data[li[i]].uv)
            b = np.array(lay.data[li[(i + 1) % len(li)]].uv)
            v0 = np.array(mesh.vertices[mesh.loops[li[i]].vertex_index].co)
            v1 = np.array(mesh.vertices[
                mesh.loops[li[(i + 1) % len(li)]].vertex_index].co)
            out.append((np.linalg.norm(b - a), np.linalg.norm(v1 - v0) * 1000.0))
    out = np.array(out)
    # The net is stored scaled so it fits the UV editor's unit square; undo
    # that here so both columns are millimetres and the comparison is a true
    # isometry check rather than a similarity one.
    out[:, 0] /= uv_mm_scale()
    return out


def uv_mm_scale():
    s = bpy.context.scene.cbu
    return bridge.uv_scale(s.bed_x, s.bed_y)


# Blender stores UVs as float32 (measured: coordinates round-trip at ~3e-8
# relative). An edge LENGTH is a difference of two large coordinates, so
# cancellation amplifies the relative error and only ABSOLUTE mm is
# meaningful here. Observed worst case across these tests is ~9e-5 mm - 90
# nanometres, about 3000x under the kerf.
#
# This is a storage limit of the UV layer, not of the solver: the core works
# in float64 (reported isometry error ~1e-15) and the exported SVG/DXF is
# generated from the core geometry, never read back from these UVs. The UV
# layer is the 2D *view*, not the source of truth.
UV_TOL_MM = 1e-3

L = uv_edge_lengths(obj.data, bridge.UV_LAYER)
err = np.abs(L[:, 0] - L[:, 1]).max()
check(err < UV_TOL_MM, f"UV net is isometric to the 3D mesh (max err {err:.2e} mm)")
check(err < 1e-4, f"UV error is at the float32 storage limit ({err:.1e} mm)")

# ------------------------------------------------------- forced seam marks
bpy.ops.object.mode_set(mode="EDIT")
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()
for e in bm.edges:
    e.select_set(False)
target = bm.edges[0]
target.select_set(True)
tv = tuple(sorted(v.index for v in target.verts))
bmesh.update_edit_mesh(obj.data)
r = bpy.ops.cbu.mark(mode="CUT")
check(r == {"FINISHED"}, "mark cut runs in edit mode")
bpy.ops.object.mode_set(mode="OBJECT")

cuts, folds = bridge.read_constraints(obj.data)
check(tv in cuts, f"forced cut round-trips through the mesh attribute {tv} in {cuts}")

bpy.ops.cbu.unfold()
# the marked edge must be a cut, i.e. NOT a fold, in the result
_, islands, parts, sheets, rep = ops._LAST[obj.name]
pm2 = ops._LAST[obj.name][0]
fold_keys = set()
for isl in islands:
    for f in isl.folds:
        pass
G = pm2.dual_graph()
kept = {tuple(sorted(d["verts"])) for a, b, d in G.edges(data=True)}
check(tv in kept, "the marked edge exists in the dual graph")
from cbu_blender.vendor.cardboard_unfolder.unfold import apply_constraints
G2, notes = apply_constraints(G, cuts, folds)
kept2 = {tuple(sorted(d["verts"])) for a, b, d in G2.edges(data=True)}
check(tv not in kept2, "forced cut removes the arc from the dual graph")

bpy.ops.cbu.clear_marks()
cuts2, _ = bridge.read_constraints(obj.data)
check(not cuts2, "clear marks empties the constraint set")

# ---------------------------------------------------------------- pack UV
# CANCELLED is a legitimate answer: nesting declines rather than take a
# layout that needs more sheets than the packing already in hand.
r = bpy.ops.cbu.pack_uv()
check(r in ({"FINISHED"}, {"CANCELLED"}), f"pack_uv runs (got {r})")
L2 = uv_edge_lengths(obj.data, bridge.UV_LAYER)
err2 = np.abs(L2[:, 0] - L2[:, 1]).max()
check(err2 < UV_TOL_MM,
      f"pack_islands with scale=False preserves isometry (max err {err2:.2e} mm)")

# ----------------------------------------------------------------- export
out = Path("/tmp/cbu_blender_out")
s.out_dir = str(out)
r = bpy.ops.cbu.export()
check(r == {"FINISHED"}, f"export runs (got {r})")
svgs = sorted(out.glob("*.svg"))
check(len(svgs) >= 1, f"wrote {len(svgs)} svg(s)")
if svgs:
    import xml.etree.ElementTree as ET
    root = ET.parse(svgs[0]).getroot()
    labels = {g.get("{http://www.inkscape.org/namespaces/inkscape}label")
              for g in root.iter("{http://www.w3.org/2000/svg}g")}
    check({"CUT", "ENGRAVE"} <= labels, f"svg layers present: {sorted(l for l in labels if l)}")
    check(not {"PERF_M", "PERF_V"} & labels,
          "no fold score layers while Score folds is off (the default)")
check((out / "manifest.json").exists(), "manifest written")

# --- fold scores: off by default, an option, their own colour in the 2D view
check(s.score_folds is False, "Score folds is off by default")
_perf_len = sum(g.length for p in ops._LAST[obj.name][2]
                for g in (p.perf_m, p.perf_v) if not g.is_empty)
_live = overlay._live_scores.get(obj.name) or []
_live_len = sum(float(np.sum(np.hypot(*np.diff(l, axis=0).T))) for l in _live)
check(_perf_len > 0 and abs(_live_len - _perf_len) < 1e-6 * _perf_len + 1e-6,
      f"the score lines reach the 2D view at their real length "
      f"({_live_len:.3f} of {_perf_len:.3f} mm)")
check("score_folds" in overlay._draw_bed.__code__.co_names
      and "_score_batch" in overlay._draw_bed.__code__.co_names,
      "and are drawn only while Score folds is on")
# Not merely unequal: the first choice, orange, was a different tuple from
# everything and still read as the bed outline and ruler beside it.
_others = [overlay.CUT_RGBA, overlay.LABEL_RGBA, overlay.SHEET_RGBA,
           overlay.AXIS_RGBA, overlay.USABLE_RGBA, *overlay.mark_colors()]
_near = min(float(np.abs(np.subtract(overlay.SCORE_RGBA[:3], c[:3])).max())
            for c in _others)
check(_near > 0.3, f"in a colour of their own (closest other colour "
                   f"differs by {_near:.2f} in some channel)")
_v, _c = overlay.stroke_segments(_live, 1.0, overlay.SCORE_RGBA)
check(_v and set(_c) == {overlay.SCORE_RGBA}, "every score segment in that colour")
s.score_folds = True
check(bpy.ops.cbu.export() == {"FINISHED"}, "export with Score folds on")
root = ET.parse(svgs[0]).getroot()
labels = {g.get("{http://www.inkscape.org/namespaces/inkscape}label")
          for g in root.iter("{http://www.w3.org/2000/svg}g")}
check({"PERF_M", "PERF_V"} & labels, "and the score layers are back in the SVG")
check("; --- SCORE" in sorted(out.glob("*.nc"))[0].read_text(),
      "and in the G-code")
s.score_folds = False

# G-code for LaserGRBL: one .nc per sheet, at the Laser panel's settings -
# read at Export, so changing a power needs no new Unfold.
ncs = sorted(out.glob("*.nc"))
check(len(ncs) == len(svgs) >= 1, f"one .nc per sheet ({len(ncs)} for {len(svgs)})")
for _f in out.glob("*.nc"):
    _f.unlink()
s.cut_power, s.cut_speed, s.cut_passes = 900.0, 450.0, 2
s.air_assist = True
check(bpy.ops.cbu.export() == {"FINISHED"}, "export with other laser settings")
_nc = sorted(out.glob("*.nc"))[0].read_text()
check("; CUT     S900 F450 x2 air" in _nc and "M4 S900" in _nc and "M8" in _nc,
      "the panel's cut power, speed, passes and air reach the file")
s.cut_power = 2000.0                    # above $30
try:
    _r = bpy.ops.cbu.export()
except RuntimeError as exc:
    _r = {"CANCELLED"}
    print("   ", exc)
check(_r == {"CANCELLED"}, "a power above Max power is refused, not written")
s.cut_power, s.cut_speed, s.cut_passes = 1000.0, 500.0, 1
s.air_assist = False
s.write_gcode = False
for _f in out.glob("*.nc"):
    _f.unlink()
check(bpy.ops.cbu.export() == {"FINISHED"} and not list(out.glob("*.nc")),
      "and none at all with Write G-code off")
s.write_gcode = True

# ------------------------------------------------------------- ico sphere
obj2 = fresh("Ico", "ico_sphere_add".replace("_add", ""), subdivisions=2, radius=0.13)
s = bpy.context.scene.cbu
s.trials = 120
s.target_height = 260.0
r = bpy.ops.cbu.unfold()
check(r == {"FINISHED"}, f"unfold on a 320-face sphere (got {r})")
check(s.last_islands >= 1, f"produced {s.last_islands} island(s), {s.last_sheets} sheet(s)")
L3 = uv_edge_lengths(obj2.data, bridge.UV_LAYER)
scale = 260.0 / (max(v.co.z for v in obj2.data.vertices) -
                 min(v.co.z for v in obj2.data.vertices)) / 1000.0
err3 = np.abs(L3[:, 0] - L3[:, 1] * scale).max()
check(err3 < UV_TOL_MM,
      f"scaled net stays isometric (max err {err3:.2e} mm, float32 UV limit)")
# scale-independent form: every UV length must be the SAME multiple of its
# 3D length. This needs no knowledge of the scale factor at all.
ratio = L3[:, 0] / np.maximum(L3[:, 1], 1e-12)
spread = (ratio.max() - ratio.min()) / ratio.mean()
check(spread < 1e-5,
      f"UV/3D length ratio is uniform to {spread:.1e} - a pure similarity")
core_iso = float(bpy.context.scene.cbu.last_report.split("isometry ")[1].split(" ")[0])
check(core_iso < 1e-12,
      f"the core itself is exact in float64 (isometry {core_iso:.0e})")

# ------------------------------------------------------------ ngon mesh
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.08, depth=0.16)
obj3 = bpy.context.active_object
pm3 = bridge.polymesh_from_object(obj3, merge_coplanar=False)
check(len(pm3.faces) == 14, f"n-gon caps preserved without merging (got {len(pm3.faces)})")
bpy.context.scene.cbu.merge_coplanar = False
bpy.context.scene.cbu.trials = 40
r = bpy.ops.cbu.unfold()
check(r == {"FINISHED"}, "unfold works on native n-gons with merging off")

# ------------------------------------------------ net layout and bed reference
# The net UV layer is written in millimetres, so it is enormous next to the
# editor's 0-1 grid. It used to be laid out as one horizontal strip, which made
# it impossible to frame: a 6-island cube net ran 12650 x 2000 mm. Rows must
# wrap instead, so the net stays roughly as wide as a sheet.
obj = fresh("Rows", "ico_sphere", subdivisions=2)
obj.scale = (0.14, 0.14, 0.14)
bpy.ops.object.transform_apply(scale=True)
bpy.context.scene.cbu.merge_coplanar = True
bpy.ops.cbu.unfold()

import numpy as _np
_d = _np.zeros(len(obj.data.loops) * 2)
obj.data.uv_layers[bridge.UV_LAYER].data.foreach_get("uv", _d)
_d = _d.reshape(-1, 2)
_w, _h = (_d.max(0) - _d.min(0)) / uv_mm_scale()        # back to mm
_sc_ = bpy.context.scene.cbu
_nsh = ops._LAST[obj.name][3]
_cols = bridge.sheet_columns(_nsh, _sc_.bed_x, _sc_.bed_y)
_rows = -(-_nsh // _cols)
_g_ = bridge.sheet_gutter(_sc_.bed_x, _sc_.bed_y)
check(_w <= _cols * _sc_.bed_x + (_cols - 1) * _g_ - 2 * _sc_.margin + 1e-6
      and _h <= _rows * _sc_.bed_y + (_rows - 1) * _g_ - 2 * _sc_.margin + 1e-6,
      f"the net fits the {_cols} x {_rows} grid of its {_nsh} sheet(s) "
      f"({_w:.0f} x {_h:.0f} mm)")
check(max(_w, _h) / max(min(_w, _h), 1e-9) < 8.0,
      f"net is a block, not a strip (aspect {max(_w,_h)/max(min(_w,_h),1e-9):.1f})")

_b = bridge.net_bounds(obj.data)
check(_b is not None and abs((_b[2] - _b[0]) / uv_mm_scale() - _w) < 1e-6,
      "net_bounds agrees with the UV layer")

# the whole point of the exercise: a sheet has to fit the editor's unit square
_sc = uv_mm_scale()
_bx, _by = bpy.context.scene.cbu.bed_x, bpy.context.scene.cbu.bed_y
check(_bx * _sc <= 1.0 and _by * _sc <= 1.0,
      f"a sheet fits the 1x1 UV grid ({_bx*_sc:.3f} x {_by*_sc:.3f})")
check(_bx * _sc > 0.5 or _by * _sc > 0.5,
      "and is not needlessly tiny inside it")
_inv = 1.0 / _sc
check(_inv == int(_inv) and (int(_inv) & (int(_inv) - 1)) == 0,
      f"the scale is a power of two (1/{_inv:.0f})")
# a power of two is exact in binary floating point, so scaling must not
# perturb a single mantissa bit - that is the reason for choosing one
_probe = np.array([123.456789012345, 0.1, 40.0, 1e-7])
check(np.all(_probe * _sc / _sc == _probe),
      "scaling by it round-trips bit for bit")
check(overlay.NET_UV == bridge.UV_LAYER,
      "overlay's net layer name tracks bridge.UV_LAYER")

# the bed reference tiles the net at true size
_v, _c = overlay.bed_segments((0.0, 0.0, 1200.0, 800.0), 600.0, 400.0, 8.0)
check(len(_v) == 4 * 16, f"2x2 sheets -> 4 tiles of 8 segments (got {len(_v)//16})")
check(len(_v) == len(_c), "every bed vertex carries a colour")
# Regression: these were 2-tuples once. The builtin shaders declare `pos` as
# vec3, and batch_for_shader given 2-component data is a buffer format
# mismatch that crashes Blender outright instead of raising - so it cannot be
# caught at draw time and has to be asserted on the data.
check(all(len(v) == 3 for v in _v),
      "bed vertices are vec3, as the builtin shaders require")
check(all(len(c) == 4 for c in _c), "bed colours are RGBA")
check(all(v[2] == 0.0 for v in _v), "bed is flat in z")
_xs = [v[0] for v in _v]
check(abs(max(_xs) - 1200.0) < 1e-6, "bed grid spans the net, in mm")
check(overlay.bed_segments((0, 0, 100, 100), 0.0, 400.0, 8.0) == ([], []),
      "a zero-width bed draws nothing instead of dividing by zero")
_v2, _ = overlay.bed_segments((0.0, 0.0, 1e9, 1e9), 600.0, 400.0, 8.0)
check(len(_v2) // 16 <= overlay.MAX_TILES,
      f"a runaway net is capped at {overlay.MAX_TILES} tiles")

# ------------------------------------------------ sheets laid out squarely
# Sheets used to stack upward whatever the bed: three 400 x 900 sheets made a
# 400 x 2700 strip. The layout picks the column count whose bounding box is
# closest to square, and the overlay draws exactly the sheets it placed.
check(bridge.sheet_columns(3, 400.0, 900.0) == 3,
      "three tall sheets sit side by side, not stacked 2700 mm high")
check(bridge.sheet_columns(3, 600.0, 400.0) == 2,
      "three wide sheets go two to a row")
check(bridge.sheet_columns(1, 400.0, 900.0) == 1, "one sheet is one sheet")


def _aspect_err(n, c, w, h):
    # sheets a gutter apart, so the arrangement is measured with the gutters
    r = -(-n // c)
    g = bridge.sheet_gutter(w, h)
    return abs(math.log((c * (w + g)) / (r * (h + g))))


_bad_layout = []
for _n in range(1, 25):
    for _w, _h in ((400.0, 900.0), (600.0, 400.0), (300.0, 300.0), (900.0, 200.0)):
        _c = bridge.sheet_columns(_n, _w, _h)
        _best = min(_aspect_err(_n, c, _w, _h) for c in range(1, _n + 1))
        if _aspect_err(_n, _c, _w, _h) > _best + 1e-9:
            _bad_layout.append((_n, _w, _h, _c))
check(not _bad_layout, f"no column count is squarer than the one chosen {_bad_layout[:3]}")
_orig = bridge.sheet_layout(5, 400.0, 900.0)
_gut = bridge.sheet_gutter(400.0, 900.0)
check(len(set(_orig)) == 5
      and all(abs(ox % (400.0 + _gut)) < 1e-6 and abs(oy % (900.0 + _gut)) < 1e-6
              for ox, oy in _orig),
      "each sheet gets its own slot, a gutter apart")
_touch = [(a, b) for i, a in enumerate(_orig) for b in _orig[i + 1:]
          if abs(a[0] - b[0]) < 400.0 + _gut - 1e-6
          and abs(a[1] - b[1]) < 900.0 + _gut - 1e-6]
check(not _touch, "no two sheets touch - the ruler lives in the gutter between them")
_park = bridge.sheet_origin(-1, 5, 400.0, 900.0)
check(_park[1] >= max(oy for _ox, oy in _orig) + 900.0,
      "a part no sheet can hold parks above the whole grid, never on a sheet")
_vs, _ = overlay.bed_segments((0.0, 0.0, 1200.0, 1800.0), 400.0, 900.0, 8.0,
                              origins=_orig)
check(len(_vs) == 5 * 16,
      f"the overlay draws the 5 sheets used, not the 6 slots of the grid ({len(_vs)//16})")

# ------------------------------------------------ face selection survives
# With UV Sync Selection off the UV editor only draws the UVs of SELECTED
# faces. An operator that borrows the selection to drive Blender's own tools
# and leaves a working subset behind therefore hides most of the net and makes
# it unpickable - the add-on looks like it broke the 2D view.
obj = fresh("SelKeep", "ico_sphere", subdivisions=2)
obj.scale = (0.14,) * 3
bpy.ops.object.transform_apply(scale=True)
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
for _p in obj.data.polygons:
    _p.select = True
obj.data.update()
_n = len(obj.data.polygons)

bpy.ops.cbu.unfold()
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
check(sum(1 for p in obj.data.polygons if p.select) == _n,
      "unfold leaves the face selection alone")

bpy.ops.cbu.pack_uv()
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
check(sum(1 for p in obj.data.polygons if p.select) == _n,
      "nesting gives the selection back after borrowing it")

for _p in obj.data.polygons:
    _p.select = False
obj.data.update()
check(bpy.ops.cbu.select_all_faces() == {"FINISHED"}, "the rescue button runs")
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
check(sum(1 for p in obj.data.polygons if p.select) == _n,
      "and brings the whole net back into view")

# ------------------------------------------------ the bed grid is anchored
# The net starts at the margin, so tiling the sheet grid from the net's own
# bounding box shifts every sheet - and its inner margin line - by exactly one
# margin. Parts sitting correctly at x=8 then appear 8 mm outside a line drawn
# at 16, and a perfectly good layout reads as ignoring the margin. Measured on
# a pikachu net: the grid spanned x[8.1, 308.1] instead of x[0, 300].
_bw, _bh = 0.586, 0.586
_offset_net = (0.0159, 0.016, 0.5567, 1.1052)     # a real net, margin included
_g = overlay.bed_grid(_offset_net, _bw, _bh)
check(_g[0] == 0.0 and _g[1] == 0.0,
      f"the sheet grid starts at the origin, not the net ({_g[0]}, {_g[1]})")
check(_g[2] >= _offset_net[2] and _g[3] >= _offset_net[3],
      "and still covers the whole net")
check(overlay.bed_grid((0.0, 0.0, 0.1, 0.1), _bw, _bh)[2] >= _bw,
      "a net smaller than one sheet still gets a whole sheet drawn")

_v, _c = overlay.bed_segments(_g, _bw, _bh, 0.0156)
_xs = [q[0] for q in _v]
_ys = [q[1] for q in _v]
check(min(_xs) == 0.0 and min(_ys) == 0.0,
      f"the drawn rectangles start at the origin ({min(_xs)}, {min(_ys)})")

# ------------------------------------------------ bed changes keep the net honest
# uv_scale is derived from the bed, so the same stored UVs mean different
# millimetres once the bed changes - a net laid out on a 300 mm bed read as
# half size on a 600 mm one. The bed reference redraws at the new size
# immediately, so the view showed the net outside a margin the packer had
# respected.
obj = fresh("BedChange", "ico_sphere", subdivisions=1)
obj.scale = (0.14,) * 3
bpy.ops.object.transform_apply(scale=True)
_s = bpy.context.scene.cbu
_s.bed_x, _s.bed_y = 300.0, 300.0
bpy.ops.cbu.unfold()
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")


def _net_mm():
    _m = props.material_from(bpy.context.scene.cbu)
    return bridge.read_net(obj.data, bridge.uv_scale(*_m.bed))


def _island_sizes(net):
    _pm, _isl = ops._LAST[obj.name][0], ops._LAST[obj.name][1]
    _loops = bridge.island_loops(obj.data, _isl, _pm)
    return {k: net[v].max(0) - net[v].min(0) for k, v in _loops.items()}


_before = _net_mm()
_s.bed_x, _s.bed_y = 600.0, 400.0
_after = _net_mm()
# Per island, not the net's overall extent: the sheets are laid out in the
# squarest grid for the bed, so a new bed can legitimately move whole sheets.
# What must not change is the size of anything - the stored millimetres.
# 1e-3 mm, not exact: the UV layer is float32 (~9e-5 mm at these sizes), and
# an island moved to a different slot rounds differently.
_sb, _sa = _island_sizes(_before), _island_sizes(_after)
_moved = [k for k in _sb if k not in _sa or np.abs(_sb[k] - _sa[k]).max() > 1e-3]
check(not _moved and _sb.keys() == _sa.keys(),
      f"the net keeps its millimetres when the bed changes (islands {_moved[:4]})")
check(_after[:, 0].min() >= _s.margin - 1e-3,
      "and still starts on the margin")

# ------------------------------------------------ the cut outline in 2D
# The UV layer holds only the islands. Glue tabs are generated in the core and
# live solely in the exported parts, so a 2D view drawn from UVs alone is
# smaller than what the laser follows - and anyone arranging islands by eye is
# optimising the wrong shape.
obj = fresh("CutView", "ico_sphere", subdivisions=2)
obj.scale = (0.14,) * 3
bpy.ops.object.transform_apply(scale=True)
bpy.ops.cbu.unfold()
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")

_groups = overlay._net_cuts.get(obj.name)
check(bool(_groups), "unfold publishes the cut outline to the 2D view")

_mat = props.material_from(bpy.context.scene.cbu)
_sc = bridge.uv_scale(*_mat.bed)
_rings = overlay.live_cut_rings(obj, _sc)
check(bool(_rings), "the cut outline resolves against the current layout")
_cut = np.vstack(_rings)
_d = np.zeros(len(obj.data.loops) * 2)
obj.data.uv_layers[bridge.UV_LAYER].data.foreach_get("uv", _d)
_net = _d.reshape(-1, 2) / _sc
# With the defaults - edge_offset 0, tabs off - the cut IS the island outline,
# so this compares two views of one shape. The net UVs are float32, hence the
# 1e-3 mm tolerance rather than 1e-6: at 500 mm coordinates the storage error
# alone is ~9e-5 mm, and an exact comparison here measures that and nothing
# else.
check(abs(_cut[:, 0].max() - _net[:, 0].max()) < 1e-3
      and abs(_cut[:, 1].max() - _net[:, 1].max()) < 1e-3,
      "at the defaults the cut sits on the island outline")

# ...and the things that DO make it bigger still do. Measured on the parts
# rather than the overlay, so it costs no second unfold.
_pmx, _islx, _partsx, _shx, _repx = ops._LAST[obj.name]
_L = ops._core()[1]
_flush = sum(p.cut.area for p in _partsx)
_m2 = props.material_from(bpy.context.scene.cbu)
_m2.edge_offset = 0.5
check(sum(p.cut.area for p in _L.build_parts(_islx, _m2)) > _flush,
      "a positive edge offset grows the cut")
_m2.edge_offset = -0.5
check(sum(p.cut.area for p in _L.build_parts(_islx, _m2)) < _flush,
      "and a negative one cuts the part back, for thick board")
_m3 = props.material_from(bpy.context.scene.cbu)
_m3.use_tabs = True
check(sum(p.cut.area for p in _L.build_parts(_islx, _m3)) > _flush,
      "glue tabs still reach past the islands")

# Interlocking fingers: the panel settings reach the core, depth follows the
# board until overridden, and a fingered net trades area rather than adding it.
_sfc = bpy.context.scene.cbu
check(_sfc.use_fingers is False and _sfc.finger_width == 15.0,
      "fingers are off by default, 15 mm wide")
_sfc.use_fingers = True
_m4 = props.material_from(_sfc)
_rp = _sfc.bl_rna.properties
check(abs(_rp["finger_clearance"].default - 0.05) < 1e-6
      and abs(_m4.finger_clearance - 0.05) < 1e-6,
      "finger clearance defaults to 0.05 mm and reaches the core")
check(_rp["relief_fixed"].hard_min == 0.0,
      "a fixed relief width can be set however thin")
_sfc.relief_adaptive, _sfc.relief_fixed = False, 0.3
_mt = props.material_from(_sfc)
_plan = sys.modules[_L.__name__.rsplit(".", 1)[0] + ".folds"].plan_folds
check(any(p.style == "relief" for i in _islx for p in _plan(i, _mt)),
      "a fixed 0.3 mm width is cut as a slot, not perforated")
_sfc.relief_adaptive, _sfc.relief_fixed = True, 3.0
check(_m4.use_fingers and _m4.finger_width == 15.0
      and abs(_m4.finger_depth_used - _sfc.thickness) < 1e-6,
      "the finger switch reaches the core, depth = board thickness")
_sfc.finger_depth_override = True
_sfc.finger_depth = 4.5
check(abs(props.material_from(_sfc).finger_depth_used - 4.5) < 1e-6,
      "...until the depth is overridden")
_sfc.finger_depth_override = False
_partsf = _L.build_parts(_islx, _m4)
_m4.use_fingers = False
_partsu = _L.build_parts(_islx, _m4)       # same frame: unpacked, no fingers
_m4.use_fingers = True
_rects = [p.cut for p in _partsf if p.cut.wkb != _partsx[_partsf.index(p)].cut.wkb]
check(bool(_rects), "fingers change the cut outline")
# The mesh is the board's mid-plane: a finger stands half the depth out and
# no further - not the whole depth, which stood a board proud of the model.
_h = _m4.finger_depth_used / 2.0
check(all(q.cut.buffer(_h + 1e-6).contains(p.cut)
          and p.cut.difference(q.cut).area > 0.0
          for p, q in zip(_partsf, _partsu) if p.cut.wkb != q.cut.wkb),
      "a finger stands half the depth out past the part, and no further")
_sfc.corner_hole = True
check(props.material_from(_sfc).corner_hole,
      "the corner hole switch reaches the core")
_sfc.corner_hole = False
check(sum(p.cut.area for p in _partsf) < _flush,
      "...so a fingered net is smaller, by exactly its notches")
check(all(len(getattr(p.cut, "geoms", [p.cut]))
          == len(getattr(q.cut, "geoms", [q.cut]))
          for p, q in zip(_partsf, _partsx)),
      "fingers never break a part into pieces")
_sfc.use_fingers = False

# vec3 or Blender dies on the draw, and this cannot be caught at draw time
_v, _c = overlay.cut_segments(_rings, _sc)
check(_v and all(len(q) == 3 for q in _v), "cut vertices are vec3")
check(len(_v) == len(_c) and all(len(q) == 4 for q in _c), "cut colours are RGBA")
check(all(q[2] == 0.0 for q in _v), "the cut outline is flat in z")

# and it must follow an island that is dragged in the UV editor, or it invites
# arranging against a shape that has moved
_pm, _isl, _parts, _s2, _r2 = ops._LAST[obj.name]
_loops = bridge.island_loops(obj.data, _isl, _pm)
_b4 = [r.mean(axis=0) for r in overlay.live_cut_rings(obj, _sc)]
_pts = bridge.read_net(obj.data, _sc)
_tgt = sorted(_loops)[0]
_pts[_loops[_tgt]] += np.array([40.0, 25.0])
bridge.write_net(obj.data, _pts, _sc)
_af = [r.mean(axis=0) for r in overlay.live_cut_rings(obj, _sc)]
check(len(_b4) == len(_af), "ring count is stable across a drag")
_shift = [np.asarray(b) - np.asarray(a) for a, b in zip(_b4, _af)]
_moved = [d for d in _shift if np.linalg.norm(d) > 1e-6]
_still = [d for d in _shift if np.linalg.norm(d) <= 1e-6]
check(bool(_moved), "some of the outline moved with the dragged island")
check(all(abs(d[0] - 40.0) < 1e-3 and abs(d[1] - 25.0) < 1e-3 for d in _moved),
      f"and it moved by exactly the drag (got {_moved[0] if _moved else None})")
check(bool(_still), "the islands that were not dragged stayed put")

# ------------------------------------------------ every icon name is real
# A typo'd icon raises inside draw(), which only happens with a real UI region
# - so a headless suite never reaches it and the panel is simply broken for
# the user. The names are checkable without drawing anything.
import re as _re
_valid = set(bpy.types.UILayout.bl_rna.functions["operator"]
             .parameters["icon"].enum_items.keys())
_ui_src = (ROOT / "cbu_blender" / "ui.py").read_bytes().decode()
_used = sorted(set(_re.findall(r'icon="([A-Z0-9_]+)"', _ui_src)))
_bad = [i for i in _used if i not in _valid]
check(not _bad, f"every icon in ui.py exists ({len(_used)} used, bad: {_bad})")

# "Seams" is what happens at a cut - tabs, fingers. The marks that tell the
# solver where to cut are Edge Constraints. Read off the draw code objects,
# which needs no UI region.
from cbu_blender import ui as _ui                        # noqa: E402
check(_ui.CBU_PT_constraints.bl_label == "Edge Constraints"
      and _ui.CBU_PT_seams.bl_label == "Seams",
      "the mark panel is Edge Constraints, the joinery panel is Seams")
# lay.prop(s, "name") stores the name as a constant, s.name as a name, so
# a panel's settings are the union of the two.
_seam_names = (_ui.CBU_PT_seams.draw.__code__.co_names
               + _ui.CBU_PT_seams.draw.__code__.co_consts)
_mat_names = (_ui.CBU_PT_material.draw.__code__.co_names
              + _ui.CBU_PT_material.draw.__code__.co_consts)
check({"use_tabs", "use_fingers", "finger_width", "tab_depth"} <= set(_seam_names),
      "glue tabs and fingers live in the Seams panel")
check(not {"use_tabs", "use_fingers"} & set(_mat_names),
      "...and not in Material any more")
check("mark" not in _ui.CBU_PT_seams.draw.__code__.co_consts
      and "cbu.mark" in _ui.CBU_PT_constraints.draw.__code__.co_consts,
      "the cut / fold marks are Edge Constraints")
_order = [c.__name__ for c in _ui.classes]
check(_order.index("CBU_PT_seams") == _order.index("CBU_PT_material") + 1,
      "Seams sits right after Material, whose thickness sets the finger depth")

# The swatch is the colour key: red is a forced cut, blue a forced fold, and
# they are the colours the mark is actually drawn in.
check("COLOR_RED" in _used and "COLOR_BLUE" in _used,
      "the mark buttons carry colour swatches")
check(overlay.CUT_MARK_RGBA[0] > overlay.CUT_MARK_RGBA[2],
      "and the cut colour really is the red one")
check(overlay.FOLD_MARK_RGBA[2] > overlay.FOLD_MARK_RGBA[0],
      "and the fold colour really is the blue one")
check(_ui_src.count("mark_key(") >= 3,
      "the key is defined once and drawn everywhere marks can be confused")

# ------------------------------------------------ forced marks draw on top
# The tint is lifted off the mesh to stop it z-fighting the faces, which put it
# in FRONT of marks drawn on the mesh itself - so the colour swallowed the one
# thing you are looking at while editing seams.
check(overlay.MARK_LIFT > 1.0,
      f"marks are lifted further than the tint (x{overlay.MARK_LIFT})")
check(overlay.CUT_MARK_RGBA != overlay.FOLD_MARK_RGBA,
      "a forced cut and a forced fold are different colours")

_eps = overlay.lift_amount(obj)
bpy.ops.object.mode_set(mode="EDIT")
_bm2 = bmesh.from_edit_mesh(obj.data)
_bm2.edges.ensure_lookup_table()
_lay2 = bridge.edge_layer(_bm2)
_bm2.edges[0][_lay2] = bridge.FORCE_CUT
_bm2.edges[1][_lay2] = bridge.FORCE_FOLD
bmesh.update_edit_mesh(obj.data)
bpy.ops.object.mode_set(mode="OBJECT")

_mp, _mc = overlay.mark_lines(obj, _eps)
check(len(_mp) == 4, f"two marked edges give four vertices ({len(_mp)})")
# Against the LIVE colours, not the constants: the panel swatch drives these
# now, and Blender stores a colour as float32 - 0.85 comes back as
# 0.8500000238, which no exact comparison with the default will ever match.
_ck, _fk = overlay.mark_colors()
check({tuple(c) for c in _mc} == {_ck, _fk},
      "each mark carries its own colour")
check(_ck != _fk, "and the two are distinguishable")
_cbu5 = bpy.context.scene.cbu
_cbu5.mark_cut_color = (1.0, 0.0, 1.0)
check(overlay.mark_colors()[0][:3] == (1.0, 0.0, 1.0),
      "the panel swatch is what the overlay draws with - the key cannot drift")
_cbu5.mark_cut_color = overlay.CUT_MARK_RGBA[:3]
_co = np.asarray([v.co[:] for v in obj.data.vertices])
_off = [float(np.linalg.norm(_co - np.asarray(q), axis=1).min()) for q in _mp]
check(min(_off) > _eps * 1.5,
      f"marks stand clear of the tint ({min(_off):.6f} vs tint {_eps:.6f})")
check(abs(min(_off) - _eps * overlay.MARK_LIFT) < 1e-9,
      "lifted by exactly MARK_LIFT along the vertex normal")

# ------------------------------------------------ and in the 2D view
_sc4 = bridge.uv_scale(*props.material_from(bpy.context.scene.cbu, obj).bed)
_segs = overlay.live_mark_segments(obj, _sc4)
check(_segs is not None and len(_segs) >= 2,
      f"forced marks reach the net too ({0 if _segs is None else len(_segs)})")
check({v for _a, _b, v in _segs} <= {1, 2}, "carrying which kind they are")
_net4 = bridge.read_net(obj.data, _sc4)
_all4 = np.vstack([np.vstack([a, b]) for a, b, _v in _segs])
check(_all4[:, 0].min() >= _net4[:, 0].min() - 1e-6
      and _all4[:, 0].max() <= _net4[:, 0].max() + 1e-6
      and _all4[:, 1].min() >= _net4[:, 1].min() - 1e-6
      and _all4[:, 1].max() <= _net4[:, 1].max() + 1e-6,
      "every mark lies on the net, not beside it")

_mb = overlay._marks_batch.__code__.co_names
check("_live_marks" in _mb and "_live_uv" not in _mb,
      "the mark batch reads the cache rather than sampling live UVs")
_dn2 = overlay._draw_bed.__code__.co_names
check("show_net_marks" in _dn2, "the 2D draw honours show_net_marks")
check(_dn2.index("_marks_batch") > _dn2.index("_label_batch"),
      "and draws the marks last, so nothing can cover them")
check(bpy.context.scene.cbu.show_net_marks is True, "shown by default")

# Marking is deliberately not a geometry change, so nothing else refreshes it.
check("refresh_marks" in ops.CBU_OT_mark.execute.__code__.co_names,
      "marking refreshes the marks rather than waiting for the next unfold")
_rm = overlay.refresh_marks.__code__.co_names
check("mark_lines" in _rm and "rebuild" not in _rm,
      "refresh_marks is the cheap half of a rebuild, not a whole one")

bpy.ops.object.mode_set(mode="EDIT")
check(overlay.live_mark_segments(obj, _sc4) is None,
      "and it refuses to sample the net in Edit Mode, like everything else")
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.cbu.clear_marks()

# ------------------------------------------------ the millimetre ruler
# The UV editor measures in UV units and the factor is the bed rounded UP to a
# power of two - 512 mm at a 300 mm bed, 1024 at 600. A length read off the
# editor against the wrong remembered constant is out by exactly two, and that
# reads as a bug in whatever was measured. A drawn ruler needs no remembering.
for _span, _want in ((300.0, 50.0), (600.0, 100.0), (1200.0, 200.0),
                     (30.0, 5.0), (3.0, 0.5)):
    _got = overlay.nice_step(_span)
    check(_got == _want,
          f"nice_step({_span:g}) is a round {_want:g}, got {_got:g}")
check(overlay.nice_step(0.0) > 0, "a zero span does not divide by zero")

# The grid and ruler read in round millimetres and adapt to zoom the way
# Blender's own grid does - lines never closer than GRID_MIN_PX.
for _ppm, _want in ((None, (10.0, 50.0)), (1.0, (10.0, 50.0)),
                    (0.5, (20.0, 100.0)), (2.0, (5.0, 10.0)),
                    (10.0, (1.0, 5.0)), (0.02, (500.0, 1000.0))):
    check(overlay.grid_steps(_ppm) == _want,
          f"grid_steps({_ppm}) is {_want}, got {overlay.grid_steps(_ppm)}")
for _ppm in (0.03, 0.3, 1.7, 4.0, 25.0):
    _mi, _ma = overlay.grid_steps(_ppm)
    check(_mi * _ppm >= overlay.GRID_MIN_PX or _mi == overlay.GRID_STEPS[-1],
          f"at {_ppm} px/mm the minor lines are at least "
          f"{overlay.GRID_MIN_PX:g} px apart ({_mi * _ppm:.1f})")

_sc3 = 1.0 / 1024.0
_W3, _H3 = 400.0, 900.0
_org3 = bridge.sheet_layout(3, _W3, _H3)
_gut3 = bridge.sheet_gutter(_W3, _H3)
_gv, _gc = overlay.sheet_grid_segments(_org3, _W3, _H3, 10.0, 50.0, _sc3)
check(_gv and len(_gv) == len(_gc) and all(len(q) == 3 for q in _gv),
      "the sheet grid emits vec3 segments with a colour each")


check(all(any(ox - 1e-6 <= q[0] / _sc3 <= ox + _W3 + 1e-6
              and oy - 1e-6 <= q[1] / _sc3 <= oy + _H3 + 1e-6 for ox, oy in _org3)
          for q in _gv),
      "every grid line lies on a sheet")
# 400 / 10 - 1 inner verticals and 900 / 10 - 1 inner horizontals per sheet
check(len(_gv) // 2 == 3 * (39 + 89),
      f"10 mm lines strictly inside each of 3 sheets ({len(_gv) // 2})")
check(sum(1 for c in _gc if c == overlay.GRID_MAJOR_RGBA) // 2 == 3 * (7 + 17),
      "every fifth line is a major")

_av, _ac = overlay.axis_segments(_org3, _W3, _H3, _sc3, 10.0, 50.0, _gut3)
check(len(_av) > 0 and len(_av) == len(_ac), "the ruler emits segments")
check(all(len(q) == 3 for q in _av), "ruler vertices are vec3")
check(all(c == overlay.AXIS_RGBA for c in _ac),
      "and are drawn in the ruler's own colour")

# Nothing may be drawn over a sheet - the ruler lives in the gutter, or it
# would sit on top of the parts it is measuring.
_inside = [q for q in _av
           if any(ox + 1e-6 < q[0] / _sc3 < ox + _W3 - 1e-6
                  and oy + 1e-6 < q[1] / _sc3 < oy + _H3 - 1e-6
                  for ox, oy in _org3)]
check(not _inside,
      f"nothing is drawn inside any sheet ({len(_inside)} vert(s))")

# The ticks are ON the grid lines: every grid x-position along a sheet's
# bottom edge has a tick leaving that edge, so the two read against each other.
_tick_x = {round(a[0] / _sc3, 6) for a, b in zip(_av[0::2], _av[1::2])
           if abs(a[0] - b[0]) < 1e-12 and abs(a[1] / _sc3 - _org3[0][1]) < 1e-6}
_grid_x = {round(a[0] / _sc3, 6) for a, b in zip(_gv[0::2], _gv[1::2])
           if abs(a[0] - b[0]) < 1e-12 and _org3[0][0] <= a[0] / _sc3 <= _org3[0][0] + _W3}
check(_grid_x and _grid_x <= _tick_x,
      f"every grid line on sheet 1 has a ruler tick on it "
      f"({len(_grid_x & _tick_x)} of {len(_grid_x)})")

# It must scale with the bed, not with the numbers: the same physical span at
# a different uv_scale gives the same ruler.
_a1, _ = overlay.axis_segments([(0.0, 0.0)], 300.0, 300.0, 1 / 512.0)
_a2, _ = overlay.axis_segments([(0.0, 0.0)], 300.0, 300.0, 1 / 1024.0)
check(len(_a1) == len(_a2),
      f"the same 300 mm reads the same at either scale ({len(_a1)} vs {len(_a2)})")
check(overlay.axis_segments([], 300.0, 300.0, 1 / 512.0) == ([], []),
      "no sheets, no ruler")

check(props.apply_blender_grid(bpy.context.scene.cbu) >= 0,
      "hiding Blender's grid survives having no UV editor open")
check(bpy.context.scene.cbu.show_bed_grid is True
      and bpy.context.scene.cbu.hide_blender_grid is True,
      "the mm grid is on, and Blender's is hidden, by default")

_cbu3 = bpy.context.scene.cbu
check(_cbu3.show_bed_ruler is True, "the ruler is on by default")
_bn = overlay._draw_bed.__code__.co_names
check("show_bed_ruler" in _bn, "the 2D draw honours show_bed_ruler")
check("axis_segments" in overlay._bed_batch.__code__.co_names,
      "and the bed batch is what carries it")

# ------------------------------------------------ annotation in the 2D view
# The engraving is generated in the core and lives only in the exported parts,
# exactly like the glue tabs - so without publishing it the first place a seam
# number appears is the SVG, and the 2D view silently shows an unannotated net.
_groups = overlay._net_cuts.get(obj.name) or []
check(any(g.get("labels") for g in _groups),
      "unfold publishes the engraving to the 2D view, not just the outline")
_nlab = sum(len(g.get("labels") or ()) for g in _groups)
check(_nlab > 0, f"{_nlab} label stroke(s) stored in island space")

_sc2 = bridge.uv_scale(*props.material_from(bpy.context.scene.cbu, obj).bed)
_strokes = overlay.live_label_strokes(obj, _sc2)
check(_strokes is not None and len(_strokes) == _nlab,
      "they resolve against the current layout like the outline does")

# Open polylines, NOT rings: closing a letter draws a chord through it.
_pts = sum(len(x) for x in _strokes)
_v, _c = overlay.stroke_segments(_strokes, _sc2)
check(len(_v) // 2 == _pts - len(_strokes),
      "stroke_segments leaves the polylines open")
check(_v and all(len(q) == 3 for q in _v), "label vertices are vec3")

# Each layer on its own switch. Returning early on show_bed used to hide the
# cut outline and the engraving with it, which is not what the box says.
_names = overlay._draw_bed.__code__.co_names
for _n in ("show_bed", "show_cut", "show_net_labels"):
    check(_n in _names, f"the 2D draw honours {_n}")
check("_label_batch" in _names and "_cut_batch" in _names,
      "and draws both layers from their caches")
_lb = overlay._label_batch.__code__.co_names
check("_live_labels" in _lb and "_live_uv" not in _lb,
      "the label batch reads the cache rather than sampling live UVs")

_cbu2 = bpy.context.scene.cbu
check(_cbu2.show_net_labels is True, "annotation is shown by default")
check(overlay.LABEL_RGBA != overlay.CUT_RGBA,
      "engraving is drawn in its own colour, not the cut's")

# ------------------------------------------------ the model identifier
# Pieces from two models cut on the same afternoon are indistinguishable
# without this: both are grey polygons whose seam numbers each start at 1.
_s = bpy.context.scene.cbu
_s.model_id = "ARMOUR"
check(props.model_id_for(_s, obj) == "ARMOUR", "the panel field wins")
_s.model_id = "  PAD  "
check(props.model_id_for(_s, obj) == "PAD", "and is trimmed")
# The fallback goes through short_id. Label length is the biggest lever on how
# many islands can carry an id, and an imported STL is routinely called
# something like OLD_VERSION_READ_DESCRIPTION_pikachu_1gen_flowalistik.
from cbu_blender.vendor.cardboard_unfolder.tabs import short_id as _short
_s.model_id = "   "
check(props.model_id_for(_s, obj) == _short(obj.name),
      f"whitespace is not an id; falls back to the object ({obj.name!r} -> "
      f"{_short(obj.name)!r})")
_s.model_id = ""
check(props.model_id_for(_s, obj) == _short(obj.name),
      "blank falls back to the object name, shortened")
check(props.material_from(_s, obj).model_id == _short(obj.name),
      "and material_from carries it into the core")
check(len(_short("OLD_VERSION_READ_DESCRIPTION_pikachu_1gen_flowalistik")) == 8,
      "a name too long to engrave becomes 8 characters")

_s.label_edge_frac = 0.4
check(props.material_from(_s, obj).label_edge_frac == pytest_approx(0.4),
      "the per-edge room setting reaches the material")
_s.label_edge_frac = 0.25
check(not (getattr(ui.CBU_PT_labels, "bl_options", set()) or set()),
      "the Annotation panel is open - the Model ID box lives there")
check(props.material_from(_s).model_id != obj.name or not obj.name,
      "without an object it does not invent one from the active selection")

# The height rule reaches the core as the user set it.
_s.label_height, _s.label_min_height = 6.0, 3.0
_m = props.material_from(_s, obj)
check(_m.label_height == 6.0 and _m.label_min_height == 3.0,
      "text height and floor reach the material")
_s.label_folds = False
check(props.material_from(_s, obj).label_folds is False,
      "and the M/V toggle does too")
_s.label_height, _s.label_min_height, _s.label_folds = 3.0, 2.0, True
_rp = _s.bl_rna.properties
check(_rp["label_height"].default == 3.0
      and _rp["label_min_height"].default == 2.0,
      "defaults are 3 mm text with a 2 mm floor")

check(any(c.__name__ == "CBU_PT_labels" for c in ui.classes),
      "the Annotation panel is registered, not just defined")

# ------------------------------------------------ the tint clears the mesh
# The island tint lies on the same planes as the mesh faces, so drawn
# unmodified it z-fights and the object's own polygons punch through the
# colour. GPU state exposes no polygon offset to Python, so the triangles are
# lifted off their own plane when the batch is built.
_sq = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 3.0, 0.0), (0.0, 3.0, 0.0)]
_n = overlay.face_normal(_sq)
check(abs(abs(_n[2]) - 1.0) < 1e-12 and abs(_n[0]) < 1e-12,
      f"face_normal finds the plane of a ring ({_n})")
check(abs(np.linalg.norm(overlay.face_normal(
          [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]))) < 1e-9,
      "a degenerate ring has no normal rather than a NaN one")

_lifted = overlay.lift(_sq, 0.25)
_d = _lifted - np.asarray(_sq, dtype=float)
check(np.allclose(np.linalg.norm(_d, axis=1), 0.25),
      "lift moves every corner by exactly the amount asked for")
check(np.allclose(_d, _d[0]),
      "and moves them all the same way, so the face stays flat and rigid")
check(abs(float(np.dot(_d[0], _n)) - 0.25) < 1e-12,
      "along the face normal, so it clears the surface it sits on")
# A ring wound the other way must still lift *off* its own plane, not into it.
check(float(np.dot(overlay.lift(_sq[::-1], 0.25)[0] - np.asarray(_sq[::-1][0]),
                   overlay.face_normal(_sq[::-1]))) > 0,
      "reversed winding lifts along its own normal too")

_a1 = overlay.lift_amount(obj)
check(_a1 > 0.0, f"lift_amount is positive for a real object ({_a1:.6g})")
# It comes from bound_box, which is local and excludes object scale - the
# overlay draws under matrix_world, so the offset must scale with the geometry
# rather than being fixed in world units.
_names = overlay.lift_amount.__code__.co_names
check("bound_box" in _names and "dimensions" not in _names,
      f"lift_amount measures the local box, not world dimensions ({_names})")
check("lift" in overlay.rebuild.__code__.co_names,
      "rebuild lifts the tint it caches")

# ------------------------------------------------ hiding the tint keeps the marks
# Turning the annotation off used to be all or nothing, and it lived in a
# closed sub-panel next to the output directory. The forced seam marks are the
# reason to have the overlay up while marking, so they switch separately from
# the colour that makes the mesh hard to read.
check(hasattr(bpy.context.scene.cbu, "overlay_fill"),
      "there is a separate switch for the face tint")
check(bpy.context.scene.cbu.overlay_fill is True,
      "and it is on by default, so nothing changes unasked")
_dn = overlay._draw.__code__.co_names
check("show_overlay" in _dn, "the 3D draw honours show_overlay")
check("overlay_fill" in _dn, "and honours overlay_fill separately")
_ui = (ROOT / "cbu_blender" / "ui.py").read_bytes().decode()
_main = _ui.split("class CBU_PT_main")[1].split(chr(10) + "class ")[0]
check("show_overlay" in _main and "overlay_fill" in _main,
      "both switches are on the main panel, not buried in a closed one")

# Sampling live UVs needs bmesh, and bmesh.from_edit_mesh inside a draw handler
# can allocate or refresh BMesh customdata - freeing blocks the draw is still
# walking. Dragging a UV island crashed Blender in BLI_mempool_free under
# BM_data_layer_add under cb_region_draw. A GPU crash cannot be caught at
# runtime, so the separation is asserted on the code: the draw path reads a
# cache that a timer fills.
_draw_names = overlay._draw_bed.__code__.co_names
check("_live_uv" not in _draw_names and "live_cut_rings" not in _draw_names,
      f"the draw handler does not sample live UVs (names: {_draw_names})")
_batch_names = overlay._cut_batch.__code__.co_names
check("live_cut_rings" not in _batch_names and "_live_uv" not in _batch_names,
      "the cut batch reads the cache rather than sampling")
check("_live_rings" in _batch_names, "and the cache is where it reads from")
check("_live_uv" in overlay.refresh_live_cuts.__code__.co_names
      or "live_cut_rings" in overlay.refresh_live_cuts.__code__.co_names,
      "refresh_live_cuts is what samples")

# No timer. Reading live Edit Mode UVs needs bmesh, and bmesh.from_edit_mesh is
# only safe at defined moments: from a draw handler it freed customdata under
# the draw, from a timer it fired mid-transform and crashed again. The
# depsgraph reports once a transform has finished, which is safe.
_src = (ROOT / "cbu_blender" / "overlay.py").read_bytes().decode()
check("bpy.app.timers" not in _src,
      "the overlay does not poll on a timer")
check("refresh_live_cuts" in cbu_blender._on_depsgraph.__code__.co_names,
      "the depsgraph handler is what drives the refresh")

# The overlay must never touch bmesh. Reading live Edit Mode UVs needs it, and
# it crashed Blender from every context tried - draw handler (customdata freed
# under the draw), timer (mid modal transform), and depsgraph_update_post
# (access violation in tbbmalloc). There is no fourth context, so the outline
# simply is not sampled in Edit Mode. A GPU/allocator crash cannot be caught at
# runtime, so this is asserted on the source.
check("import bmesh" not in _src, "the overlay does not import bmesh")
# On the code objects, not the source: a docstring explaining why bmesh is
# banned mentions from_edit_mesh, and a substring search fails on its own
# documentation. Same trap as the self.report check on the reload operator.
import types as _types
_reaches = [n for n, f in vars(overlay).items()
            if isinstance(f, _types.FunctionType)
            and "from_edit_mesh" in f.__code__.co_names]
check(not _reaches, f"no overlay function reads the edit-mode bmesh ({_reaches})")
check(overlay._live_uv(obj, [0]) is None or bpy.context.mode != "EDIT_MESH",
      "sampling returns nothing in Edit Mode rather than reaching for bmesh")

# it has to go stale with everything else, or it paints a net that has moved
obj.data.vertices[0].co.x += 0.02
obj.data.update()
cbu_blender._on_depsgraph(bpy.context.scene, bpy.context.evaluated_depsgraph_get())
check(obj.name not in overlay._net_cuts,
      "the cut outline is dropped when the mesh changes")

# ------------------------------------------------ bed-aware UV nesting
obj = fresh("Nest", "ico_sphere", subdivisions=3)
obj.scale = (0.14,) * 3
bpy.ops.object.transform_apply(scale=True)
bpy.context.scene.cbu.bed_x = 300.0
bpy.context.scene.cbu.bed_y = 300.0
bpy.ops.cbu.unfold()
_mat = props.material_from(bpy.context.scene.cbu)
_before_sheets = bpy.context.scene.cbu.last_sheets

# CANCELLED is a normal answer: the nester declines any layout needing more
# sheets than the one already in hand, and since tighten the core packing is
# often already as good as it gets. The invariant is that nesting never makes
# things worse, not that it always takes over.
_res = bpy.ops.cbu.pack_uv()
check(_res in ({"FINISHED"}, {"CANCELLED"}),
      f"bed-aware UV nesting returns cleanly ({_res})")
check(bpy.context.scene.cbu.last_sheets <= _before_sheets,
      "nesting never takes a layout needing more sheets "
      f"({bpy.context.scene.cbu.last_sheets} vs {_before_sheets})")
_pm, _isl, _parts, _sheets, _rep = ops._LAST[obj.name]

# Every part inside its own sheet. Nesting works on island outlines, but a
# glue tab reaches past them - place by island and the tab hangs over the
# margin, which is why final placement is decided by the parts.
_out = []
for p in _parts:
    x0, y0, x1, y1 = p.bounds
    if (x0 < _mat.sheet_margin - 1e-3 or y0 < _mat.sheet_margin - 1e-3
            or x1 > _mat.bed[0] - _mat.sheet_margin + 1e-3
            or y1 > _mat.bed[1] - _mat.sheet_margin + 1e-3):
        _out.append(p.island_id)
check(not _out, f"every nested part stays inside its sheet (out: {_out})")

# The check a bounding box cannot make. pack_islands, given a container it
# cannot fit, crams islands in overlapping rather than failing.
_ov = 0
_by_sheet = {}
for p in _parts:
    _by_sheet.setdefault(p.sheet, []).append(p)
for _grp in _by_sheet.values():
    for _a in range(len(_grp)):
        for _b in range(_a + 1, len(_grp)):
            if _grp[_a].cut.intersection(_grp[_b].cut).area > 1e-6:
                _ov += 1
check(_ov == 0, f"no two nested parts overlap ({_ov} pair(s))")
check(_sheets <= _before_sheets,
      f"nesting never needs more sheets ({_before_sheets} -> {_sheets})")

# the 2D view is re-emitted from the parts, so it cannot drift from the export
_sc = bridge.uv_scale(*_mat.bed)
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
_xf, _res = bridge.net_transforms(obj.data, _pm, _isl, _sc)
check(_res < 0.01, f"the net is still a rigid image of the unfold ({_res:.1e} mm)")

# ------------------------------------------------ 2D seam editing
obj = fresh("Seams2D", "ico_sphere", subdivisions=1)
obj.scale = (0.14,) * 3
bpy.ops.object.transform_apply(scale=True)
bpy.ops.cbu.unfold()


def _island_of_poly():
    pm, islands, _p, _s, _r = ops._LAST[obj.name]
    out = {}
    for isl in islands:
        for f in isl.faces:
            for s in pm.source_faces[f]:
                out[s] = isl.id
    return out


def _select_only(eidx):
    if bpy.context.mode != "EDIT_MESH":
        bpy.ops.object.mode_set(mode="EDIT")
    _bm = bmesh.from_edit_mesh(obj.data)
    _bm.edges.ensure_lookup_table()
    for x in _bm.edges:
        x.select_set(False)
    _bm.edges[eidx].select_set(True)
    bmesh.update_edit_mesh(obj.data)


bpy.ops.object.mode_set(mode="EDIT")
_bm = bmesh.from_edit_mesh(obj.data)
_bm.edges.ensure_lookup_table()
_pi = _island_of_poly()
_target = next(e.index for e in _bm.edges
               if len(e.link_faces) == 2
               and _pi.get(e.link_faces[0].index) == _pi.get(e.link_faces[1].index))
_faces = [f.index for f in _bm.edges[_target].link_faces]

def _edge_is_slit(eidx):
    """Do the two faces disagree about where this edge lives in the net?

    That - not the island count - is what a cut means. The dual graph has
    cycles, so removing one arc slits the net there while the island can stay
    in one piece through another path; asserting the island splits passes on a
    cube and fails on a sphere.
    """
    if bpy.context.mode == "EDIT_MESH":
        bpy.ops.object.mode_set(mode="OBJECT")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    lay = bm.loops.layers.uv[bridge.UV_LAYER]
    e = bm.edges[eidx]
    f1, f2 = e.link_faces

    def corners(f):
        return {l.vert.index: tuple(l[lay].uv) for l in f.loops if l.vert in e.verts}

    a, b = corners(f1), corners(f2)
    slit = any(np.linalg.norm(np.array(a[k]) - np.array(b[k])) > 1e-6 for k in a)
    bm.free()
    return slit


def _fold_edge():
    """An edge that is currently a fold inside one island - the only kind a
    separating cut can act on."""
    pm, islands, _p, _s, _r = ops._LAST[obj.name]
    core_of = {}
    for ci, grp in enumerate(pm.source_faces):
        for poly in grp:
            core_of[poly] = ci
    if bpy.context.mode != "EDIT_MESH":
        bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        a, b = (core_of.get(f.index) for f in e.link_faces)
        if a is None or b is None or a == b:
            continue
        host = next((i for i in islands if a in i.faces and b in i.faces), None)
        if host and any({fo.f0, fo.f1} == {a, b} for fo in host.folds):
            return e.index
    return None


# The two cut modes must actually differ. Re-unfolding lets the solver rebuild
# the whole tree, so the freed faces usually reattach to another island and the
# region count does not move - which is exactly the surprise this option
# exists to remove.
bpy.ops.cbu.clear_marks()
bpy.ops.cbu.unfold()
_base = bpy.context.scene.cbu.last_islands
_fe = _fold_edge()
check(_fe is not None, "found a fold edge to cut")
_select_only(_fe)
# Snapshot every island's geometry: a local edit must leave the rest of the net
# exactly where it was. That is the whole reason Cut no longer re-unfolds.
_before = {i.id: {k: v.copy() for k, v in i.faces.items()}
           for i in ops._LAST[obj.name][1]}
check(bpy.ops.cbu.edit_seam(mode="CUT") == {"FINISHED"}, "cut runs")
_detached = bpy.context.scene.cbu.last_islands
check(_detached == _base + 1,
      f"a cut adds exactly one island ({_base} -> {_detached})")

_moved = 0
for _i in ops._LAST[obj.name][1]:
    _old = _before.get(_i.id)
    if _old is None:
        continue
    for _f, _ring in _i.faces.items():
        if _f in _old and not np.allclose(_ring, _old[_f], atol=1e-9):
            _moved += 1
check(_moved == 0, f"cutting moved {_moved} face(s) that it should not have")

_pm, _isl, _parts, _sheets, _rep = ops._LAST[obj.name]
check(len({i.id for i in _isl}) == len(_isl), "island ids stay unique after a split")
# the spent fold must become a matched pair of cuts, or the glue tab on one
# side has no partner on the other
_pairs = {}
for i in _isl:
    for c in i.cuts:
        if c.seam >= 0:
            _pairs.setdefault(c.seam, 0)
            _pairs[c.seam] += 1
check(all(n == 2 for n in _pairs.values()),
      f"every seam id appears exactly twice ({sorted(_pairs.values())[:6]})")
check(sum(len(i.faces) for i in _isl) == len(_pm.faces),
      "every face still belongs to exactly one island")

bpy.ops.cbu.clear_marks()
bpy.ops.cbu.unfold()
_select_only(_target)
check(bpy.ops.cbu.edit_seam(mode="CUT") == {"FINISHED"},
      "cut a seam from the 2D view")
check(_edge_is_slit(_target), "cutting an edge slits the net along it")
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
_cuts, _folds = bridge.read_constraints(obj.data)
check(len(_cuts) == 1 and not _folds, "the cut is recorded as a forced cut")

_select_only(_target)
check(bpy.ops.cbu.edit_seam(mode="FOLD") == {"FINISHED"}, "weld a seam from the 2D view")
_after = _island_of_poly()
check(_after.get(_faces[0]) == _after.get(_faces[1]),
      "welding the same edge keeps those faces in one island")
check(not _edge_is_slit(_target), "and closes the slit in the net")
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
_cuts, _folds = bridge.read_constraints(obj.data)
check(not _cuts and len(_folds) == 1, "the weld replaced the cut, not stacked on it")

# Weld is the inverse and is equally local: cut an edge, weld it back, and the
# net must be exactly where it started. Re-unfolding would also give a valid
# answer - a different one - which is why this is measured rather than assumed.
bpy.ops.cbu.clear_marks()
bpy.ops.cbu.unfold()
_fe2 = _fold_edge()
_pre = {i.id: {k: v.copy() for k, v in i.faces.items()}
        for i in ops._LAST[obj.name][1]}
_pre_n = bpy.context.scene.cbu.last_islands
_select_only(_fe2)
check(bpy.ops.cbu.edit_seam(mode="CUT") == {"FINISHED"}, "cut, to weld back")
check(bpy.context.scene.cbu.last_islands == _pre_n + 1, "the cut took")
_select_only(_fe2)
check(bpy.ops.cbu.edit_seam(mode="FOLD") == {"FINISHED"}, "weld it back")
check(bpy.context.scene.cbu.last_islands == _pre_n,
      f"welding removes the island the cut added "
      f"({bpy.context.scene.cbu.last_islands} vs {_pre_n})")
_worst = 0.0
_unused = {i.id: i.faces for i in ops._LAST[obj.name][1]}
for _id, _pre_faces in _pre.items():
    for _f, _ring in _pre_faces.items():
        for _i2 in ops._LAST[obj.name][1]:
            if _f in _i2.faces:
                _worst = max(_worst, float(np.abs(_i2.faces[_f] - _ring).max()))
                break
check(_worst < 1e-9,
      f"cut then weld puts the net back exactly (worst {_worst:.2e} mm)")

# re-nesting on demand must not disturb the islands, only where they sit
_before_islands = bpy.context.scene.cbu.last_islands
check(bpy.ops.cbu.repack() == {"FINISHED"}, "re-pack runs on demand")
check(bpy.context.scene.cbu.last_islands == _before_islands,
      "re-packing does not re-unfold")
_pm, _isl, _parts, _sheets, _rep = ops._LAST[obj.name]
_mat = props.material_from(bpy.context.scene.cbu)
_stray = [p.island_id for p in _parts if p.sheet >= 0
          and (p.bounds[0] < _mat.sheet_margin - 1e-6
               or p.bounds[1] < _mat.sheet_margin - 1e-6
               or p.bounds[2] > _mat.bed[0] - _mat.sheet_margin + 1e-6
               or p.bounds[3] > _mat.bed[1] - _mat.sheet_margin + 1e-6)]
check(not _stray, f"re-packed parts stay inside the margins (stray: {_stray})")

# repacking twice must be idempotent - Part.xform maps island space to the
# sheet, so packing an already-packed part would compose two placements
bpy.ops.cbu.repack()
_s1 = bpy.context.scene.cbu.last_sheets
bpy.ops.cbu.repack()
check(bpy.context.scene.cbu.last_sheets == _s1, "re-packing twice is stable")

# Packing effort: what Re-pack (and so Refresh) packs with. Read on the code
# object, then exercised once at a small effort so it is not just wired.
check("pack_effort" in ops.CBU_OT_repack.execute.__code__.co_names,
      "Re-pack packs with the packing effort setting")
check("pack_effort" in ui.CBU_PT_uv.draw.__code__.co_consts
      and "pack_effort" in ui.CBU_PT_main.draw.__code__.co_consts,
      "the effort slider is beside Re-pack and beside Refresh")
_scr0 = bpy.context.scene.cbu
check(_scr0.pack_effort == 0.0, "packing effort defaults to 0 (fast)")
_scr0.pack_effort = 0.1
check(bpy.ops.cbu.repack() == {"FINISHED"}, "re-pack runs at a higher effort")
_scr0.pack_effort = 0.0

# Refresh: a parts-only setting applies to the net in hand, with no new solve
# - the user asked for it so a change like the corner hole would not mean
# re-running Optimise. A setting the SOLVE read is named, not silently
# ignored, because Refresh cannot apply it.
_scr = bpy.context.scene.cbu
_isl0 = ops._LAST[obj.name][1]
_cuts0 = [p.cut.wkb for p in ops._LAST[obj.name][2]]
_scr.use_fingers = True
check(bpy.ops.cbu.refresh() == {"FINISHED"}, "refresh runs after an unfold")
check(ops._LAST[obj.name][1] is _isl0, "refresh keeps the same net (no re-unfold)")
check([p.cut.wkb for p in ops._LAST[obj.name][2]] != _cuts0,
      "refresh applies a parts-only setting (fingers)")
check(ops.unfold_settings_changed(obj, _scr) == [],
      "a parts-only setting does not count as needing an unfold")
_bx = _scr.bed_x
_scr.bed_x = _bx + 50.0
check(ops.unfold_settings_changed(obj, _scr) == ["bed_x"],
      f"a solve setting is named as needing an unfold "
      f"({ops.unfold_settings_changed(obj, _scr)})")
check(bpy.ops.cbu.refresh() == {"FINISHED"},
      "refresh still rebuilds the parts when the solve settings moved")
_scr.bed_x = _bx
_scr.use_fingers = False
bpy.ops.cbu.refresh()
check("cbu.refresh" in ui.CBU_PT_main.draw.__code__.co_consts,
      "the main panel offers Refresh")
# The Save board slider is Optimise's alone: Unfold ignores it, so it can
# only go stale after an Optimise, and the result says when it split smaller.
check(_scr.board_bias == 0.0, "Save board defaults to large islands (0)")
bpy.ops.cbu.unfold()
_scr.board_bias = 0.6
check("board_bias" not in ops.unfold_settings_changed(obj, _scr),
      "the slider does not stale a plain Unfold")
_tb = (_scr.trials, _scr.budget)
_scr.trials, _scr.budget = 8, 2
bpy.ops.cbu.optimise()
# float32 in Blender: 0.6 is stored as 0.6000000238
check(abs(ops._SOLVED_WITH[obj.name].get("board_bias", -1) - 0.6) < 1e-6,
      "Optimise records the slider it ran with")
_scr.board_bias = 0.0
check(ops.unfold_settings_changed(obj, _scr) == ["board_bias"],
      "moving the slider after an Optimise asks for another")
check(" of bed" in _scr.last_detail or "candidate" in _scr.last_detail,
      f"the Optimise detail line is written ({_scr.last_detail!r})")
_scr.trials, _scr.budget = _tb
bpy.ops.cbu.unfold()
_held = ops._LAST.pop(obj.name)
check(not bpy.ops.cbu.refresh.poll(), "nothing to refresh before an unfold")
ops._LAST[obj.name] = _held

# ------------------------------------------------ non-planar faces
# A bent quad has no flat image at all, so it must be triangulated in the MESH.
# Splitting it core-side instead leaves one Blender polygon claimed by two core
# faces with two different placements; write_net_uv writes the whole polygon
# through source_faces, the second write wins, and the quad tears away from its
# neighbours by up to 11 mm. source_faces must stay a 1:1 partition.
obj = fresh("Bent", "plane")
bpy.ops.object.mode_set(mode="OBJECT")
obj.data.vertices[0].co.z += 0.35          # pull one corner out of plane
_pm = bridge.polymesh_from_object(obj, merge_coplanar=True)
check(_pm.planarity().max() > 1e-3, "the test face really is non-planar")
_claims = [s for grp in _pm.source_faces for s in grp]
check(len(_claims) == len(set(_claims)),
      "source_faces is a 1:1 partition of the caller's polygons")

_before = len(obj.data.polygons)
check(bpy.ops.cbu.split_nonplanar() == {"FINISHED"}, "split operator runs")
check(len(obj.data.polygons) > _before,
      f"the bent face was triangulated ({_before} -> {len(obj.data.polygons)})")
_pm2 = bridge.polymesh_from_object(obj, merge_coplanar=True)
check(_pm2.planarity().max() <= 1e-3, "nothing bent survives the split")

# a flat mesh must be left completely alone - no fold line down every panel
obj = fresh("Flat", "cube")
_n = len(obj.data.polygons)
bpy.ops.cbu.split_nonplanar()
check(len(obj.data.polygons) == _n,
      f"a flat mesh is untouched ({_n} -> {len(obj.data.polygons)})")

# ------------------------------------------------ UV islands match the core
# Blender decides UV island membership by whether the loops across an edge
# carry the SAME uv, compared as float32. Two faces of one island reach a
# shared corner down different tree paths and land a few ulps apart, so a
# 6e-14 mm disagreement used to read as a seam - and pack_islands then walked
# the halves off in different directions. A two-piece net arrived as six.
def _uv_island_count(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    lay = bm.loops.layers.uv[bridge.UV_LAYER]
    par = {f.index: f.index for f in bm.faces}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    hairline = 0
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        f1, f2 = e.link_faces

        def corners(f):
            return {l.vert.index: np.array(l[lay].uv[:], dtype=np.float32)
                    for l in f.loops if l.vert in e.verts}

        a, b = corners(f1), corners(f2)
        if set(a) != set(b):
            continue
        if all(np.array_equal(a[k], b[k]) for k in a):
            ra, rb = find(f1.index), find(f2.index)
            if ra != rb:
                par[ra] = rb
        elif max(float(np.abs(a[k] - b[k]).max()) for k in a) < 1e-5:
            hairline += 1          # same corner, disagreeing in the last bits
    n = len({find(f.index) for f in bm.faces})
    bm.free()
    return n, hairline


for _prim, _kw in (("ico_sphere", {"subdivisions": 2}), ("torus", {}),
                   ("cylinder", {"vertices": 16})):
    obj = fresh(f"UV_{_prim}", _prim, **_kw)
    obj.scale = (0.14,) * 3
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.cbu.unfold()
    _core = bpy.context.scene.cbu.last_islands
    _n, _hair = _uv_island_count(obj.data)
    check(_hair == 0, f"{_prim}: no corner disagrees in the last bits ({_hair})")
    check(_n == _core,
          f"{_prim}: Blender sees the same island count as the core "
          f"({_n} vs {_core})")
    bpy.ops.cbu.pack_uv()
    _n2, _ = _uv_island_count(obj.data)
    check(_n2 == _core,
          f"{_prim}: packing does not fragment the net ({_n2} vs {_core})")

# ------------------------------------------ stale overlay after a mesh edit
obj = fresh("Staleness", "ico_sphere", subdivisions=1)
obj.scale = (0.14,)*3
bpy.ops.object.transform_apply(scale=True)
bpy.ops.cbu.unfold()
check(obj.name in ops._LAST, "unfold cached a result to export")
check(obj.name in overlay._fingerprints, "unfold fingerprinted the geometry")
check(not overlay.geometry_changed(obj), "an untouched mesh reads as unchanged")

# ------------------------------------------ Opacity is a view setting
# It used to be baked into the vertex colours at build time and never read
# again, so the slider did nothing until the next unfold - and CBU_OT_unfold
# did not pass the setting at all, so in practice it was welded to 0.55. The
# island index is cached instead of the colour, so changing opacity re-colours
# geometry already in hand.
#
# There is no GPU in background mode, so overlay._batches is always empty here
# (available() is False and rebuild returns early). The colour rule and the
# re-colour bookkeeping are therefore tested directly, and the wiring is
# asserted on the code objects.
check(not overlay.available(), "background mode has no GPU, as assumed below")

check(overlay.island_color(3, 0.25)[3] == 0.25,
      "island_color honours the alpha it is handed")
_cols = overlay.island_colors([0, 1, 0, 2], 0.25)
check(len(_cols) == 4 and all(c[3] == 0.25 for c in _cols),
      "island_colors gives every vertex the requested opacity")
check(_cols[0] == _cols[2] and _cols[0] != _cols[1],
      "and keeps one colour per island, not per vertex")
check(overlay.island_colors([0, 1], 0.9)[0][:3]
      == overlay.island_colors([0, 1], 0.2)[0][:3],
      "opacity changes the alpha only - an island keeps its hue")

# set_alpha re-colours what is cached. Stand in for the batch the GPU would
# have built, since there is none here.
overlay._batches["__probe__"] = {"pos": [], "ids": [0, 1], "alpha": 0.55,
                                 "tris": None, "lines": None}
overlay.set_alpha(0.2)
check(overlay._batches["__probe__"]["alpha"] == 0.2,
      "set_alpha re-colours a cached annotation")
overlay._batches["__probe__"]["tris"] = "sentinel"
overlay.set_alpha(0.2)
check(overlay._batches["__probe__"]["tris"] == "sentinel",
      "an unchanged opacity rebuilds nothing - a dragged slider fires "
      "continuously")
overlay.set_alpha(0.9)
check(overlay._batches["__probe__"]["tris"] != "sentinel",
      "a changed one does rebuild")
del overlay._batches["__probe__"]

# Re-colouring must stay a pure cache operation: no mesh read, nothing that is
# unsafe to do from a property update while the user is dragging.
_sa = overlay.set_alpha.__code__.co_names
check("rebuild" not in _sa and "data" not in _sa,
      f"set_alpha does not touch the mesh ({_sa})")
check("overlay_alpha" in props.restyle_overlay.__code__.co_names
      and "set_alpha" in props.restyle_overlay.__code__.co_names,
      "the slider's update callback is what calls it")

# What rebuild actually caches can only be seen with a shader in hand, so
# stand one in. This is the half headless mode cannot reach: that the island
# index and the opacity are stored at all, which is what makes set_alpha work.
_real_sh, _real_batch = overlay._shaders, overlay.batch_for_shader
overlay._shaders = lambda: ("tri", "line")
overlay.batch_for_shader = lambda sh, kind, data: (kind, len(data["pos"]))
try:
    _pm, _isl = ops._LAST[obj.name][0], ops._LAST[obj.name][1]
    overlay.rebuild(obj, _pm, _isl, 0.3)
    _b = overlay._batches[obj.name]
    check(_b["alpha"] == 0.3, "rebuild records the opacity it built at")
    check(len(_b["pos"]) > 0 and len(_b["ids"]) == len(_b["pos"]),
          f"and one island index per vertex ({len(_b['ids'])} of {len(_b['pos'])})")
    check(set(_b["ids"]) == {i.id for i in _isl},
          "covering every island, so nothing is left uncoloured")
    overlay.set_alpha(0.8)
    check(overlay._batches[obj.name]["alpha"] == 0.8,
          "and set_alpha re-colours that cache in place")
finally:
    overlay._shaders, overlay.batch_for_shader = _real_sh, _real_batch
    overlay._batches.pop(obj.name, None)

# Both rebuild call sites must pass the setting; one of them silently did not.
# Unfold's is in `apply` rather than `execute` since the solve was split out
# onto a worker thread - the rule is about the call site, not the method name.
for _fn in (ops.CBU_OT_unfold.apply, ops.CBU_OT_edit_seam._local):
    check("overlay_alpha" in _fn.__code__.co_names,
          f"{_fn.__qualname__} passes the opacity setting through")

# Marking a seam writes an edge attribute. The depsgraph calls that a geometry
# update, but it must NOT count as modifying the object - otherwise the island
# colours vanish at the exact moment you are marking against them.
bpy.ops.object.mode_set(mode="EDIT")
_bm = bmesh.from_edit_mesh(obj.data)
_bm.edges.ensure_lookup_table()
for _e in _bm.edges:
    _e.select_set(False)
_bm.edges[0].select_set(True)
bmesh.update_edit_mesh(obj.data)
bpy.ops.cbu.mark(mode="CUT")
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
check(not overlay.geometry_changed(obj),
      "marking a seam does NOT count as modifying the object")
check(obj.name in ops._LAST, "and does not throw away the cached unfold")

# Moving a vertex does.
obj.data.vertices[0].co.x += 0.01
obj.data.update()
check(overlay.geometry_changed(obj), "moving a vertex counts as modifying it")

# so does adding or removing geometry
_before = overlay.geometry_fingerprint(obj.data)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.subdivide(number_cuts=1)
bpy.ops.object.mode_set(mode="OBJECT")
check(overlay.geometry_fingerprint(obj.data) != _before,
      "subdividing changes the fingerprint")

# and the handler wired to the depsgraph drops both caches
cbu_blender._on_depsgraph(bpy.context.scene, bpy.context.evaluated_depsgraph_get())
check(obj.name not in ops._LAST,
      "the depsgraph handler drops the cached unfold after a real edit")
check(obj.name not in overlay._fingerprints,
      "and the overlay annotation with it")

# ----------------------------------------------- constraints and locks
# The optimiser's answer IS its cut set, so write_constraints is how that
# answer becomes something the user can see, edit and re-unfold against. Two
# things have to hold or the promise is empty: the written set must replay the
# same net, and a locked edge must survive being written over.
obj = fresh("Locks", "ico_sphere", subdivisions=1)
obj.scale = (0.12,) * 3
bpy.ops.object.transform_apply(scale=True)
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.cbu.clear_marks()
bpy.ops.cbu.clear_locks()
bpy.ops.cbu.unfold()
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")

_pm, _isl, _pa, _sh, _rep = ops._LAST[obj.name]
_want = set(_rep["cut_arcs"])
check(bool(_want), "the net has cuts to write")
_written, _locked = bridge.write_constraints(obj.data, _want)
check(_locked == 0, f"nothing is locked yet (got {_locked})")
_cuts, _folds = bridge.read_constraints(obj.data)
check(_cuts == _want,
      f"the cut set round-trips ({len(_cuts)} read vs {len(_want)} written)")
check(not _folds, "and nothing was marked as a fold")

# ...and replaying it gives the same net back, which is the whole point.
_before = sorted(sorted(i.faces) for i in _isl)
bpy.ops.cbu.unfold()
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
_after = sorted(sorted(i.faces) for i in ops._LAST[obj.name][1])
check(_before == _after,
      f"re-unfolding against the written cuts rebuilds the same net "
      f"({len(_before)} islands vs {len(_after)})")

# A lock is the user telling the optimiser to keep its hands off ONE decision.
# Nothing else about it is special: it does not change what the mark means.
bpy.ops.cbu.clear_marks()
bpy.ops.object.mode_set(mode="EDIT")
_bm = bmesh.from_edit_mesh(obj.data)
_bm.edges.ensure_lookup_table()
for _e in _bm.edges:
    _e.select_set(False)
_keep = _bm.edges[0]
_keep.select_set(True)
_kv = tuple(sorted(v.index for v in _keep.verts))
bmesh.update_edit_mesh(obj.data)
check(bpy.ops.cbu.mark(mode="FOLD") == {"FINISHED"}, "mark the edge as a fold")
check(bpy.ops.cbu.lock(lock=True) == {"FINISHED"}, "lock it")
bpy.ops.object.mode_set(mode="OBJECT")

check(_kv in bridge.read_locks(obj.data), "the lock round-trips through the mesh")
_written, _locked = bridge.write_constraints(obj.data, _want)
check(_locked == 1, f"write_constraints sees the lock (got {_locked})")
_cuts, _folds = bridge.read_constraints(obj.data)
check(_kv in _folds and _kv not in _cuts,
      "a locked fold survives a cut set that asks for it to be cut")
check(_cuts == _want - {_kv},
      "and every unlocked edge took the value it was given")

bpy.ops.object.mode_set(mode="EDIT")
_bm = bmesh.from_edit_mesh(obj.data)
_bm.edges.ensure_lookup_table()
for _e in _bm.edges:
    _e.select_set(False)
_bm.edges[0].select_set(True)
bmesh.update_edit_mesh(obj.data)
check(bpy.ops.cbu.lock(lock=False) == {"FINISHED"}, "unlock it again")
bpy.ops.object.mode_set(mode="OBJECT")
check(not bridge.read_locks(obj.data), "and the lock is gone")

# Locked marks are drawn DASHED. Colour already says cut-or-fold and is the
# user's to choose, and line width is a per-draw uniform rather than a vertex
# attribute - so neither is free to carry this.
_d = overlay.dash((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 3)
check(len(_d) == 3, f"a dashed edge is 3 dashes (got {len(_d)})")
check(abs(sum(float(b[0] - a[0]) for a, b in _d) - 3.0 / 5.0) < 1e-12,
      "three equal dashes and two equal gaps cover 3/5 of the edge")
check(abs(float(_d[0][0][0])) < 1e-12 and abs(float(_d[-1][-1][0]) - 1.0) < 1e-12,
      "and a dash starts at each end, so the mark still shows where it stops")

_lk = overlay.locked_edges(obj.data)
check(len(_lk) == len(obj.data.edges) and not _lk.any(),
      "locked_edges reads all-False on an unlocked mesh")

bpy.ops.cbu.clear_marks()
bpy.ops.cbu.clear_locks()

# ------------------------------------------------- cancelling a long solve
# The optimiser mends every candidate before it can score one, so a run is tens
# of seconds and needs a way out. The solve moves to a worker thread; that is
# only legal because the core imports no bpy, so these checks are about that
# boundary rather than about the threading.
_names = ops.Solve.run.__code__.co_names
check("bpy" not in _names and "context" not in _names,
      f"Solve.run touches no Blender data (names: {_names})")
check("Exception" in _names and "error" in _names,
      "Solve.run cannot raise out of the worker thread")
_src = ops.CBU_OT_unfold.prepare.__doc__ or ""
check(hasattr(ops.CBU_OT_unfold, "prepare")
      and hasattr(ops.CBU_OT_unfold, "apply"),
      "the unfold operator is split around the solve")
check("progress" not in ops.CBU_OT_unfold.apply.__code__.co_names,
      "apply does not re-enter the solver")

# A script always takes execute(), never the modal path - which is what makes
# every other check in this file independent of a window existing.
check("prepare" in ops.CBU_OT_unfold.execute.__code__.co_names
      and "apply" in ops.CBU_OT_unfold.execute.__code__.co_names,
      "execute runs the whole job synchronously")
check("modal_handler_add" in ops.CBU_OT_unfold.invoke.__code__.co_names,
      "invoke is the modal path")
check("execute" in ops.CBU_OT_unfold.invoke.__code__.co_names,
      "...and falls straight through to execute when the optimiser is off")

# Stopping is cooperative and must still answer with a net. Drive Solve
# directly: the modal loop needs a window, the contract does not.
obj = fresh("Cancel", "ico_sphere", subdivisions=1)
obj.scale = (0.12,) * 3
bpy.ops.object.transform_apply(scale=True)
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
_pm = bridge.polymesh_from_object(obj, merge_coplanar=True, mm_per_unit=1000.0)
_mat = props.material_from(bpy.context.scene.cbu, obj)
_U = ops._core()[0]
_job = ops.Solve(_U, _pm, dict(trials=30, seed=0, mat=_mat, budget=30))
_seen = []
_real_progress = _job.progress


def _stop_at_two(done, total, note):
    _seen.append(done)
    if done >= 2:
        _job.stop()
    return _real_progress(done, total, note)


_job.progress = _stop_at_two
_job.run()
check(_job.error is None, f"a cancelled solve does not fail ({_job.error})")
check(_job.islands, "a cancelled solve still answers with a net")
check(_job.stopping, "and it knows it was stopped")
check(len(_seen) <= 3, f"it stopped when told ({len(_seen)} candidates)")
check(sum(len(i.faces) for i in _job.islands) == len(_pm.faces),
      "no face went missing when the search was cut short")

# -------------------------------------------- progress, and saying it is done
# Parts and packing are the slow half on a big model and they used to run in
# apply, on the main thread, after the progress readout had gone - so the bar
# reached the end and Blender froze at exactly the moment it looked finished.
_run = ops.Solve.run.__code__.co_names
_app = ops.CBU_OT_unfold.apply.__code__.co_names
check("build_parts" in _run and "pack" in _run,
      "the worker thread builds and packs the parts")
check("build_parts" not in _app and "pack" not in _app,
      f"apply does no packing on the main thread (names: {_app})")

# With the bed limit prepare() would pass - without it nothing is bed-split,
# and the whole net comes back as one part too big for any sheet.
_pad = 2 * _mat.tab_depth + 2 * _mat.kerf
_lim = (_mat.usable_bed[0] - _pad, _mat.usable_bed[1] - _pad)
_job = ops.Solve(_U, _pm, dict(trials=6, seed=0, mat=_mat, budget=2,
                               max_size=_lim),
                 L=ops._core()[1], mat=_mat)
_job.run()
check(_job.error is None, f"a packing solve does not fail ({_job.error})")
check(_job.parts and _job.sheets and _job.sheets >= 1,
      "the job comes back with parts and a sheet count")
check(_job.finished is not None and _job.elapsed > 0.0,
      "and knows how long it took")
check(0.0 <= _job.fraction <= 1.0, f"fraction stays in range ({_job.fraction})")

# One solve at a time: a second click must not start another on top.
_prev = ops.RUNNING
ops.RUNNING = _job
try:
    check(not ops.CBU_OT_unfold.poll(bpy.context),
          "Unfold is unavailable while a solve is running")
finally:
    ops.RUNNING = _prev
check(ops.CBU_OT_unfold.poll(bpy.context),
      "...and available again once it is done")

# The bar goes where the user is looking, and is repainted as it moves.
check("draw_progress" in ui.CBU_PT_main.draw.__code__.co_names,
      "the main panel draws the progress bar")
check("_redraw_panels" in ops.CBU_OT_unfold.modal.__code__.co_names,
      "the modal loop repaints the panels")
check("_announce" in ops.CBU_OT_unfold.modal.__code__.co_names,
      "and says so when it has finished")
for _name in ("last_optimised", "last_elapsed", "last_detail"):
    check(hasattr(bpy.context.scene.cbu, _name), f"scene remembers {_name}")

# ------------------------------------------------ Unfold stays the fast one
# Optimise used to be a scene toggle that changed what Unfold did. A scene
# property is saved with the file, so switching it on once made every later
# Unfold - in that session and in the saved .blend - run the slow path.
check(not hasattr(bpy.context.scene.cbu, "optimise"),
      "there is no stored optimise switch to leave on by accident")
check(ops.CBU_OT_unfold.optimising is False
      and ops.CBU_OT_optimise.optimising is True,
      "Unfold is plain and Optimise is the optimiser, by class")

obj = fresh("TwoButtons", "ico_sphere", subdivisions=1)
obj.scale = (0.12,) * 3
bpy.ops.object.transform_apply(scale=True)
if bpy.context.mode == "EDIT_MESH":
    bpy.ops.object.mode_set(mode="OBJECT")
_sc = bpy.context.scene.cbu
_sc.trials = 8
_sc.budget = 2

check(bpy.ops.cbu.optimise() == {"FINISHED"}, "Optimise runs")
check(_sc.last_optimised and "candidate" in _sc.last_detail,
      f"...and says it optimised ({_sc.last_detail!r})")
_opt = ops._LAST[obj.name][4]
check(_opt.get("candidates") is not None, "the report carries the search")

# The point of the whole change: after an Optimise, Unfold is still plain.
check(bpy.ops.cbu.unfold() == {"FINISHED"}, "Unfold runs after an Optimise")
_plain = ops._LAST[obj.name][4]
check(not _sc.last_optimised and not _sc.last_detail,
      "Unfold after Optimise is a plain unfold, not a second optimise")
check(_plain.get("candidates") is None and _plain.get("cost") is None,
      "...and its report has no search in it")

# ------------------------------------------------------- cutting together
# Several objects, one solve: each unfolds on its own, ONE scale for all of
# them (so a border modelled to fit a rim still fits), and every part on one
# set of sheets. The saved list says what is cut; empty means the active
# object, as before.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0, 0, 0.05))
_A = bpy.context.active_object
_A.name = "BigBox"
bpy.ops.mesh.primitive_cube_add(size=0.05, location=(0.3, 0, 0.025))
_B = bpy.context.active_object
_B.name = "SmallBox"
_s = bpy.context.scene.cbu
_s.trials = 30
_s.target_height = 0.0
_s.model_id = ""

check(ops.job_objects(bpy.context)[0] == [_B],
      "with no list, the active object is what is cut")
_A.select_set(True)
_B.select_set(True)
check(bpy.ops.cbu.job_add() == {"FINISHED"} and len(_s.job) == 2,
      "Add Selected puts both in the list")
check(bpy.ops.cbu.job_add() == {"CANCELLED"} and len(_s.job) == 2,
      "...and adding them again does not duplicate them")
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = None
check(ops.CBU_OT_unfold.poll(bpy.context),
      "Unfold is available with a list and no active object")


def _pairs_clear(names):
    """The closest two parts on one sheet come, whichever object they are
    from - the point of packing them together is that they keep the gap."""
    parts = [p for n in names for p in ops._LAST[n][2]]
    worst = float("inf")
    for i, p in enumerate(parts):
        for q in parts[i + 1:]:
            if p.sheet == q.sheet and p.sheet >= 0:
                worst = min(worst, p.cut.distance(q.cut))
    return worst


check(bpy.ops.cbu.unfold() == {"FINISHED"}, "Unfold cuts the list")
check(_A.name in ops._LAST and _B.name in ops._LAST, "both objects unfolded")
check(ops._LAST[_A.name][3] == ops._LAST[_B.name][3] == _s.last_sheets,
      f"they share the sheets ({_s.last_sheets})")
check(set(ops.group_of(_A.name)) == {_A.name, _B.name}
      and ops.group_of(_B.name) == ops.group_of(_A.name),
      "each knows what it was packed with")
_w = _pairs_clear([_A.name, _B.name])
check(_w >= _s.part_gap - 1e-6,
      f"no two parts on a sheet are closer than the gap ({_w:.2f} mm)")
check(_s.last_sheets == 1, f"two small boxes share ONE sheet ({_s.last_sheets})")
_mA = {p.model for p in ops._LAST[_A.name][2]}
_mB = {p.model for p in ops._LAST[_B.name][2]}
check(_mA and _mB and _mA.isdisjoint(_mB),
      f"each object's parts carry its own id ({_mA} / {_mB})")
check("2 objects" in _s.last_report, f"the report says so ({_s.last_report!r})")
check(overlay.shown_with(_A.name) == (_A.name, _B.name),
      "the 2D view draws the other object's parts on the shared sheets")

# A typed id gets a letter per object: island numbers restart per object, so
# HELM-0 twice is exactly the ambiguity the id exists to remove.
_s.model_id = "JOB"
_ids = [ops.job_material(_s, o, [_A, _B]).model_id for o in (_A, _B)]
check(_ids == ["JOBA", "JOBB"], f"a typed id is lettered per object ({_ids})")
_s.model_id = ""

# One scale for all: Height is measured over everything together.
_s.target_height = 200.0                  # together they stand 100 mm tall
check(bpy.ops.cbu.unfold() == {"FINISHED"}, "Unfold with a shared height")
_eA = ops._LAST[_A.name][0].extents
_eB = ops._LAST[_B.name][0].extents
check(abs(_eA[2] - 200.0) < 1e-6 and abs(_eB[2] - 100.0) < 1e-6,
      f"one factor for both: 200 and 100 mm tall (got {_eA[2]:.3f}, "
      f"{_eB[2]:.3f}) - not each scaled to 200")

# Re-pack and Refresh take the whole group: packing one alone would put it
# on top of the other's parts.
check(bpy.ops.cbu.repack() == {"FINISHED"}, "Re-pack runs on the group")
check(_pairs_clear([_A.name, _B.name]) >= _s.part_gap - 1e-6,
      "and the group still keeps the gap")
check(ops._LAST[_A.name][3] == ops._LAST[_B.name][3],
      "and still shares one sheet count")
check(bpy.ops.cbu.refresh() == {"FINISHED"}, "Refresh runs on the group")
bpy.context.view_layer.objects.active = _A     # Nest Sheets needs one
check(bpy.ops.cbu.pack_uv() == {"CANCELLED"},
      "Nest Sheets declines a group rather than nesting half of it")

# Export writes everything on the sheets, both objects.
import json as _json
import tempfile as _tf
_out = Path(_tf.mkdtemp())
_s.out_dir = str(_out)
check(bpy.ops.cbu.export() == {"FINISHED"}, "Export the group")
_man = _json.loads((_out / "manifest.json").read_text())
check({p["model"] for p in _man["parts"]} == _mA | _mB
      and len(_man["parts"]) == len(ops._LAST[_A.name][2]) +
      len(ops._LAST[_B.name][2]),
      "the manifest lists every part of both objects, by model")

# --- the yardstick
bpy.context.view_layer.objects.active = _A
check(bpy.ops.cbu.add_yardstick() == {"FINISHED"}, "Add Yardstick")
_Y = _s.yardstick
check(_Y is not None and len(_Y.data.edges) == 1 and len(_Y.data.vertices) == 2,
      "the yardstick is one edge")
check(_s.size_mode == "YARDSTICK", "and sizes by it")
check(bpy.context.active_object == _Y, "and is active, ready to move")
check(ops.job_ref(bpy.context) == _A,
      "the panel still reports on the model, not on the stick")
# Adding it kept the size: the model would have been 200 mm tall.
_k = _s.yardstick_length / bridge.yardstick_mm(_Y, _s.mm_per_unit)
check(abs(_k - 2.0) < 1e-6, f"switching to it keeps the size (x{_k:.6f})")
# Put its ends on a known 50 mm and say it is really 150 mm: x3.
_Y.location = (0, 0, 0)
_Y.data.vertices[0].co = (0.0, -0.05, 0.0)
_Y.data.vertices[1].co = (0.0, 0.0, 0.0)
_Y.data.update()
_s.yardstick_length = 150.0
_ym = bridge.yardstick_mm(_Y, 1000.0)
# 1e-4, not 1e-6: vertex coordinates are float32, and -0.05 stored that way
# reads back as 50.0000018 mm.
check(abs(_ym - 50.0) < 1e-4,
      f"it measures the distance between its two vertices, in world mm "
      f"(got {_ym:.7f})")
check(bpy.ops.cbu.unfold() == {"FINISHED"}, "Unfold sized by the yardstick")
_eA = ops._LAST[_A.name][0].extents
check(abs(_eA.max() - 300.0) < 1e-4,
      f"a 100 mm box with a 50 -> 150 mm yardstick is 300 mm ({_eA.max():.4f})")
check(_Y.name not in ops._LAST, "the yardstick is never cut")
_s.job.add().obj = _Y
check(_Y not in ops.job_objects(bpy.context)[0],
      "...not even when it is in the list")
_s.job.remove(len(_s.job) - 1)
check(not ops.unfold_settings_changed(_A, _s), "nothing stale after Unfold")
_Y.data.vertices[0].co = (0.0, -0.06, 0.0)
_Y.data.update()
check(ops.unfold_settings_changed(_A, _s) == ["yardstick_mm"],
      f"moving the yardstick needs an Unfold "
      f"({ops.unfold_settings_changed(_A, _s)})")
check("Yardstick" in ops.setting_labels(_s, ["yardstick_mm"]),
      "and the warning names it the way the panel does")
_Y.data.vertices[0].co = (0.0, -0.05, 0.0)
_Y.data.update()
_s.yardstick = None
check(bpy.ops.cbu.unfold() == {"CANCELLED"},
      "Size by Yardstick with no yardstick refuses, rather than guessing")
_s.yardstick = _Y
_s.size_mode = "HEIGHT"
_s.target_height = 0.0

# --- a flat piece cut as it is
bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=5,
                                size=0.12, location=(0.6, 0, 0.0))
_F = bpy.context.active_object
_F.name = "FlatPlate"
_F.rotation_euler = (0.4, 0.2, 0.0)       # flat, but not in an axis plane
_s.job.add().obj = _F
_s.merge_coplanar = False                 # keep every grid cell a face
check(bpy.ops.cbu.unfold() == {"FINISHED"}, "Unfold with a flat plate in the list")
_s.merge_coplanar = True
_rep = ops._LAST[_F.name][4]
_isl = ops._LAST[_F.name][1]
check(_rep.get("flat_pieces") == 1 and len(_isl) == 1 and not _isl[0].folds,
      f"the plate is one flat piece with no folds ({_rep.get('flat_pieces')}, "
      f"{len(_isl)} island(s), {len(_isl[0].faces)} face(s))")
_pf = ops._LAST[_F.name][2][0]
check(_pf.perf_m.is_empty and _pf.perf_v.is_empty,
      "and nothing on it is scored")
check(abs(_pf.cut.area - 120.0 * 120.0) < 1e-3,
      f"at its real size ({_pf.cut.area:.3f} mm2)")
check("flat piece" in _s.last_report, f"the report counts it ({_s.last_report!r})")
check(_pairs_clear([_A.name, _B.name, _F.name]) >= _s.part_gap - 1e-6,
      "three objects, still no two parts closer than the gap")
_uvF = uv_edge_lengths(_F.data, bridge.UV_LAYER)
_errF = np.abs(_uvF[:, 0] - _uvF[:, 1]).max()
check(_errF < UV_TOL_MM, f"the plate's net is its real size ({_errF:.1e} mm)")

# --- leaving the list: what was packed together stops re-packing together
bpy.ops.cbu.job_clear()
bpy.context.view_layer.objects.active = _B
check(bpy.ops.cbu.unfold() == {"FINISHED"}, "Unfold one object after the list")
check(ops.group_of(_B.name) == (_B.name,), "it is packed alone now")
check(_B.name not in ops.group_of(_A.name),
      "and the others no longer drag it into their re-pack")
bpy.context.view_layer.objects.active = _Y
check(bpy.ops.cbu.unfold() == {"CANCELLED"},
      "with no list, the yardstick as the active object is refused")


# --- a listed object left in Edit Mode behind another active one. The ESD
# gun: Add Yardstick, run from Edit Mode, made the stick active and left the
# model in Edit Mode, where its mesh arrays read EMPTY - Optimise then died
# with "foreach_set(..) sequence length mismatch given 228, needed 0".
def _ran(op):
    try:
        return op() == {"FINISHED"}
    except RuntimeError as exc:
        print("   ", exc)
        return False


def _stuck_behind_stick():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = _A
    _A.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.view_layer.objects.active = _Y    # what Add Yardstick did
    return _A.mode == "EDIT" and bpy.context.mode == "OBJECT"


_s.job.add().obj = _A
_s.job.add().obj = _B
check(_stuck_behind_stick(), "setup: a listed model in Edit Mode, the stick active")
check(_ran(bpy.ops.cbu.unfold), "Unfold reads a listed model left in Edit Mode")
check(_A.mode == "OBJECT", "and leaves it in Object Mode")
check(_stuck_behind_stick(), "setup again")
check(_ran(bpy.ops.cbu.repack), "Re-pack writes to a listed model left in Edit Mode")
bpy.ops.cbu.job_clear()
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = _A
_A.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
_s.yardstick = None
check(_ran(bpy.ops.cbu.add_yardstick), "Add Yardstick from Edit Mode")
check(_A.mode == "OBJECT",
      f"does not leave the model behind it in Edit Mode ({_A.mode})")
bpy.data.objects.remove(_s.yardstick)
bpy.ops.cbu.job_clear()
_s.size_mode = "HEIGHT"
_s.target_height = 0.0
_s.yardstick = None

# ------------------------------------------------------- reload operator
# reload_all() unregisters CBU_OT_reload itself, so the operator must not
# touch `self` afterwards - that is a use-after-free on a freed RNA type.
# test_dev.py calls reload_all() directly and so never exercised this path.
# Assert on the compiled code object, not the source text: a comment
# mentioning self.report would fool a substring search either way round.
_names = dev.CBU_OT_reload.execute.__code__.co_names
check("report" not in _names,
      f"the reload operator never touches self.report (names: {_names})")
check("timers" in _names and "register" in _names,
      "the reload operator defers the reload off its own stack")
check(bpy.ops.cbu.reload() == {"FINISHED"}, "reload operator returns cleanly")

# -------------------------------------------------------------- teardown
cbu_blender.unregister()
check(not hasattr(bpy.types.Scene, "cbu"), "unregister cleans up")

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("all addon checks passed")
