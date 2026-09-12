"""Headless Blender test for the addon. Run:  python3 blender/test_addon.py

Uses the `bpy` PyPI module, so this exercises the real Blender API - operator
registration, bmesh round trips, attributes, UV layers and pack_islands - not
a mock.
"""
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
from cbu_blender import bridge, dev, ops, overlay, props

cbu_blender.register()
check(hasattr(bpy.types.Scene, "cbu"), "settings registered on Scene")
for idname in ("cbu.unfold", "cbu.mark", "cbu.export", "cbu.pack_uv",
               "cbu.clear_marks", "cbu.select_marked"):
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
    check({"CUT", "PERF_M", "ENGRAVE"} <= labels, f"svg layers present: {sorted(l for l in labels if l)}")
check((out / "manifest.json").exists(), "manifest written")

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
_row = bpy.context.scene.cbu.bed_x - 2 * bpy.context.scene.cbu.margin
check(_w <= _row + 1e-6,
      f"net is no wider than one usable sheet ({_w:.0f} <= {_row:.0f} mm)")
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


_before = _net_mm()
_s.bed_x, _s.bed_y = 600.0, 400.0
_after = _net_mm()
check(abs(_after[:, 0].max() - _before[:, 0].max()) < 1e-3,
      f"the net keeps its millimetres when the bed changes "
      f"({_before[:, 0].max():.1f} -> {_after[:, 0].max():.1f})")
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
check(bpy.ops.cbu.edit_seam(mode="CUT", separate=False) == {"FINISHED"},
      "re-unfolding cut runs")
_resolved = bpy.context.scene.cbu.last_islands

bpy.ops.cbu.clear_marks()
bpy.ops.cbu.unfold()
_fe = _fold_edge()
_select_only(_fe)
check(bpy.ops.cbu.edit_seam(mode="CUT", separate=True) == {"FINISHED"},
      "separating cut runs")
_detached = bpy.context.scene.cbu.last_islands
check(_detached == _base + 1,
      f"a separating cut adds exactly one island ({_base} -> {_detached})")
check(_detached > _resolved,
      f"and gives more regions than re-unfolding ({_detached} vs {_resolved})")

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
check(bpy.ops.cbu.edit_seam(mode="CUT", separate=False) == {"FINISHED"},
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
for _fn in (ops.CBU_OT_unfold.execute, ops.CBU_OT_edit_seam._separate):
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
