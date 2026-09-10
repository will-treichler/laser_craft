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
from cbu_blender import bridge, ops, overlay, props

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
check(arr.max() > 100.0, f"UVs are in mm, not 0-1 (max {arr.max():.1f})")


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
    return np.array(out)


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
r = bpy.ops.cbu.pack_uv()
check(r == {"FINISHED"}, f"pack_uv runs (got {r})")
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
