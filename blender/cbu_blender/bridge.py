"""bmesh <-> core translation, and where the addon keeps its state.

State lives in MESH ATTRIBUTES, not PropertyGroups. This is not a style
preference: Blender's edit-mode undo records mesh and bmesh changes only, so
custom properties set while in Edit Mode are NOT restored by Ctrl+Z. Seam
marks stored in a PropertyGroup would survive right up until the first undo
and then quietly desynchronise from the mesh. Attributes are part of the mesh
and undo correctly.

  edge  int   cbu_constraint   0 auto | 1 force cut | 2 force fold
  face  int   cbu_island       island index, -1 before unfolding
  loop  uv    CBU_Net          the flat net, in millimetres
"""
from __future__ import annotations

import math

import numpy as np
import bmesh
import bpy
from mathutils import Matrix

# The core is imported inside the functions that use it, not here. It pulls
# in shapely and networkx, which Blender does not ship, and an eager import
# makes register() raise ModuleNotFoundError - which kills the whole add-on
# before the panel that offers to install those packages can ever draw.
# PolyMesh appears in annotations only, and those stay strings under
# `from __future__ import annotations`.

EDGE_ATTR = "cbu_constraint"
FACE_ATTR = "cbu_island"
UV_LAYER = "CBU_Net"
# mm. Corners closer than this are treated as the same corner.
#
# Two jobs. It absorbs the few-ulp disagreement between two tree paths, and it
# absorbs the residual bow of a face that is flat enough to keep (split_nonplanar
# only splits past 1e-3 relative). A 66 mm edge bowed 0.02 mm produced a 0.056 mm
# disagreement on a real part - physically meaningless against a 0.3 mm kerf and
# +/-0.5 mm costume tolerance, but enough for Blender to call it a seam and for
# pack_islands to tear the net there. Splitting such a face instead would add a
# fold line to the finished part for no physical reason.
#
# Ceiling: it must stay well under the smallest gap between two genuinely
# different corners of one island, and well under the kerf.
WELD_TOL = 0.1

AUTO, FORCE_CUT, FORCE_FOLD = 0, 1, 2


# ------------------------------------------------------------- attributes

def edge_layer(bm, create: bool = True):
    lay = bm.edges.layers.int.get(EDGE_ATTR)
    if lay is None and create:
        lay = bm.edges.layers.int.new(EDGE_ATTR)
    return lay


def face_layer(bm, create: bool = True):
    lay = bm.faces.layers.int.get(FACE_ATTR)
    if lay is None and create:
        lay = bm.faces.layers.int.new(FACE_ATTR)
    return lay


def ensure_mesh_attrs(mesh) -> None:
    if EDGE_ATTR not in mesh.attributes:
        mesh.attributes.new(EDGE_ATTR, "INT", "EDGE")
    if FACE_ATTR not in mesh.attributes:
        mesh.attributes.new(FACE_ATTR, "INT", "FACE")


def uv_selected_edges(bm, uv_name: str = UV_LAYER) -> set:
    """Mesh edge indices whose UV edge is selected in the 2D view.

    UV selection lives per loop, not per edge, and with UV Sync Selection off
    it is completely separate from the mesh selection - so "the edge I picked
    in the net" cannot be read from `edge.select` at all. Both loops of a
    shared edge are checked, since only one side is selected when the edge is
    a seam and the two halves sit apart in the net.
    """
    lay = bm.loops.layers.uv.get(uv_name)
    if lay is None:
        return set()
    out = set()
    for f in bm.faces:
        for l in f.loops:
            if getattr(l[lay], "select_edge", False):
                out.add(l.edge.index)
    return out


def read_constraints(mesh) -> tuple[set, set]:
    """Forced cuts and folds as sets of sorted (v0, v1) vertex-index pairs."""
    cuts, folds = set(), set()
    attr = mesh.attributes.get(EDGE_ATTR)
    if attr is None:
        return cuts, folds
    vals = np.zeros(len(mesh.edges), dtype=np.int32)
    attr.data.foreach_get("value", vals)
    for i, v in enumerate(vals):
        if v == AUTO:
            continue
        a, b = mesh.edges[i].vertices
        key = (min(a, b), max(a, b))
        (cuts if v == FORCE_CUT else folds).add(key)
    return cuts, folds


# ------------------------------------------------------------------ read

def polymesh_from_object(obj, merge_coplanar: bool = True,
                         mm_per_unit: float = 1000.0,
                         apply_transform: bool = True,
                         tol_deg: float = 0.5):
    """Build a core PolyMesh from a Blender object.

    Blender faces are already ordered n-gon rings, so unlike the file loader
    there is no triangulate-then-remerge round trip - the mesh goes straight
    in. Coplanar merging is still offered because imported STL is triangle
    soup.
    """
    mesh = obj.data
    mesh.calc_loop_triangles()
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)

    if apply_transform:
        M = np.array(obj.matrix_world.to_4x4())
        verts = verts @ M[:3, :3].T + M[:3, 3]
    verts = verts * mm_per_unit

    from .vendor.cardboard_unfolder.mesh import PolyMesh

    rings = [[int(v) for v in p.vertices] for p in mesh.polygons]
    # planar_tol=None: do NOT let the core split bent faces here.
    #
    # source_faces must stay a 1:1 partition of the caller's polygons, because
    # write_net_uv writes whole polygons through it. Splitting a bent quad
    # core-side leaves that one quad claimed by two core faces with two
    # different flat placements; the second write wins and tears the quad away
    # from its neighbours by up to 11 mm. Blender's mesh still holds a quad,
    # and a bent quad simply has no consistent flat image - so the split has to
    # happen in the mesh itself, via cbu.split_nonplanar.
    pm = PolyMesh.from_rings(verts, rings, merge=merge_coplanar, tol_deg=tol_deg,
                             planar_tol=None)
    return pm


# ----------------------------------------------------------------- write

def write_islands(mesh, pm: PolyMesh, islands) -> None:
    """Tag every original Blender face with the island it ended up in."""
    ensure_mesh_attrs(mesh)
    vals = np.full(len(mesh.polygons), -1, dtype=np.int32)
    for isl in islands:
        for f in isl.faces:
            for src in pm.source_faces[f]:
                vals[src] = isl.id
    mesh.attributes[FACE_ATTR].data.foreach_set("value", vals)
    mesh.update()


def uv_scale(bed_w: float, bed_h: float) -> float:
    """mm -> UV factor: the largest power of two that fits a sheet in 0-1.

    The UV editor is built around the unit square and will not zoom out far
    enough to frame a net written in raw millimetres, so the net is scaled.
    The factor is a power of two on purpose: it is exact in binary floating
    point, so scaling is a pure exponent shift and every edge length keeps
    its mantissa bit for bit. A snug factor like 1/600 would round, and this
    unfolder's whole premise is that two faces which shared a 40.00 mm edge
    still share one.

    A 600 x 400 mm bed gives 1/1024, so a sheet draws 0.586 x 0.391.
    """
    big = max(bed_w, bed_h, 1e-9)
    return 2.0 ** -math.ceil(math.log2(big))


def _shelf_offsets(islands, row_width: float | None, gap: float) -> dict:
    """Lay the islands out in rows rather than one long strip.

    The old layout put every island on a single horizontal line, so a 38
    island net ran several metres wide and a few hundred millimetres tall.
    The UV editor could not zoom out far enough to frame that, and the 0-1
    grid was useless next to it. Rows wrap at row_width, which the caller
    sets to the usable bed width, so one row of the preview is one sheet
    wide and the bed reference lines up with it.
    """
    boxes = {i.id: i.bounds() for i in islands}
    if not row_width or row_width <= 0.0:       # no bed given: aim for a square
        total = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes.values())
        row_width = max(1.0, float(np.sqrt(max(total, 1.0))) * 1.3)

    # An island wider than a sheet cannot be sheet-aligned anyway - bed
    # splitting only cuts along fold edges, so a single oversized face stays
    # oversized. Widening the row to fit it keeps the rest of the net packed
    # in rows instead of collapsing into a one-island-per-row column.
    widest = max((x1 - x0 for x0, _, x1, _ in boxes.values()), default=0.0)
    row_width = max(row_width, widest)

    # Tallest first, the same shelf heuristic layout.pack uses, so the preview
    # rows resemble the sheets the exporter will actually produce.
    order = sorted(islands, key=lambda i: -(boxes[i.id][3] - boxes[i.id][1]))
    offsets, x, y, shelf_h = {}, 0.0, 0.0, 0.0
    for isl in order:
        x0, y0, x1, y1 = boxes[isl.id]
        w, h = x1 - x0, y1 - y0
        if x > 0.0 and x + w > row_width:       # next row
            x, y, shelf_h = 0.0, y + shelf_h + gap, 0.0
        offsets[isl.id] = (x - x0, y - y0)
        x += w + gap
        shelf_h = max(shelf_h, h)
    return offsets


def net_bounds(mesh):
    """(x0, y0, x1, y1) of the net UV layer in mm, or None if there isn't one."""
    uv = mesh.uv_layers.get(UV_LAYER)
    if uv is None or not len(mesh.loops):
        return None
    d = np.zeros(len(mesh.loops) * 2)
    uv.data.foreach_get("uv", d)
    d = d.reshape(-1, 2)
    return (float(d[:, 0].min()), float(d[:, 1].min()),
            float(d[:, 0].max()), float(d[:, 1].max()))


def write_net_uv(mesh, pm: PolyMesh, islands, spread: bool = True,
                 row_width: float | None = None, gap: float = 10.0,
                 scale: float = 1.0, placement: dict | None = None,
                 sheet_height: float = 0.0) -> str:
    """Write the flat net into a UV layer, in millimetres.

    An isometric unfold IS a UV parameterisation, and UVs are stored per
    loop - which is exactly the storage an unfold needs, because cuts
    duplicate vertices and a shared corner has two different flat positions.
    Putting the net here means the UV editor becomes the 2D view for free,
    including island selection and uv.pack_islands as a true-shape nester.
    """
    uv = mesh.uv_layers.get(UV_LAYER) or mesh.uv_layers.new(name=UV_LAYER)
    # Make the net the layer the UV editor shows. uv_layers.new() does not
    # do this, so on any mesh that already had a UVMap - every Blender
    # primitive and every imported model - the 2D view keeps displaying the
    # old map and the unfold looks like it silently failed. Only `active` is
    # touched, never `active_render`, so materials still texture as before.
    mesh.uv_layers.active = uv

    # islands are unfolded into a shared origin, so they need spreading out
    #
    # With `placement` the layout IS the packing: each island goes exactly
    # where its Part was placed on its sheet, and sheets stack upward. That is
    # what makes the bed reference mean something - before this the rows were
    # laid out independently of the sheets, so islands straddled the sheet
    # divisions and sat in the margins that pack() had carefully kept clear.
    offsets = None
    if placement is None:
        offsets = (_shelf_offsets(islands, row_width, gap) if spread
                   else {i.id: (0.0, 0.0) for i in islands})

    # pm.vertices is already in the core's mm space and is indexed by the
    # original Blender vertex indices, so it is the correct source here -
    # re-deriving it from mesh.vertices would drop the object transform.
    data = np.zeros((len(mesh.loops), 2), dtype=np.float64)
    loop_island = np.full(len(mesh.loops), -1, dtype=np.int64)
    spare = max((p.sheet for p in (placement or {}).values()), default=-1) + 1
    for isl in islands:
        part = (placement or {}).get(isl.id)
        if part is None:
            ox, oy = offsets[isl.id] if offsets else (0.0, 0.0)
        else:
            # a part too big for the bed never got a sheet; park it past the
            # last real one rather than stacking it on sheet 0
            row = part.sheet if part.sheet >= 0 else spare
            ox, oy = 0.0, row * sheet_height
        for f, pos2 in isl.faces.items():
            fn = pm.plane_map(f, pos2)
            for src in pm.source_faces[f]:
                poly = mesh.polygons[src]
                idx = list(poly.loop_indices)
                pts = fn(pm.vertices[list(poly.vertices)])
                if part is not None:
                    pts = part.place(pts)
                data[idx] = pts + (ox, oy)
                loop_island[idx] = isl.id

    # Weld each shared corner to a single value.
    #
    # Two faces of the same island reach a corner they share through different
    # tree paths, so they compute it independently and land a few ulps apart.
    # Blender decides UV island membership by whether the loops across an edge
    # carry the *same* uv, as float32 - so a 6e-14 mm disagreement reads as a
    # seam, and pack_islands then walks the two halves off in different
    # directions. That is how a two-piece net arrives as six.
    #
    # Welding by (island, vertex) rather than vertex alone is what keeps real
    # seams real: a vertex on a cut belongs to two islands and must keep two
    # different positions, which is the whole point of storing the net in a
    # per-loop layer.
    #
    # The position has to be part of the key too. One island can legitimately
    # visit the same 3D vertex at two different places in the plane - that is
    # a net wrapping round and rejoining itself, the near-pinch narrow_necks
    # looks for. Keying on (island, vertex) alone fuses those two places into
    # one and drags the geometry with it: 220 mm of isometry error, caught by
    # test_addon's isometry checks. Quantising to WELD_TOL separates them
    # while still collecting the few-ulp disagreements we are here to fix; the
    # canonical value is a real member of the cluster, not the rounded key, so
    # nothing moves by more than the cluster's own width.
    if len(data):
        nv = max(len(mesh.vertices), 1)
        vidx = np.empty(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", vidx)
        ident = (loop_island + 1) * (nv + 1) + vidx
        order = np.argsort(ident, kind="stable")

        # Cluster by actual distance, not by rounding to a grid. Snapping to a
        # WELD_TOL grid looked equivalent and is not: two corners 0.056 mm
        # apart still land either side of a 0.1 mm grid line better than half
        # the time, so the weld silently did nothing on exactly the parts that
        # needed it. Groups are tiny - the loops meeting at one vertex of one
        # island - so a representative scan inside each is cheap, and a group
        # whose values already agree is skipped outright.
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and ident[order[j + 1]] == ident[order[i]]:
                j += 1
            grp = order[i:j + 1]
            i = j + 1
            if len(grp) < 2:
                continue
            block = data[grp]
            if np.array_equal(block, np.broadcast_to(block[0], block.shape)):
                continue                      # already welded, nothing to do
            reps: list = []
            for li in grp:
                for r in reps:
                    if np.abs(data[li] - data[r]).max() <= WELD_TOL:
                        data[li] = data[r]
                        break
                else:
                    reps.append(li)           # a genuinely different corner

    if scale != 1.0:
        data *= scale           # exact: scale is a power of two
    uv.data.foreach_set("uv", data.ravel())
    mesh.update()
    return UV_LAYER
