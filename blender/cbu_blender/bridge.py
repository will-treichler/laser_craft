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
# Bool per edge: the optimiser must leave this edge's mark alone.
LOCK_ATTR = "cbu_lock"
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


def lock_layer(bm, create: bool = True):
    lay = bm.edges.layers.bool.get(LOCK_ATTR)
    if lay is None and create:
        lay = bm.edges.layers.bool.new(LOCK_ATTR)
    return lay


def ensure_mesh_attrs(mesh) -> None:
    if EDGE_ATTR not in mesh.attributes:
        mesh.attributes.new(EDGE_ATTR, "INT", "EDGE")
    if FACE_ATTR not in mesh.attributes:
        mesh.attributes.new(FACE_ATTR, "INT", "FACE")
    if LOCK_ATTR not in mesh.attributes:
        mesh.attributes.new(LOCK_ATTR, "BOOLEAN", "EDGE")


def edge_keys(mesh) -> np.ndarray:
    """(n, 2) sorted vertex pairs, one row per edge, in edge-index order."""
    v = np.zeros(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", v)
    return np.sort(v.reshape(-1, 2), axis=1)


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


def read_locks(mesh) -> set:
    """Edges the optimiser must not touch, as sorted (v0, v1) pairs."""
    attr = mesh.attributes.get(LOCK_ATTR)
    if attr is None:
        return set()
    vals = np.zeros(len(mesh.edges), dtype=bool)
    attr.data.foreach_get("value", vals)
    keys = edge_keys(mesh)
    return {(int(a), int(b)) for a, b in keys[vals]}


def write_constraints(mesh, cuts, folds=()) -> tuple[int, int]:
    """Replace the unlocked marks with this cut set. Returns (written, locked).

    A net IS its cut set, so this is how an optimiser's answer survives into
    something the user can see, edit and re-unfold against. It overwrites
    unlocked marks by design - including clearing marks the optimiser did not
    ask for, because a stale FORCE_CUT left behind would silently constrain
    the next solve. Locking an edge is how you tell it to keep its hands off a
    decision you made by hand.

    Object Mode only, like everything else that touches mesh data: in Edit
    Mode the live values are in the BMesh and this writes into a datablock
    nothing is reading.
    """
    ensure_mesh_attrs(mesh)
    attr = mesh.attributes[EDGE_ATTR]
    locked = read_locks(mesh)
    cuts = {(min(a, b), max(a, b)) for a, b in cuts}
    folds = {(min(a, b), max(a, b)) for a, b in folds}

    vals = np.zeros(len(mesh.edges), dtype=np.int32)
    attr.data.foreach_get("value", vals)
    written = 0
    for i, (a, b) in enumerate(edge_keys(mesh)):
        key = (int(a), int(b))
        if key in locked:
            continue
        want = FORCE_CUT if key in cuts else (FORCE_FOLD if key in folds
                                              else AUTO)
        if vals[i] != want:
            vals[i] = want
            written += 1
    attr.data.foreach_set("value", vals)
    mesh.update()
    return written, len(locked)


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


# ----------------------------------------------------------- coordinate spaces
#
# Four of them, and mixing them up is what makes 2D layout work go wrong:
#
#   island space  mm, as unfold produced it. All islands share one origin.
#   sheet space   mm within one bed. Origin at the bed corner; the usable area
#                 is [margin, bed - margin].
#   net space     mm, sheets stacked upward: sheet s owns y in
#                 [s * bed_y, (s+1) * bed_y]. This is what the UV layer holds.
#   uv space      net space * uv_scale(bed). Only ever touched at the edges.
#
# Everything below works in NET SPACE. uv_scale is applied on read and undone
# on write, so no caller has to think in UV units.


def island_loops(mesh, islands, pm: PolyMesh) -> dict:
    """{island id: array of loop indices} - which UVs belong to which island."""
    out = {}
    for isl in islands:
        idx = []
        for f in isl.faces:
            for poly_i in pm.source_faces[f]:
                idx.extend(mesh.polygons[poly_i].loop_indices)
        out[isl.id] = np.asarray(sorted(set(idx)), dtype=np.int64)
    return out


def read_net(mesh, scale: float) -> np.ndarray:
    """The whole net as (nloops, 2) in net-space mm.

    Object Mode only: uv_layers[...].data is empty in Edit Mode, so this would
    quietly hand back zeros. write_net asserts on the same thing.
    """
    uv = mesh.uv_layers.get(UV_LAYER)
    if uv is None or not len(mesh.loops):
        return np.zeros((0, 2))
    buf = np.zeros(len(mesh.loops) * 2)
    uv.data.foreach_get("uv", buf)
    return buf.reshape(-1, 2) / (scale or 1.0)


def write_net(mesh, pts: np.ndarray, scale: float) -> None:
    """Write net-space mm back to the UV layer."""
    uv = mesh.uv_layers.get(UV_LAYER)
    if uv is None:
        return
    assert len(uv.data) == len(mesh.loops), (
        "uv_layers[...].data is empty - this only works in Object Mode")
    uv.data.foreach_set("uv", (pts * (scale or 1.0)).ravel())
    mesh.update()


def _kabsch2d(A: np.ndarray, B: np.ndarray):
    """Best RIGID motion taking A onto B, plus the worst residual in mm.

    Rigid, never a general affine fit: if an island has been scaled, an affine
    fit absorbs it with a near-zero residual and the broken isometry is
    accepted silently into the cut geometry. Rotation and translation only, so
    a scale shows up as a large residual the caller can refuse.
    """
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt = Vt.copy()
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    t = cb - R @ ca
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = t
    res = float(np.abs((A @ R.T + t) - B).max()) if len(A) else 0.0
    return M, res


def island_source_points(mesh, pm: PolyMesh, isl):
    """(island-space points, matching loop indices) for one island."""
    srcs, idx = [], []
    for f, pos2 in isl.faces.items():
        fn = pm.plane_map(f, pos2)
        for poly_i in pm.source_faces[f]:
            poly = mesh.polygons[poly_i]
            srcs.append(fn(pm.vertices[list(poly.vertices)]))
            idx.extend(poly.loop_indices)
    if not srcs:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    return np.vstack(srcs), np.asarray(idx, dtype=np.int64)


def net_transforms(mesh, pm: PolyMesh, islands, scale: float):
    """Where each island sits now: ({island id: 3x3}, worst residual mm).

    Maps ISLAND SPACE to NET SPACE. This is what lets a layout rearranged in
    the UV editor be pushed back onto the parts that actually get exported,
    instead of the 2D view being a read-only preview.
    """
    pts = read_net(mesh, scale)
    out, worst = {}, 0.0
    for isl in islands:
        A, idx = island_source_points(mesh, pm, isl)
        if not len(A):
            continue
        M, res = _kabsch2d(A, pts[idx])
        out[isl.id] = M
        worst = max(worst, res)
    return out, worst


def sheet_gutter(bed_w: float, bed_h: float) -> float:
    """Clear space between sheets in the 2D view, mm.

    The ruler lives OUTSIDE each sheet so it never covers the parts it
    measures, and sheets laid side by side leave no outside unless one is
    made. Scales with the bed, so the ruler's numbers stay readable on a big
    one; never under 40 mm.
    """
    return max(40.0, 0.12 * max(bed_w, bed_h))


def sheet_columns(n: int, bed_w: float, bed_h: float) -> int:
    """How many sheets to a row, so the whole layout is as square as it gets.

    Stacking every sheet upward is wrong for a tall bed: three 400 x 900
    sheets become a 400 x 2700 strip the UV editor has to shrink to a sliver
    to frame. Pick the column count whose bounding box has the aspect ratio
    closest to 1 - measured on a log scale, so 2:1 and 1:2 are equally bad -
    and on a tie, the one with fewer empty slots, then fewer columns.
    """
    if n <= 1 or bed_w <= 0 or bed_h <= 0:
        return 1
    g = sheet_gutter(bed_w, bed_h)
    best = None
    for c in range(1, n + 1):
        r = -(-n // c)
        key = (round(abs(math.log((c * (bed_w + g)) / (r * (bed_h + g)))), 9),
               c * r - n, c)
        if best is None or key < best[0]:
            best = (key, c)
    return best[1]


def sheet_origin(sheet: int, n: int, bed_w: float, bed_h: float) -> tuple:
    """Where sheet `sheet` of `n` sits in the 2D view, in mm.

    Row 0 at the bottom, filling left to right, a `sheet_gutter` apart. A
    part too big for any sheet (sheet < 0) parks in a row of its own above
    the whole grid, so it never lands on a real sheet.
    """
    cols = sheet_columns(max(n, 1), bed_w, bed_h)
    g = sheet_gutter(bed_w, bed_h) if bed_w > 0 and bed_h > 0 else 0.0
    px, py = bed_w + g, bed_h + g
    if sheet < 0:
        return 0.0, (-(-max(n, 1) // cols) * py if n > 0 else 0.0)
    return (sheet % cols) * px, (sheet // cols) * py


def sheet_layout(n: int, bed_w: float, bed_h: float) -> list:
    """Every sheet's origin, in order - what the bed overlay draws."""
    return [sheet_origin(i, n, bed_w, bed_h) for i in range(n)]


def write_net_uv(mesh, pm: PolyMesh, islands, spread: bool = True,
                 row_width: float | None = None, gap: float = 10.0,
                 scale: float = 1.0, placement: dict | None = None,
                 sheet_size: tuple | None = None) -> str:
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
    # where its Part was placed on its sheet, and the sheets are laid out in
    # the grid `sheet_layout` chooses - the one the bed overlay draws. That is
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
    n_sheets = max((p.sheet for p in (placement or {}).values()), default=-1) + 1
    bw, bh = sheet_size if sheet_size else (0.0, 0.0)
    for isl in islands:
        part = (placement or {}).get(isl.id)
        if part is None:
            ox, oy = offsets[isl.id] if offsets else (0.0, 0.0)
        else:
            # a part too big for the bed never got a sheet; sheet_origin parks
            # it above the grid rather than on sheet 0
            ox, oy = sheet_origin(part.sheet, n_sheets, bw, bh)
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
