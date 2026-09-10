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

import numpy as np
import bmesh
import bpy
from mathutils import Matrix

from .vendor.cardboard_unfolder.mesh import PolyMesh

EDGE_ATTR = "cbu_constraint"
FACE_ATTR = "cbu_island"
UV_LAYER = "CBU_Net"

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

    rings = [[int(v) for v in p.vertices] for p in mesh.polygons]
    pm = PolyMesh.from_rings(verts, rings, merge=merge_coplanar, tol_deg=tol_deg)
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


def write_net_uv(mesh, pm: PolyMesh, islands, spread: bool = True) -> str:
    """Write the flat net into a UV layer, in millimetres.

    An isometric unfold IS a UV parameterisation, and UVs are stored per
    loop - which is exactly the storage an unfold needs, because cuts
    duplicate vertices and a shared corner has two different flat positions.
    Putting the net here means the UV editor becomes the 2D view for free,
    including island selection and uv.pack_islands as a true-shape nester.
    """
    uv = mesh.uv_layers.get(UV_LAYER) or mesh.uv_layers.new(name=UV_LAYER)

    offsets = {}
    if spread:                      # islands are unfolded into a shared origin
        x = 0.0
        for isl in sorted(islands, key=lambda i: i.id):
            x0, y0, x1, y1 = isl.bounds()
            offsets[isl.id] = (x - x0, -y0)
            x += (x1 - x0) * 1.06 + 10.0
    else:
        offsets = {i.id: (0.0, 0.0) for i in islands}

    # pm.vertices is already in the core's mm space and is indexed by the
    # original Blender vertex indices, so it is the correct source here -
    # re-deriving it from mesh.vertices would drop the object transform.
    data = np.zeros((len(mesh.loops), 2), dtype=np.float64)
    for isl in islands:
        ox, oy = offsets[isl.id]
        for f, pos2 in isl.faces.items():
            fn = pm.plane_map(f, pos2)
            for src in pm.source_faces[f]:
                poly = mesh.polygons[src]
                idx = list(poly.loop_indices)
                data[idx] = fn(pm.vertices[list(poly.vertices)]) + (ox, oy)
    uv.data.foreach_set("uv", data.ravel())
    mesh.update()
    return UV_LAYER
