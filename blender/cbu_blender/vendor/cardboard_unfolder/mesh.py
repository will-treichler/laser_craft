"""Mesh loading, coplanar-face merging, and the dual graph.

The merge step matters more than it sounds. An STL of a cube is twelve
triangles; unfolded naively you get a fold line down the diagonal of every
face. Merging coplanar triangles back into n-gons is what makes the output
look like a designed part instead of a triangulated one.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import networkx as nx

# trimesh is imported lazily, inside the two loaders that need it. It is only
# ever used to read a mesh file; the Blender add-on builds its PolyMesh from
# bmesh instead and never touches those loaders. An eager import here made the
# add-on fail to register in Blender's Python with ModuleNotFoundError, which
# also hid the panel offering to install the missing packages.


def newell_normal(pts: np.ndarray) -> np.ndarray:
    """Robust polygon normal for an n-gon (Newell's method)."""
    n = np.zeros(3)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        n += np.cross(a, b)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def _chain_boundary(edges: np.ndarray) -> list[int] | None:
    """Chain an unordered set of boundary edges into a single vertex ring.

    Returns None if the boundary isn't one simple closed loop (a facet with a
    hole, or a non-manifold fan) - the caller then declines to merge it.
    """
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adj.values()):
        return None

    start = next(iter(adj))
    ring = [start]
    prev, cur = None, start
    while True:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        ring.append(nxt)
        prev, cur = cur, nxt
        if len(ring) > len(adj):
            return None
    return ring if len(ring) == len(adj) else None


def ring_planarity(vertices: np.ndarray, ring: list[int]) -> float:
    """Max deviation from the ring's best-fit plane, relative to its longest
    edge - the same measure PolyMesh.planarity reports, for one loose ring."""
    pts = vertices[ring]
    n = newell_normal(pts)
    dev = float(np.abs((pts - pts.mean(axis=0)) @ n).max())
    span = max(float(np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)]))
               for i in range(len(pts)))
    return dev / span if span > 1e-12 else 0.0


def split_nonplanar(vertices: np.ndarray, rings: list[list[int]],
                    tol: float = 1e-3):
    """Fan-triangulate any ring too bent to unfold rigidly.

    A non-planar quad has no flat position that keeps all four corners, so the
    two faces meeting along one of its edges disagree about where that corner
    goes - measured at 0.79 mm on a real part, well past the +/-0.5 mm this
    is built for. Splitting is not a nicety: it is the only honest handling,
    because the isometry guarantee simply does not apply to a bent face.

    Triangles come back through merge_coplanar, so genuinely flat faces that
    were triangulated on import are re-merged and do not gain a fold line.
    Only the bent ones stay split, and those have to bend somewhere.

    Returns (new_rings, origin) where origin[i] is the input ring that ring i
    came from, so source_faces can still point at the caller's own faces.
    """
    out: list[list[int]] = []
    origin: list[int] = []
    for i, r in enumerate(rings):
        if len(r) > 3 and ring_planarity(vertices, r) > tol:
            for k in range(1, len(r) - 1):      # fan from the first vertex
                out.append([r[0], r[k], r[k + 1]])
                origin.append(i)
        else:
            out.append(list(r))
            origin.append(i)
    return out, origin


def merge_coplanar(vertices: np.ndarray, rings: list[list[int]],
                   tol_deg: float = 0.5):
    """Group adjacent near-coplanar rings into single faces.

    Returns (new_rings, source_lists) where source_lists[i] lists the input
    ring indices that were merged into new_rings[i]. Callers keep that map so
    they can push results back onto the original faces (Blender needs it to
    write per-loop UVs).
    """
    normals = np.array([newell_normal(vertices[r]) for r in rings])
    parent = list(range(len(rings)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    em: dict[tuple[int, int], list[int]] = {}
    for fi, ring in enumerate(rings):
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            em.setdefault((min(a, b), max(a, b)), []).append(fi)

    cos_tol = np.cos(np.radians(tol_deg))
    for uses in em.values():
        if len(uses) != 2:
            continue
        a, b = uses
        if float(normals[a] @ normals[b]) >= cos_tol:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for fi in range(len(rings)):
        groups.setdefault(find(fi), []).append(fi)

    # Interior edges per group, in ONE pass over the edge map. This used to be
    # a comprehension over the whole of `em` inside the group loop, with an
    # `in members` list test inside that - O(groups x edges x group size). It
    # is invisible on a 300-face model and fatal on a real scan: a 500k
    # triangle STL took ~40 minutes to merge, scaling at n^1.7 (133 us/tri at
    # 1k, 450 at 16k). Same answer, one pass.
    inner_by_root: dict[int, set] = {}
    for e, u in em.items():
        if len(u) != 2:
            continue
        ra, rb = find(u[0]), find(u[1])
        if ra == rb:
            inner_by_root.setdefault(ra, set()).add(e)

    out_rings, out_src = [], []
    for root, members in groups.items():
        if len(members) == 1:
            out_rings.append(list(rings[members[0]]))
            out_src.append(members)
            continue
        inner = inner_by_root.get(root, ())
        bedges = []
        for fi in members:
            r = rings[fi]
            for i in range(len(r)):
                a, b = r[i], r[(i + 1) % len(r)]
                if (min(a, b), max(a, b)) not in inner:
                    bedges.append((a, b))
        ring = _chain_boundary(np.array(bedges)) if bedges else None
        if ring is None:                     # a hole or a fan - leave unmerged
            for fi in members:
                out_rings.append(list(rings[fi]))
                out_src.append([fi])
            continue
        n = newell_normal(vertices[ring])
        if float(n @ normals[members[0]]) < 0:
            ring = ring[::-1]
        out_rings.append(ring)
        out_src.append(members)
    return out_rings, out_src


@dataclass
class PolyMesh:
    """A mesh of planar polygonal faces, each stored as an ordered vertex ring."""
    vertices: np.ndarray                      # (V,3)
    faces: list[list[int]]                    # ordered rings, CCW seen from outside
    source_faces: list = None                 # face -> original face indices
    normals: np.ndarray = field(init=False)   # (F,3)

    def __post_init__(self):
        self.normals = np.array([newell_normal(self.vertices[r]) for r in self.faces])
        if self.source_faces is None:
            self.source_faces = [[i] for i in range(len(self.faces))]

    @classmethod
    def from_rings(cls, vertices, rings, merge: bool = True, tol_deg: float = 0.5,
                   planar_tol: float | None = 1e-3):
        vertices = np.asarray(vertices, dtype=float)
        rings = [list(r) for r in rings]

        origin = list(range(len(rings)))
        if planar_tol is not None:
            rings, origin = split_nonplanar(vertices, rings, planar_tol)

        if merge:
            rings, src = merge_coplanar(vertices, rings, tol_deg)
            # merge_coplanar indexes the rings it was handed; map back to the
            # caller's original faces or source_faces points at the wrong ones.
            src = [sorted({origin[j] for j in group}) for group in src]
            return cls(vertices, rings, src)
        return cls(vertices, rings, [[origin[i]] for i in range(len(rings))])

    def plane_map(self, face: int, pos2: np.ndarray):
        """Return f(points3d) -> points2d, the rigid map that placed this face.

        Needed to push 2D results back onto geometry that lives on the face
        but isn't in its ring - the original triangles inside a merged n-gon,
        and every polygon `write_net_uv` writes.

        FITTED to every corner of the ring (a 2-D Procrustes fit, both
        handednesses tried, the better kept), not read off the first edge. On
        a flat face the two are the same map. On a bent one they are not: the
        first version took the whole orientation from the first edge, and on
        the user's helmet that edge was 1.94 mm long - flattening the face
        shortened and turned it a little, and 372 mm along the sliver the turn
        put the face 11.8 mm from its own cut outline in the 2D view. Fitted,
        the map lands within the bend itself
        (`test_plane_map_puts_a_bent_face_where_the_unfold_put_it`).
        """
        ring = self.faces[face]
        V = self.vertices[ring]
        o = V.mean(axis=0)
        n = self.normals[face]
        u = V[np.argmax(np.linalg.norm(V - o, axis=1))] - o
        u = u - n * float(u @ n)
        u = u / np.linalg.norm(u)
        w = np.cross(n, u)
        P = np.asarray(pos2, dtype=float)
        pc = P.mean(axis=0)
        Q = P - pc
        best = None
        for k in (1.0, -1.0):
            L = np.column_stack([(V - o) @ u, k * ((V - o) @ w)])
            # rotation taking L onto Q, least squares (2-D Kabsch)
            a = float(np.sum(L[:, 0] * Q[:, 0] + L[:, 1] * Q[:, 1]))
            b = float(np.sum(L[:, 0] * Q[:, 1] - L[:, 1] * Q[:, 0]))
            th = np.arctan2(b, a)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])
            err = float(np.sum((L @ R.T - Q) ** 2))
            if best is None or err < best[0]:
                best = (err, k, R)
        _err, k, R = best

        def fn(p):
            r = np.atleast_2d(np.asarray(p, dtype=float)) - o
            loc = np.column_stack([r @ u, k * (r @ w)])
            return loc @ R.T + pc
        return fn

    # ---------------------------------------------------------------- build

    @classmethod
    def from_trimesh(cls, tm: "trimesh.Trimesh", merge_coplanar: bool = True) -> "PolyMesh":
        tm = tm.copy()
        tm.merge_vertices()
        if not merge_coplanar:
            return cls(np.asarray(tm.vertices), [list(f) for f in tm.faces])

        return cls.from_rings(np.asarray(tm.vertices),
                              [[int(v) for v in f] for f in tm.faces], merge=True)

    @classmethod
    def load(cls, path: str, merge_coplanar: bool = True) -> "PolyMesh":
        import trimesh
        tm = trimesh.load(path, force="mesh", process=True)
        return cls.from_trimesh(tm, merge_coplanar)

    # ------------------------------------------------------------ topology

    def edge_map(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """(v0,v1) sorted -> [(face_index, position_in_ring), ...]"""
        em: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for fi, ring in enumerate(self.faces):
            for i in range(len(ring)):
                a, b = ring[i], ring[(i + 1) % len(ring)]
                em.setdefault((min(a, b), max(a, b)), []).append((fi, i))
        return em

    def dual_graph(self) -> nx.Graph:
        """Faces as nodes, shared edges as arcs, carrying the signed bend angle.

        `bend` is alpha: how far the pair must fold from flat, in radians.
        Positive is convex (mountain from outside), negative concave.
        """
        G = nx.Graph()
        G.add_nodes_from(range(len(self.faces)))
        for (v0, v1), uses in self.edge_map().items():
            if len(uses) != 2:
                continue                      # boundary or non-manifold; never a fold
            (fa, ia), (fb, _) = uses
            # orient the edge along face A's winding so the sign is well defined
            ring = self.faces[fa]
            p, q = ring[ia], ring[(ia + 1) % len(ring)]
            d = self.vertices[q] - self.vertices[p]
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                continue
            d = d / nd
            na, nb = self.normals[fa], self.normals[fb]
            bend = float(np.arctan2(np.dot(np.cross(na, nb), d), np.dot(na, nb)))
            if G.has_edge(fa, fb):
                # two faces meeting along more than one edge; keep the longest
                if G[fa][fb]["length"] >= nd:
                    continue
            G.add_edge(fa, fb, verts=(v0, v1), length=float(nd), bend=bend)
        return G

    # ------------------------------------------------------------ validate

    def planarity(self) -> np.ndarray:
        """Per-face max vertex deviation from its best-fit plane, relative to
        the face's longest edge. Anything above ~1e-3 breaks the isometry
        guarantee: unfolding is only exact for genuinely planar faces."""
        out = np.zeros(len(self.faces))
        for fi, ring in enumerate(self.faces):
            pts = self.vertices[ring]
            n = self.normals[fi]
            dev = np.abs((pts - pts.mean(axis=0)) @ n).max()
            span = max(np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)])
                       for i in range(len(pts)))
            out[fi] = dev / span if span > 1e-12 else 0.0
        return out

    def check(self, tol: float = 1e-3) -> list[str]:
        problems = []
        bad = np.nonzero(self.planarity() > tol)[0]
        if len(bad):
            problems.append(
                f"{len(bad)} non-planar face(s) (worst {self.planarity().max():.2e} "
                f"> {tol:.0e}); triangulate or planarize them first")
        em = self.edge_map()
        open_edges = sum(1 for u in em.values() if len(u) == 1)
        nonman = sum(1 for u in em.values() if len(u) > 2)
        if nonman:
            problems.append(f"{nonman} non-manifold edge(s) with >2 faces")
        if open_edges:
            problems.append(f"{open_edges} open boundary edge(s) - mesh is not closed "
                            f"(fine for a shell, but tabs on those edges have no partner)")
        if not nx.is_connected(self.dual_graph()):
            problems.append("mesh has disconnected shells; each unfolds separately")
        return problems

    # --------------------------------------------------------------- utils

    @property
    def extents(self) -> np.ndarray:
        return self.vertices.max(axis=0) - self.vertices.min(axis=0)

    def scale_to_height(self, height_mm: float, axis: int = 2) -> "PolyMesh":
        cur = self.extents[axis]
        if cur < 1e-9:
            raise ValueError("mesh has no extent along the scaling axis")
        return PolyMesh(self.vertices * (height_mm / cur), self.faces,
                        self.source_faces)

    def scale(self, factor: float) -> "PolyMesh":
        return PolyMesh(self.vertices * factor, self.faces, self.source_faces)

    def __repr__(self):
        sizes = [len(r) for r in self.faces]
        return (f"PolyMesh({len(self.faces)} faces, {len(self.vertices)} verts, "
                f"ring sizes {min(sizes)}-{max(sizes)})")
