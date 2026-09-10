"""Rigid isometric unfolding of a planar-faced mesh into flat islands.

The transform is exact. Every face is placed by a rigid motion built from its
own 3D geometry relative to the parent's already-placed shared edge, so edge
lengths are preserved to floating-point and error is additive down the tree
rather than multiplicative. There is no solver and nothing to converge.

The only hard part is combinatorial: choosing which edges to cut so the 2D
layout doesn't self-overlap. No algorithm can guarantee a single-piece result
(Duerer's conjecture is open for convex solids, and non-convex counterexamples
are known), so the objective is to minimise island count, never to succeed.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .mesh import PolyMesh


# ------------------------------------------------------------------ model

@dataclass
class Fold:
    a: np.ndarray          # 2D endpoint
    b: np.ndarray
    bend: float            # signed radians; + convex (mountain), - concave (valley)
    f0: int
    f1: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


@dataclass
class Cut:
    a: np.ndarray
    b: np.ndarray
    face: int
    seam: int              # shared id with the matching cut elsewhere; -1 if open

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))


@dataclass
class Island:
    id: int
    faces: dict[int, np.ndarray] = field(default_factory=dict)   # face -> (n,2) ring
    folds: list[Fold] = field(default_factory=list)
    cuts: list[Cut] = field(default_factory=list)

    def outline(self, grid: float = 1e-9):
        """Union of the island's faces: the shape the laser cuts around.

        Adjacent faces are placed independently, so a shared corner computed
        via two different tree paths lands ~1e-14 apart. Shapely's union turns
        those hairline slivers into lost area and spurious fragments, so snap
        to a grid first. 1e-9 mm is nine orders below the kerf and five above
        the noise. Without this the union silently loses ~1% of the net.
        """
        import shapely
        from shapely.ops import unary_union
        polys = []
        for r in self.faces.values():
            p = Polygon(r)
            if not p.is_valid:
                p = p.buffer(0)
            polys.append(shapely.set_precision(p, grid))
        return unary_union(polys)

    def area(self) -> float:
        """Total flat area. Must equal the 3D surface area of these faces."""
        return float(sum(Polygon(r).area for r in self.faces.values()))

    def bounds(self):
        """(minx, miny, maxx, maxy) straight from the face coordinates -
        no boolean ops, so it can't be perturbed by union precision."""
        pts = np.vstack(list(self.faces.values()))
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    def size(self):
        x0, y0, x1, y1 = self.bounds()
        return (x1 - x0, y1 - y0)

    def translate(self, dx: float, dy: float):
        v = np.array([dx, dy])
        for k in self.faces:
            self.faces[k] = self.faces[k] + v
        for f in self.folds:
            f.a, f.b = f.a + v, f.b + v
        for c in self.cuts:
            c.a, c.b = c.a + v, c.b + v

    def rotate(self, radians: float, origin=(0.0, 0.0)):
        c, s = np.cos(radians), np.sin(radians)
        R = np.array([[c, -s], [s, c]])
        o = np.asarray(origin, dtype=float)
        for k in self.faces:
            self.faces[k] = (self.faces[k] - o) @ R.T + o
        for f in self.folds:
            f.a, f.b = (f.a - o) @ R.T + o, (f.b - o) @ R.T + o
        for c in self.cuts:
            c.a, c.b = (c.a - o) @ R.T + o, (c.b - o) @ R.T + o


# --------------------------------------------------------------- geometry

def _face_local(pm: PolyMesh, face: int, v0: int, v1: int):
    """Coordinates of a face's ring in an orthonormal in-plane frame whose
    origin is v0 and whose x axis runs v0 -> v1. Returns (t, h) per vertex,
    with h SIGNED so concave n-gons place correctly."""
    p, q = pm.vertices[v0], pm.vertices[v1]
    d3 = q - p
    L = np.linalg.norm(d3)
    d3 = d3 / L
    e3 = np.cross(pm.normals[face], d3)
    r = pm.vertices[pm.faces[face]] - p
    return r @ d3, r @ e3, L


def unfold_tree(pm: PolyMesh, tree: nx.Graph, root: int) -> dict[int, np.ndarray]:
    """Place every face of one connected dual-tree component into the plane."""
    pos: dict[int, np.ndarray] = {}

    ring = pm.faces[root]
    v0, v1 = ring[0], ring[1]
    t, h, _ = _face_local(pm, root, v0, v1)
    pos[root] = np.column_stack([t, h])

    for parent, child in nx.bfs_edges(tree, root):
        ev = tree[parent][child]["verts"]
        pr, cr = pm.faces[parent], pm.faces[child]
        if ev[0] not in pr or ev[1] not in pr or ev[0] not in cr or ev[1] not in cr:
            raise ValueError(f"dual arc {parent}-{child} names verts not on both faces")

        P2 = pos[parent][pr.index(ev[0])]
        Q2 = pos[parent][pr.index(ev[1])]
        d = Q2 - P2
        d = d / np.linalg.norm(d)
        e = np.array([-d[1], d[0]])

        # which side of the seam does the parent occupy?
        s_par = np.sign(np.sum((pos[parent] - P2) @ e))
        t, h, L = _face_local(pm, child, ev[0], ev[1])
        hm = np.sign(np.sum(h)) or 1.0
        k = -s_par * hm                       # flip the child to the far side

        pos[child] = P2 + np.outer(t, d) + k * np.outer(h, e)
    return pos


def isometry_error(pm: PolyMesh, pos: dict[int, np.ndarray]) -> float:
    """Max relative edge-length error between 3D and the 2D layout."""
    worst = 0.0
    for f, ring2 in pos.items():
        v3 = pm.vertices[pm.faces[f]]
        n = len(v3)
        for i in range(n):
            j = (i + 1) % n
            l3 = np.linalg.norm(v3[i] - v3[j])
            l2 = np.linalg.norm(ring2[i] - ring2[j])
            if l3 > 1e-12:
                worst = max(worst, abs(l3 - l2) / l3)
    return worst


def overlapping_pairs(forest: nx.Graph, pos: dict[int, np.ndarray],
                      shrink: float = 1e-6) -> set[tuple[int, int]]:
    """Face pairs whose placements collide.

    Two exclusions, both essential:

      * Faces joined by a forest arc share an edge by construction.
      * Faces in DIFFERENT components. Each island is unfolded from its own
        root into the same raw coordinate space, so islands routinely sit on
        top of each other there - and it means nothing, because packing gives
        each one its own place on a sheet. Counting those as overlaps makes a
        healthy multi-island net look catastrophically broken.
    """
    comp_of: dict[int, int] = {}
    for ci, comp in enumerate(nx.connected_components(forest)):
        for f in comp:
            comp_of[f] = ci

    polys, live = [], []
    for f in sorted(pos):
        p = Polygon(pos[f])
        if not p.is_valid:
            p = p.buffer(0)
        p = p.buffer(-shrink * max(1.0, np.sqrt(abs(p.area))))
        if not p.is_empty:
            polys.append(p)
            live.append(f)
    if len(polys) < 2:
        return set()

    idx = STRtree(polys)
    bad = set()
    for i, pi in enumerate(polys):
        for j in idx.query(pi):
            j = int(j)
            if j <= i:
                continue
            fa, fb = live[i], live[j]
            if comp_of.get(fa, -1) != comp_of.get(fb, -2):
                continue
            if forest.has_edge(fa, fb):
                continue
            if pi.intersection(polys[j]).area > 1e-9:
                bad.add((fa, fb))
    return bad


# ----------------------------------------------------------------- search

def apply_constraints(G: nx.Graph, forced_cuts: set = (),
                      forced_folds: set = ()) -> tuple[nx.Graph, list[str]]:
    """Honour hand-placed seams.

    forced_cuts / forced_folds hold sorted (v0, v1) mesh-vertex pairs. A cut
    arc is deleted from the dual outright; a fold arc is flagged so the
    spanning tree always keeps it. Forced folds that would close a cycle
    cannot all be honoured - a spanning tree has no cycles - so those are
    reported rather than silently dropped.
    """
    G = G.copy()
    cuts = {tuple(sorted(e)) for e in forced_cuts}
    folds = {tuple(sorted(e)) for e in forced_folds}
    notes = []

    for a, b, d in list(G.edges(data=True)):
        if tuple(sorted(d["verts"])) in cuts:
            G.remove_edge(a, b)

    keep = nx.Graph()
    keep.add_nodes_from(G.nodes())
    dropped = 0
    for a, b, d in G.edges(data=True):
        if tuple(sorted(d["verts"])) in folds:
            if nx.has_path(keep, a, b) if (keep.has_node(a) and keep.has_node(b)) else False:
                dropped += 1
                continue
            keep.add_edge(a, b)
            G[a][b]["forced_fold"] = True
    if dropped:
        notes.append(f"{dropped} forced fold(s) would close a cycle and were "
                     f"left to the solver - a net cannot keep every edge joined")
    if cuts:
        notes.append(f"{len(cuts)} forced cut(s) applied")
    return G, notes


def weighted_dual(pm: PolyMesh, G: nx.Graph, c: np.ndarray,
                  beta: float = 0.5, jitter: float = 0.05,
                  rng: np.random.Generator | None = None) -> nx.Graph:
    """Score dual arcs so a MAXIMUM spanning tree keeps long edges as folds
    (short cut perimeter) and cuts preferentially across a direction c.

    Randomising c between restarts is the highest-leverage knob in the search:
    the steepest-edge family is direction-parameterised, so it sweeps a
    structurally diverse slice of tree space that jittering weights does not.
    """
    W = nx.Graph()
    lmax = max((d["length"] for *_, d in G.edges(data=True)), default=1.0)
    for a, b, d in G.edges(data=True):
        v0, v1 = d["verts"]
        ed = pm.vertices[v1] - pm.vertices[v0]
        ed = ed / np.linalg.norm(ed)
        w = (1 - beta) * (d["length"] / lmax) + beta * abs(float(ed @ c))
        if rng is not None and jitter:
            w += jitter * rng.random()
        if d.get("forced_fold"):
            w += 1e3                      # a max spanning tree will always keep it
        W.add_edge(a, b, weight=w, **d)
    return W


def _forest_positions(pm: PolyMesh, forest: nx.Graph):
    pos, comps = {}, []
    for comp in nx.connected_components(forest):
        comp = sorted(comp)
        sub = forest.subgraph(comp)
        pos.update(unfold_tree(pm, sub, comp[0]))
        comps.append(comp)
    return pos, comps


def search(pm: PolyMesh, G: nx.Graph | None = None, trials: int = 250,
           seed: int = 0, verbose: bool = False):
    """Randomised-restart maximum spanning tree with overlap rejection.

    Returns (forest, pos, overlaps, trials_used). A clean result has an empty
    overlap set; otherwise the caller repairs by splitting.
    """
    G = G if G is not None else pm.dual_graph()
    rng = np.random.default_rng(seed)
    best = None
    for i in range(trials):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        W = weighted_dual(pm, G, c, rng=rng)
        T = nx.maximum_spanning_tree(W)
        pos, _ = _forest_positions(pm, T)
        bad = overlapping_pairs(T, pos)
        if not bad:
            if verbose:
                print(f"  clean unfold on trial {i + 1}")
            return T, pos, set(), i + 1
        if best is None or len(bad) < len(best[2]):
            best = (T, pos, bad, i + 1)
    return best


def repair(pm: PolyMesh, forest: nx.Graph, overlaps: set, max_cuts: int = 500):
    """Greedy minimum set cover: repeatedly cut the tree arc that breaks the
    most conflicting face-pair paths. Provably terminates - in the worst case
    every face becomes its own island."""
    forest = forest.copy()
    cuts = 0
    pos, _ = _forest_positions(pm, forest)
    bad = overlaps
    while bad and cuts < max_cuts:
        cover: dict[tuple[int, int], int] = {}
        for a, b in bad:
            try:
                path = nx.shortest_path(forest, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for u, v in zip(path, path[1:]):
                key = (min(u, v), max(u, v))
                cover[key] = cover.get(key, 0) + 1
        if not cover:
            break
        forest.remove_edge(*max(cover, key=cover.get))
        cuts += 1
        pos, _ = _forest_positions(pm, forest)
        bad = overlapping_pairs(forest, pos)
    return forest, pos, bad, cuts


# -------------------------------------------------------------- bed limit

def min_rect_size(pts: np.ndarray) -> tuple[float, float]:
    """Side lengths of the minimum-area rotated bounding box of a point cloud.
    A net that is too wide axis-aligned often fits fine on the diagonal, so
    testing the AABB alone rejects islands that would actually have fitted."""
    from shapely.geometry import MultiPoint
    if len(pts) < 3:
        return (0.0, 0.0)
    r = MultiPoint([tuple(p) for p in pts]).minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        b = pts.max(axis=0) - pts.min(axis=0)
        return (float(b[0]), float(b[1]))
    c = np.asarray(r.exterior.coords)[:4]
    e = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    return (float(e[0]), float(e[1]))


def fits(pts: np.ndarray, max_w: float, max_h: float) -> bool:
    w, h = min_rect_size(pts)
    return (w <= max_w and h <= max_h) or (h <= max_w and w <= max_h)


def split_to_fit(pm: PolyMesh, forest: nx.Graph, pos: dict[int, np.ndarray],
                 max_w: float, max_h: float, max_splits: int = 400):
    """Cut tree arcs until every island fits the bed.

    Key economy: removing an arc never moves a face relative to the others in
    its own half - the placement of each half is fixed by the tree paths that
    survive inside it. So candidate splits can be scored straight from the
    existing coordinates, with no re-unfolding.

    Splitting along a fold edge is the good kind of split: the seam lands on a
    real crease that already needs a glue flap, instead of a flat butt joint
    across the middle of a panel.
    """
    forest = forest.copy()
    splits = 0

    def comp_pts(comp):
        return np.vstack([pos[f] for f in comp])

    while splits < max_splits:
        bad = [c for c in nx.connected_components(forest)
               if not fits(comp_pts(c), max_w, max_h)]
        if not bad:
            break
        comp = max(bad, key=len)
        sub = forest.subgraph(comp).copy()
        if sub.number_of_edges() == 0:
            break                       # a single face larger than the bed

        best, best_score = None, None
        for u, v in list(sub.edges()):
            sub.remove_edge(u, v)
            half = nx.node_connected_component(sub, u)
            other = set(comp) - half
            sub.add_edge(u, v)
            if not other:
                continue
            a = min_rect_size(comp_pts(half))
            b = min_rect_size(comp_pts(other))
            # minimise the worse half's overflow, then prefer an even split
            over = max(max(a) / max_w, min(a) / max_h,
                       max(b) / max_w, min(b) / max_h)
            score = (over, abs(len(half) - len(other)))
            if best_score is None or score < best_score:
                best, best_score = (u, v), score
        if best is None:
            break
        forest.remove_edge(*best)
        splits += 1

    pos, _ = _forest_positions(pm, forest)
    still = [c for c in nx.connected_components(forest)
             if not fits(comp_pts(c), max_w, max_h)]
    return forest, pos, splits, len(still)


# ------------------------------------------------------------ island build

def build_islands(pm: PolyMesh, G: nx.Graph, forest: nx.Graph,
                  pos: dict[int, np.ndarray]) -> list[Island]:
    """Turn a dual forest plus 2D placements into Island records carrying
    explicit fold and cut segments, with cut seams paired across islands."""
    comp_of: dict[int, int] = {}
    islands: list[Island] = []
    for idx, comp in enumerate(nx.connected_components(forest)):
        isl = Island(id=idx)
        for f in sorted(comp):
            isl.faces[f] = pos[f]
            comp_of[f] = idx
        islands.append(isl)

    # folds: dual arcs retained in the forest
    for a, b, d in forest.edges(data=True):
        v0, v1 = d["verts"]
        ring = pm.faces[a]
        p2 = pos[a][ring.index(v0)]
        q2 = pos[a][ring.index(v1)]
        islands[comp_of[a]].folds.append(
            Fold(a=p2, b=q2, bend=d["bend"], f0=a, f1=b))

    # cuts: dual arcs dropped, plus genuinely open mesh boundary
    seam = 0
    for a, b, d in G.edges(data=True):
        if forest.has_edge(a, b):
            continue
        v0, v1 = d["verts"]
        for f in (a, b):
            ring = pm.faces[f]
            islands[comp_of[f]].cuts.append(
                Cut(a=pos[f][ring.index(v0)], b=pos[f][ring.index(v1)],
                    face=f, seam=seam))
        seam += 1

    dual_pairs = {(min(a, b), max(a, b)) for a, b in G.edges()}
    for (v0, v1), uses in pm.edge_map().items():
        if len(uses) == 1:                       # open boundary of a shell
            f, i = uses[0]
            ring = pm.faces[f]
            islands[comp_of[f]].cuts.append(
                Cut(a=pos[f][ring.index(v0)], b=pos[f][ring.index(v1)],
                    face=f, seam=-1))
    return islands


def unfold(pm: PolyMesh, trials: int = 250, seed: int = 0,
           max_size: tuple[float, float] | None = None,
           forced_cuts: set = (), forced_folds: set = (),
           verbose: bool = False):
    """Full unfold: search, repair overlaps, then split anything too big.

    max_size is the island limit in mm, which should already have room
    subtracted for glue tabs and kerf - Blender's addon tests the bare island
    against the full page and then adds flaps afterwards, which is why its
    output sometimes still overflows.
    """
    G_full = pm.dual_graph()
    G, notes = apply_constraints(G_full, forced_cuts, forced_folds)

    forest, pos, bad, used = search(pm, G, trials=trials, seed=seed, verbose=verbose)
    extra = 0
    if bad:
        forest, pos, bad, extra = repair(pm, forest, bad)

    splits = oversize = 0
    if max_size is not None:
        forest, pos, splits, oversize = split_to_fit(pm, forest, pos, *max_size)
        bad = overlapping_pairs(forest, pos)

    # seams are reported against the full dual so forced cuts still get tabs
    islands = build_islands(pm, G_full, forest, pos)
    report = dict(trials_used=used, extra_cuts=extra, size_splits=splits,
                  islands=len(islands), unresolved=len(bad),
                  oversize_islands=oversize, notes=notes,
                  isometry_error=isometry_error(pm, pos))
    return islands, report
