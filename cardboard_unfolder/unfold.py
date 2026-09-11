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
           seed: int = 0, verbose: bool = False,
           max_size: tuple | None = None, size_candidates: int = 6):
    """Randomised-restart maximum spanning tree with overlap rejection.

    Returns (forest, pos, overlaps, trials_used). A clean result has an empty
    overlap set; otherwise the caller repairs by splitting.

    With max_size given, a tree that merely avoids overlaps is not good
    enough. Overlap-free trees are plentiful - what separates them is how many
    pieces they cost once the bed limit is applied, and that varies by a third
    or more on real parts (a dome measured 4 pieces from the best tree and 6
    from the worst). So instead of returning the first clean tree, score up to
    size_candidates of them by the objective that actually matters - the piece
    count after bed splitting - and keep the cheapest.

    Scoring costs a bed split per candidate, so it is capped rather than run
    over every trial, and skipped entirely when there is no bed limit.
    """
    G = G if G is not None else pm.dual_graph()
    rng = np.random.default_rng(seed)
    best = None                     # fewest overlaps, used only if none are clean
    best_clean = None               # (score, T, pos, trials_used)
    seen_clean = 0
    for i in range(trials):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        W = weighted_dual(pm, G, c, rng=rng)
        T = nx.maximum_spanning_tree(W)
        pos, _ = _forest_positions(pm, T)
        bad = overlapping_pairs(T, pos)
        if not bad:
            if max_size is None:
                if verbose:
                    print(f"  clean unfold on trial {i + 1}")
                return T, pos, set(), i + 1
            seen_clean += 1
            _, _, ncuts, over = split_to_fit(pm, T, pos, *max_size, trials=3)
            # an unsplittable oversized face is worse than any piece count
            score = (over, 1 + ncuts)
            if best_clean is None or score < best_clean[0]:
                best_clean = (score, T, pos, i + 1)
            if seen_clean >= size_candidates:
                break
            continue
        if best is None or len(bad) < len(best[2]):
            best = (T, pos, bad, i + 1)

    if best_clean is not None:
        score, T, pos, used = best_clean
        if verbose:
            print(f"  best of {seen_clean} clean unfold(s): {score[1]} piece(s)")
        return T, pos, set(), used
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

def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, counter-clockwise, no repeated endpoint.

    Written out instead of taken from GEOS because its rotated-rectangle
    routine returns a degenerate sliver on some of these nets - a hull of
    2.3e5 mm2 came back as a rectangle of 7e-12 - and a wrong "it fits" lets
    an oversized island straight through to the packer. Collinear points are
    dropped, so every edge of the result is a genuine support direction.
    """
    p = np.unique(np.asarray(pts, dtype=float), axis=0)   # also sorts lexically
    if len(p) < 3:
        return p

    def chain(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                cross = ((b[0] - a[0]) * (q[1] - a[1])
                         - (b[1] - a[1]) * (q[0] - a[0]))
                if cross > 0.0:
                    break
                out.pop()
            out.append(q)
        return out

    lower, upper = chain(p), chain(p[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def min_area_rect(pts: np.ndarray) -> tuple[float, float, float, tuple]:
    """Minimum-area enclosing rectangle, as (width, height, angle, centre).

    width runs along angle (radians), height across it. Rotating calipers over
    the hull edges: the minimum-area rectangle always has a side flush with
    one of them, and the extreme along any direction is attained at a hull
    vertex, so projecting the hull alone is exact rather than a sample.
    """
    h = _convex_hull(pts)
    if len(h) == 0:
        return (0.0, 0.0, 0.0, (0.0, 0.0))
    if len(h) < 3:                       # a single point, or a collinear run
        d = h[-1] - h[0]
        length = float(np.hypot(d[0], d[1]))
        mid = (h[0] + h[-1]) / 2.0
        angle = float(np.arctan2(d[1], d[0])) if length > 0.0 else 0.0
        return (length, 0.0, angle, (float(mid[0]), float(mid[1])))

    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        length = float(np.hypot(e[0], e[1]))
        if length <= 0.0:
            continue
        d = e / length
        n = np.array([-d[1], d[0]])
        u, v = h @ d, h @ n
        u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
        w, ht = float(u1 - u0), float(v1 - v0)
        if best is None or w * ht < best[0]:
            c = d * ((u0 + u1) / 2.0) + n * ((v0 + v1) / 2.0)
            best = (w * ht, w, ht, float(np.arctan2(d[1], d[0])),
                    (float(c[0]), float(c[1])))
    if best is None:                     # every hull edge of zero length
        return (0.0, 0.0, 0.0, (float(h[0][0]), float(h[0][1])))
    return (best[1], best[2], best[3], best[4])


def min_rect_size(pts: np.ndarray) -> tuple[float, float]:
    """Side lengths of the minimum-area rotated bounding box of a point cloud.
    A net that is too wide axis-aligned often fits fine on the diagonal, so
    testing the AABB alone rejects islands that would actually have fitted."""
    if len(pts) < 3:
        return (0.0, 0.0)
    w, h, _, _ = min_area_rect(pts)
    return (w, h)


def fits(pts: np.ndarray, max_w: float, max_h: float) -> bool:
    w, h = min_rect_size(pts)
    return (w <= max_w and h <= max_h) or (h <= max_w and w <= max_h)


def _grow_partition(sub: nx.Graph, pos: dict[int, np.ndarray],
                    max_w: float, max_h: float, root: int,
                    order: str, rng) -> list:
    """Partition one tree into as few connected pieces as the bed allows.

    Bottom-up: every node starts a piece, absorbs each child's piece if the
    result still fits, and cuts that child off if it does not. Growing pieces
    to their limit is the whole point - the previous scorer preferred *even*
    splits, which is precisely how you turn one net into many small ones.

    Two properties make this sound rather than merely plausible:

    - Feasibility is downward-closed. Dropping faces can only shrink the point
      cloud, so "grow until it stops fitting" is well defined.
    - A cut can never become regrettable. After a child is cut off, its piece
      is final while the parent's piece only ever grows, and a superset of a
      set that did not fit does not fit either. So no two finished pieces
      could have been left joined - the result is maximal, not just valid.

    Only the convex hull of each piece is carried, since min_area_rect reads
    nothing else, which keeps the merge test cheap as pieces grow.
    """
    parent = {}
    for u, v in nx.dfs_edges(sub, source=root):
        parent[v] = u

    hulls: dict[int, np.ndarray] = {}
    sizes: dict[int, int] = {}
    cuts = []
    for u in nx.dfs_postorder_nodes(sub, source=root):
        hull = _convex_hull(pos[u])
        size = 1
        kids = [v for v in sub.neighbors(u) if parent.get(v) is u or parent.get(v) == u]
        if order == "small":
            kids.sort(key=lambda v: sizes[v])
        elif order == "big":
            kids.sort(key=lambda v: -sizes[v])
        else:
            rng.shuffle(kids)
        for v in kids:
            cand = np.vstack([hull, hulls[v]])
            if fits(cand, max_w, max_h):
                hull = _convex_hull(cand)
                size += sizes[v]
            else:
                cuts.append((u, v))
        hulls[u], sizes[u] = hull, size
    return cuts


def split_to_fit(pm: PolyMesh, forest: nx.Graph, pos: dict[int, np.ndarray],
                 max_w: float, max_h: float, max_splits: int = 400,
                 trials: int = 8, seed: int = 0):
    """Cut tree arcs until every island fits the bed, using as few as possible.

    Key economy: removing an arc never moves a face relative to the others in
    its own half - the placement of each half is fixed by the tree paths that
    survive inside it. So candidate splits can be scored straight from the
    existing coordinates, with no re-unfolding.

    Splitting along a fold edge is the good kind of split: the seam lands on a
    real crease that already needs a glue flap, instead of a flat butt joint
    across the middle of a panel.

    Cutting k arcs of a forest always yields exactly k more components, so
    "fewest pieces" is just "fewest cuts" - the trials below vary the root and
    the order children are absorbed in, and keep the cheapest result.
    """
    forest = forest.copy()
    rng = np.random.default_rng(seed)

    comps = [c for c in nx.connected_components(forest) if len(c) > 1]
    best_cuts = None
    for t in range(max(1, trials)):
        cuts = []
        for comp in comps:
            sub = forest.subgraph(comp)
            nodes = sorted(comp)
            if t == 0:
                root = nodes[0]
                order = "small"
            elif t == 1:
                root = nodes[0]
                order = "big"
            else:
                root = nodes[int(rng.integers(len(nodes)))]
                order = ("small", "big", "random")[t % 3]
            cuts += _grow_partition(sub, pos, max_w, max_h, root, order, rng)
        if best_cuts is None or len(cuts) < len(best_cuts):
            best_cuts = cuts
            if not cuts:
                break                   # already one piece per component

    best_cuts = (best_cuts or [])[:max_splits]
    forest.remove_edges_from(best_cuts)

    pos, _ = _forest_positions(pm, forest)
    still = [c for c in nx.connected_components(forest)
             if not fits(np.vstack([pos[f] for f in c]), max_w, max_h)]
    return forest, pos, len(best_cuts), len(still)


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

    forest, pos, bad, used = search(pm, G, trials=trials, seed=seed,
                                    verbose=verbose, max_size=max_size)
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
