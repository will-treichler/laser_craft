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
import math
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
    flat: bool = False      # a flat piece, cut as it is (`flat_island`)

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


def _edge_dir(ring, a: int, b: int) -> int:
    """+1 if the ring traverses a -> b, -1 if b -> a."""
    n = len(ring)
    return 1 if ring[(ring.index(a) + 1) % n] == b else -1


class _Lookup:
    """Two dicts read as one, newest first. Cheaper than merging them when the
    first is a handful of entries and the second is the whole net."""

    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __getitem__(self, k):
        try:
            return self.a[k]
        except KeyError:
            return self.b[k]


def place_across(pm: PolyMesh, tree: nx.Graph, pos: dict, parent: int,
                 child: int) -> np.ndarray:
    """Where `child` lands, given `parent` is already placed at pos[parent].

    The one piece of real geometry in the unfolder, so it lives in exactly one
    place: `unfold_tree` walks a whole component with it and `place_branch`
    moves a single branch with it, and neither can drift from the other.
    """
    ev = tree[parent][child]["verts"]
    pr, cr = pm.faces[parent], pm.faces[child]
    if ev[0] not in pr or ev[1] not in pr or ev[0] not in cr or ev[1] not in cr:
        raise ValueError(f"dual arc {parent}-{child} names verts not on both faces")

    P2 = pos[parent][pr.index(ev[0])]
    Q2 = pos[parent][pr.index(ev[1])]
    d = Q2 - P2
    d = d / np.linalg.norm(d)
    e = np.array([-d[1], d[0]])

    # Which side of the seam the child goes is a combinatorial fact, not
    # something to infer from the geometry. Two faces of a consistently wound
    # mesh traverse their shared edge in OPPOSITE directions, so _face_local
    # hands each of them a signed h that already points away from the other;
    # placing both with k = +1 puts them on opposite sides. Reading the two
    # directions against each other rather than trusting the mesh globally
    # also absorbs a locally flipped face.
    #
    # The sums of signed offsets this replaced were a centroid test, not a
    # side test. A concave ring can have its vertex sum on the far side of its
    # own seam, and the child was then placed on top of its parent - which is
    # not merely an overlap but a REFLECTION, the mirror image of the true
    # face, and it will not fold up. isometry_error cannot see it because a
    # reflection keeps every edge length. Measured on pikachu at 150 mm: 5 of
    # 277 faces mirrored, 7 folds back-folded, and the spurious overlaps drove
    # repair to 32 extra cuts and 15 confetti islands of one or two faces.
    t, h, L = _face_local(pm, child, ev[0], ev[1])
    k = -1.0 if _edge_dir(pr, *ev) == _edge_dir(cr, *ev) else 1.0
    return P2 + np.outer(t, d) + k * np.outer(h, e)


def place_branch(pm: PolyMesh, tree: nx.Graph, pos: dict, parent: int,
                 child: int, members) -> dict:
    """Place `child` and everything beyond it, leaving the rest of pos alone.

    `members` confines the walk, which is what stops it turning round and
    re-placing the side it came from. Welding a 5-face splinter onto a
    250-face island then costs 5 placements instead of 255 - the difference
    between a search that can afford to score its candidates and one that
    cannot.
    """
    sub = tree.subgraph(set(members) | {parent})
    out = {child: place_across(pm, tree, pos, parent, child)}
    # Look in the branch first, fall back to the caller's placement for the
    # one node outside it. Copying the whole net's positions per call would
    # cost more than the placement it is helping with.
    view = _Lookup(out, pos)
    for a, b in nx.bfs_edges(sub, child):
        if b == parent:
            continue
        out[b] = place_across(pm, tree, view, a, b)
    return out


def unfold_tree(pm: PolyMesh, tree: nx.Graph, root: int) -> dict[int, np.ndarray]:
    """Place every face of one connected dual-tree component into the plane."""
    ring = pm.faces[root]
    t, h, _ = _face_local(pm, root, ring[0], ring[1])
    pos: dict[int, np.ndarray] = {root: np.column_stack([t, h])}
    for parent, child in nx.bfs_edges(tree, root):
        pos[child] = place_across(pm, tree, pos, parent, child)
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


def mirrored_faces(pm: PolyMesh, pos: dict[int, np.ndarray]) -> list[int]:
    """Faces whose placement is a reflection rather than a rigid motion.

    The companion to isometry_error, which is blind to this: a reflection
    preserves every edge length, so a mirrored face measures perfect and is
    still the wrong part - it will not fold up. Signed area is the test. Each
    face's own winding fixes the handedness of the frame _face_local builds,
    so a placement that preserves orientation keeps the sign that frame gives
    it.
    """
    bad = []
    for f, p in pos.items():
        ring = pm.faces[f]
        t, h, _ = _face_local(pm, f, ring[0], ring[1])
        if np.sign(_signed_area(np.column_stack([t, h]))) !=                 np.sign(_signed_area(p)):
            bad.append(f)
    return bad


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


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
    # Every face, not just the ones with a surviving arc. A face whose edges
    # are ALL force-cut is isolated, and building W from edges alone dropped
    # it - out of the spanning tree, out of _forest_positions, and finally out
    # of build_islands with a KeyError. It is a perfectly good one-face island.
    # Invisible until a constraint set is complete enough to isolate a face,
    # which is precisely what replaying an optimiser's cuts does.
    W.add_nodes_from(G)
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


def _trial(pm: PolyMesh, G: nx.Graph, rng) -> tuple:
    """One randomised restart: a direction, a maximum spanning tree, its
    placement and its overlaps. The unit both search and optimise work in."""
    c = rng.normal(size=3)
    c /= np.linalg.norm(c)
    W = weighted_dual(pm, G, c, rng=rng)
    T = nx.maximum_spanning_tree(W)
    pos, _ = _forest_positions(pm, T)
    return T, pos, overlapping_pairs(T, pos)


def mend(pm: PolyMesh, G: nx.Graph, forest: nx.Graph, pos: dict,
         bad: set, splinter_width: float = 0.0, neck_width: float = 0.0,
         weights=None) -> tuple:
    """Everything between a raw tree and a net that could be cut: resolve the
    overlaps, then put back the cuts that stopped being necessary.

    Returns (forest, pos, bad, extra_cuts, rewelds). Bed splitting is
    deliberately NOT here - it depends on the bed and this does not, so a
    caller comparing candidate trees can mend them all and split once.
    """
    extra = welds = 0
    if bad:
        forest, pos, bad, extra = repair(pm, forest, bad,
                                         splinter_width=splinter_width,
                                         weights=weights)
        forest, pos, welds = reattach(pm, G, forest, pos,
                                      neck_width=neck_width)
        bad = overlapping_pairs(forest, pos)
    return forest, pos, bad, extra, welds


# Roughly how many island-packs a finalist stage may spend, and the ceiling
# on how many candidates that can buy. Packing cost is close to linear in the
# island count, so dividing by it spends about the same effort on a 2-island
# model as on a 50-island one - and spends it where it buys something, since
# that is exactly where the quick score has nothing to say.
FINALIST_WORK = 60
FINALIST_CAP = 24

# Bed-split sizes the slider adds, as fractions of the island limit, and the
# slider position each starts at: a fraction f joins once bias >= (1-f)/0.6,
# so 85% from 0.25, 70% from 0.5, 50% from 0.84. Splitting below the bed is
# the only lever that actually makes pieces smaller - lowpoly_dome split to a
# smaller limit went 4 islands / 4 sheets to 15 / 3 - and they are tried on
# the best few finalists only, because each one is another pack.
SPLIT_FRACTIONS = (0.85, 0.7, 0.5)
SPLIT_FINALISTS = 3


def split_fractions(bias: float) -> list:
    return [1.0] + [f for f in SPLIT_FRACTIONS if (1.0 - f) <= 0.6 * bias + 1e-9]


def optimise(pm: PolyMesh, G: nx.Graph, mat, trials: int = 60, seed: int = 0,
             max_size: tuple | None = None, budget: int = 10,
             finalists: int = 6, weights=None, splinter_width: float = 0.0,
             neck_width: float = 0.0, count_sheets=True, progress=None,
             verbose: bool = False, board_bias: float = 0.0):
    """Choose a tree by what the finished net costs, not by how it starts.

    `board_bias` is the slider (0 = large islands, today's behaviour; 1 =
    less board): it sets the weights when none are given
    (`cost.board_weights`) and adds smaller bed-split sizes for the best
    finalists (`split_fractions`). The sheet term blends from the whole-sheet
    count at 0 to sheet-equivalents at 1 (the last sheet counted only as far
    as it is used, after packing it low). Measured: scoring sheet-equivalents
    at 0 too traded islands for offcut on the default - icosphere2 8 -> 12
    islands, pikachu 8 -> 9 - and the user's default is large islands.


    `search` ranks candidates by how many faces overlap, and only among the
    trees that overlap not at all. On the models where that matters it does
    neither job: pikachu has no clean tree in 60 restarts, so the scoring
    branch never ran, and the tie-break it fell back on is **anti-correlated**
    with the answer - spearman -0.3 to -0.46 against the finished cost. The
    fewest-overlaps tree costs **+74% to +114%** more than the best of the same
    24 candidates. Many overlaps usually means the net wraps densely, and a
    dense wrap consolidates into a few large islands once it has been mended;
    a tree that starts tidy is often one that ends up in fragments.

    So the ranking has to happen after `mend`, and it does: the quick cost of
    the mended net ranks candidates at spearman **+0.97** on pikachu and +0.73
    on icosphere2, and picks the best of 16 outright. There is no cheaper
    signal - that was looked for and `len(bad)` is the one that was there.

    Which makes `budget` the whole design. Mending is what costs (1.7 s a
    candidate on pikachu, after the incremental rewrite took it down from 17),
    so only candidates that actually need mending are counted against it;
    clean trees are nearly free and every restart up to `trials` gets one.
Candidates are ranked BEFORE the bed split, which looks wrong and was
    measured both ways. A torus unfolds to a handful of islands and leaves the
    splitter to make 53 of them, so ranking a pre-split net there is ranking
    almost nothing - the obvious fix is to rank after a cheap split, and it
    costs 30% more time across the corpus and changes nothing it was meant to
    (torus stayed at 23 sheets, its repair cuts went 6 to 15). Kept cheap.

    The finalists ARE split properly and scored with the full cost, which the
    quick tier cannot match: it bounds splinters rather than measuring them,
    never looks at necks, and - until the packer got fast enough - could not
    see sheets at all.

    **Sheets are counted by actually packing the finalists.** Both cheap
    estimates were tried and neither ranks: area is conserved so it is the
    same constant for every candidate, and shelving the minimum-area
    rectangles reads [3, 4, 4, 4] against a real [3, 2, 2, 2] on pikachu,
    because rectangles cannot see nesting and these nets nest.

    `raster_pack` alone rather than `pack`, because it gives the SAME count -
    23, 13 and 2 sheets against pack's 23, 13 and 2 on torus, annulus and
    pikachu - for 2x to 12x less time, and because `pack` runs it anyway, so
    the final reported number can only be this one or better. A finalist then
    costs 10.6 s on a torus instead of 22.1, which is what makes six of them
    affordable, and six is what annulus needs: its best net (11 sheets) ranks
    fifth on the quick score and never reached a three-deep shortlist.

    **How many finalists is decided by how much a pack costs**, which is the
    island count and nothing else. On a small model the quick score has almost
    nothing to separate candidates by - a 100 mm cylinder scored every one of
    them 44.3 or 44.4 - so the shortlist is essentially arbitrary, and the
    thing that actually differs (one sheet or two) is invisible to it. There
    the packs are cheap and the answer is to pack far more of them: 2 islands
    buys 24 finalists. A 53-island torus pays ten seconds a pack and gets the
    floor of six.

    `finalists` is that floor and `FINALIST_WORK` the numerator. Deliberately
    a function of the ISLAND COUNT rather than of a clock: a time budget would
    make the answer depend on the machine, and a net that changes when you run
    it somewhere else is not one you can hand to anybody.

    `count_sheets=False` turns it off, which is what the term's absence used
    to be.

    `progress(done, total, note) -> bool` is called once per candidate and
    returning False stops the search, which then answers with the best it has.
    A minute of silence is indistinguishable from a hang.
    """
    from .cost import score_islands, board_weights

    bias = min(1.0, max(0.0, float(board_bias)))
    weights = weights or board_weights(bias)
    if splinter_width <= 0 and mat is not None:
        splinter_width = mat.splinter_width
    rng = np.random.default_rng(seed)

    def report(done, total, note):
        return progress is None or progress(done, total, note)

    cands, spent, used = [], 0, 0
    for i in range(trials):
        used = i + 1
        T, pos, bad = _trial(pm, G, rng)
        if bad and spent >= budget:
            # Out of mending budget, but the report still has to happen. This
            # used to `continue` straight past it, so once the budget was
            # spent the progress bar froze for every remaining trial - and
            # ESC, which is read through the same call, went deaf with it.
            if not report(i + 1, trials, f"searching trees {i + 1}/{trials}"):
                break
            continue
        if bad:
            spent += 1
        T, pos, bad, extra, welds = mend(pm, G, T, pos, bad,
                                         splinter_width=splinter_width,
                                         neck_width=neck_width,
                                         weights=weights)
        isl = build_islands(pm, G, T, pos)
        q = score_islands(isl, mat, w=weights, quick=True)
        cands.append((q.total, i, T, pos, bad, extra, welds))
        if verbose:
            print(f"  trial {i + 1}: {len(isl)} island(s), quick {q.total:.1f}")
        if not report(i + 1, trials, f"searching trees {i + 1}/{trials}"):
            break

    if not cands:
        return None

    cands.sort(key=lambda c: c[0])
    n_isl = max(1, len(build_islands(pm, G, cands[0][2], cands[0][3])))
    deep = int(min(FINALIST_CAP, max(finalists, FINALIST_WORK // n_isl)))
    if verbose:
        print(f"  {len(cands)} candidate(s), {n_isl} island(s) in the best -> "
              f"{deep} finalist(s)")
    def finish(T, pos, bad, extra, welds, i, frac):
        """Split one finalist at `frac` of the limit, pack it, score it."""
        splits = oversize = 0
        bed = []
        if max_size is not None:
            T, pos, splits, oversize, bed = split_to_fit(
                pm, T, pos, max_size[0] * frac, max_size[1] * frac)
            bad = overlapping_pairs(T, pos)
        isl = build_islands(pm, G, T, pos)
        sheets = sheet_eq = None
        n_over = 0
        if count_sheets and mat is not None:
            # Deferred, and it has to be: layout imports this module, so the
            # two can only meet inside a call. Packing throwaway copies, so
            # nothing the caller holds is moved by the measurement.
            from .layout import (build_parts, raster_pack, compact_last,
                                 sheet_equivalents)
            try:
                parts = build_parts(isl, mat, annotate=False)
                sheets, over = raster_pack(parts, mat)
                n_over = len(over)
                # Scored the way pack() will leave it: the last sheet low,
                # and only as much of it as is used.
                compact_last(parts, mat)
                sheet_eq, _off = sheet_equivalents(parts, mat)
            except Exception:                           # noqa: BLE001
                sheets = sheet_eq = None   # never worth failing a solve over
        term = None
        if sheets is not None and sheet_eq is not None:
            term = (1.0 - bias) * sheets + bias * sheet_eq
        full = score_islands(isl, mat, sheets=term, w=weights,
                             oversize=n_over)
        if verbose:
            print(f"  finalist (trial {i + 1}, split {frac:.0%}): "
                  f"{sheet_eq} sheet-eq, {full.report()}")
        return (T, pos, bad, dict(
            extra_cuts=extra, rewelds=welds, size_splits=splits,
            oversize=oversize, bed=bed, cost=full.total, terms=full.terms,
            trial=i + 1, sheets=sheets, sheet_eq=sheet_eq, split_frac=frac))

    fracs = split_fractions(board_bias) if max_size is not None else [1.0]
    n_fin = min(deep, len(cands))
    n_more = min(SPLIT_FINALISTS, n_fin) * (len(fracs) - 1)
    total = n_fin + n_more
    scored = []
    for rank, (_q, i, T, pos, bad, extra, welds) in enumerate(
            cands[:max(1, deep)]):
        scored.append((finish(T, pos, bad, extra, welds, i, 1.0),
                       (T, pos, bad, extra, welds, i)))
        if not report(rank + 1, total, f"scoring finalists {rank + 1}/{n_fin}"):
            break
    else:
        # The slider's other half: the best few finalists again at smaller
        # bed splits - smaller pieces, which pack denser. Ranked at full size
        # first, so only nets already worth having pay for another pack.
        done = len(scored)
        for (_res, raw) in sorted(scored, key=lambda s: s[0][3]["cost"])[
                :SPLIT_FINALISTS]:
            for frac in fracs[1:]:
                scored.append((finish(*raw, frac), raw))
                done += 1
                if not report(done, total,
                              f"trying smaller pieces {done - n_fin}/{n_more}"):
                    break
            else:
                continue
            break

    T, pos, bad, stats = min((s[0] for s in scored),
                             key=lambda r: r[3]["cost"])
    stats["trials_used"] = used
    stats["candidates"] = len(cands)
    stats["mended"] = spent
    return T, pos, bad, stats


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
        T, pos, bad = _trial(pm, G, rng)
        if not bad:
            if max_size is None:
                if verbose:
                    print(f"  clean unfold on trial {i + 1}")
                return T, pos, set(), i + 1
            seen_clean += 1
            _, _, ncuts, over, _ = split_to_fit(pm, T, pos, *max_size,
                                                trials=3)
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


def _face_area(pm: PolyMesh, f: int) -> float:
    """Flat area of one face, by shoelace on its 3D ring projected to its own
    plane - which is exact, because the face is planar. Unfolding is a rigid
    motion, so this is also its area anywhere in the net."""
    r = np.asarray([pm.vertices[v] for v in pm.faces[f]], dtype=float)
    # Newell: |sum of cross products| / 2 is the area of any planar polygon in
    # 3D, with no need to pick a projection plane and no risk of picking one
    # the polygon is edge-on to.
    n = np.cross(r - r.mean(0), np.roll(r, -1, axis=0) - r.mean(0)).sum(0)
    return 0.5 * float(np.linalg.norm(n))


def _side_areas(forest: nx.Graph, face_area: dict) -> dict:
    """For every arc of the forest, the face area on each side of it.

    One rooted DFS per component gives all of them at once. Removing an arc of
    a tree always yields exactly two pieces, and the piece below the child is
    just its subtree - so the areas both cuts would produce are a post-order
    sum, O(n) for the whole forest rather than O(n) per candidate arc.
    """
    out = {}
    for comp in nx.connected_components(forest):
        sub = forest.subgraph(comp)
        root = next(iter(comp))
        total = sum(face_area[f] for f in comp)
        below = {f: face_area[f] for f in comp}
        # A DFS preorder lists a parent's arc before any of its children's, so
        # walking it backwards finalises every child before its parent is used.
        for parent, child in reversed(list(nx.dfs_edges(sub, root))):
            below[parent] += below[child]
        for parent, child in nx.dfs_edges(sub, root):
            out[(min(parent, child), max(parent, child))] = (
                below[child], total - below[child])
    return out


def repair(pm: PolyMesh, forest: nx.Graph, overlaps: set, max_cuts: int = 500,
           splinter_width: float = 0.0, weights=None):
    """Cut tree arcs until nothing overlaps, preferring cuts that leave
    islands somebody can actually hold.

    Greedy set cover over the conflicting face-pair paths: every overlap is
    resolved by cutting SOME arc on the path between the two faces, so the
    paths are the sets and the arcs are the elements. Provably terminates - in
    the worst case every face becomes its own island.

    With `splinter_width` set this becomes the WEIGHTED greedy, choosing by
    coverage per unit cost rather than coverage alone. That matters more than
    it sounds: repair is where pikachu's splinters come from - 25 extra cuts
    against 0-1 from bed splitting - precisely because the unweighted rule has
    no idea what the islands it leaves behind look like, and will happily slice
    a two-face tab off the edge of a panel if that tab happens to lie on three
    conflicting paths.

    The cost of an arc is what cutting it does to the net: one more island, a
    new seam twice the arc's length, and - dominant - whatever splinter
    penalty the two halves incur that the whole did not. Judged on the
    equal-area circle, which is an upper bound on inscribed width, so this
    sees a splinter it is about to make but can miss one it already has. A
    guide for choosing between arcs, never a verdict on the result.

    Coverage per unit cost is the standard weighted set-cover greedy and keeps
    its logarithmic guarantee; taking cheapest-first instead would resolve
    nothing, and coverage-first is what we already have.
    """
    from .cost import Weights, area_width, splinter_penalty

    forest = forest.copy()
    w = weights or Weights()
    cuts = 0
    pos, _ = _forest_positions(pm, forest)
    bad = overlaps

    face_area, ref = {}, 1.0
    if splinter_width > 0:
        face_area = {f: _face_area(pm, f) for f in forest}
        ref = math.sqrt(max(sum(face_area.values()), 1e-9))

    while cuts < max_cuts:
        if not bad:
            # The fast path below can only ever be optimistic, so "nothing
            # overlaps any more" is not a conclusion it is allowed to reach on
            # its own - the loop exits through a real measurement or not at all.
            pos, _ = _forest_positions(pm, forest)
            bad = overlapping_pairs(forest, pos)
            if not bad:
                break
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

        if splinter_width <= 0:
            pick = max(cover, key=cover.get)
        else:
            sides = _side_areas(forest, face_area)
            best, pick = None, None
            for key, n in cover.items():
                a_below, a_above = sides.get(key, (0.0, 0.0))
                whole = splinter_penalty(area_width(a_below + a_above),
                                         splinter_width, w.splinter_power)
                made = (splinter_penalty(area_width(a_below), splinter_width,
                                         w.splinter_power)
                        + splinter_penalty(area_width(a_above), splinter_width,
                                           w.splinter_power))
                length = forest[key[0]][key[1]].get("length", 0.0)
                price = (w.island
                         + w.splinter * max(0.0, made - whole)
                         + w.seam * 2.0 * length / ref)
                value = n / price
                # Ties go to the arc that covers more, which is the rule this
                # degenerates to when every cut costs the same.
                if best is None or (value, n) > best:
                    best, pick = (value, n), key

        forest.remove_edge(*pick)
        cuts += 1
        # Cutting a tree arc moves NOTHING relative to anything in its own
        # half - the fact split_to_fit is already built on - so the overlap
        # set can only shrink, and only by the pairs that just landed in
        # different components, which overlapping_pairs excludes anyway.
        # Filtering is therefore the same answer as recomputing for a fraction
        # of the cost: 25 placements and 25 STRtrees on a pikachu net become
        # one or two.
        #
        # "Therefore" does a lot of work there, so it was measured: identical
        # on four of five corpus models, and on pikachu 8 of 124 cuts differed
        # by up to 6 pairs. _forest_positions RE-ROOTS every component, so the
        # coordinates afterwards are numerically different and GEOS decides a
        # borderline touch the other way. Hence the honest recompute at the top
        # of the loop - the filter may end the loop early, the measurement
        # decides whether it was right to.
        comp_of = {}
        for ci, comp in enumerate(nx.connected_components(forest)):
            for f in comp:
                comp_of[f] = ci
        bad = {(a, b) for a, b in bad if comp_of[a] == comp_of[b]}
    pos, _ = _forest_positions(pm, forest)
    return forest, pos, bad, cuts


def _neck_count(pm: PolyMesh, nodes, sub_pos, width: float) -> int:
    """Near-pinches in one placed component. Deferred import: layout owns the
    measurement and imports this module, so it cannot be imported at the top."""
    from .layout import narrow_necks
    import shapely
    from shapely.ops import unary_union
    polys = []
    for f in nodes:
        q = Polygon(sub_pos[f])
        if not q.is_valid:
            q = q.buffer(0)
        polys.append(shapely.set_precision(q, 1e-9))
    return narrow_necks(unary_union(polys), width)


def _shrunk(pos, faces, shrink: float = 1e-6):
    """Face polygons, pulled in slightly so that sharing an edge is not an
    overlap. The shrink and the 1e-9 area threshold below are copied from
    `overlapping_pairs` deliberately: two answers to the same question that
    round differently would be worse than one slow answer."""
    out, live = [], []
    for f in faces:
        q = Polygon(pos[f])
        if not q.is_valid:
            q = q.buffer(0)
        q = q.buffer(-shrink * max(1.0, np.sqrt(abs(q.area))))
        if not q.is_empty:
            out.append(q)
            live.append(f)
    return out, live


def _cross_overlaps(pos_a: dict, a_faces, host, joined: tuple) -> bool:
    """Does any face of A land on any face of B?

    Only the CROSS pairs, because A was a clean island and B was a clean
    island and welding moves A rigidly - neither side can have started
    overlapping itself. `joined` is the pair the weld joins, which shares an
    edge by construction, and is excluded exactly as `overlapping_pairs`
    excludes a forest arc; it is the only cross pair that can be one, since a
    forest has exactly one arc across the cut it is putting back.

    `host` is a prepared (polys, faces, STRtree) triple, because the island
    being welded ONTO does not move between candidates - rebuilding its
    polygons for each of them was 3.1 s of a 4.9 s pikachu mend.
    """
    pb, fb, idx = host
    pa, fa = _shrunk(pos_a, a_faces)
    if not pa or not pb:
        return False
    for i, q in enumerate(pa):
        for j in idx.query(q):
            j = int(j)
            pair = (fa[i], fb[j])
            if pair == joined or pair[::-1] == joined:
                continue
            if q.intersection(pb[j]).area > 1e-9:
                return True
    return False


def reattach(pm: PolyMesh, G: nx.Graph, forest: nx.Graph,
             pos: dict[int, np.ndarray], max_rounds: int = 4,
             neck_width: float = 0.0):
    """Put back every cut that has stopped being necessary.

    Greedy set cover cuts an arc because some pair of faces overlapped THROUGH
    IT AT THAT MOMENT. Later cuts in the same run routinely resolve the same
    conflict from the other end, and nothing ever goes back to check - so the
    net keeps a seam, and often a whole island, that no longer pays for
    anything. That is where pikachu's confetti comes from: repair makes 25
    cuts and is not allowed to regret any of them.

    So: for every dual arc that currently separates two components, try
    welding along it and keep the weld if the merged island does not overlap
    itself. Smallest island first, because welding a splinter away is worth
    more than shortening a seam.

    Every accepted weld strictly improves the net - one fewer island, one
    fewer seam, and a splinter penalty that can only fall, because the merged
    island contains both halves. So this cannot make things worse and needs no
    cost function to tell it when to stop.

    `neck_width` guards the one way a weld could make things worse: an island
    that wraps around and rejoins itself through a near-single-point pinch is
    one piece topologically and two physically, and rejoining two halves is
    how such a pinch gets made. It refuses any weld whose merged island has
    more necks than its two halves had between them.

    It is OFF by default, because it was measured and does not pay. On the
    whole corpus it changes only pikachu, and there it buys 3 necks (102 down
    to 99) for 3 splinters (2 up to 5) and four extra islands. Welding turns
    out to create very few new pinches - most of the rise from the 90 of the
    unwelded net is the same necks living in fewer, larger islands - so the
    guard mostly just refuses good welds. Kept, exposed and documented rather
    than deleted, because it is the right instrument if a model ever does tear
    there; the default is a statement about this corpus, not about the idea.

    It does NOT check the bed, and must not: before split_to_fit an island is
    routinely larger than a sheet - that is what split_to_fit is for - so a
    bed test here would reject nearly every weld on a big model. Rejoining
    more of the tree hands the splitter more freedom, not less, and the
    splitter is provably maximal for whatever tree it is given. So this runs
    before it. Returns (forest, pos, welds).

    Everything here is incremental, and it has to be. The straightforward
    version re-placed the entire merged component and re-intersected all of it
    for EVERY candidate arc: 17 s a call on a pikachu net, and the single
    reason the tree search could not afford to score its candidates with the
    cost function that ranks them correctly.
    """
    forest = forest.copy()
    pos = dict(pos)
    welds = 0

    # Component bookkeeping by hand rather than through networkx. `has_path`
    # and `node_connected_component` are both O(component) and were called
    # once per candidate arc, which was most of the rest of the old cost.
    comp_of, members = {}, {}
    for ci, comp in enumerate(nx.connected_components(forest)):
        members[ci] = set(comp)
        for f in comp:
            comp_of[f] = ci

    face_area = {f: _face_area(pm, f) for f in forest}
    area = {ci: sum(face_area[f] for f in c) for ci, c in members.items()}
    geo = {}            # comp -> (polys, faces, STRtree), dropped when it moves

    for _round in range(max_rounds):
        cands = []
        for a, b in G.edges:
            if forest.has_edge(a, b):
                continue
            ca, cb = comp_of.get(a), comp_of.get(b)
            if ca is None or cb is None or ca == cb:
                continue
            cands.append((min(area[ca], area[cb]), a, b))
        cands.sort(key=lambda t: t[0])

        made = 0
        for _size, a, b in cands:
            ca, cb = comp_of.get(a), comp_of.get(b)
            if ca is None or cb is None or ca == cb:
                continue        # an earlier weld this round already joined them
            # Move the SMALLER side onto the larger. The big island keeps the
            # coordinates it already had, so nothing else in the net shifts,
            # and the placement work is proportional to the splinter being
            # welded on rather than to the panel it is welded to.
            if len(members[ca]) > len(members[cb]):
                a, b, ca, cb = b, a, cb, ca
            moving, host = members[ca], members[cb]

            forest.add_edge(a, b, **G[a][b])
            try:
                new = place_branch(pm, forest, pos, b, a, moving)
            except ValueError:
                forest.remove_edge(a, b)
                continue

            hg = geo.get(cb)
            if hg is None:
                hp, hf = _shrunk(pos, host)
                hg = geo[cb] = (hp, hf, STRtree(hp) if hp else None)
            ok = not _cross_overlaps(new, moving, hg, (a, b))
            if ok and neck_width > 0:
                trial = dict(pos)
                trial.update(new)
                before = (_neck_count(pm, moving, trial, neck_width)
                          + _neck_count(pm, host, trial, neck_width))
                ok = _neck_count(pm, moving | host, trial, neck_width) <= before
            if not ok:
                forest.remove_edge(a, b)
                continue

            pos.update(new)
            geo.pop(cb, None)           # it has grown; its index is stale now
            geo.pop(ca, None)
            members[cb] = host | moving
            area[cb] += area[ca]
            for f in moving:
                comp_of[f] = cb
            del members[ca], area[ca]
            welds += 1
            made += 1
        if not made:
            break

    pos, _ = _forest_positions(pm, forest)
    return forest, pos, welds


def split_island(isl: Island, f0: int, f1: int, new_id: int, new_seam: int):
    """Split one island along the fold joining f0 and f1. Returns (a, b).

    The local counterpart to re-unfolding with a forced cut. Re-unfolding lets
    the solver rebuild the whole tree, so the faces freed by the cut usually
    reattach somewhere else entirely and the island count does not go up -
    correct, but not what someone dragging a seam in the 2D view expects. This
    detaches exactly the piece on the far side of that fold and leaves the rest
    of the net alone.

    An island's folds form a spanning tree over its faces, so every fold is a
    bridge: removing one always yields exactly two pieces, never zero or three.
    The fold becomes a matching pair of cuts sharing a new seam id, which is
    what pairs the glue tab with its partner.

    Returns None if those faces are not joined by a fold here - they may be
    adjacent in 3D but already cut apart in this net.
    """
    target = None
    G = nx.Graph()
    G.add_nodes_from(isl.faces)
    for fo in isl.folds:
        if target is None and {fo.f0, fo.f1} == {f0, f1}:
            target = fo
            continue
        G.add_edge(fo.f0, fo.f1)
    if target is None:
        return None

    comps = list(nx.connected_components(G))
    if len(comps) != 2:
        return None
    side_a, side_b = comps
    if target.f0 not in side_a:
        side_a, side_b = side_b, side_a

    def build(iid, members, keep_face):
        out = Island(id=iid)
        out.faces = {k: isl.faces[k] for k in sorted(members)}
        out.folds = [fo for fo in isl.folds
                     if fo is not target and fo.f0 in members and fo.f1 in members]
        out.cuts = [c for c in isl.cuts if c.face in members]
        out.cuts.append(Cut(target.a, target.b, keep_face, new_seam))
        return out

    return (build(isl.id, side_a, target.f0),
            build(new_id, side_b, target.f1))


def join_islands(a: Island, b: Island, pm: PolyMesh, G: nx.Graph,
                 f0: int, f1: int):
    """Weld island `b` onto `a` along the edge between faces f0 and f1.

    The exact inverse of split_island, and the local counterpart to
    re-unfolding with a forced fold. Re-unfolding would let the solver rebuild
    every tree, which moves the whole net - correct, but not what someone
    welding one seam in the 2D view expects. This moves `b` rigidly onto `a`
    and leaves everything else exactly where it was, so the global re-solve
    stays an explicit act.

    Where `b` goes is not a choice: f1 has exactly one placement that shares
    the edge with f0 and lies on the far side of it, and unfold_tree already
    knows how to find it. Compute that placement, then carry the rest of `b`
    with the rigid motion it implies. Two corresponding points fix a rotation
    and a translation; no reflection is possible, because both placements are
    orientation-preserving images of the same face.

    Returns None when the two are not adjacent across an uncut edge here - they
    may share an edge in 3D and already be in the same island, or be joined
    through a different path.

    The joined island CAN self-overlap: welding re-creates whatever the solver
    avoided by cutting there. That is visible rather than prevented -
    overlapping_pairs reports it and re-running the unfold is the way out.
    """
    if a is b or a.id == b.id:
        return None
    if f0 not in a.faces or f1 not in b.faces:
        return None
    if not G.has_edge(f0, f1):
        return None
    ev = G[f0][f1].get("verts")
    if ev is None:
        return None

    ring0 = pm.faces[f0]
    if ev[0] not in ring0 or ev[1] not in ring0:
        return None
    P2 = a.faces[f0][ring0.index(ev[0])]
    Q2 = a.faces[f0][ring0.index(ev[1])]

    # Where f1 belongs, by exactly the rule unfold_tree uses for a child.
    d = Q2 - P2
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None
    d = d / n
    e = np.array([-d[1], d[0]])
    t, h, _L = _face_local(pm, f1, ev[0], ev[1])
    k = -1.0 if _edge_dir(ring0, *ev) == _edge_dir(pm.faces[f1], *ev) else 1.0
    target = P2 + np.outer(t, d) + k * np.outer(h, e)

    # The rigid motion carrying b's current f1 onto that target, from the two
    # shared corners - the same two-point recovery the 2D overlay uses.
    ring1 = pm.faces[f1]
    i0, i1 = ring1.index(ev[0]), ring1.index(ev[1])
    q0, q1 = b.faces[f1][i0], b.faces[f1][i1]
    p0, p1 = target[i0], target[i1]
    u, v = q1 - q0, p1 - p0
    lu, lv = float(np.hypot(*u)), float(np.hypot(*v))
    if lu < 1e-12 or lv < 1e-12:
        return None
    c = float(u @ v) / (lu * lv)
    s = float(u[0] * v[1] - u[1] * v[0]) / (lu * lv)
    R = np.array([[c, -s], [s, c]])

    def move(pts):
        return (np.asarray(pts, dtype=float) - q0) @ R.T + p0

    out = Island(id=min(a.id, b.id))
    out.faces = dict(a.faces)
    for fi, ring in b.faces.items():
        out.faces[fi] = move(ring)

    # The two cuts that were this seam become the fold again. Drop them by the
    # face they sit on rather than by seam id: a hand-welded edge may never
    # have been a numbered pair.
    def is_seam(cut, face):
        if cut.face != face:
            return False
        ends = (np.allclose(cut.a, P2) and np.allclose(cut.b, Q2)) or \
               (np.allclose(cut.a, Q2) and np.allclose(cut.b, P2))
        return ends

    out.cuts = [c for c in a.cuts if not is_seam(c, f0)]
    for c in b.cuts:
        m = Cut(move([c.a, c.b])[0], move([c.a, c.b])[1], c.face, c.seam)
        if np.allclose(m.a, P2) and np.allclose(m.b, Q2) and c.face == f1:
            continue
        if np.allclose(m.a, Q2) and np.allclose(m.b, P2) and c.face == f1:
            continue
        out.cuts.append(m)

    out.folds = list(a.folds)
    for fo in b.folds:
        mv = move([fo.a, fo.b])
        out.folds.append(Fold(mv[0], mv[1], fo.bend, fo.f0, fo.f1))
    out.folds.append(Fold(P2, Q2, float(G[f0][f1].get("bend", 0.0)), f0, f1))
    return out


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
    # Plain floats, not numpy rows: the loop below is scalar arithmetic, and
    # indexing a numpy row per point was 90% of its time. Python floats ARE
    # float64 and the operations run in the same order, so the hull is
    # bit-identical (test_the_convex_hull_is_unchanged_by_the_fast_path).
    p = p.tolist()

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

    Returns (forest, pos, n_cuts, n_oversize, arcs). The arcs are reported and
    not merely counted because the optimiser has to tell a bed split apart from
    a repair cut: this is the good kind, landing on a real crease that already
    needs a glue flap, while a repair cut is the solver working around an
    overlap it could not otherwise resolve. Two very different things to weigh.
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
    return forest, pos, len(best_cuts), len(still), best_cuts


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


def cut_arcs(G_full: nx.Graph, forest: nx.Graph) -> list:
    """The cut set of a finished net, as sorted (v0, v1) mesh-vertex pairs.

    Every dual arc is either a fold or a cut, so the cuts are the complement of
    the fold forest - and that complement is a COMPLETE description of the net.
    Feed it back as `forced_cuts` and apply_constraints deletes exactly those
    arcs, leaving a graph that is already a forest; the spanning-tree step then
    has nothing left to choose and the seed stops mattering. That is what lets
    the optimiser hand its answer to the user as editable marks instead of as a
    number nobody can reproduce.

    Arcs deleted by an incoming constraint are included, because they are cuts
    too - the set has to describe the whole net, not the part of it this run
    happened to decide.
    """
    out = []
    for a, b in G_full.edges:
        if forest.has_edge(a, b):
            continue
        verts = G_full[a][b].get("verts")
        if verts is None:
            continue
        out.append((min(verts), max(verts)))
    return sorted(set(out))


# ------------------------------------------------------------ flat pieces

# A face may tilt this far from its piece's plane and the piece still count as
# flat. Steeper than this it is a wall, and a piece with walls is a solid -
# a box 3 mm thick is within any flatness tolerance and is still not a sheet.
FLAT_MAX_TILT_DEG = 45.0


def _plane_fit(pts: np.ndarray):
    """Least-squares plane: (centroid, unit normal, max |distance|)."""
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return c, n, float(np.abs((pts - c) @ n).max())


def flat_pieces(pm: PolyMesh, tol: float, G: nx.Graph | None = None) -> list:
    """Connected pieces of the mesh that can be cut exactly as they are.

    A piece is one connected component of the dual graph, so it shares no
    edge with anything else - nothing is ever cut off a larger part to make
    one. It is flat when every vertex lies within `tol` mm of one plane AND
    every face faces the same way within FLAT_MAX_TILT_DEG. The second test is
    what keeps a closed solid out: a 3 mm thick plate modelled as a box is
    within 3 mm of a plane, and its top, bottom and walls cannot all be laid
    on a sheet as one shape.

    Returns [(faces, centroid, normal, deviation)], normal pointing the way
    the faces do.
    """
    G = pm.dual_graph() if G is None else G
    cos_tilt = math.cos(math.radians(FLAT_MAX_TILT_DEG))
    out = []
    for comp in nx.connected_components(G):
        faces = sorted(comp)
        verts = sorted({v for f in faces for v in pm.faces[f]})
        if len(verts) < 3:
            continue
        c, n, dev = _plane_fit(pm.vertices[verts])
        if dev > tol:
            continue
        d = pm.normals[faces] @ n
        if d.mean() < 0:
            n, d = -n, -d
        if d.min() < cos_tilt:
            continue
        out.append((faces, c, n, dev))
    return out


def flat_island(pm: PolyMesh, faces, centroid, normal, iid: int) -> Island:
    """One flat piece as an island: every face projected onto the piece's
    plane, no folds, and its open boundary as cuts that pair with nothing.

    No tree, no search and no overlap test: one connected sheet facing one
    way projects onto its plane without folding over itself, so there is
    nothing to decide. The in-plane frame is right-handed about the normal,
    so every face keeps its winding (`mirrored_faces` stays empty). Exactly
    isometric for a piece that is exactly planar; within the tolerance it is
    the projection, which is what "cut it flat" means.
    """
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    P = np.column_stack([e1, e2])
    isl = Island(id=iid, flat=True)
    for f in faces:
        isl.faces[f] = (pm.vertices[pm.faces[f]] - centroid) @ P
    uses: dict = {}
    for f in faces:
        ring = pm.faces[f]
        for i in range(len(ring)):
            u, v = ring[i], ring[(i + 1) % len(ring)]
            uses.setdefault((min(u, v), max(u, v)), []).append(f)
    for (v0, v1), fs in uses.items():
        if len(fs) != 1:
            continue
        f = fs[0]
        ring = pm.faces[f]
        isl.cuts.append(Cut(a=isl.faces[f][ring.index(v0)],
                            b=isl.faces[f][ring.index(v1)], face=f, seam=-1))
    return isl


def _sub_mesh(pm: PolyMesh, keep: list) -> PolyMesh:
    """The faces `keep` as a mesh of their own, sharing pm's vertex array -
    so vertex ids, and every forced cut or fold, mean the same thing in it."""
    return PolyMesh(pm.vertices, [pm.faces[f] for f in keep],
                    [pm.source_faces[f] for f in keep])


def _renumber(islands: list, fmap: list, first_id: int) -> None:
    """Point a sub-mesh's islands back at the full mesh's face ids."""
    for k, isl in enumerate(islands):
        isl.id = first_id + k
        isl.faces = {fmap[f]: r for f, r in isl.faces.items()}
        for fd in isl.folds:
            fd.f0, fd.f1 = fmap[fd.f0], fmap[fd.f1]
        for c in isl.cuts:
            c.face = fmap[c.face]


def unfold(pm: PolyMesh, trials: int = 250, seed: int = 0,
           max_size: tuple[float, float] | None = None,
           forced_cuts: set = (), forced_folds: set = (),
           verbose: bool = False, splinter_width: float = 0.0,
           neck_width: float = 0.0, weights=None, mat=None,
           budget: int = 10, finalists: int = 6, count_sheets: bool = True,
           progress=None, board_bias: float = 0.0,
           flat_tol: float | None = None):
    """Full unfold: search, repair overlaps, then split anything too big.

    max_size is the island limit in mm, which should already have room
    subtracted for glue tabs and kerf - Blender's addon tests the bare island
    against the full page and then adds flaps afterwards, which is why its
    output sometimes still overflows.

    `splinter_width` (`Material.splinter_width`) turns on the cost-aware
    repair. It is a parameter rather than a Material because the core solver
    has no business knowing about flutes and kerfs, and because passing 0
    gives back the old unweighted behaviour exactly - which is what makes the
    two comparable on the bench.

    `mat` turns on the cost-driven tree search (`optimise`), which needs a
    whole Material because the cost function does - the bed, the splinter
    threshold, the board. Without it the old `search` runs, ranking candidates
    by overlap count, which is measurably the wrong way round; both paths are
    kept for exactly that comparison. `progress(done, total, note) -> bool`
    reports through the search and cancels it by returning False.

    `flat_tol` (mm, `Material.flat_tol`) turns on flat pieces: a connected
    piece within that distance of one plane is cut as the shape it is - one
    island, no folds, no seams - instead of being searched, and the rest of
    the mesh is unfolded as usual. None keeps the old behaviour exactly.
    """
    if flat_tol is not None:
        flats = flat_pieces(pm, flat_tol)
        if flats:
            return _unfold_with_flats(
                pm, flats, dict(
                    trials=trials, seed=seed, max_size=max_size,
                    forced_cuts=forced_cuts, forced_folds=forced_folds,
                    verbose=verbose, splinter_width=splinter_width,
                    neck_width=neck_width, weights=weights, mat=mat,
                    budget=budget, finalists=finalists,
                    count_sheets=count_sheets, progress=progress,
                    board_bias=board_bias))
    G_full = pm.dual_graph()
    G, notes = apply_constraints(G_full, forced_cuts, forced_folds)

    cost = candidates = mended = None
    split_frac = 1.0
    if mat is not None:
        forest, pos, bad, st = optimise(
            pm, G, mat, trials=trials, seed=seed, max_size=max_size,
            budget=budget, finalists=finalists, weights=weights,
            splinter_width=splinter_width, neck_width=neck_width,
            count_sheets=count_sheets, progress=progress, verbose=verbose,
            board_bias=board_bias)
        used, extra, welds = st["trials_used"], st["extra_cuts"], st["rewelds"]
        splits, oversize, bed = st["size_splits"], st["oversize"], st["bed"]
        cost = st["cost"]
        candidates, mended = st["candidates"], st["mended"]
        split_frac = st.get("split_frac", 1.0)
    else:
        forest, pos, bad, used = search(pm, G, trials=trials, seed=seed,
                                        verbose=verbose, max_size=max_size)
        forest, pos, bad, extra, welds = mend(pm, G, forest, pos, bad,
                                              splinter_width=splinter_width,
                                              neck_width=neck_width,
                                              weights=weights)
        splits = oversize = 0
        bed = []
        if max_size is not None:
            forest, pos, splits, oversize, bed = split_to_fit(pm, forest, pos,
                                                              *max_size)
            bad = overlapping_pairs(forest, pos)

    bed_arcs = sorted({tuple(sorted(G_full[a][b]["verts"]))
                       for a, b in bed if "verts" in G_full[a][b]})

    # seams are reported against the full dual so forced cuts still get tabs
    islands = build_islands(pm, G_full, forest, pos)
    report = dict(cut_arcs=cut_arcs(G_full, forest), bed_cut_arcs=bed_arcs,
                  trials_used=used, extra_cuts=extra, rewelds=welds,
                  size_splits=splits, split_frac=split_frac,
                  cost=cost, candidates=candidates, mended=mended,
                  islands=len(islands), unresolved=len(bad),
                  oversize_islands=oversize, notes=notes,
                  isometry_error=isometry_error(pm, pos))
    return islands, report


def _unfold_with_flats(pm: PolyMesh, flats: list, kw: dict):
    """unfold() with the flat pieces taken out first and laid down as they are.

    The rest is a mesh of its own (`_sub_mesh`), unfolded with every setting
    unchanged; its islands come back pointing at the full mesh's faces, and
    the flat islands are numbered after them. A flat piece has no seams, so
    seam numbers, the cut set and every repair statistic are the rest's alone.
    """
    flat_faces = {f for faces, *_ in flats for f in faces}
    rest = [f for f in range(len(pm.faces)) if f not in flat_faces]
    if rest:
        islands, report = unfold(_sub_mesh(pm, rest), **kw)
        _renumber(islands, rest, 0)
    else:
        islands, report = [], dict(
            cut_arcs=[], bed_cut_arcs=[], trials_used=0, extra_cuts=0,
            rewelds=0, size_splits=0, split_frac=1.0, cost=None,
            candidates=None, mended=None, unresolved=0, oversize_islands=0,
            notes=[], isometry_error=0.0)
    report = dict(report)
    worst = report["isometry_error"]
    for faces, c, n, _dev in flats:
        isl = flat_island(pm, faces, c, n, len(islands))
        islands.append(isl)
        worst = max(worst, isometry_error(pm, isl.faces))
        if kw.get("max_size") is not None:
            w, h = min_rect_size(np.vstack(list(isl.faces.values())))
            mw, mh = kw["max_size"]
            if not (min(w, h) <= min(mw, mh) and max(w, h) <= max(mw, mh)):
                report["oversize_islands"] = report.get("oversize_islands", 0) + 1
    report["islands"] = len(islands)
    report["isometry_error"] = worst
    report["flat_pieces"] = len(flats)
    report["flat_deviation"] = max(f[3] for f in flats)
    return islands, report
