"""Thickness-aware fold geometry for corrugated board.

Two panels of thickness t meeting at interior angle theta must lose material
at the crease or they cannot close. With bend angle alpha = 180 - theta, each
cut face rotates alpha/2 from the sheet normal and sweeps t*tan(alpha/2), so
the relief opening is

    w = 2 * d * tan(alpha/2)

where d is the depth to the effective pivot. This is exactly the sheet-metal
bend deduction at zero inside radius, and it matches the "spacing substantially
twice the thickness" specified for 90-degree V-grooved corrugated in
US 5,427,309.

Two corrections theory doesn't give you:

  * d is not t. With the far liner surviving as the hinge the board pivots
    about that liner's mid-plane, so d = t - liner/2 (~5% narrower).
  * Corrugated crushes. Flat crush resistance is 0.1-0.5 MPa against plywood's
    5-10 MPa, so the board squashes into the channel and the theoretical width
    is conservative. Folding-carton practice uses 1.5t channels where theory
    demands 2.0t. Hence the crush factor, default 0.75. This one is empirical.
"""
from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union, nearest_points

from .material import Material
from .unfold import Island, Fold


def relief_width(bend: float, mat: Material) -> float:
    """Opening width for a fold of signed bend angle (radians).

    Adaptive by default, from the derivation at the top of this module. A fixed
    width is offered because the derived one is the width the *board* needs,
    and sometimes you are cutting to a width the *machine* wants instead.
    """
    a = abs(float(bend))
    if a <= 1e-9:
        return 0.0
    if not mat.relief_adaptive:
        return min(float(mat.relief_fixed), mat.max_relief)
    if a >= np.pi - 1e-6:                 # folded flat back on itself
        return mat.max_relief
    return mat.crush * 2.0 * mat.pivot_depth * np.tan(a / 2.0)


def fold_intervals(length: float, mat: Material) -> list[tuple[float, float]]:
    """Cut intervals along a fold of the given length, as (start, end) in mm.

    Shared by both fold styles on purpose. A thin score and a wide relief slot
    are the same decision - how much of this bend line is removed and how much
    is left holding it - so one set of controls governs both. Before this the
    slot was segmented by its own bridge_len/bridge_every pair, which no UI
    exposed, so changing the perforation settings appeared to do nothing on
    any model whose folds were wide enough to become slots.

    Centred so both ends of the fold keep material.
    """
    L = float(length)
    if L <= 0.0:
        return []
    floor = min(mat.min_connect, L)
    if L <= floor + 1e-9:
        return []                         # too short to cut and still hold

    cut = max(float(mat.perf_cut), 1e-3)
    tie = mat.perf_tie
    period = cut + tie
    if period <= 1e-9:
        return []

    n = max(1, int(round((L - tie) / period)))
    span = n * period - tie               # a tie at each end
    start = (L - span) / 2.0

    out = []
    x = start
    for _ in range(n):
        s, e = max(0.0, x), min(L, x + cut)
        if e - s > 1e-6:
            out.append((s, e))
        x += period

    # Hold back min_connect of material overall. The ratio is a fraction, so on
    # a short fold it leaves a proportionally small - and absolutely tiny -
    # amount holding the joint together.
    total = sum(e - s for s, e in out)
    allowed = max(0.0, L - floor)
    if total > allowed + 1e-9 and total > 0.0:
        k = allowed / total
        out = [(( s + e) / 2.0 - (e - s) * k / 2.0,
                (s + e) / 2.0 + (e - s) * k / 2.0) for s, e in out]
        out = [(s, e) for s, e in out if e - s > 1e-6]
    return out


def flute_warning(mat: Material) -> str:
    """Perforation period against the flute pitch.

    Kept as advice rather than a clamp. Silently rescaling the user's dash
    length to pitch/3 made the control look broken - anything over 2 mm came
    back as 1.08 mm - and the constraint is real but not absolute: it matters
    for a fine score, far less for a slot with long bridges.
    """
    period = mat.perf_cut + mat.perf_tie
    limit = mat.pitch / 3.0
    if period > limit + 1e-9:
        return (f"perforation period {period:.1f}mm exceeds a third of the "
                f"{mat.pitch:.1f}mm flute pitch ({limit:.1f}mm) - consecutive "
                f"flutes will be treated very differently")
    return ""


def perf_segments(a: np.ndarray, b: np.ndarray, mat: Material,
                  phase: float = 0.0,
                  min_connect: float | None = None
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dash a fold line into cut/tie segments, centred so both ends get a tie."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-9:
        return []
    if min_connect is not None:           # bridge remnants ask for no floor
        m2 = replace(mat, min_connect=float(min_connect))
    else:
        m2 = mat
    d = (b - a) / L
    return [(a + s * d, a + e * d) for s, e in fold_intervals(L, m2)]


@dataclass
class FoldPlan:
    fold: Fold
    width: float                # relief opening, mm (0 if perf-only)
    style: str                  # "perf" | "relief"
    slot: Polygon | None        # material to remove (bridges already subtracted)
    perf: list                  # [(p,q), ...] dashed segments on remaining material
    warn: str = ""

    @property
    def is_mountain(self) -> bool:
        return self.fold.bend > 0


def _strip(a: np.ndarray, b: np.ndarray, width: float,
           round_ends: bool = False) -> Polygon:
    """Strip of given width centred on segment a-b.

    Square ends by default so a slot does not overhang the fold it relieves.
    Rounded ends are offered because a square internal corner concentrates
    stress and is where a fold starts tearing - and the beam dwells there
    anyway, so the corner is never truly sharp in the board.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = width / 2.0
    if not round_ends:
        return LineString([a, b]).buffer(r, cap_style=2, join_style=2)

    # Pull the ends in by the radius first. A round cap is drawn *beyond* the
    # line end, so buffering the full segment would push the slot width/2 past
    # the fold at each end and eat into the panels either side - a 50 mm fold
    # became a 54 mm slot.
    L = float(np.linalg.norm(b - a))
    if L <= 2.0 * r + 1e-9:               # too short to inset: a disc will do
        return LineString([a, b]).centroid.buffer(min(r, L / 2.0))
    d = (b - a) / L
    return LineString([a + d * r, b - d * r]).buffer(r, cap_style=1, join_style=1)


def _slot(a: np.ndarray, b: np.ndarray, width: float, mat: Material) -> Polygon:
    """The material a relief fold removes: one strip per cut interval.

    Built from the intervals rather than as one strip with bridges subtracted,
    so the radius lands on the ends of *every* segment. Rounding the whole slot
    and then cutting bridges out of it leaves square corners at each bridge -
    which is most of the corners, and exactly where a fold tears.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-9:
        return Polygon()
    d = (b - a) / L
    pieces = [_strip(a + s * d, a + e * d, width, round_ends=mat.round_cut_ends)
              for s, e in fold_intervals(L, mat)]
    pieces = [p for p in pieces if not p.is_empty]
    return unary_union(pieces) if pieces else Polygon()


def plan_folds(island: Island, mat: Material) -> list[FoldPlan]:
    """Decide, per fold, whether to simply perforate or to remove a relief slot."""
    plans = []
    flute = flute_warning(mat)
    for f in island.folds:
        w = relief_width(f.bend, mat)
        warn = flute
        if w > mat.max_relief:
            warn = (f"relief {w:.1f}mm exceeds max {mat.max_relief:.1f}mm at "
                    f"{np.degrees(abs(f.bend)):.0f} deg - use a double score here")
            w = mat.max_relief

        if w < mat.relief_threshold:
            plans.append(FoldPlan(f, w, "perf", None, perf_segments(f.a, f.b, mat), warn))
            continue

        slot = _slot(f.a, f.b, w, mat)
        if slot.is_empty:
            plans.append(FoldPlan(f, w, "perf", None, perf_segments(f.a, f.b, mat), warn))
            continue

        # perforate only where material survives, i.e. across the bridges
        line = LineString([f.a, f.b])
        remaining = line.difference(slot.buffer(1e-9))
        perf = []
        parts = getattr(remaining, "geoms", [remaining])
        for seg in parts:
            if seg.is_empty or seg.length < 1e-6:
                continue
            cs = np.asarray(seg.coords)
            # No min_connect floor here. These remnants ARE the bridges - the
            # material the floor exists to protect - and they are shorter than
            # it by design. Applying it would delete every dash on them, which
            # is how the PERF layer vanished from the SVG entirely.
            perf.extend(perf_segments(cs[0], cs[-1], mat, min_connect=0.0))
        plans.append(FoldPlan(f, w, "relief", slot, perf, warn))
    return plans


def apply_relief(outline, plans: list[FoldPlan], grid: float = 1e-9):
    """Subtract every relief slot from an island outline."""
    slots = [p.slot for p in plans if p.slot is not None and not p.slot.is_empty]
    if not slots:
        return outline
    cut = shapely.set_precision(unary_union(slots), grid)
    return shapely.set_precision(outline, grid).difference(cut)


def corner_relief(body, cut, kerf: float, overcut: float = 0.0,
                  grid: float = 1e-9):
    """Sever the sharp notches the kerf offset cannot reach into.

    Growing the part by kerf/2 is right along an edge and at a convex corner,
    but a notch cut into the material is an inner corner: the offset fills it,
    and the cut stops short of the apex by exactly (kerf/2) / sin(theta/2).
    That is 0.30 mm at a 60 degree notch, 0.86 mm at 20, 1.72 mm at 10 and
    8.59 mm at 2 - and the join style makes no difference whatsoever, because
    mitre, bevel and round only apply to *outer* corners. Measured on a
    pikachu net: 634 notches, median loss 0.15 mm but worst case 13.09 mm of
    material left uncut at the apex. On 3 mm board that is a fold that will
    not fold, which is the opposite of what a sharp bend needs.

    So the apex is reached explicitly, with a slit one beam wide run down the
    notch bisector from the offset boundary to the apex, and `overcut` mm past
    it. The direction comes from the nearest point on the cut boundary rather
    than from angle arithmetic - that point IS the offset apex, so the segment
    between them is the bisector, and it needs no reasoning about winding or
    which side the material lies on.

    Only notches losing more than a whole kerf are touched; below that the
    beam itself already covers the apex.
    """
    if kerf <= 0 or cut.is_empty or body.is_empty:
        return cut
    slits = _relief_slits(body, cut, kerf, overcut)
    if not slits:
        return cut

    out = _cut_slits(cut, slits, kerf, grid)
    if _piece_count(out) <= _piece_count(cut):
        return out

    # A slit through a near-pinch severs the part - the neck was only held
    # together by the material the kerf offset added, which is exactly what
    # this removes. Relief must never break a part into two, so where that
    # would happen the notch keeps its uncut apex and narrow_necks still warns
    # about the pinch. Re-applied one at a time, since only some are at fault.
    out = cut
    for s in slits:
        trial = _cut_slits(out, [s], kerf, grid)
        if _piece_count(trial) <= _piece_count(out):
            out = trial
    return out


def _relief_slits(body, cut, kerf: float, overcut: float) -> list:
    """One bisector segment per notch the beam cannot reach into."""
    coords = []
    for ring in _rings(body):
        coords.extend(list(ring.coords)[:-1])
    if not coords:
        return []
    pts = shapely.points(np.asarray(coords, dtype=float))
    bnd = cut.boundary
    shapely.prepare(cut)
    # Vectorised, because nearly every vertex is an ordinary one sitting the
    # offset distance away: doing this per vertex in Python cost more than the
    # whole rest of the core suite.
    keep = shapely.contains(cut, pts) & (shapely.distance(bnd, pts) > kerf)
    slits = []
    for i in np.nonzero(keep)[0]:
        v = coords[int(i)]
        near = nearest_points(bnd, pts[int(i)])[0]
        dx, dy = v[0] - near.x, v[1] - near.y
        n = (dx * dx + dy * dy) ** 0.5
        if n < 1e-12:
            continue
        slits.append(LineString([(near.x, near.y),
                                 (v[0] + dx / n * overcut,
                                  v[1] + dy / n * overcut)]))
    return slits


def _cut_slits(cut, slits: list, kerf: float, grid: float):
    slot = unary_union(slits).buffer(kerf / 2.0, cap_style=2, join_style=2)
    out = shapely.set_precision(
        shapely.set_precision(cut, grid).difference(
            shapely.set_precision(slot, grid)), grid)
    return _polygonal(out)


def _polygonal(geom):
    """Keep only the areal part of a difference.

    A slit that lands exactly along an edge leaves a stray LineString behind
    and the result comes back as a GeometryCollection - which has no
    .exterior, so every consumer of Part.cut breaks on it. Measured: this
    happens on a real pikachu net, and on no mesh in the synthetic corpus, so
    it will not be caught by the shapes anyone tests with.
    """
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    if not polys:
        return Polygon()
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def _piece_count(geom) -> int:
    if geom.geom_type == "Polygon":
        return 1
    return sum(1 for g in getattr(geom, "geoms", [])
               if g.geom_type == "Polygon")


def _rings(geom):
    """Every boundary ring of a Polygon or MultiPolygon, holes included."""
    out = []
    for g in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        if g.is_empty:
            continue
        out.append(g.exterior)
        out.extend(g.interiors)
    return out


def offset_cut(geom, amount: float, grid: float = 1e-9):
    """Move the cut line off the part's own edge by `amount` mm.

    Positive is outward - kerf compensation, so the beam's far side lands on
    the drawn edge and the part comes out nominal size. Negative is inward,
    which is what thick board often wants: two panels meeting at a fold have
    to share the same edge, and at 3 mm they collide unless each is cut back
    a little.

    Zero by default. At costume tolerance neither compensation is needed, and
    an outward offset is what makes a cut stop short of a sharp notch
    (corner_relief) - so paying that complexity for 0.15 mm is a poor trade
    unless the user asks for it.

    Interior holes move the opposite way automatically, because an outward
    offset of the part is an inward offset of a hole boundary. Mitre joins
    keep corners sharp rather than rounding them off.
    """
    if amount == 0.0 or geom.is_empty:
        return geom
    g = geom.buffer(amount, join_style=2, mitre_limit=8.0)
    return shapely.set_precision(g, grid)
