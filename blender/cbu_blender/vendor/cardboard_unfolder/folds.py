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
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

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


def perf_segments(a: np.ndarray, b: np.ndarray, mat: Material,
                  phase: float = 0.0,
                  min_connect: float | None = None
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dash a fold line into cut/tie segments, centred so both ends get a tie.

    The corrugated-specific constraint: keep the dash period well under the
    flute pitch. At a 4 mm period on a 6.5 mm B-flute pitch, consecutive
    flutes get wildly different treatment - some ties land on a glue line
    (locally 3-ply solid, cracks), some over a void (hinges on nothing).
    Period <= pitch/3 means every flute sees ~3 dashes and behaves alike.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-9:
        return []
    floor = mat.min_connect if min_connect is None else float(min_connect)
    if L <= floor:
        return []                         # too short to cut and still hold

    cut, tie = mat.perf_cut, mat.perf_tie
    period = cut + tie
    limit = mat.pitch / 3.0
    if period > limit:                    # rescale, preserving the cut:tie ratio
        k = limit / period
        cut, tie, period = cut * k, tie * k, limit

    d = (b - a) / L
    n = max(1, int(round((L - tie) / period)))
    span = n * period - tie               # keep a tie at each end
    start = (L - span) / 2.0 + phase % period

    out = []
    x = start
    for _ in range(n):
        s, e = max(0.0, x), min(L, x + cut)
        if e - s > 1e-6:
            out.append((s, e))
        x += period

    # Hold back min_connect of material overall. The ratio alone is a fraction,
    # so on a short fold it leaves a proportionally small - and absolutely
    # tiny - amount holding the joint together.
    total = sum(e - s for s, e in out)
    allowed = max(0.0, L - floor)
    if total > allowed + 1e-9:
        k = allowed / total if total > 0 else 0.0
        shrunk = []
        for s, e in out:
            c = (s + e) / 2.0
            half = (e - s) * k / 2.0
            if half > 1e-6:
                shrunk.append((c - half, c + half))
        out = shrunk
    return [(a + s * d, a + e * d) for s, e in out]


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


def _bridges(a: np.ndarray, b: np.ndarray, width: float, mat: Material) -> Polygon:
    """Uncut ties spanning the relief slot. These are the actual hinge."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    n = max(1, int(np.floor(L / mat.bridge_every)))
    blen = min(mat.bridge_len, L / (2 * n + 1))       # never eat the whole fold
    if n * blen < mat.min_connect:        # the floor that stops small joints tearing
        blen = min(mat.min_connect / n, L / (2 * n + 1) * 2.0, L * 0.9 / n)
    pieces = []
    for i in range(n):
        c = L * (i + 0.5) / n
        s, e = max(0.0, c - blen / 2), min(L, c + blen / 2)
        pieces.append(_strip(a + s * d, a + e * d, width * 1.5))
    return unary_union(pieces)


def plan_folds(island: Island, mat: Material) -> list[FoldPlan]:
    """Decide, per fold, whether to simply perforate or to remove a relief slot."""
    plans = []
    for f in island.folds:
        w = relief_width(f.bend, mat)
        warn = ""
        if w > mat.max_relief:
            warn = (f"relief {w:.1f}mm exceeds max {mat.max_relief:.1f}mm at "
                    f"{np.degrees(abs(f.bend)):.0f} deg - use a double score here")
            w = mat.max_relief

        if w < mat.relief_threshold:
            plans.append(FoldPlan(f, w, "perf", None, perf_segments(f.a, f.b, mat), warn))
            continue

        strip = _strip(f.a, f.b, w, round_ends=mat.round_cut_ends)
        slot = strip.difference(_bridges(f.a, f.b, w, mat))
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


def kerf_compensate(geom, kerf: float, grid: float = 1e-9):
    """Grow the part by half a kerf so the cut lands on the drawn line.

    Interior holes shrink automatically because a negative offset of a hole
    boundary is an outward offset of the part. Mitre joins keep corners sharp
    rather than rounding them off.
    """
    if kerf <= 0:
        return geom
    g = geom.buffer(kerf / 2.0, join_style=2, mitre_limit=8.0)
    return shapely.set_precision(g, grid)
