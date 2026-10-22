"""Assemble islands into cuttable parts and pack them onto laser sheets.

Packing here is deliberately simple: rotate each part to its minimum-area
bounding box, then shelf-pack the boxes. True polygon nesting (no-fit
polygon, as in SVGnest) is worth roughly +20 points of material utilisation
on net-like geometry and is the obvious upgrade, but bbox shelving is 60
lines and gets a first model onto the machine.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiLineString, LineString, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .material import Material
from .unfold import Island, min_area_rect
from .folds import (plan_folds, apply_relief, apply_corner_holes, offset_cut,
                    corner_relief)
from .tabs import place_tabs, island_labels
from .fingers import place_fingers, apply_fingers, fingered_seams


@dataclass
class Part:
    island_id: int
    cut: object                        # Polygon | MultiPolygon, kerf-compensated
    perf_m: MultiLineString            # mountain fold dashes
    perf_v: MultiLineString            # valley fold dashes
    engrave: MultiLineString           # labels
    warnings: list = field(default_factory=list)
    sheet: int = -1
    xform: np.ndarray = field(default_factory=lambda: np.eye(3))
    # Which model this came from. Island ids restart at 0 per object, so
    # once several objects share the sheets the id alone is ambiguous.
    model: str = ""

    def apply(self, fn):
        self.cut = fn(self.cut)
        self.perf_m = fn(self.perf_m)
        self.perf_v = fn(self.perf_v)
        self.engrave = fn(self.engrave)

    def rotate(self, deg, origin=(0, 0)):
        self.apply(lambda g: affinity.rotate(g, deg, origin=origin))
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        ox, oy = float(origin[0]), float(origin[1])
        self._compose(np.array([[c, -s, ox - c * ox + s * oy],
                                [s, c, oy - s * ox - c * oy],
                                [0.0, 0.0, 1.0]]))

    def translate(self, dx, dy):
        self.apply(lambda g: affinity.translate(g, dx, dy))
        self._compose(np.array([[1.0, 0.0, dx],
                                [0.0, 1.0, dy],
                                [0.0, 0.0, 1.0]]))

    def _compose(self, M: np.ndarray) -> None:
        """Track where packing put this part.

        The geometry alone is not enough to show the packing anywhere else:
        the net UV layer needs the same rigid motion applied to the island's
        faces, or the 2D view shows a layout that has nothing to do with the
        sheets actually being cut - parts straddling sheet boundaries and
        ignoring the margins.
        """
        self.xform = M @ self.xform

    def place(self, pts: np.ndarray) -> np.ndarray:
        """Push island-space points through the packing transform."""
        return pts @ self.xform[:2, :2].T + self.xform[:2, 2]

    @property
    def bounds(self):
        return self.cut.bounds

    @property
    def size(self):
        x0, y0, x1, y1 = self.cut.bounds
        return (x1 - x0, y1 - y0)


def narrow_necks(geom, min_width: float) -> int:
    """How many places the part is thinner than min_width.

    Unfolded nets routinely wrap around and rejoin through a near-single-point
    pinch. The result is topologically one piece and physically two: in
    corrugated those necks tear while you are still peeling the part off the
    bed. Eroding by half the minimum width and counting the fragments finds
    them - if the part falls apart, it was only just holding together.
    """
    if geom.is_empty:
        return 0
    eroded = geom.buffer(-min_width / 2.0)
    if eroded.is_empty:
        return 0
    n = len(eroded.geoms) if eroded.geom_type == "MultiPolygon" else 1
    return max(0, n - 1)


def inscribed_width(geom, tol: float = 0.25) -> float:
    """Diameter of the largest circle that fits inside the part, in mm.

    The honest measure of "can I pick this up and glue a tab to it". Area is
    not: a 2 x 40 mm splinter and a 9 x 9 mm square have the same area and are
    not equally handleable, and it is the splinter that gets dropped, bent and
    sworn at. Nor is the bounding box, which says nothing about a C-shape.

    This is the FATTEST point, not the thinnest - the right question for "can
    I hold it", and deliberately not the same question as "is any part of it
    fragile", which is what narrow_necks answers. An L with 8 mm legs measures
    9.4 mm, because the inscribed circle at the inner corner is larger than
    either leg: the piece has somewhere to grip even though most of it is thin.

    Bisection on erosion - if buffering inward by r empties the shape then no
    circle of radius r fits in it. About a dozen buffers, which is cheap
    enough to score every island with.
    """
    if geom.is_empty:
        return 0.0
    x0, y0, x1, y1 = geom.bounds
    lo, hi = 0.0, max(x1 - x0, y1 - y0) / 2.0 + tol
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if geom.buffer(-mid).is_empty:
            hi = mid
        else:
            lo = mid
    return 2.0 * lo


def _mls(segs) -> MultiLineString:
    """Engraved polylines, minus the ones with no length.

    Reject a stroke that goes nowhere - NOT one that comes back to where it
    started. Testing `first != last` deleted every CLOSED glyph the stroke
    font has: 0, O, D and Q vanished from every label. A part stamped PIKA-0
    engraved as "PIKA-", and seam 10 engraved as "1" - which is not a missing
    annotation but a wrong one, and the builder has no way to know. Island 0
    of a pikachu net built 206 strokes and engraved 194.
    """
    out = []
    for s in segs:
        a = np.asarray(s, dtype=float)
        if len(a) < 2:
            continue
        if float(np.abs(np.diff(a, axis=0)).max()) <= 1e-9:
            continue                    # every point on top of the last
        out.append(LineString(a))
    return MultiLineString(out) if out else MultiLineString([])


def build_parts(islands: list[Island], mat: Material,
                annotate: bool = True) -> list[Part]:
    """Island -> Part: relief slots subtracted, fingers and tabs on, kerf applied.

    `annotate=False` builds the cut and nothing else - no perforation, no
    labels, no neck warnings - for a caller that only wants to know how the
    parts PACK. The optimiser packs every finalist to count its sheets, and
    labelling was a third of each build there while changing nothing the
    packer reads. The cut is the same geometry either way
    (`test_an_unannotated_part_has_exactly_the_annotated_cut`).
    """
    plans_by = {isl.id: plan_folds(isl, mat) for isl in islands}
    bodies = {isl.id: apply_relief(isl.outline(), plans_by[isl.id])
              for isl in islands}
    # Before the fingers, so a notch keeps its wall from the hole too - and
    # the fingers are told where the holes are, to run back to them.
    # One finger depth across when fingers are on: the notch bottom is half
    # a depth in, so a hole of that radius meets it at a tangent (the user's
    # spec). One board across without fingers - nothing to be tangent to.
    holes = {}
    hole_size = mat.finger_depth_used if mat.use_fingers else mat.thickness
    if mat.corner_hole:
        for isl in islands:
            bodies[isl.id], holes[isl.id] = apply_corner_holes(
                isl, bodies[isl.id], hole_size)
    # Fingers first: a seam that interlocks needs no flap, and a flap laid
    # along a fingered edge would bury the fingers. Tabs then go on the seams
    # that could not take a finger.
    joints = (place_fingers(islands, mat, bodies, holes=holes,
                            hole_r=hole_size / 2.0)
              if mat.use_fingers else {})
    # Tabs are optional: most cardboard is taped or gets a glued backing strip,
    # and the flaps cost about a fifth of the sheet in area.
    tabs, skipped = (place_tabs(islands, mat, skip=fingered_seams(joints))
                     if mat.use_tabs else ({}, set()))

    parts = []
    for isl in islands:
        plans = plans_by[isl.id]
        warns = [p.warn for p in plans if p.warn]

        body = apply_fingers(bodies[isl.id], joints.get(isl.id, []))
        tab_polys = [t.poly for t in tabs.get(isl.id, [])]
        if tab_polys:
            body = unary_union([shapely.set_precision(body, 1e-9)] +
                               [shapely.set_precision(t, 1e-9) for t in tab_polys])

        cut = offset_cut(body, mat.edge_offset)
        if cut.is_empty:
            warns.append(
                f"edge offset {mat.edge_offset:+.2f}mm removed this part "
                f"entirely; cutting it at its own edge instead")
            cut = body
        # Only an OUTWARD offset buries a notch apex, so there is nothing to
        # relieve without one.
        if mat.corner_relief and mat.edge_offset > 0:
            cut = corner_relief(body, cut, mat.kerf, mat.corner_overcut)
        if not annotate:
            parts.append(Part(island_id=isl.id, cut=cut, perf_m=_mls([]),
                              perf_v=_mls([]), engrave=_mls([]),
                              warnings=warns, model=mat.model_id))
            continue

        perf_m, perf_v = [], []
        for p in plans:
            (perf_m if p.is_mountain else perf_v).extend(p.perf)
        # a tab folds against its own base edge
        for t in tabs.get(isl.id, []):
            from .folds import perf_segments
            perf_m.extend(perf_segments(t.base[0], t.base[1], mat))

        # One pass per island, so the three kinds compete for the same room
        # instead of being laid down blind on top of one another.
        strokes = []
        for lab in island_labels(isl, plans, mat, body):
            strokes.extend(lab.strokes)

        necks = narrow_necks(cut, max(3.0, 2.0 * mat.thickness))
        if necks and getattr(isl, "flat", False):
            # A flat piece is the shape it was drawn as: no seed or trial
            # count changes it, so the unfold advice below would be wrong.
            warns.append(
                f"{necks} place(s) narrower than "
                f"{max(3.0, 2.0*mat.thickness):.0f}mm in this flat piece - "
                f"fragile in board this thick; widen them in the model")
        elif necks:
            warns.append(
                f"{necks} neck(s) narrower than {max(3.0, 2.0*mat.thickness):.0f}mm - "
                f"the net rejoins itself through a near-pinch and will tear there. "
                f"Re-run with more trials or a different seed, or split this island.")

        parts.append(Part(
            island_id=isl.id, cut=cut,
            perf_m=_mls(perf_m), perf_v=_mls(perf_v),
            engrave=_mls(strokes), warnings=warns, model=mat.model_id))

    if skipped:
        parts[0].warnings.append(
            f"{len(skipped)} seam(s) had no room for a glue tab - tape those "
            f"(seams {sorted(skipped)[:10]}{'...' if len(skipped) > 10 else ''})")
    return parts


def orient_min_bbox(part: Part) -> None:
    """Rotate a part onto its minimum-area bounding box, so it packs tightly.

    Uses the core's own rotated-rectangle rather than the GEOS one, which
    degenerates on some nets - see min_area_rect. A bogus angle here does not
    fail loudly, it just quietly wastes sheet.
    """
    pts = shapely.get_coordinates(part.cut)
    w, h, angle, centre = min_area_rect(pts)
    if w <= 0.0 or h <= 0.0:
        return
    if h > w:                          # bring the long side onto x
        angle += np.pi / 2.0
    part.rotate(-np.degrees(angle), origin=centre)


def grow(geom, d: float, **kw):
    """`geom` buffered by `d` for a collision test - and made valid.

    A part carries its relief slots as holes. Growing it by the part gap
    shrinks every hole by the gap from both sides, so a slot exactly twice the
    gap wide - a 2 mm cut width against a 1 mm gap - collapses to zero width,
    and GEOS hands back an INVALID MultiPolygon with a zero-area piece where
    the slot was. The next overlay then throws ("found non-noded intersection
    between LINESTRING (x y, x y) and LINESTRING (x y, x y)", or "side
    location conflict") and the whole unfold fails. The repair changes no
    area at all - measured to 1e-4 mm2 on the helmet part that did it - so it
    only runs when the buffer came back broken.
    """
    g = geom.buffer(d, **kw)
    if g.is_valid:
        return g
    g = shapely.make_valid(g)
    # make_valid keeps a collapsed hole as a stray line in a
    # GeometryCollection; only the area is an obstacle.
    polys = [q for q in getattr(g, "geoms", [g])
             if q.geom_type in ("Polygon", "MultiPolygon") and not q.is_empty]
    return unary_union(polys) if polys else Polygon()


def _corner_candidates(placed_boxes, x0: float, y0: float, gap: float) -> list:
    """Positions worth trying for the next part: the sheet corner, and the
    corners of everything already down.

    A grid sweep over a 600x400 bed at any useful resolution is tens of
    thousands of shapely tests per part. In a bottom-left packing the optimum
    always touches something, so only contact positions matter - that is a few
    hundred candidates instead, and it is what makes true-shape nesting
    tractable in pure Python.
    """
    xs = {x0}
    ys = {y0}
    for bx0, by0, bx1, by1 in placed_boxes:
        # past the far edge by a full gap. Offering the edge itself is useless:
        # the two parts' half-gap buffers would overlap there by the whole gap,
        # so every such candidate is rejected and parts never share a sheet.
        xs.add(bx1 + gap)
        ys.add(by1 + gap)
        xs.add(bx0)
        ys.add(by0)
    return sorted(((y, x) for y in sorted(ys) for x in sorted(xs)))


NEST_ANGLES = (0.0, 90.0, 180.0, 270.0)


def find_spot(part: Part, placed_geo: list, placed_box: list, W: float,
              H: float, gap: float, m: float, angles=NEST_ANGLES):
    """Where this part can go among what is already down, or None.

    Returns (angle, geometry as placed). Nothing already down is consulted for
    anything but collision, so a caller can use this to slot a part into a
    finished sheet without disturbing it.
    """
    cands = _corner_candidates(placed_box, m, m, gap)
    tree = STRtree(placed_geo) if placed_geo else None
    for angle in angles:
        cand = affinity.rotate(part.cut, angle, origin="centroid")
        bx0, by0, bx1, by1 = cand.bounds
        cw, ch = bx1 - bx0, by1 - by0
        if cw > W or ch > H:
            continue
        # Buffer once per rotation, not once per candidate position. Buffering
        # is the expensive call here and there are hundreds of positions; doing
        # it inside the loop took the core suite from 36 s to 156 s on its own.
        # One side buffered by the full gap is equivalent to both by half.
        grown = grow(cand, gap, join_style=2)
        for y, x in cands:
            if x + cw > m + W or y + ch > m + H:
                continue
            dx, dy = x - bx0, y - by0
            probe = affinity.translate(grown, dx, dy)
            if tree is not None:
                hit = False
                for k in tree.query(probe):
                    if probe.intersects(placed_geo[int(k)]):
                        hit = True
                        break
                if hit:
                    continue
            return angle, affinity.translate(cand, dx, dy)
    return None


POCKET_BUDGET = 16          # deep searches per tighten call


def free_region(placed_geo: list, W: float, H: float, gap: float,
                m: float) -> list:
    """What is left of one sheet once the placed parts and their gap are out.

    Buffering the obstacles by the whole gap (rather than the candidate by
    half of it on each side) means a part that fits inside one of these pieces
    is already the full gap clear of everything - the same rule find_spot uses.
    """
    sheet = box(m, m, m + W, m + H)
    if not placed_geo:
        return [sheet]
    free = sheet.difference(grow(unary_union(placed_geo), gap))
    if free.is_empty:
        return []
    return [free] if free.geom_type == "Polygon" else list(free.geoms)


def find_pocket(part: Part, pieces: list, angles=NEST_ANGLES,
                tol: float = 0.75):
    """Thread a part into a pocket the corner candidates cannot see.

    find_spot offers the cross product of the placed parts' x and y edges.
    That is the right heuristic for building a packing against a skyline and
    the wrong one for slotting into a sheet that is already finished, where the
    opening is bounded by parts on several sides at once. Measured on pikachu
    at 150 mm: a 44.6 x 92.2 mm part had exactly one home on a sheet 44% full -
    a 5 x 2 mm window of positions at a single rotation - and not one of the
    4900 corner candidates landed in it, so a whole second sheet was spent.

    Candidates are the corners of the free region. Taking them from a
    SIMPLIFIED copy is 6x cheaper (3.06 s -> 0.50 s on that sheet) and risks
    nothing: the placement is still verified against the exact piece, so a
    simplification that strays outward can only cost a position, never put a
    part on top of another.

    Returns (angle, geometry as placed) like find_spot, or None.
    """
    area = part.cut.area
    for angle in angles:
        cand = affinity.rotate(part.cut, angle, origin="centroid")
        bx0, by0, bx1, by1 = cand.bounds
        cw, ch = bx1 - bx0, by1 - by0
        for piece in pieces:
            qx0, qy0, qx1, qy1 = piece.bounds
            if (piece.area < area or qx1 - qx0 < cw - 1e-9
                    or qy1 - qy0 < ch - 1e-9):
                continue                        # cannot hold it, whatever else
            shapely.prepare(piece)
            simp = piece.simplify(tol)
            verts = list(simp.exterior.coords)
            for ring in simp.interiors:
                verts += list(ring.coords)
            # Each corner of the part's box against each corner of the pocket:
            # which side of the part the opening is bounded on is not known.
            for vx, vy in verts:
                for ax, ay in ((0.0, 0.0), (cw, 0.0), (0.0, ch), (cw, ch)):
                    ox, oy = vx - ax, vy - ay
                    if (ox < qx0 - 1e-9 or oy < qy0 - 1e-9
                            or ox + cw > qx1 + 1e-9 or oy + ch > qy1 + 1e-9):
                        continue
                    moved = affinity.translate(cand, ox - bx0, oy - by0)
                    if piece.contains(moved):
                        return angle, moved
    return None


def _put(part: Part, angle: float, moved) -> None:
    """Apply a find_spot result to the part, keeping Part.xform in step."""
    part.rotate(angle, origin=tuple(part.cut.centroid.coords)[0])
    b = part.bounds
    part.translate(moved.bounds[0] - b[0], moved.bounds[1] - b[1])


def long_axis(mat: Material) -> int:
    """0 if the bed is longer in x, 1 if in y - or square, which keeps the
    packers' original bottom-first order. Parts are pushed toward the low end
    of this axis, so a part-used last sheet leaves its offcut as one strip,
    the full width of the short side, at the far end of the long one (the
    user's rule: "take the ratio on the long dimension")."""
    W, H = mat.usable_bed
    return 0 if W > H else 1


def sheet_equivalents(parts: list[Part], mat: Material) -> tuple[float, float]:
    """(sheets used, as a fraction; the fraction of the last sheet left free).

    Every sheet but the last counts whole. The last counts only as far along
    the long axis as its parts reach: what is beyond them is a strip of board
    the full width of the sheet, kept for another job. So 3 sheets with the
    last 40% used is 2.4 - one number for "how much board did this take",
    which an integer sheet count cannot give: it reads the same whether the
    last sheet holds one sliver or is nearly full.
    """
    on = [p for p in parts if p.sheet >= 0]
    if not on:
        return 0.0, 0.0
    last = max(p.sheet for p in on)
    ax = long_axis(mat)
    span = mat.usable_bed[ax]
    reach = max(p.bounds[2 + ax] for p in on if p.sheet == last) - mat.sheet_margin
    frac = min(1.0, max(0.0, reach / span))
    return last + frac, 1.0 - frac


def seat_sheets(parts: list[Part], mat: Material) -> None:
    """Sit each sheet's parts on their margin, together.

    Per sheet, not per part: the arrangement within a sheet is the packing and
    must not be disturbed.
    """
    m = mat.sheet_margin
    by: dict[int, list[Part]] = {}
    for p in parts:
        if p.sheet >= 0:
            by.setdefault(p.sheet, []).append(p)
    for group in by.values():
        x0 = min(p.bounds[0] for p in group)
        y0 = min(p.bounds[1] for p in group)
        if abs(x0 - m) > 1e-9 or abs(y0 - m) > 1e-9:
            for p in group:
                p.translate(m - x0, m - y0)


def tighten(parts: list[Part], sheets: int, mat: Material) -> int:
    """Move parts off the later sheets into room left on the earlier ones.

    Every packer here places a part once and never looks at it again. Shelf
    packing commits to a shelf; true-shape nesting is greedy bottom-left; and
    the add-on's Nest Sheets re-runs Blender's packer over the whole candidate
    set on every trial, so a part is dropped to the next sheet whenever that
    global re-solve happens to fail - which is not the same as there being no
    room for it.

    Measured on pikachu at 150 mm: 35 parts on a sheet only 44% full, and one
    44.6 x 92.2 mm part alone on a second sheet that fitted the first at 270
    degrees. A whole sheet of cardboard for a part that had a home.

    Three searches, cheapest first, and they answer different questions.
    `find_spot` offers corner positions against the skyline - right for a
    sheet being built, useless for one that is already full. `raster_spot`
    offers EVERY position on a 2 mm grid from one transform per sheet, which
    is what finds a pocket in the middle of a finished sheet. `find_pocket`
    stays last because it works on exact geometry where the raster is
    conservative by 0.71 of a cell, so it can still place a part that needs
    the last millimetre - and it stays budgeted, because it is the expensive
    one.

    That order matters more than it sounds. Measured on a torus at 200 mm:
    the last sheet held ONE part at 13% full while sheet 17 sat at 56%, and
    the part fitted sheet 17 perfectly well. `find_pocket` never got to try:
    POCKET_BUDGET is 16 deep searches, sheet 17 is the seventeenth target, and
    the budget was spent before the search reached it. A budget that runs out
    in sheet order is a budget that always fails on the same sheets.

    Nothing already placed is moved, so this can only tie or improve. Sheets
    left empty are renumbered away; returns the new sheet count.
    """
    W, H = mat.usable_bed
    gap, m = mat.part_gap, mat.sheet_margin

    by: dict[int, list[Part]] = {}
    for p in parts:
        if p.sheet >= 0:
            by.setdefault(p.sheet, []).append(p)
    if len(by) < 2:
        return sheets

    # Seat first, always. The free region is measured over
    # [m, m+W] x [m, m+H], so a sheet whose parts have not been put on
    # their margin yet is being measured in a frame it does not live
    # in: the new part lands correctly in that box, and the seating
    # shift that follows then drags it out of the bed. That is exactly
    # how this arrived - "part 31 ended up outside sheet 0" from the
    # add-on's nester, which seated after packing rather than before.
    seat_sheets(parts, mat)

    # Latest sheet first: emptying it is what removes a sheet, and its parts
    # have the most earlier sheets to choose from.
    budget = POCKET_BUDGET
    # Both the free region and the occupancy transform cost a pass over
    # everything on the sheet and only change when something is put down
    # there. Cache them per sheet; several strays usually fail against the
    # same one.
    pockets: dict[int, list] = {}
    grids: dict[int, tuple] = {}
    for src in sorted(by, reverse=True):
        targets = sorted(k for k in by if k < src)
        for part in list(by[src]):
            masks = None
            for dst in targets:
                geo = [q.cut for q in by[dst]]
                spot = find_spot(part, geo, [q.bounds for q in by[dst]],
                                 W, H, gap, m)
                if spot is None:
                    if dst not in grids:
                        occ, shape = sheet_occupancy(parts, dst, mat)
                        grids[dst] = (np.fft.rfft2(occ, s=shape), shape)
                    if masks is None:
                        masks = part_masks(part, mat)
                    spot = raster_spot(part, *grids[dst], mat, masks=masks)
                if spot is None and budget > 0:
                    # Exact geometry, where the raster is conservative by most
                    # of a cell. Last because it is the expensive one.
                    budget -= 1
                    if dst not in pockets:
                        pockets[dst] = free_region(geo, W, H, gap, m)
                    spot = find_pocket(part, pockets[dst],
                                       tol=max(gap / 4.0, 1e-3))
                if spot is None:
                    continue
                pockets.pop(dst, None)          # the sheet just changed
                grids.pop(dst, None)
                _put(part, *spot)
                part.sheet = dst
                by[dst].append(part)
                by[src].remove(part)
                break
        if not by[src]:
            del by[src]

    remap = {old: new for new, old in enumerate(sorted(by))}
    for old, group in by.items():
        for part in group:
            part.sheet = remap[old]
    seat_sheets(parts, mat)     # a sheet that lost parts may have lost its edge
    return len(by)


def nest(parts: list[Part], mat: Material) -> tuple[int, list[Part]]:
    """True-shape bottom-left nesting. Returns (sheet count, oversized parts).

    Shelf packing puts one bbox per shelf and wastes everything concave, which
    on these nets is most of the sheet. Here a part is placed against the
    actual outlines already down, so a spur of one island can sit in the notch
    of another.

    Parts keep their size: rotation is allowed, scaling never is, because the
    unfold is isometric and a scaled part stops mating with its neighbours.
    """
    W, H = mat.usable_bed
    gap = mat.part_gap
    m = mat.sheet_margin

    for p in parts:
        orient_min_bbox(p)

    oversized, fit = [], []
    for p in parts:
        w, h = p.size
        if (w > W or h > H) and (h > W or w > H):
            oversized.append(p)
        else:
            fit.append(p)
    fit.sort(key=lambda p: -p.cut.area)

    sheet = 0
    remaining = list(fit)
    while remaining:
        placed_geo: list = []
        placed_box: list = []
        leftovers = []
        for p in remaining:
            spot = find_spot(p, placed_geo, placed_box, W, H, gap, m)
            if spot is None:
                leftovers.append(p)
                continue
            _put(p, *spot)
            p.sheet = sheet
            placed_geo.append(p.cut)
            placed_box.append(p.bounds)
        if len(leftovers) == len(remaining):
            break                       # nothing fits an empty sheet: give up
        remaining = leftovers
        if remaining:
            sheet += 1

    for p in oversized:
        p.sheet = -1
    return (sheet + 1 if fit else 0), oversized

def shelf_pack(parts: list[Part], mat: Material) -> tuple[int, list[Part]]:
    """Shelf-pack parts onto sheets. Returns (sheet count, oversized parts)."""
    W, H = mat.usable_bed
    gap = mat.part_gap

    for p in parts:
        orient_min_bbox(p)
        w, h = p.size
        # orient_min_bbox always lays the long side on x, which is right for a
        # landscape bed and wrong for a portrait one: a 615 x 292 part on a
        # 384 x 884 bed only fits standing up, and this used to call it
        # oversized.
        if not (w <= W and h <= H) and (h <= W and w <= H):
            p.rotate(90, origin=tuple(p.cut.centroid.coords)[0])

    oversized = [p for p in parts if p.size[0] > W or p.size[1] > H]
    fit = [p for p in parts if p not in oversized]
    fit.sort(key=lambda p: -p.size[1])

    sheet, x, y, shelf_h = 0, 0.0, 0.0, 0.0
    for p in fit:
        w, h = p.size
        if x + w > W:                        # next shelf
            x, y = 0.0, y + shelf_h + gap
            shelf_h = 0.0
        if y + h > H:                        # next sheet
            sheet += 1
            x, y, shelf_h = 0.0, 0.0, 0.0
        x0, y0, _, _ = p.bounds
        p.translate(mat.sheet_margin + x - x0, mat.sheet_margin + y - y0)
        p.sheet = sheet
        x += w + gap
        shelf_h = max(shelf_h, h)

    for p in oversized:
        p.sheet = -1
    return (sheet + 1 if fit else 0), oversized


# The raster packer's two knobs, and neither wants turning up. Measured across
# the corpus: 8 or 12 angles instead of 4, and 1 mm cells instead of 2, give
# byte-identical sheet counts for 2-5x the time. The win is asking about EVERY
# position at once rather than about a handful of corners - it is not about
# resolution, and a finer sweep is exactly what plateaued before.
RASTER_RES = 2.0                        # mm per cell
RASTER_ANGLES = (0.0, 45.0, 90.0, 135.0)


def raster_mask(geom, res: float, gap: float):
    """A part as a binary grid, plus the grid's origin in mm.

    Grown by half the gap first, so that touching in the raster means `gap`
    apart in millimetres - half on each of two parts is a whole gap between
    them, the same trick `find_spot` plays with its buffers, and it means the
    collision test needs no clearance logic of its own.

    Then grown again by **half a cell diagonal**, which is what makes the mask
    a provable SUPERSET of the shape rather than a sample of it. The argument
    is exact: the nearest cell centre to any point is the centre of that
    point's own cell, at most (res/2, res/2) away, so it lies within
    res*0.7071 - hence inside the second buffer - and the cell is marked. A
    mask that under-covers would let two parts overlap by up to a cell, and a
    cell is 2 mm of real collision on a real laser.

    Dilating the sampled grid instead is the obvious move and is NOT
    sufficient: a 4-connected dilation cannot reach the cell at a convex
    corner, whose only filled neighbour is diagonal
    (`test_the_raster_mask_covers_the_whole_part` catches exactly that). This
    is also slightly less conservative than a dilation - 0.71 of a cell rather
    than a whole one - and being conservative here is measured not to matter:
    halving the cell size halves the padding and changes no sheet count.

    The origin comes back with the mask because the two must agree. Placing a
    part by one geometry's bounds while its mask was built from another's is
    an offset nothing downstream would catch.
    """
    g = grow(geom, gap / 2.0 + res * 0.7072)
    x0, y0, x1, y1 = g.bounds
    nx = max(1, int(math.ceil((x1 - x0) / res)) + 1)
    ny = max(1, int(math.ceil((y1 - y0) / res)) + 1)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    m = shapely.contains_xy(g, X.ravel(), Y.ravel()).reshape(ny, nx)
    return m, x0, y0


def mask_fft(mask: np.ndarray, shape) -> np.ndarray:
    """The transform of a part's mask, ready for `free_offsets`.

    It depends only on the mask and the sheet's grid, so a part tried against
    sheet after sheet can carry it rather than recompute it every time.
    """
    return np.fft.rfft2(mask[::-1, ::-1].astype(np.float64), s=shape)


def free_offsets(F, shape, mask: np.ndarray, Mk=None) -> np.ndarray:
    """Every grid offset at which `mask` lands on nothing, in one shot.

    Cross-correlating the sheet's occupancy with the part gives the overlap
    count at ALL offsets from a single transform pair, so the number of
    candidate positions stops being the thing you have to budget for. That is
    the whole idea: `_corner_candidates` offers a few hundred positions chosen
    by a heuristic that cannot see a pocket, and this offers every one.

    `F` is the occupancy already transformed - it is shared across a part's
    rotations, which is most of the saving once there is more than one angle.
    The 0.5 threshold is because a float transform lands on ~1e-12, not 0.
    `Mk` is the mask's own transform from `mask_fft`, when the caller has it.
    """
    gh, gw = shape
    h, w = mask.shape
    if h > gh or w > gw:
        return None
    if Mk is None:
        Mk = mask_fft(mask, (gh, gw))
    corr = np.fft.irfft2(F * Mk, s=(gh, gw))
    # corr[y + h - 1, x + w - 1] is the overlap of the mask placed at (x, y)
    return corr[h - 1:gh, w - 1:gw] < 0.5


def sheet_occupancy(parts, sheet: int, mat: Material, res: float = RASTER_RES):
    """What is already on one sheet, as a grid, ready to be correlated.

    Built from where the parts actually are rather than from a packing order,
    so it works on a finished layout - which is what `tighten` has.
    """
    W, H = mat.usable_bed
    gw, gh = int(math.ceil(W / res)), int(math.ceil(H / res))
    occ = np.zeros((gh, gw), dtype=np.float64)
    for q in parts:
        if q.sheet != sheet:
            continue
        m, mx0, my0 = raster_mask(q.cut, res, mat.part_gap)
        gx = int(round((mx0 - mat.sheet_margin) / res))
        gy = int(round((my0 - mat.sheet_margin) / res))
        sub = m[max(0, -gy):, max(0, -gx):]
        x0, y0 = max(gx, 0), max(gy, 0)
        sub = sub[:gh - y0, :gw - x0]
        if sub.size:
            occ[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] += sub
    return occ, (gh, gw)


def part_masks(part: Part, mat: Material, res: float = RASTER_RES,
               angles=RASTER_ANGLES) -> list:
    """Every rotation of one part, rasterised once.

    Rasterising is `shapely.contains_xy` over a grid and costs far more than
    the transform that consumes it, so a part asked about twenty sheets must
    be rasterised once rather than twenty times. At a 400 mm model that is the
    difference between seconds and minutes.
    """
    return [raster_mask(affinity.rotate(part.cut, a, origin="centroid"),
                        res, mat.part_gap) for a in angles]


def raster_spot(part: Part, F, shape, mat: Material, res: float = RASTER_RES,
                angles=RASTER_ANGLES, masks=None):
    """Bottom-left room for one part on an already-transformed sheet, as a
    `_put` result - or None if there is none.

    `F` is the sheet's occupancy transform, shared across every part tried
    against that sheet; `masks` is the part's own rasters, shared across every
    sheet it is tried on. Both are what make asking twenty sheets about one
    part affordable.
    """
    gh, gw = shape
    pad = mat.part_gap / 2.0 + res * 0.7072
    best = None
    for ang, (m, _mx0, _my0) in zip(angles,
                                    masks or part_masks(part, mat, res,
                                                        angles)):
        ok = free_offsets(F, shape, m)
        if ok is None or not ok.any():
            continue
        ys, xs = np.nonzero(ok)
        k = int(np.argmin(ys * gw + xs))
        cand = (int(ys[k]), int(xs[k]), ang)
        if best is None or cand[:2] < best[:2]:
            best = cand
    if best is None:
        return None
    gy, gx, ang = best
    # _put matches the part's own min corner to this box's, and the mask's
    # origin is the PADDED min corner, so the padding comes back off here.
    x = mat.sheet_margin + gx * res + pad
    y = mat.sheet_margin + gy * res + pad
    return ang, box(x, y, x + 1.0, y + 1.0)


def raster_pack(parts: list[Part], mat: Material, res: float = RASTER_RES,
                angles=RASTER_ANGLES, jitter=None) -> tuple[int, list[Part]]:
    """True-shape bottom-left over a raster, with FFT collision.

    `nest` asks "does this part fit at one of these corners"; this asks "where
    on this sheet does this part fit at all", which is a different question and
    a strictly larger set of answers. On the corpus it takes pikachu from 3
    sheets to 2 and torus from 22 to 21, ties everything else, and on pikachu
    it is also ten times faster than the packer it beats.

    Rotations are rasterised once per part rather than per placement. The
    occupancy is transformed only when a part has landed on it - a part that
    does not fit changes nothing - and each mask's transform is computed the
    first time it is needed and kept, since a large part is tried against
    sheet after sheet. Same arrays into the same transforms, so the packing is
    bit-identical to recomputing both every time; on a torus it was 11,520
    transforms.
    """
    W, H = mat.usable_bed
    gap = mat.part_gap
    gw, gh = int(math.ceil(W / res)), int(math.ceil(H / res))

    for p in parts:
        orient_min_bbox(p)
    oversized, fit = [], []
    for p in parts:
        w, h = p.size
        if (w > W or h > H) and (h > W or w > H):
            oversized.append(p)
        else:
            fit.append(p)
    if jitter is None:
        fit.sort(key=lambda p: -p.cut.area)
    else:
        # Decreasing area, shuffled a little: a restart that asks the same
        # greedy question in a slightly different order. `jitter` is a numpy
        # Generator, so a restart is reproducible from its seed.
        noise = {id(p): float(np.exp(jitter.normal(0.0, 0.3))) for p in fit}
        fit.sort(key=lambda p: -p.cut.area * noise[id(p)])

    shapes = {id(p): [[a, *raster_mask(affinity.rotate(p.cut, a,
                                                       origin="centroid"),
                                       res, gap), None] for a in angles]
              for p in fit}

    sheet = 0
    occ = np.zeros((gh, gw), dtype=np.float64)
    F = None                            # occ's transform, while it is current
    remaining = list(fit)
    while remaining:
        leftovers = []
        for p in remaining:
            if F is None:
                F = np.fft.rfft2(occ, s=(gh, gw))
            best = None
            for entry in shapes[id(p)]:
                ang, m, mx0, my0, Mk = entry
                if Mk is None and m.shape[0] <= gh and m.shape[1] <= gw:
                    Mk = entry[4] = mask_fft(m, (gh, gw))
                ok = free_offsets(F, (gh, gw), m, Mk)
                if ok is None or not ok.any():
                    continue
                ys, xs = np.nonzero(ok)
                k = int(np.argmin(ys * gw + xs))        # bottom, then left
                cand = (int(ys[k]), int(xs[k]), ang, m, mx0, my0)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            if best is None:
                leftovers.append(p)
                continue
            gy, gx, ang, m, mx0, my0 = best
            p.rotate(ang, origin=p.cut.centroid.coords[0])
            # mx0/my0 are the grid's own origin, not the part's bounds - see
            # raster_mask on why those are not the same thing.
            p.translate(gx * res - mx0 + mat.sheet_margin,
                        gy * res - my0 + mat.sheet_margin)
            p.sheet = sheet
            occ[gy:gy + m.shape[0], gx:gx + m.shape[1]] += m
            F = None
        if len(leftovers) == len(remaining):
            break                       # nothing fits an empty sheet
        remaining = leftovers
        if remaining:
            sheet += 1
            occ = np.zeros((gh, gw), dtype=np.float64)
            F = None

    for p in oversized:
        p.sheet = -1
    return (sheet + 1 if fit else 0), oversized


# Rotation sets the last-sheet pass tries, and the part orders. More angles
# than the main packer, because here they can only help: the pass keeps a
# result only if it frees more board, and a last sheet holds a handful of
# parts, not dozens.
COMPACT_ANGLE_SETS = (RASTER_ANGLES,
                      tuple(float(a) for a in range(0, 360, 45)),
                      tuple(float(a) for a in range(0, 360, 30)))


def compact_last(parts: list[Part], mat: Material,
                 res: float = RASTER_RES) -> bool:
    """Re-pack the last sheet so its parts reach as short a way along the
    bed's long axis as they can, leaving the rest as one offcut strip.

    Every packer here is bottom-left for the sake of FITTING parts, which is
    not the same as keeping them low: on a 300 mm bed the helmet's last sheet
    was a third full and reached 98% of the way up. So each part, largest
    first, goes where its FAR edge along the long axis ends lowest - not its
    near edge - ties to the low end of the short axis. A few part orders and
    rotation sets are tried, and the best is kept only if it measurably beats
    what was there, so this can only tie or free board. Returns whether it
    moved anything.
    """
    on = [p for p in parts if p.sheet >= 0]
    if not on:
        return False
    last = max(p.sheet for p in on)
    group = [p for p in on if p.sheet == last]
    before, _ = sheet_equivalents(parts, mat)
    ax = long_axis(mat)
    W, H = mat.usable_bed
    gw, gh = int(math.ceil(W / res)), int(math.ceil(H / res))
    gap = mat.part_gap

    masks: dict = {}

    def mask(i, a):
        if (i, a) not in masks:
            m, mx0, my0 = raster_mask(
                affinity.rotate(group[i].cut, a, origin="centroid"), res, gap)
            masks[(i, a)] = (m, mx0, my0,
                             mask_fft(m, (gh, gw))
                             if m.shape[0] <= gh and m.shape[1] <= gw else None)
        return masks[(i, a)]

    def dims(i):
        x0, y0, x1, y1 = group[i].bounds
        return x1 - x0, y1 - y0

    idx = range(len(group))
    orders = [sorted(idx, key=lambda i: -group[i].cut.area),
              sorted(idx, key=lambda i: -max(dims(i))),
              sorted(idx, key=lambda i: -dims(i)[ax]),
              sorted(idx, key=lambda i: -dims(i)[1 - ax])]
    best = None
    for angles in COMPACT_ANGLE_SETS:
        for order in orders:
            occ = np.zeros((gh, gw), dtype=np.float64)
            plan, reach, ok = {}, 0, True
            for i in order:
                F = np.fft.rfft2(occ, s=(gh, gw))
                pick = None
                for a in angles:
                    m, mx0, my0, Mk = mask(i, a)
                    if Mk is None:
                        continue
                    free = free_offsets(F, (gh, gw), m, Mk)
                    if free is None or not free.any():
                        continue
                    ys, xs = np.nonzero(free)
                    far = (ys + m.shape[0]) if ax == 1 else (xs + m.shape[1])
                    near = xs if ax == 1 else ys
                    k = int(np.lexsort((near, far))[0])
                    key = (int(far[k]), int(near[k]))
                    if pick is None or key < pick[0]:
                        pick = (key, a, int(ys[k]), int(xs[k]))
                if pick is None:
                    ok = False
                    break
                (f, _n), a, gy, gx = pick
                m = masks[(i, a)][0]
                occ[gy:gy + m.shape[0], gx:gx + m.shape[1]] += m
                plan[i] = (a, gx, gy)
                reach = max(reach, f)
            if ok and (best is None or reach < best[0]):
                best = (reach, plan)
    if best is None:
        return False

    state = [(p.cut, p.perf_m, p.perf_v, p.engrave, p.xform) for p in group]
    for i, (a, gx, gy) in best[1].items():
        p = group[i]
        _m, mx0, my0, _Mk = masks[(i, a)]
        p.rotate(a, origin=p.cut.centroid.coords[0])
        # mx0/my0 are the mask grid's own origin - see raster_mask
        p.translate(gx * res - mx0 + mat.sheet_margin,
                    gy * res - my0 + mat.sheet_margin)
    after, _ = sheet_equivalents(parts, mat)
    if after < before - 1e-9:
        return True
    for p, st in zip(group, state):     # no better: put it back exactly
        (p.cut, p.perf_m, p.perf_v, p.engrave, p.xform) = st
    return False


# Extra raster restarts at full packing effort. Each is one raster pack plus
# the last-sheet pass; only the best of them is tightened, because tighten is
# the expensive step.
EFFORT_RESTARTS = 24


def pack(parts: list[Part], mat: Material, effort: float = 0.0,
         progress=None) -> tuple[int, list[Part]]:
    """Pack onto sheets, taking whichever strategy wins on this net.

No strategy dominates. True-shape nesting is far better when parts are small
    enough to share a sheet (a pauldron split to 70% of the bed: 9 sheets
    shelved, 5 nested) and worse when a greedy bottom-left placement paints
    itself into a corner (drum12 split to 50%: 5 shelved, 10 nested).
    `raster_pack` beats both where it wins and ties where it does not, but it
    is the newest and the least exercised, so it earns its place by the same
    rule as the others rather than by replacing them. Running all three
    removes the risk entirely.

    Packing mutates the parts, so each strategy starts from a snapshot. The
    snapshot is references, not copies: shapely geometries are immutable and
    Part.xform is replaced rather than mutated, so there is nothing to clone.
    deepcopy here instead took the core suite from 36 s to 159 s.
    """
    def snap():
        return [(p.cut, p.perf_m, p.perf_v, p.engrave, p.sheet, p.xform)
                for p in parts]

    def restore(state):
        for p, st in zip(parts, state):
            (p.cut, p.perf_m, p.perf_v, p.engrave, p.sheet, p.xform) = st

    start_state = snap()
    trials = []
    extra = int(round(max(0.0, min(1.0, effort)) * EFFORT_RESTARTS))

    def report(done, note):
        return progress is None or progress(done, 3 + extra, note)

    for fn in (shelf_pack, nest, raster_pack):
        restore(start_state)
        for p in parts:
            p.xform = np.eye(3)
        sheets, _over = fn(parts, mat)
        # Tighten before comparing, not after choosing. Both strategies leave
        # room on an earlier sheet that a later part fits, and they leave
        # different amounts, so the winner before tightening is not
        # necessarily the winner after it.
        sheets = tighten(parts, sheets, mat)
        # And pack the last sheet low before comparing, for the same reason.
        compact_last(parts, mat)
        # Count what is actually off the sheets, not what the strategy owned up
        # to: nest and raster_pack also leave a part unplaced when it passes
        # the size test and still fits no empty sheet, without listing it.
        trials.append((sheets, sum(1 for p in parts if p.sheet < 0),
                       sheet_equivalents(parts, mat)[0], snap()))
        report(len(trials), f"packing: {fn.__name__}")

    # Packing effort: restarts of the raster packer, each with its own
    # rotation set and a shuffled part order, scored after the last-sheet pass
    # alone. The best is then tightened and has to beat the three above on
    # the same terms. On the helmet nothing here beat the tie-break below -
    # more angles, more orders, finer cells and ruin-and-recreate were all
    # measured - so effort is offered, not assumed (see CLAUDE.md).
    restarts = []
    for k in range(extra):
        restore(start_state)
        for p in parts:
            p.xform = np.eye(3)
        sheets, _over = raster_pack(parts, mat,
                                    angles=COMPACT_ANGLE_SETS[k % len(COMPACT_ANGLE_SETS)],
                                    jitter=np.random.default_rng(k))
        compact_last(parts, mat)
        restarts.append((sum(1 for p in parts if p.sheet < 0), sheets,
                         sheet_equivalents(parts, mat)[0], snap()))
        if not report(3 + k + 1, f"packing harder {k + 1}/{extra}"):
            break
    if restarts:
        _u, sheets, _e, state = min(restarts, key=lambda t: t[:3])
        restore(state)
        sheets = tighten(parts, sheets, mat)
        compact_last(parts, mat)
        trials.append((sheets, sum(1 for p in parts if p.sheet < 0),
                       sheet_equivalents(parts, mat)[0], snap()))

    # Parts left off FIRST, sheets second. The other way round, a strategy
    # that gives up on a part reports fewer sheets and wins for it - shelf
    # packing on a 400 x 900 bed dropped four helmet parts it could not stand
    # up, called it one sheet, and beat three sheets with everything on.
    # Sheets only count PLACED parts, so a dropped part is not a saving, it is
    # a part that cannot be cut.
    #
    # Then BOARD, which the whole-sheet count cannot see: two strategies that
    # both need 2 sheets tied, and the tie went to whichever ran first -
    # shelf packing. On the helmet's 14 pieces that left sheet 1 at 41% with
    # gaps a reader could see from across the room, where the raster packer
    # filled it to 59%: 1.59 sheets of board against 1.26. More angles, more
    # part orders, finer cells and a void-fill pass at 72 angles / 1 mm were
    # all measured on the same parts and bought nothing on top of this.
    trials.sort(key=lambda t: (t[1], t[0], t[2]))
    sheets, _, _eq, winner = trials[0]
    restore(winner)

    # A part too big for the bed is never placed, so it keeps whatever
    # coordinates the unfold happened to give it - which are routinely
    # negative. Nothing downstream expects that: the net is written from these
    # positions, so an oversized part drags the 2D view off into negative
    # space and every margin check on the *view* then fails for a layout the
    # packer never claimed to have made. Park it on the margin instead, still
    # flagged by sheet < 0.
    #
    # Side by side, not all on the same corner: stacked there, every oversized
    # part is drawn on top of every other one, and a view of parts that do not
    # fit then also reads as parts that overlap.
    over = [p for p in parts if p.sheet < 0]
    x = mat.sheet_margin
    for p in over:
        x0, y0, x1, _y1 = p.bounds
        p.translate(x - x0, mat.sheet_margin - y0)
        x += (x1 - x0) + mat.part_gap
    return sheets, over
