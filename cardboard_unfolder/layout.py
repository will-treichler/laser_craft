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
from .folds import plan_folds, apply_relief, kerf_compensate
from .tabs import place_tabs, seam_labels, part_labels


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


def _mls(segs) -> MultiLineString:
    segs = [s for s in segs if np.linalg.norm(np.asarray(s[-1]) - np.asarray(s[0])) > 1e-9]
    return MultiLineString([LineString(s) for s in segs]) if segs else MultiLineString([])


def build_parts(islands: list[Island], mat: Material) -> list[Part]:
    """Island -> Part: relief slots subtracted, tabs unioned on, kerf applied."""
    tabs, skipped = place_tabs(islands, mat)
    slabels = seam_labels(islands, mat)
    plabels = part_labels(islands, mat)

    parts = []
    for isl in islands:
        plans = plan_folds(isl, mat)
        warns = [p.warn for p in plans if p.warn]

        body = apply_relief(isl.outline(), plans)
        tab_polys = [t.poly for t in tabs.get(isl.id, [])]
        if tab_polys:
            body = unary_union([shapely.set_precision(body, 1e-9)] +
                               [shapely.set_precision(t, 1e-9) for t in tab_polys])

        perf_m, perf_v = [], []
        for p in plans:
            (perf_m if p.is_mountain else perf_v).extend(p.perf)
        # a tab folds against its own base edge
        for t in tabs.get(isl.id, []):
            from .folds import perf_segments
            perf_m.extend(perf_segments(t.base[0], t.base[1], mat))

        strokes = []
        for lab in slabels.get(isl.id, []) + plabels.get(isl.id, []):
            strokes.extend(lab.strokes)

        cut = kerf_compensate(body, mat.kerf)
        necks = narrow_necks(cut, max(3.0, 2.0 * mat.thickness))
        if necks:
            warns.append(
                f"{necks} neck(s) narrower than {max(3.0, 2.0*mat.thickness):.0f}mm - "
                f"the net rejoins itself through a near-pinch and will tear there. "
                f"Re-run with more trials or a different seed, or split this island.")

        parts.append(Part(
            island_id=isl.id, cut=cut,
            perf_m=_mls(perf_m), perf_v=_mls(perf_v),
            engrave=_mls(strokes), warnings=warns))

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
            spot = None
            cands = _corner_candidates(placed_box, m, m, gap)
            tree = STRtree(placed_geo) if placed_geo else None
            for angle in (0.0, 90.0, 180.0, 270.0):
                cand = affinity.rotate(p.cut, angle, origin="centroid")
                bx0, by0, bx1, by1 = cand.bounds
                cw, ch = bx1 - bx0, by1 - by0
                if cw > W or ch > H:
                    continue
                # Buffer once per rotation, not once per candidate position.
                # Buffering is the expensive call here and there are hundreds
                # of positions; doing it inside the loop took the core suite
                # from 36 s to 156 s on its own. One side buffered by the full
                # gap is equivalent to both sides by half.
                grown = cand.buffer(gap, join_style=2)
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
                    spot = (angle, dx, dy, affinity.translate(cand, dx, dy))
                    break
                if spot is not None:
                    break
            if spot is None:
                leftovers.append(p)
                continue
            angle, dx, dy, moved = spot
            p.rotate(angle, origin=tuple(p.cut.centroid.coords)[0])
            b = p.bounds
            p.translate(moved.bounds[0] - b[0], moved.bounds[1] - b[1])
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
        if w < h and h <= W and w <= H:      # lay it down if that helps it fit
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


def pack(parts: list[Part], mat: Material) -> tuple[int, list[Part]]:
    """Pack onto sheets, taking whichever strategy wins on this net.

    Neither strategy dominates. True-shape nesting is far better when parts
    are small enough to share a sheet (a pauldron split to 70% of the bed:
    9 sheets shelved, 5 nested) and worse when a greedy bottom-left placement
    paints itself into a corner (drum12 split to 50%: 5 shelved, 10 nested).
    Running both costs milliseconds and removes the risk entirely.

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
    for fn in (shelf_pack, nest):
        restore(start_state)
        for p in parts:
            p.xform = np.eye(3)
        sheets, over = fn(parts, mat)
        trials.append((sheets, len(over), snap()))

    trials.sort(key=lambda t: (t[0], t[1]))
    sheets, _, winner = trials[0]
    restore(winner)
    return sheets, [p for p in parts if p.sheet < 0]
