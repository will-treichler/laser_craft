"""Assemble islands into cuttable parts and pack them onto laser sheets.

Packing here is deliberately simple: rotate each part to its minimum-area
bounding box, then shelf-pack the boxes. True polygon nesting (no-fit
polygon, as in SVGnest) is worth roughly +20 points of material utilisation
on net-like geometry and is the obvious upgrade, but bbox shelving is 60
lines and gets a first model onto the machine.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiLineString, LineString, Polygon, box
from shapely.ops import unary_union

from .material import Material
from .unfold import Island
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

    def apply(self, fn):
        self.cut = fn(self.cut)
        self.perf_m = fn(self.perf_m)
        self.perf_v = fn(self.perf_v)
        self.engrave = fn(self.engrave)

    def rotate(self, deg, origin=(0, 0)):
        self.apply(lambda g: affinity.rotate(g, deg, origin=origin))

    def translate(self, dx, dy):
        self.apply(lambda g: affinity.translate(g, dx, dy))

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
    """Rotate a part onto its minimum-area bounding box, so it packs tightly."""
    mrr = part.cut.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return
    c = np.asarray(mrr.exterior.coords)[:4]
    edges = np.diff(np.vstack([c, c[:1]]), axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    e = edges[int(np.argmax(lengths))]
    part.rotate(-np.degrees(np.arctan2(e[1], e[0])), origin=tuple(mrr.centroid.coords)[0])


def pack(parts: list[Part], mat: Material) -> tuple[int, list[Part]]:
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
