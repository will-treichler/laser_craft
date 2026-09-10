"""Glue tabs and seam labelling.

Tabs are trapezoids grown outward from a cut edge. Placement is a real
constraint problem - a tab must not collide with its own island or with
another tab - and the literature solves side assignment as 2-SAT. For an MVP
a greedy pass with collision rejection and a shrink-then-skip fallback gets
the same answer on nearly every seam, and any seam it gives up on is simply
taped instead of glued.

Seam numbers matter more than they sound. A net of 60 pieces with unlabelled
edges is not assemblable by a human; the same net with matching numbers on
both halves of every seam is straightforward.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .material import Material
from .unfold import Island, Cut
from . import strokefont


@dataclass
class Tab:
    poly: Polygon
    seam: int
    base: tuple           # (a, b) the cut edge it hinges on
    island: int


@dataclass
class Label:
    strokes: list         # list of (n,2) arrays
    text: str


def _outward(island: Island, cut: Cut) -> np.ndarray:
    """Unit normal to the cut edge pointing away from its face."""
    d = cut.b - cut.a
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    centroid = island.faces[cut.face].mean(axis=0)
    mid = (cut.a + cut.b) / 2.0
    return -n if np.dot(centroid - mid, n) > 0 else n


def _trapezoid(a: np.ndarray, b: np.ndarray, out: np.ndarray,
               depth: float, inset_deg: float) -> Polygon:
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    inset = min(depth * np.tan(np.radians(inset_deg)), 0.42 * L)
    return Polygon([a, b,
                    b + out * depth - d * inset,
                    a + out * depth + d * inset])


def place_tabs(islands: list[Island], mat: Material,
               grid: float = 1e-9) -> tuple[dict[int, list[Tab]], list[int]]:
    """One tab per seam, on whichever side it fits. Returns (tabs by island,
    seams that got no tab)."""
    # group cut records by seam so we can choose a side
    seams: dict[int, list[tuple[int, Cut]]] = {}
    for isl in islands:
        for c in isl.cuts:
            if c.seam >= 0:
                seams.setdefault(c.seam, []).append((isl.id, c))

    by_isl = {i.id: i for i in islands}
    outlines = {i.id: shapely.set_precision(i.outline(), grid) for i in islands}
    placed: dict[int, list[Tab]] = {i.id: [] for i in islands}
    occupied = {i.id: outlines[i.id] for i in islands}
    skipped = []

    for seam in sorted(seams):
        sides = seams[seam]
        got = False
        # prefer the longer edge, then the lower island id, for determinism
        for isl_id, cut in sorted(sides, key=lambda s: (-s[1].length, s[0])):
            if cut.length < mat.tab_min_edge:
                continue
            isl = by_isl[isl_id]
            out = _outward(isl, cut)
            for shrink in (1.0, 0.7, 0.45):
                depth = mat.tab_depth * shrink
                if depth < 3.0:
                    break
                t = _trapezoid(cut.a, cut.b, out, depth, mat.tab_inset_deg)
                t = shapely.set_precision(t, grid)
                # a sliver of shared edge is fine; real area overlap is not
                if t.intersection(occupied[isl_id]).area > 1e-6:
                    continue
                placed[isl_id].append(Tab(t, seam, (cut.a, cut.b), isl_id))
                occupied[isl_id] = unary_union([occupied[isl_id], t])
                got = True
                break
            if got:
                break
        if not got:
            skipped.append(seam)
    return placed, skipped


def seam_labels(islands: list[Island], mat: Material) -> dict[int, list[Label]]:
    """Engrave the seam number just inside each cut edge, on both halves, so
    the two pieces that mate can be found by matching numbers."""
    out: dict[int, list[Label]] = {i.id: [] for i in islands}
    h = mat.label_height * 0.8
    for isl in islands:
        for c in isl.cuts:
            if c.seam < 0 or c.length < h * 2.2:
                continue
            d = c.b - c.a
            d = d / np.linalg.norm(d)
            inward = -_outward(isl, c)
            mid = (c.a + c.b) / 2.0 + inward * (h * 0.85)
            ang = np.arctan2(d[1], d[0])
            if ang > np.pi / 2 or ang < -np.pi / 2:      # keep text upright
                ang += np.pi
            txt = str(c.seam)
            out[isl.id].append(
                Label(strokefont.text_strokes(txt, h, mid, ang), txt))
    return out


def part_labels(islands: list[Island], mat: Material,
                prefix: str = "P") -> dict[int, list[Label]]:
    """A part ID engraved at the island's interior centre."""
    out: dict[int, list[Label]] = {}
    for isl in islands:
        o = isl.outline()
        pt = o.representative_point()
        txt = f"{prefix}{isl.id}"
        out[isl.id] = [Label(
            strokefont.text_strokes(txt, mat.label_height, (pt.x, pt.y)), txt)]
    return out
