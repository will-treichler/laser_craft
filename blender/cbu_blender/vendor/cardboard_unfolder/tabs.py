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
from shapely.geometry import Polygon, box
from shapely import affinity
from shapely.ops import unary_union, polylabel

from .material import Material
from .unfold import Island, Cut, min_area_rect
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
    height: float = 0.0   # mm, as drawn - NOT recoverable from the strokes,
                          # whose extent is the rotated glyph box
    kind: str = ""        # seam | fold | part


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


def fit_height(text: str, avail: float, mat: Material) -> float | None:
    """The largest legible height for `text` in `avail` mm, or None.

    Width is linear in height, so the fitting height is one division - there
    is nothing to search. Capped at label_height because bigger is not better
    on a part, and refused below label_min_height: a 1 mm character on
    corrugated board is a burnt smudge that reads as nothing, and a smudge
    where a seam number should be is worse than a bare edge, because the
    builder trusts it.
    """
    if not text or avail <= 0.0:
        return None
    unit = strokefont.text_width(text, 1.0)
    if unit <= 0.0:
        return None
    h = min(mat.label_height, avail / unit)
    return h if h >= mat.label_min_height else None


def _upright(ang: float) -> float:
    """Keep text readable rather than upside down."""
    return ang + np.pi if (ang > np.pi / 2 or ang < -np.pi / 2) else ang


def seam_labels(islands: list[Island], mat: Material) -> dict[int, list[Label]]:
    """Engrave the seam number just inside each cut edge, on both halves, so
    the two pieces that mate can be found by matching numbers.

    This is the annotation that decides whether a net is assemblable at all: a
    60-piece net with unlabelled edges is a puzzle, the same net with matching
    numbers either side of every seam is a job.
    """
    out: dict[int, list[Label]] = {i.id: [] for i in islands}
    for isl in islands:
        for c in isl.cuts:
            if c.seam < 0:
                continue
            txt = str(c.seam)
            # 80% of the edge, so a number never runs into the corners
            h = fit_height(txt, c.length * 0.8, mat)
            if h is None:
                continue
            d = c.b - c.a
            d = d / np.linalg.norm(d)
            inward = -_outward(isl, c)
            mid = (c.a + c.b) / 2.0 + inward * (h * 0.85)
            out[isl.id].append(Label(
                strokefont.text_strokes(
                    txt, h, mid, _upright(np.arctan2(d[1], d[0]))),
                txt, h, "seam"))
    return out


def fold_labels(island: Island, plans, mat: Material) -> list[Label]:
    """M or V beside every fold, so the builder knows which way it bends.

    Which way a crease goes is not recoverable from a flat sheet, and getting
    one wrong on an assembled shell means peeling it apart again. The mark
    clears the relief slot - `plans` carries each fold's actual opening, so the
    offset is measured rather than guessed - and goes on whichever side of the
    fold is inside the island, because a fold at the island's edge has
    material on one side only.
    """
    if not mat.label_folds:
        return []
    o = island.outline()
    out = []
    for plan in plans:
        f = plan.fold
        txt = "M" if plan.is_mountain else "V"
        h = fit_height(txt, f.length * 0.5, mat)
        if h is None:
            continue
        d = f.b - f.a
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            continue
        d = d / n
        perp = np.array([-d[1], d[0]])
        mid = (f.a + f.b) / 2.0
        clear = plan.width / 2.0 + h * 0.9
        ang = _upright(np.arctan2(d[1], d[0]))
        w = strokefont.text_width(txt, h)
        for side in (1.0, -1.0):
            pt = mid + perp * clear * side
            if o.contains(box(pt[0] - w / 2, pt[1] - h / 2,
                              pt[0] + w / 2, pt[1] + h / 2)):
                out.append(Label(
                    strokefont.text_strokes(txt, h, pt, ang), txt, h, "fold"))
                break
    return out


def part_labels(islands: list[Island], mat: Material,
                prefix: str = "P") -> dict[int, list[Label]]:
    """The model identifier and island number, engraved inside each island.

    Cut two models on one afternoon and the pieces are indistinguishable
    without this: both are grey polygons carrying seam numbers that each start
    at 1. `Material.model_id` names the model, and falls back to `prefix` so
    the old P0/P1 behaviour survives when nobody sets one.

    Placed at the pole of inaccessibility, not the centroid: the centroid of an
    L-shaped or horseshoe island is frequently outside it, and
    representative_point only promises to be *somewhere* inside - which on a
    long thin island is somewhere with no room. polylabel finds the point
    furthest from any edge, which is exactly where a label wants to be.
    """
    out: dict[int, list[Label]] = {}
    tag = mat.model_id.strip()
    for isl in islands:
        txt = f"{tag}-{isl.id}" if tag else f"{prefix}{isl.id}"
        o = isl.outline()
        big = (max(o.geoms, key=lambda g: g.area)
               if o.geom_type == "MultiPolygon" else o)
        out[isl.id] = []
        if big.is_empty:
            continue
        try:
            pt = polylabel(big, tolerance=max(0.05, mat.label_min_height / 10))
        except Exception:                                   # noqa: BLE001
            pt = big.representative_point()
        # Room here is two-dimensional, so unlike an edge label this one has to
        # be tried rather than divided out. Try the island's own long axis as
        # well as horizontal: these nets are full of long thin slivers with
        # plenty of room along them and none across, and horizontal-only left
        # 10 of 24 pikachu islands unlabelled.
        _w, _h, axis, _c = min_area_rect(shapely.get_coordinates(big))
        for h in _steps(mat.label_height, mat.label_min_height):
            w = strokefont.text_width(txt, h)
            for ang in _angles(axis):
                card = box(pt.x - w / 2, pt.y - h / 2,
                           pt.x + w / 2, pt.y + h / 2)
                if ang:
                    card = affinity.rotate(card, np.degrees(ang),
                                           origin=(pt.x, pt.y))
                if big.contains(card):
                    out[isl.id] = [Label(strokefont.text_strokes(
                        txt, h, (pt.x, pt.y), _upright(ang)), txt, h, "part")]
                    break
            if out[isl.id]:
                break
    return out


def _angles(axis: float):
    """Angles to try a label at, best guesses first.

    The island's own long axis and its perpendicular are where a sliver has
    room, and horizontal is easiest to read; after that just sweep. Three
    angles left 8 of 24 pikachu islands unlabelled, three of which had
    obvious room - a 22.4 x 17.7 mm island with a 7.5 mm inscribed circle
    among them.
    """
    yield 0.0
    yield axis
    yield axis + np.pi / 2
    for k in range(1, 12):
        yield k * np.pi / 12.0


def _steps(top: float, floor: float, step: float = 0.25):
    """Heights to try, largest first, never below the floor."""
    h = top
    while h >= floor - 1e-9:
        yield h
        h -= step
