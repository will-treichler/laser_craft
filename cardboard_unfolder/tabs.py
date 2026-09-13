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
import hashlib
import re
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
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


# How much of a feature a label may occupy. Both are ceilings, and the
# smaller wins. A number as long as the edge it names reads as a label for the
# whole part; at a quarter it reads as belonging to that edge. The face rule
# is what stops a big glyph on a narrow sliver: text wider than half the face
# it sits on has nowhere to be that is not touching something.
ACROSS_FRAC = 0.5       # of the face's narrow dimension; see Material


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


def short_id(name: str, n: int = 8) -> str:
    """A short, stable, engravable tag from a name of any length.

    Label length is the single biggest lever on how many islands can carry an
    id at all - at the 2 mm floor every character costs 1.7 mm of clear room -
    and a name like OLD_VERSION_READ_DESCRIPTION_pikachu_1gen_flowalistik
    cannot be engraved anywhere. A name already short enough is kept as it is,
    because a human-readable tag beats a hash; anything longer is hashed, which
    is stable across runs and still tells two models apart. Hex only, so every
    character exists in the stroke font.
    """
    base = re.sub(r"[^A-Za-z0-9]+", "", name or "").upper()
    if not base:
        return ""
    if len(base) <= n:
        return base
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:n].upper()


def _card(pt, w: float, h: float, ang: float):
    """The rectangle a label will occupy, rotated the way it will be drawn.

    Test the rotated card, not an axis-aligned one. Testing upright and then
    drawing at an angle is not a test: 9 of 209 fold marks passed it and
    still hung over the edge.
    """
    g = box(pt[0] - w / 2.0, pt[1] - h / 2.0, pt[0] + w / 2.0, pt[1] + h / 2.0)
    if ang:
        g = affinity.rotate(g, np.degrees(ang), origin=(float(pt[0]),
                                                        float(pt[1])))
    return g


def _upright(ang: float) -> float:
    """Keep text readable rather than upside down."""
    return ang + np.pi if (ang > np.pi / 2 or ang < -np.pi / 2) else ang


def _try(outline, taken: list, txt: str, h: float, pt, ang: float):
    """(strokes, card) for `txt` at `pt`, or (None, None) if it cannot go there.

    Two things disqualify a spot: leaving the island, and landing on a label
    already placed. Overlapping engraving is not a near miss - two numbers
    burned through each other read as a third number that means nothing.

    Containment is tested cheap-first. The card is the glyph cell's bounding
    box, so fitting it is sufficient but not necessary, because letters do not
    fill their cells; card-only cost 27 points of seam coverage by refusing
    labels whose glyphs had room. Collision uses the card on purpose though -
    two labels whose cells touch are too close to read even if their strokes
    happen to interleave.
    """
    w = strokefont.text_width(txt, h)
    card = _card(pt, w, h, ang)
    for t in taken:
        if card.intersects(t):
            return None, None
    strokes = strokefont.text_strokes(txt, h, pt, ang)
    if outline.contains(card):
        return strokes, card
    for s in strokes:
        if len(s) > 1 and not outline.contains(LineString(s)):
            return None, None
    return strokes, card


def _face_width(isl: Island, face: int, cache: dict) -> float:
    """The narrow dimension of one face, in mm.

    The room a label has is not the length of the edge it names; it is how
    much material is behind that edge. A 60 mm cut along a 3 mm spur has room
    for nothing.
    """
    if face in cache:
        return cache[face]
    ring = isl.faces.get(face)
    if ring is None or len(ring) < 3:
        cache[face] = float("inf")
        return cache[face]
    w, h, _ang, _c = min_area_rect(np.asarray(ring, dtype=float))
    cache[face] = float(min(w, h))
    return cache[face]


def island_labels(isl: Island, plans, mat: Material, body=None) -> list[Label]:
    """Every engraving for one island, in priority order and never overlapping.

    Order is the point. There is not room for everything on a congested net,
    so what gets dropped has to be the least useful thing:

      1. the island id - without it two models cut the same day are one pile
      2. seam numbers - without them the net is not assemblable at all
      3. M / V - useful, and the first thing to give up

    Placed against a shared occupancy map, so a fold mark can never land on a
    seam number that is already there.
    """
    # The BODY, not the outline: relief slots are holes, and a number engraved
    # across one is cut in half by the fold it was meant to explain. On a
    # 3 mm board every fold on this net becomes a slot, so the outline is a
    # poor picture of where there is material.
    o = isl.outline() if body is None else body
    if o.is_empty:
        return []
    big = (max(o.geoms, key=lambda g: g.area)
           if o.geom_type == "MultiPolygon" else o)
    taken: list = []
    out: list[Label] = []
    widths: dict = {}

    # ---------------------------------------------------------------- id
    tag = mat.model_id.strip()
    txt = f"{tag}-{isl.id}" if tag else f"P{isl.id}"
    if not big.is_empty:
        try:
            pt = polylabel(big, tolerance=max(0.05, mat.label_min_height / 10))
        except Exception:                                   # noqa: BLE001
            pt = big.representative_point()
        _w, _h, axis, _c = min_area_rect(shapely.get_coordinates(big))
        done = False
        for h in _steps(mat.label_height, mat.label_min_height):
            for ang in _angles(axis):
                strokes, card = _try(big, taken, txt, h, (pt.x, pt.y),
                                     _upright(ang))
                if strokes is not None:
                    out.append(Label(strokes, txt, h, "part"))
                    taken.append(card)
                    done = True
                    break
            if done:
                break

    # ------------------------------------------------------- seam numbers
    for c in isl.cuts:
        if c.seam < 0:
            continue
        txt = str(c.seam)
        avail = min(c.length * mat.label_edge_frac,
                    _face_width(isl, c.face, widths) * ACROSS_FRAC)
        h = fit_height(txt, avail, mat)
        if h is None:
            continue
        d = c.b - c.a
        d = d / np.linalg.norm(d)
        inward = -_outward(isl, c)
        ang = _upright(np.arctan2(d[1], d[0]))
        # Slide along the edge as well as away from it. The midpoint is the
        # obvious place and often the worst one: a neighbouring cut running
        # back at an angle takes the material right behind it while leaving
        # plenty a quarter of the way along.
        found = None
        for t in (0.5, 0.35, 0.65, 0.25, 0.75):
            base = c.a + (c.b - c.a) * t
            for k in (0.85, 1.2, 1.7, 2.4):
                strokes, card = _try(o, taken, txt, h, base + inward * (h * k),
                                     ang)
                if strokes is not None:
                    found = (strokes, card)
                    break
            if found:
                break
        if found:
            out.append(Label(found[0], txt, h, "seam"))
            taken.append(found[1])

    # --------------------------------------------------------- fold marks
    if mat.label_folds:
        for plan in plans:
            f = plan.fold
            txt = "M" if plan.is_mountain else "V"
            avail = min(f.length * mat.label_edge_frac,
                        _face_width(isl, f.f0, widths) * ACROSS_FRAC,
                        _face_width(isl, f.f1, widths) * ACROSS_FRAC)
            h = fit_height(txt, avail, mat)
            if h is None:
                continue
            d = f.b - f.a
            n = float(np.linalg.norm(d))
            if n < 1e-12:
                continue
            d = d / n
            perp = np.array([-d[1], d[0]])
            mid = (f.a + f.b) / 2.0
            ang = _upright(np.arctan2(d[1], d[0]))
            # On the line, aligned to it. Offsetting a letter clear of the
            # fold spends far more of a congested part than the letter does,
            # so step aside only for a relief slot, which is a real hole.
            clear = 0.0 if plan.width <= 0.0 else plan.width / 2.0 + h * 0.6
            for side in (1.0, -1.0):
                strokes, card = _try(o, taken, txt, h, mid + perp * clear * side,
                                     ang)
                if strokes is not None:
                    out.append(Label(strokes, txt, h, "fold"))
                    taken.append(card)
                    break
                if clear == 0.0:
                    break                   # both sides are the same spot
    return out


def seam_labels(islands: list[Island], mat: Material) -> dict[int, list[Label]]:
    """Seam numbers only. island_labels is what build_parts uses; this stays
    for callers that want one kind on its own."""
    return {i.id: [l for l in island_labels(i, (), mat) if l.kind == "seam"]
            for i in islands}


def part_labels(islands: list[Island], mat: Material,
                prefix: str = "P") -> dict[int, list[Label]]:
    """The island id only. See island_labels."""
    return {i.id: [l for l in island_labels(i, (), mat) if l.kind == "part"]
            for i in islands}


def fold_labels(island: Island, plans, mat: Material) -> list[Label]:
    """The M / V marks only. See island_labels."""
    return [l for l in island_labels(island, plans, mat) if l.kind == "fold"]


def _angles(axis: float):
    """Angles to try a label at, best guesses first.

    The island's own long axis and its perpendicular are where a sliver has
    room, and horizontal is easiest to read; after that just sweep. Three
    angles left 8 of 24 pikachu islands unlabelled, three of which had obvious
    room - a 22.4 x 17.7 mm island with a 7.5 mm inscribed circle among them.
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
