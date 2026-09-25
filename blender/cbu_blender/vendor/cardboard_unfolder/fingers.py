"""Interlocking fingers along cut seams - a box joint for cardboard.

Every cut seam has two halves, one on each island. Split the seam into
segments of about `finger_width` and alternate: on segment k one half keeps
its full length to the edge - that is the finger - and the other half is cut
back by the depth - that is the notch; on segment k+1 they swap. Assembled,
each finger fills its partner's notch and the two panels cannot slide along
the seam - they lock while the tape or glue goes on.

**A finger adds nothing.** The mesh edge is the OUTER corner where the two
panels meet, and each panel's board lies inward of its own face. So at a
right-angle corner, a panel cut to the edge already reaches the far face of
the other panel - its end is the finger - and the other panel, cut back by one
board thickness, stops against its inner face. The edge alternates between
the outline and one thickness in: a swing of ONE board thickness, peak to
trough, which is what a box joint is.

The first version grew each finger OUTWARD by the depth as well as cutting the
notch in, so the edge swung by two thicknesses - +/-3 mm, a 6 mm finger on
3 mm board - and every finger stood a board's thickness proud of the outside
of the model once assembled. Its mating test passed, because it tested the
two halves against each other and not against the board.

Complementary by construction, which is the whole requirement. The two halves
of a seam are the same 3D edge, and `build_islands` stores `Cut.a` / `Cut.b`
in the SAME vertex order on both sides (v0 then v1 of the dual arc), so a
parameter t along a -> b names the same physical point on each half. Nothing
has to be matched up geometrically after the fact.

Depth defaults to the board thickness: the notch must clear the board that
fills it, which at a right angle is exactly one thickness. At a dihedral
angle phi between the panels the two boards overlap along t / sin(phi) of
each other's outer face - 3.5 mm at 60 or 120 degrees, 6 mm at 150 - so a
shallow seam wants more; costume tolerance and crushable board mostly do not
care, and the override is there for when they do.

A segment is jointed only if the notch fits in its half (inside its own face,
a board's thickness clear of any other edge, relief slot or notch) and the
finger's material is really there in the other half (not already taken by a
relief slot or by a notch cut from another edge). Otherwise that segment is
left plain on both sides - which is simply a seam without a joint.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.ops import unary_union

from .material import Material
from .unfold import Island, Cut

GRID = 1e-9

# The narrowest finger worth cutting, as a fraction of finger_width. An edge
# too short for two full fingers still gets two narrower ones down to this;
# below it there is no joint, because one finger alone does not interlock.
MIN_FRAC = 0.5


@dataclass
class Joint:
    poly: Polygon
    seam: int
    island: int
    kind: str             # "finger" (kept to the edge) | "notch" (cut back)
    t: tuple              # (t0, t1) along the seam from Cut.a, mm
    face: int = -1        # the face whose edge it sits on
    tool: Polygon | None = None   # what a notch actually cuts: see BITE


def _ring_ccw(ring: np.ndarray) -> bool:
    x, y = ring[:, 0], ring[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) > 0.0


def _corner(ring: np.ndarray, p: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(ring - p, axis=1)))


def inward(island: Island, cut: Cut) -> np.ndarray:
    """Unit normal to the cut pointing INTO its face.

    Read from the ring order, not from the centroid: the centroid of a
    concave face can sit on the far side of its own edge, which is the exact
    mistake `unfold_tree` once made. The face's ring visits a and b as
    consecutive corners; interior is left of that direction for a
    counter-clockwise ring and right of it otherwise.
    """
    ring = np.asarray(island.faces[cut.face], dtype=float)
    i, j = _corner(ring, cut.a), _corner(ring, cut.b)
    forward = j == (i + 1) % len(ring)        # the ring walks a -> b
    d = (cut.b - cut.a) / np.linalg.norm(cut.b - cut.a)
    left = np.array([-d[1], d[0]])
    return left if forward == _ring_ccw(ring) else -left


def corner_angles(island: Island, cut: Cut) -> tuple[float, float]:
    """Interior angle of the cut's face at a and at b, radians."""
    ring = np.asarray(island.faces[cut.face], dtype=float)
    n = len(ring)
    sign = 1.0 if _ring_ccw(ring) else -1.0
    out = []
    for p in (cut.a, cut.b):
        i = _corner(ring, p)
        u = ring[(i + 1) % n] - ring[i]
        v = ring[i - 1] - ring[i]
        cross = sign * float(u[0] * v[1] - u[1] * v[0])
        ang = float(np.arctan2(cross, float(np.dot(u, v))))
        out.append(ang if ang > 0 else ang + 2 * np.pi)
    return out[0], out[1]


def corner_inset(angle: float, depth: float, wall: float) -> float:
    """How far from a corner of interior `angle` a notch `depth` deep may start.

    The notch's inner corner sits at (t, depth) in the edge's frame, and the
    face's other edge leaves the corner at `angle`; keeping `wall` of board
    between them is t*sin(angle) - depth*cos(angle) >= wall. Never less than
    the depth itself: three panels meet at a vertex, and the corner cube one
    board thick belongs to none of them - a finger there fights the third.
    """
    if angle >= np.pi - 1e-9:           # straight or reflex: nothing to hit
        return depth
    need = (wall + depth * np.cos(angle)) / max(np.sin(angle), 1e-9)
    return max(depth, float(need))


def segments(length: float, mat: Material, e0: float | None = None,
             e1: float | None = None) -> list[tuple[float, float]]:
    """Where the fingers go along an edge of this length, or [] for none.

    `e0` / `e1` are the insets at the a and b ends (see `corner_inset`); the
    depth by default. What is left is split into equal segments as close to
    `finger_width` as a whole number allows, and at least two - an
    alternating pattern needs one finger for each half.
    """
    d = mat.finger_depth_used
    w = mat.finger_width
    if w <= 0.0 or d <= 0.0:
        return []
    e0 = d if e0 is None else e0
    e1 = d if e1 is None else e1
    usable = length - e0 - e1
    if usable < 2.0 * MIN_FRAC * w:
        return []
    n = max(2, int(round(usable / w)))
    step = usable / n
    return [(e0 + k * step, e0 + (k + 1) * step) for k in range(n)]


def _rect(a, u, v, t0, t1, s0, s1) -> Polygon:
    """Rectangle in the cut's own frame: t along a->b, s along v."""
    return shapely.set_precision(Polygon([a + u * t0 + v * s0, a + u * t1 + v * s0,
                                          a + u * t1 + v * s1, a + u * t0 + v * s1]),
                                 GRID)


def _probe(rect: Polygon, a, u, v, side: float, wall: float) -> Polygon:
    """`rect` grown by `wall` in every direction but back across the edge.

    Round, because the requirement is a DISTANCE: an upright box of clearance
    pokes its corner out of a 60 degree face that the notch itself clears
    comfortably, which is every notch on a triangulated sphere.
    """
    L = 1e4
    half = Polygon([a - u * L + v * side * 1e-6, a + u * L + v * side * 1e-6,
                    a + u * L + v * side * L, a - u * L + v * side * L])
    return rect.buffer(wall, quad_segs=4).intersection(half)


def place_fingers(islands: list[Island], mat: Material,
                  bodies: dict | None = None,
                  why: dict | None = None) -> dict[int, list[Joint]]:
    """Fingers and notches for every seam that can take them, by island.

    `bodies` is each island's outline with its relief slots already taken
    out; a notch must stay clear of those slots as well as of the other
    faces. Without it the plain outline is used. `why`, if given, counts
    what was refused, by reason.
    """
    why = {} if why is None else why

    def refuse(reason):
        why[reason] = why.get(reason, 0) + 1

    by_isl = {i.id: i for i in islands}
    seams: dict[int, list[tuple[int, Cut]]] = {}
    for isl in islands:
        for c in isl.cuts:
            if c.seam >= 0:
                seams.setdefault(c.seam, []).append((isl.id, c))

    d = mat.finger_depth_used
    # Material left between a notch and anything else. One board thickness:
    # less than that and the web tears off the moment the finger is pushed
    # home. Derived, not tuned.
    wall = max(mat.thickness, 1.0)

    base = {}
    for i in islands:
        b = (bodies or {}).get(i.id)
        if b is None:
            b = i.outline()
        base[i.id] = prep(shapely.set_precision(b, GRID))
    faces = {}
    notches: dict[int, list[tuple[int, Polygon]]] = {i.id: [] for i in islands}
    kept: dict[int, list[tuple[int, Polygon]]] = {i.id: [] for i in islands}
    out: dict[int, list[Joint]] = {i.id: [] for i in islands}

    for seam in sorted(seams):
        sides = seams[seam]
        if len(sides) != 2:
            continue
        sides = sorted(sides, key=lambda s: (s[0], s[1].face))
        # Either half may be the one notched at either end, so each end is
        # inset for the sharper of the two faces' corners there.
        angs = [corner_angles(by_isl[i], c) for i, c in sides]
        e0 = max(corner_inset(a[0], d, wall) for a in angs)
        e1 = max(corner_inset(a[1], d, wall) for a in angs)
        segs = segments(sides[0][1].length, mat, e0, e1)
        if not segs:
            refuse("edge too short")
            continue
        frames = []
        for isl_id, cut in sides:
            u = (cut.b - cut.a) / cut.length
            frames.append((isl_id, cut, u, inward(by_isl[isl_id], cut)))
            key = (isl_id, cut.face)
            if key not in faces:
                faces[key] = prep(shapely.set_precision(
                    Polygon(by_isl[isl_id].faces[cut.face]), GRID).buffer(1e-6))

        for k, (t0, t1) in enumerate(segs):
            f_isl, f_cut, fu, fv = frames[k % 2]
            n_isl, n_cut, nu, nv = frames[1 - k % 2]
            # Both are the same rectangle INSIDE their own half: the notch is
            # what is cut away, the finger what is kept to fill it.
            finger = _rect(f_cut.a, fu, fv, t0, t1, 0.0, d)
            notch = _rect(n_cut.a, nu, nv, t0, t1, 0.0, d)

            # The finger's board must really be there: in its own face, not a
            # relief slot, and not already cut away by a notch from another
            # edge of the same island. Tested a hair in from the edge: the
            # rectangle's outer side lies ON the body's boundary, and a strict
            # `contains` against the unbuffered body then fails on float noise
            # - it refused 240 of icosphere2's fingers before this.
            held = _rect(f_cut.a, fu, fv, t0, t1, 1e-6, d)
            if not (faces[(f_isl, f_cut.face)].contains(held)
                    and base[f_isl].contains(held)):
                refuse("finger has no board behind it")
                continue
            if any(finger.intersection(q).area > 1e-9 for _c, q in notches[f_isl]):
                refuse("finger meets a notch")
                continue
            # The notch, and a wall round it, must be solid board of its own
            # face: not a neighbouring face, not a relief slot, not a notch
            # cut from another edge - and not another seam's finger.
            nprobe = _probe(notch, n_cut.a, nu, nv, 1.0, wall)
            if not faces[(n_isl, n_cut.face)].contains(nprobe):
                refuse("notch leaves its face")
                continue
            if not base[n_isl].contains(nprobe):
                refuse("notch meets a relief slot")
                continue
            if any(c is not n_cut and nprobe.intersects(q)
                   for c, q in notches[n_isl]):
                refuse("notch meets another notch")
                continue
            if any(c is not n_cut and notch.intersection(q).area > 1e-9
                   for c, q in kept[n_isl]):
                refuse("notch meets a finger")
                continue

            kept[f_isl].append((f_cut, finger))
            notches[n_isl].append((n_cut, notch))
            out[f_isl].append(Joint(finger, seam, f_isl, "finger", (t0, t1),
                                    f_cut.face))
            out[n_isl].append(Joint(notch, seam, n_isl, "notch", (t0, t1),
                                    n_cut.face,
                                    _rect(n_cut.a, nu, nv, t0, t1, -BITE, d)))
    return out


# How far a notch reaches PAST the edge, mm. Its outer side lies ON the
# body's edge, and after snapping to GRID the two can sit a fraction of a
# nanometre apart - the difference then leaves a hairline of board along the
# edge, a sliver the laser traces as a separate piece. Extending the notch a
# tenth of a micron outward makes it bite across the edge instead of meeting
# it. Outward only: growing it on every side shaved the fingers beside it,
# 0.17 mm2 over a cone net, which is small and is also not what was asked for.
BITE = 1e-4


def apply_fingers(body, joints: list[Joint]):
    """Cut the notches out of a part. Fingers are what is left: they add
    nothing, so a fingered part never reaches past its own outline."""
    cut = [j.tool if j.tool is not None else j.poly
           for j in joints if j.kind == "notch"]
    if not cut:
        return body
    # Floating overlay, snapped once at the end. Snapping the body FIRST puts
    # GEOS in fixed-precision mode, and a body with its relief slots is tens
    # of thousands of vertices: 10.9 s against 2.8 s on an icosahedron net,
    # for a result that differs by 1e-9 mm2. BITE is what makes the floating
    # overlay safe - nothing here meets exactly on a line.
    return shapely.set_precision(body.difference(unary_union(cut)), GRID)


def fingered_seams(joints: dict[int, list[Joint]]) -> set[int]:
    return {j.seam for js in joints.values() for j in js}
