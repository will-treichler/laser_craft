"""Interlocking fingers along cut seams - a box joint for cardboard.

Every cut seam has two halves, one on each island. Split the seam into
segments of about `finger_width` and alternate: on segment k one half grows a
finger OUTWARD by the depth and the other half has the same rectangle cut
INWARD as a notch; on segment k+1 they swap. Assembled, each finger drops
into its partner's notch and the two panels cannot slide along the seam - they
lock while the tape or glue goes on.

Complementary by construction, which is the whole requirement. The two halves
of a seam are the same 3D edge, and `build_islands` stores `Cut.a` / `Cut.b`
in the SAME vertex order on both sides (v0 then v1 of the dual arc), so a
parameter t along a -> b names the same physical point on each half. Nothing
has to be matched up geometrically after the fact.

Depth defaults to the board thickness: a finger has to pass through the
thickness of the board it mates with, which is what a box joint is. At other
dihedral angles the exact figure differs; costume tolerance does not care,
and the override is there for when it does.

A segment is only jointed if BOTH halves can take it. A finger that would hit
its own island, or a notch that would cut into a neighbouring face, a relief
slot or another notch, leaves that segment plain on both sides - never on one
side only, which would be a finger with nowhere to go.
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
    kind: str             # "finger" (added) | "notch" (removed)
    t: tuple              # (t0, t1) along the seam from Cut.a, mm
    face: int = -1        # the face whose edge it sits on


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
    # Material left between a notch and anything else, and room kept round a
    # finger. One board thickness: less than that and the web tears off the
    # moment the finger is pushed home. Derived, not tuned.
    wall = max(mat.thickness, 1.0)

    outline = {i.id: shapely.set_precision(i.outline(), GRID) for i in islands}
    base = {}
    for i in islands:
        b = (bodies or {}).get(i.id, outline[i.id])
        base[i.id] = prep(shapely.set_precision(b, GRID))
    faces = {}
    occupied = dict(outline)          # outline plus every finger accepted
    notches: dict[int, list[tuple[int, Polygon]]] = {i.id: [] for i in islands}
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
            finger = _rect(f_cut.a, fu, fv, t0, t1, 0.0, -d)
            notch = _rect(n_cut.a, nu, nv, t0, t1, 0.0, d)

            # The finger must not touch its own island anywhere but its base.
            probe = _probe(finger, f_cut.a, fu, fv, -1.0, wall)
            if probe.intersection(occupied[f_isl]).area > 1e-9:
                refuse("finger hits its island")
                continue
            # The notch, and a wall round it, must be solid board of its own
            # face: not a neighbouring face, not a relief slot, not a notch
            # cut from another edge.
            nprobe = _probe(notch, n_cut.a, nu, nv, 1.0, wall)
            if not faces[(n_isl, n_cut.face)].contains(nprobe):
                refuse("notch leaves its face")
                continue
            if not base[n_isl].contains(nprobe):
                refuse("notch meets a relief slot")
                continue
            if any(c is not n_cut and nprobe.intersects(p)
                   for c, p in notches[n_isl]):
                refuse("notch meets another notch")
                continue

            occupied[f_isl] = unary_union([occupied[f_isl], finger])
            notches[n_isl].append((n_cut, notch))
            out[f_isl].append(Joint(finger, seam, f_isl, "finger", (t0, t1),
                                    f_cut.face))
            out[n_isl].append(Joint(notch, seam, n_isl, "notch", (t0, t1),
                                    n_cut.face))
    return out


# How far a finger or notch is grown before the boolean, mm. A finger's base
# lies ON the body's edge, and after snapping to GRID the two can sit a
# fraction of a nanometre apart - the union then keeps the finger as a
# separate piece, a loose rectangle that falls out of the sheet. Growing it
# by a tenth of a micron makes it bite across the edge instead of meeting it.
BITE = 1e-4


def apply_fingers(body, joints: list[Joint]):
    """Grow the fingers onto a part and cut the notches out of it."""
    if not joints:
        return body
    add = [j.poly.buffer(BITE, join_style="mitre")
           for j in joints if j.kind == "finger"]
    cut = [j.poly.buffer(BITE, join_style="mitre")
           for j in joints if j.kind == "notch"]
    # Floating overlay, snapped once at the end. Snapping the body FIRST puts
    # GEOS in fixed-precision mode for both booleans, and a body with its
    # relief slots is tens of thousands of vertices: 10.9 s against 2.8 s on
    # an icosahedron net, for a result that differs by 1e-9 mm2. BITE is what
    # makes the floating overlay safe - nothing here meets exactly on a line.
    g = body
    if add:
        g = g.union(unary_union(add))
    if cut:
        g = g.difference(unary_union(cut))
    return shapely.set_precision(g, GRID)


def fingered_seams(joints: dict[int, list[Joint]]) -> set[int]:
    return {j.seam for js in joints.values() for j in js}
