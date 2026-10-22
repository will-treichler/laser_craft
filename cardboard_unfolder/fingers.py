"""Interlocking fingers along cut seams - a box joint for cardboard.

Every cut seam has two halves, one on each island. Split the seam into
segments of about `finger_width` and alternate: on segment k one half keeps
its full length to the edge - that is the finger - and the other half is cut
back by the depth - that is the notch; on segment k+1 they swap. Assembled,
each finger fills its partner's notch and the two panels cannot slide along
the seam - they lock while the tape or glue goes on.

**The mesh is the MID-PLANE of the board** (the user's call, 2026-10-04):
each board extends half a thickness either side of its face. So a finger
reaches HALF the depth past the mesh edge and the notch opposite it is cut
back half the depth: the edge swings by one depth in all, centred on the
mesh. At a right angle that is exact - the finger ends flush with the other
board's outer face, and the notch clears its inner face.

Two earlier models, both wrong in practice. The first grew each finger out by
a whole depth AND cut the notch a whole depth in: a 6 mm swing on 3 mm board,
a board's thickness proud of the model. The second treated the mesh as the
OUTER surface, so a finger added nothing and the notch went a whole depth in
- consistent with that assumption, but "as drawn the fingers will not
intermesh": in the 2D view every notch faced a plain edge.

A finger stands OUT of its part, into what is empty sheet in the flat - which
in a narrow V (a slit inside one island) is not empty for long: near the
apex the other half of the V is right there. A finger is therefore kept only
while it stays a part gap clear of every other board in its island, which is
what ends a joint short of a sharp corner and leaves the seam plain from
there to the point.

Complementary by construction, which is the whole requirement. The two halves
of a seam are the same 3D edge, and `build_islands` stores `Cut.a` / `Cut.b`
in the SAME vertex order on both sides (v0 then v1 of the dual arc), so a
parameter t along a -> b names the same physical point on each half. Nothing
has to be matched up geometrically after the fact.

Depth is the whole swing, half out and half in, and defaults to the board
thickness - exact at a right angle. At other angles it is not, and one global
depth is the user's decision. The figures below were measured for the OLD
outer-surface model and do not carry over to the mid-plane one unmeasured;
what they show is the shape of the dependence. With both ends cut square to their own faces, the depth
one board reaches into the other at interior angle phi is t*sin(phi) for an
obtuse corner - 2.6 mm at 120 degrees, 1.5 at 150, 0.26 at 175 on 3 mm board
- and t*cot(phi/2) for an acute one - 5.2 mm at 60, 7.2 at 45. So a shallow
seam needs LESS than a thickness, not more, and a concave seam (an inside
corner) needs none at all: the boards only touch at the edge. Measured by
intersecting the two board cross-sections, not derived by hand - the first
version of this paragraph claimed t / sin(phi), which is the wrong geometry
and says 6 mm at 150 degrees where the boards overlap by 1.5.

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
from .folds import hole_disc

GRID = 1e-9

# The narrowest finger worth cutting, as a fraction of finger_width. An edge
# too short for two full fingers still gets two narrower ones down to this;
# below it there is no joint, because one finger alone does not interlock.
MIN_FRAC = 0.5


@dataclass
class Joint:
    poly: Polygon         # a finger: what is ADDED past the edge
                          # a notch: what is cut away inside it
    seam: int
    island: int
    kind: str             # "finger" (stands out) | "notch" (cut back)
    t: tuple              # (t0, t1) along the seam from Cut.a, mm
    face: int = -1        # the face whose edge it sits on
    tool: Polygon | None = None   # what is actually unioned / cut: see BITE


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
    d = mat.finger_depth_used / 2.0     # the corner cube is half a board
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


def _relieved(p, centres) -> bool:
    return any(float(np.hypot(*(np.asarray(p) - c))) < 1e-6 for c in centres)


def _halfplane(apex, along, side, L: float = 1e4) -> Polygon:
    """Everything on `side` of the line through `apex` along `along`, moved a
    hair off the line so two fingers clipped either side of it never touch."""
    n = np.array([-along[1], along[0]]) * side
    o = np.asarray(apex) + n * 1e-6
    return Polygon([o - along * L, o + along * L,
                    o + along * L + n * L, o - along * L + n * L])


def place_fingers(islands: list[Island], mat: Material,
                  bodies: dict | None = None,
                  why: dict | None = None, holes: dict | None = None,
                  hole_r: float = 0.0,
                  trace: list | None = None) -> dict[int, list[Joint]]:
    """Fingers and notches for every seam that can take them, by island.

    `bodies` is each island's outline with its relief slots already taken
    out; a notch must stay clear of those slots as well as of the other
    faces. Without it the plain outline is used. `why`, if given, counts
    what was refused, by reason; `trace`, if given, collects (seam, (t0, t1),
    reason) for every refusal, which is what answers "why does THIS seam
    stop short" when a count cannot.

    `holes` maps an island to the vertices with a corner relief hole of
    radius `hole_r` (`folds.apply_corner_holes`). A seam ending at one runs
    its joint back to the hole instead of stopping at the corner inset. And
    where both halves of a slit meet at it - the apex of a V inside one
    island - their fingers point at each other across a gap that closes to
    nothing, so each is clipped at the V's centreline: the joint carries on
    as one interlocking line, the two halves sharing the slit between them.
    """
    holes = holes or {}
    why = {} if why is None else why
    at = {"seam": None, "t": None}

    def refuse(reason):
        why[reason] = why.get(reason, 0) + 1
        if trace is not None:
            trace.append((at["seam"], at["t"], reason))

    by_isl = {i.id: i for i in islands}
    seams: dict[int, list[tuple[int, Cut]]] = {}
    for isl in islands:
        for c in isl.cuts:
            if c.seam >= 0:
                seams.setdefault(c.seam, []).append((isl.id, c))

    # Half the depth each way from the mesh edge: out for a finger, in for a
    # notch (the mid-plane model, module docstring).
    d = mat.finger_depth_used / 2.0
    # Material left between a notch and anything else. One board thickness:
    # less than that and the web tears off the moment the finger is pushed
    # home. Derived, not tuned.
    wall = max(mat.thickness, 1.0)
    # How far a finger must stay from any other board of its island in the
    # flat sheet: the gap the packer keeps between parts, for the same reason
    # - closer and the laser fuses them.
    clear = max(mat.part_gap, mat.kerf)

    base = {}
    for i in islands:
        b = (bodies or {}).get(i.id)
        if b is None:
            b = i.outline()
        base[i.id] = prep(shapely.set_precision(b, GRID))
    faces, raw = {}, {}
    notches: dict[int, list[tuple[int, Polygon]]] = {i.id: [] for i in islands}
    # The board BEHIND each finger, inside the edge: no other seam may notch it.
    kept: dict[int, list[tuple[int, Polygon]]] = {i.id: [] for i in islands}
    # The fingers themselves, outside the edge: none may meet another.
    stand: dict[int, list[Polygon]] = {i.id: [] for i in islands}
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
        # A relieved end runs back to the VERTEX: the notch bottom, half a
        # depth in, then meets the hole (radius half a depth) at a tangent.
        # Starting at the hole's edge left the notch's end wall cutting
        # across the circle - a spike of board pointing at the vertex, the
        # user's screenshot. At the apex of a V each half's fingers stop at
        # the centreline (docstring).
        clip = {}
        near: dict[int, list] = {0: [], 1: []}
        for end, p_of in ((0, lambda c: c.a), (1, lambda c: c.b)):
            hit = [_relieved(p_of(c), holes.get(i, ())) for i, c in sides]
            if not any(hit):
                continue
            for j, (_i, c) in enumerate(sides):
                if hit[j]:
                    near[j].append(np.asarray(p_of(c), float))
            if end == 0:
                e0 = 0.0
            else:
                e1 = 0.0
            (i0, c0), (i1, c1) = sides
            apex = np.asarray(p_of(c0), float)
            if not (i0 == i1 and all(hit) and
                    np.hypot(*(apex - np.asarray(p_of(c1), float))) < 1e-6):
                continue
            far = [np.asarray(c.b if end == 0 else c.a, float) - apex
                   for _i, c in sides]
            far = [f / np.linalg.norm(f) for f in far]
            along = far[0] + far[1]
            if np.linalg.norm(along) < 1e-12:
                continue
            along = along / np.linalg.norm(along)
            for j in (0, 1):
                side = 1.0 if (along[0] * far[j][1] - along[1] * far[j][0]) > 0 \
                    else -1.0
                clip[j] = _halfplane(apex, along, side)
        # A segment reaching within a wall of the hole's edge is the joint's
        # last before the hole, and its notch is held to its own face rather
        # than to a wall from every edge of it: the face narrows to its
        # corner there, and the wall rule alone stopped the joint ~15 mm
        # short of the hole on the helmet ("notch leaves its face" across a
        # whole segment, not just beside the hole).
        # the hole's own neighbourhood: inside it a notch is not clipped to
        # its face, or the face's next edge - a fold running close beside the
        # seam - put a step in the notch bottom just short of the tangent
        free = {j: unary_union([shapely.Point(p).buffer(2.0 * hole_r)
                                for p in ps]) if ps else None
                for j, ps in near.items()}
        near = {j: unary_union([shapely.Point(p).buffer(hole_r + wall)
                                for p in ps]) if ps else None
                for j, ps in near.items()}
        segs = segments(sides[0][1].length, mat, e0, e1)
        at["seam"], at["t"] = seam, None
        if not segs:
            refuse("edge too short")
            continue
        frames = []
        for isl_id, cut in sides:
            u = (cut.b - cut.a) / cut.length
            frames.append((isl_id, cut, u, inward(by_isl[isl_id], cut)))
            key = (isl_id, cut.face)
            if key not in faces:
                raw[key] = shapely.set_precision(
                    Polygon(by_isl[isl_id].faces[cut.face]), GRID)
                faces[key] = prep(raw[key].buffer(1e-6))

        # A seam is jointed as a whole or not at all. Segments are refused
        # one at a time, so without this a seam could end with every notch on
        # ONE half - a lone notch with nothing opposite it, which is not a
        # joint, just a hole in one panel. 6 of the helmet's 44 seams did,
        # where relief slots crowd the inner ends of folds. Each segment
        # still sees the ones accepted before it, so this changes nothing
        # about which are accepted; it only takes them all back at the end.
        mark = {i: (len(notches[i]), len(kept[i]), len(stand[i]), len(out[i]))
                for i, _c, _u, _v in frames}
        halves = set()

        for k, (t0, t1) in enumerate(segs):
            at["t"] = (round(t0, 2), round(t1, 2))
            f_isl, f_cut, fu, fv = frames[k % 2]
            n_isl, n_cut, nu, nv = frames[1 - k % 2]
            # The finger stands OUT of its half by d; the notch goes d INTO
            # the other. `root` is the finger's own board just inside the
            # edge, which nothing else may cut away.
            # The finger is `c` narrower than its notch at each end - the
            # play that lets the joint go together by hand. Taken off the
            # finger alone, so the notch and every check on it are exactly
            # as at zero clearance; `root` and `reach` stay full width.
            c = min(max(mat.finger_clearance, 0.0), 0.25 * (t1 - t0))
            finger = _rect(f_cut.a, fu, fv, t0 + c, t1 - c, -d, 0.0)
            root = _rect(f_cut.a, fu, fv, t0, t1, 0.0, d)
            notch = _rect(n_cut.a, nu, nv, t0, t1, 0.0, d)
            ftool = _rect(f_cut.a, fu, fv, t0 + c, t1 - c, -d, BITE)
            ntool = _rect(n_cut.a, nu, nv, t0, t1, -BITE, d)
            # Fingers go on AFTER the holes are cut, so one running back to
            # the vertex would fill its own hole in again: take it out.
            fh = [hole_disc(p, hole_r) for p in holes.get(f_isl, ())]
            if fh:
                gone = unary_union(fh)
                if ftool.intersects(gone):
                    finger = finger.difference(gone)
                    ftool = ftool.difference(gone)
            fnear, nnear = near[k % 2], near[1 - k % 2]
            last = nnear is not None and notch.intersects(nnear)
            if last:
                # Next to a hole the face narrows to its corner: the notch is
                # clipped to its own face rather than refused, so it tapers
                # into the hole. Half a notch, or no joint here.
                fp = raw[(n_isl, n_cut.face)]
                if free[1 - k % 2] is not None:
                    fp = fp.union(free[1 - k % 2])
                full = notch.area
                notch = notch.intersection(fp)
                ntool = ntool.intersection(fp.buffer(2 * BITE))
                if notch.area < 0.5 * full:
                    refuse("notch clipped away at the hole")
                    continue
            if fnear is not None and root.intersects(fnear):
                root = root.intersection(raw[(f_isl, f_cut.face)])
            cl = clip.get(k % 2)
            if cl is not None:
                full = finger.area
                finger = finger.intersection(cl)
                ftool = ftool.intersection(cl)
                # Clipped to a sliver it fills nothing: the notch opposite
                # would face an empty slot. Half the finger, or no joint here.
                if finger.area < 0.5 * full:
                    refuse("finger clipped away at the apex")
                    continue

            # The finger's board must really be there: in its own face, not a
            # relief slot, and not already cut away by a notch from another
            # edge of the same island. Tested a hair in from the edge: the
            # rectangle's outer side lies ON the body's boundary, and a strict
            # `contains` against the unbuffered body then fails on float noise
            # - it refused 240 of icosphere2's fingers before this.
            held = _rect(f_cut.a, fu, fv, t0, t1, 1e-6, d)
            if fnear is not None and held.intersects(fnear):
                full = held.area
                held = held.intersection(raw[(f_isl, f_cut.face)])
                if held.area < 0.5 * full:
                    refuse("finger has no board behind it")
                    continue
                held = held.difference(fnear)
            if not held.is_empty and not (
                    faces[(f_isl, f_cut.face)].contains(held)
                    and base[f_isl].contains(held)):
                refuse("finger has no board behind it")
                continue
            if any(root.intersection(q).area > 1e-9 for _c, q in notches[f_isl]):
                refuse("finger meets a notch")
                continue
            # Out in the flat sheet the finger must meet nothing: no other
            # board of its island - the far side of a V, near its apex - and
            # no other finger. Widened by `clear` along the edge too, and
            # stopped a hair short of the edge, which is its own board.
            reach = _rect(f_cut.a, fu, fv, t0 - clear, t1 + clear,
                          -d - clear, -1e-6)
            if cl is not None:
                # past the centreline is the other half's, and it is jointed
                reach = reach.intersection(cl)
            if base[f_isl].intersects(reach):
                refuse("finger meets other board")
                continue
            if any(reach.intersects(q) for q in stand[f_isl]):
                refuse("finger meets another finger")
                continue
            # The notch, and a wall round it, must be solid board of its own
            # face: not a neighbouring face, not a relief slot, not a notch
            # cut from another edge - and not another seam's finger.
            nprobe = _probe(notch, n_cut.a, nu, nv, 1.0, wall)
            if not last and not faces[(n_isl, n_cut.face)].contains(nprobe):
                refuse("notch leaves its face")
                continue
            if not last and not base[n_isl].contains(nprobe):
                refuse("notch meets a relief slot")
                continue
            # The whole round clearance here, not the probe: the probe is
            # clipped to this edge's own side, and another seam's notch round
            # a corner lies past that line - 2 mm away on an annulus once the
            # notches were half as deep and could fit there. Not this seam's
            # own other half, though: in a slit that is across the slit, with
            # no board between the two to tear.
            if any(c is not n_cut and c.seam != seam
                   and notch.distance(q) < wall - 1e-9
                   for c, q in notches[n_isl]):
                refuse("notch meets another notch")
                continue
            if any(c is not n_cut and notch.intersection(q).area > 1e-9
                   for c, q in kept[n_isl]):
                refuse("notch meets a finger")
                continue

            kept[f_isl].append((f_cut, root))
            stand[f_isl].append(finger)
            notches[n_isl].append((n_cut, notch))
            out[f_isl].append(Joint(finger, seam, f_isl, "finger", (t0, t1),
                                    f_cut.face, ftool))
            out[n_isl].append(Joint(notch, seam, n_isl, "notch", (t0, t1),
                                    n_cut.face, ntool))
            halves.add(k % 2)

        if len(halves) == 1:
            refuse("notches on one half only")
            for i, (nn, nk, ns, no) in mark.items():
                del notches[i][nn:], kept[i][nk:], stand[i][ns:], out[i][no:]
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
    """Add the fingers to a part and cut the notches out of it. A finger's
    tool reaches BITE back inside the edge, so it fuses with the board
    instead of meeting it on a line - the same trap as a notch, mirrored."""
    add = [j.tool if j.tool is not None else j.poly
           for j in joints if j.kind == "finger"]
    cut = [j.tool if j.tool is not None else j.poly
           for j in joints if j.kind == "notch"]
    if add:
        body = body.union(unary_union(add))
    if not cut:
        return shapely.set_precision(body, GRID) if add else body
    # Floating overlay, snapped once at the end. Snapping the body FIRST puts
    # GEOS in fixed-precision mode, and a body with its relief slots is tens
    # of thousands of vertices: 10.9 s against 2.8 s on an icosahedron net,
    # for a result that differs by 1e-9 mm2. BITE is what makes the floating
    # overlay safe - nothing here meets exactly on a line.
    return shapely.set_precision(body.difference(unary_union(cut)), GRID)


def fingered_seams(joints: dict[int, list[Joint]]) -> set[int]:
    return {j.seam for js in joints.values() for j in js}
