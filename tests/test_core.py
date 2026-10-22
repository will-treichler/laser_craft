"""Invariant tests. These check properties that must hold exactly, not
snapshots - if any of these fail the output is not cuttable.

Run:  python -m pytest tests -q      (or just: python tests/test_core.py)
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import trimesh
import pytest
import shapely
from shapely.geometry import (Polygon, MultiLineString, box,
                              Point, LineString)
from shapely.ops import unary_union
from shapely import set_precision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardboard_unfolder import Material, PolyMesh
from cardboard_unfolder import mesh as mesh_mod
from cardboard_unfolder.unfold import (unfold, isometry_error, overlapping_pairs,
                                       search, min_rect_size, min_area_rect, fits,
                                       split_to_fit, split_island, unfold_tree,
                                       join_islands, cut_arcs,
                                       reattach, repair, _side_areas,
                                       optimise, mend, _trial,
                                       place_across, place_branch,
                                       _face_area,
                                       mirrored_faces, _face_local)
from cardboard_unfolder.folds import (relief_width, perf_segments, plan_folds,
                                      apply_relief, offset_cut,
                                      fold_intervals, flute_warning,
                                      corner_relief)
from cardboard_unfolder.tabs import (place_tabs, seam_labels,
                                     part_labels, fold_labels,
                                     fit_height, island_labels,
                                     short_id, ACROSS_FRAC)
from cardboard_unfolder.layout import (build_parts, pack, narrow_necks,
                                       _mls, inscribed_width,
                                       Part, tighten, find_spot,
                                       find_pocket, free_region,
                                       seat_sheets)
from cardboard_unfolder import strokefont
from cardboard_unfolder.fingers import (place_fingers, segments, inward,
                                        corner_inset, MIN_FRAC)
from cardboard_unfolder.unfold import Island, Cut


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def meshes():
    out = []
    for name, tm in [
        ("box", trimesh.creation.box()),
        ("icosahedron", trimesh.creation.icosahedron()),
        ("cylinder", trimesh.creation.cylinder(radius=1, height=2, sections=16)),
        ("cone", trimesh.creation.cone(radius=1, height=2, sections=12)),
        ("icosphere1", trimesh.creation.icosphere(subdivisions=1)),
        ("capsule", trimesh.creation.capsule(count=[10, 10])),
        ("annulus", trimesh.creation.annulus(r_min=.5, r_max=1, height=1, sections=14)),
        # The one real model here. Every primitive above is convex or nearly
        # so, has no concave n-gons after the coplanar merge, and closes - the
        # helmet is an open shell of 137 triangles from a real costume, which
        # is the shape of input this exists for.
        ("helmet", trimesh.load(str(EXAMPLES / "spartan_helmet_reduced.stl"),
                                force="mesh", process=True)),
    ]:
        tm.apply_scale(200.0 / tm.extents.max())
        out.append((name, tm))
    return out


ALL = meshes()
IDS = [n for n, _ in ALL]


# ------------------------------------------------------------ mesh layer

@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_faces_are_planar(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    assert pm.planarity().max() < 1e-9, "merged faces must stay planar"


def test_nonplanar_faces_are_split_but_flat_ones_are_not():
    """A bent face has no rigid flat position, so it must be triangulated.

    Flat faces must survive intact: triangulating everything and letting the
    coplanar merge put it back together is what stops a fold line appearing
    down the diagonal of every panel.
    """
    # a unit square pulled out of plane at one corner
    v = np.array([[0., 0., 0.], [100., 0., 0.], [100., 100., 8.], [0., 100., 0.]])
    rings, origin = mesh_mod.split_nonplanar(v, [[0, 1, 2, 3]], tol=1e-3)
    assert len(rings) == 2, "a bent quad must be split"
    assert origin == [0, 0], "both halves still come from the caller's face 0"

    flat = np.array([[0., 0., 0.], [100., 0., 0.], [100., 100., 0.], [0., 100., 0.]])
    rings, origin = mesh_mod.split_nonplanar(flat, [[0, 1, 2, 3]], tol=1e-3)
    assert rings == [[0, 1, 2, 3]] and origin == [0], "a flat quad is left alone"


def test_split_faces_keep_pointing_at_their_original_face():
    """source_faces must survive the split, or the add-on writes UVs onto the
    wrong polygons - it indexes the caller's mesh with exactly these values."""
    v = np.array([[0., 0., 0.], [100., 0., 0.], [100., 100., 8.], [0., 100., 0.],
                  [200., 0., 0.], [200., 100., 0.]])
    pm = PolyMesh.from_rings(v, [[0, 1, 2, 3], [1, 4, 5, 2]], merge=True)
    seen = sorted({s for grp in pm.source_faces for s in grp})
    assert seen == [0, 1], f"source faces must index the input rings, got {seen}"
    assert all(0 <= s <= 1 for grp in pm.source_faces for s in grp)


def test_nonplanar_split_keeps_the_unfold_exact():
    """The whole point: after splitting, the unfold is isometric again."""
    tm = trimesh.creation.icosphere(subdivisions=1)
    tm.apply_scale(200.0 / tm.extents.max())
    v = tm.vertices.copy()
    quads = []                      # glue triangle pairs into deliberately bent quads
    for a, b in zip(tm.faces[0::2], tm.faces[1::2]):
        quads.append(list(dict.fromkeys(list(a) + list(b))))
    quads = [q for q in quads if len(q) == 4]
    pm = PolyMesh.from_rings(v, quads, merge=True)
    assert pm.planarity().max() < 1e-3, "nothing bent should survive the split"


def test_coplanar_merge_collapses_a_cube():
    pm = PolyMesh.from_trimesh(trimesh.creation.box())
    assert len(pm.faces) == 6, "12 triangles should merge to 6 quads"
    assert all(len(r) == 4 for r in pm.faces)


def test_cube_dual_angles_are_all_ninety():
    pm = PolyMesh.from_trimesh(trimesh.creation.box())
    G = pm.dual_graph()
    assert G.number_of_edges() == 12
    for *_, d in G.edges(data=True):
        assert abs(math.degrees(d["bend"]) - 90.0) < 1e-9


# --------------------------------------------------------------- unfolding

@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_unfold_is_exactly_isometric(name, tm):
    """The whole approach rests on this: a planar-faced mesh unfolds by rigid
    motion, so every edge length survives to floating point."""
    pm = PolyMesh.from_trimesh(tm)
    islands, rep = unfold(pm, trials=120)
    assert rep["isometry_error"] < 1e-12, rep


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_no_face_is_placed_mirrored(name, tm):
    """The invariant isometry_error cannot see.

    Placing a face on the wrong side of its seam reflects it. A reflection
    preserves every edge length, so the isometry check measures perfect while
    the part is the mirror image of the real face and will not fold up.
    """
    pm = PolyMesh.from_trimesh(tm)
    T, pos, bad, _ = search(pm, trials=120)
    assert mirrored_faces(pm, pos) == []


def _concave_pair():
    """Two faces whose child is concave enough to fool a centroid test.

    The child's seam runs x 0->10. Its body is at +y, but it has a leg
    reaching to y = -6 out at x 10.5..13 - past the end of the seam, so the leg
    misses the parent entirely and the correct placement has no overlap at all.
    The sum of the child's signed offsets is still negative, which is exactly
    what a sign-of-the-sum side test gets wrong.
    """
    c, s = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
    parent = [(0, 0, 0), (0, -3, 0), (10, -3, 0), (10, 0, 0)]
    child2d = [(0, 0), (10, 0), (10.5, -6), (13, -6), (13, 2), (0, 2)]
    verts = list(parent)
    rings = [[0, 1, 2, 3]]
    ring = []
    for u, v in child2d:
        pt = (u, v * c, v * s)
        for i, w in enumerate(verts):
            if np.allclose(w, pt, atol=1e-12):
                ring.append(i)
                break
        else:
            verts.append(pt)
            ring.append(len(verts) - 1)
    rings.append(ring)
    # merge off: the two faces are not coplanar, but nothing here should be
    # rewritten under the test's feet.
    return PolyMesh.from_rings(np.array(verts, float), rings, merge=False)


def test_a_concave_child_is_not_reflected_onto_its_parent():
    pm = _concave_pair()
    G = pm.dual_graph()
    assert G.number_of_edges() == 1, "the fixture must share exactly one edge"
    ev = list(G.edges(data=True))[0][2]["verts"]

    # The fixture is only worth having if it actually trips the old rule.
    _t, h, _L = _face_local(pm, 1, ev[0], ev[1])
    assert h.sum() < 0 < np.abs(h).max(),         "fixture no longer fools a sign-of-the-sum side test"

    pos = unfold_tree(pm, G, 0)
    assert mirrored_faces(pm, pos) == [], "the child was reflected"

    a = Polygon(pos[0]).buffer(0)
    b = Polygon(pos[1]).buffer(0)
    a, b = set_precision(a, 1e-9), set_precision(b, 1e-9)
    assert a.intersection(b).area < 1e-9, "the child was folded back over it"
    assert isometry_error(pm, pos) < 1e-12


def test_a_locally_flipped_face_still_places_correctly():
    """The side is read from the two windings against each other, not from the
    mesh being globally consistent - so one reversed ring costs nothing."""
    pm = _concave_pair()
    flipped = PolyMesh.from_rings(pm.vertices,
                                  [pm.faces[0], list(reversed(pm.faces[1]))],
                                  merge=False)
    G = flipped.dual_graph()
    pos = unfold_tree(flipped, G, 0)
    a = set_precision(Polygon(pos[0]).buffer(0), 1e-9)
    b = set_precision(Polygon(pos[1]).buffer(0), 1e-9)
    assert a.intersection(b).area < 1e-9
    assert isometry_error(flipped, pos) < 1e-12


def _blank(cut, sheet):
    """A Part carrying nothing but an outline, for packing tests."""
    empty = MultiLineString([])
    return Part(island_id=id(cut) % 1000, cut=cut, perf_m=empty, perf_v=empty,
                engrave=empty, sheet=sheet)


def _sheet_with_a_pocket(mat):
    """A sheet walled in except for one opening, plus a stray that fits it.

    The opening is reachable only from inside, so its bottom-left corner is
    not on any placed part's skyline - which is exactly the case the corner
    candidates cannot offer and find_pocket exists for.
    """
    W, H = mat.usable_bed
    m, gap = mat.sheet_margin, mat.part_gap
    stray_w, stray_h = 40.0, 25.0
    pocket = box(m + 100.0, m + 100.0,
                 m + 100.0 + stray_w + 2 * gap + 1.0,
                 m + 100.0 + stray_h + 2 * gap + 1.0)
    # one big part with the pocket punched out of it
    slab = box(m, m, m + W, m + H).difference(pocket)
    stray = box(0.0, 0.0, stray_w, stray_h)
    return [_blank(slab, 0), _blank(stray, 1)]


def test_tighten_slots_a_stray_into_a_pocket_the_skyline_hides():
    mat = Material.from_flute("B")
    mat.bed = (300.0, 300.0)
    W, H = mat.usable_bed
    m, gap = mat.sheet_margin, mat.part_gap
    parts = _sheet_with_a_pocket(mat)
    stray = parts[1]

    # The cheap search must genuinely fail here, or the test proves nothing.
    assert find_spot(stray, [parts[0].cut], [parts[0].bounds],
                     W, H, gap, m) is None,         "corner candidates now find it; this fixture no longer tests find_pocket"
    assert find_pocket(stray, free_region([parts[0].cut], W, H, gap, m),
                       tol=gap / 4.0) is not None

    assert tighten(parts, 2, mat) == 1
    assert {p.sheet for p in parts} == {0}
    assert parts[0].cut.intersection(stray.cut).area < 1e-9
    assert stray.cut.distance(parts[0].cut) >= gap - 1e-6


def test_tighten_seats_before_it_measures():
    """Regression: it used to trust the caller to have done that.

    The free region is measured over [m, m+W]; parts handed in unseated live
    in a different frame, so the stray landed correctly in the box and the
    seating that followed dragged it outside the bed. Reported by the add-on's
    nester as "part 31 ended up outside sheet 0".
    """
    mat = Material.from_flute("B")
    mat.bed = (300.0, 300.0)
    m = mat.sheet_margin
    parts = _sheet_with_a_pocket(mat)
    for p in parts:                              # shove everything off-margin
        p.translate(-m + 3.0, -m - 5.0)

    tighten(parts, 2, mat)
    for p in parts:
        x0, y0, x1, y1 = p.bounds
        assert x0 >= m - 1e-6 and y0 >= m - 1e-6
        assert x1 <= mat.bed[0] - m + 1e-6 and y1 <= mat.bed[1] - m + 1e-6


def test_tighten_never_moves_a_part_it_cannot_place():
    """No room means no change - and certainly no overlap."""
    mat = Material.from_flute("B")
    mat.bed = (300.0, 300.0)
    W, H = mat.usable_bed
    m = mat.sheet_margin
    full = box(m, m, m + W, m + H)
    parts = [_blank(full, 0), _blank(box(m, m, m + W, m + H), 1)]
    before = [p.cut.bounds for p in parts]
    assert tighten(parts, 2, mat) == 2
    assert [p.sheet for p in parts] == [0, 1]
    assert [p.cut.bounds for p in parts] == before


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_packing_leaves_no_overlap_and_nothing_off_the_sheet(name, tm):
    """The property tighten must not break: it moves parts between finished
    sheets, so it is the one step that could drop a part onto another."""
    mat = Material.from_flute("B")
    mat.bed = (300.0, 300.0)
    m = mat.sheet_margin
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    parts = build_parts(islands, mat)
    pack(parts, mat)

    by = {}
    for p in parts:
        if p.sheet >= 0:
            by.setdefault(p.sheet, []).append(p)
    assert sorted(by) == list(range(len(by))), "sheet numbers have a hole"
    for group in by.values():
        geos = [set_precision(p.cut.buffer(0), 1e-9) for p in group]
        for i in range(len(geos)):
            for j in range(i + 1, len(geos)):
                assert geos[i].intersection(geos[j]).area < 1e-6
        for p in group:
            x0, y0, x1, y1 = p.bounds
            assert x0 >= m - 1e-6 and y0 >= m - 1e-6
            assert x1 <= mat.bed[0] - m + 1e-6
            assert y1 <= mat.bed[1] - m + 1e-6


import string as _string


def test_a_closed_glyph_survives_to_the_engrave_layer():
    """Regression: every 0, O, D and Q was silently deleted.

    _mls dropped any stroke whose first point equalled its last, meaning to
    reject zero-length segments - but that is the definition of a CLOSED
    stroke, which is what those glyphs are. A part stamped PIKA-0 engraved as
    "PIKA-", and seam 10 engraved as "1". Not a missing annotation but a
    wrong one, and nothing downstream can tell.
    """
    for ch in _string.digits + _string.ascii_uppercase + "-":
        strokes = strokefont.text_strokes(ch, 5.0)
        kept = list(_mls(strokes).geoms)
        assert len(kept) == len(strokes), f"{ch!r} lost a stroke"

    closed = [c for c in _string.digits + _string.ascii_uppercase
              if any(np.allclose(s[0], s[-1])
                     for s in strokefont.text_strokes(c, 5.0))]
    assert closed, "the font has no closed glyph left to protect"

    # ...and a stroke that really does go nowhere is still dropped
    assert len(list(_mls([np.array([[1.0, 1.0], [1.0, 1.0]])]).geoms)) == 0
    assert len(list(_mls([np.array([[1.0, 1.0]])]).geoms)) == 0


def test_every_label_stroke_reaches_the_part():
    """The count is the check: labels are built, then filtered on the way in,
    and a filter that eats glyphs is invisible in any area or bounds test."""
    mat = Material.from_flute("B")
    mat.model_id = "D0Q"                        # three closed glyphs
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(240.0)
    islands, _ = unfold(pm, trials=40)
    parts = build_parts(islands, mat)
    by_id = {p.island_id: p for p in parts}
    for isl in islands:
        want = sum(len(l.strokes) for l in _all_labels([isl], mat))
        got = len(list(getattr(by_id[isl.id].engrave, "geoms",
                               [by_id[isl.id].engrave])))
        if by_id[isl.id].engrave.is_empty:
            got = 0
        assert got == want, f"island {isl.id}: {want} strokes built, {got} engraved"


def _all_labels(islands, mat):
    sl, pl = seam_labels(islands, mat), part_labels(islands, mat)
    out = []
    for isl in islands:
        out += sl[isl.id] + pl[isl.id]
        out += fold_labels(isl, plan_folds(isl, mat), mat)
    return out


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_every_label_is_between_the_floor_and_the_cap(name, tm):
    """The whole point of the height rule: never bigger than asked for, never
    smaller than legible, and left off rather than shrunk past the floor."""
    mat = Material.from_flute("B")
    mat.model_id = "TEST"
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    labels = _all_labels(islands, mat)
    assert labels, "a net this size should carry some annotation"
    for lab in labels:
        assert lab.height <= mat.label_height + 1e-9, lab.text
        assert lab.height >= mat.label_min_height - 1e-9, lab.text


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_no_engraved_stroke_is_bigger_than_the_text_it_claims(name, tm):
    """Measure the geometry, not the height field it was built from.

    Rotation makes a bounding box useless here - the y extent of a rotated
    glyph is its box at that angle and reads up to 2.5x the real height, which
    is how a 4 mm cap once measured as 9.94 mm. Diameter is what a ruler
    measures and no rigid motion changes it: no single stroke of text `h` tall
    can exceed the cell diagonal.
    """
    mat = Material.from_flute("B")
    mat.model_id = "ID"
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    cap = math.hypot(strokefont.CELL_W / strokefont.CELL_H * mat.label_height,
                     mat.label_height)
    worst = 0.0
    for lab in _all_labels(islands, mat):
        for st in lab.strokes:
            a = np.asarray(st, dtype=float)
            worst = max(worst, float(
                np.linalg.norm(a[:, None] - a[None, :], axis=-1).max()))
    assert worst <= cap + 1e-9, (
        f"a stroke spans {worst:.2f} mm; {mat.label_height} mm text cannot "
        f"exceed {cap:.2f}")
    assert worst > mat.label_min_height, "nothing was engraved at all"


def test_fit_height_divides_rather_than_searches():
    mat = Material.from_flute("B")
    mat.label_height, mat.label_min_height = 4.0, 2.0
    # roomy: capped at the requested height, not grown to fill
    assert fit_height("7", 500.0, mat) == pytest.approx(4.0)
    # tight enough that the cap does not bind: the height whose width exactly
    # fills the space
    h = fit_height("12", 4.0, mat)
    assert h is not None and mat.label_min_height < h < mat.label_height
    assert strokefont.text_width("12", h) == pytest.approx(4.0, rel=1e-9)
    # too tight for the floor: omitted, not shrunk
    assert fit_height("12", 1.0, mat) is None
    assert fit_height("", 100.0, mat) is None


def test_a_label_never_leaves_its_island():
    """An engraved line outside the part is cut off the sheet and lost, and a
    seam number in a neighbour's territory is worse than none."""
    mat = Material.from_flute("B")
    mat.model_id = "X"
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(220.0)
    islands, _ = unfold(pm, trials=40)
    sl = seam_labels(islands, mat)
    pl = part_labels(islands, mat)
    for isl in islands:
        o = isl.outline().buffer(1e-6)
        labs = sl[isl.id] + pl[isl.id] + fold_labels(isl, plan_folds(isl, mat),
                                                     mat)
        assert labs, f"island {isl.id} carries no annotation at all"
        for lab in labs:
            for st in lab.strokes:
                assert o.contains(LineString(st) if len(st) > 1
                                  else Point(st[0])), \
                    f"{lab.kind} label {lab.text!r} escaped island {isl.id}"


def test_every_fold_is_marked_mountain_or_valley():
    """Which way a crease bends is not recoverable from a flat sheet."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(200.0)
    islands, _ = unfold(pm, trials=40)
    for isl in islands:
        plans = plan_folds(isl, mat)
        labs = fold_labels(isl, plans, mat)
        assert len(labs) == len(plans), "a box's folds all have room"
        assert {l.text for l in labs} <= {"M", "V"}
        for lab, plan in zip(labs, plans):
            assert lab.text == ("M" if plan.is_mountain else "V")
            assert lab.kind == "fold"

    mat.label_folds = False
    assert fold_labels(islands[0], plan_folds(islands[0], mat), mat) == []


def test_the_model_id_is_stamped_on_every_island_that_can_hold_it():
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(200.0)
    islands, _ = unfold(pm, trials=40)

    plain = part_labels(islands, mat)
    assert all(l.text.startswith("P") for v in plain.values() for l in v)

    mat.model_id = "ARM"
    named = part_labels(islands, mat)
    for isl in islands:
        assert named[isl.id], "a 200 mm box panel has room for an id"
        assert named[isl.id][0].text == f"ARM-{isl.id}"
        assert named[isl.id][0].kind == "part"

    # whitespace is not an id
    mat.model_id = "   "
    assert part_labels(islands, mat)[islands[0].id][0].text.startswith("P")


def test_a_label_that_cannot_fit_is_omitted_not_shrunk():
    """The floor is a floor: below it the label is dropped, never shrunk."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(60.0)
    islands, _ = unfold(pm, trials=40)

    # Derive the premise rather than guess it. A first attempt assumed 60
    # characters could not fit a 60 mm box - but the net of a box is a strip
    # of four panels, 240 mm long, and it fitted easily. An id whose text at
    # the floor is longer than any island's bounding-box DIAGONAL cannot fit
    # at any angle, whatever the net turned out to look like.
    diag = 0.0
    for isl in islands:
        x0, y0, x1, y1 = isl.outline().bounds
        diag = max(diag, math.hypot(x1 - x0, y1 - y0))
    per_char = (strokefont.text_width("AA", mat.label_min_height)
                - strokefont.text_width("A", mat.label_min_height))
    mat.model_id = "A" * (int(diag / per_char) + 5)
    assert strokefont.text_width(mat.model_id, mat.label_min_height) > diag

    assert all(v == [] for v in part_labels(islands, mat).values())


def test_seam_numbers_match_across_the_pair():
    """The two halves of a seam must carry the same number or the net cannot
    be assembled - that is the entire job of this annotation."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(260.0)
    islands, _ = unfold(pm, trials=40)
    labelled = {}
    for isl in islands:
        for lab in seam_labels(islands, mat)[isl.id]:
            labelled.setdefault(lab.text, 0)
            labelled[lab.text] += 1
    assert labelled, "some seams should be labelled"
    # every seam id that got labelled at all should appear on both halves
    both = sum(1 for n in labelled.values() if n == 2)
    assert both >= len(labelled) * 0.8, \
        f"only {both} of {len(labelled)} seam numbers appear on both halves"


def _labelled(islands, mat):
    """Every island's labels, built the way build_parts builds them."""
    out = {}
    for isl in islands:
        plans = plan_folds(isl, mat)
        body = apply_relief(isl.outline(), plans)
        out[isl.id] = (island_labels(isl, plans, mat, body), body)
    return out


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_nets_cut_set_replays_it_exactly_at_any_seed(name, tm):
    """The invariant the whole optimiser rests on.

    Every dual arc is either a fold or a cut, so the cut set is a COMPLETE
    description of a net: replay it as forced_cuts and apply_constraints
    deletes exactly those arcs, leaving a graph that is already a forest. The
    spanning-tree step then has nothing to choose and the seed cannot matter.

    Without this, "the optimiser writes constraints you can then hand-edit" is
    a promise the code does not keep - re-running Unfold would quietly give a
    different net and every hand edit would be against a layout that no longer
    exists.
    """
    pm = PolyMesh.from_trimesh(tm)
    islands, rep = unfold(pm, trials=40, seed=0)
    cuts = set(rep["cut_arcs"])
    assert cuts, "a closed mesh always needs cuts"

    want_members = sorted(tuple(sorted(i.faces)) for i in islands)
    want_pos = {f: r for i in islands for f, r in i.faces.items()}

    for seed in (0, 5, 91):
        again, _ = unfold(pm, trials=40, seed=seed, forced_cuts=cuts)
        assert sorted(tuple(sorted(i.faces)) for i in again) == want_members
        got = {f: r for i in again for f, r in i.faces.items()}
        worst = max(float(np.abs(got[f] - want_pos[f]).max()) for f in want_pos)
        assert worst == 0.0, f"seed {seed} moved the net by {worst:.2e} mm"


def test_a_face_cut_off_on_every_side_is_still_an_island():
    """Regression: it used to vanish, with a KeyError deep in build_islands.

    weighted_dual built its graph from EDGES, so a face whose every dual arc is
    forced cut had no node at all - it fell out of the spanning tree, out of
    _forest_positions, and finally out of build_islands. Invisible until a
    constraint set is complete enough to isolate a face, which is exactly what
    replaying an optimiser's cut set does.
    """
    pm = PolyMesh.from_trimesh(trimesh.creation.icosahedron())
    G = pm.dual_graph()
    victim = 0
    ring = [tuple(sorted(G[victim][n]["verts"])) for n in G.neighbors(victim)]
    assert len(ring) == 3, "a triangle has three neighbours to cut away"

    islands, rep = unfold(pm, trials=20, forced_cuts=set(ring))
    alone = [i for i in islands if set(i.faces) == {victim}]
    assert len(alone) == 1, "the isolated face is its own island"
    assert sum(len(i.faces) for i in islands) == len(pm.faces), \
        "and no face went missing"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_the_equal_area_circle_is_an_upper_bound_on_what_fits(name, tm):
    """The cheap tier's whole licence to exist.

    `area_width` judges a splinter from area alone, which is only safe because
    no shape of area A holds a circle wider than the one of area A - so it can
    miss a splinter but never invent one. Inside a search that means it is
    usable as a filter and useless as a verdict, and if the inequality ever
    went the other way the optimiser would start cutting away islands that are
    perfectly holdable.
    """
    from cardboard_unfolder.cost import area_width, island_area
    from cardboard_unfolder.layout import inscribed_width
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=30)
    for isl in islands:
        bound = area_width(island_area(isl))
        real = inscribed_width(isl.outline(), tol=0.01)
        assert real <= bound + 1e-6, (
            f"island {isl.id} holds {real:.3f} mm but the area bound says "
            f"{bound:.3f} mm")


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_reattach_never_makes_a_net_worse(name, tm):
    """Every weld it keeps is strictly an improvement, so this is a ratchet.

    Greedy cover cuts an arc because two faces overlapped through it at that
    moment, and later cuts routinely resolve the same conflict from the other
    end - nothing ever went back to check. The guarantee that makes it safe to
    run unconditionally is that a weld is only kept when the merged island does
    not overlap itself, so it can only remove islands and seams.
    """
    pm = PolyMesh.from_trimesh(tm)
    G = pm.dual_graph()
    forest, pos, bad, _ = search(pm, G, trials=30, seed=0)
    if bad:
        forest, pos, bad, _ = repair(pm, forest, bad)
    before = nx.number_connected_components(forest)
    seam_before = sum(d.get("length", 0.0) for *_e, d in G.edges(data=True)
                      if not forest.has_edge(*_e[:2]))

    out, pos2, welds = reattach(pm, G, forest, pos)
    after = nx.number_connected_components(out)
    assert after == before - welds, \
        f"{welds} weld(s) should remove exactly that many islands"
    assert after <= before
    assert not overlapping_pairs(out, pos2), \
        "reattach kept a weld that overlaps"
    assert isometry_error(pm, pos2) < 1e-6
    assert not mirrored_faces(pm, pos2)
    seam_after = sum(d.get("length", 0.0) for *_e, d in G.edges(data=True)
                     if not out.has_edge(*_e[:2]))
    assert seam_after <= seam_before + 1e-9, "welding cannot lengthen the seam"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_placing_one_branch_agrees_with_placing_the_whole_tree(name, tm):
    """`place_branch` is the reason a weld costs five placements, not 255.

    It has to put the branch exactly where a full `unfold_tree` rooted on the
    other side would have - not merely somewhere valid. If the two ever
    disagree, welding would nudge the net by a hair and every net would depend
    on which code path happened to place it.
    """
    pm = PolyMesh.from_trimesh(tm)
    G = pm.dual_graph()
    forest, pos, _bad, _ = search(pm, G, trials=20, seed=0)
    comp = max(nx.connected_components(forest), key=len)
    sub = forest.subgraph(comp)
    if sub.number_of_edges() == 0:
        pytest.skip("needs a component with an arc to place across")
    root = min(comp)
    whole = unfold_tree(pm, sub, root)

    # cut one arc, then put that branch back with place_branch alone
    for parent, child in nx.bfs_edges(sub, root):
        cut = sub.copy()
        cut.remove_edge(parent, child)
        branch = nx.node_connected_component(cut, child)
        got = place_branch(pm, sub, whole, parent, child, branch)
        worst = max(float(np.abs(got[f] - whole[f]).max()) for f in branch)
        assert worst == 0.0, (
            f"branch at {parent}->{child} moved by {worst:.2e} mm")
        break


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_repairs_fast_path_never_ends_on_an_overlap(name, tm):
    """Filtering the overlap set by component is optimistic and known to be.

    Cutting a tree arc moves nothing relative to its own half, so the set can
    only shrink - but `_forest_positions` re-roots every component afterwards
    and GEOS then decides a borderline touch the other way. Measured on
    pikachu: 8 of 124 cuts differed by up to 6 pairs. So repair may only LEAVE
    the loop on a real measurement, and this is that guarantee.
    """
    pm = PolyMesh.from_trimesh(tm)
    G = pm.dual_graph()
    forest, pos, bad, _ = search(pm, G, trials=20, seed=0)
    forest, pos, bad, cuts = repair(pm, forest, bad)
    assert bad == overlapping_pairs(forest, pos), \
        "repair returned an overlap set it had not actually measured"
    assert not bad, f"repair stopped with {len(bad)} overlap(s) outstanding"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_the_optimiser_answers_with_a_net_that_could_be_cut(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    G = pm.dual_graph()
    mat = Material()
    out = optimise(pm, G, mat, trials=8, seed=0, budget=3, finalists=2)
    assert out is not None
    forest, pos, bad, stats = out
    assert set(forest) == set(G), "every face must survive the search"
    assert not overlapping_pairs(forest, pos)
    assert isometry_error(pm, pos) < 1e-6
    assert not mirrored_faces(pm, pos)
    assert stats["mended"] <= 3, "the mending budget is a budget"
    assert stats["candidates"] >= 1


def test_the_board_slider_at_zero_is_todays_optimiser():
    """The user's default is large islands - today's behaviour. So the slider
    at 0 must be exactly the old weights and try no smaller split."""
    from cardboard_unfolder.cost import Weights, board_weights
    from cardboard_unfolder.unfold import split_fractions
    assert board_weights(0.0) == Weights()
    assert split_fractions(0.0) == [1.0]


def test_the_board_slider_adds_smaller_pieces_and_weighs_board_more():
    from cardboard_unfolder.cost import Weights, board_weights
    from cardboard_unfolder.unfold import split_fractions
    prev_f, prev_w = [1.0], Weights()
    for b in np.linspace(0.0, 1.0, 11):
        f, w = split_fractions(b), board_weights(b)
        assert f[0] == 1.0 and set(prev_f) <= set(f), b
        assert w.sheet >= prev_w.sheet and w.island <= prev_w.island, b
        assert w.splinter == Weights().splinter, "buildability is not traded"
        assert w.oversize == Weights().oversize
        prev_f, prev_w = f, w
    assert split_fractions(1.0) == [1.0, 0.85, 0.7, 0.5]


def test_saving_board_still_answers_with_a_net_that_could_be_cut():
    """At the far end of the slider the optimiser also splits finalists to
    smaller limits. Whatever it picks must still be a real net - every face,
    no overlap, exact - and fit the smaller limit it was split to."""
    from cardboard_unfolder.unfold import min_rect_size
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(250)
    G = pm.dual_graph()
    mat = Material()
    pad = 2 * mat.tab_depth + 2 * mat.kerf
    limit = (mat.usable_bed[0] - pad, mat.usable_bed[1] - pad)
    forest, pos, bad, st = optimise(pm, G, mat, trials=6, seed=0,
                                    max_size=limit, budget=3, finalists=2,
                                    board_bias=1.0)
    assert set(forest) == set(G)
    assert not overlapping_pairs(forest, pos)
    assert isometry_error(pm, pos) < 1e-6
    assert not mirrored_faces(pm, pos)
    f = st["split_frac"]
    assert f in (1.0, 0.85, 0.7, 0.5)
    for comp in nx.connected_components(forest):
        w, h = min_rect_size(np.vstack([pos[x] for x in comp]))
        lw, lh = limit[0] * f, limit[1] * f
        assert (w <= lw + 1e-6 and h <= lh + 1e-6) or                (h <= lw + 1e-6 and w <= lh + 1e-6), (w, h, f)


def test_the_optimiser_can_see_sheets():
    """The term it spent two failed estimates trying to avoid paying for.

    Area cannot rank candidates (it is conserved) and bounding boxes rank them
    backwards, so the count comes from actually packing each finalist. That
    costs real time, so the switch has to exist - and has to be the thing that
    decides whether the term is there at all, rather than the term quietly
    going missing.
    """
    from cardboard_unfolder.cost import Weights
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    G = pm.dual_graph()
    mat = Material()
    limit = mat.usable_bed

    on = optimise(pm, G, mat, trials=4, seed=0, max_size=limit, budget=4,
                  finalists=2, count_sheets=True)
    off = optimise(pm, G, mat, trials=4, seed=0, max_size=limit, budget=4,
                   finalists=2, count_sheets=False)
    assert on[3]["sheets"] is not None and on[3]["sheets"] >= 1
    assert off[3]["sheets"] is None
    assert "sheet" in on[3]["terms"], "the term is missing when it was asked for"
    assert "sheet" not in off[3]["terms"], "the term is there when it was not"
    # ...and it is worth what the weights say it is worth
    w = Weights()
    assert on[3]["terms"]["sheet"] == pytest.approx(
        w.sheet * on[3]["sheets"], rel=1e-9)


def test_the_optimiser_reports_every_trial_even_past_its_budget():
    """Regression: the progress bar froze once the mending budget was spent.

    A trial that needs mending after the budget has gone is skipped - and the
    skip used to `continue` straight past the progress call. So the readout
    stopped moving for the rest of the search, which on a 60-trial run with a
    budget of 10 is most of it, and ESC went deaf too, because stopping is
    read through the same call. Every trial must report, used or not.
    """
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=2))
    G = pm.dual_graph()
    mat = Material()
    seen = []

    def watch(done, total, note):
        seen.append((done, total, note))
        return True

    trials = 12
    optimise(pm, G, mat, trials=trials, seed=0, budget=1, finalists=1,
             progress=watch)
    searching = [d for d, t, n in seen if n.startswith("searching")]
    assert searching == list(range(1, trials + 1)), (
        f"trials that were skipped did not report: saw {searching}")


def test_the_optimiser_stops_when_progress_says_to():
    """A minute of silence is indistinguishable from a hang, so the core
    reports - and a caller that reports can also refuse to continue."""
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=2))
    G = pm.dual_graph()
    mat = Material()
    seen = []

    def stop_after_two(done, total, note):
        seen.append((done, total, note))
        return done < 2

    out = optimise(pm, G, mat, trials=30, seed=0, budget=30, finalists=1,
                   progress=stop_after_two)
    assert out is not None, "cancelling must still answer with the best so far"
    forest, pos, bad, stats = out
    assert stats["candidates"] == 2, \
        f"it kept going after being told to stop ({stats['candidates']})"
    assert not overlapping_pairs(forest, pos)
    assert seen and seen[0][1] == 30


def test_the_optimiser_never_loses_to_the_first_candidate():
    """It ranks, so the answer can tie the first candidate but not lose to it.

    Worth asserting because the ranking key (quick cost on the mended net) is
    NOT the key the answer is chosen by (full cost after bed splitting), and a
    two-key scheme can quietly pick the worse of two things it has both scored.
    """
    from cardboard_unfolder.cost import score_islands
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=2))
    G = pm.dual_graph()
    mat = Material()
    limit = mat.usable_bed

    one = optimise(pm, G, mat, trials=1, seed=0, max_size=limit, budget=1,
                   finalists=1)
    many = optimise(pm, G, mat, trials=6, seed=0, max_size=limit, budget=6,
                    finalists=3)
    assert many[3]["cost"] <= one[3]["cost"] + 1e-9, (
        f"scoring 6 candidates ({many[3]['cost']:.2f}) did worse than taking "
        f"the first ({one[3]['cost']:.2f})")


def test_reattach_has_nothing_left_to_do_the_second_time():
    """It runs to a fixed point, so a caller cannot improve on it by looping.

    If a second pass found more welds, the first one stopped early and the net
    handed to bed splitting is not the one the cost function was promised.
    """
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=2))
    G = pm.dual_graph()
    forest, pos, bad, _ = search(pm, G, trials=20, seed=1)
    if bad:
        forest, pos, bad, _ = repair(pm, forest, bad)
    once, pos1, _ = reattach(pm, G, forest, pos)
    twice, _, again = reattach(pm, G, once, pos1)
    assert again == 0, f"a second pass found {again} more weld(s)"
    assert set(map(frozenset, twice.edges)) == set(map(frozenset, once.edges))


def test_side_areas_accounts_for_every_face_on_both_sides_of_every_arc():
    """The number the cost-aware repair prices a cut with.

    Removing an arc of a tree yields exactly two pieces, so the two side areas
    must add up to the component - if they do not, the repair is pricing a cut
    that does not exist, and it would do it silently.
    """
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    G = pm.dual_graph()
    forest, pos, _bad, _ = search(pm, G, trials=20, seed=0)
    face_area = {f: _face_area(pm, f) for f in forest}
    sides = _side_areas(forest, face_area)
    assert len(sides) == forest.number_of_edges()
    for comp in nx.connected_components(forest):
        total = sum(face_area[f] for f in comp)
        for a, b in forest.subgraph(comp).edges:
            lo, hi = sides[(min(a, b), max(a, b))]
            assert abs(lo + hi - total) < 1e-9
            assert lo > 0 and hi > 0, "a tree arc always has faces on both sides"


def test_the_bed_splits_are_named_and_are_part_of_the_cut_set():
    """The optimiser has to weigh a bed split differently from a repair cut.

    One lands on a real crease that already needs a glue flap; the other is the
    solver working around an overlap. Counting them together says nothing about
    which lever to pull, so split_to_fit reports the arcs it cut and unfold
    passes them on - and they must be genuine members of the net's cut set, or
    they name something that is not there.
    """
    pm = PolyMesh.from_trimesh(trimesh.creation.cylinder(radius=40, height=180,
                                                         sections=24))
    islands, rep = unfold(pm, trials=20, max_size=(60.0, 60.0))
    bed = rep["bed_cut_arcs"]
    assert bed, "a 40 mm cylinder cannot fit a 60 mm bed in one piece"
    assert set(bed) <= set(rep["cut_arcs"]),         "a bed split that is not in the cut set names an edge that is folded"
    assert len(bed) == len(set(bed)), "no arc is cut twice"


def test_the_cut_set_is_the_complement_of_the_folds():
    pm = PolyMesh.from_trimesh(trimesh.creation.cylinder(radius=1, height=2,
                                                         sections=16))
    islands, rep = unfold(pm, trials=40)
    G = pm.dual_graph()
    folds = sum(len(i.folds) for i in islands)
    assert len(rep["cut_arcs"]) + folds == G.number_of_edges(), (
        f"{len(rep['cut_arcs'])} cuts + {folds} folds != "
        f"{G.number_of_edges()} arcs")
    # ...and the fold count is what a spanning forest must have
    assert folds == len(pm.faces) - len(islands)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_split_then_join_puts_the_island_back_exactly(name, tm):
    """The invariant that makes Weld a local edit rather than a re-solve.

    join_islands is the inverse of split_island, so doing both has to leave the
    net byte-for-byte where it started - otherwise welding a seam you just cut
    would nudge the whole island and every hand edit would drift.
    """
    pm = PolyMesh.from_trimesh(tm)
    G = pm.dual_graph()
    islands, _ = unfold(pm, trials=40)
    isl = max(islands, key=lambda i: len(i.faces))
    if not isl.folds:
        pytest.skip("single-face island has no fold to cut")

    fo = isl.folds[len(isl.folds) // 2]
    parts = split_island(isl, fo.f0, fo.f1, 9999, 4242)
    assert parts is not None
    a, b = parts
    assert len(a.faces) + len(b.faces) == len(isl.faces)

    j = join_islands(a, b, pm, G, fo.f0, fo.f1)
    assert j is not None, "the two halves of a fresh cut must weld back"
    assert set(j.faces) == set(isl.faces)
    assert len(j.folds) == len(isl.folds)
    assert len(j.cuts) == len(isl.cuts)
    worst = max(float(np.abs(j.faces[k] - isl.faces[k]).max()) for k in isl.faces)
    assert worst < 1e-9, f"welding moved the island by {worst:.2e} mm"


def test_a_weld_is_a_rigid_motion_not_a_re_solve():
    """Every edge length in the welded island survives, and the half that did
    not move did not move at all."""
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(200.0)
    G = pm.dual_graph()
    islands, _ = unfold(pm, trials=40)
    isl = max(islands, key=lambda i: len(i.faces))
    fo = isl.folds[3]
    a, b = split_island(isl, fo.f0, fo.f1, 9999, 4242)
    before_a = {k: v.copy() for k, v in a.faces.items()}

    j = join_islands(a, b, pm, G, fo.f0, fo.f1)
    assert j is not None
    for k, ring in before_a.items():
        assert np.allclose(j.faces[k], ring, atol=1e-12), \
            "the island being welded ONTO must not move"
    assert isometry_error(pm, j.faces) < 1e-12


def test_join_refuses_what_it_cannot_weld():
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    G = pm.dual_graph()
    islands, _ = unfold(pm, trials=40)
    isl = max(islands, key=lambda i: len(i.faces))
    # A split whose kept side has a face that does NOT touch the other side.
    # Cutting at folds[0] can leave a five-face stub where every face happens
    # to border f1, and then there is no stranger left to test with.
    found = None
    for fo in isl.folds:
        a, b = split_island(isl, fo.f0, fo.f1, 9999, 4242)
        strangers = [f for f in a.faces if f != fo.f0 and not G.has_edge(f, fo.f1)]
        if strangers:
            found = (fo, a, b, strangers[0])
            break
    assert found, "no fixture on this mesh"
    fo, a, b, stranger = found

    assert join_islands(a, a, pm, G, fo.f0, fo.f1) is None, "an island with itself"
    assert join_islands(a, b, pm, G, fo.f1, fo.f0) is None, "faces the wrong way round"
    assert join_islands(a, b, pm, G, stranger, fo.f1) is None,         "faces that do not share an edge in 3D"


def test_the_coplanar_merge_does_not_rescan_per_group():
    """A complexity regression, guarded by the clock because there is no other
    way to see it.

    Interior edges used to be recomputed inside the group loop by scanning the
    WHOLE edge map, with an `in members` list test inside that - so the cost
    was groups x edges x group size. It is invisible on the 300-face models
    everything else here uses, and fatal on anything real: a UV sphere, which
    is thousands of two-triangle coplanar groups, took **72.8 s** to merge and
    now takes 2.9 s for byte-identical output. A 500k-triangle scan extrapolated
    to about 40 minutes.

    The bound is loose on purpose. This machine does it in ~1 s; the old code
    needed well over ten at this size, so a 10x margin still catches a
    reintroduction without failing on a slow CI box.
    """
    tm = trimesh.creation.uv_sphere(count=[40, 40])
    t0 = time.perf_counter()
    pm = PolyMesh.from_trimesh(tm)
    elapsed = time.perf_counter() - t0

    assert len(pm.faces) < len(tm.faces), "the merge did nothing"
    groups = sum(1 for g in pm.source_faces if len(g) > 1)
    assert groups > 500, f"fixture no longer has many groups ({groups})"
    assert elapsed < 10.0, f"coplanar merge took {elapsed:.1f} s"


def test_inscribed_width_measures_what_a_hand_can_hold():
    """Area is the wrong measure of a fiddly part, and so is the bounding box.

    The splinter penalty is built on this, so it has to mean what it says: the
    largest circle that fits, which is the FATTEST point of the part.
    """
    assert inscribed_width(box(0, 0, 20, 20)) == pytest.approx(20.0, abs=0.3)

    # same area, wildly different to handle
    thin = box(0, 0, 40.0, 2.0)
    fat = box(0, 0, math.sqrt(80.0), math.sqrt(80.0))
    assert thin.area == pytest.approx(fat.area, rel=1e-9)
    assert inscribed_width(thin) < 3.0 < inscribed_width(fat)

    # An L with 8 mm legs measures ~9.4, not 8: the inner corner holds a bigger
    # circle than either leg. That is the intended answer - the piece has
    # somewhere to grip - and it is exactly the sort of thing a "min dimension"
    # reading of this function would get wrong.
    L = Polygon([(0, 0), (40, 0), (40, 8), (8, 8), (8, 40), (0, 40)])
    exact = 2 * 8 * math.sqrt(2) / (1 + math.sqrt(2))
    assert inscribed_width(L) == pytest.approx(exact, abs=0.3)

    assert inscribed_width(Polygon()) == 0.0


def test_the_splinter_floor_is_a_material_setting():
    """It is a statement about fingers, not geometry, so it is tunable."""
    mat = Material.from_flute("B")
    assert mat.splinter_width > 0
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(240.0)
    islands, _ = unfold(pm, trials=40)
    widths = [inscribed_width(p.cut) for p in build_parts(islands, mat)]
    assert widths and max(widths) > mat.splinter_width,         "a 240 mm sphere should give at least one properly handleable part"


def test_short_id_keeps_short_names_and_hashes_long_ones():
    """Label length is the biggest lever on how many islands carry an id, and
    an imported STL is routinely called something unengravable."""
    assert short_id("ARM") == "ARM"
    assert short_id("pikachu") == "PIKACHU"
    long = "OLD_VERSION_READ_DESCRIPTION_pikachu_1gen_flowalistik"
    got = short_id(long)
    assert len(got) == 8
    assert got == short_id(long), "must be stable across calls"
    assert got != short_id(long + "2"), "and tell two models apart"
    assert set(got) <= set("0123456789ABCDEF")
    for ch in got:
        assert strokefont.text_strokes(ch, 4.0), f"{ch!r} is not in the font"
    assert short_id("") == ""
    assert short_id("  ") == ""


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_labels_never_overlap_each_other(name, tm):
    """Two numbers burned through each other read as a third number that means
    nothing - worse than either being missing."""
    mat = Material.from_flute("B")
    mat.model_id = "T"
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    pairs = 0
    for labs, _body in _labelled(islands, mat).values():
        geo = [unary_union([LineString(s) for s in l.strokes if len(s) > 1])
               for l in labs]
        for a in range(len(geo)):
            for b in range(a + 1, len(geo)):
                pairs += 1
                assert geo[a].distance(geo[b]) > 0.0, (
                    f"{labs[a].text!r} and {labs[b].text!r} collide")
    assert pairs > 0, "nothing was labelled, so nothing was tested"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_seam_number_never_spans_more_of_its_edge_than_allowed(name, tm):
    """The rule that makes text shrink in congested areas rather than shout."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    for isl in islands:
        by_seam = {str(c.seam): c for c in isl.cuts if c.seam >= 0}
        plans = plan_folds(isl, mat)
        for lab in island_labels(isl, plans, mat,
                                 apply_relief(isl.outline(), plans)):
            if lab.kind != "seam":
                continue
            c = by_seam[lab.text]
            w = strokefont.text_width(lab.text, lab.height)
            assert w <= c.length * mat.label_edge_frac + 1e-9, (
                f"{lab.text!r} spans {w:.2f} mm of a {c.length:.2f} mm edge")


def test_a_tighter_edge_fraction_shrinks_the_text_before_dropping_it():
    """The behaviour the fraction exists for: crowd it and the text gets
    smaller, down to the floor and no further."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(260.0)
    islands, _ = unfold(pm, trials=40)

    def seam_heights(frac):
        mat.label_edge_frac = frac
        out = []
        for isl in islands:
            plans = plan_folds(isl, mat)
            out += [l.height for l in island_labels(
                isl, plans, mat, apply_relief(isl.outline(), plans))
                if l.kind == "seam"]
        return out

    # Derive the premise. At a loose fraction the 4 mm cap binds and nothing
    # shrinks however hard you squeeze - a first attempt asserted shrinking at
    # 0.2 on this mesh and found every label still at the cap, because its
    # edges are long. Pick the fraction that puts the median edge in the band
    # where the fit actually decides the height.
    lens = [c.length for isl in islands for c in isl.cuts if c.seam >= 0]
    med = float(np.median(lens))
    tight_frac = 4.0 / med          # ~4 mm of room: a 2-digit number wants 2.8

    loose = seam_heights(1.0)
    tight = seam_heights(tight_frac)
    assert loose and tight
    assert max(loose) == pytest.approx(mat.label_height),         "a loose fraction should let the cap bind"
    assert min(tight) < mat.label_height, "nothing shrank when crowded"
    assert np.mean(tight) < np.mean(loose)
    assert min(tight) >= mat.label_min_height - 1e-9, "shrank past the floor"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_fold_mark_sits_at_the_middle_of_its_fold(name, tm):
    """Centred along the edge and aligned to it. Offset across it only far
    enough to clear a relief slot, which is a real hole."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    checked = 0
    for isl in islands:
        plans = plan_folds(isl, mat)
        marks = [l for l in island_labels(isl, plans, mat,
                                          apply_relief(isl.outline(), plans))
                 if l.kind == "fold"]
        for lab in marks:
            c = np.vstack(lab.strokes).mean(axis=0)
            # the fold it belongs to is the nearest one of its own sense
            same = [p for p in plans
                    if ("M" if p.is_mountain else "V") == lab.text]
            best = min(same, key=lambda p: LineString(
                [p.fold.a, p.fold.b]).distance(Point(c)))
            f = best.fold
            d = (f.b - f.a) / np.linalg.norm(f.b - f.a)
            t = float((c - f.a) @ d) / f.length
            assert 0.3 <= t <= 0.7, (
                f"{lab.text} sits {t:.2f} along its fold, not at the middle")
            across = LineString([f.a, f.b]).distance(Point(c))
            assert across <= best.width / 2.0 + lab.height * 1.2 + 1e-6, (
                f"{lab.text} stands {across:.2f} mm off a fold with a "
                f"{best.width:.2f} mm slot")
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_no_label_is_engraved_into_a_relief_slot(name, tm):
    """A slot is a hole. A number across one is cut in half by the fold it was
    meant to explain, so labels are placed against the body, not the outline."""
    mat = Material.from_flute("B")
    mat.model_id = "T"
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)
    for labs, body in _labelled(islands, mat).values():
        room = body.buffer(1e-6)
        for lab in labs:
            for s in lab.strokes:
                if len(s) > 1:
                    assert room.contains(LineString(s)), (
                        f"{lab.kind} {lab.text!r} crosses a slot or the edge")


def test_the_island_id_outranks_the_fold_marks():
    """Priority is the point of building all three together: on a crowded part
    the id must survive and an M is what gives way."""
    mat = Material.from_flute("B")
    mat.model_id = "ZZ"
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(70.0)
    islands, _ = unfold(pm, trials=40)
    for isl in islands:
        plans = plan_folds(isl, mat)
        labs = island_labels(isl, plans, mat,
                             apply_relief(isl.outline(), plans))
        kinds = [l.kind for l in labs]
        if "part" in kinds:
            assert kinds.index("part") == 0, "the id is placed first"


def test_the_cut_sits_on_the_part_edge_by_default():
    """No offset unless asked for. A 0.15 mm kerf compensation is below what
    this is built for, and it is what makes a cut stop short of a notch."""
    mat = Material.from_flute("B")
    assert mat.edge_offset == 0.0
    pm = PolyMesh.from_trimesh(trimesh.creation.icosahedron())
    islands, _ = unfold(pm, trials=40)
    for part, isl in zip(build_parts(islands, mat), islands):
        body = apply_relief(isl.outline(), plan_folds(isl, mat))
        assert part.cut.area == pytest.approx(body.area, rel=1e-12)


def test_a_negative_edge_offset_cuts_the_part_back():
    """Thick board: two panels meeting at a shared edge collide unless each is
    cut back a little."""
    mat = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.icosahedron())
    islands, _ = unfold(pm, trials=40)

    mat.edge_offset = 0.0
    flush = sum(p.cut.area for p in build_parts(islands, mat))
    mat.edge_offset = -0.5
    smaller = sum(p.cut.area for p in build_parts(islands, mat))
    mat.edge_offset = 0.5
    bigger = sum(p.cut.area for p in build_parts(islands, mat))
    assert smaller < flush < bigger


def test_an_edge_offset_that_would_erase_a_part_falls_back_and_says_so():
    mat = Material.from_flute("B")
    mat.edge_offset = -500.0                     # far wider than any island
    pm = PolyMesh.from_trimesh(trimesh.creation.icosahedron())
    islands, _ = unfold(pm, trials=40)
    parts = build_parts(islands, mat)
    assert all(not p.cut.is_empty for p in parts)
    assert any("edge offset" in w for p in parts for w in p.warnings)


def test_corner_relief_is_inert_without_an_outward_offset():
    """Nothing buries a notch apex at 0 or below, so relief must not fire -
    it would only cut slits into a part that never lost its corners."""
    mat = Material.from_flute("B")
    mat.corner_relief = True
    pm = PolyMesh.from_trimesh(trimesh.creation.cylinder(radius=1, height=2,
                                                         sections=16))
    pm = pm.scale_to_height(200.0)
    islands, _ = unfold(pm, trials=40)
    for off in (0.0, -0.3):
        mat.edge_offset = off
        for part, isl in zip(build_parts(islands, mat), islands):
            body = apply_relief(isl.outline(), plan_folds(isl, mat))
            expect = offset_cut(body, off)
            assert part.cut.area == pytest.approx(expect.area, rel=1e-9)


def _slab_with_notch(deg, top=30.0):
    """Slab x[-30,30] y[-10,top] with a V cut down from the top, apex at (0,0)."""
    h = math.radians(deg / 2.0)
    w = top * math.tan(h)
    return Polygon([(-30, -10), (30, -10), (30, top), (w, top), (0.0, 0.0),
                    (-w, top), (-30, top)])


@pytest.mark.parametrize("deg,expect", [(90, 0.21), (60, 0.30), (30, 0.58),
                                        (20, 0.86), (10, 1.72), (5, 3.44)])
def test_kerf_offset_cannot_reach_into_a_notch(deg, expect):
    """The defect corner_relief exists for, stated as the geometry it is.

    A notch is an INNER corner, so join style is irrelevant - mitre, bevel and
    round only shape outer corners. The cut stops short by (kerf/2)/sin(t/2)
    whatever you pass.
    """
    body = _slab_with_notch(deg)
    apex = Point(0, 0)
    seen = set()
    for js, ml in ((2, 8.0), (2, 1.0), (3, 8.0), (1, 8.0)):
        cut = body.buffer(0.15, join_style=js, mitre_limit=ml)
        seen.add(round(apex.distance(cut.boundary), 6))
    assert len(seen) == 1, f"join style changed the notch: {seen}"
    assert seen.pop() == pytest.approx(expect, abs=0.01)


@pytest.mark.parametrize("deg", [60, 30, 20, 10, 5, 2])
def test_corner_relief_reaches_the_apex(deg):
    body = _slab_with_notch(deg)
    kerf = 0.30
    cut = offset_cut(body, kerf / 2)
    before = Point(0, 0).distance(cut.boundary)
    relieved = corner_relief(body, cut, kerf)
    after = Point(0, 0).distance(relieved.boundary)
    assert after <= kerf + 1e-6, (
        f"{deg} deg notch: cut still stops {after:.2f} mm short")
    if before > kerf:
        assert after < before


def test_corner_relief_leaves_ordinary_corners_alone():
    """A notch the beam already covers must not be cut at."""
    body = _slab_with_notch(120.0)
    kerf = 0.30
    cut = offset_cut(body, kerf / 2)
    assert Point(0, 0).distance(cut.boundary) < kerf
    assert corner_relief(body, cut, kerf).equals(cut)


def test_corner_overcut_runs_past_the_apex():
    body = _slab_with_notch(20.0)
    kerf = 0.30
    cut = offset_cut(body, kerf / 2)
    deep = corner_relief(body, cut, kerf, overcut=2.0)
    # 2 mm past the apex, on the far side, must now be outside the part
    assert not deep.contains(Point(0.0, -1.5))
    assert corner_relief(body, cut, kerf, overcut=0.0).contains(Point(0.0, -1.5))


def test_corner_relief_never_breaks_a_part_in_two():
    """A slit through a near-pinch would sever it: the neck is held together
    only by the material the kerf offset added, which is what relief removes.
    Better an unreachable apex than two pieces where the user expects one."""
    kerf = 0.30
    # an hourglass: two lobes joined by a neck thinner than the kerf fill
    neck = 0.05
    body = Polygon([(-20, -20), (20, -20), (20, -0.5), (neck, 0.0), (20, 0.5),
                    (20, 20), (-20, 20), (-20, 0.5), (-neck, 0.0),
                    (-20, -0.5)])
    assert body.is_valid
    cut = offset_cut(body, kerf / 2)
    out = corner_relief(body, cut, kerf)
    n_before = 1 if cut.geom_type == "Polygon" else len(cut.geoms)
    n_after = 1 if out.geom_type == "Polygon" else len(out.geoms)
    assert n_after <= n_before, "relief cut the part into pieces"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_corner_relief_keeps_parts_whole_and_costs_almost_nothing(name, tm):
    mat = Material.from_flute("B")
    mat.bed = (300.0, 300.0)
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=40)

    mat.corner_relief = False
    plain = build_parts(islands, mat)
    mat.corner_relief = True
    relieved = build_parts(islands, mat)

    def pieces(ps):
        return sum(len(p.cut.geoms) if p.cut.geom_type == "MultiPolygon" else 1
                   for p in ps)

    assert pieces(relieved) == pieces(plain), "relief fragmented a part"
    a0 = sum(p.cut.area for p in plain)
    a1 = sum(p.cut.area for p in relieved)
    assert a1 <= a0 + 1e-9, "relief added material"
    assert a1 > a0 * 0.98, f"relief ate {100 * (1 - a1 / a0):.1f}% of the net"


def test_geos_needs_set_precision_to_intersect_faces_sharing_an_edge():
    """Why every overlap measurement here snaps first.

    Two faces reach their shared corner down different tree paths and land one
    ulp apart. GEOS then returns the WHOLE of the smaller triangle as the
    intersection of two triangles that only touch along that edge - so a
    perfectly good net measures as catastrophically back-folded. This is the
    same class of GEOS failure as minimum_rotated_rectangle.
    """
    a = Polygon([(0.8410296852764221, -14.660246626375063),
                 (-2.825933799914169, -10.978942271621378),
                 (-4.754465724754761, -12.942846048485386)])
    b = Polygon([(-2.825933799914169, -10.978942271621378),
                 (0.8410296852764222, -14.660246626375063),   # 1 ulp apart
                 (6.968610201863562, -9.426195684)])
    assert a.intersection(b).area == pytest.approx(a.area),         "GEOS no longer mis-evaluates this; the snapping may be simplifiable"
    assert set_precision(a, 1e-9).intersection(
        set_precision(b, 1e-9)).area == 0.0


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_flat_area_equals_surface_area(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=120)
    flat = sum(i.area() for i in islands)
    assert flat == pytest.approx(tm.area, rel=1e-9)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_outline_union_loses_no_area(name, tm):
    """Regression: without coordinate snapping, shapely's union turns the
    ~1e-14 gaps between independently-placed faces into lost area and
    spurious fragments - about 1% of the net, silently."""
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=120)
    for isl in islands:
        assert isl.outline().area == pytest.approx(isl.area(), rel=1e-9)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_every_face_lands_in_exactly_one_island(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=120)
    seen = [f for i in islands for f in i.faces]
    assert sorted(seen) == list(range(len(pm.faces)))


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_fold_count_matches_spanning_forest(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, trials=120)
    folds = sum(len(i.folds) for i in islands)
    assert folds == len(pm.faces) - len(islands)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_no_overlaps_remain(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    _, rep = unfold(pm, trials=120)
    assert rep["unresolved"] == 0


def test_genus_one_unfolds_cleanly():
    """A torus is genus 1. That does NOT force extra islands - a dual spanning
    tree spans every face, so the net is always one connected piece; genus
    only means the cut set contains a cycle rather than being a tree. What it
    does do is make overlap much likelier, so this exercises the repair path.
    """
    tm = trimesh.creation.torus(major_radius=2, minor_radius=0.7,
                                major_sections=14, minor_sections=8)
    tm.apply_scale(200 / tm.extents.max())
    pm = PolyMesh.from_trimesh(tm)
    islands, rep = unfold(pm, trials=60)
    assert rep["unresolved"] == 0
    assert rep["isometry_error"] < 1e-12
    assert sum(i.area() for i in islands) == pytest.approx(tm.area, rel=1e-9)
    seen = [f for i in islands for f in i.faces]
    assert sorted(seen) == list(range(len(pm.faces)))


# ------------------------------------------------------------ bed limiting

@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_split_to_fit_respects_the_bed(name, tm):
    pm = PolyMesh.from_trimesh(tm)
    limit = (400.0, 260.0)
    islands, rep = unfold(pm, trials=80, max_size=limit)
    assert rep["oversize_islands"] == 0
    for isl in islands:
        pts = np.vstack(list(isl.faces.values()))
        assert fits(pts, *limit), f"island {isl.id} exceeds the bed"


def test_split_island_detaches_exactly_one_piece():
    """Cutting a fold must hand back the two halves and nothing else.

    Conserving faces and area is the point: this is a local edit, so anything
    that vanished or got duplicated would be a part you cannot cut or one you
    cut twice.
    """
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(280)
    islands, _ = unfold(pm, trials=40)
    isl = max(islands, key=lambda i: len(i.faces))
    fold = isl.folds[len(isl.folds) // 2]

    out = split_island(isl, fold.f0, fold.f1, new_id=999, new_seam=777)
    assert out is not None
    a, b = out
    assert set(a.faces) | set(b.faces) == set(isl.faces), "faces must be conserved"
    assert not (set(a.faces) & set(b.faces)), "a face cannot land in both halves"
    assert len(a.faces) and len(b.faces), "both halves must be non-empty"
    assert a.area() + b.area() == pytest.approx(isl.area(), rel=1e-12)
    assert len(a.folds) + len(b.folds) == len(isl.folds) - 1, "one fold is spent"

    # the spent fold becomes a matching pair of cuts, which is what pairs tabs
    seams_a = [c.seam for c in a.cuts if c.seam == 777]
    seams_b = [c.seam for c in b.cuts if c.seam == 777]
    assert len(seams_a) == 1 and len(seams_b) == 1

    # each half is still one connected run of faces you can fold
    for half in (a, b):
        g = nx.Graph()
        g.add_nodes_from(half.faces)
        for fo in half.folds:
            g.add_edge(fo.f0, fo.f1)
        assert nx.number_connected_components(g) == 1, "a half came out in pieces"


def test_split_island_refuses_an_edge_that_is_not_a_fold():
    """Two faces can be neighbours in 3D and already cut apart in the net."""
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(280)
    islands, _ = unfold(pm, trials=40)
    isl = max(islands, key=lambda i: len(i.faces))
    folded = {frozenset((f.f0, f.f1)) for f in isl.folds}
    pairs = [(a, b) for a in isl.faces for b in isl.faces
             if a < b and frozenset((a, b)) not in folded]
    assert pairs, "expected some face pair that is not a fold"
    a, b = pairs[0]
    assert split_island(isl, a, b, new_id=999, new_seam=777) is None


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_bed_split_is_maximal(name, tm):
    """No two finished pieces could have been left joined.

    This is the quality bar for bed splitting, and it is stronger than "every
    piece fits": a splitter that cuts every face apart also passes that one.
    Growing each piece until the bed stops it makes maximality automatic -
    after a cut the child piece is final while the parent only grows, and a
    superset of something that did not fit cannot fit.

    Checked in the ORIGINAL frame. split_to_fit re-roots every component
    afterwards, so two pieces' coordinates then live in different frames and
    stacking them measures nothing - they would appear to overlap at the
    origin and any merge would look like it fitted. Same trap as
    component-aware overlap counting.
    """
    pm = PolyMesh.from_trimesh(tm).scale_to_height(280)
    limit = (400.0, 260.0)
    forest, pos, _, _ = search(pm, pm.dual_graph(), trials=60)
    cut, _, ncuts, _, _ = split_to_fit(pm, forest, pos, *limit)

    removed = set(map(frozenset, forest.edges())) - set(map(frozenset, cut.edges()))
    assert len(removed) == ncuts, "reported split count must match the arcs removed"

    comps = [set(c) for c in nx.connected_components(cut)]
    comp_of = {f: i for i, c in enumerate(comps) for f in c}
    for e in removed:
        u, v = tuple(e)
        merged = comps[comp_of[u]] | comps[comp_of[v]]
        assert not fits(np.vstack([pos[f] for f in merged]), *limit),             f"{name}: the cut {u}-{v} was unnecessary, those pieces both fit joined"


def test_scored_tree_search_never_does_worse_than_the_first_clean_tree():
    """Scoring candidate trees must not be able to lose.

    The old behaviour was "return the first overlap-free tree". That tree is
    still the first candidate the scored search evaluates, so the best of the
    candidates can only tie it or beat it. This is what stops the extra search
    cost from ever buying a worse result.
    """
    m = Material.from_flute("B")
    pad = 2 * m.tab_depth + 2 * m.kerf
    limit = (m.usable_bed[0] - pad, m.usable_bed[1] - pad)
    for tm in (trimesh.creation.icosphere(subdivisions=1),
               trimesh.creation.cylinder(radius=1, height=2, sections=16)):
        pm = PolyMesh.from_trimesh(tm).scale_to_height(280)
        G = pm.dual_graph()

        first, pos_a, bad_a, _ = search(pm, G, trials=60, seed=3)
        scored, pos_b, bad_b, _ = search(pm, G, trials=60, seed=3, max_size=limit)
        if bad_a or bad_b:
            continue                       # no clean tree to compare against

        _, _, cuts_first, _, _ = split_to_fit(pm, first, pos_a, *limit)
        _, _, cuts_scored, _, _ = split_to_fit(pm, scored, pos_b, *limit)
        assert cuts_scored <= cuts_first, (
            f"scored search returned a worse tree: {1+cuts_scored} pieces "
            f"vs {1+cuts_first} for the first clean tree")


def test_bed_split_only_removes_existing_arcs():
    """Splitting may cut the unfold tree, never invent a join that was not
    there - a piece has to stay a connected run of faces you can fold."""
    pm = PolyMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=1))
    pm = pm.scale_to_height(280)
    forest, pos, _, _ = search(pm, pm.dual_graph(), trials=60)
    cut, _, _, _, _ = split_to_fit(pm, forest, pos, 400.0, 260.0)
    before = set(map(frozenset, forest.edges()))
    after = set(map(frozenset, cut.edges()))
    assert after <= before
    assert set(cut.nodes()) == set(forest.nodes()), "no face may be dropped"


def test_min_rect_beats_axis_aligned_for_a_diagonal():
    """A net rotated 45 degrees fits a bed its axis-aligned box would not."""
    s = 100.0
    pts = np.array([[0, 0], [s, s], [2 * s, 0], [s, -s]], dtype=float)
    aabb = pts.max(axis=0) - pts.min(axis=0)
    w, h = min_rect_size(pts)
    assert aabb[0] == pytest.approx(200.0)
    assert min(w, h) == pytest.approx(math.hypot(s, s), rel=1e-9)
    assert max(w, h) < aabb[0]


def test_min_area_rect_encloses_its_points():
    """The rectangle must contain the cloud, must not be smaller than the
    cloud's convex hull, and must never do worse than the axis-aligned box."""
    from shapely.geometry import MultiPoint
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(3, 40))
        pts = rng.normal(size=(n, 2)) * rng.uniform(1.0, 500.0)
        w, h, ang, centre = min_area_rect(pts)
        d = np.array([math.cos(ang), math.sin(ang)])
        q = pts - np.asarray(centre)
        u, v = q @ d, q @ np.array([-d[1], d[0]])
        scale = max(1.0, w, h)
        assert u.max() - u.min() <= w + 1e-9 * scale, "points stick out sideways"
        assert v.max() - v.min() <= h + 1e-9 * scale, "points stick out endwise"
        hull = MultiPoint([tuple(pt) for pt in pts]).convex_hull.area
        aabb = pts.max(axis=0) - pts.min(axis=0)
        assert w * h >= hull - 1e-6 * scale ** 2, "smaller than the hull it holds"
        assert w * h <= aabb[0] * aabb[1] + 1e-6 * scale ** 2, "worse than the AABB"


def test_min_rect_area_is_invariant_under_translation():
    """Where a net happens to sit cannot change the size of its minimum box.
    GEOS' rotated-rectangle broke precisely this: the same cylinder net gave
    687 x 436 mm centred on the origin and a zero-width sliver at its natural
    offset a few hundred mm out. The achieving rectangle can tie between two
    orientations, so the invariant is on area, which is unique."""
    rng = np.random.default_rng(11)
    for _ in range(100):
        n = int(rng.integers(3, 40))
        pts = rng.normal(size=(n, 2)) * rng.uniform(1.0, 400.0)
        w, h = min_rect_size(pts)
        base = w * h
        for off in ([300.0, 700.0], [-1200.0, 450.0], [5000.0, 5000.0]):
            w, h = min_rect_size(pts + np.array(off))
            assert w * h == pytest.approx(base, rel=1e-9)


def _reference_hull(pts):
    """The monotone chain as it was before it was handed plain floats - kept
    here, verbatim in its arithmetic, as the definition the fast path must
    reproduce."""
    p = np.unique(np.asarray(pts, dtype=float), axis=0)
    if len(p) < 3:
        return p

    def chain(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                cross = ((b[0] - a[0]) * (q[1] - a[1])
                         - (b[1] - a[1]) * (q[0] - a[0]))
                if cross > 0.0:
                    break
                out.pop()
            out.append(q)
        return out

    lower, upper = chain(p), chain(p[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def test_the_convex_hull_is_unchanged_by_the_fast_path():
    """`_convex_hull` runs its loop over Python floats rather than numpy rows,
    for speed. Bed splitting and every packer's orientation go through it, so
    "nearly the same hull" is not good enough - it has to be the same bits,
    on random clouds, on collinear runs, and on real nets."""
    from cardboard_unfolder.unfold import _convex_hull
    rng = np.random.default_rng(5)
    clouds = [rng.normal(size=(int(rng.integers(1, 300)), 2))
              * rng.uniform(1e-3, 500.0) + rng.uniform(-900, 900, 2)
              for _ in range(200)]
    clouds += [np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
               np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
               np.round(rng.uniform(0, 5, (400, 2)))]   # heavy ties
    for name, tm in ALL:
        pm = PolyMesh.from_trimesh(tm).scale_to_height(200)
        islands, _ = unfold(pm, trials=10, seed=0)
        clouds += [shapely.get_coordinates(i.outline()) for i in islands]
    for pts in clouds:
        got, want = _convex_hull(pts), _reference_hull(pts)
        assert got.shape == want.shape and np.array_equal(got, want)


def _sliver_next_to_a_triangle(lift):
    """A 372 x 1.94 mm quad whose FIRST edge is the short one, its far corner
    lifted `lift` mm out of plane, sharing its long edge with a triangle -
    the shape of polygon 129 on the user's helmet."""
    V = np.array([[0, 0, 0], [0, 1.94, 0], [372, 1.94, lift], [372, 0, 0],
                  [186, -50, 0]], dtype=float)
    pm = PolyMesh.from_rings(V, [[0, 1, 2, 3], [3, 4, 0]], merge=False,
                             planar_tol=None)
    pos = unfold_tree(pm, pm.dual_graph(), 1)       # rooted at the triangle
    return pm, pos


def test_plane_map_puts_a_bent_face_where_the_unfold_put_it():
    """The user's screenshot: the cut outline right, the face drawn beside it.
    write_net_uv pushes each Blender polygon through `plane_map`, which took
    the face's whole orientation from its FIRST edge - here 1.94 mm long. A
    bent face cannot be flattened exactly, so that edge comes out a little
    short and turned, and 372 mm along the sliver the turn was 11.8 mm.
    Fitted to every corner instead, the map lands within the bend itself."""
    pm, pos = _sliver_next_to_a_triangle(0.36)
    fn = pm.plane_map(0, pos[0])
    off = np.linalg.norm(fn(pm.vertices[pm.faces[0]]) - pos[0], axis=1).max()
    assert off < 0.5, f"corners land {off:.2f} mm from the unfold"
    # and a flat face is still exact
    pm, pos = _sliver_next_to_a_triangle(0.0)
    fn = pm.plane_map(0, pos[0])
    assert np.abs(fn(pm.vertices[pm.faces[0]]) - pos[0]).max() < 1e-9


def test_min_area_rect_survives_degenerate_clouds():
    """Collinear and coincident inputs must return a usable answer, not raise
    and not report a size that hides a real extent."""
    line = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.0], [2.5, 0.0]])
    w, h, _, _ = min_area_rect(line)
    assert max(w, h) == pytest.approx(10.0)
    assert min(w, h) == pytest.approx(0.0)

    w, h, _, _ = min_area_rect(np.zeros((5, 2)))
    assert (w, h) == (0.0, 0.0)

    diag = np.array([[0.0, 0.0], [3.0, 4.0], [1.5, 2.0]])
    w, h, _, _ = min_area_rect(diag)
    assert max(w, h) == pytest.approx(5.0)


def test_split_to_fit_splits_a_net_that_sits_off_the_origin():
    """Regression. The cylinder net at costume scale lands a few hundred mm
    from the origin, where GEOS' rotated-rectangle collapsed to a sliver: a
    687 x 436 mm island reported that it fitted a 559 x 359 mm bed, so bed
    splitting stopped one cut early and shipped an uncuttable part."""
    limit = (559.4, 359.4)
    tm = trimesh.creation.cylinder(radius=1, height=2, sections=16)
    tm.apply_scale(200.0 / tm.extents.max())
    pm = PolyMesh.from_trimesh(tm).scale_to_height(220)
    islands, rep = unfold(pm, trials=80, max_size=limit)
    assert rep["oversize_islands"] == 0
    for isl in islands:
        pts = np.vstack(list(isl.faces.values()))
        assert fits(pts, *limit),             f"island {isl.id} is {min_rect_size(pts)} on a {limit} bed"


# ------------------------------------------------------------ fold geometry

def test_tabs_are_optional():
    """Most cardboard is taped or gets a glued backing strip, and the flaps
    cost about a fifth of the sheet - so tabs off must mean no tab area."""
    tm = trimesh.creation.icosphere(subdivisions=1)
    pm = PolyMesh.from_trimesh(tm).scale_to_height(200)
    islands, _ = unfold(pm, trials=40)

    off = Material.from_flute("B", use_tabs=False)
    on = Material.from_flute("B", use_tabs=True)
    a = sum(p.cut.area for p in build_parts(islands, off))
    b = sum(p.cut.area for p in build_parts(islands, on))
    assert b > a * 1.02, f"tabs should add area (off {a:.0f}, on {b:.0f})"

    bare = sum(i.area() for i in islands)
    assert a <= bare * 1.02, "with tabs off the part is the island plus kerf only"


def test_fixed_relief_width_overrides_the_derived_one():
    """Adaptive width is what the board needs; fixed is what the machine wants."""
    m = Material.from_flute("B", relief_adaptive=False, relief_fixed=2.5)
    for deg in (20, 60, 90, 150):
        assert relief_width(math.radians(deg), m) == pytest.approx(2.5)
    m.relief_fixed = 999.0
    assert relief_width(math.radians(90), m) == pytest.approx(m.max_relief),         "a fixed width must still respect max_relief"


def test_connect_ratio_sets_how_much_material_survives():
    """0.5 means half the fold line is left connected."""
    m = Material.from_flute("B")
    a, b = np.array([0.0, 0.0]), np.array([200.0, 0.0])
    for ratio in (0.25, 0.5, 0.75):
        m.connect_ratio = ratio
        segs = perf_segments(a, b, m)
        cut = sum(float(np.linalg.norm(q - p)) for p, q in segs)
        left = (200.0 - cut) / 200.0
        assert left == pytest.approx(ratio, abs=0.08),             f"ratio {ratio}: {left:.2f} of the fold survived"


def test_short_folds_keep_min_connect_of_material():
    """A blanket ratio leaves a proportionally small - and absolutely tiny -
    amount holding a short joint together. The floor is what stops it tearing."""
    m = Material.from_flute("B", min_connect=10.0, connect_ratio=0.2)
    for length in (6.0, 12.0, 25.0, 120.0):
        a, b = np.array([0.0, 0.0]), np.array([length, 0.0])
        segs = perf_segments(a, b, m)
        cut = sum(float(np.linalg.norm(q - p)) for p, q in segs)
        assert length - cut >= min(10.0, length) - 1e-6,             f"a {length}mm fold kept only {length - cut:.1f}mm of material"
    a, b = np.array([0.0, 0.0]), np.array([8.0, 0.0])
    assert perf_segments(a, b, m) == [], "a fold shorter than the floor is left solid"


def test_round_cut_ends_changes_the_slot_not_its_span():
    """Radiused ends must not make the slot overhang the fold it relieves."""
    from cardboard_unfolder.folds import _strip
    a, b = np.array([0.0, 0.0]), np.array([50.0, 0.0])
    square = _strip(a, b, 4.0, round_ends=False)
    round_ = _strip(a, b, 4.0, round_ends=True)
    assert round_.area < square.area, "rounding removes the corners"
    assert round_.bounds[0] >= square.bounds[0] - 1e-9
    assert round_.bounds[2] <= square.bounds[2] + 1e-9
    assert round_.bounds[2] - round_.bounds[0] == pytest.approx(50.0, abs=1e-6)


def test_relief_width_matches_closed_form():
    m = Material.from_flute("B")
    m.crush = 1.0
    for deg in (15, 30, 45, 60, 90, 120):
        a = math.radians(deg)
        assert relief_width(a, m) == pytest.approx(
            2 * m.pivot_depth * math.tan(a / 2), rel=1e-12)


def test_relief_is_twice_thickness_at_ninety_degrees():
    """The one number to sanity-check against packaging practice: a 90 degree
    V-groove opens to 2t (US 5,427,309 specifies 'substantially twice the
    thickness')."""
    m = Material.from_flute("B")
    m.crush = 1.0
    assert relief_width(math.radians(90), m) == pytest.approx(2 * m.pivot_depth)


def test_straight_slot_constraint_is_satisfied():
    """A laser cuts parallel walls, not a V. The binding constraint is the top
    corner clearing the hinge: (w/2)cos(a/2) - d sin(a/2) >= 0, which yields
    the same width. Verify it closes exactly."""
    m = Material.from_flute("B")
    m.crush = 1.0
    for deg in (20, 45, 90, 130):
        a = math.radians(deg)
        w = relief_width(a, m)
        assert abs((w / 2) * math.cos(a / 2) - m.pivot_depth * math.sin(a / 2)) < 1e-12


def test_crush_factor_scales_linearly_and_is_conservative():
    m = Material.from_flute("B")
    a = math.radians(90)
    m.crush = 1.0
    full = relief_width(a, m)
    m.crush = 0.75
    assert relief_width(a, m) == pytest.approx(0.75 * full)
    assert relief_width(a, m) < full


def test_pivot_depth_uses_the_liner_midplane():
    m = Material.from_flute("B")
    assert m.pivot_depth == pytest.approx(m.thickness - m.liner / 2)
    assert m.pivot_depth < m.thickness


def test_flat_fold_needs_no_relief():
    m = Material.from_flute("B")
    assert relief_width(0.0, m) == 0.0


@pytest.mark.parametrize("grade", ["E", "B", "C"])
def test_a_coarse_perforation_warns_instead_of_being_silently_rescaled(grade):
    """The flute constraint is advice now, not a clamp.

    Dash period wants to stay well under the flute pitch, or consecutive flutes
    get very different treatment - some ties on a glue line, some over a void.
    That used to be enforced by rescaling, which made the dash-length control
    look broken: anything over 2 mm came back as 1.08 mm. The rule is real but
    not absolute - it matters for a fine score, far less for a slot with long
    bridges - so the user's number is honoured and the risk is reported.
    """
    m = Material.from_flute(grade)
    m.perf_cut = 20.0
    segs = perf_segments(np.array([0.0, 0.0]), np.array([200.0, 0.0]), m)
    assert segs, "a coarse perforation must still produce cuts"
    length = float(np.linalg.norm(segs[0][1] - segs[0][0]))
    assert length == pytest.approx(20.0, abs=1e-6),         f"asked for 20mm dashes, got {length:.2f}mm"
    assert "flute pitch" in flute_warning(m), "and it must say so"

    m.perf_cut = m.pitch / 8.0            # comfortably inside the guidance
    assert flute_warning(m) == "", "a fine perforation must not warn"

def test_perf_dashes_stay_inside_the_fold():
    m = Material.from_flute("B")
    a, b = np.array([10.0, 5.0]), np.array([60.0, 5.0])
    for p, q in perf_segments(a, b, m):
        assert 10.0 - 1e-9 <= p[0] <= 60.0 + 1e-9
        assert 10.0 - 1e-9 <= q[0] <= 60.0 + 1e-9


def test_relief_slots_keep_the_part_in_one_piece():
    """Bridges across the slot are the hinge. Without them a 90 degree fold
    would cut the part fully in two."""
    m = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(120)
    islands, _ = unfold(pm, trials=40)
    isl = islands[0]
    plans = plan_folds(isl, m)
    assert {p.style for p in plans} == {"relief"}, "90 degree folds need relief"
    after = apply_relief(isl.outline(), plans)
    assert after.geom_type == "Polygon", "bridges must keep it connected"
    assert after.area < isl.outline().area, "relief must remove material"


def test_shallow_folds_are_perforated_not_relieved():
    m = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(
        trimesh.creation.cylinder(radius=1, height=2, sections=24)).scale_to_height(200)
    islands, _ = unfold(pm, trials=40)
    styles = [p.style for p in plan_folds(islands[0], m)]
    assert styles.count("perf") > styles.count("relief")


def test_kerf_grows_the_part():
    m = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(100)
    islands, _ = unfold(pm, trials=20)
    o = islands[0].outline()
    assert offset_cut(o, m.kerf / 2).area > o.area
    assert offset_cut(o, 0.0).area == pytest.approx(o.area)


# ------------------------------------------------------------------- tabs

def test_one_tab_per_seam_on_a_cube():
    m = Material.from_flute("B")
    pm = PolyMesh.from_trimesh(trimesh.creation.box()).scale_to_height(110)
    islands, _ = unfold(pm, trials=40)
    seams = sorted({c.seam for i in islands for c in i.cuts if c.seam >= 0})
    tabs, skipped = place_tabs(islands, m)
    placed = sorted(t.seam for lst in tabs.values() for t in lst)
    assert not skipped
    assert placed == seams


def test_tabs_never_overlap_their_own_island():
    m = Material.from_flute("B")
    for tm in (trimesh.creation.box(), trimesh.creation.icosahedron()):
        tm.apply_scale(150 / tm.extents.max())
        pm = PolyMesh.from_trimesh(tm)
        islands, _ = unfold(pm, trials=40)
        tabs, _ = place_tabs(islands, m)
        for isl in islands:
            body = isl.outline()
            for t in tabs[isl.id]:
                assert t.poly.intersection(body).area < 1e-6
            lst = tabs[isl.id]
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    assert lst[i].poly.intersection(lst[j].poly).area < 1e-6


# ------------------------------------------------------------- end to end

@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_pipeline_produces_parts_that_fit_the_bed(name, tm):
    m = Material.from_flute("B")
    pad = 2 * m.tab_depth + 2 * m.kerf
    limit = (m.usable_bed[0] - pad, m.usable_bed[1] - pad)
    pm = PolyMesh.from_trimesh(tm).scale_to_height(220)
    islands, _ = unfold(pm, trials=80, max_size=limit)
    parts = build_parts(islands, m)
    sheets, oversized = pack(parts, m)
    assert not oversized, [(p.island_id, p.size) for p in oversized]
    assert sheets >= 1
    for p in parts:
        assert not p.cut.is_empty
        assert p.cut.is_valid
        x0, y0, x1, y1 = p.bounds
        assert x0 >= -1e-6 and y0 >= -1e-6
        assert x1 <= m.bed[0] + 1e-6 and y1 <= m.bed[1] + 1e-6


def _boxes_part(iid, rects, sheet=0):
    """A Part made of plain rectangles, for packing tests that are about the
    search rather than about any real net."""
    from cardboard_unfolder.layout import Part
    from shapely.ops import unary_union
    geom = unary_union([box(*r) for r in rects])
    empty = MultiLineString([])
    p = Part(island_id=iid, cut=geom, perf_m=empty, perf_v=empty,
             engrave=empty)
    p.sheet = sheet
    return p


def test_the_raster_search_finds_a_pocket_the_corner_search_cannot():
    """The capability `tighten` was missing, in isolation.

    `find_spot` offers the cross product of the placed parts' edges, which is
    the right heuristic for building against a skyline and the wrong one for a
    hole in the middle of a finished sheet - the pocket's corners are not on
    any part's corner grid. The raster search offers every position, so it
    does not care where the opening is.
    """
    from cardboard_unfolder.layout import (find_spot, raster_spot,
                                           sheet_occupancy)
    m = Material.from_flute("B")
    m.bed = (300.0, 300.0)
    m.part_gap = 1.0
    W, H = m.usable_bed
    mg = m.sheet_margin

    # a frame with a clear window in the middle of it
    lo, hi = mg, mg + W
    mid0, mid1 = mg + W * 0.38, mg + W * 0.62
    frame = _boxes_part(0, [(lo, lo, hi, mid0),            # below
                            (lo, mid1, hi, mg + H),        # above
                            (lo, mid0, mid0, mid1),        # left
                            (mid1, mid0, hi, mid1)])       # right
    side = W * 0.2                                          # fits the window
    stray = _boxes_part(1, [(0.0, 0.0, side, side)], sheet=1)

    corner = find_spot(stray, [frame.cut], [frame.cut.bounds], W, H,
                       m.part_gap, mg)
    assert corner is None, "the fixture is wrong - the skyline search found it"

    occ, shape = sheet_occupancy([frame, stray], 0, m)
    spot = raster_spot(stray, np.fft.rfft2(occ, s=shape), shape, m)
    assert spot is not None, "the raster search missed a clear central pocket"


def test_tighten_reaches_past_its_expensive_search_budget():
    """Regression: a budget spent in sheet order always fails on late sheets.

    Measured on a torus at 200 mm - the last sheet held ONE part at 13% full
    while sheet 17 sat at 56%, and the part fitted sheet 17 perfectly well.
    POCKET_BUDGET is 16 deep searches and sheet 17 is the seventeenth target,
    so the exact search was never offered it. The raster search runs before
    that budget and is not gated by it.
    """
    from cardboard_unfolder.layout import tighten, POCKET_BUDGET
    m = Material.from_flute("B")
    m.bed = (300.0, 300.0)
    m.part_gap = 1.0
    W, H = m.usable_bed
    mg = m.sheet_margin
    home = POCKET_BUDGET + 1            # one past where the budget runs out

    parts = []
    for k in range(home + 2):
        if k == home:
            mid0, mid1 = mg + W * 0.38, mg + W * 0.62
            parts.append(_boxes_part(k, [(mg, mg, mg + W, mid0),
                                         (mg, mid1, mg + W, mg + H),
                                         (mg, mid0, mid0, mid1),
                                         (mid1, mid0, mg + W, mid1)], sheet=k))
        else:                           # solid: nothing will fit beside it
            parts.append(_boxes_part(k, [(mg, mg, mg + W, mg + H)], sheet=k))
    stray = _boxes_part(999, [(0.0, 0.0, W * 0.2, W * 0.2)],
                        sheet=home + 2)
    parts.append(stray)

    before = home + 3
    after = tighten(parts, before, m)
    assert after == before - 1, (
        f"the stray should have moved onto sheet {home} and emptied its own "
        f"({before} sheets -> {after})")
    assert stray.sheet == home


def test_the_fft_collision_test_agrees_with_counting_the_overlap():
    """The one numerical claim the raster packer rests on.

    `free_offsets` says a part fits at an offset because a cross-correlation
    came back below 0.5 - which is only the right answer if that correlation
    really is the overlap count at that offset. It is a float transform, so it
    lands on ~1e-12 rather than 0, and getting the index arithmetic off by one
    would produce a packer that quietly places parts on top of each other.
    Checked against the definition, by hand, over every offset.
    """
    from cardboard_unfolder.layout import free_offsets
    rng = np.random.default_rng(3)
    occ = (rng.random((23, 29)) < 0.25).astype(float)
    mask = (rng.random((5, 7)) < 0.5)
    mask[0, 0] = mask[-1, -1] = True        # pin the extent

    gh, gw = occ.shape
    h, w = mask.shape
    got = free_offsets(np.fft.rfft2(occ, s=(gh, gw)), (gh, gw), mask)
    assert got.shape == (gh - h + 1, gw - w + 1)
    for y in range(gh - h + 1):
        for x in range(gw - w + 1):
            direct = float((occ[y:y + h, x:x + w] * mask).sum())
            assert bool(got[y, x]) == (direct == 0.0), (
                f"offset ({x}, {y}): correlation says "
                f"{'free' if got[y, x] else 'blocked'}, direct count {direct}")


def test_the_raster_mask_covers_the_whole_part():
    """It must be a SUPERSET of the shape, never a sample of it.

    Sampling cell centres alone under-covers by up to a cell, and a cell is
    2 mm of real collision on a real laser. Dilating the sampled grid is the
    obvious fix and is not sufficient: a 4-connected dilation cannot reach the
    cell at a convex corner, whose only filled neighbour is diagonal - which
    is exactly what this caught. Buffering by half a cell diagonal before
    sampling is provable instead of nearly right.
    """
    import shapely
    from cardboard_unfolder.layout import raster_mask
    from shapely.geometry import Polygon as P
    res, gap = 2.0, 1.0
    # an L, so the test says something about a concave outline as well
    geom = P([(0, 0), (40, 0), (40, 9), (9, 9), (9, 40), (0, 40)])
    m, x0, y0 = raster_mask(geom, res, gap)

    g = geom.buffer(gap / 2.0)
    dense = shapely.get_coordinates(g.exterior.segmentize(res / 3.0))
    for x, y in dense:
        ix, iy = int((x - x0) / res), int((y - y0) / res)
        assert 0 <= ix < m.shape[1] and 0 <= iy < m.shape[0], \
            f"({x:.1f}, {y:.1f}) falls outside the grid entirely"
        assert m[iy, ix], \
            f"point ({x:.1f}, {y:.1f}) is outside the mask that represents it"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_the_raster_packer_leaves_no_overlap_and_nothing_off_the_sheet(name, tm):
    from cardboard_unfolder.layout import raster_pack
    m = Material.from_flute("B")
    m.part_gap = 1.0
    pad = 2 * m.tab_depth + 2 * m.kerf
    limit = (m.usable_bed[0] - pad, m.usable_bed[1] - pad)
    pm = PolyMesh.from_trimesh(tm).scale_to_height(280)
    islands, _ = unfold(pm, trials=30, max_size=limit)
    parts = build_parts(islands, m)
    sheets, over = raster_pack(parts, m)

    placed = [p for p in parts if p.sheet >= 0]
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if a.sheet != b.sheet:
                continue
            hit = a.cut.buffer(-1e-9).intersection(b.cut.buffer(-1e-9))
            assert hit.area <= 1e-9, (
                f"parts {a.island_id} and {b.island_id} overlap by "
                f"{hit.area:.3f} mm2 on sheet {a.sheet}")
    for p in placed:
        x0, y0, x1, y1 = p.bounds
        assert x0 >= m.sheet_margin - 1e-6 and y0 >= m.sheet_margin - 1e-6
        assert x1 <= m.sheet_margin + m.usable_bed[0] + 1e-6
        assert y1 <= m.sheet_margin + m.usable_bed[1] + 1e-6


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_an_unannotated_part_has_exactly_the_annotated_cut(name, tm):
    """The optimiser packs every finalist to count its sheets, and builds
    those parts with `annotate=False` - no labels, no perforation. That is
    only sound if what the packer reads is untouched, so the cut must be the
    same geometry to the last bit, with fingers and tabs both off and on."""
    pm = PolyMesh.from_trimesh(tm).scale_to_height(200)
    for fingers, tabs in ((False, False), (True, True)):
        m = Material.from_flute("B")
        m.use_fingers, m.use_tabs = fingers, tabs
        pad = 2 * m.tab_depth + 2 * m.kerf
        islands, _ = unfold(pm, trials=20, seed=0,
                            max_size=(m.usable_bed[0] - pad,
                                      m.usable_bed[1] - pad))
        full = build_parts(islands, m)
        bare = build_parts(islands, m, annotate=False)
        assert [p.island_id for p in full] == [p.island_id for p in bare]
        for a, b in zip(full, bare):
            assert shapely.to_wkb(a.cut) == shapely.to_wkb(b.cut),                 f"{name} island {a.island_id}: cut differs without labels"
            assert b.engrave.is_empty and b.perf_m.is_empty


def _clear_of_each_other(parts, mat):
    """Every placed pair at least the part gap apart, everything on its bed."""
    W, H = mat.usable_bed
    m = mat.sheet_margin
    by = {}
    for p in parts:
        if p.sheet >= 0:
            by.setdefault(p.sheet, []).append(p)
            x0, y0, x1, y1 = p.bounds
            assert x0 >= m - 1e-6 and y0 >= m - 1e-6, p.island_id
            assert x1 <= m + W + 1e-6 and y1 <= m + H + 1e-6, p.island_id
    for group in by.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                d = group[a].cut.distance(group[b].cut)
                assert d >= mat.part_gap - 1e-6, (group[a].island_id,
                                                  group[b].island_id, d)


@pytest.mark.parametrize("bed", [(400.0, 900.0), (900.0, 400.0)],
                         ids=["portrait", "landscape"])
def test_the_last_sheet_is_packed_low_along_the_long_side(bed):
    """Three small parts spread along a last sheet. The pass packs them as
    short a way along the bed's LONG side as they go - up a portrait bed,
    across a landscape one - so the rest is one strip of board kept for
    another job (the user's rule: "take the ratio on the long dimension")."""
    from cardboard_unfolder.layout import compact_last, sheet_equivalents
    m = Material.from_flute("B")
    m.bed = bed
    long_ax = 0 if bed[0] > bed[1] else 1
    parts = []
    for k, at in enumerate((20.0, 300.0, 700.0)):
        xy = [m.sheet_margin + 10.0, m.sheet_margin + 10.0]
        xy[long_ax] = m.sheet_margin + at
        parts.append(_blank(box(xy[0], xy[1], xy[0] + 60, xy[1] + 40), 0))
    eq0, off0 = sheet_equivalents(parts, m)
    assert compact_last(parts, m)
    eq1, off1 = sheet_equivalents(parts, m)
    assert off1 > off0 and eq1 < eq0
    reach = max(p.bounds[2 + long_ax] for p in parts) - m.sheet_margin
    assert reach < 150.0, reach         # three 60 x 40 parts, side by side
    _clear_of_each_other(parts, m)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_packing_the_last_sheet_low_keeps_every_part_clear(name, tm):
    """After pack() - which now ends with the last-sheet pass - every part is
    on its bed and a part gap from every other. The raster is conservative by
    construction; this is the check that nothing it moved came to rest on
    top of anything else, on a tall bed where the pass has most to do."""
    m = Material.from_flute("B")
    m.bed = (400.0, 900.0)
    pad = 2 * m.tab_depth + 2 * m.kerf
    pm = PolyMesh.from_trimesh(tm).scale_to_height(200)
    islands, _ = unfold(pm, trials=20, seed=0,
                        max_size=(m.usable_bed[0] - pad, m.usable_bed[1] - pad))
    parts = build_parts(islands, m, annotate=False)
    pack(parts, m)
    _clear_of_each_other(parts, m)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_pack_breaks_a_sheet_tie_on_board_used(name, tm):
    """When strategies tie on parts placed and whole sheets, pack() keeps the
    one using the least board. It used to keep whichever ran first - shelf
    packing - and on the helmet that was 1.59 sheets of board where the
    raster packer needed 1.26, with visible voids on sheet 1."""
    from cardboard_unfolder.layout import (shelf_pack, nest as nest_fn,
                                           raster_pack, compact_last,
                                           sheet_equivalents)
    m = Material.from_flute("B")
    m.bed = (400.0, 900.0)
    pad = 2 * m.tab_depth + 2 * m.kerf
    pm = PolyMesh.from_trimesh(tm).scale_to_height(250)
    islands, _ = unfold(pm, trials=20, seed=0,
                        max_size=(m.usable_bed[0] - pad, m.usable_bed[1] - pad))
    each = []
    for fn in (shelf_pack, nest_fn, raster_pack):
        ps = build_parts(islands, m, annotate=False)
        n, _ = fn(ps, m)
        n = tighten(ps, n, m)
        compact_last(ps, m)
        each.append((sum(1 for p in ps if p.sheet < 0), n,
                     sheet_equivalents(ps, m)[0]))
    ps = build_parts(islands, m, annotate=False)
    n, _ = pack(ps, m)
    got = (sum(1 for p in ps if p.sheet < 0), n, sheet_equivalents(ps, m)[0])
    best = min(each)
    assert got[:2] <= best[:2]
    if got[:2] == best[:2]:
        assert got[2] <= best[2] + 1e-9, (got, each)


def test_pack_never_loses_to_any_strategy():
    """No packer dominates, so pack() runs all three and keeps the winner.

    Nesting is far better when parts can share a sheet and worse when its
    greedy placement paints itself into a corner - measured both ways on real
    nets. The raster packer beats both where it wins and ties where it does
    not. This is the property that makes adding any of them safe.
    """
    from cardboard_unfolder.layout import (shelf_pack, nest as nest_fn,
                                           raster_pack)
    m = Material.from_flute("B")
    pad = 2 * m.tab_depth + 2 * m.kerf
    for tm, frac in ((trimesh.creation.icosphere(subdivisions=1), 1.0),
                     (trimesh.creation.icosphere(subdivisions=1), 0.6),
                     (trimesh.creation.cylinder(radius=1, height=2, sections=16), 0.7)):
        pm = PolyMesh.from_trimesh(tm).scale_to_height(280)
        limit = ((m.usable_bed[0] - pad) * frac, (m.usable_bed[1] - pad) * frac)
        islands, _ = unfold(pm, trials=40, max_size=limit)

        a = build_parts(islands, m)
        b = build_parts(islands, m)
        c = build_parts(islands, m)
        d = build_parts(islands, m)
        s_shelf, _ = shelf_pack(a, m)
        s_nest, _ = nest_fn(b, m)
        s_rast, _ = raster_pack(c, m)
        s_pack, _ = pack(d, m)
        assert s_pack <= min(s_shelf, s_nest, s_rast), (
            f"pack used {s_pack} sheets, shelf {s_shelf}, nest {s_nest}, "
            f"raster {s_rast}")


def test_nested_parts_do_not_overlap():
    """Nesting places against real outlines, so this is the check that the
    gap is actually being honoured rather than parts merely not colliding."""
    from cardboard_unfolder.layout import nest as nest_fn
    m = Material.from_flute("B")
    pad = 2 * m.tab_depth + 2 * m.kerf
    pm = PolyMesh.from_trimesh(
        trimesh.creation.icosphere(subdivisions=1)).scale_to_height(280)
    limit = ((m.usable_bed[0] - pad) * 0.6, (m.usable_bed[1] - pad) * 0.6)
    islands, _ = unfold(pm, trials=40, max_size=limit)
    parts = build_parts(islands, m)
    sheets, _ = nest_fn(parts, m)
    for s in range(sheets):
        on = [p for p in parts if p.sheet == s]
        for i in range(len(on)):
            for j in range(i + 1, len(on)):
                assert on[i].cut.intersection(on[j].cut).area < 1e-6,                     f"sheet {s}: islands {on[i].island_id} and {on[j].island_id} overlap"
        for p in on:
            x0, y0, x1, y1 = p.bounds
            assert x0 >= m.sheet_margin - 1e-6 and y0 >= m.sheet_margin - 1e-6
            assert x1 <= m.bed[0] - m.sheet_margin + 1e-6
            assert y1 <= m.bed[1] - m.sheet_margin + 1e-6


def test_part_xform_reproduces_the_packing():
    """The tracked transform must BE the packing, not an approximation of it.

    Everything downstream trusts it to place the same island somewhere else -
    the net UV layer uses it to show the real sheet layout - so if it drifts
    from the geometry the 2D view silently lies about what will be cut.
    """
    import shapely
    m = Material.from_flute("B")
    tm = trimesh.creation.icosphere(subdivisions=1)
    tm.apply_scale(240 / tm.extents.max())
    pm = PolyMesh.from_trimesh(tm)
    pad = 2 * m.tab_depth + 2 * m.kerf
    islands, _ = unfold(pm, trials=60,
                        max_size=(m.usable_bed[0] - pad, m.usable_bed[1] - pad))
    parts = build_parts(islands, m)

    before = {p.island_id: shapely.get_coordinates(p.cut) for p in parts}
    pack(parts, m)
    for p in parts:
        moved = shapely.get_coordinates(p.cut)
        pred = p.place(before[p.island_id])
        assert moved.shape == pred.shape
        assert np.abs(moved - pred).max() < 1e-6, (
            f"island {p.island_id}: transform and geometry disagree by "
            f"{np.abs(moved - pred).max():.3e} mm")


def test_oversized_parts_are_parked_not_left_at_random_coordinates():
    """A part too big for the bed is never placed, so without help it keeps
    whatever coordinates the unfold gave it - routinely negative. The net is
    written from these positions, so one oversized part drags the 2D view into
    negative space and every margin check on the *view* fails for a layout the
    packer never claimed to have made."""
    m = Material.from_flute("B")
    m.bed = (120.0, 120.0)                 # far too small on purpose
    tm = trimesh.creation.icosphere(subdivisions=1)
    pm = PolyMesh.from_trimesh(tm).scale_to_height(280)
    islands, _ = unfold(pm, trials=30)
    parts = build_parts(islands, m)
    _sheets, over = pack(parts, m)
    assert over, "this bed is too small for anything - expected oversized parts"
    assert over[0].bounds[0] == pytest.approx(m.sheet_margin, abs=1e-6)
    for p in over:
        x0, y0, _x1, _y1 = p.bounds
        assert x0 >= m.sheet_margin - 1e-6, f"island {p.island_id} parked at x={x0:.1f}"
        assert y0 == pytest.approx(m.sheet_margin, abs=1e-6)
        assert p.sheet < 0, "and it must still be flagged as not placed"


def test_oversized_parts_are_parked_side_by_side():
    """Not all on the one corner: piled there, parts that do not fit also
    read as parts that overlap - 35,000 mm2 of apparent overlap on a helmet
    whose only real problem was a packer that would not stand parts up."""
    m = Material.from_flute("B")
    parts = [_blank(box(0, 0, 700 + 10 * k, 450), -1) for k in range(3)]
    _sheets, over = pack(parts, m)
    assert len(over) == 3
    for q in over:
        assert q.bounds[1] == pytest.approx(m.sheet_margin, abs=1e-6)
    for i, a in enumerate(over):
        for b in over[i + 1:]:
            assert a.cut.distance(b.cut) >= m.part_gap - 1e-6


def _slotted_part(width=2.0, seated=None):
    """A panel with eleven round-ended relief slots `width` wide, rotated the
    way packing leaves it. At twice the 1 mm part gap, growing it by the gap
    collapses every slot to zero width and GEOS returns an invalid buffer.
    Whether the overlay after it then throws depends on the exact
    coordinates, so both variants here are ones that threw on the old code:
    as rotated, and rotated then seated on a `seated` mm margin."""
    from shapely import affinity
    from cardboard_unfolder.folds import _strip
    c, d, n = np.array([100.0, 100.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    slots = unary_union([_strip(c - 30 * d + k * 9 * n, c + 30 * d + k * 9 * n,
                                width, round_ends=True) for k in range(-5, 6)])
    part = set_precision(set_precision(box(0, 0, 200, 200), 1e-9).difference(
        set_precision(slots, 1e-9)), 1e-9)
    if seated is None:
        return affinity.rotate(part, 17.3, origin=(123.4, 56.7))
    part = affinity.rotate(part, 9.2, origin="centroid")
    x0, y0, _, _ = part.bounds
    return affinity.translate(part, seated - x0, seated - y0)


def test_growing_a_part_by_the_gap_survives_slots_twice_the_gap_wide():
    """A 2 mm cut width against a 1 mm gap: the helmet's Unfold and Optimise
    both died in free_region with a GEOS TopologyException. The fix must not
    change what counts as occupied - only stop the buffer coming back broken."""
    from cardboard_unfolder.layout import grow
    part = _slotted_part(2.0)
    assert part.is_valid
    raw = part.buffer(1.0)
    g = grow(part, 1.0)
    assert g.is_valid and not g.is_empty
    assert abs(g.area - raw.area) < 1e-3, "the repair must not change what is occupied"
    pieces = free_region([part], 392.0, 892.0, 1.0, 4.0)
    assert pieces and all(q.is_valid for q in pieces)
    assert not any(q.intersects(part) for q in pieces)


@pytest.mark.parametrize("seated", [None, 4.0], ids=["rotated", "seated"])
def test_the_free_region_survives_slots_twice_the_gap_wide(seated):
    """Both of the user's messages, from the one call that raised them:
    tighten asks free_region for the room left on a sheet. Rotated, the old
    code threw "found non-noded intersection" (Unfold); rotated and seated on
    the margin, "side location conflict" (Optimise)."""
    part = _slotted_part(2.0, seated=seated)
    pieces = free_region([part], 392.0, 892.0, 1.0, 4.0)
    assert pieces and all(q.is_valid for q in pieces)
    assert not any(q.intersection(part).area > 1e-6 for q in pieces)
    free = sum(q.area for q in pieces)
    sheet = box(4.0, 4.0, 396.0, 896.0)
    assert free == pytest.approx(sheet.difference(part.buffer(1.0).buffer(0)).area,
                                 abs=1e-3)


def _long_part():
    """615 x 200 mm: fits a 400 x 900 bed only standing up."""
    return _blank(box(0, 0, 615, 200), -1)


@pytest.mark.parametrize("strategy", ["shelf_pack", "nest", "raster_pack"])
def test_every_packer_stands_a_part_up_on_a_portrait_bed(strategy):
    """orient_min_bbox lays every part long side on x, which is right for a
    landscape bed. shelf_pack never tried the other way, so on a 400 x 900 bed
    a 615 x 292 helmet part was called oversized."""
    from cardboard_unfolder import layout
    m = Material.from_flute("B")
    m.bed = (400.0, 900.0)
    part = _long_part()
    sheets, over = getattr(layout, strategy)([part], m)
    assert not over and sheets == 1 and part.sheet == 0
    x0, y0, x1, y1 = part.bounds
    W, H = m.bed
    assert x0 >= m.sheet_margin - 1e-6 and x1 <= W - m.sheet_margin + 1e-6
    assert y0 >= m.sheet_margin - 1e-6 and y1 <= H - m.sheet_margin + 1e-6


def test_pack_never_drops_a_part_to_save_a_sheet(monkeypatch):
    """Sheets count PLACED parts only, so a strategy that gives up on a part
    reports fewer of them. pack() ranked by sheets first and took that: four
    helmet parts left off, "1 sheet", beating three sheets with everything
    on. A part that is not placed is not a saving - it cannot be cut."""
    from cardboard_unfolder import layout

    def gives_up(parts, mat):
        for q in parts:
            q.sheet = -1
        return 0, list(parts)

    monkeypatch.setattr(layout, "shelf_pack", gives_up)
    m = Material.from_flute("B")
    m.bed = (400.0, 900.0)
    parts = [_long_part(), _blank(box(0, 0, 50, 50), -1)]
    sheets, over = pack(parts, m)
    assert not over, "a strategy that placed everything was available"
    assert sheets >= 1 and all(q.sheet >= 0 for q in parts)


def test_packed_parts_sit_inside_the_margins():
    """Nothing may stray into the sheet margin or across a sheet boundary -
    that is clamped material the laser cannot reach."""
    m = Material.from_flute("B")
    tm = trimesh.creation.icosphere(subdivisions=1)
    tm.apply_scale(240 / tm.extents.max())
    pm = PolyMesh.from_trimesh(tm)
    pad = 2 * m.tab_depth + 2 * m.kerf
    islands, _ = unfold(pm, trials=60,
                        max_size=(m.usable_bed[0] - pad, m.usable_bed[1] - pad))
    parts = build_parts(islands, m)
    sheets, oversized = pack(parts, m)
    for p in parts:
        if p.sheet < 0:
            continue
        x0, y0, x1, y1 = p.bounds
        assert x0 >= m.sheet_margin - 1e-6, f"island {p.island_id} crosses the left margin"
        assert y0 >= m.sheet_margin - 1e-6, f"island {p.island_id} crosses the bottom margin"
        assert x1 <= m.bed[0] - m.sheet_margin + 1e-6, "crosses the right margin"
        assert y1 <= m.bed[1] - m.sheet_margin + 1e-6, "crosses the top margin"


def test_packed_parts_do_not_collide():
    m = Material.from_flute("B")
    tm = trimesh.creation.icosphere(subdivisions=1)
    tm.apply_scale(240 / tm.extents.max())
    pm = PolyMesh.from_trimesh(tm)
    pad = 2 * m.tab_depth + 2 * m.kerf
    islands, _ = unfold(pm, trials=80,
                        max_size=(m.usable_bed[0] - pad, m.usable_bed[1] - pad))
    parts = build_parts(islands, m)
    sheets, _ = pack(parts, m)
    for s in range(sheets):
        on = [p for p in parts if p.sheet == s]
        for i in range(len(on)):
            for j in range(i + 1, len(on)):
                assert on[i].cut.intersection(on[j].cut).area < 1e-6


def test_narrow_neck_detection():
    from shapely.geometry import Polygon
    dumbbell = Polygon([(0, 0), (40, 0), (40, 19.6), (60, 19.6), (60, 0),
                        (100, 0), (100, 40), (60, 40), (60, 20.4),
                        (40, 20.4), (40, 40), (0, 40)])
    assert narrow_necks(dumbbell, 5.0) >= 1
    assert narrow_necks(Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]), 5.0) == 0


def test_writes_valid_svg_and_dxf(tmp_path):
    import xml.etree.ElementTree as ET
    import ezdxf
    from cardboard_unfolder import run

    tm = trimesh.creation.box()
    p = tmp_path / "cube.stl"
    tm.export(p)
    rep = run(p, tmp_path / "out", Material.from_flute("B"),
              height_mm=110, trials=40, verbose=False)

    svgs = sorted((tmp_path / "out").glob("*.svg"))
    assert len(svgs) == rep["sheets"] >= 1
    root = ET.parse(svgs[0]).getroot()
    assert root.get("width") == "600.0mm"
    assert root.get("viewBox") == "0 0 600.0 400.0"
    labels = {g.get("{http://www.inkscape.org/namespaces/inkscape}label")
              for g in root.iter("{http://www.w3.org/2000/svg}g")}
    assert {"CUT", "ENGRAVE"} <= labels
    assert not {"PERF_M", "PERF_V"} & labels, "fold scores are off by default"

    doc = ezdxf.readfile(sorted((tmp_path / "out").glob("*.dxf"))[0])
    used = {e.dxf.layer for e in doc.modelspace()}
    assert "CUT" in used
    assert (tmp_path / "out" / "manifest.json").exists()


def test_svg_layer_order_puts_cuts_last(tmp_path):
    """Perimeters must cut last or the part drops before its features are done."""
    import xml.etree.ElementTree as ET
    from cardboard_unfolder import run
    tm = trimesh.creation.box()
    p = tmp_path / "c.stl"
    tm.export(p)
    run(p, tmp_path / "o", Material.from_flute("B"), height_mm=100,
        trials=20, dxf=False, verbose=False)
    root = ET.parse(sorted((tmp_path / "o").glob("*.svg"))[0]).getroot()
    order = [g.get("{http://www.inkscape.org/namespaces/inkscape}label")
             for g in root.iter("{http://www.w3.org/2000/svg}g")]
    assert order[-1] == "CUT"
    assert order.index("ENGRAVE") < order.index("CUT")


# ---------------------------------------------------------------- fingers

_FINGERED = {}


def _fingered(tm, **kw):
    """A net with fingers on, built once per mesh: several invariants read
    the same one, and unfolding it again for each is most of their time."""
    key = (id(tm), tuple(sorted(kw.items())))
    if key not in _FINGERED:
        islands, _ = unfold(PolyMesh.from_trimesh(tm), trials=40, seed=0)
        mat = Material.from_flute("B", use_fingers=True, **kw)
        bodies = {i.id: apply_relief(i.outline(), plan_folds(i, mat))
                  for i in islands}
        _FINGERED[key] = (islands, mat, place_fingers(islands, mat, bodies))
    return _FINGERED[key]


_FPARTS = {}


def _fingered_parts(tm):
    """(parts with fingers, the same net's parts without)."""
    if id(tm) not in _FPARTS:
        islands, mat, _ = _fingered(tm)
        _FPARTS[id(tm)] = (build_parts(islands, mat),
                           build_parts(islands, Material.from_flute("B")))
    return _FPARTS[id(tm)]


def _in_cut_frame(cut, isl, t0, t1, s0, s1):
    """A rectangle given in the cut's own frame - t along a->b, s into the
    face - placed in the island's coordinates. Built there rather than moving
    the part: a part with its relief slots is tens of thousands of vertices."""
    u = (cut.b - cut.a) / cut.length
    v = inward(isl, cut)
    a = cut.a
    return Polygon([a + u * t0 + v * s0, a + u * t1 + v * s0,
                    a + u * t1 + v * s1, a + u * t0 + v * s1])


def _pieces(g):
    return len(getattr(g, "geoms", [g]))


def test_finger_defaults():
    """15 mm wide, as deep as the board, and off until asked for - like glue
    tabs, it changes every part outline, so it is a choice, not a default."""
    m = Material.from_flute("B")
    assert m.use_fingers is False
    assert m.finger_width == 15.0
    assert m.finger_clearance == 0.05, "the user's number after a first build"
    assert m.finger_depth_used == m.thickness == 3.0
    assert Material.from_flute("C").finger_depth_used == 4.0
    m.finger_depth = 5.5
    assert m.finger_depth_used == 3.0, "the value is ignored until overridden"
    m.finger_depth_override = True
    assert m.finger_depth_used == 5.5


def test_fingers_off_change_nothing():
    """The switch is a real switch: off is the part outline exactly."""
    islands, _ = unfold(PolyMesh.from_trimesh(ALL[0][1]), trials=40, seed=0)
    plain = build_parts(islands, Material.from_flute("B"))
    off = build_parts(islands, Material.from_flute("B", use_fingers=False,
                                                   finger_width=9.0))
    assert [p.cut.wkb for p in plain] == [p.cut.wkb for p in off]


def test_finger_segments_are_about_the_width_and_never_alone():
    mat = Material.from_flute("B")
    w, d = mat.finger_width, mat.finger_depth_used / 2.0
    for L in np.arange(5.0, 200.0, 0.7):
        segs = segments(L, mat)
        if L - 2 * d < 2 * MIN_FRAC * w:
            assert segs == [], L
            continue
        assert len(segs) >= 2, "one finger alone does not interlock"
        assert segs[0][0] == pytest.approx(d) and segs[-1][1] == pytest.approx(L - d)
        for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
            assert a1 == pytest.approx(b0)
        for t0, t1 in segs:
            assert MIN_FRAC * w - 1e-9 <= t1 - t0 <= 1.25 * w + 1e-9


def test_a_corner_inset_keeps_the_wall_to_the_next_edge():
    """At the inset, the notch's inner corner is exactly `wall` from the
    face's other edge - checked against the geometry, not the formula."""
    d, wall = 3.0, 3.0
    for deg in (20, 45, 60, 90, 120, 150):
        a = math.radians(deg)
        t = corner_inset(a, d, wall)
        other = LineString([(0, 0), (100 * math.cos(a), 100 * math.sin(a))])
        gap = Point(t, d).distance(other)
        assert t >= d
        assert gap >= wall - 1e-9, (deg, gap)
    assert corner_inset(math.radians(200), d, wall) == d, "reflex: no limit"


def test_inward_is_read_from_the_ring_not_the_centroid():
    """A C-shaped face whose centroid sits on the far side of its inner edge.
    A centroid test points that notch out of the part."""
    ring = np.array([(0, 0), (20, 0), (20, 2), (2, 2), (2, 18), (20, 18),
                     (20, 20), (0, 20)], dtype=float)
    assert Polygon(ring).centroid.x > 2.0, "the fixture must defeat a centroid test"
    for r in (ring, ring[::-1]):
        isl = Island(id=0, faces={0: r})
        cut = Cut(a=np.array([2.0, 2.0]), b=np.array([2.0, 18.0]), face=0, seam=0)
        assert np.allclose(inward(isl, cut), (-1.0, 0.0))


def _lopsided_seams(J):
    """Seams jointed on one half only: every notch on the same side, so the
    other half has no notch - and no finger worth the name - at all.

    Halves are told apart by (island, face), not island alone, because both
    halves of a slit inside one island belong to the same island."""
    halves = {}
    for js in J.values():
        for j in js:
            if j.kind == "notch":
                halves.setdefault(j.seam, set()).add((j.island, j.face))
    seams = {j.seam for js in J.values() for j in js}
    return sorted(s for s in seams if len(halves.get(s, ())) < 2)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_fingered_seam_is_notched_on_both_halves(name, tm):
    """A box joint alternates: each half carries notches for the other's
    fingers. Segments are refused one at a time - a notch that would run into
    a relief slot, say - and refusals could leave a seam with every notch on
    ONE half: a lone notch, with nothing on the other side to show it is a
    joint at all. On the helmet that was 6 of 44 seams, at the inner ends of
    folds where the slots crowd the seam. Such a seam is left plain on both
    halves, the same as one too short to joint."""
    assert _lopsided_seams(_fingered(tm)[2]) == []


def test_the_helmet_has_no_lopsided_finger_joint_at_its_own_settings():
    """The settings the lopsided joints were found at: 320 mm tall, fixed 2 mm
    relief slots, a 400 x 900 bed - every fold a slot, which is what crowds
    the seam ends."""
    tm = dict(ALL)["helmet"]
    mat = Material.from_flute("B", use_fingers=True, relief_adaptive=False,
                              relief_fixed=2.0)
    mat.bed = (400.0, 900.0)
    pad = 2 * mat.tab_depth + 2 * mat.kerf
    pm = PolyMesh.from_trimesh(tm).scale_to_height(320)
    islands, _ = unfold(pm, trials=250, seed=0,
                        max_size=(mat.usable_bed[0] - pad,
                                  mat.usable_bed[1] - pad))
    bodies = {i.id: apply_relief(i.outline(), plan_folds(i, mat))
              for i in islands}
    J = place_fingers(islands, mat, bodies)
    assert any(J.values()), "nothing jointed - the test would prove nothing"
    assert _lopsided_seams(J) == []


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_both_halves_of_a_fingered_seam_mate_exactly(name, tm):
    """Bring both halves onto their shared edge. The mesh is the board's
    mid-plane, so on every jointed segment the finger half is solid board
    from half a depth OUT past the edge to half a depth in, and the notched
    half is empty for that half depth in - the finger fills the notch."""
    islands, mat, J = _fingered(tm)
    d = mat.finger_depth_used / 2.0
    play = mat.finger_clearance
    parts = {p.island_id: p for p in _fingered_parts(tm)[0]}
    by = {i.id: i for i in islands}
    cuts = {}
    for isl in islands:
        for c in isl.cuts:
            if c.seam >= 0:
                cuts[(isl.id, c.seam, c.face)] = c
    n = 0
    for iid, js in J.items():
        for j in js:
            if j.kind != "finger":
                continue
            mine = cuts[(iid, j.seam, j.face)]
            other_key = next(k for k in cuts if k[1] == j.seam
                             and (k[0], k[2]) != (iid, j.face))
            theirs = cuts[other_key]
            t0, t1 = j.t
            e = 1e-3
            slot = _in_cut_frame(mine, by[iid], t0 + play + e, t1 - play - e,
                                 -d + e, d - e)
            A = parts[iid].cut.intersection(slot)
            assert A.area == pytest.approx(slot.area, rel=1e-6), (j.seam, j.t)
            # ...and stands `c` clear of the notch's walls on both sides:
            # the clearance that lets the joint go together by hand.
            for lo, hi in ((t0 + e, t0 + play - e), (t1 - play + e, t1 - e)):
                gap = _in_cut_frame(mine, by[iid], lo, hi, -d + e, -e)
                assert parts[iid].cut.intersection(gap).area < 1e-9, \
                    (j.seam, j.t, "no clearance beside the finger")
            # Only the notched half's OWN side of the edge: across it, a slit
            # inside one island can have the other half of the V lying in the
            # flat sheet - separate board, somewhere else once folded.
            hole = _in_cut_frame(theirs, by[other_key[0]], t0 + e, t1 - e,
                                 e, d - e)
            B = parts[other_key[0]].cut.intersection(hole)
            assert B.area < 1e-6, (j.seam, j.t)
            n += 1
    assert n > 0, "nothing was jointed, so nothing was tested"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_finger_reaches_half_a_depth_past_the_edge_and_no_further(name, tm):
    """The mesh is the board's MID-plane, so a finger stands half the depth
    out past the mesh edge - flush with the other board's outer face at a
    right angle - and a notch goes half the depth in. Not a whole depth: the
    first version stood a board's thickness proud of the model. What the
    part gains is exactly its fingers and what it loses exactly its notches:
    the edge swings by one depth in all."""
    islands, mat, J = _fingered(tm)
    h = mat.finger_depth_used / 2.0
    on, off = _fingered_parts(tm)
    for p, q in zip(on, off):
        assert q.cut.buffer(h + 1e-6).contains(p.cut), \
            f"part {p.island_id} reaches past half a depth"
    added = sum(p.cut.difference(q.cut).area for p, q in zip(on, off))
    removed = sum(q.cut.difference(p.cut).area for p, q in zip(on, off))
    fingers = sum(j.poly.area for js in J.values() for j in js
                  if j.kind == "finger")
    notched = sum(j.poly.area for js in J.values() for j in js
                  if j.kind == "notch")
    slack = sum(2 * mat.finger_clearance * h for js in J.values() for j in js
                if j.kind == "finger")
    assert notched > 0 and fingers == pytest.approx(notched - slack, rel=1e-9)
    assert added == pytest.approx(fingers, rel=1e-5)
    assert removed == pytest.approx(notched, rel=1e-5)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_finger_stays_clear_of_every_other_board_in_the_flat(name, tm):
    """A finger stands out into the sheet, and in a narrow V - a slit inside
    one island - the far half of the V is right there near the apex. The
    joint must END where a finger would come within a part gap of other
    board, and the seam run plain to the point from there; the laser would
    otherwise fuse the two. Measured against the island before fingers."""
    islands, mat, J = _fingered(tm)
    clear = max(mat.part_gap, mat.kerf)
    h = mat.finger_depth_used / 2.0
    by = {i.id: i for i in islands}
    _, off = _fingered_parts(tm)
    bare = {q.island_id: q.cut for q in off}
    n = 0
    for iid, js in J.items():
        cuts = {(c.seam, c.face): c for c in by[iid].cuts}
        for j in js:
            if j.kind != "finger":
                continue
            c = cuts[(j.seam, j.face)]
            t0, t1 = j.t
            reach = _in_cut_frame(c, by[iid], t0 - clear + 1e-6,
                                  t1 + clear - 1e-6, -h - clear + 1e-6, -1e-5)
            assert not bare[iid].intersects(reach), (j.seam, j.t)
            n += 1
    assert n > 0


def test_corner_holes_are_off_by_default_and_change_nothing_then():
    m = Material.from_flute("B", use_fingers=True)
    assert m.corner_hole is False
    tm = dict(ALL)["helmet"]
    islands, _ = unfold(PolyMesh.from_trimesh(tm), trials=40, seed=0)
    a = build_parts(islands, m)
    b = build_parts(islands, Material.from_flute("B", use_fingers=True,
                                                 corner_hole=False))
    assert [p.cut.wkb for p in a] == [p.cut.wkb for p in b]


def test_a_corner_hole_is_centred_on_every_sharp_seam_vertex():
    """The helmet's nets have slits whose sides meet at a few degrees - the
    corners where the two boards crush into each other as the slit closes.
    With the switch on, each such vertex the part can spare gets a round hole
    one board across CENTRED on it (the user's correction to a first version
    that set a teardrop just past it), and no part is ever split by one."""
    from cardboard_unfolder.folds import sharp_seam_corners
    tm = dict(ALL)["helmet"]
    islands, _ = unfold(PolyMesh.from_trimesh(tm), trials=40, seed=0)
    m0 = Material.from_flute("B")
    m1 = Material.from_flute("B", corner_hole=True)
    off, on = build_parts(islands, m0), build_parts(islands, m1)
    r = m1.thickness / 2.0
    corners = cut = 0
    for isl, p, q in zip(islands, off, on):
        assert _pieces(q.cut) == _pieces(p.cut)
        # the body the parts are built from: relief slots can sit at a corner
        body = apply_relief(isl.outline(), plan_folds(isl, m1))
        found = sharp_seam_corners(isl, body)
        corners += len(found)
        for pt, _into in found:
            disc = Point(pt).buffer(0.98 * r)
            if not q.cut.intersects(disc):
                cut += 1                        # the whole disc is gone
                continue
            # dropped only because it would have split the part
            trial = p.cut.difference(Point(pt).buffer(r))
            assert _pieces(trial) > _pieces(p.cut), pt
    assert corners > 0 and cut > 0, "the fixture has no sharp corner to relieve"


def test_a_corner_hole_joins_the_notch_bottom_with_no_spike():
    """The user's screenshot: a notch starting at the hole's EDGE leaves a
    spike of board between its end wall and the circle, pointing at the
    vertex. The notch must run from the vertex, its bottom - half a depth in
    - tangent to a hole of that same radius, so the band between the hole
    and the notch holds no board at all."""
    from cardboard_unfolder.folds import apply_corner_holes
    tm = dict(ALL)["helmet"]
    islands, _ = unfold(PolyMesh.from_trimesh(tm), trials=40, seed=0)
    m = Material.from_flute("B", use_fingers=True, corner_hole=True)
    h = m.finger_depth_used / 2.0
    holes = {}
    for i in islands:
        _b, holes[i.id] = apply_corner_holes(
            i, apply_relief(i.outline(), plan_folds(i, m)), 2 * h)
    parts = {p.island_id: p for p in build_parts(islands, m)}
    by = {i.id: i for i in islands}
    bodies = {}
    for i in islands:
        bodies[i.id], _ = apply_corner_holes(
            i, apply_relief(i.outline(), plan_folds(i, m)), 2 * h)
    J = place_fingers(islands, m, bodies, holes=holes, hole_r=h)
    checked = 0
    for iid, js in J.items():
        cuts = {(c.seam, c.face): c for c in by[iid].cuts}
        for j in js:
            if j.kind != "notch":
                continue
            c = cuts[(j.seam, j.face)]
            for t_end, p in ((0.0, c.a), (c.length, c.b)):
                if not any(np.hypot(*(np.asarray(p) - q)) < 1e-6
                           for q in holes[iid]):
                    continue
                # only a notch that reaches the hole: the corner it shares
                # with the hole, two radii along and half a depth in, must
                # hold no board - that is where the spike was
                t0, t1 = j.t
                if t_end == 0.0 and t0 >= 2 * h:
                    continue
                if t_end > 0.0 and t1 <= c.length - 2 * h:
                    continue
                lo, hi = ((0.0, 2 * h) if t_end == 0.0
                          else (c.length - 2 * h, c.length))
                band = _in_cut_frame(c, by[iid], lo, hi, 1e-3, h - 1e-3)
                band = band.intersection(Polygon(by[iid].faces[c.face]))
                left = parts[iid].cut.intersection(band).area
                assert left < 1e-4, (j.seam, j.t, left)
                checked += 1
    assert checked > 0, "no notch runs to a hole - the test proved nothing"


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_with_corner_holes_no_two_fingers_overlap(name, tm):
    """With holes on, a slit's joint runs back to the hole at its apex, where
    the two halves' fingers point at each other across a gap that closes to
    nothing - each is clipped at the V's centreline. Two fingers of one
    island never share board, and no part comes apart."""
    islands, _ = unfold(PolyMesh.from_trimesh(tm), trials=40, seed=0)
    m = Material.from_flute("B", use_fingers=True, corner_hole=True)
    from cardboard_unfolder.folds import apply_corner_holes
    bodies, holes = {}, {}
    for i in islands:
        bodies[i.id], holes[i.id] = apply_corner_holes(
            i, apply_relief(i.outline(), plan_folds(i, m)), m.thickness)
    J = place_fingers(islands, m, bodies, holes=holes, hole_r=m.thickness / 2)
    for js in J.values():
        fs = [j.poly for j in js if j.kind == "finger"]
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                assert fs[a].intersection(fs[b]).area < 1e-9
    parts = build_parts(islands, m)
    bare = build_parts(islands, Material.from_flute("B", corner_hole=True))
    assert [_pieces(p.cut) for p in parts] == [_pieces(p.cut) for p in bare]


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_finger_is_not_refused_over_float_noise_on_its_own_edge(name, tm):
    """The finger's rectangle lies ON the body's edge, so a strict `contains`
    against the unbuffered body failed on 1e-12 mm of noise and called the
    board missing - 240 of icosphere2's fingers refused, and a third of the
    cone's. On these plain nets there is always board behind a finger."""
    islands, _ = unfold(PolyMesh.from_trimesh(tm), trials=40, seed=0)
    mat = Material.from_flute("B", use_fingers=True)
    bodies = {i.id: apply_relief(i.outline(), plan_folds(i, mat)) for i in islands}
    why = {}
    place_fingers(islands, mat, bodies, why)
    assert why.get("finger has no board behind it", 0) == 0, why


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_fingers_never_break_a_part(name, tm):
    """A notch whose outer side met the edge a nanometre short used to leave a
    hairline of board, which the laser traces as a separate piece - so count
    the pieces. And every notch has exactly one finger opposite it."""
    islands, mat, J = _fingered(tm)
    on, off = _fingered_parts(tm)
    assert [_pieces(p.cut) for p in on] == [_pieces(p.cut) for p in off]
    kinds = [(j.seam, j.t) for js in J.values() for j in js]
    fingers = sorted((j.seam, j.t) for js in J.values() for j in js
                     if j.kind == "finger")
    notches = sorted((j.seam, j.t) for js in J.values() for j in js
                     if j.kind == "notch")
    assert fingers == notches and len(kinds) == 2 * len(fingers)


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_a_notch_keeps_a_wall_of_its_own_face(name, tm):
    """Inside its own face, and a board's thickness from every other edge of
    it - a fold, another seam, a relief slot - and from every other notch."""
    islands, mat, J = _fingered(tm)
    wall = mat.thickness
    by = {i.id: i for i in islands}
    for iid, js in J.items():
        notches = [j for j in js if j.kind == "notch"]
        for j in notches:
            ring = by[iid].faces[j.face]
            face = Polygon(ring)
            assert face.buffer(1e-6).contains(j.poly)
            own = [c for c in by[iid].cuts if c.seam == j.seam and c.face == j.face][0]
            n = len(ring)
            for k in range(n):
                p0, p1 = ring[k], ring[(k + 1) % n]
                on_cut = all(min(np.linalg.norm(q - own.a),
                                 np.linalg.norm(q - own.b)) < 1e-9
                             for q in (p0, p1))
                if on_cut:
                    continue
                assert LineString([p0, p1]).distance(j.poly) >= wall - 1e-6
            for o in notches:
                if o.seam != j.seam:
                    assert o.poly.distance(j.poly) >= wall - 1e-6
            # and never into a finger kept for another seam
            for f in js:
                if f.kind == "finger" and f.seam != j.seam:
                    assert f.poly.intersection(j.poly).area < 1e-6


def test_a_fingered_seam_gets_no_glue_tab():
    """Both on: a seam that interlocks needs no flap, and a flap along a
    fingered edge would bury the fingers - so tabs go only where fingers
    could not."""
    tm = [t for n, t in ALL if n == "capsule"][0]
    islands, mat, J = _fingered(tm, use_tabs=True)
    fingered = {j.seam for js in J.values() for j in js}
    from cardboard_unfolder.fingers import fingered_seams
    tabs, _ = place_tabs(islands, mat, skip=fingered_seams(J))
    tabbed = {t.seam for ts in tabs.values() for t in ts}
    assert fingered and tabbed, "the fixture must exercise both"
    assert not fingered & tabbed
    for iid, ts in tabs.items():
        for t in ts:
            for j in J.get(iid, []):
                if j.kind == "finger":
                    assert t.poly.intersection(j.poly).area < 1e-6


def test_a_deeper_override_makes_deeper_fingers():
    tm = ALL[0][1]
    for kw, want in (({}, 3.0),
                     (dict(finger_depth_override=True, finger_depth=5.0), 5.0)):
        _, mat, J = _fingered(tm, **kw)
        js = [j for v in J.values() for j in v]
        assert js
        for j in js:
            t0, t1 = j.t
            w = t1 - t0 - (2 * mat.finger_clearance if j.kind == "finger"
                           else 0.0)
            assert j.poly.area == pytest.approx(w * want / 2.0, rel=1e-6)


def test_finger_clearance_zero_is_a_snug_fit_and_it_only_narrows_fingers():
    """0 is the old joint exactly: the finger as wide as its notch. Any
    clearance comes off the finger alone - the notch, and every check it
    passed, stay as they were, so the same segments are jointed."""
    tm = ALL[0][1]
    _, m0, J0 = _fingered(tm, finger_clearance=0.0)
    _, m1, J1 = _fingered(tm, finger_clearance=0.3)
    key = lambda J: sorted((j.seam, j.island, j.kind, j.t)
                           for v in J.values() for j in v)
    assert key(J0) == key(J1)
    for j in (j for v in J0.values() for j in v):
        t0, t1 = j.t
        assert j.poly.area == pytest.approx((t1 - t0) * m0.finger_depth_used
                                            / 2.0, rel=1e-6)
    n0 = {(j.seam, j.t): j.poly.wkb for v in J0.values() for j in v
          if j.kind == "notch"}
    n1 = {(j.seam, j.t): j.poly.wkb for v in J1.values() for j in v
          if j.kind == "notch"}
    assert n0 == n1


def test_text_defaults_to_three_mm():
    """3 mm, the user's call after the first assembled model; the 2 mm floor
    is unchanged."""
    m = Material.from_flute("B")
    assert m.label_height == 3.0 and m.label_min_height == 2.0


def test_a_fixed_relief_width_is_cut_however_thin():
    """A fixed width is what the machine is asked to cut, so it is cut at
    that width - the 1.2 mm perforate-instead threshold belongs to the
    ADAPTIVE width. It used to apply to both, so a fixed 1 mm slot quietly
    became a perforation: with fold scoring off, nothing at all."""
    from cardboard_unfolder.unfold import Fold
    a, b = np.array([0.0, 0.0]), np.array([60.0, 0.0])
    isl = Island(id=0, faces={0: np.array([(0, 0), (60, 0), (30, -40)], float),
                              1: np.array([(0, 0), (30, 40), (60, 0)], float)})
    isl.folds = [Fold(a=a, b=b, f0=0, f1=1, bend=math.radians(90))]
    for w in (1.0, 0.5, 0.1, 0.02):
        # square ends, so the area is exactly width x cut length
        m = Material.from_flute("B", relief_adaptive=False, relief_fixed=w,
                                round_cut_ends=False)
        (p,) = plan_folds(isl, m)
        assert p.style == "relief", w
        cut = sum(e - s for s, e in fold_intervals(60.0, m))
        assert p.slot.area == pytest.approx(w * cut, rel=1e-6), w
    m = Material.from_flute("B", relief_adaptive=False, relief_fixed=0.0)
    assert plan_folds(isl, m)[0].style == "perf", "0 means no slot"
    # adaptive keeps its threshold: a shallow fold is only perforated
    m = Material.from_flute("B")
    isl.folds[0].bend = math.radians(15)
    assert plan_folds(isl, m)[0].style == "perf"


# ------------------------------------------------------------------- font

def test_stroke_font_covers_labels_and_scales():
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert strokefont.text_strokes(ch, 5.0), f"glyph {ch} is empty"
    s = strokefont.text_strokes("P12", 5.0, origin=(0, 0))
    pts = np.vstack(s)
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(5.0, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------------- flat pieces

def _flat_ring_mesh(tilt=0.7, offset=(5.0, 7.0, 11.0), r0=40.0, r1=80.0,
                    n=24):
    """A flat washer, triangulated - the shape the coplanar merge cannot
    rebuild, because one face with a hole is not a face. Tilted out of every
    axis plane so the plane fit has to do the work."""
    ring = Polygon([(r1 * math.cos(t), r1 * math.sin(t))
                    for t in np.linspace(0, 2 * math.pi, n, endpoint=False)],
                   [[(r0 * math.cos(t), r0 * math.sin(t))
                     for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]])
    tris = shapely.constrained_delaunay_triangles(ring)
    verts, idx, rings = [], {}, []
    for t in tris.geoms:
        r = []
        for x, y in list(t.exterior.coords)[:-1]:
            k = (round(x, 9), round(y, 9))
            if k not in idx:
                idx[k] = len(verts)
                verts.append((x, y, 0.0))
            r.append(idx[k])
        # constrained_delaunay winds either way; face +z, consistently
        if _signed2d(np.array(verts)[r]) < 0:
            r = r[::-1]
        rings.append(r)
    c, s = math.cos(tilt), math.sin(tilt)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array(verts) @ R.T + np.array(offset), rings, ring


def _signed2d(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def test_a_flat_piece_is_cut_as_the_shape_it_is():
    """One island, no folds, the hole kept, exact. Without flat pieces the
    same washer is a tree of triangles with every diagonal a fold - a 300 mm
    border was 22,303 "folds" and 6 m of perforation."""
    v, rings, shape = _flat_ring_mesh()
    pm = PolyMesh.from_rings(v, rings, merge=False)
    islands, rep = unfold(pm, flat_tol=3.0)
    assert rep["flat_pieces"] == 1 and len(islands) == 1
    isl = islands[0]
    assert not isl.folds
    assert sorted(isl.faces) == list(range(len(pm.faces)))
    out = isl.outline()
    assert abs(out.area - shape.area) < 1e-6 * shape.area
    assert len(out.interiors) == 1
    assert rep["isometry_error"] < 1e-9
    assert mirrored_faces(pm, isl.faces) == []
    assert all(c.seam == -1 for c in isl.cuts)
    part = build_parts(islands, Material.from_flute("B"))[0]
    assert part.perf_m.is_empty and part.perf_v.is_empty


def test_a_piece_within_the_tolerance_is_flat_and_beyond_it_is_not():
    v, rings, _ = _flat_ring_mesh(tilt=0.0, offset=(0, 0, 0))
    bowed = v.copy()
    bowed[:, 2] = 2.0 * (bowed[:, 0] / 80.0) ** 2        # up to 2 mm
    pm = PolyMesh.from_rings(bowed, rings, merge=False)
    from cardboard_unfolder.unfold import flat_pieces
    assert len(flat_pieces(pm, 3.0)) == 1
    assert flat_pieces(pm, 0.5) == []


def test_a_thin_closed_solid_is_not_a_flat_piece():
    """A 2 mm plate modelled as a box is within 3 mm of a plane, and its top,
    bottom and walls still cannot be one flat shape."""
    from cardboard_unfolder.unfold import flat_pieces
    pm = PolyMesh.from_trimesh(trimesh.creation.box(extents=(100, 60, 2)))
    assert flat_pieces(pm, 3.0) == []


@pytest.mark.parametrize("name,tm", ALL, ids=IDS)
def test_no_corpus_model_has_a_flat_piece(name, tm):
    """So flat_tol changes nothing on any of them."""
    from cardboard_unfolder.unfold import flat_pieces
    assert flat_pieces(PolyMesh.from_trimesh(tm), 3.0) == []


def test_flat_pieces_leave_the_rest_of_the_model_alone():
    """A flat washer beside a box, in ONE mesh: the box unfolds exactly as it
    does on its own, its faces keep their ids, and the washer is one more
    island after it."""
    box_tm = trimesh.creation.box(extents=(60, 60, 60))
    pm_box = PolyMesh.from_trimesh(box_tm)
    v, rings, _ = _flat_ring_mesh(offset=(200.0, 0.0, 0.0))
    nb = len(pm_box.vertices)
    pm = PolyMesh(np.vstack([pm_box.vertices, v]),
                  [list(r) for r in pm_box.faces] + [[i + nb for i in r]
                                                     for r in rings])
    alone, _ = unfold(pm_box, seed=3)
    both, rep = unfold(pm, seed=3, flat_tol=3.0)
    assert rep["flat_pieces"] == 1
    assert len(both) == len(alone) + 1
    for a, b in zip(alone, both):
        assert a.id == b.id and sorted(a.faces) == sorted(b.faces)
        for f in a.faces:
            assert np.array_equal(a.faces[f], b.faces[f])
        assert [(c.face, c.seam) for c in a.cuts] == [(c.face, c.seam)
                                                     for c in b.cuts]
    flat = both[-1]
    assert flat.id == len(alone)
    assert sorted(flat.faces) == list(range(len(pm_box.faces), len(pm.faces)))


def test_a_fold_that_does_not_bend_is_not_scored():
    """The annulus caps are flat rings the merge cannot rebuild, so their
    triangle diagonals are 0-degree "folds" - and each was perforated and
    marked M or V. Below `min_bend` a fold is left alone; at 0 the old
    behaviour comes back, which is what proves the rule is doing it."""
    tm = dict(ALL)["annulus"]
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, seed=0)
    flat = [f for i in islands for f in i.folds
            if abs(f.bend) < math.radians(1.0)]
    assert flat, "the fixture must have unbent folds"
    mat = Material.from_flute("B")
    for isl in islands:
        for plan in plan_folds(isl, mat):
            if abs(plan.fold.bend) < math.radians(mat.min_bend):
                assert plan.style == "flat" and not plan.perf
                assert plan.slot is None
            else:
                assert plan.style != "flat"
    old = Material.from_flute("B", min_bend=0.0)
    scored = sum(len(p.perf) for isl in islands for p in plan_folds(isl, old)
                 if abs(p.fold.bend) < math.radians(1.0))
    assert scored > 0, "at min_bend 0 the unbent folds are scored again"


def test_an_unbent_fold_gets_no_fold_mark():
    """Only the unbent folds' plans: no M or V - and the same folds planned
    at min_bend 0 do get marks, so the room was there."""
    tm = dict(ALL)["annulus"]
    pm = PolyMesh.from_trimesh(tm)
    islands, _ = unfold(pm, seed=0)
    mat = Material.from_flute("B")
    old = Material.from_flute("B", min_bend=0.0)
    marked_old = 0
    for isl in islands:
        flat = [p for p in plan_folds(isl, mat) if p.style == "flat"]
        assert fold_labels(isl, flat, mat) == []
        ids = {id(p.fold) for p in flat}
        again = [p for p in plan_folds(isl, old) if id(p.fold) in ids]
        marked_old += len(fold_labels(isl, again, old))
    assert marked_old > 0


def test_a_part_knows_its_model():
    """Island ids restart per object; on shared sheets the model says whose."""
    pm = PolyMesh.from_trimesh(trimesh.creation.box(extents=(50, 50, 50)))
    islands, _ = unfold(pm)
    parts = build_parts(islands, Material.from_flute("B", model_id="BOX"))
    assert {p.model for p in parts} == {"BOX"}


def test_a_flat_piece_is_not_told_to_re_unfold():
    """Narrow places in a flat piece are its drawing, not the unfold's - a
    different seed changes nothing, so the warning must not say it would."""
    # two discs joined by a 3 mm bar: a pinch drawn into the piece
    shape = unary_union([Point(0, 0).buffer(30), Point(100, 0).buffer(30),
                         box(0, -1.5, 100, 1.5)])
    tris = shapely.constrained_delaunay_triangles(shape)
    verts, idx, rings = [], {}, []
    for t in tris.geoms:
        r = []
        for x, y in list(t.exterior.coords)[:-1]:
            k = (round(x, 9), round(y, 9))
            if k not in idx:
                idx[k] = len(verts)
                verts.append((x, y, 5.0))
            r.append(idx[k])
        if _signed2d(np.array(verts)[r]) < 0:
            r = r[::-1]
        rings.append(r)
    pm = PolyMesh.from_rings(np.array(verts), rings, merge=False)
    islands, _ = unfold(pm, flat_tol=3.0)
    assert islands[0].flat
    warns = build_parts(islands, Material.from_flute("B"))[0].warnings
    assert any("flat piece" in w for w in warns), warns
    assert not any("seed" in w for w in warns), warns


# ------------------------------------------------------------- G-code (GRBL)

def _grbl(text):
    """What a GRBL 1.1 laser in laser mode ($32=1) does with a file: every
    lit G1 segment as (op, (x0, y0), (x1, y1), S, F, air), in order, plus
    the state it ends in. G0 never burns - laser mode turns the beam off for
    rapids - and M5 turns it off until the next M3/M4."""
    x = y = 0.0
    s = f = 0.0
    beam = air = False
    op = None
    burns = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("; ---"):
            op = line[5:].strip()
            continue
        line = line.split(";")[0].strip()
        if not line:
            continue
        words = dict((w[0], float(w[1:]) if w[1:] else 0.0)
                     for w in line.split())
        if "S" in words:
            s = words["S"]
        if "F" in words:
            f = words["F"]
        if "M" in words:
            m = int(words["M"])
            beam = {3: True, 4: True, 5: False}.get(m, beam)
            air = {7: True, 8: True, 9: False}.get(m, air)
        if "G" in words:
            g = int(words["G"])
            nx, ny = words.get("X", x), words.get("Y", y)
            if g == 1 and beam and s > 0:
                burns.append((op, (x, y), (nx, ny), s, f, air))
            if g in (0, 1):
                x, y = nx, ny
    return burns, dict(beam=beam, air=air, x=x, y=y)


def _gcode_parts(name, mat):
    tm = dict(ALL)[name]
    pm = PolyMesh.from_trimesh(tm)
    pad = 2 * mat.tab_depth + 2 * mat.kerf
    islands, _ = unfold(pm, trials=40, max_size=(mat.usable_bed[0] - pad,
                                                 mat.usable_bed[1] - pad))
    parts = build_parts(islands, mat)
    sheets, _ = pack(parts, mat)
    return parts, sheets


def _seg_len(b):
    return math.hypot(b[2][0] - b[1][0], b[2][1] - b[1][1])


def _geom_len(geoms):
    return sum(g.length for g in geoms if g is not None and not g.is_empty)


def _mid(b):
    return shapely.Point((b[1][0] + b[2][0]) / 2, (b[1][1] + b[2][1]) / 2)


@pytest.mark.parametrize("name", ["cylinder", "helmet"])
def test_gcode_burns_exactly_what_the_sheet_holds(name, tmp_path):
    """Per operation, the lit length is the geometry's length times its
    passes, and every lit segment lies on that operation's geometry - so
    nothing is missing, nothing is burned twice by accident, and no travel
    move is lit."""
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    mat.cut_passes = 2
    mat.score_folds = True
    parts, sheets = _gcode_parts(name, mat)
    for sh in range(sheets):
        f = tmp_path / f"s{sh}.nc"
        write_gcode(f, parts, mat, sh, sheets)
        burns, end = _grbl(f.read_text())
        mine = [p for p in parts if p.sheet == sh]
        want = {
            "ENGRAVE": ([p.engrave for p in mine], mat.engrave_passes),
            "SCORE": ([g for p in mine for g in (p.perf_m, p.perf_v)],
                      mat.score_passes),
            "CUT": ([p.cut.boundary for p in mine], mat.cut_passes),
        }
        for op, (geoms, n) in want.items():
            lit = [b for b in burns if b[0] == op]
            got = sum(_seg_len(b) for b in lit)
            exp = _geom_len(geoms) * n
            # coordinates are written to 0.001 mm: each end moves at most
            # 0.0005 mm per axis, so each segment by at most 2 * 0.0005 * sqrt 2
            assert got == pytest.approx(exp, abs=1.5e-3 * len(lit) + 1e-6), (op, sh)
            if not exp:
                continue
            body = unary_union([g for g in geoms if not g.is_empty])
            far = max(body.distance(_mid(b)) for b in burns if b[0] == op)
            assert far < 2e-3, f"{op}: a lit segment {far:.4f} mm off its geometry"
        assert not end["beam"], "the file must end with the laser off"
    # the fixture is not vacuous
    assert any(not p.perf_m.is_empty or not p.perf_v.is_empty for p in parts)
    assert any(not p.engrave.is_empty for p in parts)


def test_gcode_cuts_every_hole_before_its_outline(tmp_path):
    """Relief slots and corner holes are holes in the part: if the outline
    cuts through first the part drops and they are never finished. Every
    pass of every hole comes before the first pass of its outline."""
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    mat.cut_passes = 3
    mat.relief_threshold = 0.0          # every fold a slot: plenty of holes
    parts, sheets = _gcode_parts("helmet", mat)
    holes_seen = 0
    for sh in range(sheets):
        f = tmp_path / f"s{sh}.nc"
        write_gcode(f, parts, mat, sh, sheets)
        burns = [b for b in _grbl(f.read_text())[0] if b[0] == "CUT"]
        mids = [_mid(b) for b in burns]
        for p in (q for q in parts if q.sheet == sh):
            polys = p.cut.geoms if hasattr(p.cut, "geoms") else [p.cut]
            for poly in polys:
                if not poly.interiors:
                    continue
                outer = poly.exterior
                inner = MultiLineString([list(r.coords) for r in poly.interiors])
                i_out = [i for i, m in enumerate(mids) if outer.distance(m) < 1e-3]
                i_in = [i for i, m in enumerate(mids) if inner.distance(m) < 1e-3]
                assert i_out and i_in
                holes_seen += len(poly.interiors)
                assert max(i_in) < min(i_out), "an outline was cut before its holes"
    assert holes_seen, "the fixture has no holes to test"


def test_gcode_runs_engrave_then_score_then_cut_at_their_own_settings(tmp_path):
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    mat.score_folds = True
    mat.score_power, mat.score_speed = 600.0, 800.0
    parts, sheets = _gcode_parts("cylinder", mat)
    f = tmp_path / "s.nc"
    write_gcode(f, parts, mat, 0, sheets)
    text = f.read_text()
    burns, _ = _grbl(text)
    seq = [b[0] for b in burns]
    order = [op for i, op in enumerate(seq) if i == 0 or seq[i - 1] != op]
    assert order == ["ENGRAVE", "SCORE", "CUT"], order
    want = {"ENGRAVE": (250.0, 1000.0), "SCORE": (600.0, 800.0),
            "CUT": (1000.0, 500.0)}
    for b in burns:
        assert (b[3], b[4]) == want[b[0]], b[0]
    assert "M4" in text and "M3" not in text, "dynamic power is M4"
    assert all(b[3] <= mat.laser_max for b in burns)


def test_gcode_stays_on_the_bed_and_refuses_otherwise(tmp_path):
    """Origin front-left, Y away from you: the sheet's own coordinates,
    unflipped, so every point is inside the bed the parts were packed for.
    A part off the bed is a file that would run the head into the frame."""
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    parts, sheets = _gcode_parts("cylinder", mat)
    f = tmp_path / "s.nc"
    write_gcode(f, parts, mat, 0, sheets)
    W, H = mat.bed
    pts = [q for b in _grbl(f.read_text())[0] for q in (b[1], b[2])]
    assert all(-1e-3 <= x <= W + 1e-3 and -1e-3 <= y <= H + 1e-3 for x, y in pts)
    # not flipped: the lowest lit point is the lowest point of the parts
    lo = min(p.cut.bounds[1] for p in parts if p.sheet == 0)
    assert min(y for _, y in pts) == pytest.approx(lo, abs=1e-3)
    p0 = next(p for p in parts if p.sheet == 0)
    p0.translate(mat.bed[0], 0.0)
    with pytest.raises(ValueError, match="leaves"):
        write_gcode(tmp_path / "bad.nc", parts, mat, 0, sheets)


def test_gcode_refuses_power_above_the_machine_maximum(tmp_path):
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    parts, sheets = _gcode_parts("box", mat)
    mat.laser_max = 255.0               # a $30=255 machine: 1000 is not a value
    with pytest.raises(ValueError, match=r"\$30"):
        write_gcode(tmp_path / "s.nc", parts, mat, 0, sheets)


def test_gcode_air_assist_only_when_fitted_and_only_where_asked(tmp_path):
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    parts, sheets = _gcode_parts("cylinder", mat)
    f = tmp_path / "off.nc"
    write_gcode(f, parts, mat, 0, sheets)
    text = f.read_text()
    assert "M8" not in text and "M7" not in text, "no air assist fitted by default"

    mat.air_assist = True
    mat.score_folds = True
    f = tmp_path / "on.nc"
    write_gcode(f, parts, mat, 0, sheets)
    burns, end = _grbl(f.read_text())
    for b in burns:
        assert b[5] == {"ENGRAVE": False, "SCORE": True, "CUT": True}[b[0]], b[0]
    assert not end["air"] and not end["beam"], "air and laser off at the end"


def test_write_all_writes_one_gcode_file_per_sheet(tmp_path):
    from cardboard_unfolder import run
    tm = trimesh.creation.box()
    p = tmp_path / "cube.stl"
    tm.export(p)
    rep = run(p, tmp_path / "out", Material.from_flute("B"), height_mm=110,
              trials=20, dxf=False, verbose=False)
    ncs = sorted((tmp_path / "out").glob("*.nc"))
    assert len(ncs) == rep["sheets"] >= 1
    run(p, tmp_path / "none", Material.from_flute("B"), height_mm=110,
        trials=20, dxf=False, verbose=False, gcode=False)
    assert not list((tmp_path / "none").glob("*.nc"))


def test_fold_scores_are_off_by_default_and_burn_nowhere(tmp_path):
    """The user's call: a fold score is an option, off by default. At the
    cut setting the perforation dashes are slits through the board, so with
    it off no fold line reaches any output - SVG, DXF or G-code - and the
    relief slots, which are holes in the cut, still do."""
    import xml.etree.ElementTree as ET
    import ezdxf
    from cardboard_unfolder.export import write_svg, write_dxf
    from cardboard_unfolder.gcode import write_gcode
    mat = Material.from_flute("B")
    assert mat.score_folds is False
    parts, sheets = _gcode_parts("cylinder", mat)
    assert any(not p.perf_m.is_empty or not p.perf_v.is_empty for p in parts), \
        "the fixture has fold lines to leave out"

    def layers(m):
        write_svg(tmp_path / "s.svg", parts, m, 0)
        root = ET.parse(tmp_path / "s.svg").getroot()
        svg = {g.get("{http://www.inkscape.org/namespaces/inkscape}label")
               for g in root.iter("{http://www.w3.org/2000/svg}g")}
        write_dxf(tmp_path / "s.dxf", parts, m, 0)
        dxf = {e.dxf.layer for e in ezdxf.readfile(tmp_path / "s.dxf").modelspace()}
        write_gcode(tmp_path / "s.nc", parts, m, 0, sheets)
        nc = {b[0] for b in _grbl((tmp_path / "s.nc").read_text())[0]}
        return svg, dxf, nc

    svg, dxf, nc = layers(mat)
    for name, got in (("svg", svg), ("dxf", dxf), ("nc", nc)):
        assert not {"PERF_M", "PERF_V", "SCORE"} & got, (name, got)
        assert "CUT" in got and "ENGRAVE" in got, (name, got)

    mat.score_folds = True
    svg, dxf, nc = layers(mat)
    assert {"PERF_M", "PERF_V"} & svg and {"PERF_M", "PERF_V"} & dxf
    assert "SCORE" in nc


def test_a_fold_score_does_not_default_to_cutting_power():
    """Switched on, the score starts where the board is known not to cut
    through - the measured engrave setting - not at the cut setting."""
    mat = Material.from_flute("B")
    assert (mat.score_power, mat.score_speed) == (mat.engrave_power,
                                                  mat.engrave_speed)
    assert mat.score_power < mat.cut_power
