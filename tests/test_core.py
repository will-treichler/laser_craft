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


def test_pack_never_loses_to_either_strategy():
    """Neither packer dominates, so pack() runs both and keeps the winner.

    Nesting is far better when parts can share a sheet and worse when its
    greedy placement paints itself into a corner - measured both ways on real
    nets. This is the property that makes adding it safe.
    """
    from cardboard_unfolder.layout import shelf_pack, nest as nest_fn
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
        s_shelf, _ = shelf_pack(a, m)
        s_nest, _ = nest_fn(b, m)
        s_pack, _ = pack(c, m)
        assert s_pack <= min(s_shelf, s_nest), (
            f"pack used {s_pack} sheets, shelf {s_shelf}, nest {s_nest}")


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
    for p in over:
        x0, y0, _x1, _y1 = p.bounds
        assert x0 == pytest.approx(m.sheet_margin, abs=1e-6),             f"island {p.island_id} parked at x={x0:.1f}, not the margin"
        assert y0 == pytest.approx(m.sheet_margin, abs=1e-6)
        assert p.sheet < 0, "and it must still be flagged as not placed"


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
    assert {"CUT", "PERF_M", "ENGRAVE"} <= labels

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


# ------------------------------------------------------------------- font

def test_stroke_font_covers_labels_and_scales():
    for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert strokefont.text_strokes(ch, 5.0), f"glyph {ch} is empty"
    s = strokefont.text_strokes("P12", 5.0, origin=(0, 0))
    pts = np.vstack(s)
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(5.0, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
