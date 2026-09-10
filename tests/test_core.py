"""Invariant tests. These check properties that must hold exactly, not
snapshots - if any of these fail the output is not cuttable.

Run:  python -m pytest tests -q      (or just: python tests/test_core.py)
"""
import math
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import trimesh
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardboard_unfolder import Material, PolyMesh
from cardboard_unfolder.unfold import (unfold, isometry_error, overlapping_pairs,
                                       search, min_rect_size, fits)
from cardboard_unfolder.folds import (relief_width, perf_segments, plan_folds,
                                      apply_relief, kerf_compensate)
from cardboard_unfolder.tabs import place_tabs
from cardboard_unfolder.layout import build_parts, pack, narrow_necks
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


def test_min_rect_beats_axis_aligned_for_a_diagonal():
    """A net rotated 45 degrees fits a bed its axis-aligned box would not."""
    s = 100.0
    pts = np.array([[0, 0], [s, s], [2 * s, 0], [s, -s]], dtype=float)
    aabb = pts.max(axis=0) - pts.min(axis=0)
    w, h = min_rect_size(pts)
    assert aabb[0] == pytest.approx(200.0)
    assert min(w, h) == pytest.approx(math.hypot(s, s), rel=1e-9)
    assert max(w, h) < aabb[0]


# ------------------------------------------------------------ fold geometry

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
def test_perf_period_stays_under_a_third_of_flute_pitch(grade):
    """Dash period must be small against the flute pitch or consecutive flutes
    get very different treatment - some ties on a glue line, some over a void."""
    m = Material.from_flute(grade)
    segs = perf_segments(np.array([0.0, 0.0]), np.array([80.0, 0.0]), m)
    assert len(segs) > 1
    period = segs[1][0][0] - segs[0][0][0]
    assert period <= m.pitch / 3 + 1e-9


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
    assert kerf_compensate(o, m.kerf).area > o.area
    assert kerf_compensate(o, 0.0).area == pytest.approx(o.area)


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
