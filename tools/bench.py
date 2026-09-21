"""Measure the unfolder on a fixed corpus, so optimiser work can be judged.

Every claim about the optimiser is a claim about numbers on real models, and
the numbers move in several directions at once - fewer sheets can cost more
splinters, fewer islands can cost a sheet. This prints all of them together and
diffs against a saved run, so a change is measured rather than argued about.

    python tools/bench.py                       # the table
    python tools/bench.py --json base.json      # ...and save it
    python tools/bench.py --baseline base.json  # ...and show what moved

Deterministic: fixed seed, fixed trials, fixed corpus. A row that moves means
the code moved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cardboard_unfolder import Material, PolyMesh                    # noqa: E402
from cardboard_unfolder.layout import (build_parts, inscribed_width,  # noqa: E402
                                       narrow_necks, pack)
from cardboard_unfolder.unfold import mirrored_faces, unfold          # noqa: E402


def stl_triangles(path) -> int | None:
    """How many triangles a binary STL holds, from its header alone.

    A binary STL is an 80-byte header, a uint32 count, then exactly 50 bytes
    per triangle - so the size arithmetic both gives the count and proves the
    file really is binary. None means "could not tell, load it and see".

    Worth the 84 bytes: the helmet is over the budget and gets skipped, but it
    was being fully parsed first, which is most of a minute of every single
    bench run before any measuring starts.
    """
    try:
        size = path.stat().st_size
        if size < 84:
            return None
        with open(path, "rb") as fh:
            head = fh.read(84)
        # Note the leading word is NOT the test: an ASCII file starts
        # "solid", and so can a binary one whose 80-byte header happens to.
        # The size arithmetic is what settles it.
        n = int.from_bytes(head[80:84], "little")
        return n if 84 + 50 * n == size else None
    except OSError:
        return None


def corpus(height_mm: float, max_tris: int = 20000):
    """Models that stress different things, smallest first.

    Primitives are reproducible anywhere and catch regressions; the two real
    scans are what actually has concave n-gons, near-pinches and 250+ faces,
    and every interesting failure this session came from those.
    """
    import trimesh
    out = [
        ("box", trimesh.creation.box()),
        ("cone", trimesh.creation.cone(radius=1, height=2, sections=12)),
        ("cylinder", trimesh.creation.cylinder(radius=1, height=2, sections=16)),
        ("icosphere1", trimesh.creation.icosphere(subdivisions=1)),
        ("annulus", trimesh.creation.annulus(r_min=.5, r_max=1, height=1,
                                             sections=14)),
        ("capsule", trimesh.creation.capsule(count=[10, 10])),
        ("torus", trimesh.creation.torus(major_radius=1, minor_radius=.4,
                                         major_sections=16,
                                         minor_sections=10)),
        ("icosphere2", trimesh.creation.icosphere(subdivisions=2)),
    ]
    for name, path in (("pikachu",
                        "OLD_VERSION_READ_DESCRIPTION_pikachu_1gen_flowalistik.STL"),
                       ("helmet", "spartan-helmet-low-poly.stl")):
        f = ROOT / path
        if not f.is_file():
            continue
        # A budget, because "low poly" in a filename is not a fact. The spartan
        # helmet is 499,992 triangles and the coplanar merge only removes 18%
        # of them - it is a smooth scan, not a faceted model, and there is
        # nothing here to unfold until it has been decimated. Benchmarking it
        # means waiting minutes to measure a case nobody can cut.
        #
        # Read the count from the header where that is possible, so an
        # over-budget model is skipped without being parsed at all.
        n = stl_triangles(f)
        if n is not None and n > max_tris:
            say(f"  skip {name}: {n} triangles, over the {max_tris} budget "
                f"- decimate it first")
            continue
        tm = trimesh.load(str(f), force="mesh", process=True)
        if len(tm.faces) > max_tris:
            say(f"  skip {name}: {len(tm.faces)} triangles, over the "
                f"{max_tris} budget - decimate it first")
            continue
        out.append((name, tm))
    return out


def measure(name, tm, mat, height_mm, trials, seed, optimise=False,
            budget=10):
    """One row. Returns a dict of plain numbers, nothing shapely."""
    pm = PolyMesh.from_trimesh(tm).scale_to_height(height_mm)
    pad = 2 * mat.tab_depth + 2 * mat.kerf
    limit = (mat.usable_bed[0] - pad, mat.usable_bed[1] - pad)

    t0 = time.perf_counter()
    islands, rep = unfold(pm, trials=trials, seed=seed, max_size=limit,
                          splinter_width=mat.splinter_width,
                          mat=mat if optimise else None, budget=budget)
    t_unfold = time.perf_counter() - t0

    t0 = time.perf_counter()
    parts = build_parts(islands, mat)
    sheets, over = pack(parts, mat)
    t_pack = time.perf_counter() - t0

    widths = [inscribed_width(p.cut) for p in parts]
    necks = sum(narrow_necks(p.cut, max(3.0, 2.0 * mat.thickness)) for p in parts)
    seam_mm = sum(c.length for i in islands for c in i.cuts if c.seam >= 0)
    # PLACED parts only. A part too big for the bed is parked rather than
    # placed, and counting its area against the sheets it is not on reports
    # utilisations above 100% - a 400 mm annulus read 125.4%, and a 400 mm box
    # read 1190% on zero sheets, which is a model that cannot be cut at all
    # announcing the best score in the table.
    area = sum(p.cut.area for p in parts if p.sheet >= 0)
    bed_area = mat.usable_bed[0] * mat.usable_bed[1]

    # The pre-split positions are the only frame in which faces are comparable;
    # unfold() has already re-rooted every component, so measure the mirror
    # invariant on what it returned rather than recomputing a layout.
    mirrored = 0
    for isl in islands:
        mirrored += len(mirrored_faces(pm, isl.faces))

    return dict(
        model=name, faces=len(pm.faces), islands=len(islands),
        splinters=sum(1 for w in widths if w < mat.splinter_width),
        # Count alone is misleading and it took eight pikachu nets to see it:
        # the one with the FEWEST pieces under 15 mm had ten of them under 5,
        # while the one with the most had five, the rest a buildable 8-14 mm.
        # Ranking by `splinters` puts the unbuildable net first, so the table
        # has to show the shape of the distribution, not just its size.
        tiny=sum(1 for w in widths if w < mat.splinter_width / 3.0),
        min_width=round(min(widths), 1) if widths else 0.0,
        med_width=round(float(np.median(widths)), 1) if widths else 0.0,
        necks=necks, sheets=sheets,
        # Parts no sheet can hold. Invisible in every other column - not in
        # the area, not on a sheet, and counted as islands - so without a
        # column of its own the table says nothing about a model the splitter
        # cannot finish. Bed splitting only cuts along fold edges, so a single
        # FACE bigger than the bed has no answer here at all (milestone 4).
        oversize=len(over),
        util=round(100.0 * area / sheets / bed_area, 1) if sheets else 0.0,
        seam_mm=round(seam_mm),
        extra_cuts=rep["extra_cuts"], size_splits=rep["size_splits"],
        mirrored=mirrored, unresolved=rep["unresolved"],
        iso=rep["isometry_error"],
        t_unfold=round(t_unfold, 2), t_pack=round(t_pack, 2),
    )


COLS = [("model", "%-11s"), ("faces", "%5d"), ("islands", "%7d"),
        ("splinters", "%9d"), ("tiny", "%4d"), ("min_width", "%9.1f"),
        ("necks", "%5d"),
        ("sheets", "%6d"), ("oversize", "%8d"), ("util", "%5.1f"),
        ("seam_mm", "%7d"),
        ("extra_cuts", "%10d"), ("mirrored", "%8d"),
        ("t_unfold", "%8.2f"), ("t_pack", "%6.2f")]
# Lower is better for all of these; util is the one that wants to go up.
BETTER_UP = {"util"}
WATCH = ["islands", "splinters", "tiny", "sheets", "oversize", "util",
         "seam_mm", "extra_cuts",
         "mirrored"]


def say(*a):
    """Print and flush. See the note in main()."""
    print(*a, flush=True)


def header():
    return "  ".join(("%-11s" if k == "model" else
                      "%" + f[1:-1].split(".")[0] + "s") % k for k, f in COLS)


def row(r):
    return "  ".join(f % r[k] for k, f in COLS)


def deltas(new, old):
    """What moved, and whether it moved the right way."""
    by = {r["model"]: r for r in old}
    lines = []
    for r in new:
        b = by.get(r["model"])
        if b is None:
            lines.append(f"  {r['model']:<11s}  NEW")
            continue
        bits = []
        for k in WATCH:
            if k not in b:
                # A baseline saved before this column existed is a normal
                # thing to diff against - it must report what it can, not
                # refuse the whole comparison over one missing key.
                continue
            d = r[k] - b[k]
            if abs(d) < (0.05 if isinstance(d, float) else 1):
                continue
            good = (d > 0) if k in BETTER_UP else (d < 0)
            bits.append(f"{k} {b[k]}->{r[k]} {'better' if good else 'WORSE'}")
        if bits:
            lines.append(f"  {r['model']:<11s}  " + ",  ".join(bits))
    return lines or ["  nothing moved"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--height", type=float, default=200.0,
                    help="scale every model to this height, mm")
    ap.add_argument("--bed", default="300x300")
    ap.add_argument("--flute", default="B")
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--tabs", action="store_true")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tris", type=int, default=20000,
                    dest="max_tris",
                    help="skip models bigger than this")
    ap.add_argument("--optimise", action="store_true",
                    help="cost-driven tree search instead of overlap count")
    ap.add_argument("--budget", type=int, default=10,
                    help="candidates the cost search may mend")
    ap.add_argument("--only", default=None, help="substring filter on model name")
    ap.add_argument("--json", default=None, help="write the rows here")
    ap.add_argument("--baseline", default=None, help="diff against this file")
    a = ap.parse_args(argv)

    mat = Material.from_flute(a.flute)
    w, h = (float(v) for v in a.bed.lower().split("x"))
    mat.bed = (w, h)
    mat.part_gap = a.gap
    mat.use_tabs = a.tabs

    say(f"bed {w:g}x{h:g}  flute {a.flute}  gap {a.gap:g}  tabs {a.tabs}  "
          f"height {a.height:g}  trials {a.trials}  seed {a.seed}  "
          f"splinter under {mat.splinter_width:g} mm")
    say(header())
    rows = []
    t0 = time.perf_counter()
    for name, tm in corpus(a.height, a.max_tris):
        if a.only and a.only not in name:
            continue
        # printed as it goes: a silent ten-minute run looks like a hang, which
        # is the same complaint the optimiser's progress hook exists for
        r = measure(name, tm, mat, a.height, a.trials, a.seed,
                    optimise=a.optimise, budget=a.budget)
        rows.append(r)
        say(row(r))
    say(f"total {time.perf_counter() - t0:.1f} s")

    bad = [r for r in rows if r["mirrored"] or r["unresolved"] or r["oversize"]]
    for r in bad:
        say(f"  ! {r['model']}: mirrored={r['mirrored']} "
            f"unresolved={r['unresolved']} oversize={r['oversize']}")

    if a.json:
        # newline="" or Windows text mode writes CRLF, which .gitattributes
        # then normalises on commit - so the file is dirty the moment it is
        # written and every later diff carries it.
        Path(a.json).write_text(json.dumps(rows, indent=1) + "\n",
                                newline="\n")
        print(f"wrote {a.json}")
    if a.baseline:
        old = json.loads(Path(a.baseline).read_text())
        print(f"\nagainst {a.baseline}:")
        for line in deltas(rows, old):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
