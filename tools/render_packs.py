"""Render the corpus packings so they can be judged by eye.

One PNG per model, every sheet side by side, with the utilisation on each so a
half-empty sheet is obvious rather than something you have to measure. Drawn
from the real Part outlines - the thing the laser follows, tabs and relief
included - not from the island UVs, which are smaller and would flatter the
packer.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path("tools").resolve()))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from shapely.geometry import Polygon, MultiPolygon
import bench, importlib
from cardboard_unfolder.mesh import PolyMesh
from cardboard_unfolder.material import Material
from cardboard_unfolder.layout import build_parts, pack
U = importlib.import_module("cardboard_unfolder.unfold")

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/packs")
OUT.mkdir(parents=True, exist_ok=True)
ONLY = sys.argv[2] if len(sys.argv) > 2 else None

mat = Material.from_flute("B")
mat.bed = (300.0, 300.0)
mat.part_gap = 1.0
mat.use_tabs = False
pad = 2 * mat.tab_depth + 2 * mat.kerf
limit = (mat.usable_bed[0] - pad, mat.usable_bed[1] - pad)
W, H = mat.bed
m = mat.sheet_margin
BED_AREA = mat.usable_bed[0] * mat.usable_bed[1]


def draw(ax, parts, sheet):
    ax.add_patch(MplPoly([(0, 0), (W, 0), (W, H), (0, H)], closed=True,
                         fc="#fbf7ef", ec="#999999", lw=1.0, zorder=0))
    ax.add_patch(MplPoly([(m, m), (W - m, m), (W - m, H - m), (m, H - m)],
                         closed=True, fc="none", ec="#d8d2c4", lw=0.8,
                         ls="--", zorder=0))
    mine = [p for p in parts if p.sheet == sheet]
    area = 0.0
    for i, p in enumerate(mine):
        col = plt.cm.tab20((i % 20) / 20.0)
        geoms = p.cut.geoms if isinstance(p.cut, MultiPolygon) else [p.cut]
        for g in geoms:
            if not isinstance(g, Polygon) or g.is_empty:
                continue
            ax.add_patch(MplPoly(np.asarray(g.exterior.coords), closed=True,
                                 fc=col, ec="#7a2a1c", lw=0.7, alpha=0.85,
                                 zorder=1))
            for it in g.interiors:
                ax.add_patch(MplPoly(np.asarray(it.coords), closed=True,
                                     fc="#fbf7ef", ec="#7a2a1c", lw=0.5,
                                     zorder=2))
        area += p.cut.area
    ax.set_xlim(-4, W + 4)
    ax.set_ylim(-4, H + 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"sheet {sheet + 1}   {len(mine)} part(s)   "
                 f"{100 * area / BED_AREA:.0f}% full", fontsize=9)
    return area


for name, tm in bench.corpus(200.0):
    if ONLY and ONLY not in name:
        continue
    for label, kw in (("plain", {}), ("optimise", dict(mat=mat, budget=10))):
        pm = PolyMesh.from_trimesh(tm).scale_to_height(200.0)
        t = time.perf_counter()
        isl, rep = U.unfold(pm, trials=60, seed=0, max_size=limit,
                            splinter_width=mat.splinter_width, **kw)
        parts = build_parts(isl, mat)
        sheets, over = pack(parts, mat)
        dt = time.perf_counter() - t

        cols = min(max(sheets, 1), 6)
        rows = int(np.ceil(max(sheets, 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.8 * rows),
                                 squeeze=False)
        total = 0.0
        for k in range(rows * cols):
            ax = axes[k // cols][k % cols]
            if k < sheets:
                total += draw(ax, parts, k)
            else:
                ax.axis("off")
        fig.suptitle(
            f"{name} - {label} - {len(isl)} island(s), {sheets} sheet(s), "
            f"{100 * total / max(sheets, 1) / BED_AREA:.1f}% mean fill "
            f"({dt:.0f}s)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = OUT / f"pack_{name}_{label}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"{path.name}: {len(isl)} islands, {sheets} sheets", flush=True)
