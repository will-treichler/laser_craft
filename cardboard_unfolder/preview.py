"""Render sheets to PNG so you can eyeball a job before burning cardboard."""
from __future__ import annotations
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, MultiLineString

from .material import Material
from .export import _rings, _lines


def render(parts, mat: Material, sheet: int, path, dpi: int = 110, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    W, H = mat.bed
    fig, ax = plt.subplots(figsize=(W / 60, H / 60), dpi=dpi)
    ax.add_patch(MplPoly([(0, 0), (W, 0), (W, H), (0, H)], closed=True,
                         fc="#fbf7ef", ec="#bbbbbb", lw=0.8, ls="--", zorder=0))
    m = mat.sheet_margin
    ax.add_patch(MplPoly([(m, m), (W - m, m), (W - m, H - m), (m, H - m)],
                         closed=True, fc="none", ec="#dddddd", lw=0.6, zorder=0))

    mine = [p for p in parts if p.sheet == sheet]
    for p in mine:
        geoms = p.cut.geoms if isinstance(p.cut, MultiPolygon) else [p.cut]
        for g in geoms:
            if not isinstance(g, Polygon) or g.is_empty:
                continue
            ax.add_patch(MplPoly(np.asarray(g.exterior.coords), closed=True,
                                 fc="#e8d9bd", ec="#c0301c", lw=1.0, zorder=1))
            for it in g.interiors:
                ax.add_patch(MplPoly(np.asarray(it.coords), closed=True,
                                     fc="#fbf7ef", ec="#c0301c", lw=1.0, zorder=2))
        for l in _lines(p.perf_m):
            ax.plot(l[:, 0], l[:, 1], color="#2c5b9e", lw=1.1, zorder=3)
        for l in _lines(p.perf_v):
            ax.plot(l[:, 0], l[:, 1], color="#3aa0d8", lw=1.1, zorder=3)
        for l in _lines(p.engrave):
            ax.plot(l[:, 0], l[:, 1], color="#333333", lw=0.7, zorder=4)

    ax.set_xlim(-10, W + 10)
    ax.set_ylim(-10, H + 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"sheet {sheet + 1} - {len(mine)} part(s) - "
                          f"{W:.0f}x{H:.0f} mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def render_all(parts, mat: Material, sheets: int, outdir, stem="preview"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return [render(parts, mat, s, outdir / f"{stem}_{s + 1:02d}.png")
            for s in range(sheets)]
