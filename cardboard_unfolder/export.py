"""Layered SVG and DXF output, one file per sheet.

LightBurn and RDWorks both key operations off stroke colour, and LightBurn
maps DXF layers straight to its own. So set both, consistently. Multi-page
SVG isn't read reliably by laser software and DXF has no page concept at
all, hence one file per sheet.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, MultiLineString

from .material import Material
from .layout import Part

# operation -> (svg stroke, dxf aci colour, dash pattern)
LAYERS = {
    "CUT":     ("#FF0000", 1, None),
    "PERF_M":  ("#0000FF", 5, None),
    "PERF_V":  ("#00AAFF", 4, None),
    "ENGRAVE": ("#000000", 7, None),
    "REF":     ("#CCCCCC", 8, "4,3"),
}
# Document order is operation order in most laser software: engrave first so
# the part is still held by its web, perimeters last so nothing drops early.
ORDER = ["REF", "ENGRAVE", "PERF_V", "PERF_M", "CUT"]


def _rings(geom):
    if geom.is_empty:
        return []
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        out.append(np.asarray(p.exterior.coords))
        out.extend(np.asarray(i.coords) for i in p.interiors)
    return out


def _lines(geom):
    if geom is None or geom.is_empty:
        return []
    gs = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    return [np.asarray(g.coords) for g in gs if not g.is_empty]


def _path(pts, close):
    d = f"M {pts[0][0]:.4f},{pts[0][1]:.4f} " + " ".join(
        f"L {x:.4f},{y:.4f}" for x, y in pts[1:])
    return d + (" Z" if close else "")


def write_svg(path, parts, mat: Material, sheet: int, flip_y: bool = True) -> None:
    """One sheet. Y is flipped so the file reads the same way it cuts:
    SVG's Y grows downward, our geometry's grows up."""
    W, H = mat.bed
    mine = [p for p in parts if p.sheet == sheet]

    def xf(pts):
        pts = np.asarray(pts, dtype=float)
        return np.column_stack([pts[:, 0], H - pts[:, 1]]) if flip_y else pts

    buckets = {k: [] for k in LAYERS}
    buckets["REF"].append((np.array([[0, 0], [W, 0], [W, H], [0, H]]), True))
    for p in mine:
        for r in _rings(p.cut):
            buckets["CUT"].append((r, True))
        for name, geom in (("PERF_M", p.perf_m), ("PERF_V", p.perf_v),
                           ("ENGRAVE", p.engrave)):
            for l in _lines(geom):
                buckets[name].append((l, False))

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
        f'  <!-- sheet {sheet}: {len(mine)} part(s); 1 unit = 1 mm -->',
    ]
    for name in ORDER:
        items = buckets[name]
        if not items:
            continue
        stroke, _, dash = LAYERS[name]
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'  <g inkscape:groupmode="layer" inkscape:label="{name}" '
                   f'id="layer_{name.lower()}" fill="none" stroke="{stroke}" '
                   f'stroke-width="0.05"{da}>')
        for pts, close in items:
            out.append(f'    <path d="{_path(xf(pts), close)}"/>')
        out.append("  </g>")
    out.append("</svg>")
    Path(path).write_text("\n".join(out))


def write_dxf(path, parts, mat: Material, sheet: int) -> None:
    import ezdxf
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    for name, (_, aci, _) in LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name, color=aci)

    msp.add_lwpolyline([(0, 0), (mat.bed[0], 0), mat.bed, (0, mat.bed[1])],
                       close=True, dxfattribs={"layer": "REF"})
    for p in (q for q in parts if q.sheet == sheet):
        for r in _rings(p.cut):
            msp.add_lwpolyline([tuple(c) for c in r[:-1]], close=True,
                               dxfattribs={"layer": "CUT"})
        for name, geom in (("PERF_M", p.perf_m), ("PERF_V", p.perf_v),
                           ("ENGRAVE", p.engrave)):
            for l in _lines(geom):
                msp.add_lwpolyline([tuple(c) for c in l], close=False,
                                   dxfattribs={"layer": name})
    doc.saveas(path)


def write_manifest(path, parts, mat: Material, sheets: int, report: dict) -> None:
    """Assembly instructions and a recut need this; so does future you."""
    def cutlen(p):
        return sum(float(np.sum(np.linalg.norm(np.diff(r, axis=0), axis=1)))
                   for r in _rings(p.cut))
    data = {
        "material": {k: getattr(mat, k) for k in
                     ("thickness", "liner", "pitch", "kerf", "crush",
                      "relief_threshold", "bed", "sheet_margin", "part_gap")},
        "sheets": sheets,
        "unfold": report,
        "parts": [{"id": p.island_id, "sheet": p.sheet,
                   "size_mm": [round(v, 2) for v in p.size],
                   "area_mm2": round(p.cut.area, 1),
                   "cut_length_mm": round(cutlen(p), 1),
                   "warnings": p.warnings} for p in parts],
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def write_all(outdir, parts, mat: Material, sheets: int, report: dict,
              dxf: bool = True, stem: str = "sheet") -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in range(sheets):
        f = outdir / f"{stem}_{s + 1:02d}.svg"
        write_svg(f, parts, mat, s)
        written.append(str(f))
        if dxf:
            g = outdir / f"{stem}_{s + 1:02d}.dxf"
            write_dxf(g, parts, mat, s)
            written.append(str(g))
    write_manifest(outdir / "manifest.json", parts, mat, sheets, report)
    written.append(str(outdir / "manifest.json"))
    return written
