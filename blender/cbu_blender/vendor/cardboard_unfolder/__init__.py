"""cardboard_unfolder - low-poly mesh to laser-cut corrugated cardboard.

    from cardboard_unfolder import Material, run
    run("helmet.stl", "out/", Material.from_flute("B"), height_mm=280)

Pipeline: load and merge coplanar faces -> unfold to non-overlapping islands
-> thickness-aware fold relief -> glue tabs and seam numbers -> kerf -> pack
onto sheets -> layered SVG/DXF.
"""
from __future__ import annotations
from pathlib import Path

from .material import Material, FLUTES
from .mesh import PolyMesh
from .unfold import unfold, Island
from .folds import relief_width, plan_folds
from .layout import build_parts, pack, Part
from .export import write_all

__version__ = "0.1.0"
__all__ = ["Material", "FLUTES", "PolyMesh", "unfold", "build_parts",
           "pack", "write_all", "run"]


def run(mesh_path, outdir, material: Material | None = None,
        height_mm: float | None = None, scale: float | None = None,
        trials: int = 250, seed: int = 0, merge_coplanar: bool = True,
        dxf: bool = True, verbose: bool = True) -> dict:
    """Full pipeline. Returns a report dict; writes SVG/DXF/manifest to outdir."""
    mat = material or Material.from_flute("B")
    log = print if verbose else (lambda *a, **k: None)

    pm = PolyMesh.load(str(mesh_path), merge_coplanar=merge_coplanar)
    log(f"loaded {Path(mesh_path).name}: {pm}")
    for p in pm.check():
        log(f"  ! {p}")

    if height_mm is not None:
        pm = pm.scale_to_height(height_mm)
    elif scale is not None:
        pm = pm.scale(scale)
    log(f"  size {pm.extents.round(1)} mm")

    # leave room for glue tabs on both sides and a little kerf slack
    pad = 2 * mat.tab_depth + 2 * mat.kerf
    limit = (mat.usable_bed[0] - pad, mat.usable_bed[1] - pad)

    islands, rep = unfold(pm, trials=trials, seed=seed,
                          max_size=limit, verbose=verbose)
    log(f"unfolded: {rep['islands']} island(s), isometry error "
        f"{rep['isometry_error']:.1e}, {rep['trials_used']} trial(s), "
        f"{rep['extra_cuts']} repair cut(s), {rep['size_splits']} bed split(s)")
    if rep["unresolved"]:
        log(f"  ! {rep['unresolved']} overlap(s) could not be resolved")

    parts = build_parts(islands, mat)
    sheets, oversized = pack(parts, mat)
    log(f"packed {len(parts)} part(s) onto {sheets} sheet(s) of "
        f"{mat.bed[0]:.0f}x{mat.bed[1]:.0f} mm")
    for p in parts:
        for w in p.warnings:
            log(f"  ! part {p.island_id}: {w}")
    if oversized:
        log(f"  ! {len(oversized)} part(s) exceed the bed and were not placed: "
            f"{[(p.island_id, tuple(round(v) for v in p.size)) for p in oversized]}")
        log("    (bed-size splitting with splice joints is M6 - for now scale "
            "down or add cuts)")

    files = write_all(outdir, parts, mat, sheets, rep, dxf=dxf)
    log(f"wrote {len(files)} file(s) to {outdir}")

    rep.update(sheets=sheets, parts=len(parts),
               oversized=[p.island_id for p in oversized], files=files)
    return rep
