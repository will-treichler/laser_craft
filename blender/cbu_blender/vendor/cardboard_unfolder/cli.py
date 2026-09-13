"""Command line entry point:  python -m cardboard_unfolder model.stl -o out/"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from .material import Material, FLUTES
from . import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="cardboard_unfolder",
        description="Unfold a low-poly mesh into laser-cuttable corrugated cardboard.")
    ap.add_argument("mesh", help="OBJ / STL / PLY / OFF")
    ap.add_argument("-o", "--out", default="out", help="output directory")

    g = ap.add_argument_group("size")
    g.add_argument("--height", type=float, help="scale so the model is this tall, mm")
    g.add_argument("--scale", type=float, help="raw scale factor instead of --height")

    g = ap.add_argument_group("material")
    g.add_argument("--flute", default="B", choices=sorted(FLUTES),
                   help="board grade (default B, ~3mm - what shipping boxes are)")
    g.add_argument("--thickness", type=float, help="override caliper, mm")
    g.add_argument("--kerf", type=float, help="override kerf, mm (measure it)")
    g.add_argument("--crush", type=float,
                   help="crush factor 0.6-1.0; lower removes less at folds")
    g.add_argument("--tab-depth", type=float, help="glue tab depth, mm")

    g = ap.add_argument_group("annotation")
    g.add_argument("--id", dest="model_id", default=None,
                   help="model identifier engraved on every island; "
                        "defaults to the mesh file's name. Keep it SHORT - it "
                        "is what decides whether a small island can carry a "
                        "label at all")
    g.add_argument("--text-height", type=float, default=None,
                   help="engraved text height, mm (default 4)")
    g.add_argument("--text-min", type=float, default=None,
                   help="never engrave smaller than this, mm (default 2); "
                        "a label that will not fit is left off instead")
    g.add_argument("--no-fold-marks", action="store_true",
                   help="skip the M / V mark beside every fold")

    g = ap.add_argument_group("machine")
    g.add_argument("--bed", default="600x400", help="bed size WxH in mm")
    g.add_argument("--margin", type=float, help="sheet margin, mm")

    g = ap.add_argument_group("unfolding")
    g.add_argument("--trials", type=int, default=250,
                   help="randomised restarts before falling back to splitting")
    g.add_argument("--seed", type=int, default=0,
                   help="change this if a net comes out badly pinched")
    g.add_argument("--no-merge", action="store_true",
                   help="keep the original triangles instead of merging coplanar faces")
    g.add_argument("--no-dxf", action="store_true")
    g.add_argument("--preview", action="store_true", help="also write PNG previews")
    a = ap.parse_args(argv)

    mat = Material.from_flute(a.flute)
    for src, dst in (("thickness", "thickness"), ("kerf", "kerf"),
                     ("crush", "crush"), ("tab_depth", "tab_depth"),
                     ("margin", "sheet_margin")):
        v = getattr(a, src, None)
        if v is not None:
            setattr(mat, dst, v)
    try:
        w, h = (float(v) for v in a.bed.lower().split("x"))
        mat.bed = (w, h)
    except ValueError:
        ap.error(f"--bed wants WxH in mm, e.g. 600x400 (got {a.bed!r})")

    # The mesh's own name is the obvious identifier and beats nothing at all,
    # but it is often long - and label length is what decides whether a small
    # island can carry one, so --id is worth using.
    mat.model_id = a.model_id if a.model_id is not None else Path(a.mesh).stem
    if a.text_height is not None:
        mat.label_height = a.text_height
    if a.text_min is not None:
        mat.label_min_height = a.text_min
    if a.no_fold_marks:
        mat.label_folds = False

    if a.height is None and a.scale is None:
        print("note: no --height given, using the mesh's own units as mm",
              file=sys.stderr)

    rep = run(a.mesh, a.out, mat, height_mm=a.height, scale=a.scale,
              trials=a.trials, seed=a.seed, merge_coplanar=not a.no_merge,
              dxf=not a.no_dxf)

    if a.preview:
        print("(preview needs matplotlib; re-run without --preview if it is missing)")
    return 1 if rep.get("oversized") or rep.get("unresolved") else 0


if __name__ == "__main__":
    raise SystemExit(main())
