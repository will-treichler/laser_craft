"""Command line entry point:  python -m cardboard_unfolder model.stl -o out/"""
from __future__ import annotations
import argparse
from pathlib import Path

from .tabs import short_id
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

    g = ap.add_argument_group("interlocking fingers")
    g.add_argument("--fingers", action="store_true",
                   help="cut a box joint along every seam that can take one, "
                        "so the panels lock together during assembly")
    g.add_argument("--finger-width", type=float, default=None,
                   help="mm per finger (default 15)")
    g.add_argument("--finger-clearance", type=float, default=None,
                   help="play between a finger and its notch, mm per side "
                        "(default 0.05)")
    g.add_argument("--finger-depth", type=float, default=None,
                   help="mm; overrides the default, which is the board "
                        "thickness")

    g = ap.add_argument_group("annotation")
    g.add_argument("--id", dest="model_id", default=None,
                   help="model identifier engraved on every island; "
                        "defaults to the mesh file's name. Keep it SHORT - it "
                        "is what decides whether a small island can carry a "
                        "label at all")
    g.add_argument("--text-height", type=float, default=None,
                   help="engraved text height, mm (default 3)")
    g.add_argument("--text-min", type=float, default=None,
                   help="never engrave smaller than this, mm (default 2); "
                        "a label that will not fit is left off instead")
    g.add_argument("--no-fold-marks", action="store_true",
                   help="skip the M / V mark beside every fold")

    g = ap.add_argument_group("machine")
    g.add_argument("--bed", default="600x400", help="bed size WxH in mm")
    g.add_argument("--margin", type=float, help="sheet margin, mm")
    g.add_argument("--cut", metavar="S,F[,N]",
                   help="cut power (GRBL S), speed mm/min, passes "
                        "(default 1000,500,1)")
    g.add_argument("--score", metavar="S,F[,N]",
                   help="burn the fold lines at this power, speed, passes "
                        "(off unless given; e.g. 250,1000 - untested as a "
                        "score)")
    g.add_argument("--engrave", metavar="S,F[,N]",
                   help="label power, speed, passes (default 250,1000,1)")
    g.add_argument("--laser-max", type=float,
                   help="GRBL $30, the S value of full power (default 1000)")
    g.add_argument("--air", action="store_true",
                   help="air assist on (M8/M9) for cuts and scores")
    g.add_argument("--no-gcode", action="store_true",
                   help="skip the per-sheet .nc files")

    g = ap.add_argument_group("unfolding")
    g.add_argument("--trials", type=int, default=250,
                   help="randomised restarts before falling back to splitting")
    g.add_argument("--seed", type=int, default=0,
                   help="change this if a net comes out badly pinched")
    g.add_argument("--no-merge", action="store_true",
                   help="keep the original triangles instead of merging coplanar faces")
    g.add_argument("--flat-tol", type=float, default=None,
                   help="mm: a separate piece within this of one plane is cut "
                        "as the flat shape it is (default 3; negative = never)")
    g.add_argument("--min-bend", type=float, default=None,
                   help="degrees: folds bending less are left uncut (default 1)")
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
    mat.model_id = (a.model_id if a.model_id is not None
                    else short_id(Path(a.mesh).stem))
    if a.text_height is not None:
        mat.label_height = a.text_height
    if a.text_min is not None:
        mat.label_min_height = a.text_min
    if a.no_fold_marks:
        mat.label_folds = False
    for op in ("cut", "score", "engrave"):
        v = getattr(a, op)
        if v is None:
            continue
        try:
            nums = [float(x) for x in v.split(",")]
            assert len(nums) in (2, 3)
        except (ValueError, AssertionError):
            ap.error(f"--{op} wants power,speed[,passes], e.g. 1000,500,2 "
                     f"(got {v!r})")
        if op == "score":
            mat.score_folds = True
        setattr(mat, f"{op}_power", nums[0])
        setattr(mat, f"{op}_speed", nums[1])
        if len(nums) == 3:
            setattr(mat, f"{op}_passes", int(nums[2]))
    if a.laser_max is not None:
        mat.laser_max = a.laser_max
    mat.air_assist = a.air
    if a.flat_tol is not None:
        mat.flat_tol = a.flat_tol if a.flat_tol >= 0 else None
    if a.min_bend is not None:
        mat.min_bend = a.min_bend
    mat.use_fingers = a.fingers
    if a.finger_width is not None:
        mat.finger_width = a.finger_width
    if a.finger_clearance is not None:
        mat.finger_clearance = a.finger_clearance
    if a.finger_depth is not None:
        mat.finger_depth_override = True
        mat.finger_depth = a.finger_depth

    if a.height is None and a.scale is None:
        print("note: no --height given, using the mesh's own units as mm",
              file=sys.stderr)

    rep = run(a.mesh, a.out, mat, height_mm=a.height, scale=a.scale,
              trials=a.trials, seed=a.seed, merge_coplanar=not a.no_merge,
              dxf=not a.no_dxf, gcode=not a.no_gcode)

    if a.preview:
        print("(preview needs matplotlib; re-run without --preview if it is missing)")
    return 1 if rep.get("oversized") or rep.get("unresolved") else 0


if __name__ == "__main__":
    raise SystemExit(main())
