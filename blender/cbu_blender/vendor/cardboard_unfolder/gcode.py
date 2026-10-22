"""GRBL G-code, one file per sheet, for LaserGRBL or any GRBL 1.1 laser.

LaserGRBL loads an SVG with no layer management - every path at one power
and speed - so the layered SVG would cut the fold scores through, cut out the
labels and cut the bed outline. Here each operation carries its own power,
speed and passes, in the order the board needs them:

- ENGRAVE first, while every part is still held by the sheet;
- SCORE (the fold perforations) next;
- CUT last, and within each part its holes - relief slots, corner holes -
  before its outline, so nothing drops out before it is finished.

Coordinates are the sheet's own millimetres, unflipped: origin at the front
left of the bed, X to the right, Y away from you - which is how the packer
already lays a sheet out. Passes live in the file, so LaserGRBL's own passes
counter stays at 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .material import Material
from .export import _rings, _lines

OPS = ("ENGRAVE", "SCORE", "CUT")


@dataclass
class Op:
    name: str
    power: float            # S value, 0..laser_max
    speed: float            # mm/min
    passes: int
    air: bool               # air assist during this operation


def ops_from(mat: Material) -> dict:
    """The three operations as the material sets them, checked."""
    out = {
        "ENGRAVE": Op("ENGRAVE", mat.engrave_power, mat.engrave_speed,
                      mat.engrave_passes, mat.engrave_air),
        "SCORE": Op("SCORE", mat.score_power, mat.score_speed,
                    mat.score_passes, mat.score_air),
        "CUT": Op("CUT", mat.cut_power, mat.cut_speed, mat.cut_passes,
                  mat.cut_air),
    }
    for op in out.values():
        if not 0 <= op.power <= mat.laser_max:
            raise ValueError(f"{op.name} power S{op.power:g} is outside "
                             f"0..{mat.laser_max:g} (GRBL $30)")
        if op.speed <= 0:
            raise ValueError(f"{op.name} speed must be above 0 mm/min")
        if op.passes < 0:
            raise ValueError(f"{op.name} passes cannot be negative")
    return out


def _clean(pts) -> np.ndarray:
    """Drop repeated points; they are zero-length moves."""
    pts = np.asarray(pts, dtype=float)[:, :2]
    if len(pts) < 2:
        return pts
    keep = np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)]
    return pts[keep]


def _start_ring_near(ring: np.ndarray, at) -> np.ndarray:
    """A closed ring re-started at its vertex nearest `at` - the same ring,
    a shorter travel move to it."""
    body = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    k = int(np.argmin(np.hypot(*(body - at).T)))
    body = np.roll(body, -k, axis=0)
    return np.vstack([body, body[:1]])


def _order_open(paths: list, at) -> tuple[list, np.ndarray]:
    """Nearest-neighbour order over open polylines, each run either way."""
    left = list(paths)
    out = []
    while left:
        best, flip, bd = 0, False, float("inf")
        for i, p in enumerate(left):
            for f, end in ((False, p[0]), (True, p[-1])):
                d = float(np.hypot(*(end - at)))
                if d < bd:
                    best, flip, bd = i, f, d
        p = left.pop(best)
        p = p[::-1] if flip else p
        out.append(p)
        at = p[-1]
    return out, at


def sheet_plan(parts, mat: Material, sheet: int, home=(0.0, 0.0)) -> list:
    """What the laser burns on one sheet, in order: [(op, [paths])] with each
    pass of an operation already repeated. Paths are (n, 2) arrays in sheet
    mm; a closed ring repeats its first point at the end."""
    ops = ops_from(mat)
    mine = [p for p in parts if p.sheet == sheet]
    at = np.asarray(home, dtype=float)
    plan = []

    for name, geoms in (("ENGRAVE", [p.engrave for p in mine]),
                        ("SCORE", [g for p in mine for g in (p.perf_v, p.perf_m)])):
        paths = [c for g in geoms for l in _lines(g)
                 if len(c := _clean(l)) >= 2]
        op = ops[name]
        if not paths or op.passes == 0 or (name == "SCORE"
                                           and not mat.score_folds):
            continue
        ordered, at = _order_open(paths, at)
        plan.append((op, ordered * op.passes))

    op = ops["CUT"]
    if op.passes:
        # Part by part, nearest first. Within a part every pass of every hole
        # comes before the first pass of the outline: if the outline cuts
        # through early, the part drops and the holes are never finished.
        todo = []
        for p in mine:
            geoms = p.cut.geoms if hasattr(p.cut, "geoms") else [p.cut]
            for g in geoms:
                if g.is_empty:
                    continue
                holes = [_clean(r.coords) for r in g.interiors]
                todo.append(([h for h in holes if len(h) >= 3],
                             _clean(g.exterior.coords)))
        cut = []
        while todo:
            k = min(range(len(todo)), key=lambda i: float(np.min(np.hypot(
                *(todo[i][1] - at).T))))
            holes, outline = todo.pop(k)
            ordered = []
            rest = list(holes)
            while rest:
                j = min(range(len(rest)), key=lambda i: float(np.min(np.hypot(
                    *(rest[i] - at).T))))
                h = _start_ring_near(rest.pop(j), at)
                ordered.append(h)
                at = h[-1]
            cut += ordered * op.passes
            o = _start_ring_near(outline, at)
            cut += [o] * op.passes
            at = o[-1]
        if cut:
            plan.append((op, cut))
    return plan


def plan_stats(plan) -> dict:
    """Burned length per operation and an estimate of the laser-on time."""
    out = {}
    for op, paths in plan:
        L = sum(float(np.sum(np.hypot(*np.diff(p, axis=0).T))) for p in paths)
        out[op.name] = (L, 60.0 * L / op.speed)
    return out


def _f(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def write_gcode(path, parts, mat: Material, sheet: int, sheets: int = 1) -> dict:
    """One sheet as GRBL G-code. Returns plan_stats for the sheet.

    Refuses rather than writes a file that would run off the bed: every point
    is checked against the bed the parts were packed for.
    """
    W, H = mat.bed
    plan = sheet_plan(parts, mat, sheet)
    for _op, paths in plan:
        for p in paths:
            if (p[:, 0].min() < -1e-6 or p[:, 1].min() < -1e-6
                    or p[:, 0].max() > W + 1e-6 or p[:, 1].max() > H + 1e-6):
                raise ValueError(f"sheet {sheet + 1}: a path leaves the "
                                 f"{W:g} x {H:g} mm bed")
    stats = plan_stats(plan)
    ops = ops_from(mat)
    on = "M4" if mat.dynamic_power else "M3"
    n_parts = sum(1 for p in parts if p.sheet == sheet)

    out = [f"; cardboard-unfolder - sheet {sheet + 1} of {sheets}, "
           f"{n_parts} part(s), bed {W:g} x {H:g} mm",
           "; origin front-left, X right, Y away from you, millimetres",
           f"; power S0..{mat.laser_max:g} ($30), {on} "
           f"({'dynamic' if mat.dynamic_power else 'constant'} power)"]
    for name in OPS:
        op = ops[name]
        L, t = stats.get(name, (0.0, 0.0))
        if name == "SCORE" and not mat.score_folds:
            out.append("; SCORE   off - fold lines are not burned")
            continue
        out.append(f"; {name:<7} S{op.power:g} F{op.speed:g} x{op.passes}"
                   f"{' air' if mat.air_assist and op.air else ''}"
                   f" - {L / 1000:.2f} m, ~{t / 60:.1f} min")
    out.append("; passes are in the file: leave LaserGRBL's passes counter at 1")
    # G0 travels unlit only in laser mode; without it the beam stays on
    # between paths and burns a line across the sheet.
    out.append("; needs GRBL laser mode ($32=1): G0 travel is unlit only there")
    out += ["G21", "G90", "M5"]
    air_now = False
    for op, paths in plan:
        out.append(f"; --- {op.name}")
        want_air = mat.air_assist and op.air
        if want_air != air_now:
            out.append(mat.air_on if want_air else mat.air_off)
            air_now = want_air
        out.append(f"{on} S{op.power:g}")
        first = True
        for p in paths:
            out.append(f"G0 X{_f(p[0, 0])} Y{_f(p[0, 1])}")
            for x, y in p[1:]:
                if first:
                    out.append(f"G1 X{_f(x)} Y{_f(y)} F{op.speed:g}")
                    first = False
                else:
                    out.append(f"G1 X{_f(x)} Y{_f(y)}")
    out.append("M5")
    if air_now:
        out.append(mat.air_off)
    out.append("G0 X0 Y0")
    Path(path).write_text("\n".join(out) + "\n")
    return stats
