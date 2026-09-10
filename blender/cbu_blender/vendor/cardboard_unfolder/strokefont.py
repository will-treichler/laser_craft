"""A minimal single-stroke vector font for engraved part labels.

Outline fonts are the wrong tool on a laser: filled glyphs engrave 5-20x
slower than strokes and on cardboard they burn straight through. Every
serious laser workflow uses a single-stroke (Hershey-style) font. This is a
small one - digits, uppercase, and a few marks - defined as polylines on a
3 wide x 5 tall cell, which is all a part ID needs.
"""
from __future__ import annotations
import numpy as np

# Each glyph: list of polylines, coordinates on a 3x5 grid, origin bottom-left.
_G: dict[str, list[list[tuple[float, float]]]] = {
    "0": [[(0,1),(0,4),(1,5),(2,5),(3,4),(3,1),(2,0),(1,0),(0,1)]],
    "1": [[(0.5,4),(1.5,5),(1.5,0)], [(0.3,0),(2.7,0)]],
    "2": [[(0,4),(1,5),(2,5),(3,4),(3,3),(0,0),(3,0)]],
    "3": [[(0,5),(3,5),(1.5,3),(3,1.5),(2,0),(0.5,0),(0,1)]],
    "4": [[(2.2,0),(2.2,5),(0,1.6),(3,1.6)]],
    "5": [[(3,5),(0,5),(0,3),(2,3),(3,2),(3,1),(2,0),(0.5,0),(0,0.7)]],
    "6": [[(3,4.3),(2,5),(1,5),(0,4),(0,1),(1,0),(2,0),(3,1),(3,2),(2,3),(0.6,3)]],
    "7": [[(0,5),(3,5),(1.2,0)]],
    "8": [[(1,3),(0,4),(0.6,5),(2.4,5),(3,4),(2,3),(0,2),(0.5,0),(2.5,0),(3,2),(1,3)]],
    "9": [[(0,0.7),(1,0),(2,0),(3,1),(3,4),(2,5),(1,5),(0,4),(0,3),(1,2),(2.4,2)]],
    "A": [[(0,0),(1.5,5),(3,0)], [(0.6,1.8),(2.4,1.8)]],
    "B": [[(0,0),(0,5),(2.3,5),(3,4.2),(2.3,3),(0,3)], [(0,3),(2.5,3),(3,2),(2.4,0),(0,0)]],
    "C": [[(3,4),(2,5),(1,5),(0,4),(0,1),(1,0),(2,0),(3,1)]],
    "D": [[(0,0),(0,5),(2,5),(3,3.8),(3,1.2),(2,0),(0,0)]],
    "E": [[(3,5),(0,5),(0,0),(3,0)], [(0,2.7),(2.2,2.7)]],
    "F": [[(3,5),(0,5),(0,0)], [(0,2.7),(2.2,2.7)]],
    "G": [[(3,4),(2,5),(1,5),(0,4),(0,1),(1,0),(2,0),(3,1),(3,2.3),(1.6,2.3)]],
    "H": [[(0,5),(0,0)], [(3,5),(3,0)], [(0,2.7),(3,2.7)]],
    "I": [[(0.4,5),(2.6,5)], [(1.5,5),(1.5,0)], [(0.4,0),(2.6,0)]],
    "J": [[(3,5),(3,1),(2,0),(1,0),(0,1)]],
    "K": [[(0,5),(0,0)], [(3,5),(0,2.3)], [(1,3.2),(3,0)]],
    "L": [[(0,5),(0,0),(3,0)]],
    "M": [[(0,0),(0,5),(1.5,2.3),(3,5),(3,0)]],
    "N": [[(0,0),(0,5),(3,0),(3,5)]],
    "O": [[(0,1),(0,4),(1,5),(2,5),(3,4),(3,1),(2,0),(1,0),(0,1)]],
    "P": [[(0,0),(0,5),(2.3,5),(3,4.2),(2.3,2.8),(0,2.8)]],
    "Q": [[(0,1),(0,4),(1,5),(2,5),(3,4),(3,1),(2,0),(1,0),(0,1)], [(1.8,1.4),(3.2,-0.3)]],
    "R": [[(0,0),(0,5),(2.3,5),(3,4.2),(2.3,2.8),(0,2.8)], [(1.4,2.8),(3,0)]],
    "S": [[(3,4.3),(2,5),(1,5),(0,4.2),(0,3.2),(3,1.8),(3,0.8),(2,0),(1,0),(0,0.7)]],
    "T": [[(0,5),(3,5)], [(1.5,5),(1.5,0)]],
    "U": [[(0,5),(0,1),(1,0),(2,0),(3,1),(3,5)]],
    "V": [[(0,5),(1.5,0),(3,5)]],
    "W": [[(0,5),(0.6,0),(1.5,2.7),(2.4,0),(3,5)]],
    "X": [[(0,5),(3,0)], [(0,0),(3,5)]],
    "Y": [[(0,5),(1.5,2.6),(3,5)], [(1.5,2.6),(1.5,0)]],
    "Z": [[(0,5),(3,5),(0,0),(3,0)]],
    "-": [[(0.4,2.5),(2.6,2.5)]],
    ".": [[(1.3,0),(1.7,0),(1.7,0.4),(1.3,0.4),(1.3,0)]],
    "/": [[(0.2,0),(2.8,5)]],
    " ": [],
}

CELL_W, CELL_H, GAP = 3.0, 5.0, 1.2


def text_width(s: str, height: float) -> float:
    k = height / CELL_H
    n = len(s)
    return (n * CELL_W + max(0, n - 1) * GAP) * k if n else 0.0


def text_strokes(s: str, height: float = 5.0, origin=(0.0, 0.0),
                 rotation: float = 0.0, anchor: str = "center") -> list[np.ndarray]:
    """Polylines (each an (n,2) array, in mm) rendering `s`.

    anchor: 'center' centres the string on `origin`, 'start' puts the
    baseline-left corner there.
    """
    s = s.upper()
    k = height / CELL_H
    w = text_width(s, height)
    ox = -w / 2.0 if anchor == "center" else 0.0
    oy = -height / 2.0 if anchor == "center" else 0.0

    c, sn = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -sn], [sn, c]])
    o = np.asarray(origin, dtype=float)

    out, pen = [], 0.0
    for ch in s:
        for poly in _G.get(ch, _G[" "]):
            if len(poly) < 2:
                continue
            p = np.asarray(poly, dtype=float) * k
            p[:, 0] += pen + ox
            p[:, 1] += oy
            out.append(p @ R.T + o)
        pen += (CELL_W + GAP) * k
    return out
