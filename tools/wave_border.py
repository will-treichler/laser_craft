"""A Greek wave ("running dog") border as a laser-cuttable SVG, in mm.

    python tools/wave_border.py out.svg [--waves 8] [--height 40] [--bar 5]
                                        [--stroke 4.5] [--mirror] [--png]

Black is board: a bar along the top and the bottom, a bar at each end so the
border is ONE piece (the top bar would otherwise float free of the waves),
and between the bars a row of scrolls, each a thick stroke rising from the
bottom bar and curling clockwise into a spiral. Everything white is cut out.

Built from geometry, not traced, so it scales cleanly and every stroke stays
at least `--stroke` wide - narrower than ~4 mm and corrugated board tears.
The SVG is one path, outer edge plus holes, fill-rule evenodd, sized in mm.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union
from shapely import affinity


def wave(cx: float, base: float, R: float, w: float, period: float):
    """One wave, breaking to the right: a solid disc of radius R sitting on
    the bottom bar, a rising back joining it to the trough before it, and a
    white channel `w` wide that enters the disc from the front and spirals
    clockwise inward - what is left of the disc is the curling crest. The
    channel's radius shrinks by a stroke plus a gap per turn, so the black
    arm between its turns is as wide as the channel."""
    cy = base + R
    disc = Polygon([(cx + R * math.cos(t), cy + R * math.sin(t))
                    for t in np.linspace(0, 2 * math.pi, 180, endpoint=False)])
    # the back: from the trough up to the crest, a quadratic curve
    x0 = cx - period + R + 0.5 * w
    p0, p1, p2 = np.array([x0, base]), np.array([cx - 0.6 * R, base]),         np.array([cx - 0.15 * R, cy + R])
    curve = [tuple((1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s * s * p2)
             for s in np.linspace(0, 1, 40)]
    back = Polygon(curve + [(cx, cy), (cx, base)]).buffer(0)
    # the white channel: in from the front, clockwise and inward
    pitch = 2.0 * w
    pts = [(cx + R + w, cy - 0.15 * R)]
    t0 = -0.15
    t = t0
    while True:
        # one stroke in from the disc's edge, so the crest keeps half a
        # stroke of board where the channel enters - from the edge itself it
        # tapered to nothing (31 mm2 of black under 1 mm, measured)
        r = (R - w) - pitch * (t0 - t) / (2.0 * math.pi)
        if r < 0.9 * w:
            break
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
        t -= 0.03
    channel = LineString(pts).buffer(w / 2.0, quad_segs=12)
    return back.union(disc).difference(channel)


MIN_BLACK = 3.0     # mm, narrowest board left in the part
MIN_WHITE = 2.0     # mm, narrowest gap cut out of it


def narrowest(shape, limit: float) -> float:
    """Largest width w (to 0.25 mm, up to `limit`) such that no part of
    `shape` is narrower than w - measured by morphological opening."""
    w = 0.25
    while w <= limit:
        if shape.difference(shape.buffer(-w / 2).buffer(w / 2)).area > 0.5:
            return w - 0.25
        w += 0.25
    return limit


def border(waves: int, height: float, bar: float, stroke: float,
           mirror: bool = False) -> Polygon:
    band = height - 2.0 * bar          # room between the bars
    R = (band - stroke) / 2.0          # a stroke of white under the top bar
    period = 2.7 * R
    length = waves * period + 2.0 * bar + 0.5 * stroke
    shapes = [box(0, 0, length, bar),                      # bottom bar
              box(0, height - bar, length, height),        # top bar
              box(0, 0, bar, height),                      # end caps
              box(length - bar, 0, length, height)]
    inner = box(bar, bar, length - bar, height - bar)
    for i in range(waves):
        cx = bar + period * (i + 1) - R - 0.25 * stroke
        shapes.append(wave(cx, bar, R, stroke, period).intersection(inner))
    shape = unary_union(shapes).buffer(0)
    # Smooth: no black narrower than MIN_BLACK (it would tear) and no white
    # narrower than MIN_WHITE (a slot the beam cannot clear) - a feature that
    # thin disappears rather than being cut.
    shape = shape.buffer(-MIN_BLACK / 2).buffer(MIN_BLACK / 2)
    shape = shape.buffer(MIN_WHITE / 2).buffer(-MIN_WHITE / 2)
    if mirror:
        shape = affinity.scale(shape, -1, 1, origin=(length / 2.0, 0))
    return shape


def svg_path(poly: Polygon, height: float) -> str:
    def ring(coords):
        c = np.asarray(coords)[:-1]
        return ("M " + " L ".join(f"{x:.3f},{height - y:.3f}" for x, y in c)
                + " Z")
    parts = [ring(poly.exterior.coords)]
    parts += [ring(h.coords) for h in poly.interiors]
    return " ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--waves", type=int, default=8)
    ap.add_argument("--height", type=float, default=40.0, help="mm, overall")
    ap.add_argument("--bar", type=float, default=5.0, help="mm, top/bottom bars")
    ap.add_argument("--stroke", type=float, default=3.5,
                    help="mm, scroll width (and the white channel's)")
    ap.add_argument("--mirror", action="store_true", help="waves break left")
    ap.add_argument("--png", action="store_true", help="also write a preview")
    a = ap.parse_args(argv)

    shape = border(a.waves, a.height, a.bar, a.stroke, a.mirror)
    assert shape.geom_type == "Polygon", "the border must be one piece"
    x0, y0, x1, y1 = shape.bounds
    L, H = x1 - x0, y1 - y0
    out = Path(a.out)
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{L:.3f}mm" '
        f'height="{H:.3f}mm" viewBox="0 0 {L:.3f} {H:.3f}">\n'
        f'  <path fill="#000" fill-rule="evenodd" stroke="none" '
        f'd="{svg_path(shape, H)}"/>\n</svg>\n')
    # the cut-out regions themselves - not "bounding box minus the part",
    # which counts the slivers at its rounded outer corners as gaps
    from shapely.geometry import MultiPolygon
    hole = MultiPolygon([Polygon(h.coords) for h in shape.interiors])
    print(f"{out}: {L:.1f} x {H:.1f} mm, {a.waves} waves, one piece with "
          f"{len(shape.interiors)} hole(s); measured: no board narrower than "
          f"{narrowest(shape, a.stroke):.2f} mm, no gap narrower than "
          f"{narrowest(hole, a.stroke):.2f} mm")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(L / 25, H / 25))
        ax.fill(*shape.exterior.xy, color="black")
        for h in shape.interiors:
            ax.fill(*h.xy, color="white")
        ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(out.with_suffix(".png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
