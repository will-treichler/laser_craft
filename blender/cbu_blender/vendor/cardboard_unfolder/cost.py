"""What a net costs, so the optimiser has something to minimise.

`tools/bench.py` reports a table; this turns a table into one number, which is
a different and more dangerous job. Every weight here is a judgement about
what it is like to build one of these things out of cardboard, so each is
named, tunable, and reported alongside the total - a single score with no
breakdown is unfalsifiable, and the first thing anyone will want to know is
*which* term moved.

The ordering of the weights is the user's, from experience with the physical
job: a small island is the worst outcome by a distance, because it has to be
found, held, glued and aligned by hand; then sheet count, because that is
money and time; then seam length, necks, bed pressure, island count, and
compactness last - a ragged island nests perfectly well, and paying for
compactness with splinters is exactly the wrong trade.

Two evaluation tiers, and they answer different questions:

- `quick=True` never touches shapely. Splinters are judged by the
  equal-area circle, `2*sqrt(A/pi)`, which is a genuine UPPER bound on the
  inscribed width - no shape of area A holds a bigger circle - so a quick
  splinter is definitely a splinter, while a quick pass can miss one. That
  makes it safe to use as a filter in an inner loop and wrong to use as a
  verdict.
- the full pass unions each island and measures `layout.inscribed_width`,
  which is the honest "can I hold this" answer.

**Splinter COUNT is the wrong yardstick, and the count is what a table
reports.** Measured over eight pikachu nets: the net with the FEWEST pieces
under 15 mm (15 of them) had ten of those under 5 mm, while the net with the
most (22) had only five, the rest sitting between 8 and 14 mm - fiddly but
buildable. Ranking by count puts the unbuildable net first. That is why the
penalty is summed per island and steep rather than counted, and why
`tools/bench.py` reports a `tiny` column beside `splinters`.

**Sheets are counted, never estimated, and both ways of estimating them were
tried.** Area cannot do it: the total island area is conserved no matter where
the net is cut, so an area-based estimate is the same constant for every
candidate and ranks nothing. Shelving the islands' minimum-area RECTANGLES
does vary with the cut and is 918x faster than the real packer - and ranks
BACKWARDS where it matters. Measured over 20 nets: spearman 0.77 across
models, which is just model size, and within one model pikachu came out at
estimate [3, 4, 4, 4] against a real [3, 2, 2, 2]. Rectangles cannot see
nesting, and these nets nest. So the term is present only when a caller hands
in a real packing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .material import Material
from .unfold import Island, min_rect_size


def ring_area(r: np.ndarray) -> float:
    """Shoelace, agreeing with `Island.area` to 1.3e-11 mm2 on a pikachu net.

    The point is NOT speed - measured at 1.25x over shapely on that net, which
    is nothing. It is that `Island.area` builds a shapely Polygon per face, and
    the quick tier promises not to touch shapely at all: that promise is what
    lets the search run where shapely is unavailable or too slow to call
    thousands of times, and one stray import would quietly break it.
    """
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def island_area(isl: Island) -> float:
    return float(sum(ring_area(r) for r in isl.faces.values()))


@dataclass
class Weights:
    """How much each term is worth. Ordered by the user's priorities.

    Every term is dimensionless before it is weighted, so these numbers can be
    compared with each other and stay meaningful when the model changes scale.
    """
    splinter: float = 100.0    # islands too small to hold. Dominant, by intent.
    # Per part no sheet can hold. Worse than any number of sheets, because it
    # is not a cost at all - it is a part that cannot be cut, and a net with
    # one is not a net. It has to be weighted rather than ignored: the sheet
    # count only counts PLACED parts, so without this a candidate that makes
    # more parts too big to place reports FEWER sheets and scores better for
    # it. Nothing in the corpus has exercised that yet; the incentive is real
    # whether or not it has fired.
    oversize: float = 500.0
    sheet: float = 20.0        # per sheet
    seam: float = 6.0          # total cut length, in units of sqrt(net area)
    neck: float = 3.0          # per near-pinch
    bed: float = 5.0           # islands crowding or exceeding the sheet
    island: float = 1.0        # per island
    spread: float = 2.0        # how unequal the island areas are
    compact: float = 0.5       # ragged outlines. Last, and deliberately small.

    # The splinter penalty is 0 at `Material.splinter_width` and 1 at zero
    # width, with this exponent between. 2 is already steep - a 14 mm piece
    # scores 0.004 against a 3 mm piece's 0.64, a factor of 160 - and the
    # curve matters more than the weight, because it is what decides whether
    # the optimiser trades one unholdable sliver for several merely small ones.
    splinter_power: float = 2.0

    # An island using no more than this fraction of a sheet is free; one that
    # fills the sheet costs 1. Below half a sheet two islands can in principle
    # share, which is the distinction being drawn.
    bed_free_frac: float = 0.5


@dataclass
class Score:
    total: float
    terms: dict = field(default_factory=dict)
    widths: list = field(default_factory=list)
    splinters: int = 0

    def __lt__(self, other) -> bool:       # so `min(candidates)` reads plainly
        return self.total < other.total

    def report(self) -> str:
        parts = "  ".join(f"{k} {v:.2f}" for k, v in sorted(self.terms.items())
                          if v)
        return f"{self.total:.2f}  ({parts})"


def _rect_area(isl: Island) -> float:
    w, h = min_rect_size(np.vstack(list(isl.faces.values())))
    return float(w * h)


def splinter_penalty(width: float, splinter_width: float,
                     power: float = 2.0) -> float:
    """What one island costs for being too small to hold. 0 at the threshold,
    1 at zero width, steep in between.

    Summed rather than counted, and that is the whole point - see the module
    docstring for the eight nets that showed why.
    """
    if splinter_width <= 0 or width >= splinter_width:
        return 0.0
    return ((splinter_width - width) / splinter_width) ** power


def area_width(area: float) -> float:
    """The equal-area circle, `2*sqrt(A/pi)`: an UPPER bound on how wide a
    circle a shape of this area can hold. Cheap enough for an inner loop, and
    safe in one direction only - below the threshold it is certainly a
    splinter, above it may still be one."""
    return 2.0 * math.sqrt(max(area, 0.0) / math.pi)


def island_widths(islands: list[Island], quick: bool = False) -> list:
    """How wide a circle each island can hold, in mm.

    Quick mode returns the equal-area circle instead, which is an upper bound:
    use it to reject, never to accept.
    """
    if quick:
        return [area_width(island_area(i)) for i in islands]
    from .layout import inscribed_width
    return [inscribed_width(i.outline()) for i in islands]


def score_islands(islands: list[Island], mat: Material, sheets: int | None = None,
                  w: Weights | None = None, quick: bool = False,
                  widths: list | None = None, oversize: int = 0) -> Score:
    """One number for a whole net, plus the breakdown that justifies it.

    `sheets` is the packed sheet count when one is in hand; leave it None in
    the cheap tier and the term is simply absent, rather than estimated from
    area - see the module docstring for why an area estimate would rank
    nothing. `widths` lets a caller reuse measurements it already has.
    """
    w = w or Weights()
    if not islands:
        return Score(0.0)

    bed_w, bed_h = mat.usable_bed
    bed_area = bed_w * bed_h
    areas = np.array([island_area(i) for i in islands], dtype=float)
    total_area = float(areas.sum())
    # Seam length is reported against the net's own size, so the weight is a
    # pure number rather than a value per millimetre that has to be re-tuned
    # every time the model is scaled.
    ref = math.sqrt(total_area) if total_area > 0 else 1.0

    if widths is None:
        widths = island_widths(islands, quick=quick)

    W = mat.splinter_width
    splinter_terms = [splinter_penalty(x, W, w.splinter_power)
                      for x in widths if x < W]
    splinters = len(splinter_terms)

    seam_mm = sum(c.length for i in islands for c in i.cuts if c.seam >= 0)

    # Bed pressure, soft on purpose: the unroll should feel the sheet without
    # being stopped by it, because terminating a roll early to respect the bed
    # is how splinters get made. An island past the bed is not rejected here
    # either - split_to_fit will cut it, at a cost this term stands in for.
    bed = 0.0
    for isl in islands:
        rw, rh = min_rect_size(np.vstack(list(isl.faces.values())))
        if not ((rw <= bed_w and rh <= bed_h) or (rh <= bed_w and rw <= bed_h)):
            bed += 1.0
            continue
        fill = (rw * rh) / bed_area if bed_area > 0 else 0.0
        if fill > w.bed_free_frac:
            bed += (fill - w.bed_free_frac) / max(1e-9, 1.0 - w.bed_free_frac)

    # "Roughly equalize the island size" as the user asked for it, measured as
    # the coefficient of variation so it does not care how many islands there
    # are or how big the model is.
    mean = total_area / len(areas)
    spread = float(areas.std() / mean) if mean > 0 else 0.0

    # Compactness against the minimum-area rectangle, weighted by area so one
    # tiny ragged offcut cannot outvote the panel that matters.
    slack = 0.0
    for isl, a in zip(islands, areas):
        ra = _rect_area(isl)
        if ra > 0:
            slack += a * (1.0 - a / ra)
    compact = slack / total_area if total_area > 0 else 0.0

    necks = 0
    if not quick:
        from .layout import narrow_necks
        neck_w = max(3.0, 2.0 * mat.thickness)
        necks = sum(narrow_necks(i.outline(), neck_w) for i in islands)

    terms = {
        "splinter": w.splinter * float(sum(splinter_terms)),
        "seam": w.seam * (seam_mm / ref),
        "bed": w.bed * bed,
        "island": w.island * len(islands),
        "spread": w.spread * spread,
        "compact": w.compact * compact,
    }
    if not quick:
        terms["neck"] = w.neck * necks
    if sheets is not None:
        terms["sheet"] = w.sheet * float(sheets)
    if oversize:
        terms["oversize"] = w.oversize * float(oversize)

    return Score(total=float(sum(terms.values())), terms=terms,
                 widths=list(widths), splinters=splinters)
