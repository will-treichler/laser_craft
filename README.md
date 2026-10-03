# cardboard_unfolder

Low-poly mesh → laser-cut corrugated cardboard, for Halloween costume armour
and props: a hobby laser, free shipping-box cardboard, and costume tolerance
(±0.5 mm is plenty). Two layers: a headless Python core, and a Blender add-on
over it.

```bash
pip install -e ".[dev]"
python -m cardboard_unfolder helmet.stl -o out/ --height 280 --flute B
```

```python
from cardboard_unfolder import run, Material
run("helmet.stl", "out/", Material.from_flute("B"), height_mm=280)
```

Writes `sheet_01.svg` / `sheet_01.dxf` per sheet plus a `manifest.json`, with a
layer per laser operation (`CUT`, `PERF_M`, `PERF_V`, `ENGRAVE`, `REF`) keyed by
both layer name and stroke colour, because LightBurn and RDWorks disagree about
which one they read. Cuts are ordered last, so a part never drops before its
features are done.

`CLAUDE.md` is the engineering notebook: every design decision, what was
measured, and what bit. Read it before changing anything non-trivial.

## Pipeline

```
load → merge coplanar faces → unfold → mend → fold relief → fingers / tabs
     → seam numbers and labels → kerf → split to bed → pack → layered SVG / DXF
```

The core (`cardboard_unfolder/`) imports no `bpy` and is tested standalone. The
add-on (`blender/cbu_blender/`) carries a vendored copy of it so the zip is
self-contained; edit the top-level core, never the copy - `blender/watch.py`
syncs it.

## What it does

**Merges coplanar triangles first.** An STL of a cube is twelve triangles;
unfolded naively you get a fold line down the diagonal of every face.

**Unfolds by exact rigid motion.** Every planar face is rotated and translated
into place from its own 3D geometry, so edge lengths survive to ~1e-15 mm.
Parameterisation tools (LSCM, ARAP) are the wrong instrument: two faces that
shared a 40.00 mm edge must still share a 40.00 mm edge or the model won't
close.

**Searches for a good net, and mends what overlaps.** Randomised-restart
spanning trees over the dual graph; overlaps are cut by weighted set cover,
and then *regret* re-welds every cut that stopped being necessary (on a
pikachu model: 26 islands to 8, 12 splinters to 2). **Optimise** goes further
and ranks candidate nets by what they cost once mended - small pieces, sheets
used (measured by actually packing them), seam length.

**Never fails, degrades instead.** A single-piece net cannot be promised
(Dürer's conjecture), so the floor is one face per island.

**Splits anything larger than the bed** along fold edges, so the seam lands on
a crease rather than a flat butt joint.

**Packs true shapes.** Three packers - shelf, corner nesting, and an FFT
raster packer that tests every position at once - run on every pack and the
best wins, ranked by parts left unplaced first and sheets second. A tightening
pass then moves stragglers into gaps on earlier sheets. Tall (portrait) beds
are handled.

**Joins the seams.** Optional **interlocking fingers** - a box joint along
every cut seam that can take one, 15 mm fingers, depth one board thickness -
and optional **glue tabs** on the seams fingers could not take. Both off by
default; most cardboard is taped.

**Labels everything.** Seam numbers on both halves of every cut, M / V beside
every fold, and a model id plus island number on each piece, in a single-stroke
engraving font. A label that cannot fit at a legible 2 mm is left off rather
than shrunk into a smudge.

**Warns about narrow necks** - nets that wrap round and rejoin through a
near-pinch are one piece topologically and two physically.

## The Blender add-on

Install for development with `python blender/dev_install.py` (symlinks the
add-on into Blender's add-ons folder). Blender loads add-on code once at start,
so after pulling changes restart Blender or press **Reload** in the add-on's
Debug panel, then **Unfold** again.

In the sidebar's **Cardboard** tab:

- **Unfold** is the quick solve; **Optimise** is the slower search, with a
  progress bar, and ESC stops it with the best net so far.
- **Edge Constraints** - mark edges as forced cut or forced fold (red / blue),
  and lock marks so Optimise may not change them.
- **Material** - flute, thickness, kerf, crush.
- **Seams** - interlocking fingers (width; depth from the board unless
  overridden) and glue tabs.
- **Folds**, **Annotation**, **Machine & Size** (bed, margin, part gap, model
  height), **Solver**, **Debug**.

The net is written to a `CBU_Net` UV layer laid out exactly as packed, so the
UV editor is the 2D view. Its **Cardboard Net** panel draws each sheet with a
millimetre grid and a ruler reading from the sheet's corner, the real cut
outline (fingers and tabs included), the engraving, and your edge marks. From
there you can cut or weld a selected edge, re-pack, nest with Blender's packer,
and export.

## The fold geometry

Two panels of thickness `t` meeting at interior angle θ must lose material at
the crease or they cannot close. With bend angle `α = 180° − θ`:

```
w = 2 · d · tan(α/2)
```

Derived twice — once for a true V-groove, once for the straight-walled slot a
laser actually cuts (the binding constraint is the top corner clearing the
hinge, `(w/2)·cos(α/2) − d·sin(α/2) ≥ 0`, which gives the same answer). It is
also exactly the sheet-metal bend deduction at zero inside radius, and it
matches the "spacing substantially twice the thickness" specified for 90°
V-grooved corrugated in US 5,427,309.

Two corrections theory alone doesn't give you:

- **`d` is not `t`.** With the far liner surviving as the hinge, the board
  pivots about that liner's mid-plane: `d = t − liner/2`, about 5% narrower.
- **Corrugated crushes.** Flat crush resistance is 0.1–0.5 MPa against
  plywood's 5–10 MPa, so the board squashes into the channel and the
  theoretical width is conservative. Folding-carton practice uses 1.5t channels
  where theory demands 2.0t. Hence `crush`, default **0.75**. This one is
  empirical — expect to tune it.

Below `relief_threshold` (1.2 mm) the fold is just perforated and the board is
left to crush. Above it, a slot of width `w` is removed and **bridges** span it
— those bridges are the hinge, and without them a 90° fold would cut the part
in two. Or set a fixed cut width instead of the adaptive one.

## Interlocking fingers

Along a seam, alternate segments: one half keeps its full length to the edge
(the finger), the other is cut back by the depth (the notch). The mesh edge is
the outer corner, so at a right angle the panel kept to the edge already
reaches the far face of the other board - the finger adds nothing, and the
edge swings by exactly one board thickness.

At other angles the overlap that needs clearing is not one thickness: with
square-cut ends it is `t·sin θ` for an obtuse corner (1.5 mm at 150° on 3 mm
board) and `t·cot(θ/2)` for an acute one (5.2 mm at 60°), and nothing at a
concave seam. Every seam currently gets one depth - the thickness, or the
override.

## Calibrate before you trust it

Every empirical default is a guess about *your* board and *your* machine. One
afternoon here saves a week:

| Measure | How | Default |
|---|---|---|
| `kerf` | Cut a 20 mm square. `kerf = (hole − dropout) / 2` | 0.30 mm |
| `thickness` | Calipers squeeze corrugated and read 0.2–0.5 mm low — prefer the grade table | 3.0 mm (B) |
| `crush` | Cut a strip with 90° folds at C = 0.6/0.75/0.9, fold each, keep the one that closes square | 0.75 |
| `perf_cut` / `connect_ratio` | Cut a ladder of ratios, fold each 20× | 1.0 mm / 0.5 |
| Finger fit | Cut a test seam at depth `t`, `t − 0.3`, `t + 0.3` | `t` |

Design tolerance: corrugated is ±3 mm die-cut as an industry standard, and
caliper swings ±0.2–0.3 mm within one sheet. Don't design anything needing
better than ±0.5 mm.

**Fire.** Corrugated is the most flammable thing you will put in a laser — the
flutes are oxygen-filled channels that feed a flame where air assist can't
reach. Air assist on, never walk away.

## Not here yet

- **Splitting a single oversized face.** Bed splitting only cuts along fold
  edges, so one face bigger than the bed still fails. Needs an in-plane cut plus
  a splice — for cardboard, a glued backing strip.
- **Finger depth per seam angle** (see above), and a minimum depth below which
  a nearly flat seam is taped instead.
- **Better packing sequences** - annealing over order and rotation; the packer
  is greedy bottom-left today.
- **Modal drag-to-cut seam editing** in the UV editor (click-to-pick works).
- **Tab side assignment as 2-SAT** (greedy today; skipped seams get taped).
- **Flute-direction constraint** (`Material.lock_grain`, off). Irrelevant at
  costume tolerance; turn it on if a part visibly racks.

## Tests

```bash
python -m pytest tests -q                       # core invariants, ~3 min
blender --background --factory-startup --python blender/test_addon.py
blender --background --factory-startup --python blender/test_dev.py
python tools/bench.py --baseline tools/baseline.json   # the scoreboard
```

They assert invariants, not snapshots: exact isometry, no mirrored faces,
every face in exactly one island, no overlaps, nothing exceeding the bed, no
collisions between packed parts, fingers that never stand proud of the model,
labels that never leave their part, and valid layered SVG/DXF with cuts last.
