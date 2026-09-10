# cardboard_unfolder

Low-poly mesh → laser-cut corrugated cardboard. M1: the headless core.

```bash
pip install trimesh networkx shapely numpy ezdxf
python -m cardboard_unfolder helmet.stl -o out/ --height 280 --flute B
```

```python
from cardboard_unfolder import run, Material
run("helmet.stl", "out/", Material.from_flute("B"), height_mm=280)
```

Writes `sheet_01.svg` / `sheet_01.dxf` per sheet plus a `manifest.json`, with a
layer per laser operation (`CUT`, `PERF_M`, `PERF_V`, `ENGRAVE`, `REF`) keyed by
both layer name and stroke colour, because LightBurn and RDWorks disagree about
which one they read.

## Pipeline

```
load  →  merge coplanar faces  →  unfold  →  fold relief  →  tabs + seam numbers
      →  kerf  →  split to bed  →  pack  →  layered SVG / DXF
```

No `bpy` anywhere. The Blender front end (M4) will import this, not the reverse.

## What it does

**Merges coplanar triangles first.** An STL of a cube is twelve triangles;
unfolded naively you get a fold line down the diagonal of every face. Merging
back into n-gons is what makes the output look designed rather than
triangulated.

**Unfolds by exact rigid motion.** For planar faces this is not an optimisation
problem — every face is rotated and translated into place from its own 3D
geometry, and edge lengths survive to floating point. Measured isometry error
across the test suite is ≤ 3e-15. This is why parameterisation tools (LSCM,
ARAP) are the wrong instrument: distortion is their premise, and here two faces
that shared a 40.00 mm edge must still share a 40.00 mm edge or the model won't
close.

**Searches for a non-overlapping net.** Randomised-restart maximum spanning tree
over the dual graph, with Shapely `STRtree` overlap rejection. Randomising the
cut-direction vector between restarts matters far more than the search
algorithm; the steepest-edge family is direction-parameterised, so it sweeps a
structurally diverse slice of tree space. Every test mesh up to 320 faces
resolves to a single island, usually on the first or second try.

**Never fails, degrades instead.** You cannot promise a single-piece net —
Dürer's conjecture is open for convex solids and non-convex counterexamples are
known. When the search budget runs out, greedy minimum set cover cuts the tree
arc breaking the most conflicting face pairs, and repeats. Worst case is one
face per island.

**Compensates for material thickness at folds.** See below.

**Splits anything larger than the bed.** Along fold edges, so the seam lands on
a crease that already needs a glue flap rather than a flat butt joint. Uses the
minimum-area *rotated* box, so a net that is too wide axis-aligned still fits on
the diagonal.

**Warns about narrow necks.** Nets routinely wrap around and rejoin through a
near-single-point pinch — topologically one piece, physically two. Eroding by
half the minimum width and counting fragments finds them before you cut.

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
in two.

Perforation has one corrugated-specific rule: the **dash period stays under a
third of the flute pitch**. At a 4 mm period on a 6.5 mm B-flute pitch,
consecutive flutes get wildly different treatment — some ties land on a glue
line (locally 3-ply solid, cracks), some over a void (hinges on nothing).

## Calibrate before you trust it

Every empirical default is a guess about *your* board and *your* machine. One
afternoon here saves a week:

| Measure | How | Default |
|---|---|---|
| `kerf` | Cut a 20 mm square. `kerf = (hole − dropout) / 2` | 0.30 mm |
| `thickness` | Calipers squeeze corrugated and read 0.2–0.5 mm low — prefer the grade table | 3.0 mm (B) |
| `crush` | Cut a strip with 90° folds at C = 0.6/0.75/0.9, fold each, keep the one that closes square | 0.75 |
| `perf_cut` / `perf_tie` | Cut a ladder of ratios, fold each 20× | 1.0 / 0.6 mm |
| Slot fit | Comb of slots from `t−0.2` to `t−0.6` | `t − 0.4` |

Design tolerance: corrugated is ±3 mm die-cut as an industry standard, and
caliper swings ±0.2–0.3 mm within one sheet. Don't design anything needing
better than ±0.5 mm.

**Fire.** Corrugated is the most flammable thing you will put in a laser — the
flutes are oxygen-filled channels that feed a flame where air assist can't
reach. Air assist on, never walk away.

## Deliberately not here yet

- **Flute-direction constraint** (`Material.lock_grain`, currently off). Creases
  perpendicular to the flutes are crisp and repeatable; parallel ones land at an
  unknown phase in the flute pitch and can wander ±P/2 — ±3.25 mm on B flute.
  Irrelevant at costume tolerances. Turn it on if a part visibly racks.
- **True polygon nesting.** Packing is bbox shelving. No-fit-polygon nesting
  (SVGnest, `python-libnest2d`) is worth roughly +20 points of material
  utilisation on net-like geometry — these nets are nothing *but* concavity.
- **Splitting a single oversized face.** Bed splitting only cuts along fold
  edges, so one face bigger than the bed still fails. Needs an in-plane cut plus
  a splice joint — for cardboard, a glued backing strip rather than an
  interlock.
- **Tab side assignment as 2-SAT.** Currently greedy with collision rejection
  and a shrink-then-skip fallback; skipped seams get taped.
- **Blender front end** (M4) — seam painting with live re-unfold.

## Tests

```bash
python -m pytest tests -q     # 87 tests
```

They assert invariants, not snapshots: exact isometry, flat area equals surface
area, every face in exactly one island, fold count equals faces minus islands,
relief width against the closed form, perf period against flute pitch, no
overlaps, nothing exceeding the bed, no collisions between packed parts, and
valid layered SVG/DXF with cuts ordered last.
