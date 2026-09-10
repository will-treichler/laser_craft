# cardboard-unfolder — context for a code session

Turns low-poly meshes into laser-cuttable corrugated cardboard. Built for
Halloween costume armour and props; the constraints are a hobby laser, free
shipping-box cardboard, and costume-level tolerance (±0.5 mm is plenty).

## Commands

```bash
pip install -e ".[dev]"                 # core + pytest + matplotlib
python -m cardboard_unfolder mesh.stl -o out/ --height 280 --flute B

python -m pytest tests -q               # 87 core invariants (~20 s)
python blender/watch.py                 # re-run core tests on save
python blender/watch.py --addon         # also the Blender suites

pip install bpy==4.5.13                 # Blender as a Python module, ~1 GB
python blender/test_addon.py            # 38 checks, real Blender API
python blender/test_dev.py              # 29 checks on the debug tooling

python blender/dev_install.py           # symlink add-on into Blender
```

## Architecture

Two layers, and the separation is load-bearing:

- `cardboard_unfolder/` — the core. **Zero `bpy` imports.** Unit-tested
  standalone. This is the escape hatch if Blender's API churn gets bad.
- `blender/cbu_blender/` — a thin add-on over it. `vendor/cardboard_unfolder/`
  is a *copy* of the core so the zip is self-contained; `watch.py` keeps it in
  sync. Edit the top-level core, never the vendored copy.

Pipeline: load → merge coplanar faces → unfold → fold relief → tabs and seam
numbers → kerf → split to bed → pack → layered SVG/DXF.

| Module | Does |
|---|---|
| `mesh.py` | Load, merge coplanar triangles into n-gons, dual graph, planarity gate |
| `unfold.py` | Rigid unfold, overlap detection, randomised search, set-cover repair, bed splitting |
| `folds.py` | Relief width from thickness and angle, perforation, kerf |
| `tabs.py` | Glue tabs with collision rejection, seam numbering |
| `layout.py` | Parts, narrow-neck detection, shelf packing |
| `export.py` | Layered SVG + DXF + manifest |
| `strokefont.py` | Single-stroke font for engraved labels |

## Things that will bite you

**Unfolding is exact, not approximate.** Planar faces unfold by rigid motion;
edge lengths survive to ~1e-15. There is no solver. Parameterisation tools
(LSCM, ARAP) are the *wrong* instrument — distortion is their premise, and two
faces that shared a 40.00 mm edge must still share one or the model won't
close. Any change that makes `test_unfold_is_exactly_isometric` fail is wrong,
not the test.

**Shapely unions need `set_precision`.** Faces are placed independently, so a
shared corner computed via two tree paths lands ~1e-14 apart. Without snapping
to a 1e-9 grid, unions silently lose ~1% of the net and split it into
fragments. Regression: `test_outline_union_loses_no_area`.

**Overlap checks must be component-aware.** After splitting, separate islands
sit on top of each other in raw coordinates — meaningless, they get packed onto
different sheets. Counting those makes a healthy net look broken.

**Never promise a single-piece net.** Dürer's conjecture is open for convex
solids and non-convex counterexamples are proven. The objective is to minimise
island count; the floor is one face per island. The code must degrade, never
fail.

**Blender: seam marks go in mesh attributes, not PropertyGroups.** Edit-mode
undo records only mesh/bmesh changes, so custom properties set in Edit Mode are
not restored by Ctrl+Z. Marks in a PropertyGroup work until the first undo,
then desync silently. Very painful to retrofit.

**Blender UVs are float32.** The net UV layer is a *view*; export comes from
the core's float64 geometry. ~9e-5 mm error at 500 mm coordinates — 90 nm
against a 0.3 mm kerf.

**Hot reload purges `__pycache__` first.** Python validates a `.pyc` from
source mtime in whole *seconds* plus size, so a same-second edit that doesn't
change the length silently runs old code.

## The material model

Relief width at a fold: `w = crush · 2 · d · tan(α/2)` where `d = t − liner/2`
(the board pivots about the surviving liner's mid-plane) and `crush` defaults
to **0.75** because corrugated squashes into the channel. That crush factor is
the one number that is empirical rather than derived — expect to tune it
against real board. Below 1.2 mm computed width the fold is just perforated;
above it a slot is removed with bridges spanning it, and those bridges are the
hinge.

Perforation dash period stays under a third of the flute pitch, or consecutive
flutes get very different treatment.

`docs/spec.html` has the full derivation, the research behind it, and the
calibration procedure.

## State of the milestones

Done: M1 (core), M4 (Blender add-on), debug tooling.

Not done, roughly in value order:

1. **True polygon nesting.** Packing is bbox shelving at ~36% sheet use. NFP
   nesting is worth ~+20 points on nets this concave. Blender's own
   `uv.pack_islands(shape_method='CONCAVE', scale=False)` already does this and
   may be enough — see `ops.CBU_OT_pack_uv`.
2. **2D seam editing.** You can move/rotate islands in the UV editor but
   re-seaming is 3D-only. PolyZamboni's `operators.py:890` is a working
   reference for Image-Editor modal plumbing.
3. **Live re-unfold** on seam change.
4. **Splitting a single oversized face.** Bed splitting only cuts along fold
   edges, so one face bigger than the bed still fails. Needs an in-plane cut
   plus a splice — for cardboard, a glued backing strip, not an interlock.
5. **Tab side assignment as 2-SAT** (currently greedy; skipped seams get taped).
6. **Flute-direction constraint.** Deliberately off (`Material.lock_grain`);
   costume tolerance doesn't need it. Turn on if a part visibly racks.

## Conventions

- Add core features to `cardboard_unfolder/` with a test first, then surface
  them in the add-on (`props.py` → `ui.py` → `ops.py`).
- Wrap expensive stages in `with cbulog.stage(T, "name"):` so they appear in
  the add-on's Timings box.
- Tests assert invariants, not snapshots. Prefer a property that must hold
  (area conserved, nothing exceeds the bed) over a golden file.
- Empirical constants live in `material.py` and are exposed in the UI; derived
  geometry is hard-coded. Keep that line clear, and label empirical values as
  such in comments.
