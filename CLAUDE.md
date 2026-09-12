# cardboard-unfolder — context for a code session

Turns low-poly meshes into laser-cuttable corrugated cardboard. Built for
Halloween costume armour and props; the constraints are a hobby laser, free
shipping-box cardboard, and costume-level tolerance (±0.5 mm is plenty).

## Commands

```bash
pip install -e ".[dev]"                 # core + pytest + matplotlib
python -m cardboard_unfolder mesh.stl -o out/ --height 280 --flute B

python -m pytest tests -q               # 115 core invariants (~30 s)
python blender/watch.py                 # re-run core tests on save
python blender/watch.py --addon         # also the Blender suites

pip install bpy==4.5.13                 # Blender as a Python module, ~1 GB
python blender/test_addon.py            # 131 checks, real Blender API
blender --background --factory-startup --python blender/test_addon.py
                                        # same suite against an installed
                                        # Blender, no 1 GB bpy wheel needed
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

**Don't trust GEOS' `minimum_rotated_rectangle`.** It returned a degenerate
zero-width sliver for a cylinder net whose hull was 2.3e5 mm^2 - but only at
the net's natural offset a few hundred mm from the origin; the same cloud
centred on the origin came back correct. `fits()` therefore said a 687 x 436 mm
island fitted a 559 x 359 mm bed, bed splitting stopped a cut early, and an
uncuttable part reached the packer. `unfold.min_area_rect` is our own rotating
calipers over our own monotone-chain hull, and `layout.orient_min_bbox` goes
through it too. Regressions: `test_min_rect_area_is_invariant_under_translation`,
`test_split_to_fit_splits_a_net_that_sits_off_the_origin`.

**Piece count is decided by the tree, not by where you cut it.** Bed
splitting grows each piece until the bed stops it and only then cuts
(`_grow_partition`), which is provably *maximal*: after a cut the child piece
is final while the parent only grows, and a superset of something that did not
fit cannot fit. So no two finished pieces could have been left joined -
`test_bed_split_is_maximal`. Rewriting the splitter that way changed no piece
counts at all, because the old bisection already landed on maximal partitions.

The leverage was upstream. `search` used to return the *first* overlap-free
tree, and overlap-free trees are plentiful - what separates them is what they
cost after bed splitting, which varies by a third on real parts. It now scores
up to `size_candidates` clean trees by piece count and keeps the cheapest:
lowpoly_dome 6 -> 4 pieces, pauldron 8 -> 6. The first clean tree is still the
first candidate, so scoring can only tie or win
(`test_scored_tree_search_never_does_worse_than_the_first_clean_tree`). It
costs a bed split per candidate, hence the cap.

Anything measuring one piece against another must use the *pre-split* `pos`.
`split_to_fit` re-roots every component afterwards, so two pieces' coordinates
then live in different frames - stack them and they appear to overlap at the
origin, and any merge looks like it fits. This produced a convincing but
entirely false "NOT MAXIMAL" result while developing.

**Packing is not the bottleneck you would expect - part *shape* is.** On a
pauldron split to fit the bed, every island's bounding box is 65-80% of the
bed's while its area is only 32-56%. The two largest need 1053 mm of a 584 mm
bed side by side and 670 mm of 384 mm stacked, so **no two parts can share a
sheet at all** and utilisation is just the mean part area. Nesting cannot help
there; it helps a lot once parts are smaller (pauldron split to 70% of the
bed: 9 sheets shelved, 5 nested).

So fewer islands is not free. Splitting harder gives smaller, denser-packing
pieces: lowpoly_dome goes 4 islands / 4 sheets / 38% to 15 islands / 3 sheets
/ 54%. That is the real trade the optimiser has to make, and it is
model-dependent - drum12 is best left at its 4 islands, where splitting more
costs sheets.

`pack()` runs both `shelf_pack` and `nest` and keeps the winner, because
neither dominates: nesting lost 5 sheets to 10 on drum12 split to 50%, where
its greedy bottom-left placement paints itself into a corner. Regression:
`test_pack_never_loses_to_either_strategy`.

Two performance traps in that nester. Buffer the candidate once per rotation,
never inside the position loop - buffering is the expensive call and there are
hundreds of positions. And offer candidate positions a full gap *past* a
neighbour's edge: at the edge itself the two half-gap buffers overlap by the
whole gap, so every candidate is rejected and parts silently never share a
sheet. Tree scoring is also budgeted (`size_candidates`, and few split trials
while scoring); at 12 candidates x 24 trials the core suite ran 158 s for
identical piece counts.

**Overlap checks must be component-aware.** After splitting, separate islands
sit on top of each other in raw coordinates — meaningless, they get packed onto
different sheets. Counting those makes a healthy net look broken.

**Never promise a single-piece net.** Dürer's conjecture is open for convex
solids and non-convex counterexamples are proven. The objective is to minimise
island count; the floor is one face per island. The code must degrade, never
fail.

**`pip install` from inside Blender lies about succeeding.** On a stock
Windows install Blender's own site-packages is under Program Files and is not
writable without admin, so pip silently falls back to a `--user` install and
*still exits 0*. Blender sets `site.ENABLE_USER_SITE = False`, so that
directory is never on `sys.path` - the add-on reports "installed" and the
package is still missing after a restart. `deps._install` therefore passes an
explicit `--target` of `scripts/addons/modules` (user-writable, always on
`sys.path`). Any new install path needs the same treatment.

**Keep third-party imports out of the add-on's import path.** Blender ships
numpy and nothing else, so `deps.py` offers a one-click install - but that
panel only draws if `register()` survived. An eager `import shapely` (or
networkx, or trimesh) anywhere reachable from `cbu_blender/__init__.py` raises
ModuleNotFoundError during registration and takes the whole add-on down,
including the UI that would have fixed it. `bridge.py` therefore imports the
core inside `polymesh_from_object`, and `mesh.py` imports trimesh inside its
two file loaders - the add-on builds meshes from bmesh and never needs it.
Check with `blender --background --factory-startup --python-expr "import
cbu_blender; cbu_blender.register()"` in a Blender whose Python is bare.

**The overlay and the export cache go stale together.** `overlay._batches`
holds vertex positions copied at unfold time and `ops._LAST` holds the parts
Export writes, so after a mesh edit the first paints islands over geometry
that has moved and the second would cut a net that no longer matches the
object. `_on_depsgraph` drops both, the same pairing `_on_load` uses.

The trap is *what counts as an edit*. Marking a seam writes an edge attribute
and the depsgraph reports that as `is_updated_geometry`, so clearing on that
flag alone wipes the island colours at the exact moment you are marking
against them. `overlay.geometry_changed` compares element counts and vertex
positions - which marking does not touch. Regression:
`marking a seam does NOT count as modifying the object`.

**GPU batches need vec3 positions even for 2D draws.** The builtin
`FLAT_COLOR` / `POLYLINE_FLAT_COLOR` shaders declare `pos` as vec3. Handing
`batch_for_shader` 2-tuples for a flat overlay is a vertex-buffer format
mismatch that *crashes Blender outright* rather than raising, so no try/except
around the draw can save you and the failure has to be asserted on the data
instead (`bed vertices are vec3`). The bed overlay emits `(x, y, 0.0)`.

**An operator must not unregister its own class mid-execute.** `reload_all()`
unregisters every class in the add-on, `CBU_OT_reload` included, so the
`self.report(...)` that followed it dereferenced a freed RNA type - an instant
segfault whose crash log points at `_bpy_types.__getattribute__`. `report()`
also tags the UI for redraw, which ran the just-re-registered draw handlers at
the same moment, so the two faults compounded. `CBU_OT_reload.execute` now
defers the whole reload to `bpy.app.timers` and returns immediately. Note this
is invisible to `test_dev.py`, which calls `reload_all()` directly; the
operator path is covered in `test_addon.py` against `execute.__code__.co_names`,
because a comment mentioning `self.report` would fool a source-text search.

**A forced cut slits the net; it does not necessarily split an island.**
The dual graph has cycles, so removing one arc leaves the two faces connected
through another path. Asserting "cut separates the two faces into different
islands" passes on a cube and fails on a sphere - the real invariant is that
the two faces disagree about where the edge sits in the net
(`cutting an edge slits the net along it`).

That is also why `cbu.edit_seam` has two cut modes. Marking the edge and
re-unfolding lets the solver rebuild the whole tree, so the freed faces
usually reattach to a *different* island and the region count does not move,
while the rest of the layout jumps around. `separate=True` (the default)
instead calls `unfold.split_island`, which removes that one fold in place and
detaches exactly the piece on its far side. An island's folds form a spanning
tree, so every fold is a bridge and removing one always yields exactly two
pieces; the spent fold becomes a matched pair of cuts sharing a new seam id,
which is what pairs a glue tab with its partner.

**In the UV editor, `edge.select` is the wrong selection.** UV selection lives
per loop and, with UV Sync Selection off, is entirely separate from the mesh
selection - so reading `edge.select` from a 2D-view operator acts on stale
edges or none at all. `bridge.uv_selected_edges` reads the uv layer's
`select_edge`, and `ops.selected_edges` picks whichever source matches the
view the user is working in.

**`cbu.repack` rebuilds parts from the islands.** `Part.xform` maps island
space to the sheet, so re-packing already-packed parts would compose two
placements and put the net somewhere it is not. Regression:
`re-packing twice is stable`.

**Blender: seam marks go in mesh attributes, not PropertyGroups.** Edit-mode
undo records only mesh/bmesh changes, so custom properties set in Edit Mode are
not restored by Ctrl+Z. Marks in a PropertyGroup work until the first undo,
then desync silently. Very painful to retrofit.

**The net UV layer is scaled by a power of two, not written in mm.** The UV
editor is built around the unit square and will not zoom out far enough to
frame a net in raw millimetres - a 6-island cube net once ran 12650 x 2000 mm
and simply could not be seen. `bridge.uv_scale` returns the largest power of
two that fits one sheet inside 0-1 (600 x 400 mm -> 1/1024, so a sheet draws
0.586 x 0.391). Two things about that factor matter:

- **A power of two is exact.** Scaling is a pure exponent shift, so every edge
  length keeps its mantissa bit for bit and the unfold stays isometric. A snug
  factor like 1/600 would round, and float32 precision is *relative*, so the
  ~9e-5 mm storage error is unchanged by the scaling. Regression:
  `scaling by it round-trips bit for bit`.
- **It depends only on the bed, never the model.** Normalising each net to 0-1
  would make two models incomparable and the bed reference meaningless.

Anything else living in UV space must be scaled the same way - `pack_islands`
takes its margin in UV units, and the bed overlay draws a sheet at
`bed * uv_scale`. `bridge._shelf_offsets` lays islands out in rows one usable
sheet wide (in mm, before scaling) so the rows line up with that reference;
rows widen to fit an island bigger than a sheet, since bed splitting cannot
cut one of those down anyway (milestone 4). The bed itself is
`overlay.bed_segments`, toggled by `show_bed`.

**Anchor the sheet grid at the origin, never at the net's bounding box.** The
net starts at the margin, so tiling sheets from the net's min corner shifts
every sheet - and its inner margin line - by exactly one margin. Parts sitting
correctly at x=8 then draw 8 mm *outside* a line drawn at 16, and a layout the
packer got right reads as ignoring the margin entirely. On a pikachu net the
grid spanned x[8.1, 308.1] against a 300 mm bed. Nothing measurable was wrong:
parts 0.00 mm past the margin, the net -0.15 mm, the cut outline 0.00 mm - only
the reference drawn underneath them. `overlay.bed_grid` is that decision on its
own, so it can be asserted (`the sheet grid starts at the origin, not the
net`). When the 2D view and the numbers disagree, suspect the view.

**A bent face must be split in the MESH, not in the core.** `source_faces`
has to stay a 1:1 partition of the caller's polygons, because `write_net_uv`
writes whole polygons through it. Letting `PolyMesh.from_rings` split a bent
quad leaves that one Blender polygon claimed by two core faces with two
different flat placements - the second write wins and tears the quad away from
its neighbours by up to 11 mm, which reads as extra UV islands. So the add-on
passes `planar_tol=None` and offers `cbu.split_nonplanar`, which triangulates
only the bent faces in the mesh itself; the coplanar merge puts genuinely flat
triangles back together, so no panel gains a fold line down its diagonal. The
core keeps its own splitting for the STL path, where nothing maps back to a
caller's polygons. Regression: `source_faces is a 1:1 partition`.

Welding cannot rescue a bent face either. `WELD_TOL` is 0.1 mm and absorbs the
few-ulp path disagreement plus the residual bow of a face flat enough to keep;
a genuinely bent quad disagrees by ~0.8 mm, past the +/-0.5 mm this is built
for, and hiding that would be lying about the cut. Note the tolerance must
cluster by *distance*: snapping to a WELD_TOL grid looks equivalent and is
not, because two corners 0.056 mm apart still land either side of a 0.1 mm
grid line better than half the time.

**The net UV layout is the packing, not a separate layout.** `write_net_uv`
takes the `Part`s and places each island exactly where `pack()` put it, with
sheets stacked upward - so the bed reference drawn underneath is the truth.
Before this the rows were laid out independently of the sheets and islands
straddled the divisions and sat in margins that `pack()` had kept clear.
`Part.xform` tracks the packing transform for this;
`test_part_xform_reproduces_the_packing` asserts it still equals the geometry,
because a transform that drifts makes the 2D view lie about what will be cut.

**`uv_layers[...].data` is empty in Edit Mode.** The live data is in the
BMesh, so reading or writing the mesh datablock there silently does nothing -
`len(uv.data) == 0` while `len(mesh.loops) == 66`. `CBU_OT_unfold` already
drops to Object Mode and restores; any new code touching mesh data must do the
same, and any *diagnostic* must too or it will report a perfectly good net as
missing.

**The UV net is the islands; the cut is the parts.** Glue tabs are generated
in the core and exist only in the exported parts - there is no mesh geometry
for them - so a 2D view drawn from UVs alone is smaller than what the laser
follows. Measured on a real design: islands 51.1% of a sheet, parts 62.1%,
**tabs and kerf add 21% on top**, and the two largest islands stack in 278 mm
bare but 289.6 mm with tabs against a 284 mm bed. Someone arranging islands by
eye is optimising the wrong shape and will conclude the packer is broken.
`overlay.cut_segments` therefore draws the real outline over the net
(`show_cut`), fed by `ops.publish_cuts` from every path that produces parts.

That also reframes packing. Before blaming the nester, check what is being
nested: no algorithm recovers 6 mm that the tabs put there. `tab_depth`
defaults to 12 mm, which is a lot of flap on a 100-200 mm part - at 8 mm the
same design becomes geometrically stackable. The deeper fix is to grow tabs
into the gaps *after* nesting rather than baking them into the outline before
it, which takes a fifth of the area out of the packing problem entirely.

For the record on packing skill, measured on that design (4 parts, 64% of one
bed): Blender's `pack_islands` reaches 318 x 319 against a 284 x 284 bed;
raster bottom-left with 24 rotations x all 24 orders fails; 4000 randomised
restarts with escape jitter fails. Greedy bottom-left plateaus regardless of
how finely it is swept - past it needs raster occupancy with FFT collision
(fast: ~40 ms a query) driving annealing over sequence and orientation, not a
finer sweep.

**Nest the parts, not the island UVs.** `cbu.pack_uv` builds a throwaway
mesh with one flat face per `Part` outline, packs *that* with Blender's
true-shape packer, and reads the rigid motion back onto the parts. Nesting the
island UVs instead means guessing an allowance for the glue tabs, which the
packer cannot see: reserve it and nothing is eligible (a 259 mm staging box
against islands 261-271 mm across - every sheet rolled back and the operator
was a measured no-op), reserve nothing and the tabs overhang the sheet. With
part outlines the fit test is a direct answer.

Things that bite, all of which silently produce a wrong layout rather than an
error:

- **UV Sync Selection defaults ON in Blender 5.x**, and with it on
  `uv.select_all` selects the whole mesh. Force it off, restore it after.
- **Select inside Edit Mode, via bmesh.** Setting `poly.select` in Object Mode
  and *then* switching `mesh_select_mode` re-derives the selection from
  vertices and can select everything.
- **Re-fetch the uv layer after every mode switch.** Entering or leaving Edit
  Mode invalidates RNA pointers into mesh data; holding one across the switch
  crashes Blender inside `attribute_access` rather than raising.
- **`pack_islands` does not fail when it cannot fit.** With scale locked off it
  crams islands in overlapping and the result still measures inside the box, so
  the overlap test is the one that matters, not the bounding box.
- **ORIGINAL_AABB is the bounding box of the *selection*.** Leaving parked-away
  parts selected makes the container enormous and every trial fails, which
  looks exactly like "nothing fits together" (19 sheets instead of 8). There is
  no target-rectangle argument - `CUSTOM_REGION` exists but only
  `use_uv_custom_region` is exposed to Python, so the box has to be set by
  parking the candidates at the origin and pushing one to the far corner.
- **`rotate_method` must be ANY, not CARDINAL.** The core orients each part to
  its minimum-area box at an arbitrary angle, so an island 263 mm across
  belongs on a 284 mm sheet; restricted to 90 degree steps the packer rejects
  arrangements the core had already proved fit. Free rotation is only legal
  because `Material.lock_grain` is off.
- **First-fit-decreasing, not "take everything and drop the smallest".** The
  latter packs all the large parts together and wastes the rest of every sheet:
  9 sheets against 8, and three times slower.

It ties the core's shelf packing rather than beating it (8 sheets on the test
net, declining outright on others), because the binding constraint is part
*shape*, not packing skill - see above. It never takes a layout needing more
sheets than the packing already in hand, so `CANCELLED` is a normal answer.

**Blender splits a UV island wherever two loops disagree in the last bit.**
Island membership is decided by whether the loops across an edge carry the
*same* uv, compared as float32 - not "close enough". Two faces of one island
reach a shared corner down different tree paths, compute it independently and
land a few ulps apart, so a 6e-14 mm disagreement reads as a seam and
`pack_islands` walks the two halves off in opposite directions: a two-piece
net arrives as six. `write_net_uv` therefore welds shared corners to one
value.

Weld on (island, vertex, **quantised position**), never (island, vertex)
alone. One island can legitimately visit the same 3D vertex at two places in
the plane - a net wrapping round and rejoining itself, the near-pinch
`narrow_necks` warns about. Fusing those drags the geometry: 220 mm of
isometry error, which `test_addon`'s isometry checks caught immediately. The
canonical value is a real member of the cluster, not the rounded key, so
nothing moves further than the cluster's own width. Regressions:
`no corner disagrees in the last bits`, `packing does not fragment the net`.

**The cut overlay must follow islands that are dragged - but never sample
them from the draw handler.** The outline is stored per island in *island
space* with two reference loops and resolved against the current UV positions;
two points recover the rigid motion cheaply, where a full fit over every loop
would not. Reading them needs bmesh in Edit Mode, which is exactly when
islands are being dragged (`uv_layers[...].data` is empty there).

**The overlay must never touch bmesh - there is no safe context for it.**
Live Edit Mode UVs live only in the BMesh: the mesh datablock is empty there,
and so is the depsgraph's evaluated copy (both checked). Reading it crashed
Blender from every context available, three times:

- a GPU draw handler: allocates or refreshes BMesh customdata and frees blocks
  the draw is still walking - `BLI_mempool_free` under `BM_data_layer_add`
  under `cb_region_draw`.
- a timer: fires in the middle of a modal transform.
- `depsgraph_update_post`: access violation inside `tbbmalloc`.

There is no fourth context, so the outline is simply **not sampled in Edit
Mode**. `_live_uv` returns None there; the cache keeps the positions from the
last unfold, nest or Object Mode refresh and catches up the moment Edit Mode
is left. `publish_cuts` resolves once immediately, or the outline would not
appear until something else happened to change. A cosmetic overlay is never
worth a crash, and three crashes is enough evidence that this one cannot be
made live safely.

An allocator or GPU crash cannot be caught at runtime, so the rules are
asserted on the **code objects**: `the draw handler does not sample live UVs`,
`the overlay does not poll on a timer`, `the overlay does not import bmesh`,
`no overlay function reads the edit-mode bmesh`. That last one must inspect
`co_names`, not the source - the docstring explaining why bmesh is banned
mentions `from_edit_mesh`, and a substring search fails on its own
documentation, exactly like the `self.report` check on the reload operator.
If the failure kills the process, test the data or the code, not the
behaviour.

**Sticky UV select drags other islands with you.** `uv_sticky_select_mode`
defaults to Shared Location, which selects every UV sitting at the same point
- including ones belonging to a *different* island, which packing routinely
leaves touching. Moving one island then takes corners of its neighbours along.
Island select (`use_uv_select_island`, a toggle in 5.x - `uv_select_mode` is
only VERTEX/EDGE/FACE) or Sticky set to Disabled fixes it; the UV panel shows
both and warns.

**UV Sync Selection, both ways round.** It defaults ON in Blender 5.x, and in
that mode the UV editor moves the loops of *selected faces* rather than whole
islands - so dragging pulls single faces out of the net and scatters them,
which looks exactly like the net corrupting itself. Each face stays internally
rigid; it has simply left its neighbours.

Turn sync off and the opposite bites: the UV editor then draws only the UVs of
*selected* faces, so an empty selection shows an empty net that cannot be
picked at all. Any operator that borrows the face selection to drive Blender's
own tools must give it back - `cbu.pack_uv` deselects everything and selects
one trial subset at a time, and `cbu.split_nonplanar` selects only the bent
faces - or it leaves most of the net invisible. Both snapshot and restore now,
`split_nonplanar` selecting all when it has changed the face count so the old
mask no longer lines up. The UV panel warns in both states and offers
`cbu.select_all_faces` as the way out. Regressions:
`unfold leaves the face selection alone`,
`nesting gives the selection back after borrowing it`.

**Changing the bed silently rescales the 2D view.** `uv_scale` is derived
from the bed, so the *same* stored UVs mean different millimetres once the bed
changes - a net laid out on a 300 mm bed read as half size on a 600 mm one,
while the bed reference redrew at the new size immediately. The view then
showed the net sitting outside a margin the packer had respected, which reads
as "packing is ignoring the margin". `props.rewrite_net` re-emits the net on
any bed change, keeping millimetres fixed and only changing the scale they are
stored at. The layout is still for the old bed - that is what Re-pack is for.

**An oversized part keeps whatever coordinates the unfold gave it.** It is
never placed, so it is never translated, and those coordinates are routinely
negative - a repack onto a bed too small produced a net spanning
x[-136.3, 226.8]. Everything downstream reads positions from the parts, so one
oversized part drags the whole 2D view into negative space and every margin
check on the *view* fails for a layout the packer never claimed to have made.
`pack()` now parks them on the margin, still flagged by `sheet < 0`.
Regression: `test_oversized_parts_are_parked_not_left_at_random_coordinates`.

**Measure a net against `pm.vertices`, not against a recomputed world
transform.** Chasing that apparent corruption, a diagnostic that rebuilt world
coordinates as `co @ matrix_world + t` disagreed with `pm.vertices` by 0.63 mm
on an object with unapplied non-uniform scale, and reported 28 of 72 edges
"torn" with a worst case of 1.21 mm. The net was exact - 0.0000 mm against the
geometry the unfolder actually used. Compare against the same array the writer
consumed, or the measurement invents the defect it is looking for.

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

**One set of controls governs both fold styles.** A thin score and a wide
relief slot are the same decision - how much of this bend line is removed and
how much is left holding it - so `fold_intervals` produces the cut intervals
and both styles consume them. Relief slots used to carry their own
`bridge_len` / `bridge_every` pair that no UI exposed, so on any model whose
folds were wide enough to become slots (which is *every* fold once a fixed cut
width above `relief_threshold` is set) the exposed perforation controls
appeared to do nothing at all.

The slot is built as one strip per interval rather than one long strip with
bridges subtracted, so the radius lands on the ends of every segment. Rounding
the whole slot and then cutting bridges out of it leaves square corners at
every bridge - which is most of the corners, and exactly where a fold tears.

The flute-pitch rule is advice now, not a clamp. Rescaling the period to
pitch/3 silently made the dash-length control useless - anything over 2 mm came
back as 1.08 mm - and the rule matters for a fine score, far less for a slot
with long bridges. `flute_warning` reports it instead.

The fold line is controlled by a *ratio*, not by two lengths: `connect_ratio`
is the fraction left connected (0.5 cuts half of it away), because the useful
question is how much of the fold survives - that governs whether it hinges or
tears - not how long one tie happens to be. `min_connect` (default 10 mm) is
the floor underneath it: a blanket ratio leaves a short joint with a
proportionally small and absolutely tiny amount of material, and this is what
stops small joints tearing off. A fold shorter than the floor is left solid.

That floor must NOT be applied to the remnants between relief bridges. Those
remnants *are* the bridges - the material the floor exists to protect - and
they are shorter than it by design; applying it there deleted every dash and
the PERF layer vanished from the SVG entirely. `perf_segments` takes
`min_connect=0.0` for that call.

`round_cut_ends` radiuses slot ends, because a square internal corner
concentrates stress and is where a fold starts to tear. Note shapely draws a
round cap *beyond* the line end, so `_strip` pulls the ends in by the radius
first - buffering the full segment turned a 50 mm fold into a 54 mm slot that
ate into the panels either side.

`use_tabs` is **off by default**. Most cardboard is taped or gets a glued
backing strip, and the flaps cost about 22% of the sheet in area - measured on
a real design, part area 39857 mm2 without them against 50233 with.

`docs/spec.html` has the full derivation, the research behind it, and the
calibration procedure.

## State of the milestones

Done: M1 (core), M4 (Blender add-on), debug tooling.

Not done, roughly in value order:

1. **True polygon nesting.** Packing is bbox shelving at ~36% sheet use. NFP
   nesting is worth ~+20 points on nets this concave. Blender's own
   `uv.pack_islands(shape_method='CONCAVE', scale=False)` already does this and
   may be enough — see `ops.CBU_OT_pack_uv`.
2. **2D seam editing.** Partly done: `cbu.edit_seam` cuts or welds the
   selected edge from the UV editor and re-unfolds, and `cbu.repack` re-nests
   on demand. Still click-to-pick only - no modal drag-along-a-path tool.
   PolyZamboni's `operators.py:890` is a working reference for Image-Editor
   modal plumbing.
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
