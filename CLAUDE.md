# cardboard-unfolder — context for a code session

Turns low-poly meshes into laser-cuttable corrugated cardboard. Built for
Halloween costume armour and props; the constraints are a hobby laser, free
shipping-box cardboard, and costume-level tolerance (±0.5 mm is plenty).

## Commands

```bash
pip install -e ".[dev]"                 # core + pytest + matplotlib
python -m cardboard_unfolder mesh.stl -o out/ --height 280 --flute B

python -m pytest tests -q               # 288 core invariants (~3 min)
python blender/watch.py                 # re-run core tests on save
python blender/watch.py --addon         # also the Blender suites

pip install bpy==4.5.13                 # Blender as a Python module, ~1 GB
python blender/test_addon.py            # 291 checks, real Blender API
blender --background --factory-startup --python blender/test_addon.py
                                        # same suite against an installed
                                        # Blender, no 1 GB bpy wheel needed
python blender/test_dev.py              # 29 checks on the debug tooling
blender --factory-startup --python blender/gui_check.py -- out/gui
                                        # the one path no headless suite
                                        # reaches: a real click, screenshotted

python blender/dev_install.py           # symlink add-on into Blender

python tools/bench.py --json base.json  # the corpus, saved
python tools/bench.py --baseline base.json   # ...and what moved since
python tools/bench.py --optimise             # cost-driven tree search
python tools/render_packs.py [dir] [model] [height]   # draw every sheet
python tools/bench.py --height 100           # ...and at other scales
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

## The optimiser, and how it is judged

`tools/bench.py` is the scoreboard: a fixed corpus at a fixed seed, printing
islands, **splinters**, min island width, necks, sheets, utilisation, seam
length, repair cuts, mirrored faces and time. `--json` saves a run and
`--baseline` diffs against it, marking each move better or WORSE. The optimiser
moves several of these in opposite directions at once - fewer sheets can cost
splinters, fewer islands can cost a sheet - so a claim about it is a claim
about this table, not about one number.

Where it started (200 mm, 300x300 bed, gap 1, tabs off), where regret left
it, and what `--optimise` adds on top:

| model | islands | splinters | tiny | min width | sheets | seam mm |
|---|---|---|---|---|---|---|
| pikachu | 26 -> 8 -> **8** | 12 -> 2 -> **2** | 4 -> 1 -> **0** | 3.1 -> 3.3 -> **9.9** | 2 -> 3 -> **2** | 6998 -> 6176 -> 6197 |
| annulus | 35 -> 34 -> *36* | 0 | 0 | 32.9 | 13 -> 12 -> **11** | 16949 -> 16712 -> *17349* |
| icosphere2 | 9 -> 9 -> **8** | 0 | 0 | 26.8 | 4 | 9976 -> 9976 -> 9880 |
| capsule | 6 -> 6 -> *7* | 0 | 0 | 38.6 -> *18.0* | 2 | 5183 -> 5183 -> **5031** |
| torus | 57 -> 53 -> 53 | 0 | 0 | 63.4 | 21 -> 21 -> **20** | 32858 -> 31906 -> *32695* |

Half of pikachu's islands could not be held, and **repair was where they came
from**: 25 extra cuts against 0-1 from bed splitting. Phase 2's regret pass
fixed the count; the cost-driven search fixed the *shape*, taking the worst
piece from 3.3 mm to 10.3 mm and winning back the sheet that consolidating
had cost. torus is the one that goes backwards, and the reason is known and
named below: the search cannot see sheets.

`--optimise` is opt-in and the committed baseline is the plain path, because
it costs roughly 3x the time - it has to mend each candidate before it can
score one, and now pack each finalist on top of that. **No model is worse on
sheets**; what it trades away is island count and seam length, in that order,
which is the priority it was given.

**One height is one point off a curve.** The same net is a different problem
at each scale, and running only 200 mm hides both ends of it. Sheets, plain
path then `--optimise`, with parts no sheet can hold in brackets:

| model | 100 mm | 200 mm | 400 mm |
|---|---|---|---|
| icosphere1 | 2 -> **1** | 3 | 13 -> **12** |
| icosphere2 | 2 -> **1** | 4 | 12 -> **11** |
| pikachu | 1 | 3 -> **2** | 7 |
| annulus | 4 | 12 -> **11** | 22 -> *24* [28] |
| torus | 7 | 21 -> **20** | 57 -> **56** [32] |
| box / cone / cylinder | 2 / 1 / 1 | 6 / 3 / 5 | *0* [6 / 13 / 18] |

Mean sheet fill on the plain path runs 18-58% at 100 mm, 26-62% at 200 and
41-67% at 400 - **rising with scale**, which is the opposite of the intuition
that small parts pack better. They do; there are just not enough of them. At
100 mm a model needs one or two sheets and the half-empty last one is most of
the average, while at 400 mm there are dozens of sheets and only the tail is
wasted. Utilisation is a statement about how many sheets there are as much as
about the packer.

**At 400 mm the primitives cannot be cut at all.** A 400 mm box is six faces
each larger than the bed, and bed splitting only cuts along fold edges, so
nothing can be done with them - that is milestone 4, and the `oversize` column
is what makes it visible. It reported `0 sheets` and `1190% util` before:
utilisation was area over `max(sheets, 1)` counting parked parts that are on
no sheet, and zero sheets reads as the best score in the column.

Which is also a live incentive in the cost function, since the sheet count
only counts PLACED parts: without `Weights.oversize` a candidate that makes
more parts too big to place reports FEWER sheets and scores better for it.
Nothing in the corpus has exercised that yet. The weight is 500 - worse than
any number of sheets, because it is not a cost but a part that cannot be cut.

**A splinter is measured by `layout.inscribed_width`, not by area.** The
largest circle that fits is the honest answer to "can I hold this while the
glue sets": a 2 x 40 mm splinter and a 9 x 9 mm square have the same area and
only one of them is workable. It is the FATTEST point, deliberately - "is any
part of it fragile" is a different question and `narrow_necks` answers that
one. An L with 8 mm legs measures 9.4 mm, because the inner corner holds a
bigger circle than either leg, and that is the right answer: the piece has
somewhere to grip. `Material.splinter_width` is 15 mm and is a statement about
fingers, so it is tunable.

**Regret is the biggest single lever in the whole solver.** Greedy set cover
cuts an arc because two faces overlapped *through it at that moment*, and
later cuts in the same run routinely resolve the same conflict from the other
end. Nothing ever went back to check, so the net kept a seam - and usually a
whole island - that had stopped paying for anything. `unfold.reattach` is that
check: for every dual arc still separating two components, try welding along
it and keep the weld when the merged island does not overlap itself. On
pikachu that is **26 islands to 7 and 12 splinters to 2**, more than every
other optimiser change put together.

It needs no cost function to know when to stop, because every weld it keeps is
strictly an improvement - one fewer island, one fewer seam, and a splinter
penalty that can only fall, since the merged island contains both halves. It
runs to a fixed point (`test_reattach_has_nothing_left_to_do_the_second_time`)
and only where `repair` actually ran.

Two things it must not do. It must **not check the bed**: before
`split_to_fit` an island is routinely bigger than a sheet - that is what the
splitter is for - so a bed test here refuses nearly every weld on a large
model, and rejoining more of the tree hands the (provably maximal) splitter
more freedom rather than less. And its neck guard is **off by default**: an
island that wraps round and rejoins itself through a near-pinch is one piece
topologically and two physically, so refusing a weld that creates one sounds
obviously right - measured, it changes only pikachu, and there it buys 3 necks
(102 -> 99) for 3 splinters (2 -> 5) and four extra islands. Welding creates
very few new pinches; most of the rise is the same necks living in fewer,
larger islands. The parameter stays, documented, for the model that does tear.

**Cost-aware repair is the small half of that story, and worth knowing how
small.** `repair` now takes `splinter_width` and becomes the WEIGHTED greedy -
coverage per unit cost rather than coverage alone, where an arc's cost is one
more island, a seam twice its length, and the splinter penalty the two halves
incur that the whole did not. It disagrees with the unweighted rule on **19 of
pikachu's 25 cuts** and still ends at the same cut count, because by the last
few steps *every* arc on a conflicting path makes a sliver - at steps 22-24 all
ten remaining candidates priced identically at ~46. The choice repair is given
is constrained by the tree, so the tree is the lever (Phase 3), and regret is
what recovers what the tree cost. It is on wherever a Material is in hand, and
`splinter_width=0` reproduces the old rule exactly, which is what makes the
two comparable on the bench. The side areas it prices with come from one
rooted DFS per component (`_side_areas`), not one traversal per candidate arc.

**The tree search was ranking candidates backwards.** `search` scores only
the trees that overlap not at all, and falls back to fewest-overlaps
otherwise. On the model where that matters most it does neither job: pikachu
has no clean tree in 60 restarts, so the scoring branch never ran at all, and
the fallback is **anti-correlated** with the answer - spearman -0.30 to -0.46
against the finished cost, and the fewest-overlaps tree costs **+74% to +114%**
more than the best of the same 24 candidates. Many overlaps usually means the
net wraps densely, and a dense wrap consolidates into a few large islands once
mended; a tree that starts tidy often ends up in fragments.

So `unfold.optimise` ranks after `mend`, where the quick cost reaches spearman
**+0.97** on pikachu and +0.73 on icosphere2 and picks the best of 16
outright. There is no cheaper signal - that was looked for, and `len(bad)` was
the one that was there.

Which makes `budget` the design. Mending is what costs, so only candidates
that need mending count against it; clean trees are nearly free and every
restart gets one. The best `finalists` are then bed-split and scored with the
full cost. Two things that looked more principled and were measured out:

- **Ranking after a cheap bed split.** A torus unfolds to a handful of islands
  and leaves the splitter to make 53, so ranking a pre-split net there is
  ranking almost nothing. Fixing it cost 30% more time across the corpus and
  changed nothing it was meant to: torus stayed at 23 sheets and its repair
  cuts went 6 to 15.
- **Estimating sheets from bounding boxes.** Area cannot do it (it is
  conserved, so every candidate gets the same number), but shelving each
  island's minimum-area rectangle does vary with the cut and is 918x faster
  than the real packer. It also ranks BACKWARDS where it matters: over 20 nets
  spearman 0.77 across models - which is just model size - and *within*
  pikachu the estimate read [3, 4, 4, 4] against a real [3, 2, 2, 2].
  Rectangles cannot see nesting, and these nets nest.

So the sheet count is **measured, by packing each finalist** - the only
option left once both estimates had been tried. That was out of the question
while a torus pack cost 29 s and is affordable now that `raster_pack` does it
in one: it returns the SAME count as `pack` (23, 13 and 2 sheets against
pack's 23, 13 and 2 on torus, annulus and pikachu) for 2x to 12x less time,
and `pack` runs it anyway, so the reported number can only be this one or
better. A finalist costs 10.6 s on a torus rather than 22.1.

**The shortlist depth is the binding constraint, and it has to follow what a
pack costs.** Six rather than three because of annulus, whose best net - 11
sheets - ranks *fifth* on the quick score. Then a 100 mm cylinder showed the
other end of it: every candidate scored 44.3 or 44.4, so the order among them
is arbitrary, the one thing that differs (one sheet or two) is invisible to
the cheap tier, and the optimiser returned a 2-sheet net while the plain path
found a 1-sheet one. The quick score ranks well enough to shortlist and
nowhere near well enough to decide - which is the entire reason there are two
tiers - so where packs are cheap, pack far more of them.

`FINALIST_WORK // islands`, floored at `finalists` and capped at
`FINALIST_CAP`: 2 islands buys 24 finalists, a 53-island torus gets 6. A
function of the ISLAND COUNT and deliberately not of a clock - a time budget
would make the answer depend on the machine, and a net that changes when you
run it somewhere else is not one you can hand to anybody.

It does not fix everything, and the place it cannot is worth naming: a 400 mm
annulus has 62 islands, so it gets the floor of six, and its 22-sheet net is
not among them - the optimiser returns 24 where the plain path finds 22. Depth
is affordable exactly where it is least needed. `count_sheets=False` restores
the old blindness, which is what the switch is for.

Note the term changes what the optimiser *wants*: annulus goes from 26 islands
at 13 sheets to 36 islands at 11, and pays 4% more seam for it. That is the
priority ordering working, not a regression - but it is why island count and
seam are the columns that now move the wrong way.

**Mend had to get 10x faster before any of this was affordable**, and the two
fixes are both "stop recomputing what cannot have changed":

- `repair` re-placed the whole net and re-intersected it after every cut.
  Cutting a tree arc moves nothing relative to anything in its own half - the
  fact `split_to_fit` is already built on - so the overlap set can only
  shrink, and only by pairs that just landed in different components, which
  `overlapping_pairs` excludes anyway. Filtering by component is the same
  answer for a fraction of the cost. *Almost*: measured identical on four of
  five corpus models, and on pikachu 8 of 124 cuts differed by up to 6 pairs,
  because `_forest_positions` re-roots each component and GEOS then decides a
  borderline touch the other way. So the filter may end the loop early and a
  real measurement decides whether it was right to
  (`test_repairs_fast_path_never_ends_on_an_overlap`).
- `reattach` re-placed the whole merged component and re-intersected all of it
  for **every candidate arc**. Now the smaller side is moved onto the larger
  (`place_branch`, which must agree exactly with a full `unfold_tree` -
  `test_placing_one_branch_agrees_with_placing_the_whole_tree`), only the
  cross pairs are tested, and the host's polygons and STRtree are cached until
  a weld actually lands. That last one alone was 3.1 s of a 4.9 s mend.

17.3 s -> 1.69 s on a pikachu net. The one behavioural change is the repair
filter picking slightly different cuts: pikachu 7 -> 8 islands, 25 -> 26 cuts,
seam 6058 -> 6176 on the plain path.

**A long solve runs off the main thread, and only the bpy-free core can.**
`CBU_OT_unfold` is split in three: `prepare` reads the mesh, `Solve.run`
unfolds, `apply` writes the net back - and only the middle one moves. That is
the load-bearing use of "the core imports no bpy": `run` holds a PolyMesh copy
and touches nothing Blender owns, which is asserted on the code object
(`Solve.run touches no Blender data`) rather than trusted.

`execute` runs all three in order, so **a script never takes the modal path** -
`bpy.ops.cbu.unfold()` and `bpy.ops.cbu.optimise()` from Python are
EXEC_DEFAULT - and every headless check stays independent of a window
existing. `invoke` starts the thread and a 0.1 s timer only for Optimise; the
plain search is a second or two and a modal loop around it would be pure
machinery.

Three things that are easy to get wrong here:

- **Cancelling is cooperative and has to be.** The thread is inside numpy and
  shapely calls that cannot be interrupted, so ESC sets a flag, the core reads
  it between candidates, and the search answers with the best it has. So ESC
  gives you a net rather than nothing - which matters, because the first
  candidate is usually already better than the old rule's answer. The status
  bar says `ESC to stop` and then `(stopping)`.
- **`run` must not raise.** An exception on a thread nobody joins simply
  disappears, and the operator would then report success for a net that was
  never built. It stores the exception and `apply` re-raises it on the main
  thread.
- **The mesh must not change under the solve.** `prepare` takes an
  `overlay.geometry_fingerprint` and `apply` checks it, because writing a net
  onto geometry that has moved is wrong even though the solve itself was safe.

**Progress has to be shown where the user is looking, and "done" has to
look different from "not started".** The first version reported progress to
`wm.progress_update` - a number on the mouse cursor - and the status bar, and
the request that followed was "we need to show progress for the optimizer",
because both are small, at the edges, and easy to miss entirely. Four faults,
all fixed together:

- **The bar lives in the panel**, under Unfold, via `layout.progress`, with
  the phase as `step N of 3` - each phase counts from zero, so an unlabelled
  bar filling and restarting reads as the run starting over. The modal loop
  has to `tag_redraw` the sidebar itself: a worker thread changing numbers is
  not something Blender notices, and without it the bar draws once and sits.
- **Packing moved into the worker thread.** `build_parts` and `pack` are core
  calls and were running in `apply` on the main thread AFTER the readout had
  gone - on a torus that is 30 s of frozen UI at exactly the moment it looked
  finished. `Solve.run` now does both; `apply` only writes to Blender.
- **The core stopped reporting once the mending budget was spent.** A skipped
  trial `continue`d straight past the progress call, so the bar froze for the
  rest of the search - and ESC went deaf too, since stopping is read through
  the same call (`test_the_optimiser_reports_every_trial_even_past_its_budget`).
- **Done is its own state.** A popup (`Optimise finished in 8 s`, with the
  result and how many candidates were weighed) waits for you, because the
  status-bar report has scrolled away by the time anyone looks after a long
  run. The panel keeps `Optimised in N s` above the result, which is split
  over lines because the single line truncated to "2 shee...repair".

`Unfold` is unavailable while a solve is running (`ops.RUNNING`), so a second
click cannot start a second solve on top of the first.

None of the drawing is reachable headless - invoke needs a window, the bar a
panel, the popup a screen - so `blender/gui_check.py` drives a real window
through a real click and screenshots it. That is how the truncated report and
the per-phase reset were found; neither shows up in any assertion. One harness
quirk worth knowing: `Region.active_panel_category` is read-only from Python,
so it moves the panels onto the tab already showing, for the screenshot only.
It also has to skip `register()` when the installed add-on is enabled, or the
second registration raises before anything runs.

**Time columns are not comparable across sessions.** Chasing an apparent 2-4x
slowdown after this work, the giveaway was `t_pack` tripling too - and nothing
in `layout.py` had changed. A worktree at the previous commit timed identically
on the same machine (`git worktree add <dir> HEAD`), and stray background
probe processes were eating the CPU. Compare against a run taken *now*, not
against a saved baseline, and check an untouched column first.

**"Low poly" in a filename is not a fact.** The spartan helmet is **499,992
triangles** and the coplanar merge removes only 18% of them - it is a smooth
scan, and there is nothing to unfold until it has been decimated. `bench.py`
carries a triangle budget for this reason, and the add-on needs the same guard:
right now that model would simply appear to hang.

**The coplanar merge was quadratic in the wrong variable.** Interior edges were
recomputed inside the group loop by scanning the whole edge map, with an
`in members` LIST test inside that - groups x edges x group size. Invisible on
the 300-face models the suite uses, fatal on anything real:

| case | before | after |
|---|---|---|
| uv_sphere, 24,960 tri (thousands of 2-face groups) | **72.8 s** | **2.9 s** |
| helmet, 16,000 tri | 7.2 s | 1.8 s |
| scaling | 133 -> 450 us/tri (n^1.7) | flat ~110 us/tri |

Byte-identical output. The pathology needs *many multi-face groups*, so a
subdivided box (6 groups) barely shows it and a UV sphere shows it completely -
worth knowing when choosing a fixture. Regression:
`test_the_coplanar_merge_does_not_rescan_per_group`, guarded by the clock
because there is no other way to see a complexity change, with a 10x margin so
it does not flap on a slow machine.

## Things that will bite you

**Splinter COUNT ranks the unbuildable net first.** Measured over eight
pikachu nets: the one with the FEWEST pieces under 15 mm - 15 of them - had
ten of those under 5 mm, while the one with the most, 22, had only five, the
rest sitting between 8 and 14 mm, fiddly but buildable. So `cost.score_islands`
SUMS a steep per-island penalty rather than counting, and `bench.py` carries a
`tiny` column (under a third of `splinter_width`) beside `splinters`. A table
that shows only the count is not enough to rank two nets by.

`cardboard_unfolder/cost.py` is the one number the optimiser minimises, and
every weight in it is a judgement about building these things by hand, so each
is named, tunable and reported alongside the total - a score with no breakdown
is unfalsifiable. Two tiers: `quick=True` never touches shapely and judges a
splinter by the equal-area circle `2*sqrt(A/pi)`, which is a genuine UPPER
bound on the inscribed width, so it can miss a splinter but never invent one
(`test_the_equal_area_circle_is_an_upper_bound_on_what_fits`). Safe as a
filter, useless as a verdict. **Sheets are counted, never estimated**: total
island area is conserved wherever the net is cut, so any area-based sheet
estimate is the same constant for every candidate and ranks nothing at all.

**Unfolding is exact, not approximate.** Planar faces unfold by rigid motion;
edge lengths survive to ~1e-15. There is no solver. Parameterisation tools
(LSCM, ARAP) are the *wrong* instrument — distortion is their premise, and two
faces that shared a 40.00 mm edge must still share one or the model won't
close. Any change that makes `test_unfold_is_exactly_isometric` fail is wrong,
not the test.

**Which side of a seam a face goes is combinatorial, not geometric.** Two
faces of a consistently wound mesh traverse their shared edge in *opposite*
directions, so `_face_local` already hands each one a signed `h` pointing away
from the other: place both with `k = +1` and they land on opposite sides.
`unfold_tree` used to infer the side from `np.sign(np.sum(...))` of the
vertices' signed offsets, which is a centroid test, not a side test. A concave
ring can have its vertex sum on the far side of its own seam - and the child
was then placed on top of its parent.

That is not merely an overlap. Flipping the side *reflects* the face, so the
exported part is the mirror image of the real one and will not fold up - and
`isometry_error` cannot see it, because a reflection preserves every edge
length. Measured on pikachu at 150 mm: **5 of 277 faces mirrored** at an
isometry error of 2.7e-05, 7 folds back-folded, and the spurious overlaps drove
`repair` to 32 extra cuts and **15 confetti islands of one or two faces** -
44% of the islands holding 1.4% of the area. Reading the two windings against
each other rather than trusting the mesh globally also absorbs a locally
flipped ring. After: 24 islands, 23 cuts, 5 tiny, 0 mirrored, 0 back-folds.
Every synthetic mesh in the corpus is bit-identical, because the bug only
bites concave n-gons and primitives have none after the coplanar merge.

`mirrored_faces` is the missing invariant and belongs next to
`isometry_error` in any check of a net. Regressions:
`test_no_face_is_placed_mirrored`,
`test_a_concave_child_is_not_reflected_onto_its_parent`,
`test_a_locally_flipped_face_still_places_correctly`.

**GEOS mis-intersects two faces that share an edge.** This is how the above
was nearly mis-diagnosed. Two faces reach their shared corner down different
tree paths and land **one ulp** apart; `a.intersection(b)` then returns the
*whole of the smaller triangle* for two triangles that only touch along that
edge. A correct net measures as catastrophically back-folded. `set_precision`
at 1e-9 - or the small negative buffer `overlapping_pairs` already applies -
returns the right answer, 0. Same class of failure as
`minimum_rotated_rectangle`; never measure an overlap on raw unfolded
coordinates. Regression:
`test_geos_needs_set_precision_to_intersect_faces_sharing_an_edge`.

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

**Ask where a part fits, not whether it fits here.** `nest` offers each part
a few hundred CORNER positions and `find_pocket` a few more; `raster_pack`
cross-correlates the sheet's occupancy with the part's raster and gets the
overlap count at **every** offset from one transform pair. The number of
candidate positions stops being a thing to budget for, which is what every
other packer here is really limited by. On the corpus: annulus 13 sheets to
12, torus 22 to 21, everything else tied, and on pikachu it is also ten times
faster than the packer it beats (0.5 s against 6.2 s).

Two knobs, and neither wants turning up. **8 or 12 rotations instead of 4, and
1 mm cells instead of 2, give identical sheet counts for 2-5x the time.** So
the win is the question being asked, not the resolution it is asked at - and
"a finer sweep" is exactly what plateaued before. Restarting with a jittered
order buys one more sheet on torus (21 -> 20) for six times the work, which is
why the single pass is what ships.

**2 mm is a floor, though, not the middle of a plateau.** Turning the cell
size *down* changes nothing; turning it up costs sheets immediately - 3 mm
takes torus 23 to 24 and annulus 13 to 14, and 4 mm takes pikachu 2 to 3. The
padding is half a cell diagonal, so a coarser grid does not just blur the
answer, it fattens every part.

**The mask must be a provable superset of the part, and dilation is not.**
Sampling cell centres under-covers by up to a cell, and a cell is 2 mm of real
collision. The obvious fix is to dilate the sampled grid - and a 4-connected
dilation cannot reach the cell at a convex corner, whose only filled
neighbour is diagonal, which is what
`test_the_raster_mask_covers_the_whole_part` caught. Buffering by **half a
cell diagonal** before sampling is exact instead of nearly right: the nearest
cell centre to any point is the centre of that point's own cell, at most
(res/2, res/2) away, so it lies inside the buffer and the cell is marked. It
is also less conservative than a dilation, 0.71 of a cell against a whole one.

`raster_mask` returns the grid's origin along with the grid, because the two
have to agree - placing a part by one geometry's bounds while its mask came
from another's is a silent offset, and the first version had exactly that bug.
And `free_offsets` thresholds at 0.5, not 0: a float transform lands on
~1e-12. `test_the_fft_collision_test_agrees_with_counting_the_overlap` checks
the correlation against the definition at every offset, because an off-by-one
there is a packer that quietly stacks parts on top of each other.

`pack()` runs all three strategies and keeps the winner
(`test_pack_never_loses_to_any_strategy`). The raster packer is the newest and
least exercised, so it earns its place by that rule rather than by replacing
anything.

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

**Every packer here places a part once and never looks at it again.** Shelf
packing commits to a shelf, `nest` is greedy bottom-left, and the add-on's
Nest Sheets re-runs Blender's packer over the whole candidate set on every
trial - so a part is dropped to the next sheet whenever that *global* re-solve
fails, which is not the same as there being no room for it. `layout.tighten`
is the pass that reconsiders: it moves parts off the later sheets into space
left on the earlier ones, never moving anything already placed, so it can only
tie or improve. Measured on pikachu at 150 mm: Nest Sheets left 35 parts on a
sheet 44% full and one 44.6 x 92.2 mm part alone on a second sheet. Now one
sheet, 0 overlaps, closest approach 3.33 mm against a 3.00 mm gap.

**`tighten` must seat before it measures**, and does it itself rather than
trusting the caller. The free region is computed over `[m, m+W] x [m, m+H]`,
so a sheet whose parts have not been put on their margin yet is being measured
in a frame it does not live in: the stray lands correctly inside that box and
the seating shift that follows drags it out of the bed. That is how this first
appeared - `part 31 ended up outside sheet 0`. For the same reason the add-on's
bed check runs after tightening and seating, never before.

**A budget spent in sheet order always fails on the same sheets.** The torus
packing ended with ONE part alone on a sheet at 13% full while sheet 17 sat at
56% - and the part fitted sheet 17 perfectly well. `find_pocket` never got to
try it: `POCKET_BUDGET` is 16 deep searches, `tighten` walks targets in sheet
order, and sheet 17 is the seventeenth. Not a tuning problem - any fixed
budget consumed in order is a budget that never reaches the late sheets, which
are exactly the ones with room.

So `tighten` now asks three searches in cost order, and they answer different
questions. `find_spot` offers corner positions against the skyline: right for
a sheet being built, useless for one already full. `raster_spot` offers EVERY
position on a 2 mm grid from one transform per sheet, and is what finds a
pocket in the middle of a finished sheet - it runs before the budget and is
not gated by it. `find_pocket` stays last, still budgeted, because it works on
exact geometry where the raster is conservative by 0.71 of a cell, so it can
still place a part that needs the last millimetre.

Worth **a whole sheet**: torus 21 -> 20 with the optimiser, utilisation 62.1
-> 65.2. It was found by looking at a picture of the packing, not by any
number in the table - 13% on one sheet is obvious to an eye and invisible in a
mean. Regressions:
`test_the_raster_search_finds_a_pocket_the_corner_search_cannot`,
`test_tighten_reaches_past_its_expensive_search_budget`.

**Corner candidates cannot see a pocket.** `_corner_candidates` offers the
cross product of the placed parts' x and y edges, which is the right heuristic
for building a packing against a skyline and the wrong one for slotting into a
sheet that is already full, where the opening is bounded by parts on several
sides at once. On that pikachu sheet the stray had exactly one home - a 5 x 2
mm window of valid positions at a single rotation, 7 placements on a 1 mm grid
- and not one of the 4900 corner candidates landed in it. `find_pocket`
therefore works off the *free region* (`sheet - union(placed).buffer(gap)`) and
tries each corner of the part's box against each corner of the pocket.

Generate those candidates from a **simplified** copy of the free region and
verify against the exact one: 3.06 s -> 0.50 s on that sheet for the same
answer, and it cannot go wrong, because a simplification that strays outward
can only cost a position, never place a part on top of another. The free
region traces round every part outline, so it carries ~3400 vertices of
crevice detail that no part can ever occupy.

Cost, honestly: `tighten` changes no sheet count in the plain `pack()` path on
this corpus - the layouts there strand parts that genuinely do not fit, which
the raster search confirms rather than fixes - and most of its time is spent
*proving* there is no room. It earns its keep where a tail exists: 2 sheets to
1 in Nest Sheets, and torus 21 to 20 under the optimiser. It roughly doubled
`t_pack` on torus (16 s to 34 s) for that sheet. `POCKET_BUDGET` caps only the
exact search; the free region and the occupancy transform are both cached per
sheet and dropped the moment something lands there.

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

**A view setting baked into a batch is a dead control.** `overlay_alpha` was
resolved to vertex colours in `rebuild` and never read again, so the Opacity
slider did nothing until the next unfold - and `CBU_OT_unfold` did not pass the
setting at all, so it was welded to the 0.55 default no matter what the slider
said. The cache now holds the island *index* per vertex, not the colour, and
`set_alpha` re-colours geometry already in hand: no mesh read, nothing unsafe
to run from a property update while a slider is being dragged. It short-circuits
on an unchanged value because `update=` fires continuously during a drag. Any
new view setting has the same shape - if it is consumed at build time, wire an
`update=` that restyles, or it is decoration.

Note **`_batches` is always empty under `blender --background`**: `available()`
is False without a GPU and `rebuild` returns early, so nothing about the cached
batches can be asserted headless. Test the pure part (`island_colors`) directly,
and stand a stub in for `_shaders` / `batch_for_shader` to reach `rebuild`'s
bookkeeping.

**The island tint must be lifted off the mesh.** It is drawn on the *same*
planes as the faces it colours, so coplanar triangles leave the depth test to
decide per pixel - the object's own polygons punch through the colour and the
viewport looks corrupt. GPU state exposes no polygon offset to Python, so
`rebuild` lifts each ring along its own normal at build time
(`overlay.lift`, `LIFT` = 0.0015 of the object's local bounding box).
`lift_amount` reads `bound_box`, not `dimensions`: the box is local and
excludes object scale, and the overlay draws under `matrix_world`, so the
offset then scales with the geometry instead of being fixed in world units -
which matters a lot when an STL import leaves a 0.001 scale on the object. The
normal comes from Newell over the whole ring, not a cross product of the first
three corners, which is numerically junk on an n-gon whose first corners are
nearly collinear.

**Hiding the annotation must not cost you the seam marks.** `show_overlay`
gated the tint and the forced marks together, and lived in a *closed* Solver
sub-panel next to the output directory. The marks are precisely what you want
visible while marking seams; the tint is what hides the selection highlight and
the shading. So `overlay_fill` switches the colour on its own, and both sit on
the main panel. Regression: `and honours overlay_fill separately`, which reads
`_draw.__code__.co_names` - a `getattr(s, "overlay_fill", True)` puts the name
in `co_consts` instead and the check will not see it, so read the attribute
directly.

**The colour key has to be a real swatch, and the same value the overlay
draws with.** An `icon=` on a button is tinted by the theme and reads as a
letter with a smudge beside it - not a key. `ui.mark_key` draws
`mark_cut_color` / `mark_fold_color` as COLOR properties, which render as
actual colour fields, and `overlay.mark_colors` reads those same properties -
so the legend cannot drift from the thing it explains, and the colours can be
changed when the defaults clash with a model. The constants stay as the
defaults and as the headless fallback.

Compare against `mark_colors()`, never against the constants: Blender stores a
colour as **float32**, so a default of 0.85 comes back as 0.8500000238 and an
exact comparison with the constant never matches.

The key belongs *beside the buttons that make the marks* - the Cut/Weld pair
in the UV panel is `cbu.edit_seam`, which calls `cbu.mark`, so it writes
exactly the same red and blue as the Seams panel. Putting the legend at the
top of the UV panel, several boxes away from those buttons, is not a key.

**Two actions are two buttons, never one button and a switch.** The
optimiser went through two wrong shapes before `cbu.optimise` existed:

- a switch in the Solver sub-panel, which is `DEFAULT_CLOSED` - the feedback
  was "how do I run the optimiser? I only see the repack and nest buttons".
  The same mistake `show_overlay` made in that same panel.
- a toggle under Unfold, which could be seen - and was a **scene property**,
  so it was saved with the file. Switch it on once and every Unfold after
  that, in that session and in the saved .blend, quietly ran the slow path:
  "hitting unfold seems to run the optimizer". The fast button stopped being
  fast and nothing on screen said why.

Now `Unfold` is always the plain, quick solve and `Optimise` is its own
button. They are two thin operators over one plain mixin, `ops._Unfold`, with
`optimising` as a class attribute - not a scene setting (saved with the file)
and not an operator property (Blender remembers those between calls unless
told `SKIP_SAVE`). And **not** `class CBU_OT_optimise(CBU_OT_unfold)`: once a
registered operator class is subclassed and the subclass registered too,
`bpy.ops.cbu.unfold.poll()` fails with "context is incorrect" on a context
that plainly satisfies it. Mixin plus two `bpy.types.Operator` subclasses is
the shape Blender supports. Regression:
`Unfold after Optimise is a plain unfold, not a second optimise`.

**Four switches in a row truncate to nothing.** Blender gives each item in
an aligned row an equal share of the panel width, so `Show bed / Show cut /
Annotation / Marks` side by side reads as `Sho..` four times at any sane
width. The UV layer switches are a `column`.

**A typo'd icon name only fails inside `draw()`**, which needs a real UI
region - so no headless suite ever reaches it and the panel is simply broken
for the user. `every icon in ui.py exists` checks all 26 against
`UILayout.bl_rna`, which needs no drawing at all.

**Lifting the tint put it in front of the forced marks.** The island colour
is raised off the mesh by `LIFT` to stop it z-fighting the faces - and the
cut/fold marks are drawn on the mesh's own edges, so the colour then swallowed
them. They are the one thing you are actually looking at while editing seams,
so they go `MARK_LIFT` (3x) further out along the vertex normals. Further out,
not depth-test-off: disabling the test would show marks on the far side of the
model straight through it.

**Marks need their own refresh, precisely because marking is not an edit.**
`geometry_changed` correctly reports that writing an edge attribute did not
modify the object - which is what stops the island colours vanishing while you
mark against them - and that also meant the marks themselves never caught up
until the next unfold. `overlay.refresh_marks` is the cheap half of a rebuild:
one pass over the edges, no unfold, no island data. `cbu.mark` calls it, and
so does `_on_depsgraph`.

**The marks belong in the 2D view too**, drawn *last* so nothing can cover
them (`show_net_marks`). Unlike the cut outline these ride on the mesh's own
edges, so they need no island-space round trip - the net UV layer already says
where each one is, and `live_mark_segments` reads it in Object Mode under the
same rule as everything else. Both halves of a cut seam are emitted, because
both are marked and both are what the builder has to match up.

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

**Both seam edits are local, and neither re-runs the unfold.** Re-unfolding
lets the solver rebuild every tree, so the faces freed by a cut usually
reattach somewhere else entirely and the whole layout jumps - correct, but not
what someone editing one seam expects, and it throws away where every other
hand edit put things. So `cbu.edit_seam` writes the mark and edits in place;
**Unfold is the one explicit global re-solve**, and it honours the marks left
behind. That is the workflow: Unfold to see it, edit seams by hand, Unfold
again when you want the solver to reconsider.

`unfold.split_island` removes one fold in place and detaches exactly the piece
on its far side. An island's folds form a spanning tree, so every fold is a
bridge: removing one always yields exactly two pieces. The spent fold becomes a
matched pair of cuts sharing a new seam id, which is what pairs a glue tab with
its partner.

`unfold.join_islands` is its exact inverse. Where the welded island goes is not
a choice - f1 has exactly one placement sharing the edge with f0 and lying on
its far side, and `unfold_tree` already knows how to find it; the rest of the
island follows by the rigid motion that implies, recovered from the two shared
corners. No reflection is possible, because both placements are
orientation-preserving images of the same face. **Split then join is
byte-exact**: measured at 1.4e-13 mm on a real net and asserted across the
corpus at 1e-9 (`test_split_then_join_puts_the_island_back_exactly`), plus the
island being welded ONTO must not move at all. Without that, welding a seam you
just cut would nudge the net and every hand edit would drift.

A weld CAN produce a self-overlapping island - it re-creates whatever the
solver avoided by cutting there. That is visible rather than prevented:
`overlapping_pairs` reports it and Unfold is the way out.

**A net IS its cut set, and that is what makes the optimiser usable.** Every
dual arc is either a fold or a cut, so the complement of the fold forest is a
*complete* description of a net. Hand it back as `forced_cuts` and
`apply_constraints` deletes exactly those arcs, leaving a graph that is already
a forest - the spanning-tree step then has nothing left to choose and the seed
stops mattering. Measured on pikachu: 26 islands and 225 cut arcs replay at
seeds 0, 3 and 41 to the same membership and **0.00e+00 mm** of vertex
movement. `unfold.cut_arcs` is that set, `bridge.write_constraints` writes it
into the mesh, and together they are why an optimiser's answer can be handed
to the user as marks they can see and edit rather than as a number nobody can
reproduce. Arcs that an *incoming* constraint deleted are included, because
they are cuts too: the set has to describe the whole net, not the part of it
this run happened to decide. Regressions:
`test_a_nets_cut_set_replays_it_exactly_at_any_seed`,
`test_the_cut_set_is_the_complement_of_the_folds`.

**`weighted_dual` built its graph from EDGES, so a fully-isolated face
vanished.** A face whose every dual arc is force-cut had no node at all - it
fell out of the spanning tree, out of `_forest_positions`, and finally out of
`build_islands` with a bare `KeyError: 269`. Invisible until a constraint set
is complete enough to isolate a face, which is *exactly* what replaying a cut
set does, so it surfaced the moment the above was first tried. `W.add_nodes_from(G)`
is the fix. Regression:
`test_a_face_cut_off_on_every_side_is_still_an_island`.

It is not free, though, and the reason is worth remembering: **node insertion
order is a tie-break**. Adding the nodes before the edges changes the order
`nx.connected_components` walks them in, which changes the roots
`split_to_fit` draws, which changes *which* arcs it cuts. Measured on the
corpus: identical everywhere except torus, whose 52 bed splits came out 31 mm
shorter in total seam - 0.1% of 32.9 m, same islands, same sheets, same repair
cuts. Attributed by rebuilding `W` from its edges alone and re-running the
bench row, not inferred.

**A lock constrains the optimiser, nothing else.** `bridge.LOCK_ATTR`
(`cbu_lock`) is a bool per edge; `write_constraints` skips those edges and
overwrites every other one - including *clearing* marks the cut set did not
ask for, because a stale FORCE_CUT left behind would silently constrain the
next solve. Clear All Marks is a hand edit and still clears a locked edge:
locking is aimed at the solver, and giving it a second meaning would make both
unclear. Note a lock on an AUTO edge is meaningful in its own right - it says
"never force anything here".

Locked marks draw **dashed** (`overlay.dash`, three dashes and two gaps, a
dash at each end so the mark still shows where it stops). Colour is already
spoken for - it says cut or fold, and the user chooses it - and line width is
a per-*draw* uniform rather than a per-segment vertex attribute, so neither is
free to carry this. The dashing happens at sample time in `mark_lines` and
`live_mark_segments`, so `_marks_batch` stays a plain segment list and knows
nothing about locking.

**`split_to_fit` reports the arcs it cut, not just how many.** The optimiser
has to weigh a bed split differently from a repair cut: one lands on a real
crease that already needs a glue flap, the other is the solver working around
an overlap it could not resolve. A single count says nothing about which lever
to pull. They reach the report as `bed_cut_arcs`, and must be a subset of
`cut_arcs` or they name an edge that is actually folded
(`test_the_bed_splits_are_named_and_are_part_of_the_cut_set`).

**`.gitattributes` pins LF, and that is not cosmetic.** `core.autocrlf=true`
hands back CRLF on any checkout - including the implicit one inside
`git stash pop`. Every edit script in this repo opens with
`assert "\r\n" not in src`, so a stashed-and-restored file then fails its own
patch with no explanation, and `sync_vendor` copies the CRLF straight into the
vendored tree. It bit twice in one session before the file was added. Nothing
CRLF has ever been committed; check with
`git show HEAD:path | grep -c $'\r'` if in doubt.

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

**`watch.sync_vendor` compares content, not mtime.** The vendored tree is a
pure copy, so "the copy is newer" is never a reason to keep it - and something
else having touched it is exactly when it needs overwriting. `test_dev.py`
edits the vendored `material.py` to prove hot reload works and restores it in a
`finally`; a *killed* run skips that, leaving `crush = 0.61` with a fresh
mtime, and an mtime-based sync then refuses to undo it for the rest of time.
The next `test_dev` run fails three checks that read like real defects in the
material model. It now self-heals from the core and says so.

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

**The cut sits ON the part edge by default.** `Material.edge_offset` is 0:
positive is kerf compensation so the part comes out nominal size, negative
cuts it back - which is what thick board usually wants, since two panels
meeting at a shared edge collide at 3 mm unless each gives a little. It used
to be hard-wired to +kerf/2, which at 0.15 mm is below what costume tolerance
needs and was the sole cause of the notch problem below. Zero also makes
`corner_relief` inert, because nothing buries a notch apex without an outward
offset - so the default carries none of that machinery. Regressions:
`test_the_cut_sits_on_the_part_edge_by_default`,
`test_corner_relief_is_inert_without_an_outward_offset`.

Note this changes what `Part.cut` means at the defaults: it is now exactly the
island outline with relief slots taken out, so anything asserting the cut is
*bigger* than the net is only true with tabs on or a positive offset. Two
add-on checks assumed otherwise. And compare the cut against the net UVs at
1e-3 mm, not 1e-6: the UV layer is float32 and carries ~9e-5 mm of storage
error, so a tight comparison measures that and nothing else.

**A kerf offset cannot reach into a notch, and no join style fixes it.**
Growing the part by kerf/2 is right along an edge and at a convex corner. A
notch cut into the material is an *inner* corner: the offset fills it and the
cut stops short of the apex by exactly `(kerf/2) / sin(theta/2)` - 0.30 mm at
60 degrees, 0.86 at 20, 1.72 at 10, 8.59 at 2. Mitre, bevel and round shape
only *outer* corners, so `mitre_limit` changes this by nothing at all -
measured identical to six decimal places across all four join styles
(`test_kerf_offset_cannot_reach_into_a_notch`). On a pikachu net: 634 notches,
median loss 0.15 mm (fine, that is just the offset) but worst case 13.09 mm of
material left uncut at an apex. On 3 mm board that is a fold that will not
fold.

`folds.corner_relief` reaches the apex explicitly, with a slit one beam wide
down the notch bisector, `corner_overcut` mm past it. Take the direction from
`nearest_points(cut.boundary, apex)` rather than from angle arithmetic - that
point IS the offset apex, so the segment between them is the bisector, and
nothing has to reason about winding or which side the material lies on. Only
notches losing more than a whole kerf are touched; below that the beam already
covers the apex. Worst case 13.09 mm -> 0.30 mm, for 0.13% of the net's area.

Two things it must not do, both found by measuring rather than by the suite:

- **Never break a part in two.** At a near-pinch the neck is held together
  only by the material the offset added - which is exactly what relief removes
  - so a slit there severs the part: 24 parts became 48 loose pieces on the
  first measurement. The slits are applied together, and only if that raises
  the piece count are they re-applied one at a time, dropping the ones at
  fault. `narrow_necks` still warns about the pinch.
- **Keep only the areal result.** A slit landing exactly along an edge leaves
  a stray LineString and `difference` returns a GeometryCollection, which has
  no `.exterior` - every consumer of `Part.cut` then breaks. It happens on a
  real pikachu net and on nothing in the synthetic corpus.

Cost: the core suite went 34 s to 89 s when this was a per-vertex Python loop,
because nearly every vertex is an ordinary one sitting the offset distance
away. `shapely.contains` / `shapely.distance` over a point array, with the cut
prepared, brings it back to 41 s.

`round_cut_ends` radiuses slot ends, because a square internal corner
concentrates stress and is where a fold starts to tear. Note shapely draws a
round cap *beyond* the line end, so `_strip` pulls the ends in by the radius
first - buffering the full segment turned a 50 mm fold into a 54 mm slot that
ate into the panels either side.

## Annotation

Three engravings, and the net is not assemblable without the first:

- **Seam numbers**, just inside each cut edge, on both halves. A 60-piece net
  with unlabelled edges is a puzzle; the same net with matching numbers is a
  job.
- **M / V beside every fold.** Which way a crease bends is not recoverable
  from a flat sheet, and getting one wrong on an assembled shell means peeling
  it apart. The mark clears the relief slot using that fold's actual opening
  from its `FoldPlan`, and goes on whichever side is inside the island - a
  fold on the island's edge has material on one side only.
- **`model_id` plus the island number**, inside each island. Cut two models in
  one afternoon and the pieces are otherwise indistinguishable: both are grey
  polygons whose seam numbers each start at 1. In the add-on the panel field
  wins, then the object's name, then the .blend's (`props.model_id_for`); the
  CLI takes `--id` and falls back to the mesh file's name.

**Build all three kinds together, in priority order, against one occupancy
map.** `tabs.island_labels` is that pass; `seam_labels` / `part_labels` /
`fold_labels` are now thin filters over it. Built separately they were laid
down blind on top of one another - two numbers burned through each other read
as a third number that means nothing, which is worse than either being
missing. Order decides what gives way when there is no room: island id first
(without it two models cut the same day are one pile), then seam numbers
(without them the net is not assemblable), then M/V (useful, and the first
thing to drop). Collision is tested on the *card* rather than the strokes -
two labels whose cells touch are too close to read even if their glyphs happen
to interleave. Regression: `test_labels_never_overlap_each_other`.

**Place labels against the BODY, not the outline.** Relief slots are holes,
and a number engraved across one is cut in half by the fold it was meant to
explain. On a 3 mm board every fold on a real net becomes a slot, so the
outline is a poor picture of where there is material - an island id was
landing across a fold line until `build_parts` started passing the
relief-subtracted body in. Regression:
`test_no_label_is_engraved_into_a_relief_slot`.

**Two ceilings on how big a label may be, and the smaller wins.**
`Material.label_edge_frac` (0.25) caps text width at a fraction of the edge it
names - a number as long as its edge reads as a label for the whole part -
and `ACROSS_FRAC` (0.5) caps it at half the *face's* narrow dimension, which
is what stops a big glyph on a 3 mm spur. `Cut.face` and `Fold.f0/f1` say
which face to measure; `min_area_rect` gives its narrow side.

The edge fraction is the dominant lever on coverage and it is worth knowing
how dominant. On a pikachu net the median cut edge is **10.5 mm** and a
two-digit number at the 2 mm floor is 2.88 mm wide, so a quarter needs an
11.5 mm edge and only **41% of edges have one**. Measured: 24% of edges
numbered at 0.25, 36% at 0.35, 41% at 0.5. It is exposed in the panel because
that trade is a judgement, not a fact.

**Label length is the other lever, and it is bigger than it looks.**
`tabs.short_id` keeps a name that is already 8 characters or fewer and hashes
anything longer to 8 hex digits - hex because every character has to exist in
the stroke font. An imported STL is routinely called something like
`OLD_VERSION_READ_DESCRIPTION_pikachu_1gen_flowalistik`, which fits nowhere at
all. Measured on that net: `3ACB3DBA-0` reaches 54% of islands, `PIKA-0` 75%,
a bare `P0` 88%. So the panel box is not decoration - typing a short tag buys
back a fifth of the islands.

**M / V go at the middle of the fold, aligned to it.** Stepping a letter clear
of its fold spends far more of a congested part than the letter does, so the
offset is zero unless there is a relief slot to clear - and then it is exactly
the slot's half width plus a little. Regression:
`test_a_fold_mark_sits_at_the_middle_of_its_fold`.

**A label must be tested where it will actually be burned.** Engraving outside
the part is not cosmetic: the laser either cuts it off the sheet or burns it
into the neighbouring piece. Three traps, all found by measuring a real net
rather than by the suite:

- **Seam numbers had no containment test at all.** An edge long enough to hold
  the number says nothing about the material *behind* it - a narrow spur, or a
  neighbouring cut running back at an angle, leaves none. 109 of 346 seam
  numbers on a pikachu net hung outside their own island.
- **Testing an upright box and drawing rotated text is not a test.** 9 of 209
  fold marks passed it and still hung over the edge. `_card` rotates the
  rectangle the way the text will be drawn.
- **The card is sufficient but not necessary.** It is the glyph *cell's*
  bounding box and letters do not fill their cells, so card-only cost 27
  points of seam coverage (78% -> 51%) by refusing labels whose glyphs had
  room. `_place` tries the card first and falls back to the strokes
  themselves, which are exactly what gets burned.

**Slide a seam number along its edge before giving up.** The midpoint is the
obvious place and often the worst one, for the same reason as above. Trying
0.5, 0.35, 0.65, 0.25, 0.75 along the edge against four inward offsets took
seam coverage from 54% to 62% of edges - and, the number that actually
matters, **72% of seam ids carry their number on BOTH halves**. Honest
coverage on that net: 62% of seam edges, 84% of folds, 75% of islands, zero
strays. The 78% figure this replaced was inflated by the 109 strays.

**The height rule is cap, floor, or omit.** `label_height` 4 mm,
`label_min_height` 2 mm. Width is linear in height, so for an edge label the
fitting height is one division - there is nothing to search (`fit_height`). A
label that still will not fit is **left off**: a 1 mm character on corrugated
board is a burnt smudge that reads as nothing, and a smudge where a seam
number should be is worse than a bare edge, because the builder trusts it.
Measured on a pikachu net at 150 mm: 78% of seam edges, 83% of folds and 75%
of islands carry one, and every label lands between 2.00 and 4.00 mm.

**`Label` carries its own height**, because the strokes cannot be measured
afterwards - they are rotated, so their y extent is the glyph's *box* at that
angle and reads up to 2.5x the real height. A probe that measured the strokes
reported 9.94 mm glyphs from a 4 mm cap, which is a measurement bug that looks
exactly like a real one. Path length is the other wrong metric: a `P` is one
polyline that walks 8.8 mm around a 4 mm glyph. **Diameter** - the largest
distance between any two of a stroke's points - is what a ruler measures and
what no rigid motion changes; no stroke of text `h` tall can exceed the cell
diagonal, `hypot(0.6h, h)`. Verified end to end at 4 mm: core 4.000 median,
the draw buffer 4.665 max against a 4.665 ceiling with 0 of 1378 over, and
the SVG the same. Regression:
`test_no_engraved_stroke_is_bigger_than_the_text_it_claims`.

**The bed layer carries a labelled mm ruler** (`overlay.axis_segments`,
toggled by `show_bed_ruler`) for exactly the reason below: the unit is drawn
rather than remembered. Ticks run along the bottom and left edges with a
1/2/5 step from `nice_step`, numbered every major tick, and "MM" once at the
corner. They sit **outside** the sheet - below and left of the origin - so
nothing is ever drawn over the parts, which is asserted
(`nothing is drawn inside the sheet area`). The numbers are stroke-font text
like the engraving, so the ruler scales with the bed rather than the zoom: the
same 300 mm span emits the same ticks at 1/512 and 1/1024.

**The UV unit square is the bed rounded UP to a power of two, and that
changes with the bed.** A 300 mm bed gives 1/512, so 1 UV unit is 512 mm; a
600 mm bed gives 1/1024. Reading a glyph as 0.0078 UV and multiplying by the
1024 from this file's `uv_scale` example returns exactly 8.00 mm for 4 mm
text - a clean factor of two, and a very easy thing to conclude is a bug in
the text. The UV panel prints the live figure (`UV editor scale: 1 UV unit
= N mm`); trust that over any remembered constant.

**Two scale factors, and each label has to name its space.** The Size panel's
`mm per scene unit` (1000) says how big the MODEL is in the 3D viewport; the
UV panel's figure (512 at a 300 mm bed) says how the NET is stored in the UV
editor. They are unrelated, and when both read "mm per unit" the add-on looked
like it contradicted itself - the user asked which one was right. Neither can
be made to match the other: the UV factor must be a power of two (exact) that
fits one sheet in 0-1, and it depends on the bed, never the model.

**Island labels need a point and an angle, not just a point.** The centroid of
an L-shaped or horseshoe island is frequently outside it and
`representative_point` only promises to be *somewhere* inside - which on a long
thin island is somewhere with no room. `polylabel` gives the point furthest
from any edge. Then sweep angles: these nets are full of slivers with plenty of
room along them and none across, and horizontal-only left 10 of 24 pikachu
islands unlabelled, three of which had obvious room (one was 22.4 x 17.7 mm
with a 7.5 mm inscribed circle). Long axis, its perpendicular, then every 15
degrees took it from 58% to 75%. The rest are genuinely 5-19 mm slivers.

Label *length* is the real lever on coverage, which is why the tooltip says to
keep `model_id` short - "PIKA-0" needs 9.6 x 2.0 mm of clear room at the floor.

**The engraving has to be published to the 2D view, like the cut outline.**
It is generated in the core and lives solely in the exported parts - so
without `publish_cuts` carrying it, the first place a seam number appears is
the SVG and the UV editor silently shows an unannotated net. It rides in the
same per-island group as the cut rings, in island space, and `_resolve`
recovers the rigid motion once for both. Draw it with `stroke_segments`, not
`cut_segments`: these are OPEN polylines, and wrapping the last point back to
the first draws a chord straight through every letter. Toggled by
`show_net_labels` in the UV panel beside `show_bed` and `show_cut`.

Each of those three is its own switch. `_draw_bed` used to return early when
`show_bed` was off, which hid the cut outline and the engraving with it - not
what the checkbox says.

**Never reject an engraved stroke for closing on itself.** `_mls` dropped any
stroke whose first point equalled its last, meaning to reject zero-length
segments - but that is the definition of a **closed** stroke, which is what
0, O, D and Q are in this font. Every one of them was silently deleted: a part
stamped `PIKA-0` engraved as `PIKA-`, and seam 10 engraved as `1`. That is not
a missing annotation but a **wrong** one, and nothing downstream can tell.
Island 0 of a pikachu net built 206 strokes and engraved 194. Test the stroke
*count* through to `Part.engrave`; a filter that eats glyphs is invisible to
any area or bounds check. Regressions:
`test_a_closed_glyph_survives_to_the_engrave_layer`,
`test_every_label_stroke_reaches_the_part`.

`use_tabs` is **off by default**. Most cardboard is taped or gets a glued
backing strip, and the flaps cost about 22% of the sheet in area - measured on
a real design, part area 39857 mm2 without them against 50233 with.

`docs/spec.html` has the full derivation, the research behind it, and the
calibration procedure.

## State of the milestones

Done: M1 (core), M4 (Blender add-on), debug tooling.

Not done, roughly in value order:

1. **2D seam editing.** Partly done: `cbu.edit_seam` cuts or welds the
   selected edge from the UV editor and re-unfolds, and `cbu.repack` re-nests
   on demand. Still click-to-pick only - no modal drag-along-a-path tool.
   PolyZamboni's `operators.py:890` is a working reference for Image-Editor
   modal plumbing.
2. **Better sequences for the raster packer.** It is still greedy
   bottom-left in decreasing-area order; jittered restarts already find one
   more sheet on torus, and annealing over sequence and rotation is the
   documented way past that. The collision test is no longer what limits it.
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
- Anything that can run for tens of seconds takes
  `progress(done, total, note) -> bool` and calls it per unit of work;
  returning False stops the work and answers with the best it has.
- Tests assert invariants, not snapshots. Prefer a property that must hold
  (area conserved, nothing exceeds the bed) over a golden file.
- Empirical constants live in `material.py` and are exposed in the UI; derived
  geometry is hard-coded. Keep that line clear, and label empirical values as
  such in comments.
