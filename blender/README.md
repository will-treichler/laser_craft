# Cardboard Unfolder — Blender add-on (M4 prototype)

## Install

1. `Edit ▸ Preferences ▸ Add-ons ▸ ▾ ▸ Install from Disk…` → pick
   `cbu_blender.zip`
2. Enable **Cardboard Unfolder**
3. Open the sidebar in the 3D viewport (**N**) → **Cardboard** tab
4. First run will say dependencies are missing — click **Install
   Dependencies** (pulls `shapely`, `networkx`, `ezdxf` into Blender's Python),
   then restart Blender

Tested against Blender **4.5 LTS**. `bl_info` claims 3.6+; the only version-
sensitive bits are the GPU shader names and `uv.pack_islands` arguments, both
of which degrade gracefully.

## Use

**Unfold** — select a mesh, press it. You get:

- islands tinted by index in the 3D viewport
- the flat net written to a UV layer called `CBU_Net`, **in millimetres**
- a one-line report: islands, sheets, isometry error, how much repair was needed

**Mark seams** — go to Edit Mode, select edges, then **Cut** / **Fold** / **Auto**.
This is the whole reason to be in Blender: you inherit Alt-click edge loops,
Ctrl-click shortest path, box and lasso select, Select Sharp Edges, and undo,
none of which had to be written. Unfold again to see the effect. Marked edges
draw red (forced cut) and blue (forced fold).

**The 2D view is the UV editor.** An isometric unfold *is* a UV
parameterisation, and UVs are stored per-loop — exactly the storage an unfold
needs, because cuts duplicate vertices and a shared corner has two different
flat positions. So the net lands somewhere Blender already knows how to
display and edit, with island selection and transforms for free.

**Pack Net (UV)** runs Blender's own `uv.pack_islands` with
`shape_method='CONCAVE'` (true shape, not bounding boxes),
`rotate_method='CARDINAL'` (90° steps) and — importantly — **`scale=False`**.
That last flag is what keeps the unfold isometric. It is a genuinely good
nester and it comes free with Blender.

> **Move and rotate islands freely. Do not scale them.** Scaling breaks the
> isometry and the parts stop mating. The add-on's own operators are
> rigid-only; Blender's `S` key is not.

**Export Sheets** writes layered SVG (+ DXF) and a `manifest.json`, one file
per sheet.

## Two things worth knowing

**Seam marks live in mesh attributes, not properties.** Blender's edit-mode
undo records mesh and bmesh changes only — custom properties set while in Edit
Mode are *not* restored by Ctrl+Z (confirmed by Blender devs as by-design, not
a bug). Marks stored in a PropertyGroup would survive right up until the first
undo and then silently desynchronise. So they're stored as an integer edge
attribute `cbu_constraint`, which is part of the mesh and undoes correctly.
Island assignment is a face attribute `cbu_island`. Both survive save/load and
object duplication for free.

**The UV layer is a view, not the source of truth.** Blender stores UVs as
float32; measured round-trip error is ~9e-5 mm at 500 mm coordinates — 90
nanometres, roughly 3000× under a 0.3 mm kerf. Harmless, but it means the
exported SVG/DXF is generated from the core's float64 geometry, never read
back from the UVs. The core's own isometry error is ~1e-15.

## Debugging

### Where output actually goes

This trips everyone up once:

| | Shows |
|---|---|
| `self.report()` | Status bar + Info editor. One-line results only. |
| `print()`, tracebacks, the add-on's log | **System console** — a *different* window from Blender's Python Console |
| Python Console (Scripting workspace) | Interactive poking. Does **not** capture operator output. |

Reaching the system console: **Windows** → `Window ▸ Toggle System Console`.
**macOS / Linux** → there is no menu item; you must have launched Blender from
a terminal:

```bash
/Applications/Blender.app/Contents/MacOS/Blender      # macOS
blender                                               # Linux
```

If you launched from an icon and don't want to restart, turn on **Also log to
file** in the Debug panel — the log lands next to your output files as
`cbu.log`.

### The fast loop: symlink + Reload

Installing the zip *copies* the code, so every edit needs a reinstall. Link it
instead and the file you edit is the file Blender runs:

```bash
python3 blender/dev_install.py          # auto-detects your Blender version
```

Then: edit → **Cardboard ▸ Debug ▸ Reload Add-on** → run it again. No restart,
scene preserved. Reload takes ~30 ms and covers the vendored core too.

One trap it handles for you: Python decides a `.pyc` is still valid from the
source's mtime **in whole seconds** plus its size. Edit a file, reload, then
edit again within the same second without changing its length — flipping
`0.75` to `0.61`, say — and the cached bytecode still looks valid, so your
reload silently runs the **old** code. `Reload Add-on` purges `__pycache__`
first. (This bit me while testing, which is why it's handled.)

### The faster loop: don't open Blender at all

Most bugs — unfolding, fold geometry, packing, export — are reachable
headlessly and turn around in seconds:

```bash
python3 blender/watch.py            # re-runs core tests on every save
python3 blender/watch.py --addon    # also the headless Blender suite
```

`watch.py` also keeps `blender/cbu_blender/vendor/` in sync with the core, so
you can edit `cardboard_unfolder/` directly and both loops see it.

### Debug panel

- **Reload Add-on** — described above
- **Run Self-Test** — unfolds a known cube in memory and checks isometry,
  area and part generation. Two seconds; run it after any reload that touched
  the core
- **Copy Debug Info** — JSON snapshot to the clipboard: versions, dependency
  versions, every setting, mesh stats, per-island results, timings. Paste it
  when something misbehaves
- **Timings** — per-stage wall clock from the last unfold, so a slow stage is
  obvious as features land
- **Log level** — `WARNING` default; `INFO` for stage progress; `DEBUG` for
  per-island detail

### Step debugging in VS Code

Debug panel → **Start Debug Server** (needs `debugpy`; the dependency
installer includes it), then attach:

```json
{ "name": "Attach to Blender", "type": "debugpy", "request": "attach",
  "connect": { "host": "127.0.0.1", "port": 5678 } }
```

Breakpoints in the add-on and in the vendored core both work. Note that
Blender's UI blocks while you're stopped at a breakpoint — that's normal, not
a hang.

### Interactive poking

The Scripting workspace console can reach live state:

```python
import cbu_blender
from cbu_blender import ops, bridge, dev
pm, islands, parts, sheets, rep = ops._LAST[C.active_object.name]
islands[0].size(), islands[0].area()          # mm, mm2
[p.warnings for p in parts]
dev.state_dict(C)["timings_ms"]
bridge.read_constraints(C.active_object.data) # forced cuts, folds
```

### Adding a feature without breaking things

1. Write it in `cardboard_unfolder/` (the `bpy`-free core), with a test in
   `tests/`. `watch.py` gives you a sub-second loop.
2. Only then surface it in the add-on — a property in `props.py`, a control in
   `ui.py`, a call in `ops.py`.
3. Wrap any expensive stage in `with cbulog.stage(T, "name"):` so it shows up
   in the Timings box.
4. Run all three suites before you cut cardboard:

```bash
python3 -m pytest tests -q        # 87 core invariants
python3 blender/test_addon.py     # 38 checks against the real Blender API
python3 blender/test_dev.py       # 29 checks on the debug tooling itself
```

## Layout

```
blender/
  dev_install.py  symlink the add-on into Blender (live editing)
  watch.py        headless test loop, auto-syncs vendor/
  test_addon.py   real-Blender API tests
  test_dev.py     tests for the debug tooling
  cbu_blender/
    __init__.py   registration + load_post cache clearing
    props.py      settings (Scene PropertyGroup - safe, never edited in Edit Mode)
    bridge.py     bmesh <-> core; attribute and UV storage
    ops.py        operators
    ui.py         panels (3D sidebar + UV editor sidebar)
    overlay.py    GPU island tinting; fails soft if the GPU is unavailable
    log.py        logger + stage timing
    dev.py        hot reload, self-test, state dump, debugpy
    deps.py       dependency check + installer
    vendor/
      cardboard_unfolder/   the core - zero bpy imports
```

The core has no `bpy` anywhere and is unit-tested on its own (87 tests). That
discipline is the escape hatch: if Blender's API churn becomes intolerable the
same core drives a standalone tool without a rewrite.

## Testing

`blender/test_addon.py` runs against the real Blender API via the `bpy` PyPI
module — operator registration, bmesh round trips, attributes, UV layers,
`pack_islands`, and export — not a mock:

```bash
pip install bpy==4.5.13
python3 blender/test_addon.py
```

## Not here yet

- **Editing the net from the 2D side.** You can move and rotate islands, but
  re-seaming still happens in the 3D viewport. Per-edge picking in the Image
  Editor is the next step; PolyZamboni's `operators.py` has a working
  reference for the plumbing (`region.view2d` mapping, hover, G/R modals).
- **Live re-unfold** on seam change — currently you press Unfold again.
- **Bundled wheels.** A 4.2+ extension manifest can ship `shapely` et al. per
  platform, removing the install step. Needs a wheel set per OS.
