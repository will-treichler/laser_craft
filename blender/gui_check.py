"""Drive the real, windowed Unfold -> modal -> progress -> done path.

Nothing headless can reach it: invoke needs a window, the progress bar needs
a panel to draw in, the finished popup needs a screen - so test_addon.py can
assert on the code objects but never on what the user actually sees. This
opens Blender with a UI, starts an optimised unfold the way a click would,
screenshots it mid-run and after, writes a log, and quits.

    blender --factory-startup --python blender/gui_check.py -- out/gui

Look at gui_mid.png and gui_done.png. One thing it does that the add-on never
does: the sidebar tab cannot be switched from Python (`active_panel_category`
is read-only), so the panels are moved onto the tab that is already showing,
for the screenshot only.
"""
import sys
import pathlib
import bpy

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = (pathlib.Path(sys.argv[sys.argv.index("--") + 1])
       if "--" in sys.argv else REPO / "out" / "gui")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "blender"))
import cbu_blender                                      # noqa: E402
from cbu_blender import ops                             # noqa: E402

if not hasattr(bpy.types.Scene, "cbu"):   # the installed copy may be enabled already
    cbu_blender.register()

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2)
obj = bpy.context.active_object
obj.scale = (0.1,) * 3
bpy.ops.object.transform_apply(scale=True)
s = bpy.context.scene.cbu
s.optimise = True
s.trials = 30
s.budget = 6

state = {"started": False, "mid": False, "done_at": None, "t": 0.0,
         "log": []}


def view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                region = next(r for r in area.regions if r.type == "WINDOW")
                return win, area, region
    return None, None, None


def shot(name):
    win, area, region = view3d()
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.screen.screenshot(filepath=str(OUT / name))
    state["log"].append(name)


def tick():
    state["t"] += 0.25
    win, area, region = view3d()
    if not state["started"] and state["t"] >= 1.5:
        # Open the sidebar on our tab, where the bar is drawn.
        area.spaces.active.show_region_ui = True
        # The sidebar tab cannot be switched from Python (read-only), so for
        # the screenshot only, move our 3D-view panels onto the tab that IS
        # showing. Harness-only; nothing in the add-on does this.
        from cbu_blender import ui as cui
        panels = [c for c in cui.classes
                  if getattr(c, "bl_space_type", "") == "VIEW_3D"]
        for c in reversed(panels):
            bpy.utils.unregister_class(c)
        for c in panels:
            c.bl_category = "Item"
            bpy.utils.register_class(c)
        state["log"].append(f"moved {len(panels)} panel(s) to Item")
        with bpy.context.temp_override(window=win, area=area, region=region):
            res = bpy.ops.cbu.unfold("INVOKE_DEFAULT")
        state["log"].append(f"invoke -> {res}")
        state["started"] = True
        return 0.25
    if state["started"] and ops.RUNNING is not None and not state["mid"] \
            and ops.RUNNING.elapsed >= 1.0:
        shot("gui_mid.png")
        state["log"].append(f"mid: {ops.RUNNING.note} "
                            f"{100 * ops.RUNNING.fraction:.0f}% "
                            f"{ops.RUNNING.elapsed:.1f}s")
        state["mid"] = True
    if state["started"] and ops.RUNNING is None and state["done_at"] is None:
        state["done_at"] = state["t"]
    if state["done_at"] is not None and state["t"] - state["done_at"] >= 1.0:
        shot("gui_done.png")
        state["log"].append(f"report: {s.last_report} | {s.last_detail} | "
                            f"optimised={s.last_optimised} "
                            f"elapsed={s.last_elapsed:.1f}")
        (OUT / "gui_log.txt").write_text("\n".join(state["log"]))
        bpy.ops.wm.quit_blender()
        return None
    if state["t"] > 240:
        state["log"].append("TIMEOUT")
        (OUT / "gui_log.txt").write_text("\n".join(state["log"]))
        bpy.ops.wm.quit_blender()
        return None
    return 0.25


bpy.app.timers.register(tick, first_interval=0.5)
