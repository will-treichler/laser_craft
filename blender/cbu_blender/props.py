"""Settings. These live on the Scene as a PropertyGroup, which is safe -
unlike seam marks, they are never edited from inside Edit Mode, so the
edit-mode undo limitation doesn't apply to them."""
from __future__ import annotations

import os

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, StringProperty, FloatVectorProperty)

from . import log as cbulog

_LOG_LEVELS = cbulog.LEVELS


def _apply_log(s) -> None:
    cbulog.set_level(s.log_level)
    if s.log_to_file:
        d = bpy.path.abspath(s.out_dir) or "."
        os.makedirs(d, exist_ok=True)
        cbulog.set_file(os.path.join(d, "cbu.log"))
    else:
        cbulog.set_file(None)

FLUTE_ITEMS = [
    ("F", "F  (1.0 mm)", "Fine, behaves like thick paperboard"),
    ("E", "E  (1.5 mm)", "Detail work; folds from a plain perforation"),
    ("B", "B  (3.0 mm)", "What shipping boxes are made of - the sweet spot"),
    ("C", "C  (4.0 mm)", "Stiffer, but every relief is 33% wider"),
    ("BC", "BC (6.5 mm)", "Double wall - only for very large parts"),
]


def rewrite_net(self, context):
    """Re-emit the 2D net when the bed changes.

    uv_scale is derived from the bed, so the *same* stored UVs mean different
    millimetres after the bed is changed - a net laid out on a 300 mm bed reads
    as half size on a 600 mm one. The bed reference redraws at the new size
    immediately, so without this the view shows the net sitting outside a
    margin that the packer did respect. Positions in mm are untouched; only
    the scale they are stored at changes. The layout itself is still for the
    old bed, which is what Re-pack is for.
    """
    obj = context.active_object if context else None
    if obj is None or obj.type != "MESH":
        return
    try:
        from . import bridge, overlay, ops
        cached = ops._LAST.get(obj.name)
        if not cached:
            return
        pm, islands, parts, _sheets, _rep = cached
        if obj.mode != "OBJECT":
            return                      # uv_layers data is empty in Edit Mode
        mat = material_from(self)
        bridge.write_net_uv(obj.data, pm, islands,
                            scale=bridge.uv_scale(*mat.bed),
                            placement={p.island_id: p for p in parts},
                            sheet_height=mat.bed[1])
        overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data))
        overlay.refresh_live_cuts()
    except Exception:                                   # noqa: BLE001
        pass                    # a settings tweak must never break the session


class CBUSettings(bpy.types.PropertyGroup):
    # --- material ---
    flute: EnumProperty(name="Flute", items=FLUTE_ITEMS, default="B")
    thickness: FloatProperty(
        name="Thickness", default=3.0, min=0.2, max=15.0, unit="NONE",
        description="Board caliper in mm. Calipers squeeze corrugated and "
                    "read 0.2-0.5 mm low; prefer the flute preset")
    kerf: FloatProperty(
        name="Kerf", default=0.30, min=0.0, max=2.0,
        description="mm. Measure it: cut a 20 mm square, "
                    "kerf = (hole - dropout) / 2")
    crush: FloatProperty(
        name="Crush factor", default=0.75, min=0.4, max=1.0,
        description="Corrugated squashes into the fold channel, so the "
                    "theoretical relief width is conservative. Empirical - "
                    "0.6 soft or recycled board, 1.0 stiff or double wall")
    use_tabs: BoolProperty(
        name="Glue tabs", default=False,
        description=("Add glue flaps along the seams. Off for tape or a glued "
                     "backing strip, which is how most cardboard goes "
                     "together - the flaps cost about a fifth of the sheet"))
    tab_depth: FloatProperty(name="Tab depth", default=12.0, min=0.0, max=60.0)

    # --- fold relief ---
    relief_adaptive: BoolProperty(
        name="Adaptive cut width", default=True,
        description=("Width from board thickness and bend angle, which is what "
                     "the board needs to close. Off uses a fixed width, which "
                     "is what you want when cutting to a known blade width"))
    relief_fixed: FloatProperty(
        name="Cut width", default=3.0, min=0.1, max=30.0,
        description="Fixed relief width, used when Adaptive is off")
    perf_cut: FloatProperty(
        name="Perforation length", default=1.0, min=0.1, max=50.0,
        description="Length of each cut along a fold line")
    connect_ratio: FloatProperty(
        name="Connect ratio", default=0.5, min=0.0, max=0.95,
        description=("Fraction of the fold line left connected. 0.5 cuts half "
                     "the material away; higher is a stiffer fold"))
    min_connect: FloatProperty(
        name="Min connect width", default=10.0, min=0.0, max=100.0,
        description=("Material that must survive a fold, in total. A blanket "
                     "ratio leaves a short joint with almost nothing holding "
                     "it; this floor is what stops it tearing off"))
    round_cut_ends: BoolProperty(
        name="Radius cut ends", default=True,
        description=("Round the ends of relief slots. A square internal corner "
                     "concentrates stress and is where a fold starts to tear"))
    corner_relief: BoolProperty(
        name="Reach sharp corners", default=True,
        description=("Run a slit down the bisector of a sharp notch so the cut "
                     "reaches its apex. Growing the part by half a kerf cannot: "
                     "it fills the notch, leaving material at the apex that "
                     "stops the fold - 0.9 mm at a 20 degree notch, 1.7 mm at "
                     "10, and metres of it at a near-pinch"))
    corner_overcut: FloatProperty(
        name="Corner overcut", default=0.0, min=0.0, max=5.0,
        description=("Run the corner slit this far PAST the apex, in mm. 0 "
                     "reaches it exactly, which is enough for B flute; thicker "
                     "board folding sharply may want a little more"))

    # --- machine ---
    bed_x: FloatProperty(name="Bed X", default=600.0, min=50.0,
                         update=rewrite_net)
    bed_y: FloatProperty(name="Bed Y", default=400.0, min=50.0,
                         update=rewrite_net)
    margin: FloatProperty(name="Sheet margin", default=8.0, min=0.0)
    part_gap: FloatProperty(
        name="Part gap", default=3.0, min=0.0,
        description="mm between parts. The char halo sets this, not the "
                    "kerf - 1 mm will scorch the web between neighbours")

    # --- size ---
    mm_per_unit: FloatProperty(
        name="mm per unit", default=1000.0, min=0.0001,
        description="Blender units to millimetres. 1000 if 1 unit = 1 metre")
    target_height: FloatProperty(
        name="Target height", default=0.0, min=0.0,
        description="Scale the model to this height in mm. 0 = use the "
                    "object's real size")
    apply_transform: BoolProperty(name="Apply object transform", default=True)

    # --- unfolding ---
    merge_coplanar: BoolProperty(
        name="Merge coplanar faces", default=True,
        description="Group coplanar triangles into n-gons. Without this an "
                    "imported STL gets a fold line down every face diagonal")
    merge_tol: FloatProperty(name="Coplanar tolerance", default=0.5,
                             min=0.0, max=20.0, description="degrees")
    trials: IntProperty(
        name="Trials", default=250, min=1, max=5000,
        description="Randomised restarts before falling back to splitting")
    seed: IntProperty(name="Seed", default=0, min=0,
                      description="Change this if a net comes out badly pinched")
    limit_to_bed: BoolProperty(
        name="Split to fit bed", default=True,
        description="Cut extra seams, along fold edges, until every island "
                    "fits the laser bed")

    # --- output ---
    out_dir: StringProperty(name="Output", subtype="DIR_PATH", default="//cbu_out")
    write_dxf: BoolProperty(name="Also write DXF", default=True)

    # --- overlay ---
    separate_on_cut: BoolProperty(
        name="Cut separates", default=True,
        description=("Cut detaches the piece on the far side as its own "
                     "island and leaves the rest of the net where it is. "
                     "Off re-runs the solver, which may reattach that piece "
                     "to a different island instead"))

    show_cut: BoolProperty(
        name="Show cut outline", default=True,
        description=("Draw what the laser actually follows - the islands plus "
                     "their glue tabs and kerf. The UV net alone shows only "
                     "the islands, which is about 20% smaller"))

    show_overlay: BoolProperty(
        name="Show islands", default=True,
        description=("Draw the island annotation in the 3D viewport. Off "
                     "hides the tint and the forced seam marks together"))
    overlay_fill: BoolProperty(
        name="Tint faces", default=True,
        description=("Fill each island with its colour. Off keeps the forced "
                     "seam marks but leaves the mesh itself unpainted, so the "
                     "selection highlight and the shading stay readable"))
    show_bed: BoolProperty(
        name="Show bed", default=True,
        description="Draw the laser bed, at true size, behind the net in the "
                    "UV editor. The net is in millimetres, so islands can be "
                    "read straight against it")
    overlay_alpha: FloatProperty(
        name="Opacity", default=0.55, min=0.0, max=1.0,
        description="How strongly the island tint covers the mesh",
        update=restyle_overlay)

    # --- last run, for the panel readout ---
    last_report: StringProperty(default="")
    last_islands: IntProperty(default=0)
    last_sheets: IntProperty(default=0)
    last_warnings: StringProperty(default="")
    last_timings: StringProperty(default="")

    # --- debugging ---
    log_level: EnumProperty(
        name="Log level",
        items=[(k, lbl, desc) for k, lbl, desc in _LOG_LEVELS],
        default="WARNING", update=lambda self, ctx: _apply_log(self))
    log_to_file: BoolProperty(
        name="Also log to file", default=False,
        update=lambda self, ctx: _apply_log(self),
        description="Write the log beside the output files. The only way to "
                    "see output when Blender was launched from a desktop icon "
                    "rather than a terminal")
    debug_port: IntProperty(name="Debug port", default=5678, min=1024, max=65535)


def material_from(s: CBUSettings):
    from .vendor.cardboard_unfolder.material import Material
    m = Material.from_flute(s.flute)
    m.thickness = s.thickness
    m.kerf = s.kerf
    m.crush = s.crush
    m.tab_depth = s.tab_depth
    m.use_tabs = s.use_tabs
    m.relief_adaptive = s.relief_adaptive
    m.relief_fixed = s.relief_fixed
    m.perf_cut = s.perf_cut
    m.connect_ratio = s.connect_ratio
    m.min_connect = s.min_connect
    m.round_cut_ends = s.round_cut_ends
    m.corner_relief = s.corner_relief
    m.corner_overcut = s.corner_overcut
    m.bed = (s.bed_x, s.bed_y)
    m.sheet_margin = s.margin
    m.part_gap = s.part_gap
    return m


def restyle_overlay(self, context):
    """Re-colour the viewport annotation when Opacity moves.

    A re-colour of cached geometry - no mesh access, nothing that is unsafe to
    do from a property update. Without this the slider did nothing until the
    next unfold, because the alpha was baked into the vertex colours.
    """
    try:
        from . import overlay
        overlay.set_alpha(self.overlay_alpha)
    except Exception:                                       # noqa: BLE001
        pass                    # a settings tweak must never break the session


def sync_thickness(self, context):
    from .vendor.cardboard_unfolder.material import FLUTES
    context.scene.cbu.thickness = FLUTES[context.scene.cbu.flute]["thickness"]


classes = (CBUSettings,)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.cbu = bpy.props.PointerProperty(type=CBUSettings)


def unregister():
    # Defensive: the hot-reload loop calls this repeatedly, and a bare `del`
    # on a missing attribute would abort the reload halfway - leaving Blender
    # with nothing registered and no obvious way back.
    if hasattr(bpy.types.Scene, "cbu"):
        del bpy.types.Scene.cbu
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
