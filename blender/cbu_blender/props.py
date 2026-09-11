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
    tab_depth: FloatProperty(name="Tab depth", default=12.0, min=0.0, max=60.0)

    # --- machine ---
    bed_x: FloatProperty(name="Bed X", default=600.0, min=50.0)
    bed_y: FloatProperty(name="Bed Y", default=400.0, min=50.0)
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

    show_overlay: BoolProperty(name="Show islands", default=True)
    show_bed: BoolProperty(
        name="Show bed", default=True,
        description="Draw the laser bed, at true size, behind the net in the "
                    "UV editor. The net is in millimetres, so islands can be "
                    "read straight against it")
    overlay_alpha: FloatProperty(name="Opacity", default=0.55, min=0.0, max=1.0)

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
    m.bed = (s.bed_x, s.bed_y)
    m.sheet_margin = s.margin
    m.part_gap = s.part_gap
    return m


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
