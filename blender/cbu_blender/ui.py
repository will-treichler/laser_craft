"""Sidebar panels. 3D viewport for the workflow; UV/Image editor for the net."""
from __future__ import annotations

import bpy

from . import bridge, deps, log as cbulog

CAT = "Cardboard"


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CAT


class CBU_PT_main(_Base, bpy.types.Panel):
    bl_label = "Cardboard Unfolder"
    bl_idname = "CBU_PT_main"

    def draw(self, context):
        lay = self.layout
        s = context.scene.cbu

        missing = deps.missing()
        if missing:
            box = lay.box()
            box.alert = True
            box.label(text="Missing Python packages", icon="ERROR")
            box.label(text=", ".join(missing))
            box.operator("cbu.install_deps", icon="IMPORT")
            box.label(text="Blender restart may be needed afterwards.")
            return

        obj = context.active_object
        if obj is None or obj.type != "MESH":
            lay.label(text="Select a mesh object", icon="INFO")
            return

        col = lay.column(align=True)
        col.scale_y = 1.4
        col.operator("cbu.unfold", icon="MOD_UVPROJECT")
        # A bent face has no flat image, so it has to be triangulated in the
        # mesh - the core cannot paper over it. Offered next to Unfold because
        # that is where the warning about it appears.
        lay.operator("cbu.split_nonplanar", icon="MOD_TRIANGULATE")

        if s.last_report:
            box = lay.box()
            box.label(text=s.last_report, icon="CHECKMARK")
            if s.last_warnings:
                for w in s.last_warnings.split(" | "):
                    if w:
                        r = box.row()
                        r.alert = True
                        r.label(text=w, icon="ERROR")

        row = lay.row(align=True)
        row.operator("cbu.pack_uv", icon="UV_ISLANDSEL")
        row.operator("cbu.export", icon="EXPORT")


class CBU_PT_seams(_Base, bpy.types.Panel):
    bl_label = "Seams"
    bl_parent_id = "CBU_PT_main"

    def draw(self, context):
        lay = self.layout
        if context.mode != "EDIT_MESH":
            lay.label(text="Enter Edit Mode to mark seams", icon="INFO")
            lay.label(text="Alt-click selects an edge loop")
        else:
            row = lay.row(align=True)
            row.operator("cbu.mark", text="Cut", icon="UNLINKED").mode = "CUT"
            row.operator("cbu.mark", text="Fold", icon="LINKED").mode = "FOLD"
            row.operator("cbu.mark", text="Auto").mode = "AUTO"
            lay.operator("cbu.select_marked", icon="RESTRICT_SELECT_OFF")
        lay.operator("cbu.clear_marks", icon="X")


class CBU_PT_material(_Base, bpy.types.Panel):
    bl_label = "Material"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.use_property_split = True
        lay.prop(s, "flute")
        lay.prop(s, "thickness")
        lay.prop(s, "kerf")
        lay.prop(s, "crush")
        lay.separator()
        lay.prop(s, "use_tabs")
        sub = lay.column()
        sub.enabled = s.use_tabs
        sub.prop(s, "tab_depth")
        lay.separator()
        lay.label(text="Every value here is empirical.", icon="INFO")
        lay.label(text="Cut a calibration set before trusting it.")


class CBU_PT_folds(_Base, bpy.types.Panel):
    bl_label = "Folds"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.use_property_split = True
        lay.prop(s, "relief_adaptive")
        sub = lay.column()
        sub.enabled = not s.relief_adaptive
        sub.prop(s, "relief_fixed")
        lay.separator()
        lay.prop(s, "perf_cut")
        lay.prop(s, "connect_ratio")
        lay.prop(s, "min_connect")
        lay.prop(s, "round_cut_ends")


class CBU_PT_machine(_Base, bpy.types.Panel):
    bl_label = "Machine & Size"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.use_property_split = True
        r = lay.row(align=True)
        r.prop(s, "bed_x", text="Bed")
        r.prop(s, "bed_y", text="")
        lay.prop(s, "margin")
        lay.prop(s, "part_gap")
        lay.separator()
        lay.prop(s, "mm_per_unit")
        lay.prop(s, "target_height")
        lay.prop(s, "apply_transform")


class CBU_PT_solver(_Base, bpy.types.Panel):
    bl_label = "Solver"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.use_property_split = True
        lay.prop(s, "merge_coplanar")
        sub = lay.column()
        sub.enabled = s.merge_coplanar
        sub.prop(s, "merge_tol")
        lay.prop(s, "trials")
        lay.prop(s, "seed")
        lay.prop(s, "limit_to_bed")
        lay.separator()
        lay.prop(s, "show_overlay")
        lay.prop(s, "overlay_alpha")
        lay.prop(s, "show_bed")
        lay.separator()
        lay.prop(s, "out_dir")
        lay.prop(s, "write_dxf")


class CBU_PT_debug(_Base, bpy.types.Panel):
    bl_label = "Debug"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout

        col = lay.column(align=True)
        col.scale_y = 1.2
        col.operator("cbu.reload", icon="FILE_REFRESH")
        row = lay.row(align=True)
        row.operator("cbu.selftest", icon="CHECKMARK")
        row.operator("cbu.dump_state", icon="COPYDOWN")

        if s.last_timings:
            box = lay.box()
            box.label(text="Timings", icon="TIME")
            for part in s.last_timings.split("  "):
                if part.strip():
                    box.label(text=part.strip())

        lay.separator()
        lay.use_property_split = True
        lay.prop(s, "log_level")
        lay.prop(s, "log_to_file")

        lay.separator()
        box = lay.box()
        box.label(text="Step debugging", icon="CONSOLE")
        box.prop(s, "debug_port")
        box.operator("cbu.debugpy", icon="PLAY")
        box.label(text="Then attach from VS Code.")

        lay.separator()
        box = lay.box()
        box.label(text="print() goes to the SYSTEM console:", icon="INFO")
        box.label(text="Windows: Window > Toggle System Console")
        box.label(text="macOS/Linux: launch Blender from a terminal")


def s_cbu(context):
    return context.scene.cbu


class CBU_PT_uv(bpy.types.Panel):
    """The net lives in a UV layer, so the UV editor is the 2D view."""
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = CAT
    bl_label = "Cardboard Net"

    @classmethod
    def poll(cls, context):
        return context.space_data.mode == "UV"

    def draw(self, context):
        lay = self.layout
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            lay.label(text="Select a mesh object", icon="INFO")
            return
        if bridge.UV_LAYER not in obj.data.uv_layers:
            lay.label(text="Nothing unfolded yet", icon="INFO")
            lay.operator("cbu.unfold", icon="MOD_UVPROJECT")
            return
        s = s_cbu(context)
        mm = 1.0 / bridge.uv_scale(s.bed_x, s.bed_y)
        lay.label(text=f'UV layer "{bridge.UV_LAYER}" is the net')
        lay.label(text=f"1 UV unit = {mm:.0f} mm  (bed {s.bed_x:.0f}x{s.bed_y:.0f})")
        row = lay.row(align=True)
        row.prop(s, "show_bed")
        row.prop(s, "show_cut")

        # UV Sync Selection defaults ON in Blender 5.x, and in that mode the UV
        # editor moves the loops of *selected faces* rather than whole islands.
        # Dragging then pulls single faces out of the net and scatters them,
        # which reads as the net corrupting itself.
        ts = context.scene.tool_settings
        if not ts.use_uv_select_sync:
            # with sync off the UV editor only draws the UVs of SELECTED faces,
            # so an empty selection shows an empty net and nothing can be picked
            me = obj.data
            if not any(p.select for p in me.polygons):
                box = lay.box()
                box.alert = True
                box.label(text="Nothing selected", icon="ERROR")
                box.label(text="With UV Sync off the editor only")
                box.label(text="shows selected faces. In Edit Mode")
                box.label(text="press A to select all.")
                box.operator("cbu.select_all_faces", icon="SELECT_EXTEND")
        if ts.use_uv_select_sync:
            box = lay.box()
            box.alert = True
            box.label(text="UV Sync is on", icon="ERROR")
            box.label(text="Dragging moves selected faces,")
            box.label(text="not whole islands - it will pull")
            box.label(text="the net apart. Turn it off:")
            box.prop(ts, "use_uv_select_sync", text="UV Sync Selection")

        box = lay.box()
        box.label(text="Selected edge", icon="EDGESEL")
        if context.mode != "EDIT_MESH":
            box.label(text="Enter Edit Mode to pick edges", icon="INFO")
        else:
            row = box.row(align=True)
            op = row.operator("cbu.edit_seam", text="Cut", icon="UNLINKED")
            op.mode = "CUT"
            op.separate = s.separate_on_cut
            op = row.operator("cbu.edit_seam", text="Weld", icon="LINKED")
            op.mode = "FOLD"
            box.prop(s, "separate_on_cut")
            if not context.scene.tool_settings.use_uv_select_sync:
                box.label(text="reading the UV selection", icon="UV")
            else:
                box.label(text="UV sync on: reading the 3D selection",
                          icon="INFO")
        if s.separate_on_cut:
            box.label(text="Cut detaches that piece; the rest stays put.")
        else:
            box.label(text="Cut re-unfolds; the freed piece may")
            box.label(text="reattach elsewhere and the layout moves.")

        lay.separator()
        col = lay.column(align=True)
        col.operator("cbu.repack", icon="MOD_ARRAY")
        col.operator("cbu.pack_uv", icon="UV_ISLANDSEL")
        lay.operator("cbu.export", icon="EXPORT")

        box = lay.box()
        box.label(text="Move and rotate islands freely.", icon="INFO")
        box.label(text="Do NOT scale - that breaks the")
        box.label(text="isometry and parts stop mating.")


classes = (CBU_PT_main, CBU_PT_seams, CBU_PT_material, CBU_PT_folds,
           CBU_PT_machine, CBU_PT_solver, CBU_PT_debug, CBU_PT_uv)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
