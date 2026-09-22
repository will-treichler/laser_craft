"""Sidebar panels. 3D viewport for the workflow; UV/Image editor for the net."""
from __future__ import annotations

import bpy

from . import bridge, deps, props, log as cbulog

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
        # Beside the button it modifies, not in the closed Solver sub-panel
        # next to the output directory. It went there first and the only
        # feedback was "how do I run the optimiser?" - a mode for Unfold that
        # you cannot see from Unfold is a mode nobody runs. Same mistake as
        # show_overlay, which lived in that panel and hid the seam marks.
        lay.prop(s, "optimise", toggle=True, icon="MOD_ARRAY")
        if s.optimise:
            lay.label(text="Unfold is slower; ESC stops it", icon="INFO")
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

        # These are view controls, not solver settings - they were buried in a
        # closed sub-panel next to the output directory, which is not where you
        # look when the annotation is in the way of the thing you are editing.
        lay.separator()
        row = lay.row(align=True)
        row.prop(s, "show_overlay", text="", icon="HIDE_OFF" if s.show_overlay
                 else "HIDE_ON")
        sub = row.row(align=True)
        sub.enabled = s.show_overlay
        sub.prop(s, "overlay_fill", toggle=True, icon="SHADING_SOLID")
        sub.prop(s, "overlay_alpha", text="", slider=True)


def mark_key(lay, s, cells: int = 2):
    """The colour key for forced marks, beside the buttons that make them.

    Real swatches, not icons: an icon is tinted by the theme and reads as a
    letter with a smudge next to it, which is not a key. These are the very
    values the overlay draws with (`overlay.mark_colors`), so the legend
    cannot drift from the thing it explains - and they can be changed if the
    defaults clash with a model's own colours.

    A thin strip directly under each button, not a labelled colour field: the
    button already says which is which, so the swatch only has to say what
    colour that is. `cells` is how many buttons are above it, so the strips
    line up with the two they belong to.
    """
    row = lay.row(align=True)
    row.scale_y = 0.4              # a strip under the button, not a form field
    row.prop(s, "mark_cut_color", text="")
    row.prop(s, "mark_fold_color", text="")
    for _ in range(cells - 2):     # blanks so the strips stay under their own
        row.label(text="")         # button when there are more than two


class CBU_PT_seams(_Base, bpy.types.Panel):
    bl_label = "Seams"
    bl_parent_id = "CBU_PT_main"

    def draw(self, context):
        lay = self.layout
        s = context.scene.cbu
        if context.mode != "EDIT_MESH":
            lay.label(text="Enter Edit Mode to mark seams", icon="INFO")
            lay.label(text="Alt-click selects an edge loop")
        else:
            # The swatch IS the key: these are the colours the mark is drawn
            # in, in the viewport and over the net. COLOR_RED / COLOR_BLUE are
            # Blender's own swatch icons, so the button carries its own legend
            # and there is nothing to remember.
            row = lay.row(align=True)
            row.operator("cbu.mark", text="Cut", icon="COLOR_RED").mode = "CUT"
            row.operator("cbu.mark", text="Fold",
                         icon="COLOR_BLUE").mode = "FOLD"
            row.operator("cbu.mark", text="Auto",
                         icon="RADIOBUT_OFF").mode = "AUTO"
            lay.operator("cbu.select_marked", icon="RESTRICT_SELECT_OFF")
        mark_key(lay, s, cells=3)
        lay.operator("cbu.clear_marks", icon="X")

        # A lock says nothing about the mark - it says the OPTIMISER may not
        # change it. Clear All Marks is a hand edit and still clears a locked
        # edge; locking only ever constrains the solver.
        box = lay.box()
        box.label(text="Optimiser", icon="LOCKED")
        if context.mode == "EDIT_MESH":
            row = box.row(align=True)
            row.operator("cbu.lock", text="Lock", icon="LOCKED").lock = True
            row.operator("cbu.lock", text="Unlock",
                         icon="UNLOCKED").lock = False
            box.operator("cbu.select_locked", icon="RESTRICT_SELECT_OFF")
        box.label(text="Locked marks draw dashed")
        box.operator("cbu.clear_locks", icon="X")


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
        lay.separator()
        lay.prop(s, "edge_offset")
        # Nothing buries a notch apex without an OUTWARD offset, so the corner
        # controls have nothing to do at 0 or below.
        sub = lay.column()
        sub.enabled = s.edge_offset > 0
        sub.prop(s, "corner_relief")
        row = sub.column()
        row.enabled = s.corner_relief
        row.prop(s, "corner_overcut")


class CBU_PT_labels(_Base, bpy.types.Panel):
    bl_label = "Annotation"
    bl_parent_id = "CBU_PT_main"
    # Open. The Model ID box is here, and a collapsed panel is where settings
    # go to not be found.

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.use_property_split = True
        obj = context.active_object
        lay.prop(s, "model_id")
        if not (s.model_id or "").strip():
            eff = props.model_id_for(s, obj)
            lay.label(text=f"using: {eff}" if eff else "no id available",
                      icon="INFO")
        lay.prop(s, "label_height")
        lay.prop(s, "label_min_height")
        lay.prop(s, "label_edge_frac", slider=True)
        lay.prop(s, "label_folds")


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
        # The switch itself is on the main panel, beside Unfold; this is the
        # knob that only matters once it is on.
        sub = lay.column()
        sub.enabled = s.optimise
        sub.prop(s, "budget")
        lay.prop(s, "limit_to_bed")
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
        # Named for the SPACE it describes. The Size panel has "mm per scene
        # unit" at 1000 and this said "1 UV unit = 512 mm", and two numbers
        # both reading "mm per unit" in one add-on read as a contradiction.
        # They are different spaces: the 3D scene, and the UV editor's unit
        # square, which is sized by the bed (rounded up to a power of two, so
        # the scaling is exact) and never by the model.
        lay.label(text=f"UV editor scale: 1 UV unit = {mm:.0f} mm")
        lay.label(text=f"   (fits one {s.bed_x:.0f} x {s.bed_y:.0f} mm sheet; "
                       f"set by the bed, not the model)")
        # Stacked, not in a row. Four switches side by side truncate to
        # "Sho.." at any sane panel width, and the key that goes with the
        # marks belongs beside the buttons that make them, further down.
        col = lay.column(align=True)
        col.prop(s, "show_bed")
        sub = col.column(align=True)
        sub.enabled = s.show_bed
        sub.prop(s, "show_bed_ruler")
        col.prop(s, "show_cut")
        col.prop(s, "show_net_labels", text="Annotation")
        col.prop(s, "show_net_marks")

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
            else:
                # Sticky select defaults to Shared Location, which selects every
                # UV sitting at the same point - including ones belonging to a
                # different island, so dragging one island takes corners of its
                # neighbours with it. Island mode is what you want for layout.
                col = lay.column(align=True)
                # Island selection is a toggle in 5.x, not a value of
                # uv_select_mode - that enum is only VERTEX/EDGE/FACE.
                col.prop(ts, "use_uv_select_island", text="Island select")
                col.prop(ts, "uv_sticky_select_mode", text="Sticky")
                if (ts.uv_sticky_select_mode == "SHARED_LOCATION"
                        and not ts.use_uv_select_island):
                    warn = col.box()
                    warn.alert = True
                    warn.label(text="Sticky: Shared Location", icon="ERROR")
                    warn.label(text="drags corners of other islands")
                    warn.label(text="that sit at the same point.")
                    warn.label(text="Turn on Island select, or set")
                    warn.label(text="Sticky to Disabled.")
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
            # Same swatches as the Seams panel, because these write the same
            # marks: cbu.edit_seam calls cbu.mark, so Cut is the red forced
            # cut and Weld the blue forced fold.
            row = box.row(align=True)
            op = row.operator("cbu.edit_seam", text="Cut", icon="COLOR_RED")
            op.mode = "CUT"
            op = row.operator("cbu.edit_seam", text="Weld", icon="COLOR_BLUE")
            op.mode = "FOLD"
            mark_key(box, s)
            if not context.scene.tool_settings.use_uv_select_sync:
                box.label(text="reading the UV selection", icon="UV")
            else:
                box.label(text="UV sync on: reading the 3D selection",
                          icon="INFO")
        box.label(text="Both edit in place. The rest of the net",
                  icon="INFO")
        box.label(text="stays put; press Unfold to re-solve.")

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
           CBU_PT_labels, CBU_PT_machine, CBU_PT_solver,
           CBU_PT_debug, CBU_PT_uv)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
