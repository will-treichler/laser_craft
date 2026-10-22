"""Sidebar panels. 3D viewport for the workflow; UV/Image editor for the net."""
from __future__ import annotations

import bpy

from . import bridge, build, deps, props, log as cbulog

CAT = "Cardboard"


def _running():
    """The background solve, if one is in flight. Imported late: ops imports
    this module's panels, so the dependency cannot go the other way at the
    top of the file."""
    from . import ops
    return ops.RUNNING


# The phases a run goes through, in order, as (label, note prefix). The
# prefixes are the notes the core and Solve.run report with.
PHASES = (("search", "searching trees"),
          ("score", "scoring finalists"),
          ("pack", "packing sheets"))


def report_lines(report: str) -> list:
    """The one-line result split so it fits a sidebar. At any normal panel
    width the single line truncated to "2 shee...repair", which hides the
    number you ran it for."""
    parts = [p for p in report.split(" - ") if p]
    if len(parts) <= 2:
        return [", ".join(parts)] if parts else []
    return [", ".join(parts[:2])] + parts[2:]


def draw_progress(lay, job) -> None:
    """A real progress bar, where you are looking.

    It used to go only to the mouse-cursor readout and the status bar, which
    are small, at the edges, and easy to miss entirely - the user asked for
    progress while it was already being reported. `layout.progress` is
    Blender 4.0+; older builds get the same numbers as text.
    """
    box = lay.box()
    # Each phase counts from zero, so without a step number the bar filling
    # and then starting again reads as the run having restarted.
    # With several objects the note leads with whose search it is
    # ("Helmet: searching trees"), so look for the phase, not at the start.
    step = next((i for i, (_k, key) in enumerate(PHASES)
                 if key in (job.note or "")), None)
    head = "Stopping..." if job.stopping else "Optimising..."
    if step is not None:
        head += f"  step {step + 1} of {len(PHASES)}"
    box.label(text=head, icon="TIME")
    note = job.note or "starting"
    if job.total:
        text = f"{note}  ({100 * job.fraction:.0f}%)"
        if hasattr(box, "progress"):
            box.progress(factor=job.fraction, type="BAR", text=text)
        else:
            box.label(text=text)
    else:
        box.label(text=note)        # a phase with no count, e.g. packing
    box.label(text=f"{job.elapsed:.0f} s elapsed"
                   + ("" if job.stopping else "  -  ESC to stop"))


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CAT


class CBU_UL_job(bpy.types.UIList):
    """The saved list of objects cut together."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index=0, flt_flag=0):
        o = item.obj
        if o is None:
            layout.label(text="(deleted)", icon="ERROR")
            return
        layout.label(text=o.name, icon="OUTLINER_OB_MESH")
        from . import ops
        if o.name in ops._LAST:
            layout.label(text=f"{len(ops._LAST[o.name][1])} isl.")


def draw_job(lay, context) -> None:
    """What Unfold will cut: the saved list, or the active object.

    A list rather than the selection: what goes on the sheets together is a
    decision about the job, and re-selecting it before every Unfold is how
    half a costume gets cut. Empty is the old behaviour - the active object.
    """
    s = context.scene.cbu
    box = lay.box()
    if len(s.job):
        box.label(text=f"Cut together ({len(s.job)})", icon="GROUP")
        row = box.row()
        row.template_list("CBU_UL_job", "", s, "job", s, "job_index",
                          rows=2)
        side = row.column(align=True)
        side.operator("cbu.job_add", text="", icon="ADD")
        side.operator("cbu.job_remove", text="", icon="REMOVE")
        side.separator()
        side.operator("cbu.job_clear", text="", icon="X")
    else:
        obj = context.active_object
        row = box.row()
        row.label(text=f"Cutting: {obj.name}" if obj is not None
                  and obj.type == "MESH" else "Cutting: (nothing)",
                  icon="OUTLINER_OB_MESH")
        row.operator("cbu.job_add", text="Add", icon="ADD")
    # The size in one line, because it now depends on more than one box -
    # and those are in Machine & Size, a closed panel. Short: at sidebar
    # width a longer line truncated.
    if s.size_mode == "YARDSTICK":
        text = (f"Size: yardstick = {s.yardstick_length:.0f} mm"
                if s.yardstick is not None else "Size: yardstick not set")
    elif s.target_height > 0:
        text = f"Size: {s.target_height:.0f} mm tall"
    else:
        text = "Size: real size"
    box.label(text=text, icon="DRIVER_DISTANCE")


def job_ref(context):
    from . import ops
    return ops.job_ref(context)


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

        draw_job(lay, context)
        obj = job_ref(context)
        if obj is None:
            lay.label(text="Select a mesh object", icon="INFO")
            return

        col = lay.column(align=True)
        col.scale_y = 1.4
        col.operator("cbu.unfold", icon="MOD_UVPROJECT")
        # Two actions, two buttons. Optimise was first a switch in the closed
        # Solver panel (nobody could find it), then a toggle under Unfold -
        # which was saved with the scene, so once it had been switched on
        # every Unfold ran the slow path. Unfold is always the quick one.
        lay.operator("cbu.optimise", icon="MOD_ARRAY")
        # Optimise's own setting, so it sits with Optimise: Unfold ignores it.
        lay.prop(s, "board_bias", slider=True)
        # Settings that only shape the parts apply to the net in hand; the
        # ones the solve read need a new one, and say so here rather than in
        # a report that has scrolled away.
        col = lay.column(align=True)
        col.operator("cbu.refresh", icon="FILE_REFRESH")
        col.prop(s, "pack_effort", slider=True)
        from . import ops as _ops
        stale = _ops.unfold_settings_changed(obj, s)
        if stale:
            warn = lay.row()
            warn.alert = True
            warn.label(text="Needs Unfold: " + _ops.setting_labels(s, stale),
                       icon="ERROR")
        job = _running()
        if job is not None:
            draw_progress(lay, job)
        # A bent face has no flat image, so it has to be triangulated in the
        # mesh - the core cannot paper over it. Offered next to Unfold because
        # that is where the warning about it appears.
        lay.operator("cbu.split_nonplanar", icon="MOD_TRIANGULATE")

        if s.last_report and _running() is None:
            box = lay.box()
            if s.last_optimised:
                # Its own header, because an optimised net and a plain one
                # otherwise look identical - there was no way to tell a run
                # had finished, or that it had been optimised at all.
                box.label(text=f"Optimised in {s.last_elapsed:.0f} s",
                          icon="CHECKMARK")
                if s.last_detail:
                    box.label(text=s.last_detail)
                for line in report_lines(s.last_report):
                    box.label(text=line)
            else:
                lines = report_lines(s.last_report)
                box.label(text=lines[0] if lines else "", icon="CHECKMARK")
                for line in lines[1:]:
                    box.label(text=line)
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


class CBU_PT_constraints(_Base, bpy.types.Panel):
    # "Edge Constraints", not "Seams": these marks tell the SOLVER where it
    # must and must not cut. What happens at a cut once it is made - glue
    # tabs, interlocking fingers - is the Seams panel. Both used to share the
    # one word, and the panel called Seams had nothing about seams in it.
    bl_label = "Edge Constraints"
    bl_parent_id = "CBU_PT_main"

    def draw(self, context):
        lay = self.layout
        s = context.scene.cbu
        if context.mode != "EDIT_MESH":
            lay.label(text="Enter Edit Mode to mark edges", icon="INFO")
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
        lay.label(text="Every value here is empirical.", icon="INFO")
        lay.label(text="Cut a calibration set before trusting it.")


class CBU_PT_seams(_Base, bpy.types.Panel):
    """How two halves of a cut go back together: glue tabs, fingers."""
    bl_label = "Seams"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.use_property_split = True
        lay.prop(s, "use_fingers")
        sub = lay.column()
        sub.enabled = s.use_fingers
        sub.prop(s, "finger_width")
        sub.prop(s, "finger_clearance")
        sub.prop(s, "finger_depth_override")
        row = sub.row()
        row.enabled = s.use_fingers and s.finger_depth_override
        if s.finger_depth_override:
            row.prop(s, "finger_depth")
        else:
            # Show the number actually used, not a field that does nothing.
            row.label(text=f"Depth {s.thickness:.1f} mm (board thickness)")
        lay.prop(s, "corner_hole")
        lay.separator()
        lay.prop(s, "use_tabs")
        sub = lay.column()
        sub.enabled = s.use_tabs
        sub.prop(s, "tab_depth")
        if s.use_fingers and s.use_tabs:
            lay.label(text="Tabs go only where fingers could not", icon="INFO")


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
        lay.prop(s, "min_bend")
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
        from . import ops
        objs, _err = ops.job_objects(context)
        lay.prop(s, "model_id")
        if len(objs) > 1:
            # One id per object: island numbers restart for each, so the
            # id is what tells their pieces apart on the shared sheets.
            ids = [ops.job_material(s, o, objs).model_id or "-" for o in objs]
            lay.label(text="using: " + ", ".join(ids), icon="INFO")
        elif not (s.model_id or "").strip():
            eff = props.model_id_for(s, objs[0] if objs else None)
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
        draw_size(lay, context)
        lay.prop(s, "apply_transform")


class CBU_PT_laser(_Base, bpy.types.Panel):
    bl_label = "Laser"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout
        lay.prop(s, "laser_max")
        lay.prop(s, "dynamic_power")
        # power / speed / passes per operation, in the order they run
        col = lay.column(align=True)
        hdr = col.row(align=True)
        for t in ("", "Power", "mm/min", "Passes"):
            hdr.label(text=t)
        for op, name in (("engrave", "Engrave"), ("score", "Score"),
                         ("cut", "Cut")):
            r = col.row(align=True)
            if op == "score":
                # off by default: the row says so and greys out
                r.prop(s, "score_folds", text=name)
                r = r.row(align=True)
                r.enabled = s.score_folds
            else:
                r.label(text=name)
            r.prop(s, f"{op}_power", text="")
            r.prop(s, f"{op}_speed", text="")
            r.prop(s, f"{op}_passes", text="")
        lay.prop(s, "air_assist")
        sub = lay.column()
        sub.enabled = s.air_assist
        r = sub.row(align=True)
        r.prop(s, "engrave_air", toggle=True)
        sc = r.row(align=True)
        sc.enabled = s.score_folds
        sc.prop(s, "score_air", toggle=True)
        r.prop(s, "cut_air", toggle=True)
        r = sub.row(align=True)
        r.prop(s, "air_on")
        r.prop(s, "air_off")
        lay.prop(s, "write_gcode")


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
        lay.prop(s, "flat_tol")
        lay.prop(s, "trials")
        lay.prop(s, "seed")
        # Only Optimise reads this; the button itself is on the main panel.
        lay.prop(s, "budget")
        lay.prop(s, "limit_to_bed")
        lay.separator()
        lay.prop(s, "out_dir")
        lay.prop(s, "write_dxf")
        lay.prop(s, "write_gcode")


class CBU_PT_debug(_Base, bpy.types.Panel):
    bl_label = "Debug"
    bl_parent_id = "CBU_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = context.scene.cbu
        lay = self.layout

        # Which code is running - the first thing to rule out when a fix
        # appears not to have worked (see build.py).
        box = lay.box()
        col = box.column(align=True)
        first, *rest = build.stamp_lines()
        col.label(text=first, icon="INFO")
        for line in rest:
            col.label(text=line, icon="BLANK1")
        if build.stale():
            warn = col.row()
            warn.alert = True
            warn.label(text="Changed on disk: Reload", icon="ERROR")

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
        # Short: the longer "(fits one 400 x 900 mm sheet; set by the bed...)"
        # truncated to "900... bed" at a normal sidebar width.
        lay.label(text="   (set by the bed, not the model)")
        # Stacked, not in a row. Four switches side by side truncate to
        # "Sho.." at any sane panel width, and the key that goes with the
        # marks belongs beside the buttons that make them, further down.
        col = lay.column(align=True)
        col.prop(s, "show_bed")
        sub = col.column(align=True)
        sub.enabled = s.show_bed
        sub.prop(s, "show_bed_ruler")
        sub.prop(s, "show_bed_grid")
        sub.prop(s, "hide_blender_grid")
        col.prop(s, "show_cut")
        col.prop(s, "show_net_labels", text="Annotation")
        sub = col.column(align=True)
        sub.enabled = s.score_folds
        sub.prop(s, "show_net_scores")
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
            # Same swatches as the Edge Constraints panel: these write the same
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
        col.prop(s, "pack_effort", slider=True)
        col.operator("cbu.repack", icon="MOD_ARRAY")
        col.operator("cbu.pack_uv", icon="UV_ISLANDSEL")
        lay.operator("cbu.export", icon="EXPORT")

        box = lay.box()
        box.label(text="Move and rotate islands freely.", icon="INFO")
        box.label(text="Do NOT scale - that breaks the")
        box.label(text="isometry and parts stop mating.")


def draw_size(lay, context) -> None:
    """Size by height or by yardstick - one factor for everything cut
    together, shown as the number it comes to so a typo is caught before an
    Unfold rather than after a cut."""
    from . import ops
    s = context.scene.cbu
    row = lay.row()
    row.prop(s, "size_mode", expand=True)
    objs, _err = ops.job_objects(context)
    if s.size_mode == "HEIGHT":
        lay.prop(s, "target_height", text="Height (world Z)")
        if s.target_height <= 0:
            lay.label(text="0 = the model's real size", icon="INFO")
        elif objs:
            lo, hi = ops.world_z_mm(objs, s.mm_per_unit, s.apply_transform)
            if hi - lo > 1e-9:
                lay.label(text=f"Scale x{s.target_height / (hi - lo):.4g}",
                          icon="INFO")
        return
    row = lay.row(align=True)
    row.prop(s, "yardstick", text="Yardstick")
    row.operator("cbu.add_yardstick", text="", icon="ADD")
    lay.prop(s, "yardstick_length")
    if s.yardstick is None:
        r = lay.row()
        r.alert = True
        r.label(text="Add or pick a yardstick", icon="ERROR")
        return
    try:
        got = bridge.yardstick_mm(s.yardstick, s.mm_per_unit,
                                  s.apply_transform)
    except ValueError as exc:
        r = lay.row()
        r.alert = True
        r.label(text=str(exc), icon="ERROR")
        return
    k = s.yardstick_length / got
    lay.label(text=f"In scene {got:.4g} mm, scale x{k:.4g}", icon="INFO")
    if objs:
        lo, hi = ops.world_z_mm(objs, s.mm_per_unit, s.apply_transform)
        lay.label(text=f"Height then {k * (hi - lo):.0f} mm", icon="INFO")


classes = (CBU_UL_job, CBU_PT_main, CBU_PT_constraints, CBU_PT_material,
           CBU_PT_seams,
           CBU_PT_folds,
           CBU_PT_labels, CBU_PT_machine, CBU_PT_laser, CBU_PT_solver,
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
