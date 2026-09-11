"""Operators. Seam marking runs in Edit Mode off the normal edge selection,
so it inherits loop select, shortest path, box/lasso and undo for free."""
from __future__ import annotations

import traceback

import bmesh
import bpy
import numpy as np
from bpy.props import EnumProperty

from . import bridge, overlay
from . import log as cbulog
from .props import material_from

_LAST = {}          # object name -> (pm, islands, parts, sheets, report)
LAST_TIMINGS = cbulog.Timings()


def _core():
    """Import the core submodules by path.

    Not `from ... import unfold` - the package re-exports an `unfold`
    FUNCTION, which shadows the `unfold` MODULE of the same name.
    """
    import importlib
    base = __package__ + ".vendor.cardboard_unfolder"
    return (importlib.import_module(base + ".unfold"),
            importlib.import_module(base + ".layout"),
            importlib.import_module(base + ".export"))


def _active_mesh(context):
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None, "select a mesh object"
    if len(obj.data.polygons) == 0:
        return None, "mesh has no faces"
    return obj, None


# --------------------------------------------------------------- marking

class CBU_OT_mark(bpy.types.Operator):
    bl_idname = "cbu.mark"
    bl_label = "Mark Edges"
    bl_description = "Set the selected edges as forced cuts, forced folds, or automatic"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(items=[
        ("CUT", "Cut", "Always cut here"),
        ("FOLD", "Fold", "Never cut here"),
        ("AUTO", "Auto", "Let the solver decide"),
    ], default="CUT")

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == "MESH"
                and context.mode == "EDIT_MESH")

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bridge.edge_layer(bm)
        val = {"CUT": bridge.FORCE_CUT, "FOLD": bridge.FORCE_FOLD,
               "AUTO": bridge.AUTO}[self.mode]
        n = 0
        for e in bm.edges:
            if e.select:
                e[lay] = val
                n += 1
        bmesh.update_edit_mesh(obj.data)
        if n == 0:
            self.report({"WARNING"}, "no edges selected")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{n} edge(s) -> {self.mode.lower()}")
        return {"FINISHED"}


class CBU_OT_split_nonplanar(bpy.types.Operator):
    bl_idname = "cbu.split_nonplanar"
    bl_label = "Split Non-Planar Faces"
    bl_description = ("Triangulate only the faces too bent to unfold flat. "
                      "Flat faces are left alone, so no panel gains a fold "
                      "line down its diagonal")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        me = obj.data
        from .vendor.cardboard_unfolder.mesh import ring_planarity

        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        bad = [p.index for p in me.polygons
               if len(p.vertices) > 3
               and ring_planarity(co, list(p.vertices)) > PLANAR_TOL]
        if not bad:
            self.report({"INFO"}, "every face is already flat enough")
            return {"CANCELLED"}

        prev = context.mode
        if prev != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for p in me.polygons:
            p.select = False
        for i in bad:
            me.polygons[i].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY",
                                           ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")
        if prev == "EDIT_MESH":
            bpy.ops.object.mode_set(mode="EDIT")
        self.report({"INFO"}, f"split {len(bad)} non-planar face(s)")
        return {"FINISHED"}


class CBU_OT_clear_marks(bpy.types.Operator):
    bl_idname = "cbu.clear_marks"
    bl_label = "Clear All Marks"
    bl_description = "Reset every edge to automatic"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        if context.mode == "EDIT_MESH":
            bm = bmesh.from_edit_mesh(obj.data)
            lay = bridge.edge_layer(bm)
            for e in bm.edges:
                e[lay] = bridge.AUTO
            bmesh.update_edit_mesh(obj.data)
        else:
            bridge.ensure_mesh_attrs(obj.data)
            a = obj.data.attributes[bridge.EDGE_ATTR]
            a.data.foreach_set("value", np.zeros(len(obj.data.edges), dtype=np.int32))
            obj.data.update()
        return {"FINISHED"}


class CBU_OT_select_marked(bpy.types.Operator):
    bl_idname = "cbu.select_marked"
    bl_label = "Select Marked"
    bl_description = "Select every edge that carries a forced cut or fold"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bridge.edge_layer(bm)
        for e in bm.edges:
            e.select_set(e[lay] != bridge.AUTO)
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        return {"FINISHED"}


# --------------------------------------------------------------- unfold

class CBU_OT_unfold(bpy.types.Operator):
    bl_idname = "cbu.unfold"
    bl_label = "Unfold"
    bl_description = ("Flatten the mesh into non-overlapping islands, write "
                      "the net to a UV layer, and colour the islands in the viewport")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == "MESH")

    def execute(self, context):
        obj, err = _active_mesh(context)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        was_edit = context.mode == "EDIT_MESH"
        if was_edit:
            bpy.ops.object.mode_set(mode="OBJECT")

        s = context.scene.cbu
        mat = material_from(s)
        U, L, _ = _core()
        log = cbulog.get()
        T = cbulog.Timings()
        LAST_TIMINGS.clear()
        try:
            with cbulog.stage(T, "read mesh"):
                pm = bridge.polymesh_from_object(
                    obj, merge_coplanar=s.merge_coplanar,
                    mm_per_unit=s.mm_per_unit,
                    apply_transform=s.apply_transform, tol_deg=s.merge_tol)
                if s.target_height > 0:
                    pm = pm.scale_to_height(s.target_height)
            log.info("mesh: %d face(s), %d vert(s), extents %s mm",
                     len(pm.faces), len(pm.vertices), pm.extents.round(1))

            with cbulog.stage(T, "check"):
                problems = pm.check()
            for p in problems:
                log.warning("%s", p)
                self.report({"WARNING"}, p)

            cuts, folds = bridge.read_constraints(obj.data)
            if cuts or folds:
                log.info("constraints: %d forced cut(s), %d forced fold(s)",
                         len(cuts), len(folds))

            limit = None
            if s.limit_to_bed:
                pad = 2 * mat.tab_depth + 2 * mat.kerf
                limit = (mat.usable_bed[0] - pad, mat.usable_bed[1] - pad)
                log.debug("island size limit %.1f x %.1f mm", *limit)

            with cbulog.stage(T, "unfold"):
                islands, rep = U.unfold(pm, trials=s.trials, seed=s.seed,
                                        max_size=limit, forced_cuts=cuts,
                                        forced_folds=folds)
            for isl in islands:
                log.debug("island %d: %d face(s), %.0f x %.0f mm, %.0f mm2",
                          isl.id, len(isl.faces), *isl.size(), isl.area())

            with cbulog.stage(T, "parts"):
                parts = L.build_parts(islands, mat)
            with cbulog.stage(T, "pack"):
                sheets, oversized = L.pack(parts, mat)
            with cbulog.stage(T, "write mesh"):
                bridge.write_islands(obj.data, pm, islands)
                # rows one usable-sheet wide, so the preview lines up with the
                # bed reference drawn underneath it
                # lay the net out exactly where packing put it, so the bed
                # reference under it is the truth and not a decoration
                bridge.write_net_uv(obj.data, pm, islands,
                                    scale=bridge.uv_scale(*mat.bed),
                                    placement={p.island_id: p for p in parts},
                                    sheet_height=mat.bed[1])
                overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data))

            _LAST[obj.name] = (pm, islands, parts, sheets, rep)
            with cbulog.stage(T, "overlay"):
                overlay.rebuild(obj, pm, islands)
            LAST_TIMINGS.update(T)
            s.last_timings = T.report()
            log.info("timings: %s", s.last_timings)

            s.last_islands = rep["islands"]
            s.last_sheets = sheets
            s.last_report = (
                f"{rep['islands']} island(s) - {sheets} sheet(s) - "
                f"isometry {rep['isometry_error']:.0e} - "
                f"{rep['trials_used']} trial(s), {rep['extra_cuts']} repair, "
                f"{rep['size_splits']} bed split(s)")
            warns = [w for p in parts for w in p.warnings]
            warns += [f"{len(oversized)} part(s) larger than the bed"] if oversized else []
            warns += rep.get("notes", [])
            s.last_warnings = " | ".join(warns[:4])
            for w in warns[:6]:
                self.report({"WARNING"}, w)

            self.report({"INFO"}, s.last_report)
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            log.error("unfold failed", exc_info=True)
            self.report({"ERROR"},
                        f"{type(exc).__name__}: {exc}  (full traceback in the "
                        f"system console)")
            return {"CANCELLED"}
        finally:
            if was_edit:
                bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}


class CBU_OT_export(bpy.types.Operator):
    bl_idname = "cbu.export"
    bl_label = "Export Sheets"
    bl_description = "Write layered SVG (and DXF) plus a manifest, one file per sheet"

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.name in _LAST)

    def execute(self, context):
        obj = context.active_object
        s = context.scene.cbu
        _, _, E = _core()
        pm, islands, parts, sheets, rep = _LAST[obj.name]
        try:
            out = bpy.path.abspath(s.out_dir)
            files = E.write_all(out, parts, material_from(s), sheets, rep,
                                dxf=s.write_dxf, stem=obj.name.lower())
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"wrote {len(files)} file(s) to {out}")
        return {"FINISHED"}


class CBU_OT_pack_uv(bpy.types.Operator):
    bl_idname = "cbu.pack_uv"
    bl_label = "Pack Net (UV)"
    bl_description = ("Nest the net with Blender's own true-shape island packer. "
                      "Scale is locked off so the unfold stays isometric")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and \
            bridge.UV_LAYER in obj.data.uv_layers

    def execute(self, context):
        obj = context.active_object
        obj.data.uv_layers.active = obj.data.uv_layers[bridge.UV_LAYER]
        prev = context.mode
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        try:
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(
                udim_source="ORIGINAL_AABB", rotate=True,
                rotate_method="CARDINAL",       # 90 degree steps only
                scale=False,                    # never rescale: isometry
                margin_method="ADD",
                # UV units, not mm: scale it the same way the net was
                margin=context.scene.cbu.part_gap * bridge.uv_scale(
                    context.scene.cbu.bed_x, context.scene.cbu.bed_y),
                shape_method="CONCAVE")
        except (RuntimeError, TypeError) as exc:
            self.report({"WARNING"}, f"pack_islands: {exc}")
        finally:
            if prev != "EDIT_MESH":
                bpy.ops.object.mode_set(mode="OBJECT")
        self.report({"INFO"}, "packed (scale locked, 90 degree rotations)")
        return {"FINISHED"}


PLANAR_TOL = 1e-3      # same gate PolyMesh.check uses

classes = (CBU_OT_mark, CBU_OT_clear_marks, CBU_OT_select_marked,
           CBU_OT_split_nonplanar, CBU_OT_unfold, CBU_OT_export, CBU_OT_pack_uv)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    _LAST.clear()
    LAST_TIMINGS.clear()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
