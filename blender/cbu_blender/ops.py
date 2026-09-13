"""Operators. Seam marking runs in Edit Mode off the normal edge selection,
so it inherits loop select, shortest path, box/lasso and undo for free."""
from __future__ import annotations

import traceback

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty

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

def selected_edges(context, bm):
    """The edges the user means, from whichever view they are working in.

    In the UV editor with sync off the mesh selection is stale and unrelated,
    so reading edge.select there would silently act on the wrong edges - or on
    none at all.
    """
    sp = getattr(context, "space_data", None)
    if (sp is not None and getattr(sp, "type", "") == "IMAGE_EDITOR"
            and not context.scene.tool_settings.use_uv_select_sync):
        idx = bridge.uv_selected_edges(bm)
        if idx:
            bm.edges.ensure_lookup_table()
            return [bm.edges[i] for i in sorted(idx)]
    return [e for e in bm.edges if e.select]


def save_face_selection(me):
    """Snapshot which faces are selected.

    Operators here select faces to drive Blender's own tools, and with UV Sync
    Selection off the UV editor only shows the UVs of *selected* faces. Leaving
    a working subset selected therefore hides most of the net and makes it
    unselectable - the tool appears to have broken the 2D view.
    """
    sel = np.zeros(len(me.polygons), dtype=bool)
    me.polygons.foreach_get("select", sel)
    return sel


def restore_face_selection(me, sel) -> None:
    if sel is None or len(sel) != len(me.polygons):
        return
    me.polygons.foreach_set("select", sel)
    me.update()


def publish_cuts(obj, parts, mat, pm=None, islands=None):
    """Hand the real cut outlines to the 2D overlay, in ISLAND space.

    The UV layer only holds the islands; glue tabs are generated in the core
    and live solely in the exported parts, so a 2D view drawn from UVs alone is
    smaller than what the laser follows. Stored in island space with two
    reference loops per island so the overlay can follow the island when it is
    dragged in the UV editor - a stale outline invites arranging against a
    shape that has moved.
    """
    if pm is None or islands is None:
        overlay.set_net_cuts(obj.name, None)
        return
    by_id = {p.island_id: p for p in parts}
    groups = []
    for isl in islands:
        part = by_id.get(isl.id)
        if part is None:
            continue
        src_pts, loop_idx = bridge.island_source_points(obj.data, pm, isl)
        if len(loop_idx) < 2:
            continue
        # two reference loops far enough apart to pin down a rotation
        j = int(np.argmax(np.linalg.norm(src_pts - src_pts[0], axis=1)))
        if j == 0:
            continue
        # the part sits wherever packing put it; bring it back to island space
        M = np.asarray(part.xform, dtype=np.float64)
        inv = np.linalg.inv(M)
        rings = []
        g = part.cut
        if g is None or g.is_empty:
            continue
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for poly in polys:
            for ring in [poly.exterior] + list(poly.interiors):
                pts = np.asarray(ring.coords)[:-1]
                if len(pts) > 2:
                    rings.append(pts @ inv[:2, :2].T + inv[:2, 2])
        if rings:
            groups.append({"refs": [int(loop_idx[0]), int(loop_idx[j])],
                           "ref_pts": (src_pts[0], src_pts[j]),
                           "rings": rings})
    overlay.set_net_cuts(obj.name, groups)
    # Resolve them straight away. The cache is what the draw handler reads, and
    # it is only otherwise filled on a depsgraph update - so without this the
    # outline does not appear until something else happens to change.
    overlay.refresh_live_cuts()


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
        for e in selected_edges(context, bm):
            e[lay] = val
            n += 1
        bmesh.update_edit_mesh(obj.data)
        if n == 0:
            self.report({"WARNING"}, "no edges selected")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{n} edge(s) -> {self.mode.lower()}")
        return {"FINISHED"}


class CBU_OT_select_all_faces(bpy.types.Operator):
    bl_idname = "cbu.select_all_faces"
    bl_label = "Select All Faces"
    bl_description = ("Select every face, so the whole net is visible and "
                      "selectable in the UV editor with UV Sync off")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        prev = context.mode
        if prev != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for p in obj.data.polygons:
            p.select = True
        obj.data.update()
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="SELECT")
        if prev != "EDIT_MESH":
            bpy.ops.object.mode_set(mode="OBJECT")
        self.report({"INFO"}, f"selected {len(obj.data.polygons)} face(s)")
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
        keep = save_face_selection(me)
        for p in me.polygons:
            p.select = False
        for i in bad:
            me.polygons[i].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY",
                                           ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")
        # the mesh has new faces, so the old mask no longer lines up; select
        # everything rather than leave the user with a hidden net
        if len(me.polygons) == len(keep):
            restore_face_selection(me, keep)
        else:
            for p in me.polygons:
                p.select = True
            me.update()
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
        mat = material_from(s, obj)
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
                publish_cuts(obj, parts, mat, pm, islands)

            _LAST[obj.name] = (pm, islands, parts, sheets, rep)
            with cbulog.stage(T, "overlay"):
                overlay.rebuild(obj, pm, islands, s.overlay_alpha)
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


class CBU_OT_edit_seam(bpy.types.Operator):
    bl_idname = "cbu.edit_seam"
    bl_label = "Edit Seam"
    bl_description = ("Force the selected edge(s) and re-unfold, so the change "
                      "shows in the net immediately")
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(items=[
        ("CUT", "Cut", "Cut along this edge"),
        ("FOLD", "Weld", "Never cut here - joins the two islands into one"),
    ], default="CUT")
    separate: BoolProperty(
        name="Separate into a new island", default=True,
        description=("Detach the piece on the far side of the cut as its own "
                     "island, leaving the rest of the net alone. Off re-runs "
                     "the solver, which is free to reattach that piece "
                     "elsewhere - fewer islands, but the whole layout moves"))

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == "MESH"
                and context.mode == "EDIT_MESH")

    def execute(self, context):
        if self.mode == "CUT" and self.separate:
            return self._separate(context)
        r = bpy.ops.cbu.mark(mode=self.mode)
        if "FINISHED" not in r:
            self.report({"WARNING"}, "nothing selected in this view")
            return {"CANCELLED"}
        return bpy.ops.cbu.unfold()

    def _separate(self, context):
        """Cut the fold locally instead of re-unfolding.

        Marking the edge and re-unfolding lets the solver rebuild the whole
        tree, so the faces freed by the cut usually reattach to a different
        island and the region count does not go up at all. Cutting the fold in
        place detaches exactly the piece the user pointed at.
        """
        obj = context.active_object
        if obj.name not in _LAST:
            self.report({"WARNING"}, "unfold first")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        picks = [(e.index, [f.index for f in e.link_faces])
                 for e in selected_edges(context, bm) if len(e.link_faces) == 2]
        if not picks:
            self.report({"WARNING"}, "select an edge between two faces")
            return {"CANCELLED"}

        # record it so a later re-unfold keeps the cut
        bpy.ops.cbu.mark(mode="CUT")

        U, L, _ = _core()
        pm, islands, _parts, _sheets, rep = _LAST[obj.name]
        core_of = {}
        for ci, grp in enumerate(pm.source_faces):
            for poly in grp:
                core_of[poly] = ci

        done = 0
        for _eidx, (p0, p1) in picks:
            a, b = core_of.get(p0), core_of.get(p1)
            if a is None or b is None or a == b:
                continue            # inside one merged n-gon: nothing to cut
            host = next((i for i in islands if a in i.faces and b in i.faces), None)
            if host is None:
                continue            # already in different islands
            nid = max(i.id for i in islands) + 1
            seam = max([c.seam for i in islands for c in i.cuts] + [-1]) + 1
            out = U.split_island(host, a, b, nid, seam)
            if out is None:
                continue
            islands[islands.index(host)] = out[0]
            islands.append(out[1])
            done += 1

        if not done:
            self.report({"WARNING"}, "that edge is already cut in this net")
            return {"CANCELLED"}

        s = context.scene.cbu
        mat = material_from(s, obj)
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            parts = L.build_parts(islands, mat)
            sheets, _over = L.pack(parts, mat)
            bridge.write_islands(obj.data, pm, islands)
            bridge.write_net_uv(obj.data, pm, islands,
                                scale=bridge.uv_scale(*mat.bed),
                                placement={p.island_id: p for p in parts},
                                sheet_height=mat.bed[1])
            _LAST[obj.name] = (pm, islands, parts, sheets, rep)
            overlay.rebuild(obj, pm, islands, s.overlay_alpha)
            overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data))
            publish_cuts(obj, parts, mat, pm, islands)
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        finally:
            bpy.ops.object.mode_set(mode="EDIT")

        s.last_islands = len(islands)
        s.last_sheets = sheets
        self.report({"INFO"},
                    f"detached {done} piece(s) - {len(islands)} island(s), "
                    f"{sheets} sheet(s)")
        return {"FINISHED"}


class CBU_OT_repack(bpy.types.Operator):
    bl_idname = "cbu.repack"
    bl_label = "Re-pack Sheets"
    bl_description = ("Run the nesting again on the current islands, without "
                      "re-unfolding. Useful after editing seams")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.name in _LAST

    def execute(self, context):
        obj = context.active_object
        s = context.scene.cbu
        mat = material_from(s, obj)
        _, L, _ = _core()
        pm, islands, _old, _sheets, rep = _LAST[obj.name]

        was_edit = context.mode == "EDIT_MESH"
        if was_edit:
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            # Rebuild from the islands rather than re-packing the packed
            # parts: Part.xform has to map island space to the sheet, and
            # packing an already-packed part would compose two placements and
            # put the net somewhere it is not.
            parts = L.build_parts(islands, mat)
            sheets, oversized = L.pack(parts, mat)
            bridge.write_net_uv(obj.data, pm, islands,
                                scale=bridge.uv_scale(*mat.bed),
                                placement={p.island_id: p for p in parts},
                                sheet_height=mat.bed[1])
            _LAST[obj.name] = (pm, islands, parts, sheets, rep)
            overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data))
            publish_cuts(obj, parts, mat, pm, islands)
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        finally:
            if was_edit:
                bpy.ops.object.mode_set(mode="EDIT")

        s.last_sheets = sheets
        if oversized:
            self.report({"WARNING"},
                        f"{sheets} sheet(s), {len(oversized)} part(s) too big")
        else:
            self.report({"INFO"}, f"packed onto {sheets} sheet(s)")
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
            files = E.write_all(out, parts, material_from(s, obj), sheets, rep,
                                dxf=s.write_dxf, stem=obj.name.lower())
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"wrote {len(files)} file(s) to {out}")
        return {"FINISHED"}


class CBU_OT_pack_uv(bpy.types.Operator):
    bl_idname = "cbu.pack_uv"
    bl_label = "Nest Sheets"
    bl_description = ("Bin-pack the real cut outlines with Blender's true-shape "
                      "packer, filling one bed at a time. Scale is locked off so "
                      "the unfold stays isometric")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == "MESH" and obj.name in _LAST
                and bridge.UV_LAYER in obj.data.uv_layers)

    def execute(self, context):
        obj = context.active_object
        s = context.scene.cbu
        mat = material_from(s, obj)
        _, L, _ = _core()
        pm, islands, old_parts, old_sheets, rep = _LAST[obj.name]
        sc = bridge.uv_scale(*mat.bed)

        prev = context.mode
        if prev != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        keep = save_face_selection(obj.data)
        parts = L.build_parts(islands, mat)
        try:
            sheets, err = self._nest(context, parts, mat, sc)
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        finally:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            restore_face_selection(obj.data, keep)

        if err or sheets > old_sheets:
            # keep whatever we had: the core's packing is always valid, and a
            # nesting that needs more sheets is not worth taking
            if prev == "EDIT_MESH":
                bpy.ops.object.mode_set(mode="EDIT")
            self.report({"WARNING"}, err or
                        f"nesting needed {sheets} sheets, keeping {old_sheets}")
            return {"CANCELLED"}

        bridge.write_net_uv(obj.data, pm, islands, scale=sc,
                            placement={p.island_id: p for p in parts},
                            sheet_height=mat.bed[1])
        _LAST[obj.name] = (pm, islands, parts, sheets, rep)
        overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data))
        publish_cuts(obj, parts, mat, pm, islands)
        s.last_sheets = sheets
        if prev == "EDIT_MESH":
            bpy.ops.object.mode_set(mode="EDIT")
        self.report({"INFO"}, f"nested onto {sheets} sheet(s) "
                              f"(was {old_sheets})")
        return {"FINISHED"}

    # ------------------------------------------------------------------
    def _nest(self, context, parts, mat, sc):
        """Fill one bed at a time with Blender's packer. Returns (sheets, err).

        Nests the parts' own cut outlines, carried on a throwaway mesh, rather
        than the island UVs. That is the whole point: a glue tab reaches past
        the island outline, so nesting islands means guessing an allowance for
        geometry the packer cannot see - too generous and nothing is eligible,
        too mean and the tabs overhang the sheet. Here what is nested is what
        is cut, so the fit test is a direct answer.
        """
        from shapely.geometry import Polygon
        _, L, _ = _core()

        rings = [self._ring(p) for p in parts]
        if any(len(r) < 3 for r in rings):
            return 0, "a part has no usable outline"
        proxy, loops = self._build_proxy(rings, sc)
        try:
            W, H = mat.usable_bed
            order = sorted(range(len(parts)), key=lambda i: -Polygon(rings[i]).area)
            remaining, sheet, placed = list(order), 0, {}
            while remaining:
                # First-fit-decreasing: grow the sheet one part at a time,
                # biggest first. Starting from "everything, then drop the
                # smallest until it fits" packs all the large parts together
                # and wastes the rest of every sheet - 9 sheets against 8 for
                # the plain shelf packer.
                won, got = [], None
                for i in list(remaining):
                    trial = won + [i]
                    self._stage(proxy, loops, trial, sc, W, H)
                    self._pack(context, proxy, trial, mat, sc)
                    cand = self._read(proxy, loops, sc)
                    if self._valid(cand, trial, W, H, mat):
                        won, got = trial, cand
                if not won:                     # one part alone will not fit
                    won = [remaining[0]]
                    self._stage(proxy, loops, won, sc, W, H)
                    got = self._read(proxy, loops, sc)
                elif got is None:
                    got = self._read(proxy, loops, sc)
                else:
                    # the winning arrangement was the last accepted trial; put
                    # it back, since later rejected trials moved things
                    self._stage(proxy, loops, won, sc, W, H)
                    self._pack(context, proxy, won, mat, sc)
                    got = self._read(proxy, loops, sc)
                for i in won:
                    placed[i] = (sheet, got[i])
                remaining = [i for i in remaining if i not in set(won)]
                sheet += 1

            err = self._apply(parts, rings, placed, mat)
            if err:
                return sheet, err
            # Every trial above re-packs the whole candidate set from scratch,
            # so a part is dropped to the next sheet whenever Blender's global
            # re-solve fails - which is not the same as there being no room.
            # Measured on pikachu at 150 mm: 35 parts on a sheet 44% full and
            # one 44.6 x 92.2 mm part alone on a second. Slot the strays into
            # the space already there, without moving anything down.
            sheet = L.tighten(parts, sheet, mat)
            return sheet, self._within_bed(parts, mat)
        finally:
            bpy.data.objects.remove(proxy, do_unlink=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _ring(part):
        g = part.cut
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        return np.asarray(g.exterior.coords)[:-1]

    @staticmethod
    def _build_proxy(rings, sc):
        """One flat face per part, spread out so each is its own UV island."""
        me = bpy.data.meshes.new("cbu_nest_proxy")
        bm = bmesh.new()
        cursor = 0.0
        for ring in rings:
            lo = ring.min(axis=0)
            vs = [bm.verts.new((float(x - lo[0]) + cursor, float(y - lo[1]), 0.0))
                  for x, y in ring]
            bm.faces.new(vs)
            cursor += float((ring.max(axis=0) - lo)[0]) + 50.0
        bm.to_mesh(me)
        bm.free()
        proxy = bpy.data.objects.new("cbu_nest_proxy", me)
        bpy.context.collection.objects.link(proxy)

        uv = me.uv_layers.new(name="pack")
        co = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)[:, :2]
        data = np.zeros((len(me.loops), 2))
        for li, loop in enumerate(me.loops):
            data[li] = co[loop.vertex_index]
        uv.data.foreach_set("uv", (data * sc).ravel())
        me.update()
        loops = [np.asarray(list(p.loop_indices)) for p in me.polygons]
        return proxy, loops

    @staticmethod
    def _uv(proxy):
        """Always re-fetch by name. Entering or leaving Edit Mode invalidates
        RNA pointers into mesh data, and holding one across the switch crashes
        Blender inside attribute access instead of raising."""
        return proxy.data.uv_layers["pack"]

    @classmethod
    def _read(cls, proxy, loops, sc):
        uv = cls._uv(proxy)
        buf = np.zeros(len(proxy.data.loops) * 2)
        uv.data.foreach_get("uv", buf)
        flat = buf.reshape(-1, 2) / sc
        return {i: flat[idx] for i, idx in enumerate(loops)}

    @classmethod
    def _stage(cls, proxy, loops, ids, sc, W, H):
        """Park the candidates at the origin and push one to the far corner,
        which is the only way to tell pack_islands what container to use."""
        uv = cls._uv(proxy)
        buf = np.zeros(len(proxy.data.loops) * 2)
        uv.data.foreach_get("uv", buf)
        flat = buf.reshape(-1, 2) / sc
        for i in ids:
            flat[loops[i]] -= flat[loops[i]].min(axis=0)
        last = ids[-1]
        flat[loops[last]] += np.array([W, H]) - flat[loops[last]].max(axis=0)
        # park everything else far away so it cannot join the bounding box
        for i in range(len(loops)):
            if i not in set(ids):
                flat[loops[i]] -= flat[loops[i]].min(axis=0)
                flat[loops[i]] += np.array([0.0, -10000.0])
        uv.data.foreach_set("uv", (flat * sc).ravel())
        proxy.data.update()

    @staticmethod
    def _pack(context, proxy, ids, mat, sc):
        prev_active = context.view_layer.objects.active
        for ob in context.selected_objects:
            ob.select_set(False)
        proxy.select_set(True)
        context.view_layer.objects.active = proxy
        ts = context.scene.tool_settings
        sync_was = ts.use_uv_select_sync
        ts.use_uv_select_sync = False
        bpy.ops.object.mode_set(mode="EDIT")
        ts.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")
        # Only the candidates. ORIGINAL_AABB is the bounding box of the
        # *selection*, so leaving the parked-away parts selected makes the
        # container enormous and every trial fails - which looks exactly like
        # "nothing fits together" and cost 19 sheets instead of 8.
        bm = bmesh.from_edit_mesh(proxy.data)
        bm.faces.ensure_lookup_table()
        want = set(ids)
        for f in bm.faces:
            f.select_set(f.index in want)
        bmesh.update_edit_mesh(proxy.data)
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            udim_source="ORIGINAL_AABB", rotate=True, rotate_method="ANY",
            scale=False, margin_method="ADD", margin=mat.part_gap * sc,
            shape_method="CONCAVE")
        bpy.ops.object.mode_set(mode="OBJECT")
        ts.use_uv_select_sync = sync_was
        proxy.select_set(False)
        context.view_layer.objects.active = prev_active

    @staticmethod
    def _valid(got, ids, W, H, mat):
        """Inside one bed, and genuinely not touching.

        pack_islands does not fail when it cannot fit - it crams the islands in
        overlapping and the result still measures inside the box - so the
        overlap test is the one that matters.
        """
        from shapely.geometry import Polygon
        pts = np.vstack([got[i] for i in ids])
        span = pts.max(axis=0) - pts.min(axis=0)
        if span[0] > W + 1e-6 or span[1] > H + 1e-6:
            return False
        polys = [Polygon(got[i]) for i in ids]
        for a in range(len(polys)):
            pa = polys[a]
            if not pa.is_valid:
                pa = pa.buffer(0)
            for b in range(a + 1, len(polys)):
                pb = polys[b]
                if not pb.is_valid:
                    pb = pb.buffer(0)
                if pa.intersection(pb).area > 1e-6:
                    return False
        return True

    @staticmethod
    def _apply(parts, rings, placed, mat):
        """Push each nested position onto its Part as a rigid motion."""
        from shapely import affinity as _aff
        by_sheet = {}
        for i, (sheet, pts) in placed.items():
            M, res = bridge._kabsch2d(rings[i], pts)
            if res > 0.5:
                return (f"part {parts[i].island_id} did not come back rigid "
                        f"({res:.2f} mm)")
            part = parts[i]
            part.apply(lambda g, M=M: _aff.affine_transform(
                g, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]]))
            part.xform = M.copy()
            part.sheet = sheet
            by_sheet.setdefault(sheet, []).append(part)
        return None

    @staticmethod
    def _within_bed(parts, mat):
        """Every part inside its sheet's usable area, or the complaint.

        Runs after tightening and seating, never before: seating is what puts
        a sheet on its margin, so checking first reports a good layout as
        outside the bed.
        """
        m0 = mat.sheet_margin
        for part in parts:
            x0, y0, x1, y1 = part.bounds
            if (x0 < m0 - 1e-3 or y0 < m0 - 1e-3
                    or x1 > mat.bed[0] - m0 + 1e-3
                    or y1 > mat.bed[1] - m0 + 1e-3):
                return f"part {part.island_id} ended up outside sheet {part.sheet}"
        return None


PLANAR_TOL = 1e-3      # same gate PolyMesh.check uses

classes = (CBU_OT_mark, CBU_OT_clear_marks, CBU_OT_select_marked,
           CBU_OT_select_all_faces,
           CBU_OT_split_nonplanar, CBU_OT_unfold, CBU_OT_edit_seam,
           CBU_OT_repack, CBU_OT_export, CBU_OT_pack_uv)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    _LAST.clear()
    LAST_TIMINGS.clear()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
