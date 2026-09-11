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
        mat = material_from(s)
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
        mat = material_from(s)
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
    bl_description = ("Re-nest each sheet with Blender's true-shape packer, "
                      "keeping every part inside its own bed. Scale is locked "
                      "off so the unfold stays isometric")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == "MESH" and obj.name in _LAST
                and bridge.UV_LAYER in obj.data.uv_layers)

    def execute(self, context):
        obj = context.active_object
        s = context.scene.cbu
        mat = material_from(s)
        _, L, _ = _core()
        pm, islands, parts, sheets, rep = _LAST[obj.name]
        sc = bridge.uv_scale(*mat.bed)

        prev = context.mode
        if prev != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            sheet_of = {p.island_id: max(p.sheet, 0) for p in parts}
            tightened = self._tighten(context, obj, pm, islands, mat, sc, sheet_of)
            new_parts, err = self._readback(obj, pm, islands, mat, sc, sheet_of, L)
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        finally:
            if context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

        if err:
            # never ship a layout we could not verify - the old one is valid
            bridge.write_net_uv(obj.data, pm, islands, scale=sc,
                                placement={p.island_id: p for p in parts},
                                sheet_height=mat.bed[1])
            self.report({"ERROR"}, err)
            if prev == "EDIT_MESH":
                bpy.ops.object.mode_set(mode="EDIT")
            return {"CANCELLED"}

        sheets = max(sheet_of.values()) + 1 if sheet_of else 0
        # Re-emit the net from the parts. The 2D view then shows exactly what
        # will be cut, by construction rather than by agreement.
        bridge.write_net_uv(obj.data, pm, islands, scale=sc,
                            placement={p.island_id: p for p in new_parts},
                            sheet_height=mat.bed[1])
        _LAST[obj.name] = (pm, islands, new_parts, sheets, rep)
        overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data))
        s.last_sheets = sheets
        if prev == "EDIT_MESH":
            bpy.ops.object.mode_set(mode="EDIT")
        self.report({"INFO"}, f"re-nested {tightened} of {sheets} sheet(s)")
        return {"FINISHED"}

    # ------------------------------------------------------------------
    def _tighten(self, context, obj, pm, islands, mat, sc, sheet_of):
        """Re-nest each sheet in place. Returns how many sheets improved.

        Starts from the core's assignment on purpose. Handed a container it
        cannot fit, with scale locked off, pack_islands does not fail - it
        crams the islands in overlapping, and the result still measures inside
        the box. A bounding-box test alone therefore accepted 19 islands on one
        sheet with 20 overlapping pairs. The core's assignment is already
        valid, so re-nesting *within* a sheet can only improve it, and anything
        failing validation is rolled back sheet by sheet.
        """
        # Pack the ISLANDS, but what gets cut is PARTS - islands plus glue
        # tabs and kerf, which stick out past the island outline. Nesting to
        # the full usable bed therefore produces a layout whose parts overhang
        # the margin, which the read-back then rightly refuses. Hold back the
        # same allowance the unfolder uses for its bed limit.
        pad = 2 * mat.tab_depth + 2 * mat.kerf
        W = mat.usable_bed[0] - pad
        H = mat.usable_bed[1] - pad
        m0 = mat.sheet_margin                 # where finished sheets start
        stage_m = m0 + pad / 2.0              # inset used only while nesting
        loops = bridge.island_loops(obj.data, islands, pm)
        by_id = {i.id: i for i in islands}

        # UV Sync Selection defaults to ON in Blender 5.x, and with it on
        # uv.select_all selects the whole mesh - so packing one sheet repacked
        # the entire net and undid every sheet placed before it. Off, the face
        # selection gates which UVs exist to be packed, which is the whole
        # mechanism this relies on.
        ts = context.scene.tool_settings
        sync_was = ts.use_uv_select_sync
        ts.use_uv_select_sync = False

        groups = {}
        for iid, sh in sheet_of.items():
            groups.setdefault(sh, []).append(iid)

        improved = 0
        for sheet, ids in sorted(groups.items()):
            ids = [i for i in ids if i in loops and len(loops[i])]
            if not ids:
                continue
            pts = bridge.read_net(obj.data, sc)
            saved = {i: pts[loops[i]].copy() for i in ids}

            if len(ids) > 1:
                before = self._span(pts, loops, ids)
                self._stage(obj, loops, ids, sc, W, H, stage_m)
                self._pack_selected(context, obj, ids, mat, sc)
                pts = bridge.read_net(obj.data, sc)
                span = self._span(pts, loops, ids)
                fits = span[0] <= W + 1e-6 and span[1] <= H + 1e-6
                ok = fits and self._disjoint(obj, pm, [by_id[i] for i in ids], sc)
                if not ok:
                    for i in ids:
                        pts[loops[i]] = saved[i]
                    bridge.write_net(obj.data, pts, sc)
                elif span[0] * span[1] < before[0] * before[1] - 1e-6:
                    improved += 1

            self._park(obj, loops, ids, sheet, mat, sc, m0)
        ts.use_uv_select_sync = sync_was
        return improved

    @staticmethod
    def _span(pts, loops, ids):
        sub = pts[np.concatenate([loops[i] for i in ids])]
        lo, hi = sub.min(axis=0), sub.max(axis=0)
        return (hi[0] - lo[0], hi[1] - lo[1])

    @staticmethod
    def _park(obj, loops, ids, sheet, mat, sc, m0):
        """Place these islands at the bottom-left of their own sheet band.

        Absolute, computed from where they are right now. Nudging them by the
        sheet offset on top of whatever a previous layout left behind applies
        it twice - which is exactly how parts ended up at y = -430 on sheet 5.
        """
        pts = bridge.read_net(obj.data, sc)
        idx = np.concatenate([loops[i] for i in ids])
        lo = pts[idx].min(axis=0)
        pts[idx] += np.array([m0 - lo[0], m0 - lo[1] + sheet * mat.bed[1]])
        bridge.write_net(obj.data, pts, sc)

    @staticmethod
    def _stage(obj, loops, ids, sc, W, H, m0):
        """Make the selection's bounding box exactly one bed.

        pack_islands has no target-rectangle argument: with ORIGINAL_AABB it
        rearranges inside the bounding box of whatever is selected. So the box
        is set by parking every candidate at the origin and pushing one out to
        the far corner.
        """
        pts = bridge.read_net(obj.data, sc)
        for i in ids:
            lo = pts[loops[i]].min(axis=0)
            pts[loops[i]] += np.array([m0 - lo[0], m0 - lo[1]])
        last = ids[-1]
        hi = pts[loops[last]].max(axis=0)
        pts[loops[last]] += np.array([(m0 + W) - hi[0], (m0 + H) - hi[1]])
        bridge.write_net(obj.data, pts, sc)

    @staticmethod
    def _pack_selected(context, obj, ids, mat, sc):
        attr = obj.data.attributes.get(bridge.FACE_ATTR)
        vals = np.zeros(len(obj.data.polygons), dtype=np.int32)
        if attr is not None:
            attr.data.foreach_get("value", vals)
        want = set(ids)

        # Select inside Edit Mode, after the select-mode switch. Setting
        # poly.select in Object Mode and *then* switching to face mode makes
        # Blender re-derive the face selection from the vertices, which can
        # quietly select the whole mesh - and then pack_islands repacks every
        # island instead of this sheet's.
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f.select_set(int(vals[f.index]) in want)
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            udim_source="ORIGINAL_AABB", rotate=True,
            rotate_method="CARDINAL", scale=False, margin_method="ADD",
            # Islands are what is packed, parts are what is cut: a glue tab
            # protrudes up to tab_depth beyond the island outline, so leaving
            # only part_gap between islands lets two tabs collide.
            margin=(mat.part_gap + 2.0 * mat.tab_depth) * sc,
            shape_method="CONCAVE")
        bpy.ops.object.mode_set(mode="OBJECT")

    @staticmethod
    def _disjoint(obj, pm, isls, sc) -> bool:
        """Do these islands actually keep out of each other's way?

        The check a bounding box cannot make, and the one whose absence let a
        sheet of 19 overlapping islands through. Uses the real outlines at
        their recovered positions.
        """
        from shapely import affinity as _aff
        xf, res = bridge.net_transforms(obj.data, pm, isls, sc)
        if res > 0.5:
            return False
        polys = []
        for isl in isls:
            M = xf.get(isl.id)
            if M is None:
                return False
            polys.append(_aff.affine_transform(
                isl.outline(),
                [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]]))
        for a in range(len(polys)):
            for b in range(a + 1, len(polys)):
                if polys[a].intersection(polys[b]).area > 1e-6:
                    return False
        return True

    @staticmethod
    def _readback(obj, pm, islands, mat, sc, sheet_of, L):
        """Push the 2D layout onto freshly built parts. Returns (parts, error)."""
        from shapely import affinity as _aff
        xf, res = bridge.net_transforms(obj.data, pm, islands, sc)
        if res > 0.5:
            return None, (f"layout is not a rigid motion of the net "
                          f"({res:.2f} mm) - was an island scaled?")
        parts = L.build_parts(islands, mat)
        for part in parts:
            M = xf.get(part.island_id)
            if M is None:
                continue
            part.apply(lambda g, M=M: _aff.affine_transform(
                g, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]]))
            part.xform = M.copy()
            part.sheet = sheet_of.get(part.island_id, 0)
            part.translate(0.0, -part.sheet * mat.bed[1])   # net -> sheet space

        # Final placement is decided by the PARTS, not the islands. Parking by
        # island outline puts the island edge on the margin and leaves the glue
        # tab hanging outside it - the islands are what gets nested, but the
        # parts are what gets cut.
        m0 = mat.sheet_margin
        per_sheet = {}
        for part in parts:
            per_sheet.setdefault(part.sheet, []).append(part)
        for group in per_sheet.values():
            x0 = min(p.bounds[0] for p in group)
            y0 = min(p.bounds[1] for p in group)
            for part in group:
                part.translate(m0 - x0, m0 - y0)

        for part in parts:
            x0, y0, x1, y1 = part.bounds
            if (x0 < m0 - 1e-3 or y0 < m0 - 1e-3
                    or x1 > mat.bed[0] - m0 + 1e-3
                    or y1 > mat.bed[1] - m0 + 1e-3):
                return None, (f"island {part.island_id} ended up outside "
                              f"sheet {part.sheet}")

        # Final word on validity, checked on the geometry that is actually
        # cut. The per-sheet check during nesting compares island outlines,
        # which cannot see two glue tabs reaching into the same gap.
        for sheet, group in per_sheet.items():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    if group[a].cut.intersection(group[b].cut).area > 1e-6:
                        return None, (
                            f"islands {group[a].island_id} and "
                            f"{group[b].island_id} overlap on sheet {sheet}")
        return parts, None


PLANAR_TOL = 1e-3      # same gate PolyMesh.check uses

classes = (CBU_OT_mark, CBU_OT_clear_marks, CBU_OT_select_marked,
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
