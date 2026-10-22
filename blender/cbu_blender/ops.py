"""Operators. Seam marking runs in Edit Mode off the normal edge selection,
so it inherits loop select, shortest path, box/lasso and undo for free."""
from __future__ import annotations

import threading
import time
import traceback

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty
from mathutils import Vector

from . import bridge, overlay
from . import log as cbulog
from .props import material_from

_LAST = {}          # object name -> (pm, islands, parts, sheets, report)
# object name -> every object name solved and packed with it, itself included.
# Its parts in _LAST are its own; the sheets are shared with the rest.
_GROUP = {}
LAST_TIMINGS = cbulog.Timings()

# The settings a solve reads: the mesh as built (merge, scale, transform),
# the bed-split limit (bed, margin, and the tab and kerf allowance), and the
# search itself. Everything else - fingers, holes, tabs, relief, labels - only
# shapes the parts, so Refresh can apply it to the net already in hand.
UNFOLD_KEYS = ("merge_coplanar", "merge_tol", "mm_per_unit", "apply_transform",
               "size_mode", "target_height", "yardstick_length", "flat_tol",
               "limit_to_bed", "bed_x", "bed_y", "margin",
               "tab_depth", "kerf", "trials", "seed", "budget")
# Inputs to a solve that are not settings: which objects were cut together,
# and how long the yardstick was in the scene. Named as the panel would.
EXTRA_KEYS = {"objects": "Objects", "yardstick_mm": "Yardstick"}
_SOLVED_WITH = {}   # object name -> {key: value} as the last solve read them


def unfold_settings(s, obj=None) -> dict:
    out = {k: getattr(s, k) for k in UNFOLD_KEYS}
    if obj is not None:
        out["objects"] = job_names(s, obj)
    if s.size_mode == "YARDSTICK":
        out["yardstick_mm"] = _yardstick_or_none(s)
    return out


def setting_labels(s, keys) -> str:
    """The panel's own names for some settings - "Bed X", not "bed_x"."""
    return ", ".join(EXTRA_KEYS[k] if k in EXTRA_KEYS
                     else s.bl_rna.properties[k].name for k in keys)


def _current(s, obj, k):
    if k == "objects":
        return job_names(s, obj)
    if k == "yardstick_mm":
        return _yardstick_or_none(s)
    return getattr(s, k)


def unfold_settings_changed(obj, s) -> list:
    """Names of the solve's settings that have changed since `obj` was last
    unfolded - what a Refresh cannot apply."""
    was = _SOLVED_WITH.get(obj.name) if obj is not None else None
    if was is None:
        return []
    out = []
    for k in was:
        a, b = _current(s, obj, k), was[k]
        if isinstance(a, float) and isinstance(b, float):
            moved = abs(a - b) > 1e-6 * max(1.0, abs(b))
        else:
            moved = a != b
        if moved:
            out.append(k)
    return out


# ------------------------------------------------------- what is cut, how big

def job_objects(context):
    """The objects cut together: the saved list, or the active object when
    the list is empty. Returns (objects, error). The yardstick is never cut,
    even when it is in the list."""
    s = context.scene.cbu
    stick = s.yardstick
    listed = [it.obj for it in s.job if it.obj is not None]
    if listed:
        objs = []
        for o in listed:
            if (o.type == "MESH" and o != stick and o not in objs
                    and context.scene.objects.get(o.name) is not None):
                objs.append(o)
        if not objs:
            return [], ("the list has nothing to cut - the yardstick never is, "
                        "and deleted objects are skipped")
    else:
        o = context.active_object
        if o is None or o.type != "MESH":
            return [], "select a mesh object"
        if o == stick:
            return [], ("that is the yardstick, which is never cut - select "
                        "the model")
        objs = [o]
    for o in objs:
        if len(o.data.polygons) == 0:
            return [], f"{o.name} has no faces"
    return objs, None


def job_ref(context):
    """The object Re-pack, Refresh, Export and the panel act for: the first
    cut object in the list, or the active object. Not simply the active
    object - right after Add Yardstick that is the stick."""
    s = context.scene.cbu
    for it in s.job:
        o = it.obj
        if o is not None and o.type == "MESH" and o != s.yardstick:
            return o
    obj = context.active_object
    return obj if obj is not None and obj.type == "MESH" else None


def job_names(s, obj) -> tuple:
    """Who `obj` would be cut with now - for the staleness check, which has
    no context to hand. Same rule as job_objects, by name."""
    stick = s.yardstick
    listed = [it.obj.name for it in s.job
              if it.obj is not None and it.obj.type == "MESH"
              and it.obj != stick]
    if listed:
        return tuple(dict.fromkeys(listed))
    return (obj.name,) if obj is not None else ()


def _yardstick_or_none(s):
    try:
        return bridge.yardstick_mm(s.yardstick, s.mm_per_unit,
                                   s.apply_transform)
    except Exception:                                   # noqa: BLE001
        return None


def object_mode(context, objs) -> bool:
    """Every object in `objs`, and the yardstick, in Object Mode - not just
    the active one. Returns whether the active object was in Edit Mode, to
    put it back after.

    An object in Edit Mode keeps its live data in a BMesh and its mesh's
    attribute and UV arrays read as EMPTY - `foreach_set(..) given 228,
    needed 0`. With a list, the active object need not be one of the
    objects cut: Add Yardstick from Edit Mode used to leave the model in Edit
    Mode with the stick active, and taking only the active object out of it
    then left the model's mesh unreadable."""
    was_edit = context.mode == "EDIT_MESH"
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    stuck = [o for o in list(objs) + [context.scene.cbu.yardstick]
             if o is not None and o.mode != "OBJECT"]
    if stuck:
        vl = context.view_layer
        active = vl.objects.active
        try:
            for o in stuck:
                vl.objects.active = o
                bpy.ops.object.mode_set(mode="OBJECT")
        finally:
            vl.objects.active = active
    return was_edit


def size_factor(s, pms) -> float:
    """ONE scale for everything cut together, so the pieces keep their sizes
    relative to each other: a border modelled to fit a helmet's rim still
    fits it. Height is measured over all of them; the yardstick is measured
    exactly as the model is (same transform, same mm per unit)."""
    if s.size_mode == "YARDSTICK":
        if s.yardstick is None:
            raise ValueError("Size by Yardstick, but no yardstick is set - "
                             "Add Yardstick, or pick one in Machine & Size")
        got = bridge.yardstick_mm(s.yardstick, s.mm_per_unit,
                                  s.apply_transform)
        return s.yardstick_length / got
    if s.target_height > 0:
        lo = min(float(pm.vertices[:, 2].min()) for pm in pms)
        hi = max(float(pm.vertices[:, 2].max()) for pm in pms)
        if hi - lo < 1e-9:
            raise ValueError("nothing being cut has any height along Z - "
                             "set Height to 0, or use a yardstick")
        return s.target_height / (hi - lo)
    return 1.0


def job_material(s, obj, objs):
    """The object's Material. With several objects a typed Model ID gets a
    letter each (HELMA, HELMB), because island numbers restart per object
    and two pieces both stamped HELM-0 are the problem the id solves."""
    mat = material_from(s, obj)
    tag = (s.model_id or "").strip()
    if tag and len(objs) > 1:
        mat.model_id = tag + chr(ord("A") + objs.index(obj) % 26)
    return mat


def group_of(name) -> tuple:
    """Everything `name` was last packed with, itself included."""
    return _GROUP.get(name, (name,))


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
        # The engraving rides along. It is generated in the core and lives only
        # in the exported parts, exactly like the tabs - so without this the 2D
        # view shows no seam numbers, no M/V and no model id, and the first
        # place they appear is the SVG.
        def lines(*geoms):
            out = []
            for g in geoms:
                if g is None or g.is_empty:
                    continue
                for ln in getattr(g, "geoms", [g]):
                    pts = np.asarray(ln.coords)
                    if len(pts) >= 2:
                        out.append(pts @ inv[:2, :2].T + inv[:2, 2])
            return out
        labels = lines(part.engrave)
        # The fold score lines, published whether or not they are burned:
        # the overlay draws them only while Score folds is on, so the switch
        # shows and hides them without a Refresh.
        scores = lines(part.perf_m, part.perf_v)
        if rings:
            groups.append({"refs": [int(loop_idx[0]), int(loop_idx[j])],
                           "ref_pts": (src_pts[0], src_pts[j]),
                           "rings": rings, "labels": labels,
                           "scores": scores})
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
        # The marks are the overlay's whole point while seams are being
        # edited, and marking is deliberately NOT a geometry change - so
        # nothing else would refresh them until the next unfold.
        overlay.refresh_marks()
        self.report({"INFO"}, f"{n} edge(s) -> {self.mode.lower()}")
        return {"FINISHED"}


class CBU_OT_lock(bpy.types.Operator):
    bl_idname = "cbu.lock"
    bl_label = "Lock Edges"
    bl_description = ("Keep the optimiser away from the selected edges. "
                      "Whatever they are marked as now - cut, fold or auto - "
                      "is what it has to work around")
    bl_options = {"REGISTER", "UNDO"}

    lock: BoolProperty(name="Lock", default=True)

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == "MESH"
                and context.mode == "EDIT_MESH")

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bridge.lock_layer(bm)
        n = 0
        for e in selected_edges(context, bm):
            e[lay] = self.lock
            n += 1
        bmesh.update_edit_mesh(obj.data)
        if n == 0:
            self.report({"WARNING"}, "no edges selected")
            return {"CANCELLED"}
        # A lock changes how the mark is DRAWN (dashed) and, like marking, is
        # not a geometry change - so nothing else would refresh it.
        overlay.refresh_marks()
        self.report({"INFO"},
                    f"{n} edge(s) {'locked' if self.lock else 'unlocked'}")
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


class CBU_OT_clear_locks(bpy.types.Operator):
    bl_idname = "cbu.clear_locks"
    bl_label = "Unlock All"
    bl_description = "Give every edge back to the optimiser"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        if context.mode == "EDIT_MESH":
            bm = bmesh.from_edit_mesh(obj.data)
            lay = bridge.lock_layer(bm)
            for e in bm.edges:
                e[lay] = False
            bmesh.update_edit_mesh(obj.data)
        else:
            bridge.ensure_mesh_attrs(obj.data)
            a = obj.data.attributes[bridge.LOCK_ATTR]
            a.data.foreach_set("value",
                               np.zeros(len(obj.data.edges), dtype=bool))
            obj.data.update()
        overlay.refresh_marks()
        return {"FINISHED"}


class CBU_OT_select_locked(bpy.types.Operator):
    bl_idname = "cbu.select_locked"
    bl_label = "Select Locked"
    bl_description = "Select every edge the optimiser has been told to leave alone"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "EDIT_MESH"

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        lay = bridge.lock_layer(bm)
        for e in bm.edges:
            e.select_set(bool(e[lay]))
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
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

# The job running in the background, if any. The panel reads it to draw a
# progress bar, and Unfold's poll reads it so a second click cannot start a
# second solve on top of the first.
RUNNING = None


class Member:
    """One object's share of a Solve: its mesh, its search settings, its
    Material (the model id differs per object), and what came back."""

    def __init__(self, name, pm, kwargs, mat):
        self.name, self.pm, self.kwargs, self.mat = name, pm, kwargs, mat
        self.islands = self.rep = self.parts = None


class Solve:
    """The bpy-free half of an unfold, packaged so it can run off the main
    thread while the operator keeps taking events.

    The core imports no bpy at all, and that separation - the whole reason the
    two layers are split - is what makes this legal rather than merely lucky.
    Everything that touches Blender data happens before (`CBU_OT_unfold.prepare`)
    or after (`apply`) on the main thread; only `run` moves, and it touches
    nothing but the PolyMesh copy it was handed.

    Cancelling is cooperative: `stop()` sets a flag, the core's progress hook
    reads it between candidates and returns False, and the search answers with
    the best candidate it has rather than nothing. So ESC gives you a net, not
    an empty result - which matters, because the first candidate is usually
    already better than the old rule's answer.
    """

    def __init__(self, U, pm, kwargs, L=None, mat=None, timings=None,
                 members=None):
        # One Member per object cut together; the arguments describe the
        # first, which is all a single-object solve (or a test) ever has.
        self.members = members or [Member(None, pm, kwargs, mat)]
        self.U, self.L, self.mat, self.T = U, L, mat, timings
        self.error = None
        self.sheets = self.oversized = None
        self.done = self.total = 0
        self.note = ""
        self.started = time.monotonic()
        self.finished = None
        self._stop = False
        self.ui = None          # optional main-thread callback, set by the caller

    # The single-object view, for callers that only ever had one.
    pm = property(lambda self: self.members[0].pm)
    kwargs = property(lambda self: self.members[0].kwargs)
    islands = property(lambda self: self.members[0].islands)
    rep = property(lambda self: self.members[0].rep)

    @property
    def parts(self):
        """Every member's parts - what was packed together."""
        if any(m.parts is None for m in self.members):
            return None
        return [p for m in self.members for p in m.parts]

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total) if self.total else 0.0

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def stop(self):
        self._stop = True

    @property
    def stopping(self) -> bool:
        return self._stop

    def progress(self, done, total, note):
        self.done, self.total, self.note = done, total, note
        if self.ui is not None:
            try:
                self.ui(done, total, note)
            except Exception:                           # noqa: BLE001
                pass        # a progress readout must never break a solve
        return not self._stop

    def run(self):
        """Runs on a worker thread. Must not raise out of it: an exception on
        a thread nobody joins disappears, and the operator would report
        success for a net that was never built.

        Parts and packing happen HERE, not in `apply`. They are core calls and
        bpy-free, and on a big model they are the slow half - a torus packs in
        30 s - so leaving them on the main thread meant the progress bar
        reached the end and Blender then froze for half a minute, at exactly
        the moment it looked finished.
        """
        try:
            many = len(self.members) > 1
            for m in self.members:
                # Each object unfolds on its own - they share no edge, so
                # nothing one decides can matter to another - and only the
                # packing is shared. The note says whose search is running.
                def hook(done, total, note, _m=m):
                    return self.progress(
                        done, total, f"{_m.name}: {note}" if many else note)
                with _stage(self.T, "unfold"):
                    m.islands, m.rep = self.U.unfold(m.pm, progress=hook,
                                                     **m.kwargs)
            if self.L is not None and self.mat is not None:
                # No per-part hook in the packer, so this phase has a label and
                # an elapsed time rather than a moving bar - still better than
                # a bar sitting at 100% while nothing appears to happen.
                self.done, self.total, self.note = 0, 0, "packing sheets"
                with _stage(self.T, "parts"):
                    for m in self.members:
                        m.parts = self.L.build_parts(m.islands,
                                                     m.mat or self.mat)
                with _stage(self.T, "pack"):
                    self.sheets, self.oversized = self.L.pack(self.parts,
                                                              self.mat)
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self.error = exc
        finally:
            self.finished = time.monotonic()


def _stage(T, name):
    """cbulog.stage when there is a Timings to fill, a no-op when there is
    not - a Solve built by hand in a test has none."""
    import contextlib
    return cbulog.stage(T, name) if T is not None else contextlib.nullcontext()


class _Unfold:
    """Everything Unfold and Optimise share. A plain mixin, NOT an Operator.

    The obvious shape - `class CBU_OT_optimise(CBU_OT_unfold)` - breaks the
    parent: once a registered operator class is subclassed and the subclass
    registered too, `bpy.ops.cbu.unfold.poll()` fails with "context is
    incorrect" on a context that satisfies it. Two thin classes over a mixin
    is the pattern Blender supports.
    """
    bl_options = {"REGISTER", "UNDO"}

    # Which path this is. A class attribute rather than a scene setting or an
    # operator property: a scene setting is saved with the file and made the
    # fast button slow the moment it was switched on once, and an operator
    # property is remembered between calls by Blender unless told not to be.
    optimising = False

    @classmethod
    def poll(cls, context):
        if RUNNING is not None:
            return False
        if any(it.obj is not None for it in context.scene.cbu.job):
            return True
        return (context.active_object is not None
                and context.active_object.type == "MESH")

    def execute(self, context):
        """The synchronous path, and the only one a script ever takes.

        `bpy.ops.cbu.unfold()` from Python runs EXEC_DEFAULT, which comes
        straight here - so every headless check exercises this and none of
        them depend on a modal loop or a window.
        """
        job = self.prepare(context)
        if job is None:
            return {"CANCELLED"}
        wm = context.window_manager
        started = [False]

        def show(done, total, _note):
            # wm.progress_* drives the cursor readout, the only thing that
            # updates from inside a blocking execute - a panel does not redraw
            # until the operator returns.
            if not started[0]:
                wm.progress_begin(0, max(1, total))
                started[0] = True
            wm.progress_update(done)

        job.ui = show
        try:
            job.run()
        finally:
            if started[0]:
                try:
                    wm.progress_end()
                except Exception:                       # noqa: BLE001
                    pass
        return self.apply(context, job)

    def invoke(self, context, event):
        """Modal only where it buys something: a cancellable optimiser run.

        The plain search is a second or two and a modal loop around it would
        be pure machinery. The optimiser mends every candidate before it can
        score one, so it runs for tens of seconds and the user needs both a
        readout and a way out.
        """
        if not self.optimising:
            return self.execute(context)
        job = self.prepare(context)
        if job is None:
            return {"CANCELLED"}

        global RUNNING
        RUNNING = self._job = job
        self._thread = threading.Thread(target=job.run, daemon=True)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.progress_begin(0, 100)
        self._thread.start()
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        wm = context.window_manager
        if event.type in {"ESC"} and event.value == "PRESS":
            # Cooperative, and it has to be: the thread is inside numpy and
            # shapely calls that cannot be interrupted, so it stops at the
            # next candidate boundary. The core then answers with the best
            # candidate so far, which is why this applies the result rather
            # than throwing the work away.
            if not self._job.stopping:
                self._job.stop()
                self.report({"INFO"}, "stopping after this candidate...")
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        job = self._job
        if self._thread.is_alive():
            try:
                wm.progress_update(int(100 * job.fraction))
                context.workspace.status_text_set(
                    f"Optimising: {job.note or 'starting'}  "
                    f"({job.elapsed:.0f} s)"
                    + ("  - stopping..." if job.stopping else "  - ESC to stop"))
            except Exception:                           # noqa: BLE001
                pass
            _redraw_panels(context)
            return {"RUNNING_MODAL"}

        self._cleanup(context)
        result = self.apply(context, job)
        if "FINISHED" in result:
            _announce(context, job)
        _redraw_panels(context)
        return result

    def cancel(self, context):
        """Blender's own cancel - a file load or a quit, not the user's ESC."""
        if getattr(self, "_job", None) is not None:
            self._job.stop()
        self._cleanup(context)

    def _cleanup(self, context):
        global RUNNING
        RUNNING = None
        wm = context.window_manager
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        try:
            wm.progress_end()
            context.workspace.status_text_set(None)
        except Exception:                               # noqa: BLE001
            pass

    def prepare(self, context):
        """Everything that touches Blender data, before the solve.

        Returns a Solve, or None after reporting why not. Kept apart from
        `apply` so the solve between them can run on a worker thread.
        """
        objs, err = job_objects(context)
        if err:
            self.report({"ERROR"}, err)
            return None

        self.was_edit = object_mode(context, objs)

        s = context.scene.cbu
        U, L, _ = _core()
        log = cbulog.get()
        T = cbulog.Timings()
        LAST_TIMINGS.clear()
        names = [o.name for o in objs]
        self.obj_name = names[0]
        self.names = names
        mats = [job_material(s, o, objs) for o in objs]
        self.mat = mat = mats[0]
        self.mats = dict(zip(names, mats))
        self.T = T
        # read now, not when the solve lands: a long Optimise outlives edits
        self.solved_with = unfold_settings(s, objs[0])
        self.solved_with["objects"] = tuple(names)
        if self.optimising:
            # only an Optimise read it, so only then can it go stale
            self.solved_with["board_bias"] = s.board_bias
        try:
            with cbulog.stage(T, "read mesh"):
                pms = [bridge.polymesh_from_object(
                    o, merge_coplanar=s.merge_coplanar,
                    mm_per_unit=s.mm_per_unit,
                    apply_transform=s.apply_transform, tol_deg=s.merge_tol)
                    for o in objs]
                k = size_factor(s, pms)
                if k != 1.0:
                    pms = [pm.scale(k) for pm in pms]
            self.scale = k
            for o, pm in zip(objs, pms):
                log.info("%s: %d face(s), %d vert(s), extents %s mm", o.name,
                         len(pm.faces), len(pm.vertices), pm.extents.round(1))

            with cbulog.stage(T, "check"):
                problems = [(f"{o.name}: {p}" if len(objs) > 1 else p)
                            for o, pm in zip(objs, pms) for p in pm.check()]
            for p in problems:
                log.warning("%s", p)
                self.report({"WARNING"}, p)

            limit = None
            if s.limit_to_bed:
                pad = 2 * mat.tab_depth + 2 * mat.kerf
                limit = (mat.usable_bed[0] - pad, mat.usable_bed[1] - pad)
                log.debug("island size limit %.1f x %.1f mm", *limit)

            members = []
            for o, pm, m in zip(objs, pms, mats):
                cuts, folds = bridge.read_constraints(o.data)
                if cuts or folds:
                    log.info("%s constraints: %d forced cut(s), %d forced "
                             "fold(s)", o.name, len(cuts), len(folds))
                members.append(Member(o.name, pm, dict(
                    trials=s.trials, seed=s.seed, max_size=limit,
                    forced_cuts=cuts, forced_folds=folds,
                    splinter_width=m.splinter_width,
                    mat=m if self.optimising else None, budget=s.budget,
                    board_bias=s.board_bias if self.optimising else 0.0,
                    flat_tol=m.flat_tol), m))

        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            log.error("unfold failed", exc_info=True)
            self.report({"ERROR"},
                        f"{type(exc).__name__}: {exc}  (full traceback in the "
                        f"system console)")
            if self.was_edit:
                bpy.ops.object.mode_set(mode="EDIT")
            return None

        # The mesh must not change under a solve that is already running on
        # its own thread, so remember what it looked like and check on the way
        # back out. The solve itself is safe either way - it holds a PolyMesh
        # copy - but writing a net onto geometry that has moved is not.
        self.fingerprints = {o.name: overlay.geometry_fingerprint(o.data)
                             for o in objs}
        return Solve(U, members[0].pm, members[0].kwargs, L=L, mat=mat,
                     timings=T, members=members)

    def apply(self, context, job):
        """Everything that touches Blender data, after the solve."""
        s = context.scene.cbu
        mat = self.mat
        log = cbulog.get()
        T = self.T
        try:
            objs = [bpy.data.objects.get(n) for n in self.names]
            if any(o is None or o.type != "MESH" for o in objs):
                raise RuntimeError("an object went away while unfolding")
            if job.error is not None:
                raise job.error
            for o in objs:
                if overlay.geometry_fingerprint(o.data) != self.fingerprints[o.name]:
                    raise RuntimeError(f"{o.name} changed while unfolding - "
                                       f"run Unfold again")
            if job.parts is None:
                raise RuntimeError("the solve finished without packing - "
                                   "this is a bug, not a setting")
            sheets, oversized = job.sheets, job.oversized
            group = tuple(self.names)
            _regroup(group)
            with cbulog.stage(T, "write mesh"):
                for o, m in zip(objs, job.members):
                    for isl in m.islands:
                        log.debug("%s island %d: %d face(s), %.0f x %.0f mm, "
                                  "%.0f mm2", o.name, isl.id, len(isl.faces),
                                  *isl.size(), isl.area())
                    bridge.write_islands(o.data, m.pm, m.islands)
                    _write_member(o, m.pm, m.islands, m.parts, sheets, mat,
                                  group)
                    _LAST[o.name] = (m.pm, m.islands, m.parts, sheets, m.rep)
                    _SOLVED_WITH[o.name] = self.solved_with
                try:
                    from .props import apply_blender_grid
                    apply_blender_grid(s, context)
                except Exception:                       # noqa: BLE001
                    pass            # a view setting must never fail an unfold

            with cbulog.stage(T, "overlay"):
                for o, m in zip(objs, job.members):
                    overlay.rebuild(o, m.pm, m.islands, s.overlay_alpha)
            LAST_TIMINGS.update(T)
            s.last_timings = T.report()
            log.info("timings: %s", s.last_timings)

            reps = [m.rep for m in job.members]
            rep = reps[0]
            n_isl = sum(r["islands"] for r in reps)
            s.last_islands = n_isl
            s.last_sheets = sheets
            flats = sum(r.get("flat_pieces", 0) for r in reps)
            s.last_report = (
                (f"{len(objs)} objects - " if len(objs) > 1 else "")
                + f"{n_isl} island(s) - {sheets} sheet(s) - "
                f"isometry {max(r['isometry_error'] for r in reps):.0e} - "
                f"{sum(r['trials_used'] for r in reps)} trial(s), "
                f"{sum(r['extra_cuts'] for r in reps)} repair, "
                f"{sum(r['size_splits'] for r in reps)} bed split(s)"
                + (f" - {flats} flat piece(s)" if flats else "")
                + (f" - scale x{self.scale:.4g}" if self.scale != 1.0 else ""))
            warns = [w for p in job.parts for w in p.warnings]
            warns += [f"{len(oversized)} part(s) larger than the bed"] if oversized else []
            warns += [n for r in reps for n in r.get("notes", [])]
            s.last_warnings = " | ".join(warns[:4])
            for w in warns[:6]:
                self.report({"WARNING"}, w)

            s.last_optimised = self.optimising
            s.last_elapsed = job.elapsed
            if self.optimising:
                n, m = rep.get("candidates"), rep.get("mended")
                frac = rep.get("split_frac", 1.0)
                s.last_detail = (
                    ("stopped early - " if job.stopping else "")
                    + f"best of {n} candidate(s), {m} mended"
                    + (f", pieces {frac:.0%} of bed" if frac < 1.0 else "")
                    if n is not None else "")
            else:
                s.last_detail = ""
            if job.stopping:
                self.report({"INFO"}, "stopped early - " + s.last_report)
            else:
                self.report({"INFO"}, s.last_report)
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            log.error("unfold failed", exc_info=True)
            self.report({"ERROR"},
                        f"{type(exc).__name__}: {exc}  (full traceback in the "
                        f"system console)")
            return {"CANCELLED"}
        finally:
            if self.was_edit:
                bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}


def _regroup(names) -> None:
    """`names` are packed together now. Anything they were packed with
    before and are not now keeps its own layout - still valid, its parts did
    not move - but must stop re-packing them, or a Re-pack there would drag
    them back onto its sheets."""
    names = tuple(names)
    for n in names:
        for old in _GROUP.get(n, ()):
            if old not in names:
                rest = tuple(x for x in _GROUP.get(old, ()) if x not in names)
                _GROUP[old] = rest or (old,)
                overlay.set_companions(old, [x for x in rest if x != old])


def _write_member(obj, pm, islands, parts, sheets, mat, group) -> None:
    """Write one object's share of a packing back to it: the net UVs where
    its parts landed on the shared sheets, the bed, and the cut outline."""
    bridge.write_net_uv(obj.data, pm, islands,
                        scale=bridge.uv_scale(*mat.bed),
                        placement={p.island_id: p for p in parts},
                        sheet_size=mat.bed)
    overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data),
                           sheets=sheets)
    publish_cuts(obj, parts, mat, pm, islands)
    _GROUP[obj.name] = tuple(group)
    overlay.set_companions(obj.name, [n for n in group if n != obj.name])


def pack_group(context, names, effort=0.0, progress=None, islands=None):
    """Rebuild every member's parts from its islands and pack them all onto
    one set of sheets - Re-pack, Refresh and a seam edit all go through here,
    so editing one object never takes it off the sheets it shares.

    `islands` overrides one member's islands ({name: islands}), for a seam
    edit that has not been stored yet. Returns (sheets, oversized). The
    caller has already left Edit Mode: UV data is empty there.
    """
    s = context.scene.cbu
    _U, L, _ = _core()
    objs = [bpy.data.objects.get(n) for n in names]
    missing = [n for n, o in zip(names, objs)
               if o is None or n not in _LAST]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} changed or went away since "
                           f"the unfold - run Unfold again")
    held = []
    for o in objs:
        pm, isl, _parts, _sh, rep = _LAST[o.name]
        isl = (islands or {}).get(o.name, isl)
        m = job_material(s, o, objs)
        held.append((o, pm, isl, rep, m, L.build_parts(isl, m)))
    mat = held[0][4]
    _regroup(names)
    allparts = [p for h in held for p in h[5]]
    sheets, oversized = L.pack(allparts, mat, effort=effort, progress=progress)
    for o, pm, isl, rep, m, parts in held:
        _write_member(o, pm, isl, parts, sheets, mat, names)
        _LAST[o.name] = (pm, isl, parts, sheets, rep)
    return sheets, oversized


def _redraw_panels(context) -> None:
    """Panels only repaint when something asks them to, and a worker thread
    changing numbers is not something Blender notices. Without this the
    progress bar is drawn once and then sits still."""
    try:
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type in {"VIEW_3D", "IMAGE_EDITOR"}:
                    area.tag_redraw()
    except Exception:                                   # noqa: BLE001
        pass


def _announce(context, job) -> None:
    """Say it is finished, in a way that cannot be missed.

    The status-bar report is transient and sits at the bottom edge of the
    screen; after a run long enough to walk away from, it has scrolled off
    before anyone looks. A popup waits for you. Only on the modal path - a
    script or a quick plain unfold has nobody waiting on it.
    """
    s = context.scene.cbu
    lines = [s.last_report, s.last_detail]

    def draw(menu, _context):
        for text in lines:
            if text:
                menu.layout.label(text=text)

    title = ("Optimise stopped" if job.stopping else "Optimise finished") \
        + f" in {job.elapsed:.0f} s"
    try:
        context.window_manager.popup_menu(draw, title=title, icon="CHECKMARK")
    except Exception:                                   # noqa: BLE001
        pass


class CBU_OT_unfold(_Unfold, bpy.types.Operator):
    bl_idname = "cbu.unfold"
    bl_label = "Unfold"
    bl_description = ("Flatten the mesh into non-overlapping islands, write "
                      "the net to a UV layer, and colour the islands in the "
                      "viewport. The quick one - see Optimise for a better net")
    optimising = False


class CBU_OT_optimise(_Unfold, bpy.types.Operator):
    bl_idname = "cbu.optimise"
    bl_label = "Optimise"
    bl_description = ("Unfold, but search for a better net: fewer small "
                      "pieces and fewer sheets. Slower - it mends and scores "
                      "many candidates, and packs the best of them. Shows "
                      "progress, and ESC stops it with the best net so far")
    optimising = True


class CBU_OT_edit_seam(bpy.types.Operator):
    bl_idname = "cbu.edit_seam"
    bl_label = "Edit Seam"
    bl_description = ("Cut or weld the selected edge(s) in place. The rest of "
                      "the net does not move and the unfold is NOT re-run - "
                      "press Unfold when you want the solver to reconsider")
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(items=[
        ("CUT", "Cut", "Cut here, detaching the piece on the far side"),
        ("FOLD", "Weld", "Never cut here - welds the two islands into one"),
    ], default="CUT")

    @classmethod
    def poll(cls, context):
        return (context.active_object is not None
                and context.active_object.type == "MESH"
                and context.mode == "EDIT_MESH")

    def execute(self, context):
        """Both directions are local, and neither re-runs the unfold.

        Re-unfolding would let the solver rebuild every tree, so the faces
        freed by a cut usually reattach somewhere else entirely and the whole
        layout jumps - correct, but not what someone editing one seam expects,
        and it throws away every other hand edit's placement. Cut detaches
        exactly the piece pointed at; Weld moves one island rigidly onto the
        other. Unfold stays the one explicit global re-solve, and it honours
        the marks these leave behind.
        """
        return self._local(context)

    def _local(self, context):
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

        # Record it first, so a later Unfold keeps the decision. This is the
        # whole point of editing in place: the mark persists, the layout does
        # not move, and the solver only reconsiders when asked.
        bpy.ops.cbu.mark(mode=self.mode)

        U, L, _ = _core()
        pm, islands, _parts, _sheets, rep = _LAST[obj.name]
        G = pm.dual_graph() if self.mode == "FOLD" else None
        core_of = {}
        for ci, grp in enumerate(pm.source_faces):
            for poly in grp:
                core_of[poly] = ci

        done = 0
        for _eidx, (p0, p1) in picks:
            a, b = core_of.get(p0), core_of.get(p1)
            if a is None or b is None or a == b:
                continue            # inside one merged n-gon: nothing to do
            if self.mode == "CUT":
                host = next((i for i in islands
                             if a in i.faces and b in i.faces), None)
                if host is None:
                    continue        # already in different islands
                nid = max(i.id for i in islands) + 1
                seam = max([c.seam for i in islands for c in i.cuts] + [-1]) + 1
                out = U.split_island(host, a, b, nid, seam)
                if out is None:
                    continue
                islands[islands.index(host)] = out[0]
                islands.append(out[1])
            else:
                ia = next((i for i in islands if a in i.faces), None)
                ib = next((i for i in islands if b in i.faces), None)
                if ia is None or ib is None or ia is ib:
                    continue        # already one island
                out = U.join_islands(ia, ib, pm, G, a, b)
                if out is None:
                    continue
                islands.remove(ia)
                islands.remove(ib)
                islands.append(out)
            done += 1

        if not done:
            self.report({"WARNING"},
                        "that edge is already cut in this net" if self.mode == "CUT"
                        else "those faces are already in one island")
            return {"CANCELLED"}

        s = context.scene.cbu
        object_mode(context, [o for o in (bpy.data.objects.get(n)
                                          for n in group_of(obj.name))
                              if o is not None])
        try:
            # The whole group re-packs: the sheets are shared, so packing
            # this object alone would put it on top of the others' parts.
            sheets, _over = pack_group(context, group_of(obj.name),
                                       islands={obj.name: islands})
            bridge.write_islands(obj.data, pm, islands)
            overlay.rebuild(obj, pm, islands, s.overlay_alpha)
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        finally:
            bpy.ops.object.mode_set(mode="EDIT")

        s.last_islands = sum(len(_LAST[n][1]) for n in group_of(obj.name))
        s.last_sheets = sheets
        self.report({"INFO"},
                    f"{'detached' if self.mode == 'CUT' else 'welded'} "
                    f"{done} - {len(islands)} island(s), {sheets} sheet(s)")
        return {"FINISHED"}


class CBU_OT_repack(bpy.types.Operator):
    bl_idname = "cbu.repack"
    bl_label = "Re-pack Sheets"
    bl_description = ("Run the nesting again on the current islands, without "
                      "re-unfolding. Useful after editing seams")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = job_ref(context)
        return obj is not None and obj.name in _LAST

    def execute(self, context):
        obj = job_ref(context)
        s = context.scene.cbu

        was_edit = object_mode(
            context, [o for o in (bpy.data.objects.get(n)
                                  for n in group_of(obj.name)) if o is not None])
        try:
            # Rebuild from the islands rather than re-packing the packed
            # parts: Part.xform has to map island space to the sheet, and
            # packing an already-packed part would compose two placements and
            # put the net somewhere it is not. Every object packed with this
            # one comes too - they share the sheets.
            # Effort can mean a minute on a big net, on the main thread: at
            # least show the count on the cursor while it works.
            wm = context.window_manager
            wm.progress_begin(0, 100)

            def tick(done, total, _note):
                wm.progress_update(int(100 * done / max(1, total)))
                return True
            try:
                sheets, oversized = pack_group(context, group_of(obj.name),
                                               effort=s.pack_effort,
                                               progress=tick)
            finally:
                wm.progress_end()
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


class CBU_OT_refresh(bpy.types.Operator):
    bl_idname = "cbu.refresh"
    bl_label = "Refresh"
    bl_description = ("Rebuild the parts from the current net with the "
                      "current settings - fingers, corner holes, tabs, relief, "
                      "labels - and re-pack, without re-unfolding. Settings "
                      "the unfold itself reads (bed, height, trials...) still "
                      "need Unfold or Optimise; an Optimise was chosen with "
                      "the settings it ran with, and Refresh keeps that net")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = job_ref(context)
        return obj is not None and obj.name in _LAST and RUNNING is None

    def execute(self, context):
        res = bpy.ops.cbu.repack()
        if "FINISHED" not in res:
            return res
        changed = unfold_settings_changed(job_ref(context),
                                          context.scene.cbu)
        if changed:
            self.report({"WARNING"},
                        f"Parts refreshed, but "
                        f"{setting_labels(context.scene.cbu, changed)} changed "
                        f"since the last solve - Unfold or Optimise to apply "
                        f"{'it' if len(changed) == 1 else 'them'}")
        else:
            self.report({"INFO"}, f"Refreshed: {context.scene.cbu.last_sheets} "
                                  f"sheet(s)")
        return {"FINISHED"}


class CBU_OT_export(bpy.types.Operator):
    bl_idname = "cbu.export"
    bl_label = "Export Sheets"
    bl_description = "Write layered SVG (and DXF) plus a manifest, one file per sheet"

    @classmethod
    def poll(cls, context):
        obj = job_ref(context)
        return obj is not None and obj.name in _LAST

    def execute(self, context):
        obj = job_ref(context)
        s = context.scene.cbu
        _, _, E = _core()
        group = group_of(obj.name)
        missing = [n for n in group if n not in _LAST]
        if missing:
            self.report({"ERROR"},
                        f"{', '.join(missing)} changed since the unfold - its "
                        f"parts would be missing from the sheets. Unfold again")
            return {"CANCELLED"}
        # Every object packed onto these sheets, or the sheets are cut with
        # holes where the others' parts were meant to go.
        parts = [p for n in group for p in _LAST[n][2]]
        _pm, _islands, _parts, sheets, rep = _LAST[obj.name]
        if len(group) > 1:
            rep = dict(rep, objects=list(group))
        stem = obj.name.lower() if len(group) == 1 else (
            bpy.path.display_name_from_filepath(bpy.data.filepath).lower()
            or "job")
        try:
            out = bpy.path.abspath(s.out_dir)
            files = E.write_all(out, parts, material_from(s, obj), sheets, rep,
                                dxf=s.write_dxf, stem=stem,
                                gcode=s.write_gcode)
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
        if len(group_of(obj.name)) > 1:
            # It nests one mesh's UVs with Blender's packer; the other
            # objects' parts are on other meshes, so it cannot see them.
            self.report({"WARNING"}, "Nest Sheets works on one object - "
                                     "Re-pack packs several together")
            return {"CANCELLED"}
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
                            sheet_size=mat.bed)
        _LAST[obj.name] = (pm, islands, parts, sheets, rep)
        overlay.set_net_bounds(obj.name, bridge.net_bounds(obj.data),
                               sheets=sheets)
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


# ----------------------------------------------------- the list, the stick

class CBU_OT_job_add(bpy.types.Operator):
    bl_idname = "cbu.job_add"
    bl_label = "Add Selected"
    bl_description = ("Add the selected mesh objects to the list of what is "
                      "cut together. Everything in the list is unfolded, "
                      "scaled by one factor and packed onto shared sheets")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.cbu
        have = {it.obj for it in s.job if it.obj is not None}
        picked = [o for o in context.selected_objects if o.type == "MESH"]
        if not picked and context.active_object is not None                 and context.active_object.type == "MESH":
            picked = [context.active_object]
        added = 0
        for o in picked:
            if o in have or o == s.yardstick:
                continue
            s.job.add().obj = o
            have.add(o)
            added += 1
        s.job_index = max(0, len(s.job) - 1)
        if not added:
            self.report({"WARNING"}, "nothing new to add (the yardstick is "
                                     "never cut)")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{len(s.job)} object(s) cut together")
        return {"FINISHED"}


class CBU_OT_job_remove(bpy.types.Operator):
    bl_idname = "cbu.job_remove"
    bl_label = "Remove"
    bl_description = "Take the highlighted object out of the list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.cbu.job) > 0

    def execute(self, context):
        s = context.scene.cbu
        i = min(s.job_index, len(s.job) - 1)
        s.job.remove(i)
        s.job_index = max(0, min(i, len(s.job) - 1))
        return {"FINISHED"}


class CBU_OT_job_clear(bpy.types.Operator):
    bl_idname = "cbu.job_clear"
    bl_label = "Clear List"
    bl_description = ("Empty the list: Unfold goes back to cutting just the "
                      "active object")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.cbu.job) > 0

    def execute(self, context):
        context.scene.cbu.job.clear()
        context.scene.cbu.job_index = 0
        return {"FINISHED"}


def world_z_mm(objs, mm_per_unit: float, apply_transform: bool):
    """(lowest, highest) z over every vertex of `objs`, in mm - what Height
    measures, without building a PolyMesh."""
    lo, hi = np.inf, -np.inf
    for o in objs:
        me = o.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        if apply_transform:
            M = np.array(o.matrix_world.to_4x4())
            co = co @ M[:3, :3].T + M[:3, 3]
        if len(co):
            lo, hi = min(lo, co[:, 2].min()), max(hi, co[:, 2].max())
    return lo * mm_per_unit, hi * mm_per_unit


class CBU_OT_add_yardstick(bpy.types.Operator):
    bl_idname = "cbu.add_yardstick"
    bl_label = "Add Yardstick"
    bl_description = ("Add a one-edge reference at the 3D cursor and size by "
                      "it. Put its ends on a dimension you know - snap to "
                      "vertices for an exact one - and type its real length. "
                      "It is never cut")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.cbu
        # Out of Edit Mode first: the stick becomes the active object, and a
        # model left in Edit Mode behind it is one nothing can read.
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        objs, _err = job_objects(context)
        # A yardstick means a saved job: once it is the active object (it is
        # about to be, so it can be moved straight away), "the active object"
        # would be the stick, and the model has to be named some other way.
        if not len(s.job):
            for o in objs:
                s.job.add().obj = o
        # Span the model along X, so it is visible and roughly the right size.
        if objs:
            xs = []
            for o in objs:
                for c in o.bound_box:
                    xs.append((o.matrix_world @ Vector(c)).x)
            span = max(xs) - min(xs) or 1.0
        else:
            span = 1.0
        # Keep the size it had: the stick's length now, at the scale the
        # model would get anyway, so switching to it changes nothing until a
        # length is typed.
        k = 1.0
        if objs and s.size_mode == "HEIGHT" and s.target_height > 0:
            lo, hi = world_z_mm(objs, s.mm_per_unit, s.apply_transform)
            if hi - lo > 1e-9:
                k = s.target_height / (hi - lo)

        me = bpy.data.meshes.new("CBU Yardstick")
        me.from_pydata([(-span / 2, 0, 0), (span / 2, 0, 0)], [(0, 1)], [])
        stick = bpy.data.objects.new("CBU Yardstick", me)
        stick.location = context.scene.cursor.location
        stick.show_in_front = True          # inside a helmet it is otherwise hidden
        stick.show_name = True
        context.collection.objects.link(stick)
        for o in context.selected_objects:
            o.select_set(False)
        stick.select_set(True)
        context.view_layer.objects.active = stick
        s.yardstick = stick
        s.size_mode = "YARDSTICK"
        s.yardstick_length = bridge.yardstick_mm(
            stick, s.mm_per_unit, s.apply_transform) * k
        self.report({"INFO"}, "Yardstick added: put its ends on a known "
                              "dimension and type its real length")
        return {"FINISHED"}

classes = (CBU_OT_mark, CBU_OT_clear_marks, CBU_OT_select_marked,
           CBU_OT_lock, CBU_OT_clear_locks, CBU_OT_select_locked,
           CBU_OT_select_all_faces,
           CBU_OT_split_nonplanar, CBU_OT_unfold, CBU_OT_optimise,
           CBU_OT_edit_seam,
           CBU_OT_repack, CBU_OT_refresh, CBU_OT_export, CBU_OT_pack_uv,
           CBU_OT_job_add, CBU_OT_job_remove, CBU_OT_job_clear,
           CBU_OT_add_yardstick)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    _LAST.clear()
    _GROUP.clear()
    LAST_TIMINGS.clear()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
