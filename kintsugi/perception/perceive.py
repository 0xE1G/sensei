"""Perception: masks -> point clouds -> object descriptors the skills use.

Two segmenters sit behind one interface:
  * "sim"  - the simulator's own segmentation (ideal masks), used for bulk trials.
  * "sam"  - SAM 3 through the Meta API (added once the API key is available).
Descriptors are computed from depth geometry in both cases, so skills never
read true simulator state. "ground_truth" mode exists only for the scripted
expert and for measuring the perception gap.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .camera import Camera


def _geom_to_object(model, object_ids):
    """geom -> object id, including geoms of an object's moving child bodies (drawer, door)."""
    roots = {model.body(oid).id: oid for oid in object_ids}
    lut = {}
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b > 0 and b not in roots:
            b = model.body_parentid[b]
        if b in roots:
            lut[g] = roots[b]
    return lut


PART_NAMES = ("handle", "front")


def _geom_to_part(model, lut):
    """geom -> (object id, part) for named parts (<id>__handle_*, <id>__front)."""
    out = {}
    for g, oid in lut.items():
        name = model.geom(g).name or ""
        for part in PART_NAMES:
            if name.startswith(f"{oid}__{part}"):
                out[g] = (oid, part)
    return out


def describe_part(points: np.ndarray, obj_center, panel: bool = False, normal=None) -> dict | None:
    """A part seen in depth: its 3D centre, size, and the horizontal direction pointing out of the object.
    panel=True (a flat front): the normal is perpendicular to the panel's long horizontal axis, turned away from
    the object's centre. Otherwise the given normal (e.g. the front it is mounted on) or centre -> part."""
    if points is None or len(points) < 6:
        return None
    lo, hi = np.percentile(points, 3, axis=0), np.percentile(points, 97, axis=0)
    c = (lo + hi) / 2
    out_dir = c[:2] - np.asarray(obj_center, float)
    if panel and len(points) >= 20:
        xy = points[:, :2] - points[:, :2].mean(0)
        _, v = np.linalg.eigh(np.cov(xy.T))
        n = np.array([-v[1, -1], v[0, -1]])               # perpendicular to the long axis
        n = n if n @ out_dir >= 0 else -n
    elif normal is not None:
        n = np.asarray(normal, float)
    else:
        n = out_dir
    n = n / max(np.linalg.norm(n), 1e-6)
    tangent = np.array([-n[1], n[0]])
    proj = (points[:, :2] - c[:2]) @ tangent
    # tip: the outermost points along the normal (a handle's graspable post sticks out furthest)
    out_ = points[:, :2] @ n
    tip_pts = points[out_ >= np.percentile(out_, 85)]
    tip = [float(v) for v in np.median(tip_pts, axis=0)]
    return {"center": [float(v) for v in c], "tip": tip, "normal": [float(n[0]), float(n[1])],
            "width": float(np.percentile(proj, 97) - np.percentile(proj, 3)), "height": float(hi[2] - lo[2]),
            "n_points": int(len(points))}


def describe(points: np.ndarray, table_z: float = 0.0) -> dict | None:
    """Shape descriptor from a world-frame point cloud of one object."""
    if points is None or len(points) < 15:
        return None
    pts = points[points[:, 2] > table_z + 0.002]
    if len(pts) < 15:
        # A flat marker on the table (a coloured zone): every point is at table height. Describe its footprint.
        if np.percentile(points[:, 2], 98) < table_z + 0.004:
            lo, hi = np.percentile(points[:, :2], 2, axis=0), np.percentile(points[:, :2], 98, axis=0)
            ext = hi - lo
            return {"center": [float(v) for v in (lo + hi) / 2], "top_z": table_z + 0.001, "height": 0.001,
                    "width": float(np.hypot(*ext)), "extent_xy": [float(ext[0]), float(ext[1])], "yaw": 0.0,
                    "axis_yaw": 0.0, "extent_major": float(ext.max()), "extent_minor": float(ext.min()),
                    "elongation": float(ext.max() / max(ext.min(), 1e-3)), "flat": True, "marker": True,
                    "hollow": False, "n_points": int(len(points))}
        return None
    top = float(np.percentile(pts[:, 2], 98))
    xy = pts[:, :2]
    lo, hi = np.percentile(xy, 2, axis=0), np.percentile(xy, 98, axis=0)
    # Centre from the top band (rim or top face) when it is well seen: a handle or other lower part must not drag
    # the centre sideways (a mug's handle made the hollow test look in the wrong place).
    band = xy[pts[:, 2] > top - 0.01]
    if len(band) >= 20:
        c = (np.percentile(band, 2, axis=0) + np.percentile(band, 98, axis=0)) / 2
    else:
        c = (lo + hi) / 2
    r = np.linalg.norm(xy - c, axis=1)
    R = float(np.percentile(r, 97))                       # full extent (incl. handles): footprint and width
    R_top = float(np.percentile(np.linalg.norm(band - c, axis=1), 97)) if len(band) >= 20 else R
    inner = pts[r < 0.4 * R_top]
    centre_z = float(np.median(inner[:, 2])) if len(inner) > 5 else top
    deep_centre = (top - centre_z) > max(0.012, 0.35 * top)
    # Ring test: the top surface has (almost) no points near the middle -> an opening.
    top_pts = xy[pts[:, 2] > top - 0.006]
    ring = len(top_pts) > 20 and np.mean(np.linalg.norm(top_pts - c, axis=1) < 0.5 * R_top) < 0.04
    hollow = (deep_centre or ring) and R_top > 0.025
    # Yaw of the top face from its minimum-area bounding rectangle. The principal axis is undefined for a square
    # top (a cube came out up to 45 deg off, so a gripper closed across the diagonal); the rectangle is not.
    topf = xy[pts[:, 2] > top - 0.006]
    yaw = 0.0
    if len(topf) > 10:
        q = topf - topf.mean(0)
        best = None
        for a in np.radians(np.arange(0.0, 90.0, 1.0)):
            u, v = q @ [math.cos(a), math.sin(a)], q @ [-math.sin(a), math.cos(a)]
            area = (np.percentile(u, 98) - np.percentile(u, 2)) * (np.percentile(v, 98) - np.percentile(v, 2))
            if best is None or area < best[0]:
                best = (area, a, np.ptp(u) >= np.ptp(v))
        _, a, u_long = best
        yaw = float(a if u_long else a + math.pi / 2)        # along the rectangle's longer side
    ext = hi - lo
    # Principal axes of the footprint (all points, not just the top face): long/short extent and elongation.
    cov = np.cov((xy - xy.mean(0)).T)
    w, v = np.linalg.eigh(cov)
    major, minor = v[:, -1], v[:, 0]
    pm, pn = (xy - c) @ major, (xy - c) @ minor
    e_major = float(np.percentile(pm, 98) - np.percentile(pm, 2))
    e_minor = float(np.percentile(pn, 98) - np.percentile(pn, 2))
    d = {"center": [float(c[0]), float(c[1])], "top_z": top, "height": top - table_z,
         "width": float(2 * R), "extent_xy": [float(ext[0]), float(ext[1])], "yaw": yaw,
         "axis_yaw": float(math.atan2(major[1], major[0])), "extent_major": e_major, "extent_minor": e_minor,
         "elongation": float(e_major / max(e_minor, 1e-3)), "flat": bool(top - table_z < 0.02),
         "hollow": bool(hollow), "n_points": int(len(pts))}
    if hollow:
        rim = pts[pts[:, 2] > top - 0.006]
        rr = np.linalg.norm(rim[:, :2] - c, axis=1)
        d["rim_radius"] = float(np.percentile(rr, 90))
        d["rim_z"] = top
        d["floor_z"] = centre_z
        # Wall slope (radius change per metre of depth), from the outer radius 2 cm below the rim.
        band = pts[(pts[:, 2] > top - 0.024) & (pts[:, 2] < top - 0.016)]
        if len(band) > 10:
            rb = float(np.percentile(np.linalg.norm(band[:, :2] - c, axis=1), 90))
            d["wall_slope"] = float(np.clip((d["rim_radius"] - rb) / 0.02, 0.0, 1.5))
        else:
            d["wall_slope"] = 0.0
    return d


def _primitive_surface(gtype: int, size, rng, n: int = 8000) -> np.ndarray | None:
    """Points spread over the surface of a box / cylinder / sphere geom, in the geom's frame."""
    G = mujoco.mjtGeom
    if gtype == G.mjGEOM_BOX:
        h = np.asarray(size[:3], float)
        pts = rng.uniform(-h, h, (n, 3))
        areas = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]])
        ax = rng.choice(3, n, p=areas / areas.sum())
        pts[np.arange(n), ax] = h[ax] * rng.choice([-1.0, 1.0], n)
        return pts
    if gtype == G.mjGEOM_CYLINDER:
        r, hz = float(size[0]), float(size[1])
        side = rng.random(n) < (2 * math.pi * r * 2 * hz) / (2 * math.pi * r * 2 * hz + 2 * math.pi * r * r)
        th = rng.uniform(0, 2 * math.pi, n)
        rr = np.where(side, r, r * np.sqrt(rng.random(n)))
        z = np.where(side, rng.uniform(-hz, hz, n), hz * rng.choice([-1.0, 1.0], n))
        return np.stack([rr * np.cos(th), rr * np.sin(th), z], axis=1)
    if gtype == G.mjGEOM_SPHERE:
        v = rng.normal(size=(n, 3))
        return float(size[0]) * v / np.linalg.norm(v, axis=1, keepdims=True)
    return None


class Perception:
    def __init__(self, scene, mode: str = "sim", cameras=("front", "top"), width=320, height=240):
        if mode == "sam":                    # never fall back silently to simulator masks
            from .sam import availability
            ok, why = availability()
            if not ok:
                raise RuntimeError(why)
        self.scene, self.mode = scene, mode
        self.calls = 0
        self.cams = [Camera(scene.model, scene.data, c, width, height) for c in cameras] if mode != "ground_truth" else []
        self.lut = _geom_to_object(scene.model, list(scene.objects))
        self.part_lut = _geom_to_part(scene.model, self.lut)
        # Objects with named parts get one extra camera on the robot's side (cabinet fronts face the robot); it is
        # used for part points only, so object descriptors keep coming from the same cameras as before.
        self.part_cams = ([Camera(scene.model, scene.data, "near", width, height)]
                          if self.part_lut and mode != "ground_truth" else [])
        self.last_points = {}        # object id -> world-frame point cloud from the latest perceive()

    def perceive(self, object_ids=None) -> dict:
        """Return {object_id: descriptor}. One call = one SAM request per camera when mode == 'sam'."""
        object_ids = list(object_ids or self.scene.objects)
        self.calls += 1
        if self.mode == "ground_truth":
            return {o: self._truth(o) for o in object_ids}
        if self.mode == "sam":
            raise NotImplementedError("SAM segmenter not wired yet: add the Meta API key, then implement sam.py")
        pts = {o: [] for o in object_ids}
        for cam in self.cams:
            seg, depth = cam.segmentation(), cam.depth()
            for o in object_ids:
                gids = [g for g, oo in self.lut.items() if oo == o]
                mask = np.isin(seg, gids)
                if mask.sum() > 0:
                    pts[o].append(cam.points(depth, mask))
        out = {}
        for o in object_ids:
            P = np.concatenate(pts[o]) if pts[o] else None
            self.last_points[o] = P
            out[o] = describe(P)
        self._add_parts(out, object_ids)
        return out

    def _add_parts(self, out, object_ids):
        wanted = {(o, p) for o, p in self.part_lut.values() if o in object_ids and out.get(o)}
        if not wanted:
            return
        pts = {k: [] for k in wanted}
        for cam in self.cams + self.part_cams:
            seg, depth = cam.segmentation(), cam.depth()
            for key in wanted:
                gids = [g for g, k in self.part_lut.items() if k == key]
                mask = np.isin(seg, gids)
                if mask.sum() > 0:
                    pts[key].append(cam.points(depth, mask))
        # the front first: a handle faces the way the front it is mounted on faces
        for (o, part) in sorted(pts, key=lambda k: k[1] != "front"):
            chunks = pts[(o, part)]
            front = out[o].get("parts", {}).get("front")
            d = describe_part(np.concatenate(chunks) if chunks else None, out[o]["center"], panel=part == "front",
                              normal=front["normal"] if front and part != "front" else None)
            if d:
                out[o].setdefault("parts", {})[part] = d

    def _truth(self, oid) -> dict:
        m, d = self.scene.model, self.scene.data
        meta = self.scene.objects[oid]
        b = m.body(oid).id
        pos = d.xpos[b]
        R = d.xmat[b].reshape(3, 3)
        yaw = float(math.atan2(R[1, 0], R[0, 0]))
        e_major, e_minor = meta.get("extent_major", meta["width"]), meta.get("extent_minor", meta["width"])
        out = {"center": [float(pos[0]), float(pos[1])], "height": meta["height"], "width": meta["width"],
               "yaw": yaw, "axis_yaw": yaw, "extent_major": e_major, "extent_minor": e_minor,
               "elongation": e_major / max(e_minor, 1e-3), "flat": meta["height"] < 0.02, "hollow": meta["hollow"],
               "top_z": float(pos[2] + meta.get("bottom_offset", 0) + meta["height"])}
        if meta["template"] == "bowl":
            out["wall_slope"] = math.tan(math.radians(meta["params"]["flare_deg"]))
        elif meta["template"] in ("cup", "mug"):
            out["wall_slope"] = math.tan(math.radians(2))
        for k in ("rim_radius", "rim_z", "floor_z", "half_x", "half_y"):
            if k in meta:
                out[k] = meta[k] + (float(pos[2]) if k in ("rim_z", "floor_z") else 0.0)
        if meta.get("articulated"):
            out["parts"] = self._truth_parts(oid)
        elif meta.get("asset") or (not meta["fixed"] and R[2, 2] < math.cos(math.radians(25))):
            # an imported mesh, or any object that is not standing (lying on its side, tipped over): the template's
            # upright geometry does not describe it; use its true surface
            mesh = self._truth_mesh(oid)
            if mesh:
                out.update(mesh)
        return out

    def _truth_mesh(self, oid) -> dict | None:
        """An imported mesh object's true descriptor: its body origin is not its centre and its shape is not a
        primitive, so the true visible surface (area-sampled from the meshes at the true pose, reduced to a
        top-down height map) goes through the same describe() as perception, without occlusion or depth noise."""
        m, d = self.scene.model, self.scene.data
        cache = self.__dict__.setdefault("_mesh_pts", {})
        b = m.body(oid).id
        if oid not in cache:
            rng = np.random.default_rng(0)
            local = []
            G = mujoco.mjtGeom
            has_mesh = any(o == oid and m.geom_type[g] == G.mjGEOM_MESH and m.geom_group[g] <= 2
                           for g, o in self.lut.items())
            for g, o in self.lut.items():
                if o != oid or m.geom_group[g] > 2:
                    continue
                if m.geom_type[g] == G.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                    f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                    a, b_, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
                    area = 0.5 * np.linalg.norm(np.cross(b_ - a, c - a), axis=1)
                    if area.sum() <= 0:
                        continue
                    k = rng.choice(len(f), 30000, p=area / area.sum())
                    u, w = rng.random((2, len(k)))
                    flip = u + w > 1
                    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
                    pts = a[k] + u[:, None] * (b_[k] - a[k]) + w[:, None] * (c[k] - a[k])
                elif not has_mesh:                       # a primitive-shape object (built-in template)
                    pts = _primitive_surface(int(m.geom_type[g]), m.geom_size[g], rng)
                    if pts is None:
                        continue
                else:
                    continue
                # geom frame -> body frame (fixed for a rigid object)
                Rg = d.geom_xmat[g].reshape(3, 3); Rb = d.xmat[b].reshape(3, 3)
                local.append((pts @ Rg.T + d.geom_xpos[g] - d.xpos[b]) @ Rb)
            cache[oid] = np.concatenate(local) if local else None
        L = cache[oid]
        if L is None:
            return None
        W = L @ d.xmat[b].reshape(3, 3).T + d.xpos[b]
        cell = np.floor(W[:, :2] / 0.003).astype(np.int64)
        order = np.lexsort((-W[:, 2], cell[:, 1], cell[:, 0]))      # highest point per 3 mm cell
        cs = cell[order]
        first = np.ones(len(cs), bool)
        first[1:] = np.any(cs[1:] != cs[:-1], axis=1)
        desc = describe(W[order][first])
        return desc or None

    def _truth_parts(self, oid) -> dict:
        """True front / handle of a moving part (same fields as describe_part): the front faces its local -x."""
        m, d = self.scene.model, self.scene.data
        names = {g: m.geom(g).name or "" for g in range(m.ngeom)}
        f = next(g for g, n in names.items() if n.startswith(f"{oid}__front"))
        R = d.geom_xmat[f].reshape(3, 3)
        n = -R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-6)
        fc, fs = d.geom_xpos[f], m.geom_size[f]
        front = {"center": [float(v) for v in fc], "tip": [float(v) for v in fc], "normal": [float(n[0]), float(n[1])],
                 "width": float(2 * fs[1]), "height": float(2 * fs[2])}
        hs_ = [g for g, n in names.items() if n.startswith(f"{oid}__handle")]
        posts = [g for g in hs_ if "post" in names[g]]
        h = posts[0] if posts else max(hs_, key=lambda g: float(d.geom_xpos[g][:2] @ n))   # the part sticking out most
        hc, hs = d.geom_xpos[h], m.geom_size[h]
        # tip = middle of the post the fingers pinch (what depth perception reports as the outermost points)
        handle = {"center": [float(v) for v in hc], "tip": [float(v) for v in hc], "normal": front["normal"],
                  "width": float(2 * hs[0]), "height": float(2 * hs[2])}
        return {"front": front, "handle": handle}

    def close(self):
        for c in self.cams + self.part_cams:
            c.close()
