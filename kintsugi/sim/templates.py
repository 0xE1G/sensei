"""Parametric object templates built from simple shapes with MjSpec.

Every template takes a small dict of parameters (the language model will fill
these in later) and adds one body to the scene. Each template also returns a
`meta` dict with the object's true geometry, used ONLY by the scripted expert,
the goal checks and the failure labeler -- never by the robot's skills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

COLORS = {
    "red": [0.85, 0.15, 0.15, 1], "green": [0.15, 0.7, 0.25, 1], "blue": [0.15, 0.35, 0.9, 1],
    "yellow": [0.95, 0.8, 0.1, 1], "orange": [0.95, 0.5, 0.1, 1], "purple": [0.55, 0.25, 0.75, 1],
    "white": [0.92, 0.92, 0.92, 1], "gray": [0.55, 0.55, 0.55, 1], "brown": [0.55, 0.38, 0.22, 1],
    "black": [0.12, 0.12, 0.12, 1],
}

# Allowed ranges per template. Anything the model asks for is clipped into these,
# so every generated object is one we have tested to simulate stably.
RANGES = {
    "block":    {"size": (0.03, 0.065), "mass": (0.02, 0.3), "friction": (0.4, 1.5)},
    "ball":     {"radius": (0.018, 0.035), "mass": (0.01, 0.2), "friction": (0.4, 1.5)},
    "cylinder": {"radius": (0.015, 0.035), "height": (0.05, 0.16), "mass": (0.02, 0.4), "friction": (0.4, 1.5)},
    "plate":    {"radius": (0.065, 0.09), "thickness": (0.008, 0.014), "mass": (0.05, 0.3), "friction": (0.3, 0.8)},
    "bar":      {"length": (0.08, 0.22), "width": (0.02, 0.045), "height": (0.02, 0.045), "mass": (0.02, 0.3),
                 "friction": (0.4, 1.5)},
    "bowl":     {"radius": (0.055, 0.10), "depth": (0.035, 0.07), "wall": (0.004, 0.012), "flare_deg": (10, 35),
                 "mass": (0.05, 0.4), "friction": (0.4, 1.5)},
    "mug":      {"radius": (0.035, 0.048), "height": (0.08, 0.11), "wall": (0.005, 0.008), "handle": (0.02, 0.035),
                 "mass": (0.08, 0.35), "friction": (0.4, 1.5)},
    "cup":      {"radius": (0.03, 0.05), "height": (0.07, 0.12), "wall": (0.004, 0.01), "mass": (0.04, 0.3),
                 "friction": (0.4, 1.5)},
    "bin":      {"width": (0.12, 0.26), "depth": (0.12, 0.26), "wall_height": (0.04, 0.10)},
    "tray":     {"width": (0.14, 0.30), "depth": (0.12, 0.24)},
    "zone":     {"width": (0.08, 0.25), "depth": (0.08, 0.25)},
    # Objects with a moving part (a fixed cabinet + a part on a joint). Their front faces the robot.
    "drawer":   {"width": (0.16, 0.26), "depth": (0.14, 0.20), "height": (0.08, 0.12), "travel": (0.08, 0.15),
                 "damping": (0.5, 3.0)},
    "door":     {"width": (0.12, 0.20), "depth": (0.12, 0.18), "height": (0.10, 0.16), "hinge_side": (0.0, 1.0),
                 "damping": (0.02, 0.2)},
}

DEFAULTS = {
    "block": {"size": 0.045, "mass": 0.08, "friction": 1.0, "color": "red"},
    "ball": {"radius": 0.025, "mass": 0.05, "friction": 1.0, "color": "orange"},
    "cylinder": {"radius": 0.025, "height": 0.10, "mass": 0.1, "friction": 1.0, "color": "white"},
    "bar": {"length": 0.15, "width": 0.03, "height": 0.03, "mass": 0.08, "friction": 1.0, "color": "yellow"},
    "plate": {"radius": 0.075, "thickness": 0.01, "mass": 0.12, "friction": 0.5, "color": "white"},
    "bowl": {"radius": 0.075, "depth": 0.05, "wall": 0.007, "flare_deg": 22, "mass": 0.15, "friction": 1.0,
             "color": "white"},
    "cup": {"radius": 0.038, "height": 0.09, "wall": 0.006, "mass": 0.1, "friction": 1.0, "color": "white"},
    "mug": {"radius": 0.04, "height": 0.095, "wall": 0.006, "handle": 0.028, "mass": 0.18, "friction": 1.0,
            "color": "red"},
    "bin": {"width": 0.20, "depth": 0.16, "wall_height": 0.06, "color": "brown"},
    "tray": {"width": 0.20, "depth": 0.16, "color": "gray"},
    "zone": {"width": 0.14, "depth": 0.14, "color": "red"},
    "drawer": {"width": 0.20, "depth": 0.16, "height": 0.10, "travel": 0.13, "damping": 1.5, "color": "white"},
    "door": {"width": 0.16, "depth": 0.15, "height": 0.13, "hinge_side": 1.0, "damping": 0.08, "color": "white"},
}

# Moving part per articulated template: joint type and the unit a goal on it uses (m for slides, deg for hinges).
ARTICULATED = {"drawer": {"part": "drawer", "type": "slide", "unit": "m"},
               "door": {"part": "door", "type": "hinge", "unit": "deg"}}

# Templates that are containers or targets default to fixed to the table.
FIXED_BY_DEFAULT = {"bin", "tray", "zone", "drawer", "door"}

OBJ_FRICTION = lambda f: [float(f), 0.005, 0.0001]


@dataclass
class ObjectSpec:
    id: str
    template: str
    params: dict = field(default_factory=dict)
    role: str = "item"          # "item" (to be moved) or "target"
    fixed: bool | None = None    # None -> template default


def resolve_params(template: str, params: dict) -> dict:
    if template not in DEFAULTS:
        raise ValueError(f"Unknown template '{template}'. Known: {sorted(DEFAULTS)}")
    p = dict(DEFAULTS[template])
    p.update(params or {})
    for k, (lo, hi) in RANGES[template].items():
        p[k] = float(np.clip(float(p[k]), lo, hi))
    if isinstance(p.get("color"), str):
        if p["color"] not in COLORS:
            raise ValueError(f"Unknown color '{p['color']}'. Known: {sorted(COLORS)}")
    return p


def _rgba(p):
    c = p.get("color", "white")
    return list(COLORS[c]) if isinstance(c, str) else list(c)


def _quat_yaw(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out.tolist()


def _ring(body, n, r_inner, base_z, wall_h, wall_t, flare, rgba, mass_total, friction, prefix):
    """Ring of n thin boxes forming a (possibly flared) wall. Returns outer rim radius at the top."""
    r_mid = r_inner + wall_t / 2
    chord = 2 * (r_mid + wall_h * math.sin(flare)) * math.tan(math.pi / n) * 1.08
    for i in range(n):
        th = 2 * math.pi * i / n
        q = _quat_mul(_quat_yaw(th), [math.cos(flare / 2), 0, math.sin(flare / 2), 0])
        rad = r_mid + (wall_h / 2) * math.sin(flare)
        z = base_z + (wall_h / 2) * math.cos(flare)
        body.add_geom(
            name=f"{prefix}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[wall_t / 2, chord / 2, wall_h / 2],
            pos=[rad * math.cos(th), rad * math.sin(th), z], quat=q,
            rgba=rgba, mass=mass_total / n, friction=friction, condim=4,
        )
    return r_mid + wall_h * math.sin(flare) + wall_t / 2


def _handle(body, oid, x_front, y, z, rgba):
    """A pull handle: an arm sticking out of the front (-x) and a vertical post hanging from it, which a gripper
    pointing down can pinch from above (fingers either side of the post)."""
    G = mujoco.mjtGeom
    reach, th, post_h = 0.05, 0.01, 0.036
    body.add_geom(name=f"{oid}__handle_arm", type=G.mjGEOM_BOX, size=[reach / 2, th / 2, th / 2],
                  pos=[x_front - reach / 2, y, z], rgba=rgba, mass=0.01, friction=[1.2, 0.005, 0.0001], condim=4)
    body.add_geom(name=f"{oid}__handle_post", type=G.mjGEOM_BOX, size=[th / 2, th / 2, post_h / 2],
                  pos=[x_front - reach + th / 2, y, z - post_h / 2 + th / 2], rgba=rgba, mass=0.01,
                  friction=[1.2, 0.005, 0.0001], condim=4)


def _add_articulated(body, oid, t, p, rgba):
    """A fixed cabinet shell (open at the front, which faces local -x) plus a moving child body: a drawer on a
    slide joint or a door on a hinge. Geoms of the moving part are named <id>__<part>_..., the handle <id>__handle_*,
    the visible front panel <id>__front, so perception can find parts and the monitor can read the joint."""
    G = mujoco.mjtGeom
    W, D, H, tk = p["width"], p["depth"], p["height"], 0.01
    shell = [rgba[0] * 0.85, rgba[1] * 0.85, rgba[2] * 0.85, 1]
    for name, pos_, size in [("bottom", [0, 0, tk / 2], [D / 2, W / 2, tk / 2]),
                             ("top", [0, 0, H - tk / 2], [D / 2, W / 2, tk / 2]),
                             ("left", [0, W / 2 - tk / 2, H / 2], [D / 2, tk / 2, H / 2]),
                             ("right", [0, -W / 2 + tk / 2, H / 2], [D / 2, tk / 2, H / 2]),
                             ("back", [D / 2 - tk / 2, 0, H / 2], [tk / 2, W / 2, H / 2])]:
        body.add_geom(name=f"{oid}_{name}", type=G.mjGEOM_BOX, pos=pos_, size=size, rgba=shell)
    inner_w, inner_h = W - 2 * tk, H - 2 * tk
    handle_rgba = [0.2, 0.2, 0.22, 1]
    if t == "drawer":
        # Drawer: front panel flush with the cabinet front + bottom, sides and back; slides out along local -x.
        dr = body.add_body(name=f"{oid}__drawer", pos=[-D / 2, 0, H / 2])
        dr.add_joint(name=f"{oid}__drawer", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                     range=[0, p["travel"]], limited=True, damping=p["damping"], armature=0.01)
        fw, fh = inner_w / 2 - 0.002, inner_h / 2 - 0.002
        dr.add_geom(name=f"{oid}__front", type=G.mjGEOM_BOX, pos=[0, 0, 0], size=[0.006, fw, fh], rgba=rgba, mass=0.08)
        L = D - 0.03
        for name, pos_, size in [("floor", [L / 2, 0, -fh + 0.003], [L / 2, fw - 0.004, 0.003]),
                                 ("side0", [L / 2, fw - 0.004, 0], [L / 2, 0.003, fh - 0.01]),
                                 ("side1", [L / 2, -fw + 0.004, 0], [L / 2, 0.003, fh - 0.01]),
                                 ("rear", [L, 0, 0], [0.003, fw - 0.004, fh - 0.01])]:
            dr.add_geom(name=f"{oid}__drawer_{name}", type=G.mjGEOM_BOX, pos=pos_, size=size, rgba=rgba, mass=0.05)
        _handle(dr, oid, -0.006, 0.0, fh * 0.35, handle_rgba)
        rng_ = [0.0, p["travel"]]
    else:
        # Door: a panel over the front opening, hinged on one side edge; opening swings the free edge toward -x.
        side = 1.0 if p["hinge_side"] >= 0.5 else -1.0
        dr = body.add_body(name=f"{oid}__door", pos=[-D / 2 - 0.007, side * W / 2, H / 2])
        dr.add_joint(name=f"{oid}__door", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, -side],
                     range=[0, 1.9], limited=True, damping=p["damping"], armature=0.002)
        dr.add_geom(name=f"{oid}__front", type=G.mjGEOM_BOX, pos=[0, -side * W / 2, 0],
                    size=[0.006, W / 2 - 0.002, H / 2 - 0.002], rgba=rgba, mass=0.06)
        _handle(dr, oid, -0.006, -side * (W - 0.03), H * 0.18, handle_rgba)
        rng_ = [0.0, 1.9]
        p["hinge_side"] = side
    a = ARTICULATED[t]
    return {"height": H, "width": max(W, D), "hollow": False, "bottom_offset": 0.0, "half_x": D / 2,
            "half_y": W / 2, "rim_z": H, "floor_z": tk,
            "articulated": {"part": a["part"], "joint": f"{oid}__{a['part']}", "type": a["type"], "unit": a["unit"],
                            "range": rng_, "body": f"{oid}__{a['part']}"}}


def add_object(spec: mujoco.MjSpec, obj: ObjectSpec, pos, yaw: float = 0.0) -> dict:
    """Add one object body to the spec at table position `pos` (x, y). Returns true-geometry meta."""
    t = obj.template
    p = resolve_params(t, obj.params)
    fixed = obj.fixed if obj.fixed is not None else (t in FIXED_BY_DEFAULT)
    rgba = _rgba(p)
    x, y = float(pos[0]), float(pos[1])
    body = spec.worldbody.add_body(name=obj.id, pos=[x, y, 0.0], quat=_quat_yaw(yaw))
    if not fixed:
        body.add_freejoint(name=f"{obj.id}_free")
    fr = OBJ_FRICTION(p.get("friction", 1.0))
    meta = {"id": obj.id, "template": t, "params": p, "fixed": fixed, "role": obj.role, "yaw": yaw,
            "color": p.get("color")}
    G = mujoco.mjtGeom
    if t == "block":
        s = p["size"]
        body.pos = [x, y, s / 2]
        body.add_geom(name=f"{obj.id}_g", type=G.mjGEOM_BOX, size=[s / 2] * 3, rgba=rgba, mass=p["mass"],
                      friction=fr, condim=4)
        meta.update(height=s, width=s, hollow=False, bottom_offset=-s / 2)
    elif t == "plate":
        # A wide flat disc: wider than the gripper opens, too thin to get the fingers under.
        r, th = p["radius"], p["thickness"]
        body.pos = [x, y, th / 2]
        body.add_geom(name=f"{obj.id}_g", type=G.mjGEOM_CYLINDER, size=[r, th / 2, 0], rgba=rgba,
                      mass=p["mass"], friction=fr, condim=4)
        meta.update(height=th, width=2 * r, hollow=False, bottom_offset=-th / 2)
    elif t == "bar":
        L_, W_, H_ = p["length"], p["width"], p["height"]
        body.pos = [x, y, H_ / 2]
        body.add_geom(name=f"{obj.id}_g", type=G.mjGEOM_BOX, size=[L_ / 2, W_ / 2, H_ / 2], rgba=rgba,
                      mass=p["mass"], friction=fr, condim=4)
        meta.update(height=H_, width=L_, hollow=False, bottom_offset=-H_ / 2, extent_major=L_, extent_minor=W_)
    elif t == "ball":
        r = p["radius"]
        body.pos = [x, y, r]
        body.add_geom(name=f"{obj.id}_g", type=G.mjGEOM_SPHERE, size=[r, 0, 0], rgba=rgba, mass=p["mass"],
                      friction=fr, condim=6)
        meta.update(height=2 * r, width=2 * r, hollow=False, bottom_offset=-r)
    elif t == "cylinder":
        r, h = p["radius"], p["height"]
        body.pos = [x, y, h / 2]
        body.add_geom(name=f"{obj.id}_g", type=G.mjGEOM_CYLINDER, size=[r, h / 2, 0], rgba=rgba,
                      mass=p["mass"], friction=fr, condim=4)
        meta.update(height=h, width=2 * r, hollow=False, bottom_offset=-h / 2)
    elif t in ("bowl", "cup", "mug"):
        if t == "bowl":
            r_base, wall_h, flare = p["radius"] * 0.6, p["depth"], math.radians(p["flare_deg"])
        else:
            r_base, wall_h, flare = p["radius"] - p["wall"], p["height"], math.radians(2)
        wt, base_t = p["wall"], 0.006
        body.add_geom(name=f"{obj.id}_base", type=G.mjGEOM_CYLINDER,
                      size=[r_base + wt, base_t / 2, 0], pos=[0, 0, base_t / 2],
                      rgba=rgba, mass=p["mass"] * 0.3, friction=fr, condim=4)
        rim_r = _ring(body, 20 if t == "bowl" else 16, r_base, base_t, wall_h, wt, flare, rgba,
                      p["mass"] * 0.7, fr, obj.id)
        rim_z = base_t + wall_h * math.cos(flare)
        meta.update(height=rim_z, width=2 * rim_r, hollow=True, rim_radius=rim_r, rim_z=rim_z,
                    wall=wt, inner_radius=r_base, floor_z=base_t, bottom_offset=0.0)
        if t == "mug":
            # Handle on the body's +x side: two arms out from the wall and an upright bar joining them.
            reach, th = p["handle"], 0.008
            z_lo, z_hi = base_t + 0.25 * wall_h, base_t + 0.8 * wall_h
            for i, z in enumerate((z_lo, z_hi)):
                body.add_geom(name=f"{obj.id}_harm{i}", type=G.mjGEOM_BOX, size=[reach / 2, th / 2, th / 2],
                              pos=[rim_r + reach / 2 - th / 2, 0, z], rgba=rgba, mass=0.01, friction=fr, condim=4)
            body.add_geom(name=f"{obj.id}_hbar", type=G.mjGEOM_BOX, size=[th / 2, th / 2, (z_hi - z_lo) / 2 + th / 2],
                          pos=[rim_r + reach - th / 2, 0, (z_lo + z_hi) / 2], rgba=rgba, mass=0.01, friction=fr,
                          condim=4)
            meta.update(handle=True, handle_reach=reach)
    elif t == "bin":
        w, d, h = p["width"], p["depth"], p["wall_height"]
        t_ = 0.008
        body.add_geom(name=f"{obj.id}_base", type=G.mjGEOM_BOX, size=[w / 2, d / 2, t_ / 2],
                      pos=[0, 0, t_ / 2], rgba=rgba)
        for i, (px, py, sx, sy) in enumerate([(w / 2, 0, t_ / 2, d / 2), (-w / 2, 0, t_ / 2, d / 2),
                                              (0, d / 2, w / 2, t_ / 2), (0, -d / 2, w / 2, t_ / 2)]):
            body.add_geom(name=f"{obj.id}_wall{i}", type=G.mjGEOM_BOX, size=[sx, sy, h / 2],
                          pos=[px, py, h / 2], rgba=rgba)
        meta.update(height=h, width=max(w, d), hollow=True, half_x=w / 2, half_y=d / 2, rim_z=h, floor_z=t_)
    elif t == "tray":
        w, d = p["width"], p["depth"]
        t_, lip = 0.01, 0.015
        body.add_geom(name=f"{obj.id}_base", type=G.mjGEOM_BOX, size=[w / 2, d / 2, t_ / 2],
                      pos=[0, 0, t_ / 2], rgba=rgba)
        for i, (px, py, sx, sy) in enumerate([(w / 2, 0, 0.004, d / 2), (-w / 2, 0, 0.004, d / 2),
                                              (0, d / 2, w / 2, 0.004), (0, -d / 2, w / 2, 0.004)]):
            body.add_geom(name=f"{obj.id}_lip{i}", type=G.mjGEOM_BOX, size=[sx, sy, lip / 2],
                          pos=[px, py, lip / 2], rgba=rgba)
        meta.update(height=lip, width=max(w, d), hollow=False, half_x=w / 2, half_y=d / 2, rim_z=lip, floor_z=t_)
    elif t in ARTICULATED:
        meta.update(_add_articulated(body, obj.id, t, p, rgba))
    elif t == "zone":
        w, d = p["width"], p["depth"]
        rgba[3] = 0.6
        body.add_geom(name=f"{obj.id}_g", type=G.mjGEOM_BOX, size=[w / 2, d / 2, 0.0005], pos=[0, 0, 0.0005],
                      rgba=rgba, contype=0, conaffinity=0)
        meta.update(height=0.001, width=max(w, d), hollow=False, half_x=w / 2, half_y=d / 2, rim_z=0.001,
                    floor_z=0.001)
    meta["pos"] = [x, y]
    return meta
