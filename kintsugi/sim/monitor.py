"""Privileged monitoring: true-state goal checks and failure labels.

Used by the evaluator and the failure labeler only. Skills never see this.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

FINGER_BODIES = ("left_finger", "right_finger")


class Monitor:
    def __init__(self, scene):
        self.scene = scene
        m = scene.model
        self.obj_body = {o: m.body(o).id for o in scene.objects}
        self.body_obj = {b: o for o, b in self.obj_body.items()}
        self.finger_ids = {m.body(n).id for n in FINGER_BODIES}
        robot_root = m.body("link0").id
        self.robot_bodies = {b for b in range(m.nbody) if self._is_desc(b, robot_root)}
        self.arm_bodies = self.robot_bodies - self.finger_ids
        self.collisions = []          # (t, robot_body, object)
        self.start_pos = {o: scene.data.xpos[b].copy() for o, b in self.obj_body.items()}
        self.held = None

    def _is_desc(self, b, root):
        m = self.scene.model
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return b == root

    def on_step(self, robot):
        d, m = self.scene.data, self.scene.model
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for rb, ob in ((b1, b2), (b2, b1)):
                if rb in self.arm_bodies and ob in self.body_obj:
                    if not self.collisions or self.collisions[-1][1:] != (int(rb), self.body_obj[ob]):
                        self.collisions.append((round(robot.t, 2), int(rb), self.body_obj[ob]))

    # ---------- true state ----------
    def pos(self, oid):
        return self.scene.data.xpos[self.obj_body[oid]].copy()

    def finger_contact(self, oid) -> bool:
        d, m = self.scene.data, self.scene.model
        b = self.obj_body[oid]
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
            if b in pair and pair & self.finger_ids:
                return True
        return False

    def upright(self, oid, tol_deg=25) -> bool:
        R = self.scene.data.xmat[self.obj_body[oid]].reshape(3, 3)
        return math.degrees(math.acos(np.clip(R[2, 2], -1, 1))) < tol_deg

    def inside(self, oid, target) -> bool:
        """True if object `oid` rests in/on `target` (bowl, cup, bin, tray, zone)."""
        tm = self.scene.objects[target]
        p = self.pos(oid)
        tp = self.pos(target)
        om = self.scene.objects[oid]
        bottom = p[2] + om.get("bottom_offset", 0)
        if tm["template"] in ("bowl", "cup", "mug"):
            r = math.hypot(p[0] - tp[0], p[1] - tp[1])
            # Inside the rim's footprint and resting in the bowl (or on another item in it).
            return r < tm["rim_radius"] and tp[2] - 0.01 < bottom < tp[2] + tm["rim_z"] + 0.5 * om["height"]
        R = self.scene.data.xmat[self.obj_body[target]].reshape(3, 3)
        local = R.T @ (p - tp)
        return (abs(local[0]) < tm["half_x"] and abs(local[1]) < tm["half_y"]
                and tp[2] - 0.01 < bottom < tp[2] + tm["rim_z"] + 0.03)

    def disturbed(self, exclude=(), tol=0.03):
        out = []
        for o, p0 in self.start_pos.items():
            if o in exclude:
                continue
            if np.linalg.norm(self.pos(o)[:2] - p0[:2]) > tol:
                out.append(o)
        return out

    def check_constraints(self, constraints) -> list[dict]:
        """Task constraints, e.g. "don't knock over the bottle": stays_upright, not_disturbed (moved < 3 cm)."""
        res = []
        for c in constraints:
            if c["pred"] == "stays_upright":
                ok = self.upright(c["obj"])
            elif c["pred"] == "not_disturbed":
                ok = np.linalg.norm(self.pos(c["obj"])[:2] - self.start_pos[c["obj"]][:2]) < 0.03 and self.upright(c["obj"])
            else:
                raise ValueError(c["pred"])
            res.append({**c, "ok": bool(ok)})
        return res

    def check_goal(self, goal) -> list[dict]:
        res = []
        for g in goal:
            if g["pred"] in ("inside", "on"):
                ok = self.inside(g["obj"], g["target"])
                if ok and g.get("upright"):
                    ok = self.upright(g["obj"])
            elif g["pred"] == "upright":
                ok = self.upright(g["obj"])
            else:
                raise ValueError(g["pred"])
            res.append({**g, "ok": bool(ok)})
        return res
