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
        # moving parts (drawer, door) are child bodies: map them to their object too
        for b in range(m.nbody):
            p = b
            while p > 0 and p not in self.body_obj:
                p = m.body_parentid[p]
            if p > 0 and b not in self.body_obj:
                self.body_obj[b] = self.body_obj[p]
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
        bodies = {b for b, o in self.body_obj.items() if o == oid}
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
            if pair & bodies and pair & self.finger_ids:
                return True
        return False

    def moving_pos(self, oid):
        """Position of the part that moves: the moving child of an articulated object, else the object itself."""
        art = self.scene.objects[oid].get("articulated")
        if art:
            return self.scene.data.xpos[self.scene.model.body(art["body"]).id].copy()
        return self.pos(oid)

    def joint_value(self, oid, part=None) -> float:
        """Opening of an articulated object's moving part: metres for a slide, degrees for a hinge."""
        art = self.scene.objects[oid].get("articulated")
        if not art or (part and part != art["part"]):
            raise ValueError(f"{oid} has no moving part {part or ''}".strip())
        m = self.scene.model
        q = float(self.scene.data.qpos[m.jnt_qposadr[m.joint(art["joint"]).id]])
        return math.degrees(q) if art["type"] == "hinge" else q

    def upright(self, oid, tol_deg=25) -> bool:
        """Standing on its base, within tol_deg. An object whose ends are alike (a plain cylinder) also counts when it
        stands on its other end: upside down it is the same shape."""
        R = self.scene.data.xmat[self.obj_body[oid]].reshape(3, 3)
        c = abs(R[2, 2]) if self.scene.objects[oid].get("symmetric_ends") else R[2, 2]
        return math.degrees(math.acos(np.clip(c, -1, 1))) < tol_deg

    def inside(self, oid, target) -> bool:
        """True if object `oid` rests in/on `target` (bowl, cup, mug, bin, tray, zone, or on top of a block,
        bar or cylinder: stacking)."""
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
        if tm["template"] in ("block", "bar", "cylinder"):
            # Stacked: centred over the target's top face and resting on it.
            top_t = tp[2] + tm.get("bottom_offset", 0) + tm["height"]
            half = (tm.get("extent_major", tm["width"]) / 2, tm.get("extent_minor", tm["width"]) / 2)
            return abs(local[0]) < half[0] and abs(local[1]) < half[1] and abs(bottom - top_t) < 0.015
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

    # direction from the target for each relation, seen from the robot (+x away from it, +y to its left)
    RELATIONS = {"left_of": (0.0, 1.0), "right_of": (0.0, -1.0), "in_front_of": (-1.0, 0.0), "behind": (1.0, 0.0)}

    def relation(self, obj, target, pred) -> tuple[bool, float]:
        """obj beside target on that side: 4-30 cm further that way (centre to centre), within 10 cm sideways."""
        u = np.array(self.RELATIONS[pred])
        d = self.pos(obj)[:2] - self.pos(target)[:2]
        along = float(d @ u)
        perp = float(np.linalg.norm(d - along * u))
        return bool(0.04 <= along <= 0.30 and perp <= 0.10), along

    def check_goal(self, goal) -> list[dict]:
        """Goals are deterministic checks on simulator state. on / inside / upright as before, plus composable
        measurable terms, each compared with op (">=" or "<=") against value:
          joint     opening of an object's moving part (m for a drawer slide, degrees for a door hinge)
          distance  horizontal distance between obj and target (m)
          height    height of obj's bottom above the table (m)"""
        res = []
        for g in goal:
            measured = None
            if g["pred"] in ("inside", "on"):
                ok = self.inside(g["obj"], g["target"])
                if ok and g.get("upright"):
                    ok = self.upright(g["obj"])
            elif g["pred"] == "upright":
                ok = self.upright(g["obj"])
            elif g["pred"] in self.RELATIONS:
                ok, measured = self.relation(g["obj"], g["target"], g["pred"])
            elif g["pred"] in ("joint", "distance", "height"):
                if g["pred"] == "joint":
                    measured = self.joint_value(g["obj"], g.get("part"))
                elif g["pred"] == "distance":
                    measured = float(np.linalg.norm(self.pos(g["obj"])[:2] - self.pos(g["target"])[:2]))
                else:
                    measured = float(self.pos(g["obj"])[2] + self.scene.objects[g["obj"]].get("bottom_offset", 0))
                ok = measured >= g["value"] if g.get("op", ">=") == ">=" else measured <= g["value"]
            else:
                raise ValueError(g["pred"])
            res.append({**g, "ok": bool(ok), **({"measured": round(measured, 4)} if measured is not None else {})})
        return res
