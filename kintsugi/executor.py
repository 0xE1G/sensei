"""Run one episode: build scene, perceive, plan, execute skills, check, label failures.

Two ways to drive an episode, sharing every skill call, check, label and piece of evidence:
  * a plan (list of subtasks of skill calls) from the planner, or
  * a plan PROGRAM written by the model (kintsugi.program), whose pick / place calls use the step methods below.
"""
from __future__ import annotations

import time

import numpy as np

from .perception.perceive import Perception
from .sim.monitor import Monitor
from .sim.recorder import Recorder
from .sim.robot import Robot
from .sim.scene import build_scene
from .skills import library as L
from .tasks.planner import make_plan


class Episode:
    """One scene with a robot, perception and the skill library, driven step by step."""

    def __init__(self, task: dict, seed: int = 0, perception_mode: str = "sim", allowed_grasps=None,
                 overrides=None, record_poses=False, record_video=False, perturb=None, extra_cards=None,
                 library=None, performance=None, positions=None, patches=None):
        self.t0 = time.time()
        self.task, self.seed, self.perception_mode = task, seed, perception_mode
        self.allowed_grasps, self.overrides = allowed_grasps, overrides or {}
        self.scene = build_scene(task, seed=seed, perturb=perturb, positions=positions)
        self.monitor = Monitor(self.scene)
        self.rec = Recorder(self.scene, poses=record_poses, video=record_video) if (record_poses or record_video) else None
        self.robot = Robot(self.scene, recorder=self.rec, monitor=self.monitor)
        self.robot.wait(0.3)                                   # let objects settle
        self.monitor.start_pos = {o: self.monitor.pos(o) for o in self.scene.objects}
        self.perc = Perception(self.scene, mode=perception_mode)
        self.cards = {**(library or L.load_cards()), **(extra_cards or {})}
        self.percepts = self.perc.perceive()
        if performance is None:
            from .skills.router import load_performance
            performance = load_performance()
        if patches is None:
            from .skills.router import load_patches
            patches = load_patches()
        self.performance, self.patches = performance, patches
        self.records = []
        self.constraints = task.get("constraints", [])
        self.violated = set()
        self.used_cards = set()

    # ------------------------------------------------------------------ perception
    def describe(self, obj, exclude=()):
        """Fresh percept of `obj` plus its neighbours (other perceived objects), for skills that keep clear."""
        desc = self.perc.perceive([obj])[obj]
        if desc is None:
            return None
        return {**desc, "neighbors": [{"id": o, "center": d["center"], "width": d["width"], "height": d["height"]}
                                      for o, d in self.percepts.items() if d and o != obj and o not in exclude]}

    def goal_for(self, obj):
        return next((g for g in self.task["goal"] if g.get("obj") == obj), None)

    # ------------------------------------------------------------------ step methods
    def begin(self, obj, target=None, routing=None, desc=None):
        """Start a subtask for `obj`. Returns its record (None-safe: label set if the object is not perceived)."""
        srec = {"obj": obj, "target": target, "trace": [], "label": None, "ok": False, "routing": routing}
        desc = desc if desc is not None else self.describe(obj, exclude=(target,) if target else ())
        srec["_desc"] = desc
        if desc is None:
            srec["label"] = "object_not_perceived"
            return srec
        srec["_z0"] = self.monitor.pos(obj)[2]
        srec["_before"] = {o: self.monitor.pos(o) for o in self.scene.objects}
        srec["_n_coll"] = len(self.monitor.collisions)
        srec["_grasp_offset"], srec["_width"], srec["_failed"] = 0.0, 0.0, False
        return srec

    def _trace(self, srec, name, p, ts, res):
        self.used_cards.add(name)
        srec["trace"].append({"skill": name, "params": p, "t0": round(ts, 2), "t1": round(self.robot.t, 2),
                              "ok": res.ok, "check": res.check,
                              "info": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.info.items()}})
        if self.rec:
            self.rec.event(self.robot.t, "skill_end", skill=name, ok=res.ok, obj=srec["obj"])
        if not res.ok:
            srec["_failed"] = True
        return res.ok

    def grasp(self, srec, name, params=None) -> bool:
        ts = self.robot.t
        p = L.card_params(self.cards[name], params)
        res = L.grasp_impl(self.cards[name])(self.robot, srec["_desc"], p)
        res.skill = name
        srec["_grasp_offset"] = res.info.get("grasp_offset", 0.0)
        srec["_width"] = res.info.get("finger_width", 0.0)
        if not res.ok:
            srec["label"] = _label_grasp(self.monitor, srec["obj"], res, srec["_n_coll"])
        return self._trace(srec, name, p, ts, res)

    def lift(self, srec, params=None) -> bool:
        ts = self.robot.t
        p = L.card_params(self.cards["lift"], params)
        res = L.lift(self.robot, p, srec["_width"])
        if not res.ok or self.monitor.pos(srec["obj"])[2] < srec["_z0"] + 0.05:
            srec["label"] = "slip" if self.monitor.pos(srec["obj"])[2] < srec["_z0"] + 0.05 else "lift_check_failed"
            res.ok = False
        return self._trace(srec, "lift", p, ts, res)

    def place(self, srec, tdesc, params=None, mode="place", slot=(0.0, 0.0)) -> bool:
        ts = self.robot.t
        srec["_tdesc"] = tdesc
        p = L.card_params(self.cards["place_in"], params)
        res = L.place_in(self.robot, tdesc, p, srec["_grasp_offset"], mode=mode, slot=slot)
        return self._trace(srec, "place_in", p, ts, res)

    def finish(self, srec, goal=None):
        """Settle, check the goal (if any) and the constraints, collect evidence, go back to the ready pose."""
        if srec.get("_desc") is None:
            self.records.append(_public(srec))
            return srec
        obj, target = srec["obj"], srec.get("target")
        tdesc = srec.get("_tdesc") or (self.percepts.get(target) if target else None)
        self.robot.wait(0.6)
        goal_ok = self.monitor.check_goal([goal])[0]["ok"] if goal else True
        moved = [o for o in self.scene.objects if o not in (obj, target)
                 and np.linalg.norm(self.monitor.pos(o)[:2] - srec["_before"][o][:2]) > 0.03]
        # Observable place check: perceive the item once after release.
        after = self.perc.perceive([obj])[obj]
        srec["place_check"] = _observed_inside(after, tdesc) if (after and tdesc) else False
        srec["ok"] = bool(goal_ok and not srec["_failed"])
        if not srec["ok"] and srec["label"] is None:
            srec["label"] = "drop" if self.monitor.pos(obj)[2] < 0.03 and not goal_ok else "wrong_place"
        # Constraint violations are pinned to the sub-task during which they first happened.
        new = [c for c in self.monitor.check_constraints(self.constraints)
               if not c["ok"] and (c["pred"], c["obj"]) not in self.violated]
        if new:
            self.violated |= {(c["pred"], c["obj"]) for c in new}
            srec["constraint_violations"] = [f"{c['pred']}({c['obj']})" for c in new]
            srec["ok"] = False
            srec["label"] = srec["label"] or "constraint_violation"
        n_coll = srec["_n_coll"]
        srec["disturbed"] = moved
        srec["arm_collisions"] = [c[2] for c in self.monitor.collisions[n_coll:]][:10]
        # Observable evidence for the investigator: what perception saw before and after, and arm contacts.
        srec["observed"] = _observed(srec["_desc"], after, tdesc, srec["place_check"]) if tdesc else None
        srec["contacts"] = [{"t": c[0], "part": self.scene.model.body(c[1]).name, "object": c[2]}
                            for c in self.monitor.collisions[n_coll:]][:10]
        self.records.append(_public(srec))
        L.go_ready(self.robot)
        return srec

    # ------------------------------------------------------------------ fixed plan
    def run_plan(self):
        plan = make_plan(self.task, self.percepts, self.cards, self.allowed_grasps, self.overrides,
                         self.performance, self.patches)
        for st in plan:
            if st.get("error"):
                self.records.append({"obj": st["obj"], "target": st["target"], "trace": [], "label": st["error"],
                                     "ok": False, "routing": st.get("routing")})
                continue
            srec = self.begin(st["obj"], st["target"], st.get("routing"))
            if srec["_desc"] is not None:
                for step in st["steps"]:
                    name = step["skill"]
                    if self.cards[name]["kind"] in ("grasp", "grasp_primitive"):
                        ok = self.grasp(srec, name, step.get("params"))
                    elif name == "lift":
                        ok = self.lift(srec, step.get("params"))
                    elif name == "place_in":
                        ok = self.place(srec, self.percepts[st["target"]], step.get("params"),
                                        step.get("mode", "place"), step.get("slot", (0.0, 0.0)))
                    else:
                        raise ValueError(name)
                    if not ok:
                        break
            self.finish(srec, st["goal"])

    # ------------------------------------------------------------------ result
    def record(self, extra=None) -> dict:
        final = self.monitor.check_goal(self.task["goal"])
        final_constraints = self.monitor.check_constraints(self.constraints)
        rec = {
            "task": self.task["name"], "prompt": self.task.get("prompt"), "seed": self.seed,
            "perception": self.perception_mode, "split": self.task.get("split"),
            "card_versions": {n: self.cards[n]["version"] for n in sorted(self.used_cards)},
            "success": all(g["ok"] for g in final) and all(c["ok"] for c in final_constraints),
            "constraints": final_constraints, "goals_met": sum(g["ok"] for g in final), "goals_total": len(final),
            "subtasks": self.records, "perception_calls": self.perc.calls, "sim_time_s": round(self.robot.t, 2),
            "wall_time_s": round(time.time() - self.t0, 2),
            "objects": {k: {kk: vv for kk, vv in v.items() if kk != "params"} | {"params": v["params"]}
                        for k, v in self.scene.objects.items()},
            "percepts": self.percepts, "task_spec": self.task, **(extra or {}),
        }
        return rec

    def close(self):
        self.perc.close()
        if self.rec:
            self.rec.close()


def run_episode(task: dict, seed: int = 0, perception_mode: str = "sim", allowed_grasps=None,
                overrides=None, record_poses=False, record_video=False, perturb=None,
                extra_cards=None, library=None, performance=None, positions=None,
                patches=None, program=None) -> tuple[dict, Recorder]:
    """`library` replaces the active skill library (a snapshot, e.g. the seed library); `extra_cards` adds
    cards that are not in it (exploration candidates, references); `positions` ({id: (x, y, yaw)}) rebuilds an
    exact earlier layout (one-change replays); `program` (or task["program"]) runs a model-written plan program
    instead of the planner's fixed plan."""
    ep = Episode(task, seed, perception_mode, allowed_grasps, overrides, record_poses, record_video, perturb,
                 extra_cards, library, performance, positions, patches)
    program = program or task.get("program")
    extra = None
    if program:
        from .program import run_program
        extra = {**run_program(ep, program), "program_source": task.get("program_source")}
    else:
        ep.run_plan()
    record = ep.record(extra)
    ep.close()
    return record, ep.rec


def _public(srec):
    return {k: v for k, v in srec.items() if not k.startswith("_")}


def _label_grasp(monitor, obj, res, n_coll):
    if not res.info.get("reached", True):
        # Short of the pose while touching the object or anything else: blocked, not out of reach.
        touching = monitor.finger_contact(obj) or bool(monitor.collisions[n_coll:])
        return "collision" if touching else "unreachable"
    if res.check.startswith("precondition"):
        return "precondition_failed"
    if any(c[2] == obj for c in monitor.collisions[n_coll:]):
        return "collision"
    if not monitor.finger_contact(obj):
        return "missed_grasp"
    return "weak_grasp"


def _observed(before, after, target, on_target):
    """Perception-only summary of what happened to the item (no simulator state)."""
    keep = lambda d: {k: (round(d[k], 4) if isinstance(d[k], float) else d[k]) for k in
                      ("center", "top_z", "height", "width", "hollow", "elongation") if k in d} if d else None
    out = {"before": keep(before), "after": keep(after), "on_target_observed": bool(on_target)}
    if before and after:
        out["moved_xy"] = round(float(np.linalg.norm(np.subtract(after["center"], before["center"]))), 4)
        out["height_ratio"] = round(after["height"] / max(before["height"], 1e-3), 3)    # < 0.7: tipped over
        out["dist_to_target_xy"] = round(float(np.linalg.norm(np.subtract(after["center"], target["center"]))), 4)
    return out


def _observed_inside(item, target):
    c = np.array(item["center"])
    tc = np.array(target["center"])
    if "rim_radius" in target:
        return bool(np.linalg.norm(c - tc) < target["rim_radius"])
    ex = np.array(target.get("extent_xy", [target["width"]] * 2)) / 2
    return bool(np.all(np.abs(c - tc) < ex))
