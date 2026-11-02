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
                 library=None, performance=None, positions=None, patches=None, recovery=0, live=None,
                 feasibility=False):
        """live: a LiveWriter streaming poses to the web app while this episode runs.
        recovery: retries per item after a failed step (re-perceive, next verified grasp); 0 in experiments,
        so searches and gates measure the skills themselves.
        feasibility: the fixed plan acts on feasibility verdicts (a verified motion where no grasp fits, joint goals
        through their opening skill); off in experiments, like recovery."""
        self.t0 = time.time()
        self.recovery = int(recovery)
        self.act_on_feasibility = bool(feasibility)
        self.task, self.seed, self.perception_mode = task, seed, perception_mode
        self.allowed_grasps, self.overrides = allowed_grasps, overrides or {}
        self.scene = build_scene(task, seed=seed, perturb=perturb, positions=positions)
        self.monitor = Monitor(self.scene)
        self.rec = (Recorder(self.scene, poses=record_poses, video=record_video, live=live)
                    if (record_poses or record_video or live is not None) else None)
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
        self.order = list(task.get("order") or [])    # required completion sequence (item ids), if any
        self.completed = []                            # items in the order their goal was first reached
        from .feasibility import assess
        self.feasibility = assess(task, self.percepts, self.cards, allowed_grasps)   # per goal, before any motion

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
        if res.check == "object_lost_in_transport":
            srec["label"] = "drop"                      # slipped out while carried (time in the trace info)
        return self._trace(srec, "place_in", p, ts, res)

    def push(self, srec, name, tdesc, params=None) -> bool:
        """Run a push card: model-written primitive code in the motion sandbox."""
        from .primitive_code import run_primitive
        ts = self.robot.t
        srec["_tdesc"] = tdesc
        card = self.cards[name]
        p = L.card_params(card, params)
        obj, goal = srec["obj"], self.goal_for(srec["obj"])
        audit, last = [], [self.monitor.moving_pos(obj)]

        def on_move(t0, t1):
            # Evaluation only (simulator state, never shown to the primitive): did this motion move the object,
            # and was the goal already met before it started?
            pos = self.monitor.moving_pos(obj)
            met_before = bool(audit and audit[-1]["goal_met"])
            met = bool(goal) and self.monitor.check_goal([goal])[0]["ok"]
            audit.append({"t0": round(t0, 2), "t1": round(t1, 2), "moved_cm": round(100 * float(np.linalg.norm(pos[:2] - last[0][:2])), 1),
                          "goal_met": met, "after_goal": met_before})
            last[0] = pos

        info = run_primitive(self.robot, card["code"], srec["_desc"], tdesc, p, on_move=on_move,
                             look=lambda: self.perc.perceive([obj])[obj])
        info["audit"] = motion_audit(audit)
        ok = info["error"] is None
        if not ok:
            srec["label"] = "primitive_error"
        return self._trace(srec, name, p, ts, L.SkillResult(name, ok, "primitive_ran" if ok else "primitive_error",
                                                            info))

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
        if goal_ok and obj not in self.completed:
            self.completed.append(obj)
            # Out of sequence: an item that must come earlier has not reached its goal yet.
            if obj in self.order:
                missing = [o for o in self.order[:self.order.index(obj)] if o not in self.completed]
                if missing:
                    srec["ok"] = False
                    srec["label"] = srec["label"] or "wrong_order"
                    srec["order_missing"] = missing
        # how far the object's parts moved, from perception (a drawer's handle comes out as the drawer opens)
        if srec["_desc"].get("parts") and after and after.get("parts"):
            srec["parts_moved_cm"] = {k: round(100 * float(np.linalg.norm(np.subtract(after["parts"][k]["center"][:2],
                                                                                     v["center"][:2]))), 1)
                                      for k, v in srec["_desc"]["parts"].items() if k in after["parts"]}
        if not srec["ok"] and srec["label"] is None and goal and goal.get("pred") not in ("on", "inside"):
            srec["label"] = "goal_not_met"                  # a measured goal (e.g. opened by N cm) not reached
        if not srec["ok"] and srec["label"] is None:
            # "drop" only for something that was lifted (a pushed plate is always low)
            lifted = any(t["skill"] == "lift" and t["ok"] for t in srec["trace"])
            srec["label"] = "drop" if lifted and self.monitor.pos(obj)[2] < 0.03 and not goal_ok else "wrong_place"
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
    # ------------------------------------------------------------------ recovery helpers
    RETRYABLE = {"missed_grasp", "weak_grasp", "collision", "slip", "lift_check_failed", "drop", "wrong_place"}

    def grasp_ranking(self, desc):
        """Grasp cards that fit this percept, best measured first (the router's order)."""
        from .skills.router import route
        from .tasks.planner import fits, grasp_preference
        allowed = self.allowed_grasps or grasp_preference(self.cards)
        fitting = [n for n in allowed if fits(self.cards[n], desc)]
        if not fitting:
            return allowed[:1]
        _, why = route(fitting, desc, self.performance)
        return sorted(fitting, key=lambda n: -why["scores"][n])

    def params_for(self, skill, desc):
        from .skills.router import patch_for, shape_group
        return {**patch_for(self.patches, skill, shape_group(desc)), **self.overrides.get(skill, {})}

    def should_retry(self, srec) -> bool:
        """Recovery decides from what the robot can observe, never from the simulator's verdict: retry only if
        perception does not see the item on its target, it is still perceivable, and it has not tipped over
        (a toppled bowl looks like a different, solid object and no skill rights objects yet)."""
        if srec.get("label") not in self.RETRYABLE or srec.get("place_check"):
            return False
        obs = srec.get("observed") or {}
        if obs.get("after") is None:
            return False
        return obs.get("height_ratio", 1.0) >= 0.7

    def retry_grasp(self, desc, tried):
        """Next grasp for a retry: the best-ranked one not tried yet, else the best one again (chance failures)."""
        ranked = self.grasp_ranking(desc)
        return next((g for g in ranked if g not in tried), ranked[0])

    def target_desc(self, target):
        """Containers keep their first percept (their contents occlude them); objects that can move (a stack base)
        are perceived again at placing time."""
        spec = next((o for o in self.task["objects"] if o["id"] == target), {})
        if spec.get("template") in ("block", "bar", "cylinder"):
            return self.perc.perceive([target])[target] or self.percepts[target]
        return self.percepts[target]

    # ------------------------------------------------------------------ fixed plan
    def run_motion(self, obj, target, card, routing=None) -> dict:
        """One subtask done by a motion card (push, open_<x>): perceive, run the primitive, check the goal."""
        srec = self.begin(obj, target, routing)
        if srec["_desc"] is None:
            return self.finish(srec, self.goal_for(obj))
        from .skills.router import patch_for, shape_group
        tdesc = self.perc.perceive([target])[target] if target else None
        prm = {**patch_for(self.patches, card, shape_group(srec["_desc"])), **self.overrides.get(card, {})}
        self.push(srec, card, tdesc, prm)
        return self.finish(srec, self.goal_for(obj))

    def run_plan(self):
        plan = make_plan(self.task, self.percepts, self.cards, self.allowed_grasps, self.overrides,
                         self.performance, self.patches, self.feasibility if self.act_on_feasibility else None)
        for st in plan:
            if st.get("error"):
                self.records.append({"obj": st["obj"], "target": st["target"], "trace": [], "label": st["error"],
                                     "ok": False, "routing": st.get("routing")})
                continue
            if st.get("motion"):
                self.run_motion(st["obj"], st["target"], st["motion"], st["routing"])
                continue
            steps, tried = st["steps"], []
            for attempt in range(1 + self.recovery):
                srec = self.begin(st["obj"], st["target"], st.get("routing") if attempt == 0 else None)
                if attempt:
                    # Recovery: the object was re-perceived (it may have fallen or tipped); next verified grasp.
                    if srec["_desc"] is None:
                        self.finish(srec, st["goal"])
                        break
                    g = self.retry_grasp(srec["_desc"], tried)
                    srec["routing"] = {"reason": f"recovery attempt {attempt}", "card": g}
                    steps = [{"skill": g, "params": self.params_for(g, srec["_desc"])},
                             {"skill": "lift", "params": self.params_for("lift", srec["_desc"])},
                             {**st["steps"][2], "params": self.params_for("place_in", srec["_desc"])}]
                srec["attempt"] = attempt
                if srec["_desc"] is not None:
                    for step in steps:
                        name = step["skill"]
                        if self.cards[name]["kind"] in ("grasp", "grasp_primitive"):
                            tried.append(name)
                            ok = self.grasp(srec, name, step.get("params"))
                        elif name == "lift":
                            ok = self.lift(srec, step.get("params"))
                        elif name == "place_in" and st.get("relation"):
                            from .tasks.planner import relation_spot
                            td = self.perc.perceive([st["target"]])[st["target"]]
                            ok = td is not None and self.place(srec, relation_spot(td, srec["_desc"], st["relation"]),
                                                               step.get("params"), "place", (0.0, 0.0))
                        elif name == "place_in":
                            ok = self.place(srec, self.target_desc(st["target"]), step.get("params"),
                                            step.get("mode", "place"), step.get("slot", (0.0, 0.0)))
                        else:
                            raise ValueError(name)
                        if not ok:
                            break
                self.finish(srec, st["goal"])
                if not self.should_retry(srec):
                    break

    # ------------------------------------------------------------------ result
    def record(self, extra=None) -> dict:
        if not (extra or {}).get("missing_capability"):
            # the fixed plan found a goal no library skill covers (feasibility): record it like a program's use() call
            miss = next((r for r in self.records if r.get("label") == "missing_capability"), None)
            if miss and (miss.get("routing") or {}).get("capability"):
                cap = miss["routing"]["capability"]
                extra = {**(extra or {}), "missing_capability": cap,
                         "missing_call": {"capability": cap, "obj": miss["obj"], "target": miss.get("target")}}
        final = self.monitor.check_goal(self.task["goal"])
        # A goal reached right after its step but lost by the end (a drawer sliding back, an object falling off a
        # tray later): the step did not really succeed. Label it, so repairs see the failure.
        for g in final:
            if not g["ok"]:
                st = next((r for r in reversed(self.records) if r.get("obj") == g.get("obj") and r.get("ok")), None)
                if st is not None:
                    st["ok"], st["label"] = False, "goal_lost"
                    if "measured" in g:
                        st["final_measured"] = g["measured"]
        final_constraints = self.monitor.check_constraints(self.constraints)
        done = [o for o in self.completed if o in self.order]
        order_ok = done == [o for o in self.order if o in done] and len(done) == len(self.order)
        rec = {
            "task": self.task["name"], "prompt": self.task.get("prompt"), "seed": self.seed,
            "perception": self.perception_mode, "split": self.task.get("split"),
            "card_versions": {n: self.cards[n]["version"] for n in sorted(self.used_cards)},
            "success": all(g["ok"] for g in final) and all(c["ok"] for c in final_constraints)
                       and (order_ok or not self.order),
            "completion_order": self.completed, **({"order_ok": order_ok} if self.order else {}),
            "constraints": final_constraints, "goals_met": sum(g["ok"] for g in final), "goals_total": len(final),
            "subtasks": self.records, "perception_calls": self.perc.calls, "sim_time_s": round(self.robot.t, 2),
            "wall_time_s": round(time.time() - self.t0, 2),
            "objects": {k: {kk: vv for kk, vv in v.items() if kk != "params"} | {"params": v["params"]}
                        for k, v in self.scene.objects.items()},
            "percepts": self.percepts, "task_spec": self.task, "recovery": self.recovery,
            "feasibility": self.feasibility, "act_on_feasibility": self.act_on_feasibility,
            "recovery_attempts": sum(1 for r in self.records if r.get("attempt")), **(extra or {}),
        }
        return rec

    def close(self):
        self.perc.close()
        if self.rec:
            self.rec.close()


def run_episode(task: dict, seed: int = 0, perception_mode: str = "sim", allowed_grasps=None,
                overrides=None, record_poses=False, record_video=False, perturb=None,
                extra_cards=None, library=None, performance=None, positions=None,
                patches=None, program=None, recovery=0, live=None, feasibility=False) -> tuple[dict, Recorder]:
    """`library` replaces the active skill library (a snapshot, e.g. the seed library); `extra_cards` adds
    cards that are not in it (exploration candidates, references); `positions` ({id: (x, y, yaw)}) rebuilds an
    exact earlier layout (one-change replays); `program` (or task["program"]) runs a model-written plan program
    instead of the planner's fixed plan."""
    ep = Episode(task, seed, perception_mode, allowed_grasps, overrides, record_poses, record_video, perturb,
                 extra_cards, library, performance, positions, patches, recovery, live, feasibility)
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


def motion_audit(moves: list[dict]) -> dict:
    """Wasted motion of a primitive, measured in simulation after the fact.
    idle: motions after the goal was already met that moved nothing (repeating a push through empty air); the
    final lift-away is excluded. overshoot: motions after the goal that still moved the object (it stays a goal
    only if it is still met at the end, which the success check covers)."""
    total = sum(m["t1"] - m["t0"] for m in moves) or 1e-6
    after = [m for m in moves if m["after_goal"]]
    if after and after[-1]["moved_cm"] < 0.5:
        after = after[:-1]                                   # the last lift-away
    idle = [m for m in after if m["moved_cm"] < 0.5]
    push = [m for m in after if m["moved_cm"] >= 0.5]
    wasted = sum(m["t1"] - m["t0"] for m in idle)
    return {"moves": len(moves), "total_s": round(total, 2), "wasted_s": round(wasted, 2),
            "wasted_frac": round(wasted / total, 3), "idle_moves_after_goal": len(idle),
            "overshoot_moves": len(push), "overshoot_cm": round(sum(m["moved_cm"] for m in push), 1), "per_move": moves}


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
