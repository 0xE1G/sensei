"""Run one episode: build scene, perceive, plan, execute skills, check, label failures."""
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


def run_episode(task: dict, seed: int = 0, perception_mode: str = "sim", allowed_grasps=None,
                overrides=None, record_poses=False, record_video=False, perturb=None,
                extra_cards=None, library=None, performance=None, positions=None,
                patches=None) -> tuple[dict, Recorder]:
    """`library` replaces the active skill library (a snapshot, e.g. the seed library); `extra_cards` adds
    cards that are not in it (exploration candidates, references); `positions` ({id: (x, y, yaw)}) rebuilds an
    exact earlier layout (one-change replays)."""
    t0 = time.time()
    scene = build_scene(task, seed=seed, perturb=perturb, positions=positions)
    monitor = Monitor(scene)
    rec = Recorder(scene, poses=record_poses, video=record_video) if (record_poses or record_video) else None
    robot = Robot(scene, recorder=rec, monitor=monitor)
    robot.wait(0.3)                                   # let objects settle
    monitor.start_pos = {o: monitor.pos(o) for o in scene.objects}
    perc = Perception(scene, mode=perception_mode)
    cards = {**(library or L.load_cards()), **(extra_cards or {})}
    percepts = perc.perceive()
    if performance is None:
        from .skills.router import load_performance
        performance = load_performance()
    if patches is None:
        from .skills.router import load_patches
        patches = load_patches()
    plan = make_plan(task, percepts, cards, allowed_grasps, overrides, performance, patches)

    subtask_records = []
    constraints = task.get("constraints", [])
    violated = set()
    for st in plan:
        srec = {"obj": st["obj"], "target": st["target"], "trace": [], "label": None, "ok": False,
                "routing": st.get("routing")}
        if st.get("error"):
            srec["label"] = st["error"]
            subtask_records.append(srec)
            continue
        desc = perc.perceive([st["obj"]])[st["obj"]]     # refresh: earlier placements may have moved things
        tdesc = percepts[st["target"]]
        if desc is not None:                             # other perceived objects, for skills that keep clear of them
            desc = {**desc, "neighbors": [{"id": o, "center": d["center"], "width": d["width"], "height": d["height"]}
                                          for o, d in percepts.items() if d and o not in (st["obj"], st["target"])]}
        if desc is None:
            srec["label"] = "object_not_perceived"
            subtask_records.append(srec)
            continue
        z0 = monitor.pos(st["obj"])[2]
        before = {o: monitor.pos(o) for o in scene.objects}
        n_coll = len(monitor.collisions)
        grasp_offset, width_at_grasp, failed = 0.0, 0.0, False
        for step in st["steps"]:
            name = step["skill"]
            ts = robot.t
            p = L.card_params(cards[name], step.get("params"))
            if cards[name]["kind"] in ("grasp", "grasp_primitive"):
                res = L.grasp_impl(cards[name])(robot, desc, p)
                res.skill = name
                grasp_offset = res.info.get("grasp_offset", 0.0)
                width_at_grasp = res.info.get("finger_width", 0.0)
                if not res.ok:
                    srec["label"] = _label_grasp(monitor, st["obj"], res, n_coll)
            elif name == "lift":
                res = L.lift(robot, p, width_at_grasp)
                if not res.ok or monitor.pos(st["obj"])[2] < z0 + 0.05:
                    srec["label"] = "slip" if monitor.pos(st["obj"])[2] < z0 + 0.05 else "lift_check_failed"
                    res.ok = False
            elif name == "place_in":
                res = L.place_in(robot, tdesc, p, grasp_offset, mode=step.get("mode", "place"),
                                 slot=step.get("slot", (0.0, 0.0)))
            else:
                raise ValueError(name)
            srec["trace"].append({"skill": name, "params": p, "t0": round(ts, 2), "t1": round(robot.t, 2),
                                  "ok": res.ok, "check": res.check,
                                  "info": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.info.items()}})
            if rec:
                rec.event(robot.t, "skill_end", skill=name, ok=res.ok, obj=st["obj"])
            if not res.ok:
                failed = True
                break
        robot.wait(0.6)
        goal_ok = monitor.check_goal([st["goal"]])[0]["ok"]
        moved = [o for o in scene.objects if o not in (st["obj"], st["target"])
                 and np.linalg.norm(monitor.pos(o)[:2] - before[o][:2]) > 0.03]
        # Observable place check: perceive the item once after release.
        after = perc.perceive([st["obj"]])[st["obj"]]
        srec["place_check"] = _observed_inside(after, tdesc) if after else False
        srec["ok"] = bool(goal_ok and not failed)
        if not srec["ok"] and srec["label"] is None:
            srec["label"] = "drop" if monitor.pos(st["obj"])[2] < 0.03 and not goal_ok else "wrong_place"
        # Constraint violations are pinned to the sub-task during which they first happened.
        new = [c for c in monitor.check_constraints(constraints)
               if not c["ok"] and (c["pred"], c["obj"]) not in violated]
        if new:
            violated |= {(c["pred"], c["obj"]) for c in new}
            srec["constraint_violations"] = [f"{c['pred']}({c['obj']})" for c in new]
            srec["ok"] = False
            srec["label"] = srec["label"] or "constraint_violation"
        srec["disturbed"] = moved
        srec["arm_collisions"] = [c[2] for c in monitor.collisions[n_coll:]][:10]
        # Observable evidence for the investigator: what perception saw before and after, and arm contacts.
        srec["observed"] = _observed(desc, after, tdesc, srec["place_check"])
        srec["contacts"] = [{"t": c[0], "part": scene.model.body(c[1]).name, "object": c[2]}
                            for c in monitor.collisions[n_coll:]][:10]
        subtask_records.append(srec)
        L.go_ready(robot)

    final = monitor.check_goal(task["goal"])
    final_constraints = monitor.check_constraints(constraints)
    record = {
        "task": task["name"], "prompt": task.get("prompt"), "seed": seed, "perception": perception_mode,
        "split": task.get("split"),
        "card_versions": {s["skill"]: cards[s["skill"]]["version"] for st in plan for s in st["steps"]},
        "success": all(g["ok"] for g in final) and all(c["ok"] for c in final_constraints),
        "constraints": final_constraints, "goals_met": sum(g["ok"] for g in final), "goals_total": len(final),
        "subtasks": subtask_records, "perception_calls": perc.calls, "sim_time_s": round(robot.t, 2),
        "wall_time_s": round(time.time() - t0, 2), "objects": {k: {kk: vv for kk, vv in v.items() if kk != "params"}
                                                             | {"params": v["params"]} for k, v in scene.objects.items()},
        "percepts": percepts, "task_spec": task,
    }
    perc.close()
    if rec:
        rec.close()
    return record, rec


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
