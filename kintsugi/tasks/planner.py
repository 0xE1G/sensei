"""Plan: turn a task spec + perception into ordered subtasks of skill calls.

Rule-based stand-in for Muse Spark. The output format (a list of subtasks,
each a list of skill calls with parameters) is what Spark will produce later.
"""
from __future__ import annotations

import math

SEED_PREFERENCE = ["top_grasp", "side_wrap"]


def grasp_preference(cards: dict) -> list[str]:
    """Seed grasps first, then promoted variants in the order they were added. Primitives are never planned."""
    grasps = [n for n, c in cards.items() if c.get("kind") == "grasp"]
    return [n for n in SEED_PREFERENCE if n in grasps] + sorted(n for n in grasps if n not in SEED_PREFERENCE)


def fits(card: dict, desc: dict) -> bool:
    """Does the perceived shape meet the card's preconditions?"""
    req = card.get("requires", {})
    if "hollow" in req and bool(desc.get("hollow")) != req["hollow"]:
        return False
    if req.get("rim") and "rim_radius" not in desc:
        return False
    if "max_width" in req and desc["width"] > req["max_width"]:
        return False
    if "min_width" in req and desc["width"] < req["min_width"]:
        return False
    if "min_height" in req and desc["height"] < req["min_height"]:
        return False
    if "max_height" in req and desc["height"] > req["max_height"]:
        return False
    if "min_elongation" in req and desc.get("elongation", 1.0) < req["min_elongation"]:
        return False
    if "max_elongation" in req and desc.get("elongation", 1.0) > req["max_elongation"]:
        return False
    return True


SIZE_SLACK = 0.25        # nearest-verified fallback: a size bound may be exceeded by at most 25 % of its value
MIN_EVIDENCE = 8         # ... and only for a card with at least this many measured trials on the object's shape group


def near_fit(card: dict, desc: dict, slack: float = SIZE_SLACK) -> bool:
    """fits(), except that width / height bounds may be exceeded by `slack` of the bound; the kind of object (hollow,
    rim, elongation) must match exactly."""
    req = card.get("requires", {})
    if "hollow" in req and bool(desc.get("hollow")) != req["hollow"]:
        return False
    if req.get("rim") and "rim_radius" not in desc:
        return False
    if "min_elongation" in req and desc.get("elongation", 1.0) < req["min_elongation"]:
        return False
    if "max_elongation" in req and desc.get("elongation", 1.0) > req["max_elongation"]:
        return False
    for key, v, upper in (("max_width", desc["width"], True), ("min_width", desc["width"], False),
                          ("max_height", desc["height"], True), ("min_height", desc["height"], False)):
        if key in req and (v > req[key] * (1 + slack) if upper else v < req[key] * (1 - slack)):
            return False
    return True


def choose_grasp(desc: dict, allowed: list[str], cards: dict, performance=None) -> tuple[str, dict]:
    """Among allowed grasps whose preconditions fit, the router picks by measured success on the object's shape
    group (preference order breaks ties and decides where there is no evidence).
    Nothing fits: the grasp verified on this KIND of object (measured on the same shape group, >= MIN_EVIDENCE trials,
    better than the default there) whose size range is only slightly exceeded (near_fit), else the default grasp.
    A held-out bowl a little larger than every bowl a card was tuned on should get the bowl grasp, not a top grasp
    known to fail on bowls; a block never gets a cylinder grasp (no evidence on the block's shape group)."""
    from ..skills.router import route, score, shape_group
    fitting = [n for n in allowed if fits(cards[n], desc)]
    if not fitting:
        perf = performance or {}
        group = shape_group(desc)
        base = score(perf, allowed[0], group)
        near = [n for n in allowed if near_fit(cards[n], desc)
                and (perf.get(n, {}).get(group) or {}).get("trials", 0) >= MIN_EVIDENCE and score(perf, n, group) > base]
        if near:
            best = max(near, key=lambda n: score(perf, n, group))
            return best, {"reason": f"no card fits; nearest verified for {group} (size just outside its range)",
                          "card": best, "group": group, "score": round(score(perf, best, group), 3)}
        # Nothing fits: fall back to the first allowed skill (it may well fail -- that is a finding).
        return allowed[0], {"reason": "no card fits; fallback"}
    return route(fitting, desc, performance or {})


RELATIONS = {"left_of": (0.0, 1.0), "right_of": (0.0, -1.0), "in_front_of": (-1.0, 0.0), "behind": (1.0, 0.0)}
SIDES = {"left": "left_of", "right": "right_of", "front": "in_front_of", "in_front": "in_front_of", "behind": "behind",
         "back": "behind"}


def relation_spot(target_desc: dict, obj_desc: dict | None, pred: str, gap: float = 0.04) -> dict:
    """A place-target descriptor on the table beside `target_desc` on that side (from perception only): the target's
    half extent + the object's half width + a gap away from its centre."""
    u = RELATIONS[pred]
    ex = target_desc.get("extent_xy") or [target_desc["width"]] * 2
    half_t = abs(u[0]) * ex[0] / 2 + abs(u[1]) * ex[1] / 2
    half_o = (obj_desc or {}).get("width", 0.05) / 2
    d = max(0.06, half_t + half_o + gap)
    c = target_desc["center"]
    return {"center": [c[0] + u[0] * d, c[1] + u[1] * d], "top_z": 0.0, "width": 0.06, "extent_xy": [0.06, 0.06],
            "spot_for": pred}


def make_plan(task: dict, percepts: dict, cards: dict, allowed_grasps=None, overrides=None,
              performance=None, patches=None, feasibility=None) -> list[dict]:
    """patches: parameter patches per (skill, shape group of the item); explicit overrides win over patches.
    feasibility: verdicts from feasibility.assess to act on (motion cards where no grasp fits, joint goals through
    their open_<template> card, missing capabilities reported up front); None keeps the plain grasp plan."""
    from ..skills.router import patch_for, pick_motion, shape_group
    allowed = allowed_grasps or grasp_preference(cards)
    overrides = overrides or {}
    templates = {o["id"]: o["template"] for o in task["objects"]}
    subtasks = []
    verdicts = {v["obj"]: v for v in feasibility or []}

    def motion_step(g, v):
        desc = percepts.get(g["obj"])
        card = pick_motion(cards, v["skills"], performance or {}, shape_group(desc)) if desc else v["skills"][0]
        return {"obj": g["obj"], "target": g.get("target"), "goal": g, "motion": card,
                "routing": {"reason": f"feasibility: {v['reason']}", "card": card}}
    # Moving parts first (opening a drawer does not depend on anything else being placed).
    for g in task["goal"]:
        v = verdicts.get(g["obj"])
        if g["pred"] in ("inside", "on") or not v:
            continue
        if v["verdict"] == "motion":
            subtasks.append(motion_step(g, v))
        elif v["verdict"] == "missing":
            subtasks.append({"obj": g["obj"], "target": None, "error": "missing_capability", "steps": [],
                             "routing": {"reason": f"feasibility: {v['reason']}", "capability": v.get("capability")}})
    goals = [g for g in task["goal"] if g["pred"] in ("inside", "on") or g["pred"] in RELATIONS]
    # Far objects first, so the arm does not sweep over things it already placed.
    goals.sort(key=lambda g: -math.hypot(*percepts[g["obj"]]["center"]) if percepts.get(g["obj"]) else 0)
    # A required sequence (task "order") overrides the distance order.
    if task.get("order"):
        seq = task["order"]
        goals.sort(key=lambda g: seq.index(g["obj"]) if g["obj"] in seq else len(seq))
    # A target that is itself an item to be moved (a stack base) must be placed before what goes on it.
    items, ordered = {g["obj"] for g in goals}, []
    while goals:
        ready = [g for g in goals if g["target"] not in items or g["target"] in {o["obj"] for o in ordered}]
        nxt = ready[0] if ready else goals[0]
        ordered.append(nxt)
        goals.remove(nxt)
    goals = ordered
    slots = _container_slots(goals, percepts)
    for g in goals:
        desc = percepts.get(g["obj"])
        if desc is None:
            subtasks.append({"obj": g["obj"], "target": g["target"], "error": "object_not_perceived", "steps": []})
            continue
        # "grasp_for" forces a grasp for one object only (the explorer trying a candidate on the failing item).
        forced = overrides.get("grasp_for", {}).get(g["obj"]) or overrides.get("grasp_skill")
        v = verdicts.get(g["obj"])
        if not forced and v and v["verdict"] == "motion":
            subtasks.append(motion_step(g, v))
            continue
        grasp, why = (forced, {"reason": "forced"}) if forced else choose_grasp(desc, allowed, cards, performance)
        mode = "drop" if templates[g["target"]] == "bin" else "place"
        group = shape_group(desc)
        prm = lambda s: {**patch_for(patches, s, group), **overrides.get(s, {})}
        applied = {s: patch_for(patches, s, group) for s in (grasp, "lift", "place_in") if patch_for(patches, s, group)}
        if applied:
            why = {**why, "patches": applied}
        subtasks.append({
            "obj": g["obj"], "target": g["target"], "goal": g, "routing": why,
            **({"relation": g["pred"]} if g["pred"] in RELATIONS else {}),
            "steps": [{"skill": grasp, "params": prm(grasp)},
                      {"skill": "lift", "params": prm("lift")},
                      {"skill": "place_in", "mode": mode, "slot": slots[g["obj"]], "params": prm("place_in")}],
        })
    return subtasks


def _container_slots(goals, percepts):
    """Give each item its own spot in its container, so the second block does not land on the first."""
    by_target = {}
    for g in goals:
        by_target.setdefault(g["target"], []).append(g["obj"])
    slots = {}
    for target, items in by_target.items():
        t = percepts.get(target) or {}
        if len(items) == 1:
            slots[items[0]] = [0.0, 0.0]
            continue
        if "rim_radius" in t:
            rad = 0.32 * t["rim_radius"]
        else:
            ex = t.get("extent_xy", [t.get("width", 0.1)] * 2)
            rad = 0.25 * min(ex)
        for i, o in enumerate(items):
            a = 2 * math.pi * i / len(items) + math.pi / 2
            slots[o] = [rad * math.cos(a), rad * math.sin(a)]
    return slots
