"""Feasibility before running: for each goal, which kind of skill can do it with this library, judged from
perception only (sizes vs the gripper and the skills' preconditions, reach, articulated parts).

    grasp    a grasp card's preconditions fit the object (the router picks among them)
    motion   no grasp is needed or none fits, but a verified motion card covers it (push onto a surface, open_<x>)
    fallback no grasp fits and no motion covers it: the planner still tries its default grasp (likely to fail; the
             failure is what the repair loop works from)
    missing  the goal needs a capability the library does not have (e.g. opening a drawer with no open_drawer card)
    unseen / unreachable  not perceived / outside the arm's reach

The verdicts are explanations shown before a run; with `act=True` the fixed plan also uses them (motion instead of
a fallback grasp, joint goals through their open_<template> card). Experiments keep act off, so their numbers and
the seed-library demo (bowl fails with top_grasp) are unchanged.
"""
from __future__ import annotations

import math

from .skills.library import OPEN_WIDTH
from .tasks.planner import fits, grasp_preference

SURFACES = {"tray", "zone"}            # targets an object can be slid onto
REACH_R = (0.30, 0.75)                 # m from the robot base; the placer keeps objects inside (0.33, 0.72)


def motion_cards(cards: dict, kind: str) -> list[str]:
    return sorted(n for n, c in cards.items() if c.get("kind") == kind and c.get("impl") == "generated")


def _why_no_grasp(desc: dict, cards: dict) -> str:
    w = desc.get("extent_minor", desc["width"])
    if w > OPEN_WIDTH:
        return f"{100 * w:.1f} cm across at its narrowest, the gripper opens {100 * OPEN_WIDTH:.0f} cm"
    if desc.get("flat") or desc["height"] < 0.012:
        return f"only {100 * desc['height']:.1f} cm tall, too thin to get the fingers around"
    reqs = {n: c.get("requires", {}) for n, c in cards.items() if c.get("kind") == "grasp"}
    return "no grasp skill's preconditions fit: " + "; ".join(f"{n} needs {r}" for n, r in reqs.items() if r)


def assess(task: dict, percepts: dict, cards: dict, allowed=None) -> list[dict]:
    """One verdict per goal: {obj, pred, verdict, skill(s), reason}."""
    templates = {o["id"]: o["template"] for o in task["objects"]}
    allowed = allowed or grasp_preference(cards)
    out = []
    for g in task["goal"]:
        obj, pred = g["obj"], g["pred"]
        v = {"obj": obj, "pred": pred, "target": g.get("target")}
        desc = percepts.get(obj)
        if desc is None:
            out.append({**v, "verdict": "unseen", "reason": "not perceived"})
            continue
        r = math.hypot(*desc["center"][:2])
        if not REACH_R[0] <= r <= REACH_R[1]:
            out.append({**v, "verdict": "unreachable", "reason": f"{100 * r:.0f} cm from the base, outside the arm's reach"})
            continue
        if pred in ("joint", "distance", "height", "upright"):
            kind = f"open_{templates[obj]}" if pred == "joint" and g.get("op", ">=") == ">=" else None
            have = motion_cards(cards, kind) if kind else []
            out.append({**v, "verdict": "motion" if have else "missing", "capability": kind, "skills": have,
                        "reason": (f"moves its part with {have[0]}" if have else
                                   f"needs a skill for '{kind or pred}', which the library does not have yet")})
            continue
        fitting = [n for n in allowed if fits(cards[n], desc)]
        if fitting:
            out.append({**v, "verdict": "grasp", "skills": fitting, "reason": f"{', '.join(fitting)} fit"})
            continue
        why = _why_no_grasp(desc, cards)
        push = motion_cards(cards, "push")
        if push and pred == "on" and templates.get(g.get("target")) in SURFACES:
            out.append({**v, "verdict": "motion", "capability": "push", "skills": push,
                        "reason": f"no grasp fits ({why}); slide it with {push[0]}"})
        else:
            out.append({**v, "verdict": "fallback", "skills": allowed[:1],
                        "reason": f"no grasp fits ({why}); will try {allowed[0]} anyway"})
    return out
