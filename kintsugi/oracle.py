"""Solvability oracle: a scripted expert with privileged state.

It runs the same scene (same task, seed and perturbation) with ground-truth object geometry instead of perception and
with the hand-written seed + reference skills (never cards the explorer discovered, so it stays independent of the
loop it judges). A robot failure counts only where the oracle succeeds: then the scene was solvable and the robot's
own skills fell short.
"""
from __future__ import annotations

import functools

from .skills.library import LIB_DIR, load_cards, seed_library


@functools.lru_cache(maxsize=1)
def oracle_library() -> dict:
    return {**seed_library(), **load_cards(LIB_DIR / "reference")}


def run_oracle(task: dict, seed: int, perturb=None) -> dict:
    """The scene counts as solvable if any attempt succeeds: first with the planner's default grasps, then with each
    other grasp skill preferred in turn (a fresh episode on the same scene each time). An item counts as solvable if
    any attempt placed it."""
    from .executor import run_episode
    from .tasks.planner import grasp_preference
    lib = oracle_library()
    prefs = grasp_preference(lib)
    # (allowed grasps, parameter overrides). The shallow top grasp is the expert's answer to tall objects, where a
    # grasp at half height drives the palm into the object's top.
    # The axis grasp (hand-written primitive defaults: middle of the long axis, closing across it) is its answer to
    # long objects, which a top grasp folded to +-45 deg cannot hold when they lie rotated.
    attempts = [(None, None), (None, {"top_grasp": {"depth_frac": 0.2}})] + \
        [([g] + [h for h in prefs if h != g], None) for g in prefs[1:]] + [(None, {"grasp_skill": "axis_pinch"})] +         [(None, {"top_grasp": {"clear_side": 1}})]      # keeps the hand away from obstacles ("don't knock it over")
    lib = {**lib, **load_cards(LIB_DIR / "primitives")}
    out = {"success": False, "subtask_ok": {}, "skills": {}, "labels": [], "attempts": 0, "wall_time_s": 0.0}
    # Last resort for tasks where order matters (stacking): the stand-in plan program, bases placed first.
    from .agents.jobs.plan_program import rule as rule_program
    attempts += [(None, {"program": rule_program({"task": task})["program"]})]
    for allowed, overrides in attempts:
        program = (overrides or {}).pop("program", None) if overrides and "program" in overrides else None
        try:
            rec, _ = run_episode(task, seed=seed, perception_mode="ground_truth", library=lib, perturb=perturb,
                                 allowed_grasps=allowed, overrides=overrides or None, performance={}, patches={},
                                 program=program or task.get("program"))
        except Exception as e:                   # an unbuildable scene is unsolvable, not a crash
            return {**out, "error": f"{type(e).__name__}: {e}"}
        out["attempts"] += 1
        out["wall_time_s"] = round(out["wall_time_s"] + rec["wall_time_s"], 2)
        for s in rec["subtasks"]:
            if "obj" in s and s["ok"] and not out["subtask_ok"].get(s["obj"]):
                out["skills"][s["obj"]] = s["trace"][0]["skill"] if s.get("trace") else None
            if "obj" in s:
                out["subtask_ok"][s["obj"]] = out["subtask_ok"].get(s["obj"], False) or s["ok"]
        out["labels"] = [s["label"] for s in rec["subtasks"] if s.get("label")]
        if rec["success"]:
            out.update(success=True, goals_met=rec["goals_met"], goals_total=rec["goals_total"])
            break
    return out


def valid_failure(result: dict, obj: str | None = None) -> bool:
    """True if a trial failed (whole task, or one item's step) where the oracle succeeded on the same scene."""
    o = result.get("oracle")
    if not o:
        return False
    if obj is None:
        return not result["success"] and o["success"]
    return not result["subtask_ok"].get(obj) and bool(o["subtask_ok"].get(obj))
