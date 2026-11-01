"""Composite cards: plan programs that earned trust, stored and reused like skills.

A program written by the model (or the stand-in) is safety-checked and every run is scored, but it is only TRUSTED for
a kind of task after verification: it must succeed on fresh scenes (train and held-out seed blocks) and must not do
worse than the fixed plan on the same scenes (paired test). Then it is saved as a composite card in
skill_library/composites/ with its evidence and lineage, and later requests with the same task signature reuse it
without asking the model again.

Signature: the goal structure in terms of object kinds, e.g. ["on:block->block", "on:block->block"] for a 3-block stack,
so "stack the blocks" and "build a tower, largest at the bottom" with the same objects share a composite. A program
that hard-codes object ids only applies where those ids exist.
"""
from __future__ import annotations

import ast
import json
import time

from .runner import ROOT, TrialPool, trial_cfg, wilson
from .search.explore import mcnemar_p
from .skills.library import LIB_DIR
from .tasks.splits import split_seeds

COMP_DIR = LIB_DIR / "composites"
MIN_RATE = 0.8            # verification success needed on fresh scenes
ALPHA = 0.05


def signature(task: dict) -> list[str]:
    kinds = {o["id"]: o["template"] for o in task["objects"]}
    sig = sorted(f"{g['pred']}:{kinds.get(g['obj'])}->{kinds.get(g.get('target'))}" for g in task["goal"]
                 if g.get("pred") in ("inside", "on"))
    sig += sorted(f"{c['pred']}:{kinds.get(c['obj'])}" for c in task.get("constraints", []))
    return sig


def program_ids(code: str) -> set[str]:
    """Object ids a program hard-codes (string arguments named id=, or the first argument of find_one/find)."""
    ids = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    ids.add(kw.value.value)
    return ids


def applicable(card: dict, task: dict) -> bool:
    return card["signature"] == signature(task) and program_ids(card["program"]) <= {o["id"] for o in task["objects"]}


def load_composites() -> dict:
    if not COMP_DIR.exists():
        return {}
    return {c["name"]: c for c in (json.loads(p.read_text()) for p in sorted(COMP_DIR.glob("*.json")))}


def find_composite(task: dict) -> dict | None:
    """The best verified composite for this task (highest verification success), or None."""
    cands = [c for c in load_composites().values() if applicable(c, task)]
    return max(cands, key=lambda c: c["evidence"]["rate"]) if cands else None


def spec_hash(task: dict) -> str:
    import hashlib
    keep = {k: task[k] for k in ("objects", "goal", "constraints") if k in task}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def quick_check(task: dict, program: str, workers=None, n=4, min_ok=3) -> dict:
    """Retrieve, then verify: a composite applied to a task spec it was not verified on must pass a few fresh
    scenes first (same signature does not mean same goals: "smallest at the bottom" looks like "largest")."""
    out = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S_quick_check")
    cfg = trial_cfg(task["name"], out, tag="quick", task_spec={**task, "program": program}, save_failures=False,
                    success_sample=0, save_video=False, save_poses=False)
    seeds = split_seeds("dev", n, start=900)
    with TrialPool(workers) as pool:
        res = list(pool.map([(cfg, s) for s in seeds]))
    k = sum(r["success"] for r in res)
    return {"success": k, "trials": n, "pass": k >= min_ok}


def _name(task):
    base = "".join(ch if ch.isalnum() else "_" for ch in task["name"].lower()).strip("_") or "composite"
    return base[:40]


def verify_program(task: dict, program: str, workers=None, n_train=16, n_heldout=16, source=None,
                   run_id=None, promote=True) -> dict:
    """Run the program and the fixed plan on the same fresh scenes; promote a composite card if it passes."""
    from .program import check_program
    errs = check_program(program)
    if errs:
        return {"pass": False, "reason": "static check failed", "errors": errs}
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S_verify_program")
    out = ROOT / "runs" / run_id
    seeds = split_seeds("train", n_train, start=500) + split_seeds("dev", n_heldout, start=500)
    common = dict(save_failures=False, success_sample=0, save_video=False, save_poses=False)
    prog_cfg = trial_cfg(task["name"], out, tag="program", task_spec={**task, "program": program}, **common)
    plan_cfg = trial_cfg(task["name"], out, tag="fixed_plan", task_spec={k: v for k, v in task.items()
                                                                         if k != "program"}, **common)
    with TrialPool(workers) as pool:
        res = {(r["tag"], r["seed"]): r for r in pool.map([(prog_cfg, s) for s in seeds] + [(plan_cfg, s) for s in seeds])}
    p_ok = [bool(res[("program", s)]["success"]) for s in seeds]
    f_ok = [bool(res[("fixed_plan", s)]["success"]) for s in seeds]
    k, n = sum(p_ok), len(p_ok)
    held = p_ok[n_train:]
    p_worse = mcnemar_p(p_ok, f_ok)                     # is the program significantly WORSE than the fixed plan?
    p_better = mcnemar_p(f_ok, p_ok)
    gate = {"program": {"success": k, "trials": n, "rate": round(k / n, 3), "ci95": wilson(k, n)},
            "heldout": {"success": sum(held), "trials": len(held)},
            "fixed_plan": {"success": sum(f_ok), "trials": n, "rate": round(sum(f_ok) / n, 3)},
            "p_worse_than_fixed": p_worse, "p_better_than_fixed": p_better, "seeds": seeds}
    gate["pass"] = k / n >= MIN_RATE and sum(held) / len(held) >= MIN_RATE and p_worse >= ALPHA
    gate["reason"] = ("verified" if gate["pass"] else
                      f"needs >= {MIN_RATE:.0%} overall and on held-out scenes, and not worse than the fixed plan")
    report = {"run_id": run_id, "task": task["name"], "prompt": task.get("prompt"), "signature": signature(task),
              "program": program, "source": source, "gate": gate}
    if gate["pass"] and promote:
        report["card"] = _promote(task, program, gate, source, run_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def _promote(task, program, gate, source, run_id):
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    name = _name(task)
    path = COMP_DIR / f"{name}.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old:
        (COMP_DIR / "history").mkdir(exist_ok=True)
        (COMP_DIR / "history" / f"{name}_v{old['version']}.json").write_text(json.dumps(old, indent=2))
    card = {"name": name, "kind": "composite", "version": (old["version"] + 1) if old else 1,
            "description": f"Verified plan program for: {task.get('prompt') or task['name']}",
            "signature": signature(task), "program": program, "verified_spec": spec_hash(task),
            "evidence": {"rate": gate["program"]["rate"], "success": gate["program"]["success"],
                         "trials": gate["program"]["trials"], "heldout": gate["heldout"],
                         "fixed_plan_rate": gate["fixed_plan"]["rate"],
                         "p_better_than_fixed": gate["p_better_than_fixed"]},
            "lineage": (old["lineage"] if old else []) + [{
                "version": (old["version"] + 1) if old else 1, "source": "plan_program", "run_id": run_id,
                "date": time.strftime("%Y-%m-%d"), "written_by": source, "prompt": task.get("prompt")}]}
    path.write_text(json.dumps(card, indent=2))
    return {"name": name, "version": card["version"], "path": str(path.relative_to(ROOT))}
