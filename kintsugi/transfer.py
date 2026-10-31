"""Transfer: reuse the library on an object type it has never seen, and measure what that saves.

    1. perceive a few new objects of the type (features only, no name)
    2. retrieval: the model ranks the library's fitting grasp skills by features and evidence, and explains its choice
    3. verification: a few trials of the chosen skill on new objects (test-split seeds) before it is trusted
    4. baseline: what a from-scratch repair costs: the explorer from the seed library (no transferred skills), dry run

Reported: verification trials and success vs the from-scratch explorer's trials (search trials and total) and its
outcome. Output: runs/<id>/report.json.
"""
from __future__ import annotations

import collections
import json
import time

from .runner import ROOT, TrialPool, trial_cfg, wilson
from .tasks.splits import split_seeds


def _log(msg):
    print(f"[transfer] {msg}", flush=True)


def run_transfer(template: str, workers=None, client=None, n_verify=10, scratch=True, run_id=None) -> dict:
    from .agents.llm import ModelClient
    from .perception.perceive import Perception
    from .search.explore import Explorer
    from .sim.scene import build_scene
    from .skills.library import load_cards, seed_library
    from .skills.router import load_patches, load_performance, shape_group
    from .tasks.planner import fits
    from .tasks.specs import make_task

    client = client or ModelClient()
    region = {"template": template, "params": {}}
    run_id = run_id or time.strftime(f"%Y%m%d-%H%M%S_transfer_{template}")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    cards, perf, patches = load_cards(), load_performance(), load_patches()
    t0 = time.time()

    # 1. perceive new objects
    descs = []
    for s in split_seeds("test", 4, start=1000):
        sc = build_scene(make_task("probe", seed=s, region=region), seed=s)
        per = Perception(sc)
        descs.append(per.perceive([template])[template])
        per.close()
    groups = collections.Counter(shape_group(d) for d in descs)
    keep = ("width", "height", "hollow", "rim_radius", "wall_slope", "elongation", "extent_major", "extent_minor")
    obj = {**{k: round(descs[0][k], 4) if isinstance(descs[0][k], float) else descs[0][k] for k in keep if k in descs[0]},
           "shape_group": groups.most_common(1)[0][0]}

    # 2. retrieval
    grasps = {n: c for n, c in cards.items() if c.get("kind") == "grasp"}
    fitting = [n for n, c in grasps.items() if sum(fits(c, d) for d in descs) >= len(descs) / 2] or list(grasps)
    cands = []
    for n in fitting:
        c = grasps[n]
        trig = next((e.get("trigger") for e in reversed(c.get("lineage", [])) if e.get("trigger")), None)
        cands.append({"card": n, "description": c.get("description", ""), "requires": c.get("requires", {}),
                      "evidence": {g: f"{e['success']}/{e['trials']}" for g, e in perf.get(n, {}).items()},
                      "found_on": f"{trig['obj']} ({trig['task']})" if trig else "hand-written seed skill"})
    res = client.complete("retrieve", {"object": obj, "candidates": cands})
    choice = res.output["choice"]
    _log(f"retrieval ({res.meta['backend']}): {choice} - {res.output['explanation']}")

    # 3. verification on new objects
    seeds = split_seeds("test", n_verify)
    cfg = trial_cfg("probe", out, save_failures=False, success_sample=0, save_video=False, save_poses=False,
                    tag="verify", library=cards, performance=perf, patches=patches,
                    task_kwargs={"region": region}, overrides={"grasp_for": {template: choice}})
    with TrialPool(workers) as pool:
        rr = list(pool.map([(cfg, s) for s in seeds]))
    k = sum(bool(r["subtask_ok"].get(template)) for r in rr)
    labels = collections.Counter(l for r in rr for l in r["labels"])
    verified = k / len(rr) >= 0.8
    _log(f"verification: {choice} {k}/{len(rr)} on new {template}s ({dict(labels)}) -> "
         f"{'TRUSTED' if verified else 'not trusted'}")
    report = {"run_id": run_id, "template": template, "perceived_example": obj, "groups": dict(groups),
              "candidates": cands, "retrieval": {"output": res.output, "meta": res.meta},
              "verification": {"card": choice, "success": k, "trials": len(rr), "ci95": wilson(k, len(rr)),
                               "labels": dict(labels), "trusted": verified, "seeds": seeds}}

    # 4. from-scratch baseline: the explorer from the seed library, nothing promoted
    if scratch:
        ex = Explorer("probe", workers=workers, library=seed_library(), performance={}, patches={},
                      task_kwargs={"region": region}, run_id=f"{run_id}/scratch")
        ex.promote = lambda card, *a: ex.report.update(promoted={"card": card, "dry_run": True})
        ex.promote_routing = lambda name, *a: ex.report.update(routed={"card": name, "dry_run": True})
        ex.promote_patch = lambda chosen, *a: ex.report.update(patched={"name": chosen["name"], "dry_run": True})
        er = ex.run()
        ph = er["phases"]
        n_before = ph.get("before", {}).get("trials", 0)
        n_gates = sum(g["fixes_cluster"]["trials"] + 2 * g["held_out"]["trials"]
                      + sum(t["after"]["trials"] for t in g["no_regression"]["tasks"].values())
                      for g in ph.get("gates", []))
        report["scratch"] = {"outcome": er.get("outcome"), "trials_total": er["trials_simulated"],
                             "trials_search": er["trials_simulated"] - n_before - n_gates,
                             "trials_before": n_before, "trials_gates": n_gates,
                             "found": (er.get("promoted") or {}).get("card", {}).get("name")
                             if isinstance((er.get("promoted") or {}).get("card"), dict) else None,
                             "report": f"runs/{run_id}/scratch/report.json"}
        _log(f"from scratch: {er.get('outcome')} after {er['trials_simulated']} trials "
             f"({report['scratch']['trials_search']} search)")
    report["wall_time_s"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report
