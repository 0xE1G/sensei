"""Auto-repair: failures from ordinary task runs trigger investigation and repair, without anyone choosing what.

    1. collect failed steps from trial records (any runs: named tasks, typed-request specs, probe scenes)
    2. cluster them by (object type, grasp skill) per task source
    3. replay each failure exactly with the oracle: only failures the oracle can solve count
    4. for every cluster with >= min_failures valid failures:
         investigator on up to max_cases of them -> confirmed one-change fixes (hints)
         explorer on the cluster (probe failures -> a region around the failing objects; named task -> itself;
         typed-request spec -> task "spec") seeded with the hints -> routing / patch / new card, through the gates
Report: runs/<id>/report.json.
"""
from __future__ import annotations

import collections
import json
import pathlib
import time

from .runner import ROOT, TrialPool, trial_cfg
from .sim.templates import RANGES
from .tasks.specs import TASKS


def _log(msg):
    print(f"[auto_repair] {msg}", flush=True)


def collect_failures(run_dirs, tag=None) -> list[dict]:
    out = []
    for d in run_dirs:
        for p in sorted(pathlib.Path(d).glob("trials/*.json")):
            rec = json.loads(p.read_text())
            if tag and tag not in (rec.get("tag") or ""):
                continue
            spec = rec.get("task_spec")
            if not spec or rec.get("success"):
                continue
            kinds = {o["id"]: o["template"] for o in spec["objects"]}
            for st in rec["subtasks"]:
                if st.get("label") and st.get("trace") and not st.get("ok"):
                    out.append({"spec": spec, "seed": rec["seed"], "obj": st["obj"], "kind": kinds.get(st["obj"]),
                                "skill": st["trace"][0]["skill"], "label": st["label"], "source": str(p),
                                "perturb": rec.get("perturb")})
                    break                                     # the first failed step of the episode
    return out


def _source_key(f):
    s = f["spec"]
    if s.get("scenario"):
        return "probe"
    return s["name"] if s["name"] in TASKS else f"spec:{s['name']}"


def repair_target(cluster):
    """(task, task_kwargs) for the explorer."""
    f0 = cluster[0]
    if _source_key(f0) == "probe":
        t = f0["kind"]
        region = {"template": t, "params": {}}
        for k in RANGES[t]:
            vals = [f["spec"]["scenario"]["params"][k] for f in cluster if k in f["spec"]["scenario"]["params"]]
            if vals:
                region["params"][k] = [round(min(vals), 4), round(max(vals), 4)]
        return "probe", {"region": region}
    if f0["spec"]["name"] in TASKS:
        return f0["spec"]["name"], {}
    return "spec", {"task_spec": {k: v for k, v in f0["spec"].items() if k != "program"}}


def auto_repair(run_dirs, workers=None, client=None, min_failures=8, max_cases=8, dry_run=False, run_id=None,
                tag=None):
    from .agents.llm import ModelClient
    from .investigate import Investigator
    from .search.explore import Explorer
    client = client or ModelClient()
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S_auto_repair")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    report = {"run_id": run_id, "inputs": [str(d) for d in run_dirs], "clusters": []}
    fails = collect_failures(run_dirs, tag)
    _log(f"{len(fails)} failed episodes in {len(run_dirs)} run(s)")

    # Oracle: replay each failing scene exactly; keep only failures it can solve.
    jobs = []
    for i, f in enumerate(fails):
        cfg = trial_cfg("spec", out, tag=f"replay{i}", save_failures=False, success_sample=0, save_video=False,
                        save_poses=False, task_spec={k: v for k, v in f["spec"].items()}, oracle="on_failure",
                        perturb=f["perturb"])
        jobs.append((cfg, f["seed"]))
    with TrialPool(workers) as pool:
        res = {r["tag"]: r for r in pool.map(jobs)}
    valid = []
    for i, f in enumerate(fails):
        r = res[f"replay{i}"]
        if not r["success"] and r["oracle"] and r["oracle"]["subtask_ok"].get(f["obj"]):
            valid.append(f)
    groups = collections.defaultdict(list)
    for f in valid:
        groups[(_source_key(f), f["kind"], f["skill"], json.dumps(f["perturb"], sort_keys=True))].append(f)
    _log(f"{len(valid)} valid (oracle-solvable) failures; clusters: "
         f"{ {f'{k[0]} {k[1]}/{k[2]}': len(v) for k, v in groups.items()} }")

    for (src, kind, skill, _), cl in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        entry = {"source": src, "kind": kind, "skill": skill, "perturb": cl[0]["perturb"], "valid_failures": len(cl),
                 "labels": dict(collections.Counter(f["label"] for f in cl))}
        report["clusters"].append(entry)
        if len(cl) < min_failures:
            entry["action"] = f"skipped: fewer than {min_failures} valid failures"
            continue
        # Investigate: confirmed one-change fixes become hints for the explorer.
        inv = Investigator(workers, client=client, run_id=f"{run_id}/investigate_{kind}_{skill}")
        ir = inv.run([{"task_spec": f["spec"], "seed": f["seed"], "perturb": f["perturb"],
                       "source": {"cluster": f"{kind}/{skill}"}} for f in cl[:max_cases]])
        hints = [h["test"] for c in ir["cases"] for h in c.get("hypotheses", []) if h.get("verdict") == "confirmed"]
        entry["investigation"] = {"run_id": ir["run_id"], **{k: ir["metrics"][k] for k in
                                  ("diagnosed", "top1_label_accuracy", "confirmed", "rejected")}}
        task, kw = repair_target(cl)
        entry["repair_target"] = {"task": task, "kwargs": {k: v for k, v in kw.items() if k != "task_spec"}}
        _log(f"cluster {kind}/{skill} ({len(cl)} failures, {entry['labels']}): {len(hints)} confirmed fixes; "
             f"explorer on {task} {entry['repair_target']['kwargs']}")
        ex = Explorer(task, workers=workers, task_kwargs=kw, hints=hints, perturb=cl[0]["perturb"],
                      run_id=f"{run_id}/explore_{kind}_{skill}")
        if dry_run:
            ex.promote = lambda card, *a: ex.report.update(promoted={"card": card, "dry_run": True})
            ex.promote_routing = lambda name, *a: ex.report.update(routed={"card": name, "dry_run": True})
            ex.promote_patch = lambda ch, *a: ex.report.update(patched={"name": ch["name"], "dry_run": True})
        er = ex.run()
        entry["repair"] = {"run_id": er["run_id"], "outcome": er.get("outcome"),
                           "routed": er.get("routed"), "patched": er.get("patched"),
                           "promoted": (er.get("promoted") or {}).get("card", {}).get("name")
                           if isinstance((er.get("promoted") or {}).get("card"), dict) else None}
        _log(f"cluster {kind}/{skill}: {entry['repair']['outcome']}")
        (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report
