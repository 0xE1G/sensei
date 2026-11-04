"""Model evaluation suite: the same inputs to each backend arm, scored automatically.

    python scripts/eval_models.py task_spec [--arms rule spark spark_low spark_low_schema]
    python scripts/eval_models.py programs  [--arms ...] [-n 8]

task_spec  20 typed requests with the expected outcome (ok with the right goal structure, or refuse, or clarify);
           a spec counts as correct only if its status matches and, for ok, every expected goal / constraint / pose
           check holds (templates, not ids, are compared).
programs   tasks that need different abilities (ordering, stacking, relations, a push, standing up, opening and
           closing a drawer): the arm writes a plan program from the prompt, the program runs on n fresh scenes
           (seeds 700+, never used by any gate), next to the fixed plan on the same scenes.
Arms: rule (stand-ins), spark (Muse Spark, model-chosen reasoning effort), spark_low (reasoning
effort low), spark_low_schema (+ answers constrained to the job schema). Report: runs/<id>_eval_<suite>/report.json (latency, tokens, repairs per call).
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.llm import ModelClient, ModelRejected  # noqa: E402
from kintsugi.runner import ROOT  # noqa: E402

ARMS = {"rule": dict(backend="rule"),
        "wonder": dict(backend="spark", model="muse-wonder-1.0"),
        "spark": dict(backend="spark"),
        "spark_low": dict(backend="spark", effort="low"),
        "spark_low_schema": dict(backend="spark", effort="low", schema_output=True)}


def _goals(spec):
    kinds = {o["id"]: o["template"] for o in spec["objects"]}
    return [(g["pred"], kinds.get(g["obj"]), kinds.get(g.get("target"))) for g in spec["goal"]], kinds


def has(pred, obj, target=None, n=1):
    """At least n goals pred(obj-template, target-template); obj/target may be a tuple of acceptable templates."""
    def ok(spec):
        gs, _ = _goals(spec)
        m = lambda v, want: want is None or v == want or (isinstance(want, tuple) and v in want)
        return sum(p == pred and m(o, obj) and m(t, target) for p, o, t in gs) >= n
    return ok


def constraint_on(template):
    return lambda spec: any({o["id"]: o["template"] for o in spec["objects"]}.get(c["obj"]) == template
                            for c in spec.get("constraints", []))


def lying(template):
    return lambda spec: any(o["template"] == template and o.get("pose") == "lying" for o in spec["objects"])


def ordered(n):
    return lambda spec: len(spec.get("order") or []) >= n


def _part(template):
    from kintsugi.sim.templates import ARTICULATED
    return ARTICULATED.get(template, {}).get("part")


def joint(part, op):
    """A joint goal on any object whose moving part is `part` (the drawer template, an imported nightstand, ...)."""
    return lambda spec: any(g["pred"] == "joint" and g.get("op", ">=") == op and
                            _part({o["id"]: o["template"] for o in spec["objects"]}.get(g["obj"])) == part
                            for g in spec["goal"])


def starts_open(part):
    return lambda spec: any(_part(o["template"]) == part and float((o.get("params") or {}).get("open", 0)) > 0
                            for o in spec["objects"])


TRAYS = ("tray", "zone")
TASK_SPEC_CASES = [
    ("Put the red block on the tray.", "ok", [has("on", "block", TRAYS)]),
    ("Sort the blocks and the bowl onto the trays of the same colour.", "ok",
     [has("on", "block", TRAYS, 2), has("on", "bowl", TRAYS)]),
    ("Put the balls in the basket.", "ok", [has("inside", "ball", ("bin", "easter_basket", "storage_bin"), 2)]),
    ("Stack the blocks, largest at the bottom.", "ok", [has("on", "block", "block", 2)]),
    ("Put the mug on the tray, don't knock over the bottle.", "ok", [has("on", "mug", TRAYS), constraint_on("cylinder")]),
    ("Open the drawer.", "ok", [joint("drawer", ">=")]),
    ("Close the door.", "ok", [joint("door", "<="), starts_open("door")]),
    ("Move the plate onto the blue zone.", "ok", [has("on", "plate", "zone")]),
    ("The bottle fell over. Stand it back up.", "ok", [has("upright", "cylinder"), lying("cylinder")]),
    ("Put the block to the left of the bowl.", "ok", [has("left_of", "block", "bowl")]),
    ("Put the yellow block behind the red block.", "ok", [has("behind", "block", "block")]),
    ("First put the red block in the bowl, then the green block.", "ok", [has("inside", "block", "bowl", 2), ordered(2)]),
    ("Put the cup in the bowl.", "ok", [has("inside", "cup", "bowl")]),
    ("Put two blocks on the tray and the ball in the basket.", "ok",
     [has("on", "block", TRAYS, 2), has("inside", "ball", ("bin", "easter_basket", "storage_bin"))]),
    ("Put the pill bottle on the tray.", "ok", [has("on", "pill_bottle", TRAYS)]),
    ("Pour water into the cup.", "refuse", []),
    ("Fold the towel and put it in the basket.", "refuse", []),
    ("Cut the apple in half.", "refuse", []),
    ("Unscrew the lid of the jar.", "refuse", []),
    ("Tidy up.", "clarify", []),
]


def eval_task_spec(arms, out):
    rows = {}
    for arm in arms:
        client = ModelClient(**ARMS[arm])
        res = []
        for prompt, want, checks in TASK_SPEC_CASES:
            t0 = time.time()
            try:
                r = client.complete("task_spec", {"prompt": prompt})
                o, meta = r.output, r.meta
                status = o["status"]
                ok_checks = [bool(c(o["task"])) for c in checks] if status == "ok" else []
                correct = status == want and all(ok_checks)
                err = None
            except (ModelRejected, Exception) as e:
                status, ok_checks, correct, meta, err = "error", [], False, {}, f"{type(e).__name__}: {str(e)[:200]}"
            res.append({"prompt": prompt, "expected": want, "status": status, "checks": ok_checks, "correct": correct,
                        "latency_s": meta.get("latency_s", round(time.time() - t0, 2)), "usage": meta.get("usage"),
                        "repairs": meta.get("repairs"), "cached": meta.get("cached"), "error": err})
            print(f"  {arm:16s} {'OK ' if correct else 'BAD'} {status:8s} {prompt[:60]}", flush=True)
        k = sum(r["correct"] for r in res)
        lat = [r["latency_s"] for r in res if r["latency_s"] and not r["cached"]]
        rows[arm] = {"correct": k, "total": len(res), "median_latency_s": sorted(lat)[len(lat) // 2] if lat else None,
                     "completion_tokens": sum((r["usage"] or {}).get("completion_tokens", 0) for r in res),
                     "repairs": sum(r["repairs"] or 0 for r in res), "errors": sum(r["status"] == "error" for r in res),
                     "cases": res}
        print(f"{arm}: {k}/{len(res)} correct, median {rows[arm]['median_latency_s']} s", flush=True)
    return rows


# ------------------------------------------------------------------------------------------------ programs
def _program_tasks():
    from kintsugi.tasks.specs import make_task
    rel = {"name": "block_left_of_bowl", "prompt": "Put the block to the left of the bowl.", "constraints": [],
           "objects": [{"id": "block", "template": "block", "role": "item", "params": {"color": "red"}},
                       {"id": "bowl", "template": "bowl", "role": "target", "params": {}}],
           "goal": [{"pred": "left_of", "obj": "block", "target": "bowl"}]}
    order = {"name": "ordered_sort", "prompt": "First put the blue block in the bowl, then the red block, then the green block.",
             "constraints": [], "order": ["block_blue", "block_red", "block_green"],
             "objects": [{"id": "bowl", "template": "bowl", "role": "target", "fixed": True, "params": {"radius": 0.09}}]
             + [{"id": f"block_{c}", "template": "block", "role": "item", "params": {"color": c, "size": 0.04}}
                for c in ("red", "green", "blue")],
             "goal": [{"pred": "inside", "obj": f"block_{c}", "target": "bowl"} for c in ("red", "green", "blue")]}
    out = {"stack_by_size": make_task("stack_by_size", 0), "ordered_sort": order, "block_left_of_bowl": rel,
           "slide_plate": make_task("slide_plate", 0), "stand_up": make_task("stand_up", 0),
           "open_drawer": make_task("open_drawer", 0), "close_drawer": make_task("close_drawer", 0)}
    for t in out.values():
        t.pop("layout_fixed", None)
    return out


def eval_programs(arms, out, n):
    from kintsugi.runner import TrialPool, run_batch
    tasks = _program_tasks()
    seeds = list(range(700, 700 + n))
    rows = {}
    with TrialPool() as pool:
        fixed = {}
        for name, task in tasks.items():          # the fixed plan on the same scenes, once
            s = run_batch("spec", task_spec=task, seeds=seeds, pool=pool, save_video=False, save_poses=False,
                          save_failures=False, success_sample=0, progress=False, feasibility=True,
                          run_id=f"{out.name}/fixed_{name}")
            fixed[name] = {"success": s["success"], "trials": s["trials"]}
            print(f"  fixed plan  {name:20s} {s['success']}/{s['trials']}", flush=True)
        for arm in arms:
            client = ModelClient(**ARMS[arm])
            res = {}
            for name, task in tasks.items():
                t0 = time.time()
                try:
                    r = client.complete("plan_program", {"prompt": task["prompt"], "task": task})
                    prog, meta = r.output["program"], r.meta
                except Exception as e:
                    res[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}", "success": 0, "trials": n}
                    print(f"  {arm:16s} {name:20s} ERROR {res[name]['error'][:80]}", flush=True)
                    continue
                s = run_batch("spec", task_spec={**task, "program": prog}, seeds=seeds, pool=pool, save_video=False,
                              save_poses=False, save_failures=False, success_sample=0, progress=False,
                              feasibility=True, run_id=f"{out.name}/{arm}_{name}")
                res[name] = {"success": s["success"], "trials": s["trials"], "program": prog,
                             "latency_s": meta.get("latency_s"), "usage": meta.get("usage"), "repairs": meta.get("repairs"),
                             "labels": s["failure_labels"]}
                print(f"  {arm:16s} {name:20s} {s['success']}/{s['trials']}  ({meta.get('latency_s')} s)", flush=True)
            k = sum(v["success"] for v in res.values())
            rows[arm] = {"success": k, "trials": n * len(tasks), "tasks": res}
            print(f"{arm}: {k}/{n * len(tasks)} scenes", flush=True)
    return {"fixed_plan": fixed, "arms": rows}


# ------------------------------------------------------------------------------------------------ skills
SKILL_CASES = {"push": "slide_plate", "stand_up": "stand_up", "open_drawer": "open_drawer", "close_drawer": "close_drawer"}


def eval_skills(arms, out, kinds):
    """Create each capability from scratch (the library without any card of that kind), dry run: nothing is written."""
    from kintsugi.create_skill import SkillCreator
    from kintsugi.skills.library import load_cards
    rows = {}
    for arm in arms:
        rows[arm] = {}
        for kind in kinds:
            lib = {n: c for n, c in load_cards().items() if c.get("kind") != kind}
            client = ModelClient(**ARMS[arm])
            t0 = time.time()
            sc = SkillCreator(kind, SKILL_CASES[kind], client=client, library=lib, run_id=f"{out.name}/{arm}_{kind}")
            sc.promote = lambda card, *rest, sc=sc: sc.report.update(promoted={"card": card["name"], "dry_run": True})
            try:
                rep = sc.run()
            except Exception as e:
                rep = {"outcome": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            props = rep.get("phases", {}).get("proposals", [])
            gate = rep.get("phases", {}).get("gate") or {}
            rows[arm][kind] = {"outcome": rep.get("outcome"), "promoted": (rep.get("promoted") or {}).get("card"),
                               "rounds": max([p.get("attempt", 1) for p in props] or [0]),
                               "proposals": len(props), "rejected": sum("rejected" in p for p in props),
                               "held_out": gate.get("held_out"), "trials": rep.get("trials_simulated"),
                               "wall_time_s": round(time.time() - t0), "model_stats": client.stats,
                               "error": rep.get("error")}
            print(f"{arm:16s} {kind:12s} {rows[arm][kind]['outcome']}  rounds {rows[arm][kind]['rounds']}  "
                  f"{rows[arm][kind]['wall_time_s']} s", flush=True)
            (out / "partial.json").write_text(json.dumps(rows, indent=1, default=float))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["task_spec", "programs", "skills"])
    ap.add_argument("--kinds", nargs="*", default=list(SKILL_CASES))
    ap.add_argument("--arms", nargs="*", default=["rule", "spark", "spark_low", "spark_low_schema"])
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    out = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + f"_eval_{a.suite}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = (eval_task_spec(a.arms, out) if a.suite == "task_spec" else eval_programs(a.arms, out, a.n)
            if a.suite == "programs" else eval_skills(a.arms, out, a.kinds))
    rep = {"suite": a.suite, "arms": a.arms, "results": rows, "seconds": round(time.time() - t0)}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
