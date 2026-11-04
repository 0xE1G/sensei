"""Headline harness: every headline number from one command, into one table (runs/<id>_headlines/table.json + .md).

    python scripts/headlines.py                   # quick (~15 min): re-runs the cheap headlines, reuses the latest
                                                  # report of each expensive one (marked "reused" with its run id)
    python scripts/headlines.py --full --repeats 3   # re-runs everything; repeats give mean and spread (hours)
    python scripts/headlines.py --only tasks scanned_repair --repeats 3
    python scripts/headlines.py --final           # + the frozen test sets (scanned test objects, bowl/cup test bands):
                                                  # run ONCE, after everything else is frozen
    python scripts/headlines.py --list            # the headlines, how each is measured, and its last result

Each headline runs the same script and arguments as when it was first reported (fixed seeds; repeats shift the seed
block or, for model jobs, re-ask without the cache), reads the report the script writes, and records value, n,
spread, run ids, backend and code version. Model calls and tokens per headline come from the reports themselves, so
the table also says what a full re-run on Spark costs. Historical numbers that cannot be re-measured (the
failure-search trend over earlier library states) are copied from their saved reports and marked as such.
"""
import argparse
import glob
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable


def _run(args, env=None, timeout=6 * 3600):
    """Run a script; return (report path from its last 'report:' / '->' line, stdout tail)."""
    p = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, **(env or {})}, timeout=timeout)
    out = p.stdout + p.stderr
    m = re.findall(r"(?:report:|->)\s*(\S*report\.json)", out)
    if p.returncode != 0 or not m:
        raise RuntimeError(f"{' '.join(args)} failed (exit {p.returncode}): {out[-800:]}")
    path = pathlib.Path(m[-1])
    return (path if path.is_absolute() else ROOT / path), out[-2000:]


def _latest(pattern):
    files = sorted(glob.glob(str(ROOT / "runs" / pattern)), key=os.path.getmtime)
    return pathlib.Path(files[-1]) if files else None


def _read(p):
    return json.loads(pathlib.Path(p).read_text())


def _latest_where(pattern, pred):
    """The newest saved report matching `pattern` whose content satisfies pred(report)."""
    for f in sorted(glob.glob(str(ROOT / "runs" / pattern)), key=os.path.getmtime, reverse=True):
        try:
            if pred(_read(f)):
                return pathlib.Path(f)
        except Exception:
            continue
    return None


def _rid(p):
    return pathlib.Path(p).parent.relative_to(ROOT / "runs").as_posix()


# ---------------------------------------------------------------------------------------------------- headlines
# Each: measure(rep) -> report path (runs the experiment); parse(report) -> {metric: value, ...};
# reuse = glob of the latest saved report; cost = "cheap" (re-run in quick mode) or "expensive".

def m_tasks(rep):
    """The current library on the named tasks, as the Studio runs them (feasibility on, no retries), fresh seeds."""
    from kintsugi.runner import TrialPool, run_batch
    from kintsugi.tasks.splits import split_seeds
    out = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + f"_headline_tasks_r{rep}")
    out.mkdir(parents=True)
    tasks = {"sort_with_bowl": None, "sort_by_color": None, "balls_in_basket": None, "stack_by_size": None,
             "bowl_on_tray": "dev", "cup_on_tray": "dev", "slide_plate": None, "stand_up": None,
             "open_drawer": None, "close_drawer": None}
    rows = {}
    with TrialPool() as pool:
        for t, split in tasks.items():
            seeds = split_seeds(split, 16, 3000 + 100 * rep) if split else list(range(3000 + 100 * rep, 3016 + 100 * rep))
            s = run_batch(t, seeds=seeds, pool=pool, feasibility=True, save_video=False, save_poses=False,
                          save_failures=False, success_sample=0, progress=False, run_id=f"{out.name}/{t}")
            rows[t] = {"success": s["success"], "trials": s["trials"], "labels": s["failure_labels"]}
    (out / "report.json").write_text(json.dumps({"tasks": rows, "rep": rep}, indent=1))
    return out / "report.json"


def p_tasks(r):
    k = sum(v["success"] for v in r["tasks"].values())
    n = sum(v["trials"] for v in r["tasks"].values())
    return {"success_rate": round(k / n, 3), "success": k, "trials": n,
            **{f"task:{t}": f"{v['success']}/{v['trials']}" for t, v in r["tasks"].items()}}


def m_demo_repair(rep):
    return _run(["scripts/explore.py", "sort_with_bowl", "--from-seed", "--dry-run", "--seed", str(rep)])[0]


def p_demo_repair(r):
    g = next((g for g in reversed(r["phases"].get("gates", [])) if g.get("pass")), (r["phases"].get("gates") or [{}])[-1])
    fc, ho = g.get("fixes_cluster", {}), g.get("held_out", {})
    return {"outcome": r.get("outcome"), "found": g.get("card"),
            "failed_trials_after": f"{fc.get('success')}/{fc.get('trials')}",
            "held_out_after": f"{ho.get('success')}/{ho.get('trials')}",
            "held_out_before": f"{(ho.get('before') or {}).get('success')}/{(ho.get('before') or {}).get('trials')}",
            "trials_simulated": r.get("trials_simulated")}


def m_transfer(rep):
    return _run(["scripts/transfer.py", "mug", "--backend", "spark"])[0]


def p_transfer(r):
    v, sc = r.get("verification") or {}, r.get("scratch") or {}
    if not sc:                                   # the from-scratch baseline is long: take the latest one that has it
        p = _latest_where("*_transfer_mug/report.json", lambda x: x.get("scratch"))
        sc = {**_read(p)["scratch"], "from_run": _rid(p)} if p else {}
    return {"skill": (r.get("retrieval") or {}).get("output", {}).get("choice"),
            "verify": f"{v.get('success')}/{v.get('trials')}", "trials_with_library": v.get("trials"),
            "scratch_trials_total": sc.get("trials_total"), "scratch_trials_search": sc.get("trials_search"),
            "scratch_from": sc.get("from_run", "this run"),
            "backend": ((r.get("retrieval") or {}).get("meta") or {}).get("backend")}


def m_skills(rep):
    env = {"KINTSUGI_NO_CACHE": "1"} if rep else None
    return _run(["scripts/eval_models.py", "skills", "--arms", "spark"], env=env)[0]


def p_skills(r):
    arm = r["results"].get("spark") or next(iter(r["results"].values()))
    out = {"promoted": sum(v["outcome"] == "promoted" for v in arm.values()), "kinds": len(arm),
           "round1": sum(v["outcome"] == "promoted" and v["rounds"] == 1 for v in arm.values())}
    for k, v in arm.items():
        h = v.get("held_out") or {}
        out[f"skill:{k}"] = f"{v['outcome']} r{v['rounds']} held-out {h.get('success')}/{h.get('trials')}"
    out["completion_tokens"] = sum((v.get("model_stats") or {}).get("completion_tokens", 0) for v in arm.values())
    return out


def m_diag_seed(rep):
    return _run(["scripts/investigate.py", "--from-search", "runs/20261030-225954_failsearch/report.json",
                 "--library", "seed", "--backend", "spark", "--vision", "on"])[0]


def p_diag(r):
    m = r["metrics"]
    return {"top1_label_accuracy": m["top1_label_accuracy"], "diagnosed": m["diagnosed"],
            "confirmed_first_look": m.get("cases_confirmed_first_look"),
            "confirmed_after_second_look": m["cases_with_a_confirmed_cause"], "backend": r.get("backend")}


def m_failsearch(rep):
    return _run(["scripts/failure_search.py", "--backend", "spark", "--budget", "160", "--seed", str(rep)])[0]


def p_failsearch(r):
    out = {}
    for s, v in r["strategies"].items():
        out[f"{s}_per100"] = v["metrics"]["valid_failures_per_100"]
        out[f"{s}_clusters"] = v["metrics"]["distinct_clusters"]
    return out


def m_scanned_repair(rep):
    return _run(["scripts/scanned_benchmark.py", "--split", "repair", "-n", "8", "--seed0", str(8 * rep)])[0]


def m_scanned_test(rep):
    return _run(["scripts/scanned_benchmark.py", "--split", "test", "-n", "8", "--seed0", str(8 * rep)])[0]


def p_scanned(r):
    return {"success": f"{r['success']}/{r['trials']}", "success_rate": round(r["success"] / r["trials"], 3),
            "expert_solvable": r.get("solvable")}


def m_models_task_spec(rep):
    return _run(["scripts/eval_models.py", "task_spec", "--arms", "rule", "claude", "spark", "spark_low"])[0]


def m_models_programs(rep):
    return _run(["scripts/eval_models.py", "programs", "--arms", "rule", "claude", "spark", "spark_low"])[0]


def p_models(r):
    out = {}
    for arm, v in r["results"].get("arms", r["results"]).items():
        if "correct" in v:
            out[arm] = f"{v['correct']}/{v['total']}"
        elif "success" in v:
            out[arm] = f"{v['success']}/{v['trials']}"
    if "fixed_plan" in r["results"]:
        fp = r["results"]["fixed_plan"]
        out["fixed_plan"] = f"{sum(v['success'] for v in fp.values())}/{sum(v['trials'] for v in fp.values())}"
    return out


def m_test_bands(rep):
    from kintsugi.runner import TrialPool, run_batch
    from kintsugi.tasks.splits import split_seeds
    out = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "_headline_test_bands")
    out.mkdir(parents=True)
    rows = {}
    with TrialPool() as pool:
        for t in ("bowl_on_tray", "cup_on_tray"):
            s = run_batch(t, seeds=split_seeds("test", 24), pool=pool, feasibility=True, oracle=True, save_video=False,
                          save_poses=False, save_failures=False, success_sample=0, progress=False,
                          run_id=f"{out.name}/{t}")
            rows[t] = {"success": s["success"], "trials": s["trials"], "oracle": s.get("oracle")}
    (out / "report.json").write_text(json.dumps({"tasks": rows}, indent=1))
    return out / "report.json"


def history_failsearch():
    """The failure-search trend over earlier library states (each round measured at the time; not re-measurable)."""
    from server.app import progress
    return {"rounds": progress().get("search", [])}


HEADLINES = {
    "tasks": dict(title="Current library on 10 named tasks (Studio settings, fresh seeds)", measure=m_tasks,
                  parse=p_tasks, reuse="*_headline_tasks_r*/report.json", cost="cheap", repeatable=True),
    "scanned_repair": dict(title="Scanned household objects, repair set (20 objects x 8)", measure=m_scanned_repair,
                           parse=p_scanned, reuse="*_scanned_bench_repair/report.json", cost="cheap", repeatable=True),
    "models_task_spec": dict(title="Prompt -> task spec, 20 requests (model comparison)", measure=m_models_task_spec,
                             parse=p_models, reuse="*_eval_task_spec/report.json", cost="cheap", repeatable=False),
    "demo_repair": dict(title="Demo repair from the seed library (bowl: top_grasp fails -> new grasp, gated)",
                        measure=m_demo_repair, parse=p_demo_repair, reuse="*_explore_sort_with_bowl/report.json",
                        cost="expensive", repeatable=True),
    "transfer": dict(title="Transfer to an unseen mug: trials with the library vs from scratch", measure=m_transfer,
                     parse=p_transfer, reuse=lambda: _latest_where("*_transfer_mug/report.json", lambda x: ((x.get("retrieval") or {}).get("meta") or {}).get("backend") == "spark"), cost="expensive", repeatable=False),
    "skills": dict(title="New skills from scratch by Spark (push, stand up, open / close drawer)", measure=m_skills,
                   parse=p_skills, reuse=lambda: _latest_where("*_eval_skills/report.json", lambda x: "spark" in x["results"]), cost="expensive", repeatable=True),
    "diagnosis": dict(title="Diagnosis of seed-library failures (Spark, second look with images)", measure=m_diag_seed,
                      parse=p_diag, reuse=lambda: _latest_where("*_investigate/report.json", lambda x: x.get("library") == "seed" and x.get("backend") == "spark" and x.get("second_look", "images") == "images" and x["metrics"].get("diagnosed", 0) >= 15 and not x["run_id"].endswith("200139_investigate")), cost="expensive", repeatable=False),
    "failure_search": dict(title="Failure search on the current library (random / Optuna / Spark, 160 each)",
                           measure=m_failsearch, parse=p_failsearch, reuse=lambda: _latest_where("*_failsearch/report.json", lambda x: len(x.get("strategies", {})) == 3 and not x.get("from_seed") and not x.get("obstacle") and not x.get("templates")),
                           cost="expensive", repeatable=True),
    "models_programs": dict(title="Plan programs, 7 tasks x 8 scenes (model comparison)", measure=m_models_programs,
                            parse=p_models, reuse="*_eval_programs/report.json", cost="expensive", repeatable=False),
    "test_scanned": dict(title="FROZEN TEST: 10 never-repaired scanned objects", measure=m_scanned_test,
                         parse=p_scanned, reuse="*_scanned_bench_test/report.json", cost="final", repeatable=False),
    "test_bands": dict(title="FROZEN TEST: bowl and cup test bands (24 each)", measure=m_test_bands,
                       parse=lambda r: {t: f"{v['success']}/{v['trials']}" for t, v in r["tasks"].items()},
                       reuse="*_headline_test_bands/report.json", cost="final", repeatable=False),
}


def _spread(values):
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) < 2:
        return None
    return {"mean": round(statistics.mean(nums), 3), "sd": round(statistics.stdev(nums), 3), "n": len(nums)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-run the expensive headlines too")
    ap.add_argument("--final", action="store_true", help="also run the frozen test sets (once)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    from kintsugi.version import code_version
    if a.list:
        for k, h in HEADLINES.items():
            p = h["reuse"]() if callable(h["reuse"]) else _latest(h["reuse"])
            print(f"{k:18s} [{h['cost']:9s}] {h['title']}\n{'':31s}last: {_rid(p) if p else '-'}")
        return 0
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_headlines"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    table = {"run_id": run_id, "mode": "final" if a.final else "full" if a.full else "quick", "repeats": a.repeats,
             "code_version": code_version(), "started": time.strftime("%Y-%m-%d %H:%M"), "headlines": {}}
    for key, h in HEADLINES.items():
        if a.only and key not in a.only:
            continue
        if h["cost"] == "final" and not a.final:
            continue
        rerun = bool(a.only) or h["cost"] == "cheap" or a.full or (h["cost"] == "final" and a.final)
        entry = {"title": h["title"], "cost": h["cost"]}
        try:
            if rerun:
                reps = a.repeats if h["repeatable"] else 1
                runs = []
                for rep in range(reps):
                    t0 = time.time()
                    print(f"[headlines] {key} (run {rep + 1}/{reps})", flush=True)
                    path = h["measure"](rep)
                    runs.append({"run": _rid(path), "seconds": round(time.time() - t0), **h["parse"](_read(path))})
                entry.update(status="measured", runs=runs)
                if len(runs) > 1:
                    entry["spread"] = {k: s for k in runs[0] if (s := _spread([r.get(k) for r in runs]))}
            else:
                p = h["reuse"]() if callable(h["reuse"]) else _latest(h["reuse"])
                if p is None:
                    entry.update(status="missing")
                else:
                    entry.update(status="reused", runs=[{"run": _rid(p), "date": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), **h["parse"](_read(p))}])
        except Exception as e:
            entry.update(status="error", error=f"{type(e).__name__}: {str(e)[:600]}")
        table["headlines"][key] = entry
        print(f"[headlines] {key}: {entry['status']}", flush=True)
        (out / "table.json").write_text(json.dumps(table, indent=1, default=str))
    try:
        table["history"] = {"failure_search": history_failsearch()}
    except Exception as e:
        table["history"] = {"error": f"{type(e).__name__}: {e}"}
    table["finished"] = time.strftime("%Y-%m-%d %H:%M")
    (out / "table.json").write_text(json.dumps(table, indent=1, default=str))
    lines = [f"# Kintsugi headlines ({table['mode']}, {table['started']}, code {table['code_version'].get('code_hash')})", ""]
    for k, e in table["headlines"].items():
        lines.append(f"## {e['title']}  [{e['status']}]")
        for r in e.get("runs", []):
            lines.append("- " + ", ".join(f"{kk}: {vv}" for kk, vv in r.items()))
        if e.get("spread"):
            lines.append("- spread: " + ", ".join(f"{kk} {v['mean']} +- {v['sd']} (n={v['n']})" for kk, v in e["spread"].items()))
        if e.get("error"):
            lines.append(f"- error: {e['error']}")
        lines.append("")
    (out / "table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: runs/{run_id}/table.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
