"""Does recovery inside a run help? Same scenes with recovery off vs on: success, paired test, time per episode.

    python scripts/measure_recovery.py                   # default task set
    python scripts/measure_recovery.py --retries 2 --n-probe 160
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.runner import ROOT, TrialPool, default_workers, trial_cfg, wilson  # noqa: E402
from kintsugi.search.explore import mcnemar_p  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--n-probe", type=int, default=160, help="random probe scenes (every object type)")
    ap.add_argument("--n-task", type=int, default=24)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--hard", action="store_true",
                    help="harder conditions: slippery heavy objects (friction x0.35-0.7, mass x1.3-2)")
    ap.add_argument("--tasks", nargs="+", default=None, help="only these tasks")
    a = ap.parse_args()
    out = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S_measure_recovery")
    tasks = [("probe", a.n_probe), ("sort_with_bowl", a.n_task), ("stack_by_size", a.n_task),
             ("sort_by_color", a.n_task // 2), ("balls_in_basket", a.n_task // 2), ("cup_on_tray", a.n_task // 2)]
    if a.tasks:
        tasks = [(t, n) for t, n in tasks if t in a.tasks]
    common = dict(save_failures=False, success_sample=0, save_video=False, save_poses=False)
    if a.hard:
        common["perturb"] = {"friction": [0.35, 0.7], "mass": [1.3, 2.0]}
        out = out.with_name(out.name + "_hard")
    report, rows = {"retries": a.retries, "tasks": {}}, []
    with TrialPool(a.workers) as pool:
        for task, n in tasks:
            seeds = list(range(5000, 5000 + n))
            jobs = [(trial_cfg(task, out, tag=f"{task}_off", **common), s) for s in seeds] + \
                   [(trial_cfg(task, out, tag=f"{task}_on", recovery=a.retries, **common), s) for s in seeds]
            res = {(r["tag"], r["seed"]): r for r in pool.map(jobs)}
            off = [res[(f"{task}_off", s)] for s in seeds]
            on = [res[(f"{task}_on", s)] for s in seeds]
            k0, k1 = sum(r["success"] for r in off), sum(r["success"] for r in on)
            p = mcnemar_p([r["success"] for r in off], [r["success"] for r in on])
            t0 = sum(r["wall_time_s"] for r in off) / n
            t1 = sum(r["wall_time_s"] for r in on) / n
            report["tasks"][task] = {"n": n, "off": k0, "on": k1, "off_ci95": wilson(k0, n), "on_ci95": wilson(k1, n),
                                     "p_on_better": p, "time_off_s": round(t0, 2), "time_on_s": round(t1, 2)}
            rows.append(f"{task:<16} {k0:>3}/{n:<4} -> {k1:>3}/{n:<4}  p={p:<8.3g} time/episode {t0:.1f}s -> {t1:.1f}s")
            print(rows[-1], flush=True)
    tot0 = sum(v["off"] for v in report["tasks"].values())
    tot1 = sum(v["on"] for v in report["tasks"].values())
    ntot = sum(v["n"] for v in report["tasks"].values())
    print(f"{'ALL':<16} {tot0:>3}/{ntot:<4} -> {tot1:>3}/{ntot:<4}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"report: {out.relative_to(ROOT)}/report.json")


if __name__ == "__main__":
    main()
