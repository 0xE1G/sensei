"""Unseen-object benchmark on the scanned objects: each loose object alone, put on a tray, with the current library.

Usage:  python scripts/fetch_assets.py --scanned            # once
        python scripts/scanned_benchmark.py [-n 8] [--objects mug_ace bowl_black ...] [--library seed|current]
Per object: perception centre error vs the true mesh, success, expert-solvable scenes (oracle), failure labels and the
grasp skill used. Report: runs/<id>_scanned_bench/report.json. None of these objects was used to build or tune a skill.
"""
import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from kintsugi.perception.perceive import Perception  # noqa: E402
from kintsugi.runner import TrialPool, run_batch  # noqa: E402
from kintsugi.sim.assets import CATALOGUE, assets  # noqa: E402
from kintsugi.sim.scene import build_scene  # noqa: E402
from kintsugi.sim.templates import FIXED_BY_DEFAULT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def spec(template: str) -> dict:
    return {"name": f"scanned_{template}", "prompt": f"Put the {template.replace('_', ' ')} on the tray.",
            "constraints": [], "goal": [{"pred": "on", "obj": "obj", "target": "tray"}],
            "objects": [{"id": "obj", "template": template, "role": "item", "params": {}},
                        {"id": "tray", "template": "tray", "role": "target", "params": {}}]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--objects", nargs="*")
    ap.add_argument("--library", choices=["current", "seed"], default="current")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    catalogue = json.loads(CATALOGUE.read_text())["objects"] if CATALOGUE.exists() else {}
    names = a.objects or [k for k in catalogue if k in assets() and k not in FIXED_BY_DEFAULT]
    if not names:
        print("No scanned objects found: run  python scripts/fetch_assets.py --scanned")
        return 1
    library = None
    if a.library == "seed":
        from kintsugi.skills.library import seed_library
        library = seed_library()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_scanned_bench"
    pool, rows, t0 = TrialPool(a.workers), {}, time.time()
    for t in names:
        sc = build_scene(spec(t), seed=0)
        ps, pg = Perception(sc, "sim").perceive(), Perception(sc, "ground_truth").perceive()
        err = float(np.linalg.norm(np.subtract(ps["obj"]["center"], pg["obj"]["center"]))) if ps.get("obj") else None
        s = run_batch("probe", n=a.n, pool=pool, task_spec=spec(t), library=library, oracle=True, save_video=False,
                      save_poses=True, success_sample=a.n, progress=False, run_id=f"{run_id}/{t}")
        labels, grasps = collections.Counter(), collections.Counter()
        for f in sorted((ROOT / "runs" / run_id / t / "trials").glob("*.json")):
            r = json.loads(f.read_text())
            for st in r.get("subtasks", []):
                g = next((x["skill"] for x in st.get("trace", []) if x["skill"] not in ("lift", "place_in", "drop_in")), None)
                grasps[g or "-"] += 1
                if st.get("label") and not r["success"]:
                    labels[st["label"]] += 1
        o = s.get("oracle") or {}
        rows[t] = {"success": s["success"], "trials": a.n, "solvable": o.get("solved"),
                   "valid_failures": o.get("valid_failures"),
                   "centre_err_mm": None if err is None else round(1000 * err, 1),
                   "hollow": (ps.get("obj") or {}).get("hollow"), "labels": dict(labels), "grasps": dict(grasps),
                   "size_cm": [round(100 * assets()[t]["width"], 1), round(100 * assets()[t]["height"], 1)]}
        print(f"{t:16s} {s['success']}/{a.n}  solvable {o.get('solved')}  err {rows[t]['centre_err_mm']} mm  "
              f"{dict(labels) or ''}", flush=True)
    k = sum(r["success"] for r in rows.values())
    n = sum(r["trials"] for r in rows.values())
    solv = sum(r["solvable"] or 0 for r in rows.values())
    report = {"run_id": run_id, "library": a.library, "objects": rows, "success": k, "trials": n, "solvable": solv,
              "seconds": round(time.time() - t0)}
    out = ROOT / "runs" / run_id / "report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\n{k}/{n} succeed ({solv} solvable by the expert), {len(rows)} objects, {report['seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
