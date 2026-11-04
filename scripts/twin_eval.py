"""Measure the photo -> digital-twin path on rendered scenes whose true layout is known.

    python scripts/twin_eval.py [--backend spark] [-n 8] [--no-sam] [--photos <dir>]

Each case: a named task's scene (fixed seed), rendered from the photographer's place (behind the robot, robot hidden,
two camera heights), sent through kintsugi.twin.build exactly like a user's photo, then compared with the truth:
objects found (template + colour), position error after aligning the layout centres (the twin centres the layout in
the reach by design), and whether left/right and front/behind relations between objects are kept. --photos also runs
every image in a folder (real photos: no truth, the report keeps the twin for a look). Writes runs/<id>_twin_eval/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROBOT_BODIES = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger",
                "right_finger"}
CASES = [("sort_with_bowl", 3), ("sort_by_color", 5), ("balls_in_basket", 2), ("bowl_on_tray", 7),
         ("cup_on_tray", 4), ("stack_by_size", 6), ("slide_plate", 1), ("sort_with_bowl", 11)]
VIEWS = {"oblique": 40, "high": 65}           # camera elevation above the table, degrees


def render_case(task_name, seed, elevation, out: pathlib.Path):
    import mujoco
    from PIL import Image
    from kintsugi.sim.scene import build_scene
    from kintsugi.tasks import specs
    task = getattr(specs, task_name)(seed=seed)
    sc = build_scene(task, seed=seed)
    m, d = sc.model, sc.data
    mujoco.mj_forward(m, d)
    for g in range(m.ngeom):                       # the photo is of a table without the robot
        if m.body(m.geom_bodyid[g]).name in ROBOT_BODIES:
            m.geom_rgba[g, 3] = 0.0
    r = mujoco.Renderer(m, 960, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.5, 0.0, 0.02]
    cam.distance = 1.05 if elevation < 50 else 0.95
    cam.azimuth = 0.0                              # looking along +x: from the robot's side, like the photographer
    cam.elevation = -float(elevation)
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(out, quality=92)
    r.close()
    truth = {oid: (float(d.xpos[m.body(oid).id][0]), float(d.xpos[m.body(oid).id][1])) for oid in sc.objects}
    kinds = {o["id"]: (o["template"], (o.get("params") or {}).get("color")) for o in task["objects"]}
    return truth, kinds


def score(report, truth, kinds):
    """Objects found + layout error (centres aligned) + kept relations."""
    task = report.get("task") or {}
    fixed = task.get("layout_fixed") or {}
    objs = {o["id"]: o for o in task.get("objects", [])}
    # match twin objects to true objects by template + colour first, then by position
    tw = {i: (fixed[i][0], fixed[i][1]) for i in fixed}
    if not tw or not truth:
        return {"found": 0, "objects": len(truth)}
    tc = np.mean(list(tw.values()), axis=0)
    gc = np.mean(list(truth.values()), axis=0)
    pairs, used = {}, set()
    cand = []
    for i, p in tw.items():
        t_i = (objs[i]["template"], (objs[i].get("params") or {}).get("color"))
        for g, q in truth.items():
            same = (t_i[0] == kinds[g][0]) + (t_i[1] == kinds[g][1])
            dist = math.dist(np.array(p) - tc, np.array(q) - gc)
            cand.append((-same, dist, i, g))
    for _, dist, i, g in sorted(cand):
        if i not in pairs and g not in used:
            pairs[i] = g
            used.add(g)
    errs = [math.dist(np.array(tw[i]) - tc, np.array(truth[g]) - gc) for i, g in pairs.items()]
    tmpl_ok = sum(objs[i]["template"] == kinds[g][0] for i, g in pairs.items())
    col_ok = sum((objs[i].get("params") or {}).get("color") == kinds[g][1] for i, g in pairs.items() if kinds[g][1])
    col_n = sum(1 for g in pairs.values() if kinds[g][1])
    rel_ok = rel_n = 0
    ids = list(pairs)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            gi, gj = truth[pairs[i]], truth[pairs[j]]
            for ax in (0, 1):
                if abs(gi[ax] - gj[ax]) > 0.05:
                    rel_n += 1
                    rel_ok += (np.sign(tw[i][ax] - tw[j][ax]) == np.sign(gi[ax] - gj[ax]))
    return {"objects": len(truth), "found": len(tw), "template_ok": tmpl_ok, "colour_ok": col_ok, "colour_n": col_n,
            "pos_err_cm_mean": round(100 * float(np.mean(errs)), 1), "pos_err_cm_max": round(100 * float(np.max(errs)), 1),
            "relations_ok": int(rel_ok), "relations": int(rel_n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="spark")
    ap.add_argument("-n", type=int, default=len(CASES))
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--photos", default=None, help="folder of real photos (no truth; twins kept for a look)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from kintsugi import twin
    out = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S_twin_eval")
    out.mkdir(parents=True)
    jobs = []
    for task_name, seed in CASES[:args.n]:
        for view, elev in VIEWS.items():
            d = out / f"{task_name}_{seed}_{view}"
            d.mkdir()
            truth, kinds = render_case(task_name, seed, elev, d / "render.jpg")
            jobs.append({"case": d.name, "dir": d, "image": d / "render.jpg", "truth": truth, "kinds": kinds,
                         "elevation": elev})
    for p in sorted(pathlib.Path(args.photos).glob("*")) if args.photos else []:
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            d = out / f"photo_{p.stem}"
            d.mkdir()
            jobs.append({"case": d.name, "dir": d, "image": p, "truth": None, "kinds": None})

    def one(j):
        t0 = time.time()
        try:
            rep = twin.build(j["image"], "", backend=args.backend, out_dir=j["dir"], use_sam=not args.no_sam)
        except Exception as e:
            return {"case": j["case"], "error": f"{type(e).__name__}: {e}"}
        row = {"case": j["case"], "status": rep.get("status"), "seconds": round(time.time() - t0, 1),
               "sam_calls": rep.get("twin", {}).get("sam_calls"), "notes": rep.get("twin", {}).get("notes"),
               "prompt": ((rep.get("task") or {}).get("prompt")), "spark_latency_s": rep.get("meta", {}).get("latency_s"),
               "sam_boxes": sum(d.get("source") == "sam" for d in rep.get("twin", {}).get("detections", []))}
        if j["truth"] is not None and rep.get("status") == "ok":
            row.update(score(rep, j["truth"], j["kinds"]))
        print(json.dumps(row), flush=True)
        return row

    with cf.ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    scored = [r for r in rows if "pos_err_cm_mean" in r]
    summary = {"cases": len(rows), "ok": sum(r.get("status") == "ok" for r in rows), "backend": args.backend,
               "sam": not args.no_sam}
    if scored:
        tot = lambda k: sum(r[k] for r in scored)
        summary.update(objects=tot("objects"), found=tot("found"), template_ok=tot("template_ok"),
                       colour_ok=tot("colour_ok"), colour_n=tot("colour_n"),
                       pos_err_cm_mean=round(float(np.mean([r["pos_err_cm_mean"] for r in scored])), 1),
                       pos_err_cm_median_case=round(float(np.median([r["pos_err_cm_mean"] for r in scored])), 1),
                       relations_ok=tot("relations_ok"), relations=tot("relations"),
                       by_view={v: round(float(np.mean([r["pos_err_cm_mean"] for r in scored if r["case"].endswith(v)])), 1)
                                for v in VIEWS if any(r["case"].endswith(v) for r in scored)})
    (out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
    print("SUMMARY", json.dumps(summary))
    print("report:", out / "report.json")


if __name__ == "__main__":
    main()
