"""Parallel trial runner. Each trial = one episode with its own seed, run in its own process.

Saved per run (runs/<run_id>/):
  summary.json                  success rates, failure label counts
  trials/<seed>.json            one small record per trial (always)
  media/<seed>_poses.json.gz    poses for the Three.js replay   (failures, + sampled successes)
  media/<seed>.mp4              video                           (failures, + sampled successes)
"""
from __future__ import annotations

import collections
import json
import math
import multiprocessing as mp
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_DEFAULT_WORKERS = 8      # measured: above ~8 parallel renderers throughput drops on a laptop GPU


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, MAX_DEFAULT_WORKERS))


def _pin_threads():
    """One BLAS thread per simulation process; many processes x many threads is slower."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")


class TrialPool:
    """Worker processes kept alive across batches (spawning and importing MuJoCo is slow on Windows).

    map(jobs) runs (cfg, seed) jobs and yields their small result dicts as they finish.
    """

    def __init__(self, workers: int | None = None):
        self.workers = workers or default_workers()
        self.pool = None
        if self.workers > 1:
            _pin_threads()
            self.pool = mp.get_context("spawn").Pool(self.workers)

    def map(self, jobs):
        if self.pool is None:
            yield from map(_trial, jobs)
        else:
            yield from self.pool.imap_unordered(_trial, jobs)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _trial(args):
    from .executor import run_episode
    from .tasks.specs import make_task
    from .oracle import run_oracle
    from .version import code_version, config_hash
    cfg, seed = args
    task = cfg.get("task_spec") or make_task(cfg["task"], seed=seed, **cfg.get("task_kwargs", {}))
    want_media = cfg["save_failures"] or cfg["success_sample"] > 0
    try:
        record, rec = run_episode(task, seed=seed, perception_mode=cfg["perception"],
                              allowed_grasps=cfg.get("allowed_grasps"), overrides=cfg.get("overrides"),
                              record_poses=want_media and cfg["save_poses"],
                              record_video=want_media and cfg["save_video"], perturb=cfg.get("perturb"),
                              extra_cards=cfg.get("extra_cards"), library=cfg.get("library"),
                              performance=cfg.get("performance"), positions=cfg.get("positions"),
                              patches=cfg.get("patches"))
    except Exception as e:                       # a broken scene is a result, not a crash
        record, rec = {"task": cfg["task"], "seed": seed, "success": False, "goals_met": 0,
                       "goals_total": len(task["goal"]), "error": f"{type(e).__name__}: {e}",
                       "subtasks": [{"label": "episode_error"}], "wall_time_s": 0.0}, None
    # oracle=True: every scene; oracle="on_failure": only where the robot failed (the only case where it matters)
    if cfg.get("oracle") and not (cfg["oracle"] == "on_failure" and record["success"]):
        record["oracle"] = run_oracle(task, seed, perturb=cfg.get("perturb"))
    record["code_version"] = code_version()
    record["config_hash"] = config_hash({k: v for k, v in cfg.items() if k not in ("out_dir", "tag")})
    out = pathlib.Path(cfg["out_dir"])
    (out / "trials").mkdir(parents=True, exist_ok=True)
    keep = (not record["success"] and cfg["save_failures"]) or (
        record["success"] and cfg["success_sample"] > 0 and seed % cfg["success_sample"] == 0)
    if rec is not None and keep:
        record["media"] = rec.save(out / "media", f"{seed:05d}", save_poses=cfg["save_poses"],
                                   save_video=cfg["save_video"])
    record["tag"] = cfg.get("tag")
    name = f"{cfg['tag']}_{seed:06d}" if cfg.get("tag") else f"{seed:05d}"
    (out / "trials" / f"{name}.json").write_text(json.dumps(record, indent=1, default=float))
    return {"seed": seed, "tag": cfg.get("tag"), "success": record["success"], "goals_met": record["goals_met"],
            "goals_total": record["goals_total"], "labels": [s["label"] for s in record["subtasks"] if s.get("label")],
            "subtask_ok": {s["obj"]: s["ok"] for s in record["subtasks"] if "obj" in s},
            "grasp": {s["obj"]: s["trace"][0]["skill"] for s in record["subtasks"] if s.get("trace")},
            "oracle": ({k: record["oracle"][k] for k in ("success", "subtask_ok")} if record.get("oracle") else None),
            "wall_time_s": record["wall_time_s"]}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def trial_cfg(task: str, out_dir, perception: str = "sim", allowed_grasps=None, overrides=None,
              save_failures=True, success_sample=20, save_video=True, save_poses=True, task_kwargs=None,
              perturb=None, extra_cards=None, tag=None, library=None, oracle=False, task_spec=None,
              performance=None, positions=None, patches=None) -> dict:
    return dict(task=task, perception=perception, allowed_grasps=allowed_grasps, overrides=overrides,
                save_failures=save_failures, success_sample=success_sample, save_video=save_video,
                save_poses=save_poses, out_dir=str(out_dir), task_kwargs=task_kwargs or {}, perturb=perturb,
                extra_cards=extra_cards, tag=tag, library=library, oracle=oracle, task_spec=task_spec,
                performance=performance, positions=positions, patches=patches)


def run_batch(task: str, n: int = 8, seed0: int = 0, workers: int | None = None, perception: str = "sim",
              allowed_grasps=None, overrides=None, save_failures=True, success_sample=20, save_video=True,
              save_poses=True, run_id: str | None = None, task_kwargs=None, perturb=None, progress=True,
              extra_cards=None, seeds=None, pool: TrialPool | None = None, library=None, oracle=False,
              task_spec=None) -> dict:
    """oracle=True also runs the solvability oracle on every scene; task_spec runs a given spec (e.g. from a prompt)
    instead of building the named task per seed (object placement still varies with the seed)."""
    workers = pool.workers if pool else (workers or default_workers())
    run_id = run_id or time.strftime(f"%Y%m%d-%H%M%S_{task}")
    out_dir = ROOT / "runs" / run_id
    cfg = trial_cfg(task, out_dir, perception, allowed_grasps, overrides, save_failures, success_sample,
                    save_video, save_poses, task_kwargs, perturb, extra_cards, library=library, oracle=oracle,
                    task_spec=task_spec)
    seeds = list(seeds) if seeds is not None else list(range(seed0, seed0 + n))
    n = len(seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**cfg, "n": n, "seeds": seeds, "workers": workers}, indent=1))
    t0 = time.time()
    results = []
    own = pool is None
    pool = pool or TrialPool(workers)
    try:
        for r in pool.map([(cfg, s) for s in seeds]):
            results.append(r)
            if progress:
                _progress(results, n)
    finally:
        if own:
            pool.close()
    k = sum(r["success"] for r in results)
    labels = collections.Counter(l for r in results for l in r["labels"])
    gm = sum(r["goals_met"] for r in results)
    gt = sum(r["goals_total"] for r in results)
    summary = {"run_id": run_id, "task": task, "trials": n, "workers": workers, "success": k, "seeds": seeds,
               "success_rate": round(k / n, 3), "success_ci95": wilson(k, n),
               "goal_rate": round(gm / max(gt, 1), 3), "goal_ci95": wilson(gm, gt),
               "failure_labels": dict(labels.most_common()), "wall_time_s": round(time.time() - t0, 1),
               "trials_per_min": round(60 * n / max(time.time() - t0, 1e-6), 1)}
    if oracle:
        from .oracle import valid_failure
        checked = [r for r in results if r["oracle"]]
        summary["oracle"] = {"checked": len(checked), "solved": sum(r["oracle"]["success"] for r in checked),
                             "valid_failures": sum(valid_failure(r) for r in results),
                             "unsolvable": sum(not r["oracle"]["success"] for r in checked)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _progress(results, n):
    k = sum(r["success"] for r in results)
    print(f"  {len(results)}/{n} trials  success {k}/{len(results)}", flush=True)
