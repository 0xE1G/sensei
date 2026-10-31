"""Run a task many times in parallel and print the success rate and failure labels.

Examples:
    python scripts/run_task.py sort_by_color -n 8
    python scripts/run_task.py bowl_on_tray -n 20 --grasps top_grasp
    python scripts/run_task.py bowl_on_tray -n 20 --grasps rim_pinch --workers 4
    python scripts/run_task.py balls_in_basket -n 10 --no-video --success-sample 0
    python scripts/run_task.py bowl_on_tray -n 16 --split dev            # held-out bowls (seeds 100000+)
    python scripts/run_task.py bowl_on_tray -n 8 --reference --grasps rim_pinch_handwritten
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.runner import default_workers, run_batch  # noqa: E402
from kintsugi.skills.library import LIB_DIR, load_cards  # noqa: E402
from kintsugi.tasks.specs import TASKS  # noqa: E402
from kintsugi.tasks.splits import split_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("-n", type=int, default=8, help="number of trials (one seed each)")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (with --split: index within the split)")
    ap.add_argument("--split", choices=["train", "dev", "test"], default=None,
                    help="draw objects from a frozen split (experiments/splits.json)")
    ap.add_argument("--oracle", action="store_true",
                    help="also run the solvability oracle per scene; failures count only where it succeeds")
    ap.add_argument("--reference", action="store_true",
                    help="also load hand-written reference cards (skill_library/reference/)")
    ap.add_argument("--workers", type=int, default=default_workers(), help="parallel simulations")
    ap.add_argument("--grasps", nargs="+", default=None, help="allowed grasp skills, in order of preference")
    ap.add_argument("--perception", default="sim", choices=["sim", "ground_truth"])
    ap.add_argument("--no-video", action="store_true", help="do not save MP4s")
    ap.add_argument("--no-poses", action="store_true", help="do not save pose logs for the 3D replay")
    ap.add_argument("--no-failures", action="store_true", help="do not save media for failures")
    ap.add_argument("--success-sample", type=int, default=20, help="save media for 1 in N successes (0 = none)")
    ap.add_argument("--set", nargs="*", default=[], help="parameter overrides, e.g. rim_pinch.depth=0.02")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=")
        skill, param = k.split(".")
        overrides.setdefault(skill, {})[param] = float(v)
    seeds = split_seeds(a.split, a.n, a.seed0) if a.split else None
    extra = load_cards(LIB_DIR / "reference") if a.reference else None
    s = run_batch(a.task, n=a.n, seed0=a.seed0, workers=a.workers, perception=a.perception,
                  allowed_grasps=a.grasps, overrides=overrides or None, save_failures=not a.no_failures,
                  success_sample=a.success_sample, save_video=not a.no_video, save_poses=not a.no_poses,
                  seeds=seeds, extra_cards=extra, oracle=a.oracle)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
