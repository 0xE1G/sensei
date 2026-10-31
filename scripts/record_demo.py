"""Record the demo runs with pose logs for the web app's 3D replay (runs/demo/...).

    python scripts/record_demo.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.composites import load_composites  # noqa: E402
from kintsugi.runner import TrialPool, run_batch  # noqa: E402
from kintsugi.skills.library import seed_library  # noqa: E402
from kintsugi.tasks.specs import make_task  # noqa: E402

PUSH_PROGRAM = """plate = find_one(kind='plate')
zone = find_one(kind='zone')
if not pick(plate) and has_skill('push'):
    push(plate, zone)"""

if __name__ == "__main__":
    media = dict(save_failures=True, success_sample=1, save_video=False, save_poses=True, progress=False)
    stack_prog = next(iter(load_composites().values()))["program"]
    runs = [
        ("demo/1_bowl_before_repair", "sort_with_bowl", dict(library=seed_library(), performance={}, patches={})),
        ("demo/2_bowl_after_repair", "sort_with_bowl", {}),
        ("demo/3_stack_program", "stack_by_size", dict(task_spec={**make_task("stack_by_size", seed=0),
                                                                  "program": stack_prog, "program_source": "composite"})),
        ("demo/4_plate_created_push", "slide_plate", dict(task_spec={**make_task("slide_plate", seed=1),
                                                                     "program": PUSH_PROGRAM})),
        ("demo/5_mug_transfer", "probe", dict(task_kwargs={"region": {"template": "mug", "params": {}}})),
        ("demo/6_bar_axis_grasp", "probe", dict(task_kwargs={"region": {"template": "bar", "params": {}}})),
    ]
    with TrialPool() as pool:
        for run_id, task, kw in runs:
            s = run_batch(task, seeds=[1, 2, 3], run_id=run_id, pool=pool, **media, **kw)
            print(f"{run_id}: {s['success']}/{s['trials']}", flush=True)
