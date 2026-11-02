"""Task specs: the JSON the language model will write from a prompt.

Until the Meta API key is added, these generators stand in for Muse Spark.
The format is exactly what the model will be asked to produce, so swapping the
generator for a model call changes nothing downstream.
"""
from __future__ import annotations

import numpy as np

from .splits import sample_params, which_split

COLOR_CYCLE = ["red", "green", "blue", "yellow", "purple", "orange"]


def sort_by_color(colors=("red", "green", "blue"), per_color=2, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    objs, goal = [], []
    for c in colors:
        objs.append({"id": f"bowl_{c}", "template": "bowl", "role": "target", "fixed": True,
                     "params": {"radius": 0.095, "depth": 0.045, "color": c}})
    for c in colors:
        for i in range(per_color):
            oid = f"block_{c}_{i + 1}"
            objs.append({"id": oid, "template": "block", "role": "item",
                         "params": {"size": float(rng.uniform(0.038, 0.05)), "color": c}})
            goal.append({"pred": "inside", "obj": oid, "target": f"bowl_{c}"})
    return {"name": "sort_by_color", "prompt": "Sort the blocks into the bowls of the same colour.",
            "objects": objs, "goal": goal, "constraints": []}


def balls_in_basket(n=4, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    objs = [{"id": "basket", "template": "bin", "role": "target",
             "params": {"width": 0.22, "depth": 0.18, "wall_height": 0.06, "color": "brown"}}]
    goal = []
    for i in range(n):
        oid = f"ball_{i + 1}"
        objs.append({"id": oid, "template": "ball", "role": "item",
                     "params": {"radius": float(rng.uniform(0.02, 0.03)), "color": COLOR_CYCLE[i % 6]}})
        goal.append({"pred": "inside", "obj": oid, "target": "basket"})
    return {"name": "balls_in_basket", "prompt": "Put the balls in the basket.", "objects": objs, "goal": goal,
            "constraints": []}


def bowl_on_tray(seed=0, radius=None, split=None) -> dict:
    """The bowl's shape comes from the split its seed belongs to (experiments/splits.json)."""
    rng = np.random.default_rng(seed)
    split = split or which_split(seed)
    p = sample_params("bowl", split, rng)
    if radius:
        p["radius"] = radius
    return {"name": "bowl_on_tray", "prompt": "Put the bowl on the tray.", "split": split,
            "objects": [{"id": "tray", "template": "tray", "role": "target",
                         "params": {"width": 0.26, "depth": 0.22, "color": "gray"}},
                        {"id": "bowl", "template": "bowl", "role": "item", "params": {**p, "color": "white"}}],
            "goal": [{"pred": "on", "obj": "bowl", "target": "tray", "upright": True}], "constraints": []}


def cup_on_tray(seed=0, split=None) -> dict:
    rng = np.random.default_rng(seed)
    split = split or which_split(seed)
    return {"name": "cup_on_tray", "prompt": "Put the cup on the tray.", "split": split,
            "objects": [{"id": "tray", "template": "tray", "role": "target",
                         "params": {"width": 0.22, "depth": 0.18, "color": "gray"}},
                        {"id": "cup", "template": "cup", "role": "item",
                         "params": {**sample_params("cup", split, rng), "color": "blue"}}],
            "goal": [{"pred": "on", "obj": "cup", "target": "tray", "upright": True}], "constraints": []}


def single_block(seed=0) -> dict:
    return {"name": "single_block", "prompt": "Put the red block in the bin.",
            "objects": [{"id": "bin", "template": "bin", "role": "target", "params": {"color": "brown"}},
                        {"id": "block", "template": "block", "role": "item", "params": {"size": 0.045, "color": "red"}}],
            "goal": [{"pred": "inside", "obj": "block", "target": "bin"}], "constraints": []}


def sort_with_bowl(seed=0, split=None) -> dict:
    """The demo task: two blocks and a bowl, each onto the tray of its own colour."""
    rng = np.random.default_rng(seed)
    split = split or which_split(seed)
    objs = [{"id": "tray_red", "template": "tray", "role": "target", "params": {"width": 0.15, "depth": 0.14, "color": "red"}},
            {"id": "tray_green", "template": "tray", "role": "target",
             "params": {"width": 0.15, "depth": 0.14, "color": "green"}},
            {"id": "tray_blue", "template": "tray", "role": "target", "params": {"width": 0.25, "depth": 0.22, "color": "blue"}},
            {"id": "block_red", "template": "block", "role": "item",
             "params": {"size": float(rng.uniform(0.038, 0.05)), "color": "red"}},
            {"id": "block_green", "template": "block", "role": "item",
             "params": {"size": float(rng.uniform(0.038, 0.05)), "color": "green"}},
            {"id": "bowl_blue", "template": "bowl", "role": "item",
             "params": {**sample_params("bowl", split, rng), "color": "blue"}}]
    goal = [{"pred": "on", "obj": "block_red", "target": "tray_red"},
            {"pred": "on", "obj": "block_green", "target": "tray_green"},
            {"pred": "on", "obj": "bowl_blue", "target": "tray_blue", "upright": True}]
    return {"name": "sort_with_bowl", "prompt": "Sort the blocks and the bowl onto the trays of the same colour.",
            "split": split, "objects": objs, "goal": goal, "constraints": []}


PROBE_TEMPLATES = ("block", "ball", "cylinder", "bowl", "cup", "bar", "mug")


def region_scenario(seed: int, region: dict) -> dict:
    """An object drawn from a failure-search cluster's region: {"template": t, "params": {k: [lo, hi]}}, or for
    imported objects {"templates": [t1, t2, ...]} (whole objects as the held-out unit).

    Held-out objects come from interleaved bands: the region's first shape parameter is cut into five bands; train
    seeds draw from bands 1, 3 and 5, dev and test seeds from bands 2 and 4. Gate objects are never tuned on, and lie
    between tuned values (a card's preconditions cover the span of the shapes it was tuned on).
    Position is left to the placer.
    """
    from ..sim.templates import RANGES
    rng = np.random.default_rng(seed)
    if region.get("templates"):
        # Imported objects have no shape parameters: held-out objects are whole objects, interleaved the same way
        # (train seeds draw from the 1st, 3rd, ... object, dev/test seeds from the 2nd, 4th, ...). One object only:
        # held-out differs by placement and physics, not shape.
        ts = list(region["templates"])
        pick = ts[0::2] if which_split(seed) == "train" or len(ts) < 2 else ts[1::2]
        # one item id for every object, so a repair can address "the item" whichever object a seed draws
        return {"template": str(pick[rng.integers(len(pick))]), "params": dict(region.get("params") or {}), "id": "item"}
    t = region["template"]
    bounds = {k: list(region.get("params", {}).get(k, RANGES[t][k])) for k in RANGES[t]}
    key = next((k for k in RANGES[t] if k not in ("mass", "friction")), None)
    if key is not None:                     # an imported model has no shape parameter: no bands (placement, physics)
        lo, hi = bounds[key]
        band = rng.choice([0, 2, 4] if which_split(seed) == "train" else [1, 3])
        bounds[key] = [lo + band * (hi - lo) / 5, lo + (band + 1) * (hi - lo) / 5]
    scn = {"template": t, "params": {k: float(rng.uniform(a, b)) for k, (a, b) in bounds.items()}}
    if "obstacle" in region:                 # a bottle near the item, as in the failure-search cluster
        ob = region["obstacle"]
        scn["pos"] = [float(rng.uniform(0.40, 0.60)), float(rng.uniform(-0.20, 0.20))]
        scn["yaw"] = float(rng.uniform(-np.pi / 2, np.pi / 2))
        scn["obstacle"] = {k: float(rng.uniform(*ob[k])) for k in ("dist", "radius", "height")}
        scn["obstacle"]["angle"] = float(rng.uniform(-np.pi, np.pi))
    return scn


def probe(seed=0, scenario=None, region=None) -> dict:
    """One item, chosen by the failure search, to its target: a tray (a bin for balls, which roll off trays).

    scenario = {"template": t, "params": {...}, "pos": [x, y], "yaw": rad}; the item is pre-placed and the target is laid
    out around it. Without a scenario the item is drawn uniformly (any probe template, full tested ranges).
    """
    from ..sim.templates import RANGES
    rng = np.random.default_rng(seed)
    if scenario is None and region is not None:
        scenario = region_scenario(seed, region)
    if scenario is None:
        t = PROBE_TEMPLATES[rng.integers(len(PROBE_TEMPLATES))]
        scenario = {"template": t, "params": {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RANGES[t].items()}}
    t = scenario["template"]
    oid = scenario.get("id", t)
    if t == "ball":
        target = {"id": "bin", "template": "bin", "role": "target",
                  "params": {"width": 0.22, "depth": 0.18, "wall_height": 0.06, "color": "brown"}}
        goal = {"pred": "inside", "obj": oid, "target": "bin"}
    else:
        target = {"id": "tray", "template": "tray", "role": "target",
                  "params": {"width": 0.26, "depth": 0.22, "color": "gray"}}
        goal = {"pred": "on", "obj": oid, "target": "tray", "upright": t in ("bowl", "cup", "cylinder", "mug")}
    task = {"name": "probe", "prompt": f"Put the {t} on the {target['id']}.", "scenario": scenario,
            "split": which_split(seed),
            "objects": [target, {"id": oid, "template": t, "role": "item",
                                 "params": {**scenario["params"], "color": "orange"}}],
            "goal": [goal], "constraints": []}
    if scenario.get("pos"):
        task["layout_fixed"] = {oid: (float(scenario["pos"][0]), float(scenario["pos"][1]),
                                    float(scenario.get("yaw", 0.0)))}
    ob = scenario.get("obstacle")
    if ob and scenario.get("pos"):
        # A bottle the robot must leave alone, at a given distance and direction from the item.
        bx = scenario["pos"][0] + ob["dist"] * np.cos(ob["angle"])
        by = scenario["pos"][1] + ob["dist"] * np.sin(ob["angle"])
        task["objects"].append({"id": "bottle", "template": "cylinder", "role": "item",
                                "params": {"radius": ob["radius"], "height": ob["height"], "color": "green"}})
        task["layout_fixed"]["bottle"] = (float(bx), float(by), 0.0)
        task["constraints"] = [{"pred": "stays_upright", "obj": "bottle"}, {"pred": "not_disturbed", "obj": "bottle"}]
        task["obstacles"] = ["bottle"]
        task["prompt"] = task["prompt"][:-1] + " without knocking over the bottle."
    return task


def stack_by_size(seed=0) -> dict:
    """Three blocks of clearly different sizes, stacked largest at the bottom. Order matters: the fixed planner
    places items in distance order, so it can bury a block it still has to move."""
    rng = np.random.default_rng(seed)
    sizes = {"block_large": float(rng.uniform(0.058, 0.064)), "block_medium": float(rng.uniform(0.046, 0.050)),
             "block_small": float(rng.uniform(0.034, 0.038))}
    colors = {"block_large": "blue", "block_medium": "green", "block_small": "red"}
    objs = [{"id": i, "template": "block", "role": "item", "params": {"size": s, "color": colors[i]}}
            for i, s in sizes.items()]
    goal = [{"pred": "on", "obj": "block_medium", "target": "block_large"},
            {"pred": "on", "obj": "block_small", "target": "block_medium"}]
    return {"name": "stack_by_size", "prompt": "Stack the blocks, largest at the bottom.", "objects": objs,
            "goal": goal, "constraints": []}


def slide_plate(seed=0) -> dict:
    """A plate (wide, thin: no grasp fits) onto a coloured zone 15-25 cm away. Needs a push."""
    rng = np.random.default_rng(seed)
    p = {"radius": float(rng.uniform(0.065, 0.085)), "thickness": float(rng.uniform(0.008, 0.014)),
         "mass": float(rng.uniform(0.06, 0.2)), "friction": float(rng.uniform(0.3, 0.7)), "color": "white"}
    for _ in range(200):
        r, th = rng.uniform(0.42, 0.58), rng.uniform(-0.45, 0.45)
        px, py = r * np.cos(th), r * np.sin(th)
        d, phi = rng.uniform(0.15, 0.25), rng.uniform(-np.pi, np.pi)
        zx, zy = px + d * np.cos(phi), py + d * np.sin(phi)
        if 0.36 <= zx <= 0.66 and abs(zy) <= 0.30 and 0.38 <= np.hypot(zx, zy) <= 0.66:
            break
    return {"name": "slide_plate", "prompt": "Slide the plate onto the blue zone.",
            "objects": [{"id": "zone", "template": "zone", "role": "target",
                         "params": {"width": 0.16, "depth": 0.16, "color": "blue"}},
                        {"id": "plate", "template": "plate", "role": "item", "params": p}],
            "goal": [{"pred": "on", "obj": "plate", "target": "zone"}], "constraints": [],
            "layout_fixed": {"plate": (float(px), float(py), 0.0), "zone": (float(zx), float(zy), 0.0)}}


def isolate(task: dict, obj: str) -> dict:
    """The sub-task of one item: the same object (same parameters), its target, and the obstacles the task's
    constraints protect (they are part of the step's difficulty); other items are removed.
    The explorer repairs a failing step here, where trials are cheap, then verifies on the full task."""
    goal = [g for g in task["goal"] if g["obj"] == obj]
    keep = {obj} | {g["target"] for g in goal} | set(task.get("obstacles", []))
    constraints = [c for c in task.get("constraints", []) if c["obj"] in keep]
    out = {**task, "name": f"{task['name']}[{obj}]", "objects": [o for o in task["objects"] if o["id"] in keep],
           "goal": goal, "constraints": constraints, "isolated_from": task["name"]}
    if "layout_fixed" in task:
        out["layout_fixed"] = {k: v for k, v in task["layout_fixed"].items() if k in keep}
    return out


def open_drawer(seed=0) -> dict:
    """A cabinet with a drawer, front facing the robot; open the drawer at least 8 cm."""
    rng = np.random.default_rng(seed)
    p = {"width": float(rng.uniform(0.16, 0.26)), "depth": float(rng.uniform(0.14, 0.20)),
         "height": float(rng.uniform(0.08, 0.12)), "travel": float(rng.uniform(0.11, 0.15)),
         "damping": float(rng.uniform(0.5, 3.0)), "color": "white"}
    return {"name": "open_drawer", "prompt": "Open the drawer.",
            "objects": [{"id": "drawer", "template": "drawer", "role": "item", "params": p}],
            "goal": [{"pred": "joint", "obj": "drawer", "op": ">=", "value": 0.08}], "constraints": []}


def open_door(seed=0) -> dict:
    """A cabinet with a hinged door (hinge on either side), front facing the robot; open it at least 50 degrees."""
    rng = np.random.default_rng(seed)
    p = {"width": float(rng.uniform(0.12, 0.20)), "depth": float(rng.uniform(0.12, 0.18)),
         "height": float(rng.uniform(0.10, 0.16)), "hinge_side": float(rng.integers(0, 2)),
         "damping": float(rng.uniform(0.02, 0.2)), "color": "white"}
    return {"name": "open_door", "prompt": "Open the door.",
            "objects": [{"id": "door", "template": "door", "role": "item", "params": p}],
            "goal": [{"pred": "joint", "obj": "door", "op": ">=", "value": 50.0}], "constraints": []}


def close_drawer(seed=0) -> dict:
    """The drawer starts fully open; push it closed (at most 1 cm open)."""
    t = open_drawer(seed)
    t["objects"][0]["params"]["open"] = 1.0
    return {**t, "name": "close_drawer", "prompt": "Close the drawer.",
            "goal": [{"pred": "joint", "obj": "drawer", "op": "<=", "value": 0.01}]}


def close_door(seed=0) -> dict:
    """The door starts open (about 80 degrees); swing it closed (at most 10 degrees open)."""
    t = open_door(seed)
    t["objects"][0]["params"]["open"] = 0.85
    return {**t, "name": "close_door", "prompt": "Close the door.",
            "goal": [{"pred": "joint", "obj": "door", "op": "<=", "value": 10.0}]}


def stand_up(seed=0) -> dict:
    """A bottle (cylinder) lying on its side, anywhere in reach, any direction; stand it back up on one of its ends."""
    rng = np.random.default_rng(seed)
    p = {"radius": float(rng.uniform(0.018, 0.032)), "height": float(rng.uniform(0.08, 0.16)),
         "mass": float(rng.uniform(0.05, 0.3)), "friction": float(rng.uniform(0.6, 1.2)), "color": "green"}
    return {"name": "stand_up", "prompt": "Stand the bottle back up.",
            "objects": [{"id": "bottle", "template": "cylinder", "role": "item", "pose": "lying", "params": p}],
            "goal": [{"pred": "upright", "obj": "bottle"}], "constraints": []}


TASKS = {"sort_by_color": sort_by_color, "balls_in_basket": balls_in_basket, "bowl_on_tray": bowl_on_tray,
         "cup_on_tray": cup_on_tray, "single_block": single_block, "sort_with_bowl": sort_with_bowl,
         "probe": probe, "stack_by_size": stack_by_size, "slide_plate": slide_plate,
         "open_drawer": open_drawer, "open_door": open_door, "close_drawer": close_drawer, "close_door": close_door,
         "stand_up": stand_up}


def make_task(name: str, seed: int = 0, isolate_obj: str | None = None, **kw) -> dict:
    """name "spec" = a task spec given as task_spec=... (e.g. written from a typed request); the seed then only
    varies the placement."""
    if name == "spec":
        task = dict(kw["task_spec"])
        task.pop("program", None)
    elif name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Known: {sorted(TASKS)} or 'spec'")
    else:
        task = TASKS[name](seed=seed, **kw)
    return isolate(task, isolate_obj) if isolate_obj else task
