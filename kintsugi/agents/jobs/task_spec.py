"""Job "task_spec": a typed request -> validated task spec JSON, a clarifying question, or a refusal."""
from __future__ import annotations

import re

from ...sim.templates import ARTICULATED, COLORS, DEFAULTS, RANGES
from ..llm import Job, load_prompt, register

CONTAINERS_INSIDE = {"bowl", "cup", "bin"}
SURFACES_ON = {"tray", "zone"}
STACKABLE = {"block", "bar", "cylinder"}      # "on" another object = stacking
CONSTRAINTS = ("not_disturbed", "stays_upright")
RELATION_PREDS = ("left_of", "right_of", "in_front_of", "behind")    # seen from the robot
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

SCHEMA = {
    "type": "object",
    "required": ["status"],
    "properties": {
        "status": {"enum": ["ok", "clarify", "refuse"]},
        "question": {"type": "string"},
        "reason": {"type": "string"},
        "task": {
            "type": "object",
            "required": ["name", "objects", "goal"],
            "properties": {
                "name": {"type": "string"},
                "prompt": {"type": "string"},
                "objects": {"type": "array", "minItems": 1, "maxItems": 10, "items": {
                    "type": "object", "required": ["id", "template"],
                    "properties": {"id": {"type": "string"}, "template": {"enum": sorted(DEFAULTS)},
                                   "role": {"enum": ["item", "target", "obstacle"]},
                                   "params": {"type": "object"}, "fixed": {"type": "boolean"},
                                   "pose": {"enum": ["upright", "lying"]}}}},
                "goal": {"type": "array", "items": {
                    "type": "object", "required": ["pred", "obj"],
                    "properties": {"pred": {"enum": ["inside", "on", "joint", "distance", "height", "upright",
                                                     "left_of", "right_of", "in_front_of", "behind"]},
                                   "obj": {"type": "string"}, "target": {"type": "string"},
                                   "upright": {"type": "boolean"}, "part": {"type": "string"},
                                   "op": {"enum": [">=", "<="]}, "value": {"type": "number"}}}},
                "constraints": {"type": "array", "items": {
                    "type": "object", "required": ["pred", "obj"],
                    "properties": {"pred": {"enum": list(CONSTRAINTS)}, "obj": {"type": "string"}}}},
                "order": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def _imported_container(template: str) -> bool:
    """A hollow imported model (a basket, a bin, a scanned bowl): a valid target for 'inside'."""
    from ...sim.assets import assets
    a = assets().get(template)
    return bool(a and a["meta"].get("hollow") and not a.get("joint"))


def refresh_templates():
    """Templates registered after start-up (downloaded objects) become valid in the schema (list updated in place)."""
    enum = SCHEMA["properties"]["task"]["properties"]["objects"]["items"]["properties"]["template"]["enum"]
    enum[:] = sorted(DEFAULTS)


def catalog_text() -> str:
    lines = []
    from ...sim.assets import assets
    for t in sorted(DEFAULTS):
        rng = ", ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in RANGES[t].items())
        extra = ""
        if t in assets():                          # an imported model: fixed size, described by its file
            a = assets()[t]
            extra = (f" (imported model, {100 * a['width']:.0f} x {100 * a['height']:.0f} cm"
                     + (f", {DEFAULTS[t]['color']} (fixed by its texture)" if DEFAULTS[t].get("color") else "")
                     + (f", moving part '{ARTICULATED[t]['part']}'" if t in ARTICULATED else "")
                     + (", hollow container" if _imported_container(t) else "")
                     + (f": {a['meta']['description']}" if a["meta"].get("description") else "") + ")")
        lines.append(f"- {t}: {rng}{extra}")
    lines.append(f"Colours: {', '.join(sorted(COLORS))}.")
    return "\n".join(lines)


PROMPT_VERSION, _TEMPLATE = load_prompt("task_spec")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def render(inputs):
    return _SYSTEM.replace("{catalog}", catalog_text()), _USER.replace("{prompt}", inputs["prompt"].strip())


def validate(out, inputs, build: bool = True) -> list[str]:
    """Checks beyond the schema. Every message says what to change, since it is fed back to the model.
    build=False skips the random-layout check (a photo's twin places the objects where the photo shows them)."""
    st = out["status"]
    if st == "clarify":
        return [] if out.get("question", "").strip() else ["status clarify needs a non-empty 'question'"]
    if st == "refuse":
        return [] if out.get("reason", "").strip() else ["status refuse needs a non-empty 'reason'"]
    if "task" not in out:
        return ["status ok needs a 'task' object"]
    t, errs = out["task"], []
    ids = [o["id"] for o in t["objects"]]
    for o in t["objects"]:
        if not ID_RE.match(o["id"]):
            errs.append(f"object id '{o['id']}' must be snake_case (lowercase letters, digits, _)")
        if o.get("pose") == "lying":
            from ...sim.templates import lying_extent
            if lying_extent(o["template"], o.get("params") or {}) is None:
                errs.append(f"{o['id']}: a {o['template']} cannot lie on its side (pose 'lying' only for loose "
                            f"cylinders, cups, mugs and imported loose objects)")
        allowed = set(DEFAULTS[o["template"]])
        for k, v in (o.get("params") or {}).items():
            if k not in allowed:
                errs.append(f"{o['id']}: template {o['template']} has no parameter '{k}' (allowed: {sorted(allowed)})")
            elif k == "color":
                if v not in COLORS:
                    errs.append(f"{o['id']}: unknown colour '{v}' (known: {sorted(COLORS)})")
            elif not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(f"{o['id']}: parameter '{k}' must be a number")
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errs.append(f"duplicate object ids: {sorted(dup)}")
    by_id = {o["id"]: o for o in t["objects"]}
    goal_objs = []
    for g in t["goal"]:
        errs += _check_term(g, by_id)
        for k in ("obj", "target"):
            if k in g and g[k] not in by_id:
                errs.append(f"goal refers to unknown object '{g[k]}'")
        if g["pred"] in RELATION_PREDS and "target" not in g:
            errs.append(f"goal '{g['pred']}' for {g['obj']} needs a target (the object it goes beside)")
        if g["pred"] in RELATION_PREDS and by_id.get(g["obj"], {}).get("role", "item") != "item":
            errs.append(f"goal object '{g['obj']}' must have role item")
        if g["pred"] in ("inside", "on") and "target" not in g:
            errs.append(f"goal '{g['pred']}' for {g['obj']} needs a target")
        if g["pred"] in ("inside", "on") and g["obj"] in by_id and g.get("target") in by_id:
            tt = by_id[g["target"]]["template"]
            if g["pred"] == "inside" and tt not in CONTAINERS_INSIDE and not _imported_container(tt):
                errs.append(f"'inside' needs a bowl, cup, bin or a hollow imported object as target, not {tt} "
                            f"({g['target']}); use 'on' for trays and zones")
            if g["pred"] == "on" and tt not in SURFACES_ON | CONTAINERS_INSIDE | STACKABLE:
                errs.append(f"'on' needs a tray, zone, or a block/bar/cylinder to stack on, not {tt} ({g['target']})"
                            + ("; use 'inside' for a hollow imported object" if _imported_container(tt) else ""))
            if by_id[g["obj"]].get("role", "item") != "item":
                errs.append(f"goal object '{g['obj']}' must have role item")
        goal_objs.append(g["obj"])
    items = [o["id"] for o in t["objects"] if o.get("role", "item") == "item"]
    stack_bases = {g["target"] for g in t["goal"] if "target" in g and g["pred"] in ("inside", "on")}
    for i in items:
        n = goal_objs.count(i)
        if n > 1 or (n == 0 and i not in stack_bases):
            errs.append(f"item '{i}' must have exactly one goal (has {n}); only the bottom of a stack may have none")
    if not t["goal"]:
        errs.append("a task needs at least one goal")
    for c in t.get("constraints", []):
        if c["obj"] not in by_id:
            errs.append(f"constraint refers to unknown object '{c['obj']}'")
    order = t.get("order") or []
    if order:
        if len(set(order)) != len(order):
            errs.append("'order' lists an item twice")
        for i in order:
            if i not in goal_objs:
                errs.append(f"'order' item '{i}' has no goal; list only goal items, in the required sequence")
        if len(order) < 2:
            errs.append("'order' needs at least two items (omit it when there is no sequence)")
    if errs or not build:
        return errs
    # The placer must be able to lay the scene out (reach, no overlaps).
    from ...sim.scene import build_scene
    try:
        build_scene(to_task(out, inputs), seed=0)
    except Exception as e:
        errs.append(f"the scene cannot be built: {e}. Use fewer or smaller objects.")
    return errs


def _check_term(g, by_id) -> list[str]:
    """Composable measurable goal terms (joint / distance / height): op + value, within what the object allows."""
    if g["pred"] not in ("joint", "distance", "height"):
        return []
    errs = []
    if g.get("op") not in (">=", "<=") or not isinstance(g.get("value"), (int, float)):
        return [f"goal '{g['pred']}' on {g['obj']} needs op ('>=' or '<=') and a numeric value"]
    o = by_id.get(g["obj"])
    if g["pred"] == "joint" and o:
        art = ARTICULATED.get(o["template"])
        if not art:
            return [f"'{g['obj']}' ({o['template']}) has no moving part; 'joint' goals need one of {sorted(ARTICULATED)}"]
        if g.get("part", art["part"]) != art["part"]:
            errs.append(f"{g['obj']} has moving part '{art['part']}', not '{g.get('part')}'")
        hi = art.get("max") or (RANGES[o["template"]]["travel"][1] if art["unit"] == "m" else 105.0)
        lo_ok = g["value"] >= 0 if g["op"] == "<=" else g["value"] > 0
        if not (lo_ok and g["value"] <= hi):
            errs.append(f"joint value for {g['obj']} must be in (0, {hi}] {art['unit']} (0 allowed for '<=')")
    if g["pred"] == "distance" and "target" not in g:
        errs.append(f"goal 'distance' for {g['obj']} needs a target")
    if g["pred"] in ("distance", "height") and not 0 <= g["value"] <= 0.6:
        errs.append(f"'{g['pred']}' value must be metres between 0 and 0.6")
    return errs


def to_task(out, inputs) -> dict:
    """The executor's task dict from a validated 'ok' output."""
    t = out["task"]
    objs = [{"id": o["id"], "template": o["template"], "role": "item" if o.get("role") == "obstacle" else o.get("role", "item"),
             "params": dict(o.get("params") or {}), **({"fixed": o["fixed"]} if "fixed" in o else {}),
             **({"pose": o["pose"]} if o.get("pose") == "lying" else {})}
            for o in t["objects"]]
    return {"name": t["name"], "prompt": t.get("prompt") or inputs["prompt"], "objects": objs,
            "goal": [dict(g) for g in t["goal"]], "constraints": [dict(c) for c in t.get("constraints", [])],
            **({"order": list(t["order"])} if t.get("order") else {}),
            "obstacles": [o["id"] for o in t["objects"] if o.get("role") == "obstacle"]}


# ---------------------------------------------------------------- rule stand-in
REFUSE = [(r"\b(pour|water|liquid|juice|fill)\b", "Liquids are not supported in this simulation."),
          (r"\b(fold|cloth|towel|rope|cable)\b", "Deformable objects are not supported."),
          (r"\b(cut|slice|knife)\b", "Cutting is outside what this robot can do."),
          (r"\b(open|unscrew)\b.*\b(jar|lid|bottle)\b", "Unscrewing jars, lids or bottles is not supported.")]


def rule(inputs) -> dict:
    """Keyword stand-in covering the built-in tasks. Same output format as the model."""
    from ...tasks import specs
    p = inputs["prompt"].lower()
    for pat, reason in REFUSE:
        if re.search(pat, p):
            return {"status": "refuse", "reason": reason}
    has = lambda *ws: all(re.search(rf"\b{w}", p) for w in ws)
    if (has("open") or has("close") or has("shut")) and (has("drawer") or has("door")):
        closing = has("close") or has("shut")
        task = {(False, True): specs.open_drawer, (False, False): specs.open_door,
                (True, True): specs.close_drawer, (True, False): specs.close_door}[(closing, has("drawer"))](seed=0)
        return {"status": "ok", "task": {"name": task["name"], "prompt": inputs["prompt"], "objects": [
            {"id": o["id"], "template": o["template"], "role": o.get("role", "item"),
             "params": {k: v for k, v in o["params"].items() if k == "open"}} for o in task["objects"]],
            "goal": task["goal"], "constraints": []}}
    if has("stack") and has("block"):
        task = specs.stack_by_size(seed=0)
    elif has("sort") and has("bowl") and has("tray"):
        task = specs.sort_with_bowl(seed=0)
    elif has("sort") and has("block"):
        task = specs.sort_by_color(seed=0)
    elif has("ball") and (has("basket") or has("bin")):
        task = specs.balls_in_basket(seed=0)
    elif has("bowl") and has("tray"):
        task = specs.bowl_on_tray(seed=0)
    elif has("cup") and has("tray"):
        task = specs.cup_on_tray(seed=0)
    elif has("block") and (has("bin") or has("box")):
        task = specs.single_block(seed=0)
    else:
        return {"status": "clarify", "question": "Which objects should I use, and where should each one go?"}
    objs = [{"id": o["id"], "template": o["template"], "role": o.get("role", "item"), "params": o.get("params", {})}
            for o in task["objects"]]
    constraints = []
    if re.search(r"(knock|tip|topple)\w*\s+(over\s+)?the\s+(bottle|can)|(bottle|can)\b.*\b(alone|untouched)", p):
        objs.append({"id": "bottle", "template": "cylinder", "role": "obstacle",
                     "params": {"radius": 0.03, "height": 0.14, "color": "green" if "green" not in p else "yellow"}})
        constraints = [{"pred": "stays_upright", "obj": "bottle"}, {"pred": "not_disturbed", "obj": "bottle"}]
    return {"status": "ok", "task": {"name": task["name"], "prompt": inputs["prompt"], "objects": objs,
                                     "goal": task["goal"], "constraints": constraints}}


JOB = register(Job("task_spec", SCHEMA, render, validate, rule, PROMPT_VERSION))
