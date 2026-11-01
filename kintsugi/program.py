"""Plan programs (Code as Policies): the model writes a short Python program over a small robot API.

    blocks = sorted(find(kind="block"), key=lambda b: b.size, reverse=True)
    for lower, upper in zip(blocks, blocks[1:]):
        if pick(upper):
            place(lower)

The program never commands motors. `pick` and `place` run the same verified skills, checks, labels and evidence
as the fixed planner (Episode step methods); the router chooses grasps unless the program names one, and parameter
patches apply. Safety: the syntax tree is checked against a whitelist before running (API calls, plain Python
control flow, no imports, no private attributes), and execution has a line budget and a skill-call budget.
"""
from __future__ import annotations

import ast
import math
import sys

API_NAMES = ("find", "find_one", "pick", "place", "place_at", "push", "use", "has_skill", "next_best_grasp", "holding", "distance",
             "left_of", "right_of", "in_front_of", "behind", "say")
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in
                 ("len", "range", "sorted", "min", "max", "abs", "sum", "enumerate", "zip", "list", "reversed",
                  "round", "any", "all", "float", "int", "str", "bool", "dict", "tuple", "set", "True", "False",
                  "None")}
OBJ_ATTRS = ("id", "kind", "color", "role", "x", "y", "size", "height", "top", "hollow", "long")
ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass,
    ast.FunctionDef, ast.Return, ast.arguments, ast.arg, ast.Lambda, ast.Call, ast.keyword, ast.Name, ast.Load,
    ast.Store, ast.Attribute, ast.Subscript, ast.Slice, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp, ast.comprehension, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.IfExp, ast.JoinedStr, ast.FormattedValue, ast.operator, ast.unaryop, ast.boolop,
    ast.cmpop, ast.expr_context)
MAX_LINES = 50_000
MAX_SKILL_CALLS = 40


class ProgramError(Exception):
    pass


def check_program(code: str) -> list[str]:
    """Static checks; every message says what to change (they are fed back to the model)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]
    errs, defined = [], set(API_NAMES) | set(SAFE_BUILTINS)
    for node in ast.walk(tree):                      # names the program binds itself
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, (ast.FunctionDef,)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            errs.append(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed "
                        f"(no imports, classes, try/except, with, global)")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errs.append(f"line {node.lineno}: private attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
            errs.append(f"line {node.lineno}: unknown name '{node.id}' (API: {', '.join(API_NAMES)})")
        elif isinstance(node, ast.FunctionDef) and node.decorator_list:
            errs.append(f"line {node.lineno}: decorators are not allowed")
    return sorted(set(errs))


class Obj:
    """What a program sees of an object: identity from the task spec, geometry from perception."""

    def __init__(self, spec, desc):
        self.id, self.kind = spec["id"], spec["template"]
        self.color = (spec.get("params") or {}).get("color")
        self.role = spec.get("role", "item")
        d = desc or {}
        self.x, self.y = (d.get("center") or [float("nan")] * 2)
        self.size = d.get("width", float("nan"))
        self.height = d.get("height", float("nan"))
        self.top = d.get("top_z", float("nan"))
        self.hollow = bool(d.get("hollow"))
        self.long = d.get("elongation", 1.0) > 2.0

    def __repr__(self):
        return f"Obj({self.id})"


def run_program(ep, code: str) -> dict:
    """Run `code` against episode `ep`. Returns record fields: program, program_log, program_error."""
    from .skills.router import patch_for, shape_group
    from .tasks.planner import choose_grasp, fits, grasp_preference

    log, state = [], {"held": None, "skill_calls": 0, "placed": {}, "replace": {}}
    specs = {o["id"]: o for o in ep.task["objects"]}
    allowed = ep.allowed_grasps or grasp_preference(ep.cards)
    from .tasks.planner import _container_slots
    # Each item bound for a shared container gets its own spot, decided up front from the goals.
    slots = {k: tuple(v) for k, v in _container_slots(
        [g for g in ep.task["goal"] if g.get("pred") in ("inside", "on") and g["target"] in ep.percepts
         and specs.get(g["target"], {}).get("template") not in ("block", "bar", "cylinder")], ep.percepts).items()}

    def _id(o):
        oid = o.id if isinstance(o, Obj) else o
        if oid not in specs:
            raise ProgramError(f"unknown object {oid!r}; known: {sorted(specs)}")
        return oid

    def _budget():
        state["skill_calls"] += 1
        if state["skill_calls"] > MAX_SKILL_CALLS:
            raise ProgramError(f"more than {MAX_SKILL_CALLS} pick/place calls")

    def find(kind=None, color=None, role=None, id=None):
        descs = ep.perc.perceive(list(specs))
        out = [Obj(s, descs.get(i)) for i, s in specs.items()
               if (kind is None or s["template"] == kind) and (id is None or i == id)
               and (role is None or s.get("role", "item") == role)
               and (color is None or (s.get("params") or {}).get("color") == color)]
        log.append({"call": "find", "args": {"kind": kind, "color": color, "role": role, "id": id},
                    "result": [o.id for o in out]})
        return out

    def find_one(kind=None, color=None, role=None, id=None):
        out = find(kind, color, role, id)
        if not out:
            raise ProgramError(f"find_one found nothing for kind={kind} color={color} role={role} id={id}")
        return out[0]

    def _params(skill, desc):
        return {**patch_for(ep.patches, skill, shape_group(desc)), **ep.overrides.get(skill, {})}

    def pick(obj, grasp=None):
        _budget()
        if state["held"] is not None:
            raise ProgramError("pick called while already holding something; place it first")
        oid = _id(obj)
        if grasp is not None and (grasp not in ep.cards or ep.cards[grasp].get("kind") != "grasp"):
            raise ProgramError(f"unknown grasp {grasp!r}; library grasps: {grasp_preference(ep.cards)}")
        # Acting on feasibility: no grasp fits this object but a verified motion covers its goal (a plate onto a
        # tray -> push): do not try a grasp that would hit it; the program's fallback (push / use) runs instead.
        v = next((f for f in ep.feasibility if f["obj"] == oid), None)
        if grasp is None and ep.act_on_feasibility and v and v["verdict"] == "motion":
            log.append({"call": "pick", "obj": oid, "ok": False, "label": "no_grasp_fits", "skipped": True,
                        "reason": v["reason"]})
            return False
        tried = []
        # With recovery on (and no grasp forced by the program), a failed pick is retried on a fresh percept with
        # the next verified grasp.
        for attempt in range(1 + (ep.recovery if grasp is None else 0)):
            srec = ep.begin(oid)
            srec["attempt"] = attempt
            desc = srec["_desc"]
            if desc is None:
                ep.finish(srec, ep.goal_for(oid))
                log.append({"call": "pick", "obj": oid, "ok": False, "label": "object_not_perceived"})
                return False
            if grasp is not None:
                g, why = grasp, {"reason": "chosen by the program"}
            elif attempt == 0:
                g, why = choose_grasp(desc, allowed, ep.cards, ep.performance)
            else:
                g = ep.retry_grasp(desc, tried)
                why = {"reason": f"recovery attempt {attempt}", "card": g}
            srec["routing"] = why
            tried.append(g)
            ok = ep.grasp(srec, g, _params(g, desc)) and ep.lift(srec, _params("lift", desc))
            log.append({"call": "pick", "obj": oid, "grasp": g, "ok": ok, "label": srec["label"], "attempt": attempt})
            if ok:
                state["held"] = srec
                return True
            ep.finish(srec, ep.goal_for(oid))
            if not ep.should_retry(srec):
                return False
        return False

    def _place(tdesc, target_id, mode):
        _budget()
        srec = state["held"]
        if srec is None:
            raise ProgramError("place called while holding nothing; pick first")
        srec["target"] = target_id
        n = state["placed"].get(target_id, 0)
        goal = ep.goal_for(srec["obj"])
        if goal and goal.get("target") == target_id:
            slot = slots.get(srec["obj"], (0.0, 0.0))          # same spots as the fixed planner, whatever the order
        elif n and target_id and "rim_radius" not in tdesc:
            ex = tdesc.get("extent_xy", [tdesc["width"]] * 2)
            a = 2 * math.pi * n / 4 + math.pi / 2              # an unplanned extra item: spread it
            slot = (0.25 * min(ex) * math.cos(a), 0.25 * min(ex) * math.sin(a))
        else:
            slot = (0.0, 0.0)
        ok = ep.place(srec, tdesc, _params("place_in", srec["_desc"]), mode, slot)
        ep.finish(srec, ep.goal_for(srec["obj"]))
        state["held"] = None
        state["placed"][target_id] = n + 1
        log.append({"call": "place", "obj": srec["obj"], "target": target_id, "ok": bool(srec["ok"]),
                    "label": srec["label"]})
        return bool(srec["ok"] or (ok and ep.goal_for(srec["obj"]) is None))

    def place(target):
        tid = _id(target)
        tdesc = ep.perc.perceive([tid])[tid]
        if tdesc is None:
            raise ProgramError(f"target {tid} is not perceived")
        oid = state["held"]["obj"] if state["held"] else None
        ok = _place(tdesc, tid, "drop" if specs[tid]["template"] == "bin" else "place")
        # Recovery: dropped on the way or landed off the target -> pick it up again and place it again (bounded).
        last = ep.records[-1] if ep.records else {}
        while (not ok and oid and ep.recovery and state["replace"].get(oid, 0) < ep.recovery
               and ep.should_retry(last)):
            state["replace"][oid] = state["replace"].get(oid, 0) + 1
            if not pick(oid):
                return False
            ok = _place(ep.perc.perceive([tid])[tid] or tdesc, tid, "drop" if specs[tid]["template"] == "bin" else "place")
            last = ep.records[-1]
        return ok

    def place_at(x, y):
        return _place({"center": [float(x), float(y)], "top_z": 0.0, "width": 0.06, "extent_xy": [0.06, 0.06]},
                      None, "place")

    def use(capability, obj, target=None, skill=None):
        """A motion no grasp/place can do (push, slide, ...): run the library's verified model-written skill of that
        kind. If there is none, the program stops with "missing capability: <kind>" and the call is recorded, so
        Kintsugi can create the skill (create_skill.py) for exactly this request and these objects."""
        from .skills.router import pick_motion, shape_group as group
        _budget()
        capability = str(capability).strip().lower().replace(" ", "_")
        if state["held"] is not None:
            raise ProgramError(f"{capability} needs an empty gripper; place what you are holding first")
        oid = _id(obj)
        tid = _id(target) if target is not None else None
        cands = [n for n, c in ep.cards.items() if c.get("kind") == capability and c.get("impl") == "generated"]
        if not cands:
            state["missing_call"] = {"capability": capability, "obj": oid, "target": tid}
            raise ProgramError(f"missing capability: {capability}")
        if skill is not None and skill not in cands:
            raise ProgramError(f"unknown {capability} skill {skill!r}; library {capability} skills: {cands}")
        srec = ep.begin(oid, tid)
        if srec["_desc"] is None:
            ep.finish(srec, ep.goal_for(oid))
            return False
        name = skill or pick_motion(ep.cards, cands, ep.performance, group(srec["_desc"]))
        srec["routing"] = {"reason": "push skill", "card": name}
        tdesc = ep.perc.perceive([tid])[tid] if tid else None
        ep.push(srec, name, tdesc, _params(name, srec["_desc"]))
        ep.finish(srec, ep.goal_for(oid))
        log.append({"call": "use", "capability": capability, "obj": oid, "target": tid, "skill": name, "ok": bool(srec["ok"]),
                    "label": srec["label"]})
        return bool(srec["ok"])

    def next_best_grasp(obj):
        from .skills.router import route
        oid = _id(obj)
        desc = ep.describe(oid)
        fitting = [n for n in allowed if desc and fits(ep.cards[n], desc)]
        if len(fitting) < 2:
            return None
        _, why = route(fitting, desc, ep.performance)
        ranked = sorted(why["scores"], key=lambda n: -why["scores"][n])
        return ranked[1]

    def holding():
        return state["held"] is not None

    api = {"find": find, "find_one": find_one, "pick": pick, "place": place, "place_at": place_at, "use": use,
           "push": lambda obj, target=None, skill=None: use("push", obj, target, skill),
           "has_skill": lambda kind: any(c.get("kind") == kind for c in ep.cards.values()),
           "next_best_grasp": next_best_grasp, "holding": holding,
           "distance": lambda a, b: math.hypot(a.x - b.x, a.y - b.y),
           "left_of": lambda a, b: a.y > b.y, "right_of": lambda a, b: a.y < b.y,      # seen from the robot
           "in_front_of": lambda a, b: a.x < b.x, "behind": lambda a, b: a.x > b.x,
           "say": lambda *a: log.append({"call": "say", "text": " ".join(map(str, a))})}

    # Which program line each skill call came from and when it ran (simulation time), for the replay's program view.
    call_lines = []

    def _traced(name, fn):
        def wrapper(*a, **k):
            f = sys._getframe(1)
            line = f.f_lineno if f.f_code.co_filename == "<plan program>" else None
            t0 = ep.robot.t
            try:
                return fn(*a, **k)
            finally:
                if line is not None:
                    call_lines.append({"call": name, "line": line, "t0": round(t0, 2), "t1": round(ep.robot.t, 2)})
        return wrapper
    for n in ("pick", "place", "place_at", "use", "push"):
        api[n] = _traced(n, api[n])

    error = None
    errs = check_program(code)
    if errs:
        error = "static check failed: " + "; ".join(errs)
    else:
        lines = [0]

        def tracer(frame, event, arg):
            if frame.f_code.co_filename == "<plan program>" and event == "line":
                lines[0] += 1
                if lines[0] > MAX_LINES:
                    raise ProgramError(f"line budget exceeded ({MAX_LINES})")
            return tracer
        old = sys.gettrace()
        sys.settrace(tracer)
        try:
            exec(compile(code, "<plan program>", "exec"), {"__builtins__": SAFE_BUILTINS, **api})
        except ProgramError as e:
            error = str(e)
        except Exception as e:                      # a bug in the program is a result, not a crash
            error = f"{type(e).__name__}: {e}"
        finally:
            sys.settrace(old)
    if state["held"] is not None:                    # finished while holding something
        srec = state["held"]
        srec["label"] = srec["label"] or "still_held"
        srec["_failed"] = True
        ep.finish(srec, ep.goal_for(srec["obj"]))
    attempted = {r["obj"] for r in ep.records}
    for g in ep.task["goal"]:
        if g.get("obj") and g["obj"] not in attempted:
            ep.records.append({"obj": g["obj"], "target": g.get("target"), "trace": [], "ok": False,
                               "label": "program_error" if error else "not_attempted", "routing": None})
    missing = error.split(":", 1)[1].strip() if error and error.startswith("missing capability") else None
    return {"program": code, "program_log": log, "program_lines": call_lines, "program_error": error,
            "missing_capability": missing,
            "missing_call": state.get("missing_call") if missing else None}
