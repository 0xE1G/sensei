"""Job "plan_program": a task spec -> a short plan program over the robot API (Code as Policies)."""
from __future__ import annotations

import json

from ...program import check_program
from ..llm import Job, load_prompt, register

SCHEMA = {"type": "object", "required": ["program"],
          "properties": {"program": {"type": "string", "minLength": 1}, "explanation": {"type": "string"}}}

PROMPT_VERSION, _TEMPLATE = load_prompt("plan_program")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def _task_view(task):
    """What the model needs: objects (id, kind, colour, role, size parameters), goals, constraints."""
    return {"objects": [{"id": o["id"], "kind": o["template"], "role": o.get("role", "item"),
                         "params": o.get("params", {})} for o in task["objects"]],
            "goal": task["goal"], "constraints": task.get("constraints", [])}


def library_capabilities() -> list[str]:
    """Kinds of verified model-written skills in the active library (push, ...)."""
    from ...skills import library as L
    return sorted({c["kind"] for c in L.load_cards().values() if c.get("impl") == "generated"})


def render(inputs):
    caps = inputs.get("capabilities")
    caps = library_capabilities() if caps is None else caps
    return (_SYSTEM.replace("{capabilities}", ", ".join(caps) or "none"),
            _USER.replace("{prompt}", inputs.get("prompt") or "")
                     .replace("{task}", json.dumps(_task_view(inputs["task"]), indent=1)))


def validate(out, inputs) -> list[str]:
    errs = check_program(out["program"])
    ids = {o["id"] for o in inputs["task"]["objects"]}
    if not errs and not any(i in out["program"] for i in ids) and "find(" not in out["program"]:
        errs.append("the program never refers to the task's objects (use find(...) or their ids)")
    return errs


def rule(inputs) -> dict:
    """Stand-in: one pick/place per goal, ordered so every target that is itself moved is placed first."""
    goals = [g for g in inputs["task"]["goal"] if g.get("pred") in ("inside", "on")]
    items = {g["obj"] for g in goals}
    done, order = set(), []
    while len(order) < len(goals):
        ready = [g for g in goals if g not in order and (g["target"] not in items or g["target"] in done)]
        if not ready:                                   # a cycle: keep the given order
            ready = [g for g in goals if g not in order]
        order.append(ready[0])
        done.add(ready[0]["obj"])
    lines = []
    kinds = {o["id"]: o["template"] for o in inputs["task"]["objects"]}
    for g in order:
        if kinds.get(g["obj"]) == "plate":            # no grasp lifts a plate: it needs a motion, not a pick
            lines += [f'use("push", find_one(id="{g["obj"]}"), find_one(id="{g["target"]}"))']
            continue
        lines += [f'obj = find_one(id="{g["obj"]}")',
                  "ok = pick(obj)",
                  "if not ok and next_best_grasp(obj):",
                  "    ok = pick(obj, grasp=next_best_grasp(obj))",
                  "if ok:",
                  f'    place(find_one(id="{g["target"]}"))',
                  'elif has_skill("push"):',
                  f'    push(obj, find_one(id="{g["target"]}"))']
    return {"program": "\n".join(lines),
            "explanation": "one pick and place per goal, bases before what goes on them, one retry per pick"}


JOB = register(Job("plan_program", SCHEMA, render, validate, rule, PROMPT_VERSION))
