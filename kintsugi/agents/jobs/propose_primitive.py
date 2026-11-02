"""Job "propose_primitive": a missing capability -> parameterized motion code (searched and gated before use)."""
from __future__ import annotations

import ast
import json
import re

from ...primitive_code import check_primitive
from ..llm import Job, load_prompt, register

PARAM = {"type": "object", "required": ["default", "range"],
         "properties": {"default": {"type": "number"}, "range": {"type": "array", "items": {"type": "number"},
                                                                  "minItems": 2, "maxItems": 2},
                        "meaning": {"type": "string"}}}
SCHEMA = {"type": "object", "required": ["name", "code", "params"],
          "properties": {"name": {"type": "string"}, "code": {"type": "string", "minLength": 20},
                         "params": {"type": "object", "minProperties": 1, "additionalProperties": PARAM},
                         "check": {"type": "string"}, "explanation": {"type": "string"}}}

PROMPT_VERSION, _TEMPLATE = load_prompt("propose_primitive")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def render(inputs):
    fb = inputs.get("feedback")
    fb_text = (f"\nYour previous proposal was searched and tested in simulation. What happened:\n{fb}\n"
               "Propose an improved motion (or wider ranges) that addresses these failures.") if fb else ""
    others = inputs.get("other_strategies")
    if others:
        fb_text += ("\nOther proposals already being tested use these strategies; propose a genuinely DIFFERENT one "
                    "(a different way of contacting or moving the object):\n" + "\n".join(f"- {o}" for o in others))
    if inputs.get("goal"):
        fb_text = f"\nSuccess is measured as: {json.dumps(inputs['goal'])}" + fb_text
    return _SYSTEM, (_USER.replace("{capability}", inputs["capability"]).replace("{why}", inputs["why"])
                     .replace("{obj}", json.dumps(inputs["obj"])).replace("{target}", json.dumps(inputs.get("target")))
                     .replace("{feedback}", fb_text))


def params_used(code: str) -> set[str]:
    used = set()
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "p"
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            used.add(node.slice.value)
    return used


def validate(out, inputs) -> list[str]:
    errs = check_primitive(out["code"])
    if not re.match(r"^[a-z][a-z0-9_]{2,40}$", out["name"]):
        errs.append("name must be snake_case (3-40 lowercase letters, digits, _)")
    for k, v in out["params"].items():
        lo, hi = v["range"]
        if not lo < hi:
            errs.append(f"param {k}: range must have lo < hi")
        elif not lo <= v["default"] <= hi:
            errs.append(f"param {k}: default {v['default']} is outside its range {v['range']}")
    if not errs:
        missing = params_used(out["code"]) - set(out["params"])
        if missing:
            errs.append(f"the code reads p[...] keys that are not declared in params: {sorted(missing)}")
    return errs


RULE_CODE = '''def primitive(obj, target, p):
    cx, cy = obj['center']
    tx, ty = target['center']
    d = max(hypot(tx - cx, ty - cy), 1e-6)
    ux, uy = (tx - cx) / d, (ty - cy) / d
    yaw = atan2(uy, ux)
    back = obj['width'] / 2 + p['standoff']
    sx, sy = cx - ux * back, cy - uy * back
    z = p['height']
    open_gripper()
    move(sx, sy, z + 0.08, yaw=yaw, speed=0.3)
    move(sx, sy, z, yaw=yaw, speed=0.15)
    travel = d + p['standoff'] + p['overshoot']
    ex, ey = sx + ux * travel, sy + uy * travel
    move(ex, ey, z, yaw=yaw, speed=p['speed'])
    move(ex, ey, z + 0.08, yaw=yaw, speed=0.3)'''


def rule(inputs) -> dict:
    """Stand-in: opening a part (drawer, door) -> grasp the handle and pull; standing a fallen object up -> grasp
    and turn it; otherwise a two-finger push."""
    cap = inputs["capability"].split(":")[0]
    if cap.startswith("stand_up"):
        from .reorient_motions import stand_in
        return stand_in()
    if (inputs.get("obj") or {}).get("parts", {}).get("handle") and ("open" in cap or "close" in cap):
        from .open_motions import stand_in
        return stand_in(cap)
    return _push_rule(inputs)


def _push_rule(inputs) -> dict:
    """Stand-in: a two-finger push. Open fingers spread across the push direction give two contacts (a disc
    pushed at one point spins away); start behind the object, slide it the distance to the target."""
    return {"name": "push_two_finger", "code": RULE_CODE,
            "params": {"height": {"default": 0.012, "range": [0.008, 0.03], "meaning": "tool height while pushing (m)"},
                       "standoff": {"default": 0.02, "range": [0.0, 0.05], "meaning": "start gap behind the object (m)"},
                       "overshoot": {"default": 0.0, "range": [-0.05, 0.05], "meaning": "extra travel beyond the target (m)"},
                       "speed": {"default": 0.1, "range": [0.03, 0.25], "meaning": "push speed (m/s)"}},
            "check": "the object's perceived centre ends inside the target",
            "explanation": "open fingers across the push direction, start behind the object, slide it to the target"}


JOB = register(Job("propose_primitive", SCHEMA, render, validate, rule, PROMPT_VERSION))
