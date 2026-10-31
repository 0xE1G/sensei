"""Job "diagnose": observable evidence of a failed step -> ranked causes, each with a one-change test."""
from __future__ import annotations

import json

from ...sim.templates import RANGES
from ..llm import Job, load_prompt, register

LABELS = ["missed_grasp", "weak_grasp", "collision", "unreachable", "slip", "lift_check_failed", "drop",
          "wrong_place", "constraint_violation"]
TEST = {"type": "object", "required": ["type"],
        "properties": {"type": {"enum": ["skill_param", "object_param", "alternative_skill"]},
                       "skill": {"type": "string"}, "param": {"type": "string"}, "value": {"type": "number"},
                       "factor": {"type": "number"}, "card": {"type": "string"}}}
SCHEMA = {"type": "object", "required": ["hypotheses"],
          "properties": {"summary": {"type": "string"},
                         "hypotheses": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                             "type": "object", "required": ["cause", "predicted_label", "test"],
                             "properties": {"cause": {"type": "string"}, "predicted_label": {"enum": LABELS},
                                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                            "test": TEST}}}}}

PROMPT_VERSION, _TEMPLATE = load_prompt("diagnose")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def render(inputs):
    return _SYSTEM, _USER.replace("{evidence}", json.dumps(inputs, indent=1))


def validate(out, inputs) -> list[str]:
    errs = []
    skills = {s["skill"]: s for s in inputs["steps"]}
    template = inputs["object"]["template"]
    for i, h in enumerate(out["hypotheses"]):
        t = h["test"]
        if t["type"] == "skill_param":
            s = skills.get(t.get("skill"))
            if s is None:
                errs.append(f"hypothesis {i}: skill must be one of the trace's skills {sorted(skills)}")
            elif t.get("param") not in s["param_ranges"]:
                errs.append(f"hypothesis {i}: {t.get('skill')} has no parameter '{t.get('param')}' "
                            f"(has {sorted(s['param_ranges'])})")
            elif "value" not in t:
                errs.append(f"hypothesis {i}: skill_param test needs a numeric 'value'")
        elif t["type"] == "object_param":
            if t.get("param") not in RANGES[template]:
                errs.append(f"hypothesis {i}: object ({template}) has no parameter '{t.get('param')}' "
                            f"(has {sorted(RANGES[template])})")
            elif ("value" in t) == ("factor" in t):
                errs.append(f"hypothesis {i}: object_param test needs exactly one of 'value' or 'factor'")
        elif t["type"] == "alternative_skill":
            if t.get("card") not in inputs["library_grasps"] or t.get("card") == inputs["grasp_used"]:
                errs.append(f"hypothesis {i}: card must be another grasp from {inputs['library_grasps']}")
    return errs


def rule(inputs) -> dict:
    """Heuristic stand-in reading the same evidence: finger width, reach, contacts, what perception saw after."""
    st = {s["skill"]: s for s in inputs["steps"]}
    g = st[inputs["grasp_used"]]
    per, obs = inputs["perceived"], inputs.get("observed") or {}
    alts = [c for c in inputs["library_grasps"] if c != inputs["grasp_used"]]
    hyps = []
    info = g["info"]
    depth_param = next((p for p in ("depth_frac", "depth") if p in g["param_ranges"]), None)
    if not g["ok"]:
        if not info.get("reached", True):
            if depth_param and per["height"] > 0.085:
                v = 0.2 if depth_param == "depth_frac" else 0.015
                hyps.append({"cause": "the object is taller than the fingers: grasping at that depth drives the palm "
                                      "into its top", "predicted_label": "collision", "confidence": 0.7,
                             "test": {"type": "skill_param", "skill": g["skill"], "param": depth_param, "value": v}})
            hyps.append({"cause": "the hand hit the object on the way down (grasp pose too close to the object)",
                         "predicted_label": "collision" if inputs["arm_contacts"] else "unreachable",
                         "confidence": 0.4,
                         "test": ({"type": "alternative_skill", "card": alts[0]} if alts else
                                  {"type": "object_param", "param": next(iter(inputs["object_params_known"])),
                                   "factor": 0.8})})
        elif info.get("finger_width", 1) < 0.003:
            key = next(k for k in RANGES[inputs["object"]["template"]] if k not in ("mass", "friction"))
            hyps.append({"cause": "the object is wider than the gripper opening, so the fingers closed on nothing"
                         if per["width"] > 0.072 else "the grasp point missed the object",
                         "predicted_label": "missed_grasp", "confidence": 0.7,
                         "test": {"type": "object_param", "param": key, "factor": 0.6}})
            if alts:
                hyps.append({"cause": "a different grasp type fits this shape", "predicted_label": "missed_grasp",
                             "confidence": 0.4, "test": {"type": "alternative_skill", "card": alts[0]}})
        else:
            hyps.append({"cause": "the fingers touched the object but the grip was too weak",
                         "predicted_label": "weak_grasp", "confidence": 0.5,
                         "test": {"type": "skill_param", "skill": g["skill"], "param": "grip_force", "value": 40}})
    elif "lift" in st and not st["lift"]["ok"]:
        hyps.append({"cause": "the object slipped out while lifting: grip too weak for its weight",
                     "predicted_label": "slip", "confidence": 0.6,
                     "test": {"type": "skill_param", "skill": g["skill"], "param": "grip_force", "value": 40}})
    else:
        tipped = obs.get("height_ratio", 1.0) < 0.7
        far = obs.get("dist_to_target_xy", 0.0) > 0.12
        if tipped:
            hyps.append({"cause": "the object tipped over at release: released too high above the target",
                         "predicted_label": "wrong_place", "confidence": 0.6,
                         "test": {"type": "skill_param", "skill": "place_in", "param": "clearance", "value": 0.0}})
        if far or not tipped:
            hyps.append({"cause": "the object fell out during transport: moved too fast for the grip",
                         "predicted_label": "drop" if far else "wrong_place", "confidence": 0.5,
                         "test": {"type": "skill_param", "skill": "place_in", "param": "speed", "value": 0.1}})
        hyps.append({"cause": "the grip was too weak to carry it", "predicted_label": "drop", "confidence": 0.3,
                     "test": {"type": "skill_param", "skill": g["skill"], "param": "grip_force", "value": 40}})
    return {"summary": f"{inputs['grasp_used']} on the {inputs['object']['template']}: failed at "
                       f"{inputs['failed_step']}", "hypotheses": hyps[:3]}


JOB = register(Job("diagnose", SCHEMA, render, validate, rule, PROMPT_VERSION))
