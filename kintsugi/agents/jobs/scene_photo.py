"""Job "scene_photo": a photo of a real table (+ an optional request) -> a task spec whose objects also say where they
are in the photo (label for the segmenter, a pixel box, the real width). kintsugi/twin.py turns the boxes into table
positions. The task part is checked by the task_spec validator, so the rest of the pipeline sees an ordinary task."""
from __future__ import annotations

import copy

from ..llm import Job, load_prompt, register
from . import task_spec

SCHEMA = copy.deepcopy(task_spec.SCHEMA)
SCHEMA["properties"]["camera_elevation_deg"] = {"type": "number"}
_obj = SCHEMA["properties"]["task"]["properties"]["objects"]["items"]
_obj["required"] = ["id", "template", "label", "box"]
# share the template list with task_spec, so objects registered after start-up (downloads) are valid here too
_obj["properties"]["template"] = task_spec.SCHEMA["properties"]["task"]["properties"]["objects"]["items"]["properties"]["template"]
_obj["properties"].update({"label": {"type": "string"}, "width_m": {"type": "number"},
                           "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}})

PROMPT_VERSION, _TEMPLATE = load_prompt("scene_photo")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def render(inputs):
    req = inputs.get("prompt", "").strip() or "(none: choose the most natural task for this scene)"
    system = (_SYSTEM.replace("{catalog}", task_spec.catalog_text())
              .replace("{width}", str(inputs["width"])).replace("{height}", str(inputs["height"])))
    return system, _USER.replace("{prompt}", req)


def validate(out, inputs) -> list[str]:
    errs = task_spec.validate(out, {"prompt": inputs.get("prompt") or "photo"}, build=False)   # twin.py builds it
    if errs or out["status"] != "ok":
        return errs
    w, h = inputs["width"], inputs["height"]
    for o in out["task"]["objects"]:
        x1, y1, x2, y2 = o["box"]
        if not (0 <= x1 < x2 <= w * 1.02 and 0 <= y1 < y2 <= h * 1.02):
            errs.append(f"{o['id']}: box {o['box']} must be [x1, y1, x2, y2] inside the {w} x {h} image with x1<x2, y1<y2")
        if not o["label"].strip():
            errs.append(f"{o['id']}: label must name the object for the segmenter, e.g. 'blue mug'")
    return errs


def rule(inputs) -> dict:
    """The rule backend cannot see images."""
    return {"status": "clarify", "question": "Reading a photo needs a vision model: choose Muse Spark (or Claude) "
                                             "as the model, or type the task instead."}


JOB = register(Job("scene_photo", SCHEMA, render, validate, rule, PROMPT_VERSION))
