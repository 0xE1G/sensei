"""Job "retrieve": a new object's perceived features + fitting library skills -> which to verify first, and why."""
from __future__ import annotations

import json

from ..llm import Job, load_prompt, register

SCHEMA = {"type": "object", "required": ["choice", "explanation"],
          "properties": {"choice": {"type": "string"}, "ranking": {"type": "array", "items": {"type": "string"}},
                         "explanation": {"type": "string"}}}

PROMPT_VERSION, _TEMPLATE = load_prompt("retrieve")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def render(inputs):
    return _SYSTEM, (_USER.replace("{object}", json.dumps(inputs["object"], indent=1))
                     .replace("{candidates}", json.dumps(inputs["candidates"], indent=1)))


def validate(out, inputs) -> list[str]:
    names = [c["card"] for c in inputs["candidates"]]
    errs = []
    if out["choice"] not in names:
        errs.append(f"choice must be one of {names}")
    bad = [r for r in out.get("ranking", []) if r not in names]
    if bad:
        errs.append(f"ranking has unknown skills {bad}; use only {names}")
    return errs


def rule(inputs) -> dict:
    """Stand-in: the router's evidence for the object's shape group, then library order."""
    g = inputs["object"]["shape_group"]

    def score(c):
        e = c["evidence"].get(g)
        if not e:
            return 0.5
        k, n = map(int, e.split("/"))
        return (k + 1) / (n + 2)
    ranked = sorted(inputs["candidates"], key=lambda c: -score(c))
    best = ranked[0]
    why = (f"measured {best['evidence'][g]} on shape group {g}" if g in best["evidence"]
           else f"preconditions fit ({best['requires']}); no evidence yet for {g}")
    return {"choice": best["card"], "ranking": [c["card"] for c in ranked], "explanation": why}


JOB = register(Job("retrieve", SCHEMA, render, validate, rule, PROMPT_VERSION))
