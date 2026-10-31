"""Job "search_proposal": results of the failure search so far -> regions to test in the next batch."""
from __future__ import annotations

import json
import math

from ..llm import Job, load_prompt, register

REGION = {"type": "object", "required": ["template", "count"],
          "properties": {"template": {"type": "string"}, "count": {"type": "integer", "minimum": 1},
                         "params": {"type": "object"},
                         "x": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "y": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "yaw": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}}
SCHEMA = {"type": "object", "required": ["regions"],
          "properties": {"rationale": {"type": "string"},
                         "regions": {"type": "array", "minItems": 1, "items": REGION}}}

PROMPT_VERSION, _TEMPLATE = load_prompt("search_proposal")
_SYSTEM, _USER = _TEMPLATE.split("\n---\n", 1)


def _fmt_rows(rows):
    return "\n".join(json.dumps(r) for r in rows) if rows else "(none yet)"


def render(inputs):
    space = "\n".join(f"- {t}: " + ", ".join(f"{k} {lo}-{hi}" for k, (lo, hi) in ps.items())
                      for t, ps in inputs["space"].items())
    user = (_USER.replace("{batch}", str(inputs["batch"])).replace("{left}", str(inputs["budget_left"]))
            .replace("{space}", space).replace("{x}", str(inputs["xy"]["x"])).replace("{y}", str(inputs["xy"]["y"]))
            .replace("{stats}", _fmt_rows([{"template": t, **v} for t, v in inputs["stats"].items()]))
            .replace("{failures}", _fmt_rows(inputs["failures"]))
            .replace("{successes}", _fmt_rows(inputs["successes"])))
    return _SYSTEM, user


def _inside(rng, bounds, tol=None):
    """lo <= hi within bounds, allowing rounding (1% of the range): 1.571 for pi/2 is fine; values are clipped
    when sampled."""
    tol = 0.01 * (bounds[1] - bounds[0]) if tol is None else tol
    return len(rng) == 2 and rng[0] <= rng[1] and bounds[0] - tol <= rng[0] and rng[1] <= bounds[1] + tol


def validate(out, inputs) -> list[str]:
    errs, space = [], inputs["space"]
    total = sum(r["count"] for r in out["regions"])
    if total != inputs["batch"]:
        errs.append(f"region counts add up to {total}, must be exactly {inputs['batch']}")
    for i, r in enumerate(out["regions"]):
        t = r["template"]
        if t not in space:
            errs.append(f"region {i}: unknown template '{t}' (allowed: {sorted(space)})")
            continue
        for k, v in (r.get("params") or {}).items():
            if k not in space[t]:
                errs.append(f"region {i}: {t} has no parameter '{k}' (allowed: {sorted(space[t])})")
            elif not (isinstance(v, list) and _inside(v, space[t][k])):
                errs.append(f"region {i}: {t}.{k} must be [lo, hi] inside {space[t][k]}")
        for ax in ("x", "y"):
            if ax in r and not _inside(r[ax], inputs["xy"][ax]):
                errs.append(f"region {i}: {ax} must be [lo, hi] inside {inputs['xy'][ax]}")
        if "yaw" in r and not _inside(r["yaw"], [-1.5708, 1.5708]):
            errs.append(f"region {i}: yaw must be [lo, hi] inside [-1.5708, 1.5708]")
    return errs


def rule(inputs) -> dict:
    """Stand-in: an upper-confidence bandit over templates; templates with failures get regions shrunk around them."""
    space, stats, batch = inputs["space"], inputs["stats"], inputs["batch"]
    n_total = sum(v["n"] for v in stats.values()) + 1
    score = {}
    for t in space:
        s = stats.get(t, {"n": 0, "valid_failures": 0})
        score[t] = (s["valid_failures"] + 1) / (s["n"] + 2) + math.sqrt(2 * math.log(n_total) / (s["n"] + 1))
    total = sum(score.values())
    counts = {t: int(batch * score[t] / total) for t in space}
    for t in sorted(space, key=lambda t: -score[t]):           # hand out the remainder to the best scores
        if sum(counts.values()) >= batch:
            break
        counts[t] += 1
    regions = []
    for t, c in counts.items():
        if c <= 0:
            continue
        fails = [f for f in inputs["failures"] if f["template"] == t]
        if fails and c >= 2:                                    # half around the failures, half anywhere
            box = {}
            for k, (lo, hi) in space[t].items():
                vals = [f["params"][k] for f in fails if k in f["params"]]
                if vals:
                    pad = 0.15 * (hi - lo)
                    box[k] = [max(lo, min(vals) - pad), min(hi, max(vals) + pad)]
            regions.append({"template": t, "count": c - c // 2, "params": box})
            regions.append({"template": t, "count": c // 2})
        else:
            regions.append({"template": t, "count": c})
    return {"rationale": "bandit stand-in: more trials where failures were found, the rest spread for exploration",
            "regions": regions}


JOB = register(Job("search_proposal", SCHEMA, render, validate, rule, PROMPT_VERSION))
