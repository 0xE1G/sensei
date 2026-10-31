"""Router: choose among skill cards that fit an object by their measured success on its shape group.

Evidence lives in skill_library/performance.json, separate from the cards (it changes which variant is used, not
what a variant does):

    {"<card>": {"<shape group>": {"success": k, "trials": n, "sources": ["<run id>", ...]}}}

A shape group is a coarse, perception-only description (hollow or solid, narrow or wide for the gripper, short or
tall), so evidence gathered on one object carries over to similar ones. Score = (k + 1) / (n + 2); a card without
evidence for the group scores 0.5, and ties keep the library's preference order, so routing only changes where
there is evidence.
"""
from __future__ import annotations

import copy
import json

from .library import LIB_DIR, OPEN_WIDTH

PERF_FILE = LIB_DIR / "performance.json"
TALL = 0.085            # m; above this a grasp at half height puts the palm on the object's top
LONG = 2.0            # elongation (long / short extent) above which an object counts as long


def shape_group(desc: dict) -> str:
    g = "-".join(("hollow" if desc.get("hollow") else "solid",
                  "wide" if desc["width"] > OPEN_WIDTH - 0.008 else "narrow",
                  "tall" if desc["height"] > TALL else "short"))
    return g + ("-long" if desc.get("elongation", 1.0) > LONG else "")


def load_performance(path=None) -> dict:
    p = path or PERF_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def save_performance(perf: dict, path=None):
    (path or PERF_FILE).write_text(json.dumps(perf, indent=1, sort_keys=True))


def with_evidence(perf: dict, card: str, group: str, success: int, trials: int, source: str) -> dict:
    """A copy of `perf` with evidence added (counts accumulate; sources are kept for the lineage view)."""
    out = copy.deepcopy(perf)
    e = out.setdefault(card, {}).setdefault(group, {"success": 0, "trials": 0, "sources": []})
    e["success"] += int(success)
    e["trials"] += int(trials)
    if source not in e["sources"]:
        e["sources"].append(source)
    return out


def score(perf: dict, card: str, group: str) -> float:
    e = (perf or {}).get(card, {}).get(group)
    return 0.5 if not e else (e["success"] + 1) / (e["trials"] + 2)


PATCH_FILE = LIB_DIR / "patches.json"


def load_patches(path=None) -> dict:
    """Parameter patches per (skill, shape group), e.g. release lower for tall narrow objects:
    {"<skill>": {"<group>": {"params": {...}, "evidence": {...}, "sources": [...], "triggers": [...], "diff": {...}}}}
    They change a skill's parameters only for objects of that group; promoted through the same gates as cards."""
    p = path or PATCH_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def save_patches(patches: dict, path=None):
    (path or PATCH_FILE).write_text(json.dumps(patches, indent=1, sort_keys=True))


def patch_for(patches: dict, skill: str, group: str) -> dict:
    return dict(((patches or {}).get(skill, {}).get(group) or {}).get("params", {}))


def route(candidates: list[str], desc: dict, perf: dict) -> tuple[str, dict]:
    """Best candidate for this object; candidates are in preference order. Returns (card, why)."""
    g = shape_group(desc)
    scored = [(score(perf, c, g), -i, c) for i, c in enumerate(candidates)]
    best = max(scored)[2]
    return best, {"group": g, "scores": {c: round(s, 3) for s, _, c in scored}}
