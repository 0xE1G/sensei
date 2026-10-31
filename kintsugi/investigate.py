"""Failure investigator: why did a step fail? Evidence -> hypotheses -> one-change replays -> evidence ledger.

For each failed trial (a task spec + seed):
  1. Replay the trial exactly (same task, seed and layout) to reproduce the failure and collect observable evidence:
     perception before/after, every skill's parameters and checks, finger width, arm contacts, routing.
  2. Model job "diagnose" (rule stand-in, Claude or Spark) proposes up to three ranked causes, each with a predicted
     failure label and ONE change that should make the step succeed if the cause is right.
  3. Each change is replayed on the very same scene (positions pinned): step succeeds -> CONFIRMED, else REJECTED.
  4. An evidence ledger labels every statement measured / hypothesis / confirmed / rejected.
Scores: top-1 label accuracy against the simulator's label (which the model never sees), confirmed vs rejected
counts. Output: runs/<id>/report.json, plus one ledger per case.
"""
from __future__ import annotations

import collections
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .runner import ROOT, TrialPool, trial_cfg, wilson
from .sim.templates import DEFAULTS, RANGES


def _log(msg):
    print(f"[investigate] {msg}", flush=True)


def _r(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return [_r(x) for x in v]
    if isinstance(v, dict):
        return {k: _r(x) for k, x in v.items()}
    return v


def build_evidence(record: dict, obj: str, cards: dict) -> dict:
    """Observable evidence for one failed step. No simulator verdict, no true object state."""
    st = next(s for s in record["subtasks"] if s.get("obj") == obj)
    spec_obj = next(o for o in record["task_spec"]["objects"] if o["id"] == obj)
    steps = []
    for t in st["trace"]:
        card = cards[t["skill"]]
        steps.append({"skill": t["skill"], "params": _r(t["params"]), "ok": t["ok"], "check": t["check"],
                      "info": _r(t["info"]),
                      "param_ranges": {k: v["range"] for k, v in card["params"].items()},
                      "param_meaning": {k: v.get("meaning") or v.get("unit") for k, v in card["params"].items()
                                        if v.get("meaning") or v.get("unit")}})
    failed = next((t["skill"] for t in st["trace"] if not t["ok"]), "final check after release")
    keep = ("center", "top_z", "height", "width", "extent_major", "extent_minor", "elongation", "axis_yaw",
            "hollow", "rim_radius", "wall_slope")
    per = record["percepts"].get(obj) or {}
    # Derived facts, all from perception + proprioception: arithmetic a diagnoser should not have to redo.
    derived = {}
    if st["trace"]:
        gi = st["trace"][0]["info"]
        if "tcp_z" in gi and "top_z" in per:
            derived["fingertips_aimed_below_top_m"] = round(per["top_z"] - gi["tcp_z"], 4)
            derived["finger_length_below_palm_m"] = 0.05
        if gi.get("reached") is False and "pos_err" in gi:
            derived["descent_stopped_short_by_m"] = round(gi["pos_err"], 4)
        if "finger_width" in gi:
            derived["fingers_closed_on_nothing"] = gi["finger_width"] < 0.003
        # The gripper closes across the narrow side of the footprint, not the circumscribed width (a rotated block's
        # "width" is its diagonal).
        derived["object_narrow_side_m"] = round(per.get("extent_minor", per.get("width", 0)), 4)
        derived["object_wider_than_gripper_opening"] = derived["object_narrow_side_m"] > 0.072
    # Other perceived objects near the item (a bottle to keep clear of, ...), with the gap between them.
    target = st.get("target")
    neighbors = []
    for o, d in (record.get("percepts") or {}).items():
        if not d or o in (obj, target) or not per:
            continue
        gap = float(np.linalg.norm(np.subtract(d["center"], per["center"]))) - d["width"] / 2 - per.get("width", 0) / 2
        if gap < 0.15:
            neighbors.append({"id": o, "height": round(d["height"], 3), "width": round(d["width"], 3),
                              "gap_m": round(gap, 3)})
    if neighbors:
        derived["nearest_neighbor_gap_m"] = min(n["gap_m"] for n in neighbors)
    obs = st.get("observed") or {}
    if obs.get("height_ratio") is not None:
        derived["tipped_over_after"] = obs["height_ratio"] < 0.7
    return {"task": record.get("prompt"), "derived": derived,
            "object": {"id": obj, "template": spec_obj["template"]},
            "object_params_known": {k: v for k, v in _r(spec_obj.get("params", {})).items() if k != "color"},
            "perceived": {k: _r(per[k]) for k in keep if k in per},
            "grasp_used": st["trace"][0]["skill"] if st["trace"] else None,
            "routing": st.get("routing"),
            "steps": steps, "failed_step": failed,
            "observed": st.get("observed"), "arm_contacts": st.get("contacts", []), "nearby_objects": neighbors,
            "constraints": record.get("task_spec", {}).get("constraints", []),
            "library_grasps": sorted(n for n, c in cards.items() if c.get("kind") == "grasp")}


def describe_test(t: dict) -> str:
    if t["type"] == "skill_param":
        return f"{t['skill']}.{t['param']} = {t['value']}"
    if t["type"] == "object_param":
        return f"object {t['param']} " + (f"= {t['value']}" if "value" in t else f"x {t['factor']}")
    return f"use {t['card']} instead"


class Investigator:
    def __init__(self, workers=None, library=None, performance=None, library_name="current", client=None,
                 run_id=None, model_threads=4):
        from .agents.llm import ModelClient
        from .skills.library import load_cards
        from .skills.router import load_performance
        self.cards = library or load_cards()
        self.perf = load_performance() if performance is None else performance
        from .skills.router import load_patches
        self.patches = load_patches() if performance is None else {}
        self.library_name = library_name
        self.client = client or ModelClient()
        self.model_threads = model_threads
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S_investigate")
        self.out = ROOT / "runs" / self.run_id
        (self.out / "ledgers").mkdir(parents=True, exist_ok=True)
        self.pool = TrialPool(workers)
        self.t0 = time.time()

    def _cfg(self, tag, spec, **kw):
        return trial_cfg(spec["name"], self.out, save_failures=False, success_sample=0, save_video=False,
                         save_poses=False, tag=tag, library=self.cards, performance=self.perf, task_spec=spec,
                         patches=self.patches, **kw)

    def _record(self, tag, seed):
        return json.loads((self.out / "trials" / f"{tag}_{seed:06d}.json").read_text())

    def _diagnose(self, ev):
        from .agents.llm import ModelRejected
        try:
            res = self.client.complete("diagnose", ev)
            return {"output": res.output, "meta": res.meta}
        except ModelRejected as e:
            return {"rejected": e.reasons}
        except Exception as e:                       # backend down etc.: record, do not crash the batch
            return {"rejected": [f"{type(e).__name__}: {e}"]}

    def _replay_cfg(self, tag, case, base, obj, test):
        spec = copy.deepcopy(case["task_spec"])
        positions = {k: (v["pos"][0], v["pos"][1], v["yaw"]) for k, v in base["objects"].items()}
        overrides, clipped = None, False
        if test["type"] == "skill_param":
            lo, hi = self.cards[test["skill"]]["params"][test["param"]]["range"]
            clipped = not lo <= float(test["value"]) <= hi
            overrides = {test["skill"]: {test["param"]: float(test["value"])}}
        elif test["type"] == "alternative_skill":
            overrides = {"grasp_for": {obj: test["card"]}}
        else:
            o = next(o for o in spec["objects"] if o["id"] == obj)
            base_v = o["params"].get(test["param"], DEFAULTS[o["template"]].get(test["param"]))
            want = float(test["value"]) if "value" in test else float(base_v) * float(test["factor"])
            lo, hi = RANGES[o["template"]][test["param"]]
            clipped = not lo - 1e-9 <= want <= hi + 1e-9
            o["params"][test["param"]] = want
        return self._cfg(tag, spec, overrides=overrides, positions=positions, perturb=case.get("perturb")), clipped

    def run(self, cases: list[dict]) -> dict:
        try:
            return self._run(cases)
        finally:
            self.pool.close()

    def _run(self, cases):
        # 1. reproduce
        jobs = [(self._cfg(f"case{i}_base", c["task_spec"], perturb=c.get("perturb")), c["seed"])
                for i, c in enumerate(cases)]
        list(self.pool.map(jobs))
        items = []
        for i, c in enumerate(cases):
            base = self._record(f"case{i}_base", c["seed"])
            st = next((s for s in base["subtasks"] if not s.get("ok") and s.get("label") and s.get("trace")), None)
            items.append({"i": i, "case": c, "base": base, "st": st})
        repro = [it for it in items if it["st"]]
        _log(f"{len(repro)}/{len(cases)} failures reproduced by exact replay")
        # 2. diagnose
        for it in repro:
            it["evidence"] = build_evidence(it["base"], it["st"]["obj"], self.cards)
        with ThreadPoolExecutor(self.model_threads) as ex:
            diags = list(ex.map(self._diagnose, [it["evidence"] for it in repro]))
        for it, d in zip(repro, diags):
            it["diagnosis"] = d
        # 3. one-change replays
        jobs, keys = [], []
        for it in repro:
            if "output" not in it["diagnosis"]:
                continue
            hyps = sorted(it["diagnosis"]["output"]["hypotheses"], key=lambda h: -h.get("confidence", 0))
            it["hyps"] = hyps
            for j, h in enumerate(hyps):
                tag = f"case{it['i']}_h{j}"
                try:
                    cfg, h["test_clipped"] = self._replay_cfg(tag, it["case"], it["base"], it["st"]["obj"], h["test"])
                except Exception as e:
                    h["replay_error"] = f"{type(e).__name__}: {e}"
                    continue
                jobs.append((cfg, it["case"]["seed"]))
                keys.append((it, j, tag))
        list(self.pool.map(jobs))
        for it, j, tag in keys:
            rec = self._record(tag, it["case"]["seed"])
            s = next(s for s in rec["subtasks"] if s.get("obj") == it["st"]["obj"])
            h = it["hyps"][j]
            # A failed replay whose change was clipped to the allowed range never created the tested condition:
            # it neither confirms nor rejects the cause.
            h["verdict"] = "confirmed" if s["ok"] else ("inconclusive" if h.get("test_clipped") else "rejected")
            h["replay"] = {"step_ok": s["ok"], "label": s.get("label"), "change": describe_test(h["test"])}
        # 4. ledger and scores
        cases_out = [self._ledger(it) for it in items]
        report = {"run_id": self.run_id, "library": self.library_name, "backend": self.client.backend,
                  "model": self.client.model, "metrics": _metrics(cases_out), "cases": cases_out,
                  "model_stats": self.client.stats, "wall_time_s": round(time.time() - self.t0, 1)}
        (self.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
        return report

    def _ledger(self, it):
        c, base, st = it["case"], it["base"], it["st"]
        out = {"case": it["i"], "seed": c["seed"], "source": c.get("source"), "task": base.get("prompt"),
               "reproduced": bool(st)}
        if not st:
            return out
        ev = it["evidence"]
        obs = ev.get("observed") or {}
        ledger = [{"status": "measured", "text": f"{ev['failed_step']} failed its check"
                   + (f" ({t['check']})" if (t := next((x for x in st['trace'] if not x['ok']), None)) else "")},
                  {"status": "measured", "text": f"perceived {ev['object']['template']}: width "
                   f"{100 * ev['perceived'].get('width', 0):.1f} cm, height {100 * ev['perceived'].get('height', 0):.1f} cm"},
                  {"status": "measured", "text": f"finger width after the grasp "
                   f"{1000 * st['trace'][0]['info'].get('finger_width', 0):.0f} mm; reached the grasp pose: "
                   f"{st['trace'][0]['info'].get('reached')}"}]
        if obs.get("height_ratio") is not None:
            ledger.append({"status": "measured", "text": f"after: moved {100 * obs['moved_xy']:.0f} cm, height ratio "
                           f"{obs['height_ratio']}, {100 * obs['dist_to_target_xy']:.0f} cm from the target"})
        if ev["arm_contacts"]:
            ledger.append({"status": "measured", "text": "arm contacts: " + ", ".join(
                sorted({f"{x['part']}->{x['object']}" for x in ev["arm_contacts"]}))})
        d = it.get("diagnosis", {})
        if "rejected" in d:
            ledger.append({"status": "rejected", "text": f"diagnosis output rejected: {d['rejected']}"})
        for h in it.get("hyps", []):
            ledger.append({"status": "hypothesis", "text": h["cause"], "predicted_label": h["predicted_label"],
                           "confidence": h.get("confidence")})
            if "verdict" in h:
                status = h["verdict"] if h["verdict"] in ("confirmed", "rejected") else "hypothesis"
                note = " [inconclusive: the change was limited by the allowed range]" if h["verdict"] == "inconclusive" else ""
                ledger.append({"status": status, "note": note.strip() or None,
                               "text": f"replay with {h['replay']['change']}{note}: step "
                               f"{'succeeded' if h['replay']['step_ok'] else 'still failed (' + str(h['replay']['label']) + ')'}",
                               "hypothesis": h["cause"]})
        sim_label = st["label"]
        top = it.get("hyps", [{}])[0] if it.get("hyps") else {}
        out.update(evidence=ev, simulator_label=sim_label, summary=d.get("output", {}).get("summary"),
                   hypotheses=it.get("hyps", []), ledger=ledger, model_meta=d.get("meta"),
                   top_label=top.get("predicted_label"), top_label_correct=top.get("predicted_label") == sim_label,
                   any_label_correct=any(h["predicted_label"] == sim_label for h in it.get("hyps", [])),
                   cluster=f"{ev['object']['template']}/{ev['grasp_used']}")
        (self.out / "ledgers" / f"case{it['i']:03d}.json").write_text(json.dumps(out, indent=1, default=float))
        return out


def _metrics(cases):
    rep = [c for c in cases if c.get("reproduced") and c.get("hypotheses")]
    n = len(rep)
    top = sum(c["top_label_correct"] for c in rep)
    hyps = [h for c in rep for h in c["hypotheses"] if "verdict" in h]
    conf = sum(h["verdict"] == "confirmed" for h in hyps)
    inconc = sum(h["verdict"] == "inconclusive" for h in hyps)
    top_conf = sum(c["hypotheses"][0].get("verdict") == "confirmed" for c in rep)
    by_label = collections.defaultdict(lambda: [0, 0])
    for c in rep:
        by_label[c["simulator_label"]][0] += c["top_label_correct"]
        by_label[c["simulator_label"]][1] += 1
    return {"cases": len(cases), "reproduced": sum(bool(c.get("reproduced")) for c in cases), "diagnosed": n,
            "top1_label_accuracy": round(top / max(n, 1), 3), "top1_ci95": wilson(top, n),
            "any_hypothesis_label": round(sum(c["any_label_correct"] for c in rep) / max(n, 1), 3),
            "hypotheses_tested": len(hyps), "confirmed": conf, "rejected": len(hyps) - conf - inconc,
            "inconclusive_range_limited": inconc,
            "cases_with_a_confirmed_cause": sum(any(h.get("verdict") == "confirmed" for h in c["hypotheses"]) for c in rep),
            "top_hypothesis_confirmed": top_conf,
            "by_simulator_label": {k: f"{v[0]}/{v[1]}" for k, v in by_label.items()}}
