"""Revise a model-written skill that fails on new conditions, without breaking where it already works.

The same lifecycle as creating a skill, started from an existing card instead of from nothing:

    before    the failing request with the current library: how often it fails (a rare failure is not revised)
    settings  the card's own code with other parameter values (search + Optuna on the failing scenes)
    rewrite   if no setting is enough: the model revises the code, given the code, the failing evidence
              (perception + proprioception) and the part-level motion API; a few strategies, searched and tuned
    gate      FIX: failing request, held-out scenes, new vs current library (McNemar, improve, >= 70%)
              HOLD: the scenes the old card was verified on, new vs old: not significantly worse, >= 70%
              REGRESSION: the fixed regression tasks unchanged
    promote   a NEW card beside the old one (lineage: parent card + version, why, evidence). The old card stays;
              both get evidence for the object's shape group and the router uses whichever measured better.

Never replace, add a version and route: a fix for short drawers cannot silently break long ones, because it must
hold on the long ones to be promoted, and the old card is still there.
"""
from __future__ import annotations

import collections
import copy
import json
import time

import numpy as np

from .create_skill import CAPABILITIES, SkillCreator, _log
from .search.explore import ALPHA, REGRESSION_TASKS, mcnemar_p
from .agents.llm import ModelRejected
from .runner import ROOT, wilson
from .skills.library import load_cards, save_card
from .tasks.splits import split_seeds


class SkillReviser(SkillCreator):
    def __init__(self, card_name, task_spec, program, obj=None, target=None, workers=None, client=None,
                 run_id=None, min_fail_rate=None, **kw):
        lib = load_cards()
        self.old = copy.deepcopy(lib[card_name])
        self.kind = self.old["kind"]
        # every card of this kind is left out when a candidate runs, so the candidate is the one used
        self.base_lib = {n: c for n, c in lib.items() if c.get("kind") != self.kind}
        self.min_fail_rate = min_fail_rate
        super().__init__(self.kind, workers=workers, client=client, task_spec=task_spec, program=program, obj=obj,
                         target=target, run_id=run_id or time.strftime(f"%Y%m%d-%H%M%S_revise_{card_name}"), **kw)
        self.report.update(mode="revise", parent={"card": card_name, "version": self.old.get("version", 1)})
        self.orig_task, self.orig_spec, self.orig_program = self._original()

    # ------------------------------------------------------------------ where the old card was verified
    def _original(self):
        """The request the old card was created and gated on: its lineage task spec, else the named task."""
        from .tasks.specs import TASKS
        lin = (self.old.get("lineage") or [{}])[0]
        prog = lin.get("program") or (CAPABILITIES.get(self.kind) or {}).get("program", "")
        if lin.get("task_spec"):
            spec = lin["task_spec"]
            obj = next((g["obj"] for g in spec["goal"]), None)
            return None, spec, prog.format(obj=obj, target=None) if "{" in prog else prog
        name = lin.get("task") if lin.get("task") in TASKS else (self.kind if self.kind in TASKS else None)
        if name is None:
            return None, None, None
        from .tasks.specs import make_task
        goal = make_task(name, seed=0)["goal"][0]
        return name, None, prog.format(obj=goal["obj"], target=goal.get("target"))

    def _orig_jobs(self, tag, card, seeds):
        from .tasks.specs import make_task
        jobs = []
        for s in seeds:
            t = make_task(self.orig_task, seed=s) if self.orig_task else make_task("spec", seed=s, task_spec=self.orig_spec)
            cfg = self._cfg(tag, card)
            cfg["task_spec"] = {**t, "program": self.orig_program}
            jobs.append((cfg, s))
        res = {r["seed"]: r for r in self.pool.map(jobs)}
        self.report["trials_simulated"] += len(jobs)
        return [res[s] for s in seeds]

    # ------------------------------------------------------------------ plumbing
    def _cfg(self, tag, card=None, params=None, perturb=None, task=None, program=True):
        cfg = super()._cfg(tag, card, params, perturb, task, program)
        if card is not None:
            cfg["library"] = self.base_lib
        return cfg

    # ------------------------------------------------------------------ phases
    def before(self):
        seeds = split_seeds("train", 8)
        res = self._run_program("before", None, None, seeds)
        recs = self._records("before", seeds)
        k = sum(r["success"] for r in res)
        labels = collections.Counter(l for r in res for l in r["labels"])
        ph = {"success": k, "trials": len(res), "labels": dict(labels), "missing_capability": None,
              "program_error": None, "evidence": self._evidence(recs)}
        hi = wilson(len(res) - k, len(res))[1]
        ph["rare"] = bool(self.min_fail_rate and hi < self.min_fail_rate)
        self.report["phases"]["before"] = ph
        _log(f"revise {self.old['name']}: current library {k}/{len(res)} on the failing request ({dict(labels)})")
        self._save()
        return ph, None

    def _evidence(self, recs) -> str:
        """Observable facts of the failed trials: labels, how far the parts moved (perception), what the hand did
        (the part-level motions' own reports and the hand's path: proprioception)."""
        lines = []
        for r in [r for r in recs if not r["success"]][:6]:
            s = next((s for s in r["subtasks"] if s.get("obj") == self.obj), r["subtasks"][0])
            info = s["trace"][0]["info"] if s.get("trace") else {}
            bits = [f"label {s.get('label')}"]
            if s.get("final_measured") is not None:
                bits.append(f"goal measured at the end of the episode {s['final_measured']:.3f} (needed {self._goal_text()})")
            if s.get("parts_moved_cm"):
                bits.append("parts moved (perceived after release, cm) " + json.dumps(s["parts_moved_cm"]))
            o = s.get("observed") or {}
            if o.get("moved_xy") is not None:
                bits.append(f"object moved {100 * o['moved_xy']:.1f} cm")
            if info.get("part_motions"):
                bits.append("part-level motions " + json.dumps(info["part_motions"]))
            if info.get("hand_path_mm"):
                bits.append("hand path (mm, after each motion) " + json.dumps(info["hand_path_mm"][:14]))
            if info.get("error"):
                bits.append(f"code error {info['error']}")
            lines.append("  - " + "; ".join(bits))
        return "\n".join(lines)

    def _goal_text(self):
        g = next((g for g in self._task(0)["goal"] if g.get("obj") == self.obj), {})
        return f"{g.get('pred')} {g.get('op', '')} {g.get('value', g.get('target', ''))}"

    def _orig_evidence(self) -> str:
        if not (self.orig_task or self.orig_spec):
            return ""
        seeds = split_seeds("train", 3, start=60)
        self._orig_jobs("orig_view", None, seeds)
        recs = self._records("orig_view", seeds)
        ok = [r for r in recs if r["success"]]
        moved = [next((s.get("parts_moved_cm") for s in r["subtasks"] if s.get("parts_moved_cm")), None) for r in ok]
        return (f"On the scenes it was verified on it succeeds {len(ok)}/{len(recs)}"
                + (f"; parts moved there (cm): {[m for m in moved if m]}" if any(moved) else "") + ".")

    def propose(self, before, feedback=None, others=None):
        obj, target = self._examples()
        goal = next((g for g in self._task(0)["goal"] if g.get("obj") == self.obj), None)
        why = (f"the library's skill {self.old['name']} for '{self.kind}' fails on a new request "
               f"('{(self.task_spec or {}).get('prompt') or self.task_label}'): {before['success']}/{before['trials']} "
               f"succeed, labels {before['labels']}. Revise it so it also works here, without breaking the cases it "
               f"already handles. {self.report.get('orig_view', '')}\nObserved in failed trials:\n{before['evidence']}\n"
               f"Current code of {self.old['name']}:\n{self.old['code']}\nCurrent parameter values: "
               f"{ {k: v['default'] for k, v in self.old['params'].items()} }")
        res = self.client.complete("propose_primitive", {"capability": f"{self.kind}: {self.cap['description']}",
                                                         "why": why, "obj": obj, "target": target, "goal": goal,
                                                         "feedback": feedback, "other_strategies": others})
        out = res.output
        name = out["name"] if out["name"].startswith(self.kind) else f"{self.kind}_{out['name']}"
        card = {"name": name, "version": 1, "kind": self.kind, "impl": "generated", "code": out["code"],
                "description": out.get("explanation", ""), "check": out.get("check", ""),
                "params": {k: {"default": v["default"], "range": v["range"], "meaning": v.get("meaning", "")}
                           for k, v in out["params"].items()}}
        return card, res.meta

    def gate(self, card):
        # FIX: the failing request, held-out scenes
        ds = split_seeds("dev", self.n_dev, start=40)
        after = self._run_program("gate_dev", card, None, ds)
        before = self._run_program("gate_dev_before", None, None, ds)
        a_ok, b_ok = [bool(r["success"]) for r in after], [bool(r["success"]) for r in before]
        p = mcnemar_p(b_ok, a_ok)
        fix = {"success": sum(a_ok), "trials": len(a_ok), "before": sum(b_ok), "p_value": p,
               "ci95": wilson(sum(a_ok), len(a_ok)),
               "wasted_frac": round(float(np.mean([r.get("wasted_frac", 0.0) for r in after])), 3),
               "pass": p < ALPHA and sum(a_ok) / len(a_ok) >= 0.7}
        # HOLD: where the old card was verified, the new one must not be worse
        hold = {"pass": True, "note": "no recorded original request"}
        if self.orig_task or self.orig_spec:
            hs = split_seeds("dev", self.n_dev, start=70)
            na = self._orig_jobs("hold_new", card, hs)
            ob = self._orig_jobs("hold_old", None, hs)
            n_ok, o_ok = [bool(r["success"]) for r in na], [bool(r["success"]) for r in ob]
            worse_p = mcnemar_p(n_ok, o_ok)              # small = the new card is significantly worse
            hold = {"success": sum(n_ok), "trials": len(n_ok), "before": sum(o_ok), "p_worse": worse_p,
                    "pass": worse_p >= ALPHA and sum(n_ok) / len(n_ok) >= 0.7}
        reg = {"tasks": {}, "pass": True}
        for t in REGRESSION_TASKS:
            seeds = list(range(4))
            r0 = list(self.pool.map([(self._cfg(f"reg0_{t}", task=t), s) for s in seeds]))
            r1 = list(self.pool.map([(self._cfg(f"reg1_{t}", card, task=t), s) for s in seeds]))
            self.report["trials_simulated"] += 8
            g0, g1 = sum(r["goals_met"] for r in r0), sum(r["goals_met"] for r in r1)
            reg["tasks"][t] = {"before_goals": g0, "after_goals": g1, "pass": g1 >= g0}
            reg["pass"] &= g1 >= g0
        g = {"held_out": fix, "hold": hold, "no_regression": reg, "pass": fix["pass"] and hold["pass"] and reg["pass"]}
        self.report["phases"]["gate"] = g
        _log(f"gate: fix {fix['success']}/{fix['trials']} vs {fix['before']}/{fix['trials']} (p={p:.3g}) "
             f"{'pass' if fix['pass'] else 'FAIL'}; hold on original {hold.get('success')}/{hold.get('trials')} vs "
             f"{hold.get('before')} {'pass' if hold['pass'] else 'FAIL'}; regression {'pass' if reg['pass'] else 'FAIL'}")
        self._save()
        return g

    def promote(self, card, meta, attempts, gate, source="revision"):
        from .skills.router import load_performance, save_performance, shape_group, with_evidence
        lib = load_cards()
        while card["name"] in lib:                        # a new card beside the old one, never an overwrite
            card["name"] = f"{card['name']}_r{attempts}" if not card["name"][-1].isdigit() else card["name"] + "x"
        fix, hold = gate["held_out"], gate["hold"]
        card["lineage"] = [{"version": 1, "source": source, "run_id": self.run_id, "date": time.strftime("%Y-%m-%d"),
                            "parent": self.report["parent"], "written_by": meta, "attempts": attempts,
                            "capability": self.kind, "task": self.task_label, "task_spec": self.task_spec,
                            "program": self.program, "description": self.cap["description"],
                            "why": f"{self.report['parent']['card']} failed {self.report['phases']['before']['labels']} "
                                   f"on '{(self.task_spec or {}).get('prompt') or self.task_label}'",
                            "trials_simulated": self.report["trials_simulated"],
                            "results": {"fix": [fix["before"], fix["success"], fix["trials"]], "p_value": fix["p_value"],
                                        "hold": [hold.get("before"), hold.get("success"), hold.get("trials")]}}]
        path = save_card(card)
        # Evidence for both cards on the object's shape group: the router now prefers the better-measured one.
        recs = self._records("gate_dev", split_seeds("dev", self.n_dev, start=40))
        desc = next((r["percepts"].get(self.obj) for r in recs if r.get("percepts", {}).get(self.obj)), None)
        if desc:
            g = shape_group(desc)
            perf = load_performance()
            new_k = fix["success"] + (hold.get("success") or 0)
            old_k = fix["before"] + (hold.get("before") or 0)
            n = fix["trials"] + (hold.get("trials") or 0)
            perf = with_evidence(perf, card["name"], g, new_k, n, self.run_id)
            perf = with_evidence(perf, self.old["name"], g, old_k, n, self.run_id)
            save_performance(perf)
            self.report["evidence"] = {"group": g, card["name"]: [new_k, n], self.old["name"]: [old_k, n]}
        self.report["promoted"] = {"card": card["name"], "path": str(path.relative_to(ROOT)),
                                   "parent": self.report["parent"]}
        _log(f"PROMOTED {card['name']} beside {self.old['name']} -> {path.relative_to(ROOT)}")
        self._save()

    def run(self):
        try:
            before, _ = self.before()
            if before["success"] == before["trials"]:
                self.report["outcome"] = "not_reproduced"
                return self.report
            if before["rare"]:
                self.report["outcome"] = "rare_failure"
                return self.report
            self.report["orig_view"] = self._orig_evidence()
            # 1. settings: the same code with other parameter values
            tuned = copy.deepcopy(self.old)
            tuned["name"] = f"{self.old['name']}_tuned"
            tuned.pop("lineage", None)
            rows, cands = self.search(tuned, "settings")
            if rows[0]["success"] == rows[0]["trials"]:
                params, _ = self.tune(tuned, rows, cands)
                for k, v in params.items():
                    tuned["params"][k]["default"] = round(v, 4)
                g = self.gate(tuned)
                if g["pass"]:
                    self.promote(tuned, {"backend": "search", "note": "parameters only"}, 0, g,
                                 source="revision (parameters)")
                    self.report["outcome"] = "promoted"
                    return self.report
                _log("new settings alone did not pass the gates; asking for a revised motion")
            # 2. rewrite: revised code, a few strategies, then one more revision with the gate's feedback
            feedback = None
            self.report["outcome"] = "no_working_proposal"
            for attempt in (1, 2):
                k = self.n_strategies if attempt == 1 else 1
                options, others = [], []
                rejected = []
                for i in range(k):
                    try:
                        card, meta = self.propose(before, feedback, others or None)
                    except ModelRejected as e:           # an invalid proposal is a failed strategy, not a crash
                        rejected.append(str(e))
                        self.report["phases"].setdefault("proposals", []).append(
                            {"attempt": attempt, "option": i, "rejected": str(e), "feedback": feedback})
                        _log(f"{'revision'} {attempt}{'abc'[i] if k > 1 else ''} rejected: {str(e)[:200]}")
                        continue
                    if card["name"] in [o[0]["name"] for o in options] or card["name"] == self.old["name"]:
                        card["name"] = f"{card['name']}_{attempt}{'abc'[i]}"
                    tag = f"r{attempt}{'abc'[i]}"
                    self.report["phases"].setdefault("proposals", []).append(
                        {"attempt": attempt, "option": i, "card": copy.deepcopy(card), "meta": meta, "feedback": feedback})
                    _log(f"revision {tag} ({meta['backend']}): {card['name']} - {card['description'][:110]}")
                    rows, cands = self.search(card, tag)
                    options.append((card, meta, rows, cands, tag))
                    others.append(f"{card['name']}: {card['description'][:160]}")
                if not options:
                    feedback = ("every proposal was rejected before simulation (fix these and follow the motion API "
                                "rules exactly): " + " | ".join(r[:300] for r in rejected))
                    continue
                card, meta, rows, cands, tag = max(options, key=lambda o: (o[2][0]["success"], sum(r["success"] > 0 for r in o[2])))
                if rows[0]["success"] == 0:
                    feedback = self.feedback(rows, tag)
                    continue
                params, _ = self.tune(card, rows, cands)
                for kk, v in params.items():
                    card["params"][kk]["default"] = round(v, 4)
                g = self.gate(card)
                if g["pass"]:
                    self.promote(card, meta, attempt, g)
                    self.report["outcome"] = "promoted"
                    return self.report
                self.report["outcome"] = "failed_gates"
                feedback = self.gate_feedback(card, g) + (
                    f"\nOn the scenes the old skill was verified on, the revision succeeded "
                    f"{g['hold'].get('success')}/{g['hold'].get('trials')} (the old skill {g['hold'].get('before')}); "
                    "it must keep working there." if not g["hold"]["pass"] else "")
            return self.report
        finally:
            self._save()
            self.pool.close()
