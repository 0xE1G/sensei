"""Create a missing skill: the model proposes a primitive, simulation tunes and verifies it, the gates decide.

    before    the task with the current library fails (and the program reports the missing capability)
    propose   model job "propose_primitive": motion-API code + parameter ranges + check
    search    candidates: the proposal's defaults + uniform samples over its ranges, a few train scenes each
    feedback  if nothing succeeds, a summary of what happened goes back to the model once (Eureka-style reflection)
    tune      Optuna on the best candidates (perturbed physics), confirmed on fresh scenes
    gate      held-out scenes: after (program using the new card) vs before (current library), paired McNemar test,
              >= 70% success; regression tasks unchanged
    promote   skill_library/<name>.json: kind = the capability, impl "generated", the code, tuned defaults, lineage
              (source "model-written primitive", who wrote it, attempts, search and gate evidence)

The task is a named task (scripts/create_skill.py) or a typed request's task spec with the plan program that asked
for the capability (the studio: program use("<capability>", obj, target) found no skill of that kind).
"""
from __future__ import annotations

import collections
import copy
import json
import time

import numpy as np

from .runner import ROOT, TrialPool, trial_cfg, wilson
from .search.explore import ALPHA, REGRESSION_TASKS, TUNE_PERTURB, mcnemar_p
from .skills.library import LIB_DIR, save_card
from .tasks.splits import split_seeds

WASTE_WEIGHT = 0.1     # tuning objective: success rate - 0.1 x wasted-motion fraction (a tie-breaker)

CAPABILITIES = {
    "open_drawer": {"kind": "open_drawer",
                    "description": "pull a drawer open: grasp its handle and slide the drawer out along the "
                                   "front's outward normal",
                    "program": "use('open_drawer', find_one(id='{obj}'))"},
    "open_door": {"kind": "open_door",
                  "description": "swing a hinged door open: grasp its handle and pull it along the arc around the "
                                 "hinge (a vertical axis at one side edge of the front)",
                  "program": "use('open_door', find_one(id='{obj}'))"},
    "push": {"kind": "push",
             "description": "move an object along the table to a target location without grasping it (the object "
                            "cannot be grasped: no grasp in the library fits it)",
             "program": "push(find_one(id='{obj}'), find_one(id='{target}'))"},
}


def _log(msg):
    print(f"[create_skill] {msg}", flush=True)


class SkillCreator:
    def __init__(self, capability, task_name=None, workers=None, client=None, n_candidates=40, search_seeds=3,
                 tune_trials=16, tune_seeds=4, n_dev=16, run_id=None, seed=0, n_strategies=3,
                 task_spec=None, program=None, obj=None, target=None, description=None):
        """task_spec + program + obj/target: a typed request whose plan called use(capability, obj, target);
        description: what the capability must do (default: from CAPABILITIES, else from the request)."""
        from .agents.llm import ModelClient
        self.task_spec = ({k: v for k, v in task_spec.items() if k not in ("program", "program_source")}
                          if task_spec else None)
        self.cap = CAPABILITIES.get(capability) or {
            "kind": capability, "description": description or (
                "move the object as the request asks, without the library's grasp/place skills "
                f"(request: {(self.task_spec or {}).get('prompt') or task_name!r})")}
        self.capability, self.task_name = capability, task_name
        self.task_label = task_name or (self.task_spec or {}).get("name", "spec")
        self.client = client or ModelClient()
        self.n_candidates, self.search_seeds = n_candidates, search_seeds
        self.tune_trials, self.tune_seeds, self.n_dev = tune_trials, tune_seeds, n_dev
        self.n_strategies = n_strategies
        self.rng = np.random.default_rng(seed)
        self.run_id = run_id or time.strftime(f"%Y%m%d-%H%M%S_create_{capability}")
        self.out = ROOT / "runs" / self.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool = TrialPool(workers)
        goal = self._task(0)["goal"][0]
        self.obj, self.target = obj or goal["obj"], target if (obj or target) else goal.get("target")
        self.program = program or self.cap["program"].format(obj=self.obj, target=self.target)
        self.report = {"run_id": self.run_id, "capability": capability, "task": self.task_label,
                       "description": self.cap["description"], "program": self.program, "phases": {},
                       "trials_simulated": 0}
        self.t0 = time.time()

    # ------------------------------------------------------------------ plumbing
    def _task(self, seed):
        """The task for one scene: a named task's spec, or the typed request's spec (placement varies by seed)."""
        from .tasks.specs import make_task
        if self.task_spec:
            return make_task("spec", seed=seed, task_spec=self.task_spec)
        return make_task(self.task_name, seed=seed)

    def _cfg(self, tag, card=None, params=None, perturb=None, task=None, program=True):
        extra = {card["name"]: card} if card else None
        overrides = {card["name"]: params} if (card and params) else None
        if task is None and self.task_spec:
            task, kw = "spec", {"task_spec": self.task_spec}
        else:
            task, kw = task or self.task_name, {}
        return trial_cfg(task, self.out, save_failures=False, success_sample=0, save_video=False,
                         save_poses=False, tag=tag, extra_cards=extra, overrides=overrides, perturb=perturb,
                         task_kwargs=kw)

    def _run_program(self, tag, card, params, seeds, perturb=None):
        """The task's own spec + the capability program, as a task_spec per seed."""
        jobs = []
        for s in seeds:
            spec = {**self._task(s), "program": self.program}
            cfg = self._cfg(tag, card, params, perturb)
            cfg["task_spec"] = spec
            jobs.append((cfg, s))
        res = {r["seed"]: r for r in self.pool.map(jobs)}
        self.report["trials_simulated"] += len(jobs)
        return [res[s] for s in seeds]

    def _records(self, tag, seeds):
        return [json.loads((self.out / "trials" / f"{tag}_{s:06d}.json").read_text()) for s in seeds]

    def _save(self):
        self.report["wall_time_s"] = round(time.time() - self.t0, 1)
        (self.out / "report.json").write_text(json.dumps(self.report, indent=1, default=float))

    # ------------------------------------------------------------------ phases
    def before(self):
        seeds = split_seeds("train", 8)
        res = list(self.pool.map([(self._cfg("before"), s) for s in seeds]))
        self.report["trials_simulated"] += len(seeds)
        labels = collections.Counter(l for r in res for l in r["labels"])
        # And with a program that asks for the capability: it must report it as missing.
        prog = self._run_program("before_program", None, None, seeds[:1])[0]
        rec = self._records("before_program", seeds[:1])[0]
        ph = {"success": sum(r["success"] for r in res), "trials": len(res), "labels": dict(labels),
              "missing_capability": rec.get("missing_capability"), "program_error": rec.get("program_error")}
        self.report["phases"]["before"] = ph
        _log(f"before: current library {ph['success']}/{ph['trials']} ({dict(labels)}); a program asking for "
             f"{self.capability}: {ph['missing_capability'] or ph['program_error']}")
        self._save()
        return ph, prog

    def _examples(self):
        from .perception.perceive import Perception
        from .sim.scene import build_scene
        for s in range(20):                     # the first layout that fits on the table
            try:
                scene = build_scene(self._task(s), seed=s)
                break
            except RuntimeError:
                continue
        per = Perception(scene)
        d = per.perceive([o for o in (self.obj, self.target) if o])
        per.close()
        keep = ("center", "width", "height", "top_z", "axis_yaw", "extent_major", "extent_minor", "parts")
        view = lambda o: ({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d[o].items() if k in keep}
                          if o and d.get(o) else None)
        return view(self.obj), view(self.target)

    def propose(self, before, feedback=None, others=None):
        obj, target = self._examples()
        why = (f"the task '{self.task_label}' fails with the current library ({before['success']}/{before['trials']}; "
               f"grasp failures {before['labels']}); a plan asking for {self.capability} reports: "
               f"{before['missing_capability'] or before['program_error']}")
        goal = next((g for g in self._task(0)["goal"] if g.get("obj") == self.obj), None)
        res = self.client.complete("propose_primitive", {"capability": f"{self.capability}: {self.cap['description']}",
                                                         "why": why, "obj": obj, "target": target, "goal": goal,
                                                         "feedback": feedback, "other_strategies": others})
        out = res.output
        card = {"name": out["name"] if out["name"].startswith(self.capability) else f"{self.capability}_{out['name']}",
                "version": 1, "kind": self.capability, "impl": "generated", "code": out["code"],
                "description": out.get("explanation", ""), "check": out.get("check", ""),
                "params": {k: {"default": v["default"], "range": v["range"], "meaning": v.get("meaning", "")}
                           for k, v in out["params"].items()}}
        return card, res.meta

    def search(self, card, attempt):
        names = list(card["params"])
        base = {k: v["default"] for k, v in card["params"].items()}
        cands = [dict(base)] + [{k: float(self.rng.uniform(*card["params"][k]["range"])) for k in names}
                                for _ in range(self.n_candidates)]
        seeds = split_seeds("train", self.search_seeds, start=10)
        rows = []
        jobs = []
        for i, c in enumerate(cands):
            for s in seeds:
                cfg = self._cfg(f"a{attempt}_cand{i}", card, c)
                cfg["task_spec"] = {**self._task(s), "program": self.program}
                jobs.append((cfg, s))
        res = {(r["tag"], r["seed"]): r for r in self.pool.map(jobs)}
        self.report["trials_simulated"] += len(jobs)
        for i, c in enumerate(cands):
            rr = [res[(f"a{attempt}_cand{i}", s)] for s in seeds]
            rows.append({"id": i, "params": {k: round(v, 4) for k, v in c.items()},
                         "wasted": round(float(np.mean([r.get("wasted_frac", 0.0) for r in rr])), 3),
                         "success": sum(r["success"] for r in rr), "trials": len(rr),
                         "labels": dict(collections.Counter(l for r in rr for l in r["labels"]))})
        rows.sort(key=lambda r: (-r["success"], r["wasted"]))     # equal success: less wasted motion first
        self.report["phases"].setdefault("search", []).append({"attempt": attempt, "seeds": seeds, "candidates": rows})
        _log(f"search (attempt {attempt}): best {rows[0]['success']}/{rows[0]['trials']}, "
             f"{sum(r['success'] > 0 for r in rows)}/{len(rows)} candidates succeed at least once")
        self._save()
        return rows, cands

    def feedback(self, rows, attempt):
        """What happened, from observable evidence, for the model's second try."""
        recs = []
        for r in rows[:5]:
            recs += self._records(f"a{attempt}_cand{r['id']}", split_seeds("train", self.search_seeds, start=10))
        obs = [s.get("observed") or {} for rec in recs for s in rec["subtasks"]]
        errs = collections.Counter(s["trace"][0]["info"].get("error") for rec in recs for s in rec["subtasks"]
                                   if s.get("trace") and s["trace"][0]["info"].get("error"))
        moved = [o.get("moved_xy") for o in obs if o.get("moved_xy") is not None]
        dist = [o.get("dist_to_target_xy") for o in obs if o.get("dist_to_target_xy") is not None]
        parts = [s.get("parts_moved_cm") or {} for rec in recs for s in rec["subtasks"]]
        if any(parts):
            names = sorted({k for p in parts for k in p})
            return (f"- {len(recs)} trials of the 5 best candidates, none met the goal.\n- code errors: {dict(errs) or 'none'}\n"
                    + "".join(f"- {n} moved on average {np.mean([p.get(n, 0) for p in parts]):.1f} cm (perception)\n" for n in names)
                    + f"- labels: {dict(collections.Counter(l for r in rows for l in r['labels']))}")
        return (f"- {len(recs)} trials of the 5 best candidates, none reached the target.\n"
                f"- code errors: {dict(errs) or 'none'}\n"
                f"- object moved on average {100 * np.mean(moved):.1f} cm; ended {100 * np.mean(dist):.1f} cm from "
                f"the target centre on average\n- labels: {dict(collections.Counter(l for r in rows for l in r['labels']))}"
                if moved else f"- the object never moved; code errors: {dict(errs) or 'none'}")

    def gate_feedback(self, card, gate):
        """The proposal worked sometimes but not reliably: its code, its tuned values, and the held-out failures."""
        ds = split_seeds("dev", self.n_dev, start=40)
        fails = [r for r in self._records("gate_dev", ds) if not r["success"]]
        lines = []
        for r in fails[:8]:
            s = r["subtasks"][0]
            o = s.get("observed") or {}
            info = s["trace"][0]["info"] if s.get("trace") else {}
            b, a = (o.get("before") or {}).get("center"), (o.get("after") or {}).get("center")
            t = (r.get("percepts", {}).get(self.target) or {}).get("center")
            geo = ""
            if b and a and t:
                d = np.subtract(t, b)
                u = d / max(np.linalg.norm(d), 1e-6)
                disp = np.subtract(a, b)
                along = float(disp @ u)
                side = float(np.linalg.norm(disp - along * u))
                geo = (f"moved {100 * along:.0f} cm along the line to the target (needed {100 * np.linalg.norm(d):.0f})"
                       f" and {100 * side:.0f} cm sideways")
            fw = info.get("finger_width_mm_while_moving")
            lines.append(f"  - {s['label']}: {geo or 'object moved ' + str(round(100 * (o.get('moved_xy') or 0))) + ' cm'}"
                         + (f"; finger opening during the moves (mm): {fw}" if fw else "")
                         + (f"; code error {info.get('error')}" if info.get("error") else ""))
        h = gate["held_out"]
        waste = (f"\nWasted motion: {100 * h.get('wasted_frac', 0):.0f}% of the motion time came after the object had "
                 "already reached the target; use look() to stop once it has arrived." if h.get("wasted_frac", 0) > 0.1 else "")
        return (f"It succeeded on {h['success']}/{h['trials']} held-out scenes (needs 70%).{waste} Failures:\n"
                + "\n".join(lines) + f"\nTuned values: { {k: v['default'] for k, v in card['params'].items()} }\n"
                f"Your previous code:\n{card['code']}\nRevise the motion so it works reliably.")

    def tune(self, card, rows, cands):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        names = list(card["params"])
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        for r in rows[:3]:
            study.enqueue_trial(cands[r["id"]])
        seeds = split_seeds("train", self.tune_seeds, start=20)
        history = []
        while len(history) < self.tune_trials:
            ts = [study.ask() for _ in range(min(3, self.tune_trials - len(history)))]
            for t in ts:
                p = {k: t.suggest_float(k, *card["params"][k]["range"]) for k in names}
                rr = self._run_program(f"tune{t.number}", card, p, seeds, TUNE_PERTURB)
                # success first; wasted motion only breaks ties (penalty < one success out of tune_seeds)
                v = sum(r["success"] for r in rr) / len(rr) - WASTE_WEIGHT * float(np.mean([r.get("wasted_frac", 0.0) for r in rr]))
                study.tell(t, v)
                history.append({"trial": t.number, "params": {k: round(x, 4) for k, x in p.items()}, "value": v})
        best = max(history, key=lambda h: (h["value"], -h["trial"]))
        cseeds = split_seeds("train", 8, start=30)
        conf = self._run_program("confirm", card, best["params"], cseeds, TUNE_PERTURB)
        k = sum(r["success"] for r in conf)
        self.report["phases"]["tune"] = {"history": history, "best": best, "confirm": [k, len(conf)]}
        _log(f"tune: best {best['value']:.2f} on tuning scenes, {k}/{len(conf)} on fresh perturbed scenes")
        self._save()
        return best["params"], k / len(conf)

    def gate(self, card):
        ds = split_seeds("dev", self.n_dev, start=40)
        after = self._run_program("gate_dev", card, None, ds)
        before = list(self.pool.map([(self._cfg("gate_dev_before"), s) for s in ds]))
        self.report["trials_simulated"] += 2 * len(ds)
        a_ok = [bool(r["success"]) for r in after]
        b_ok = [bool(r["success"]) for r in before]
        p = mcnemar_p(b_ok, a_ok)
        waste = round(float(np.mean([r.get("wasted_frac", 0.0) for r in after])), 3)
        held = {"success": sum(a_ok), "trials": len(a_ok), "ci95": wilson(sum(a_ok), len(a_ok)), "wasted_frac": waste,
                "before": sum(b_ok), "p_value": p, "pass": p < ALPHA and sum(a_ok) / len(a_ok) >= 0.7}
        reg = {"tasks": {}, "pass": True}
        for t in REGRESSION_TASKS:
            seeds = list(range(4))
            r0 = list(self.pool.map([(self._cfg(f"reg0_{t}", task=t), s) for s in seeds]))
            r1 = list(self.pool.map([(self._cfg(f"reg1_{t}", card, task=t), s) for s in seeds]))
            self.report["trials_simulated"] += 8
            g0, g1 = sum(r["goals_met"] for r in r0), sum(r["goals_met"] for r in r1)
            ok = g1 >= g0
            reg["tasks"][t] = {"before_goals": g0, "after_goals": g1, "pass": ok}
            reg["pass"] &= ok
        g = {"held_out": held, "no_regression": reg, "pass": held["pass"] and reg["pass"]}
        self.report["phases"]["gate"] = g
        _log(f"gate: held-out {held['success']}/{held['trials']} vs {held['before']}/{held['trials']} before "
             f"(p={p:.3g}) {'pass' if held['pass'] else 'FAIL'}; regression {'pass' if reg['pass'] else 'FAIL'}")
        self._save()
        return g

    def promote(self, card, meta, attempts, gate):
        card["lineage"] = [{"version": 1, "source": "model-written primitive", "run_id": self.run_id,
                            "date": time.strftime("%Y-%m-%d"), "written_by": meta, "attempts": attempts,
                            "capability": self.capability, "task": self.task_label,
                            "task_spec": self.task_spec, "program": self.program,
                            "description": self.cap["description"],
                            "trials_simulated": self.report["trials_simulated"],
                            "results": {"held_out": [gate["held_out"]["before"], gate["held_out"]["success"],
                                                     gate["held_out"]["trials"]], "p_value": gate["held_out"]["p_value"]}}]
        path = save_card(card)
        self.report["promoted"] = {"card": card["name"], "path": str(path.relative_to(ROOT))}
        _log(f"PROMOTED {card['name']} -> {path.relative_to(ROOT)}")
        self._save()

    def run(self):
        try:
            before, _ = self.before()
            feedback = None
            self.report["outcome"] = "no_working_proposal"
            # Two proposals at most. The second sees what happened to the first: nothing worked in the search, or
            # it worked but failed the gates (its own code + the held-out failures, to revise rather than restart).
            for attempt in (1, 2):
                # Round 1: several genuinely different strategies (Eureka-style sampling); search picks among them.
                # Round 2: one revision of the best, with what happened to it.
                k = self.n_strategies if attempt == 1 else 1
                options, others = [], []
                for i in range(k):
                    card, meta = self.propose(before, feedback, others or None)
                    if card["name"] in [o[0]["name"] for o in options]:
                        card["name"] = f"{card['name']}_{i + 1}"
                    tag = f"{attempt}{'abc'[i]}" if k > 1 else str(attempt)
                    self.report["phases"].setdefault("proposals", []).append(
                        {"attempt": attempt, "option": i, "card": copy.deepcopy(card), "meta": meta,
                         "feedback": feedback})
                    _log(f"proposal {tag} ({meta['backend']}): {card['name']} with params {list(card['params'])}"
                         f" - {card['description'][:110]}")
                    rows, cands = self.search(card, tag)
                    options.append((card, meta, rows, cands, tag))
                    others.append(f"{card['name']}: {card['description'][:160]}")
                # Best option: most successes of its best candidate, then how many candidates work at all.
                card, meta, rows, cands, tag = max(options, key=lambda o: (o[2][0]["success"], sum(r["success"] > 0 for r in o[2])))
                if k > 1:
                    _log(f"search picked {card['name']} among {[o[0]['name'] for o in options]}")
                if rows[0]["success"] == 0:
                    feedback = self.feedback(rows, tag)
                    _log(f"no candidate succeeded; feeding back:\n{feedback}")
                    continue
                params, conf = self.tune(card, rows, cands)
                for k, v in params.items():
                    card["params"][k]["default"] = round(v, 4)
                g = self.gate(card)
                if g["pass"]:
                    self.promote(card, meta, attempt, g)
                    self.report["outcome"] = "promoted"
                    return self.report
                self.report["outcome"] = "failed_gates"
                feedback = self.gate_feedback(card, g)
                _log(f"gates failed; feeding back:\n{feedback}")
            return self.report
        finally:
            self._save()
            self.pool.close()
