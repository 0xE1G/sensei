"""Exploration and repair: from a failure cluster to a verified, promoted skill card.

    before   run the task with the active library on train objects -> failure cluster (object, skill, shape)
    existing first try the library's other cards that fit the object; if one already works, the repair is a
             routing change (evidence in skill_library/performance.json), gated like any other change
    patch    if the step fails after the grasp (drop, tip, slip) or the investigator confirmed causes: search
             parameter patches for the object's shape group (release height, speeds, grip force, grasp depth),
             seeded with the confirmed one-change fixes; smallest best patch is gated (skill_library/patches.json)
    sample   perceive the failing objects, sample feature_pinch candidates, keep those that fit the gripper
             on most of them (geometry) and that the arm can reach (IK)
             (candidate trials run on the failing item's sub-task alone: same object, its target, nothing else)
    families group survivors by the feature they use and how the fingers close; simulate the best few
             of each family on train objects -> success map
    tune     Optuna (TPE) on the best family, seeded with its best candidates, on fixed train seeds
    gate     on the full task; a candidate card is promoted only if it (a) fixes the failure cluster, (b) leaves the other
             tasks unchanged, (c) holds on held-out (dev) objects. Otherwise the next family is tried.
    promote  new card version in skill_library/ with lineage; the old version goes to history/

Every trial is a normal episode record (runs/<run_id>/trials/<tag>_<seed>.json). The full story, including
rejected candidates and failed gates, is in runs/<run_id>/report.json for the web app.
Language-model hooks (Spark naming families, proposing patches) slot in at `sample` and `families`; the
search runs without them.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import pathlib
import time

import numpy as np

from ..runner import ROOT, TrialPool, trial_cfg, wilson
from ..skills import library as L
from ..skills.router import (load_patches, load_performance, save_patches, save_performance,
                             shape_group as group_of, with_evidence)
from ..tasks.planner import fits
from ..tasks.specs import make_task
from ..tasks.splits import split_seeds
from .grasp_sampler import (family_of, geometric_fit, sample_candidates, search_params, shape_group,
                            surface_candidates)
from .reach import ReachChecker

REGRESSION_TASKS = ("sort_by_color", "balls_in_basket", "single_block")
BASELINE_DIR = ROOT / "experiments" / "baselines"
# Tuning and its confirmation run on harder objects than nominal (slippery, heavy), so the objective
# separates candidates that all succeed under nominal conditions.
TUNE_PERTURB = {"friction": [0.5, 1.0], "mass": [1.0, 1.8]}
POST_GRASP = {"drop", "wrong_place", "slip", "lift_check_failed"}
MIN_CLUSTER = 10        # failed cases the paired gate needs
N_DEV_PATCH = 48        # held-out objects for patch gates (patch targets fail less often than missing skills)


def _log(msg):
    print(f"[explore] {msg}", flush=True)


def _rate(results):
    k, n = sum(r["success"] for r in results), len(results)
    gm, gt = sum(r["goals_met"] for r in results), sum(r["goals_total"] for r in results)
    return {"success": k, "trials": n, "rate": round(k / max(n, 1), 3), "ci95": wilson(k, n),
            "goal_rate": round(gm / max(gt, 1), 3), "goal_ci95": wilson(gm, gt)}


def _obj_rate(results, obj):
    """Success of one item's sub-task (the repaired step), as opposed to the whole task."""
    k, n = sum(bool(r["subtask_ok"].get(obj)) for r in results), len(results)
    return {"success": k, "trials": n, "rate": round(k / max(n, 1), 3), "ci95": wilson(k, n)}


ALPHA = 0.05


def mcnemar_p(before_ok, after_ok) -> float:
    """One-sided exact McNemar test that `after` is better than `before` on the same cases (paired)."""
    from math import comb
    b = sum((not x) and y for x, y in zip(before_ok, after_ok))      # fail -> success
    c = sum(x and (not y) for x, y in zip(before_ok, after_ok))      # success -> fail
    n = b + c
    return 1.0 if n == 0 else round(sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n, 6)


def _apart(a, b) -> bool:
    """Two candidates of a family are far enough apart to both be worth simulating."""
    if "angle_deg" in a["params"]:
        return abs((a["params"]["angle_deg"] - b["params"]["angle_deg"] + 180) % 360 - 180) > 30
    return abs(a["params"].get("along", 0) - b["params"].get("along", 0)) > 0.25 or \
        abs(a["params"].get("close_deg", 0) - b["params"].get("close_deg", 0)) > 20


def _library_hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def elongation_bounds(els) -> dict:
    """Shape preconditions beyond size: a card found only on long objects (bars) must not take over round ones of
    the same size (plates), and one found only on round objects must not take over long ones. 20-25% margin."""
    if els and min(els) > 1.5:
        return {"min_elongation": round(0.8 * min(els), 2)}
    if els and max(els) < 1.5:
        return {"max_elongation": round(1.25 * max(els), 2)}
    return {}


class Explorer:
    def __init__(self, task: str = "bowl_on_tray", workers: int | None = None, n_before: int = 24,
                 n_candidates: int = 800, per_family: int = 4, family_seeds: int = 4, max_families: int = 4,
                 tune_trials: int = 32, tune_seeds: int = 6, n_dev: int = 16, n_regression: int = 8,
                 min_feasible: float = 0.75, run_id: str | None = None, seed: int = 0, force: bool = False,
                 library: dict | None = None, performance: dict | None = None, task_kwargs: dict | None = None,
                 patches: dict | None = None, hints: list | None = None, patch_trials: int = 20,
                 perturb: dict | None = None, min_fail_rate: float | None = None):
        """`min_fail_rate`: stop early as "rare_failure" when the valid-failure rate is confidently below it (Wilson
        upper bound < min_fail_rate): the gates need ~10 failed cases, so a rare failure costs a long search that
        cannot pass them. None = off (experiments).
        `library`: the card set every trial uses (default: the active library, snapshotted now).
        `performance`: routing evidence (default: skill_library/performance.json; {} for the seed library).
        `task_kwargs`: e.g. {"region": ...} to repair a failure-search cluster with region-sampled probe scenes."""
        self.task_kwargs = task_kwargs or {}
        self.perf = load_performance() if performance is None else performance
        self.patches = load_patches() if patches is None else patches
        self.hints = hints or []            # confirmed one-change tests from the investigator
        self.perturb = perturb              # physics conditions to repair under (e.g. where the failures happened)
        self.patch_trials = patch_trials
        self.min_fail_rate = min_fail_rate
        self.task = task
        self.n_before, self.n_candidates, self.per_family = n_before, n_candidates, per_family
        self.family_seeds, self.max_families = family_seeds, max_families
        self.tune_trials, self.tune_seeds, self.n_dev, self.n_regression = tune_trials, tune_seeds, n_dev, n_regression
        self.min_feasible, self.force = min_feasible, force
        self.rng = np.random.default_rng(seed)
        suffix = f"_{self.task_kwargs['region']['template']}" if "region" in self.task_kwargs else ""
        self.run_id = run_id or time.strftime(f"%Y%m%d-%H%M%S_explore_{task}{suffix}")
        self.out = ROOT / "runs" / self.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool = TrialPool(workers)
        self.cards = library or L.load_cards()
        # Search primitives (skill_library/primitives/): each candidate belongs to one; families compete across them.
        self.primitives = L.load_cards(L.LIB_DIR / "primitives")
        self.report = {"run_id": self.run_id, "task": task, "task_kwargs": self.task_kwargs, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "library_before": {n: c["version"] for n, c in self.cards.items()},
                       "phases": {}, "trials_simulated": 0}
        self.t0 = time.time()
        self.obj = None                   # the failing item, set by before()

    # ------------------------------------------------------------------ plumbing
    def _run(self, jobs):
        """jobs: list of (cfg, seed). Returns results in job order."""
        res = {}
        for r in self.pool.map(jobs):
            res[(r["tag"], r["seed"])] = r
        self.report["trials_simulated"] += len(jobs)
        return [res[(c.get("tag"), s)] for c, s in jobs]

    def _cfg(self, tag, task=None, task_kwargs=None, **kw):
        """Trials of the task under repair get its task_kwargs (region); other tasks only their own."""
        tk = {**self.task_kwargs, **(task_kwargs or {})} if task is None else (task_kwargs or {})
        kw.setdefault("performance", self.perf)
        kw.setdefault("patches", self.patches)
        if kw.get("perturb") is None:
            kw["perturb"] = self.perturb
        return trial_cfg(task or self.task, self.out, save_failures=False, success_sample=0, save_video=False,
                         save_poses=False, tag=tag, library=self.cards, task_kwargs=tk, **kw)

    def _evidence(self, tag, seeds, obj) -> dict:
        """Success of `obj`'s step per shape group, read from saved trial records: {group: [k, n]}."""
        ev = {}
        for s in seeds:
            r = json.loads((self.out / "trials" / f"{tag}_{s:06d}.json").read_text())
            d = (r.get("percepts") or {}).get(obj)
            if not d:
                continue
            e = ev.setdefault(group_of(d), [0, 0])
            e[0] += int(any(st.get("obj") == obj and st["ok"] for st in r["subtasks"]))
            e[1] += 1
        return ev

    def _with(self, perf, card, evidence):
        for g, (k, n) in evidence.items():
            perf = with_evidence(perf, card, g, k, n, self.run_id)
        return perf

    def _candidate_jobs(self, tag, params, seeds, perturb=None, primitive="feature_pinch"):
        """Candidate trials: the failing item's sub-task alone, with the candidate forced for that item."""
        cfg = self._cfg(tag, overrides={"grasp_for": {self.obj: primitive}, primitive: params},
                        extra_cards=dict(self.primitives), perturb=perturb, task_kwargs={"isolate_obj": self.obj})
        return [(cfg, s) for s in seeds]

    def _save(self):
        self.report["wall_time_s"] = round(time.time() - self.t0, 1)
        (self.out / "report.json").write_text(json.dumps(self.report, indent=1, default=float))

    # ------------------------------------------------------------------ phases
    def before(self):
        # Pre-declared stopping rule: keep collecting (batches of n_before, at most 96 trials) until at least
        # MIN_CLUSTER valid failures, so the paired gate has enough failed cases to show a fix.
        seeds, res = [], []
        while True:
            new = split_seeds("train", self.n_before, start=len(seeds))
            res += self._run([(self._cfg("before", oracle=True), s) for s in new])
            seeds += new
            n_fail = sum(not r["success"] and r["oracle"] and r["oracle"]["success"] for r in res)
            if n_fail >= MIN_CLUSTER or len(seeds) >= 96 or n_fail == 0 or self._rare(n_fail, len(seeds)):
                break
            _log(f"before: {n_fail} valid failures in {len(seeds)} trials; collecting more (need {MIN_CLUSTER})")
        recs = [json.loads((self.out / "trials" / f"before_{s:06d}.json").read_text()) for s in seeds]
        failing = [r for r in recs if not r["success"]]
        # Cluster = (item, grasp skill). Labels are kept per cluster: one cause can show as several labels
        # (a top grasp on a bowl closes on air, or its open fingers land on the wall). Only valid failures count:
        # the item's step failed where the oracle succeeded on the same scene.
        cluster, labels, cseeds = collections.Counter(), collections.defaultdict(collections.Counter), {}
        n_unsolvable = collections.Counter()
        for r in failing:
            for st in r["subtasks"]:
                if st["label"] and st["trace"] and not r["oracle"]["subtask_ok"].get(st["obj"]):
                    n_unsolvable[st["obj"]] += 1
                elif st["label"] and st["trace"]:
                    key = (st["obj"], st["trace"][0]["skill"])
                    cluster[key] += 1
                    labels[key][st["label"]] += 1
                    cseeds.setdefault(key, []).append(r["seed"])
        top = cluster.most_common(1)[0][0] if cluster else None
        ph = {"seeds": seeds, **_rate(res), "oracle_solved": sum(r["oracle"]["success"] for r in res),
              "excluded_unsolvable": dict(n_unsolvable),
              "clusters": [{"obj": o, "skill": s, "count": c, "labels": dict(labels[(o, s)])}
                           for (o, s), c in cluster.most_common()]}
        if top:
            obj, skill = top
            self.obj = obj
            fseeds = cseeds[top]
            group = shape_group(next(r for r in failing if r.get("percepts", {}).get(obj))["percepts"][obj])
            lab = labels[top]
            self._cluster_descs = [r["percepts"][obj] for r in failing
                                   if r["seed"] in fseeds and r.get("percepts", {}).get(obj)]
            # How the failing skill did on this item, per shape group, over all before trials that used it.
            ev = {}
            for r in recs:
                for st in r["subtasks"]:
                    if st.get("obj") == obj and st.get("trace") and st["trace"][0]["skill"] == skill \
                            and r.get("percepts", {}).get(obj):
                        e = ev.setdefault(group_of(r["percepts"][obj]), [0, 0])
                        e[0] += int(st["ok"])
                        e[1] += 1
            ph["cluster"] = {"obj": obj, "skill": skill, "skill_version": self.cards.get(skill, {}).get("version"),
                             "label": lab.most_common(1)[0][0], "labels": dict(lab), "seeds": fseeds, "shape": group,
                             "evidence": ev,
                             "statement": f"{skill} fails on {obj} in {len(fseeds)}/{len(seeds)} train trials, "
                                          f"all solvable by the oracle "
                                          f"({', '.join(f'{l} {c}' for l, c in lab.most_common())}); "
                                          f"shape: hollow={group['hollow']} rim={group['rim']}"}
        if top and self._rare(n_fail, len(seeds)):
            hi = wilson(n_fail, len(seeds))[1]
            ph["rare"] = {"valid_failures": n_fail, "trials": len(seeds), "rate_ci95_high": hi,
                          "min_fail_rate": self.min_fail_rate}
            _log(f"before: {n_fail} valid failures in {len(seeds)} trials: failure rate below "
                 f"{self.min_fail_rate:.0%} (95% upper bound {hi:.1%}); too rare to repair at this budget")
        self.report["phases"]["before"] = ph
        _log(f"before: {ph['success']}/{ph['trials']} succeed; cluster: {ph.get('cluster', {}).get('statement')}")
        self._save()
        return ph.get("cluster")

    def _rare(self, n_fail, n):
        return self.min_fail_rate is not None and n_fail > 0 and wilson(n_fail, n)[1] < self.min_fail_rate

    def existing(self, cluster):
        """Other cards in the library whose preconditions fit the failing objects, tried on train objects."""
        obj, descs = cluster["obj"], self._cluster_descs
        cands = [n for n, c in self.cards.items() if c.get("kind") == "grasp" and n != cluster["skill"]
                 and descs and sum(fits(c, d) for d in descs) >= 0.75 * len(descs)]
        seeds = split_seeds("train", self.family_seeds + self.tune_seeds)
        rows = []
        for n in cands:
            tag = f"existing_{n}"
            rr = self._run([(self._cfg(tag, overrides={"grasp_for": {obj: n}}, task_kwargs={"isolate_obj": obj}), s)
                            for s in seeds])
            rows.append({"card": n, **_obj_rate(rr, obj), "evidence": self._evidence(tag, seeds, obj),
                         "labels": dict(collections.Counter(l for x in rr for l in x["labels"]))})
            _log(f"existing {n}: {rows[-1]['success']}/{rows[-1]['trials']} on {obj}")
        rows.sort(key=lambda r: -r["rate"])
        self.report["phases"]["existing"] = {"seeds": seeds, "candidates": rows}
        self._save()
        return rows

    def sample(self, cluster):
        from ..perception.perceive import Perception
        from ..sim.scene import build_scene
        obj = cluster["obj"]
        scenes = []
        for s in cluster["seeds"][:6]:
            sc = build_scene(make_task(self.task, seed=s, isolate_obj=obj, **self.task_kwargs), seed=s)
            per = Perception(sc)
            d = per.perceive([obj])[obj]
            scenes.append((s, d, per.last_points[obj]))
            per.close()
        # Per primitive: half uniform over its parameters, half anchored on perceived surface points.
        _, d0, pts0 = scenes[0]
        cands, prims, sources = [], [], []
        n_p = self.n_candidates // len(self.primitives)
        for pname, prim in self.primitives.items():
            n_u = n_p // 2
            new = sample_candidates(prim, n_u, self.rng) + surface_candidates(prim, pts0, d0, n_p - n_u, self.rng)
            cands += new
            prims += [pname] * len(new)
            sources += ["uniform"] * n_u + ["surface"] * (n_p - n_u)
        reach = ReachChecker()
        rows = []
        for i, p in enumerate(cands):
            pn = prims[i]
            fits = [geometric_fit(pts, d, p, pn) for _, d, pts in scenes]
            frac = sum(f["ok"] for f in fits) / len(fits)
            fam = family_of(scenes[0][1], p, pn)
            row = {"id": i, "primitive": pn, "source": sources[i],
                   "params": {k: round(p[k], 4) for k in search_params(self.primitives[pn])}, "family": fam["name"],
                   "anchor": fam["anchor"], "closing": fam["closing"], "geom_frac": round(frac, 2),
                   "geom_reason": collections.Counter(f["reason"] for f in fits).most_common(1)[0][0],
                   "score": round(float(np.mean([f.get("score", 0.0) for f in fits])), 4)}
            if frac >= self.min_feasible:
                rr = [reach.reachable(d, p, pn) for _, d, _ in scenes]
                row["reach_frac"] = round(sum(r["ok"] for r in rr) / len(rr), 2)
            rows.append(row)
        survivors = [r for r in rows if r.get("reach_frac", 0) >= self.min_feasible]
        reasons = collections.Counter(r["geom_reason"] for r in rows if r["geom_frac"] < self.min_feasible)
        ph = {"n_sampled": len(rows), "n_geometric": sum(r["geom_frac"] >= self.min_feasible for r in rows),
              "n_survivors": len(survivors), "reject_reasons": dict(reasons.most_common()),
              "scene_seeds": [s for s, _, _ in scenes],
              "survivor_families": dict(collections.Counter(r["family"] for r in survivors).most_common()),
              "candidates": rows}
        self.report["phases"]["sample"] = ph
        _log(f"sample: {len(rows)} candidates -> {ph['n_geometric']} fit the gripper -> {len(survivors)} reachable; "
             f"families {ph['survivor_families']}")
        self._save()
        self._params = {i: {**{k: v["default"] for k, v in self.primitives[prims[i]]["params"].items()}, **cands[i]}
                        for i in range(len(cands))}
        self._prim = dict(enumerate(prims))
        return survivors

    def families(self, survivors):
        by_fam = collections.defaultdict(list)
        for r in sorted(survivors, key=lambda r: -r["score"]):
            by_fam[r["family"]].append(r)
        seeds = split_seeds("train", self.family_seeds)
        picked = []
        for fam, rows in by_fam.items():
            chosen = []                   # best-scoring, spread over the object
            for r in rows:
                if all(_apart(r, c) for c in chosen):
                    chosen.append(r)
                if len(chosen) == self.per_family:
                    break
            picked += chosen
        jobs = []
        for r in picked:
            jobs += self._candidate_jobs(f"cand{r['id']}", self._params[r["id"]], seeds, primitive=r["primitive"])
        res = self._run(jobs)
        by_tag = collections.defaultdict(list)
        for (cfg, _), x in zip(jobs, res):
            by_tag[cfg["tag"]].append(x)
        trials = []
        for r in picked:
            rr = by_tag[f"cand{r['id']}"]
            r["sim"] = {**_rate(rr), "labels": dict(collections.Counter(l for x in rr for l in x["labels"]))}
            trials.append(r)
        fam_summary = []
        for fam in by_fam:
            rows = [r for r in trials if r["family"] == fam]
            k, n = sum(r["sim"]["success"] for r in rows), sum(r["sim"]["trials"] for r in rows)
            fam_summary.append({"family": fam, "primitive": rows[0]["primitive"], "anchor": rows[0]["anchor"],
                                "closing": rows[0]["closing"],
                                "candidates": len(rows), "success": k, "trials": n, "rate": round(k / max(n, 1), 3),
                                "ci95": wilson(k, n), "best": max(rows, key=lambda r: r["sim"]["rate"])["id"]})
        fam_summary.sort(key=lambda f: (-f["rate"], -f["candidates"]))
        self.report["phases"]["families"] = {"seeds": seeds, "families": fam_summary,
                                             "success_map": [{k: r[k] for k in ("id", "family", "params", "sim")}
                                                             for r in trials]}
        for f in fam_summary:
            _log(f"family {f['family']:<18} {f['success']}/{f['trials']} over {f['candidates']} candidates")
        self._save()
        return fam_summary, trials

    def tune(self, family, trials):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        rows = sorted([r for r in trials if r["family"] == family["family"]], key=lambda r: -r["sim"]["rate"])
        pname = family["primitive"]
        prm = self.primitives[pname]["params"]
        best0 = self._params[rows[0]["id"]]
        # Search box: the family's region (anchor and closing fixed by its candidates), other params free.
        sp = search_params(self.primitives[pname])
        lo_hi = {k: [min(r["params"][k] for r in rows), max(r["params"][k] for r in rows)] for k in sp}
        if pname != "feature_pinch":      # generic: the family's bounding box, widened by 15% of each range
            space = {}
            for k in sp:
                lo, hi = prm[k]["range"]
                pad = 0.15 * (hi - lo)
                space[k] = (max(lo, lo_hi[k][0] - pad), min(hi, lo_hi[k][1] + pad))
            space["grip_force"] = (15.0, 40.0)
        else:
            space = {
                "edge_offset": (max(prm["edge_offset"]["range"][0], lo_hi["edge_offset"][0] - 0.01),
                                min(prm["edge_offset"]["range"][1], lo_hi["edge_offset"][1] + 0.01)),
                "angle_deg": tuple(prm["angle_deg"]["range"]),
                "depth": (max(prm["depth"]["range"][0], lo_hi["depth"][0] - 0.01),
                          min(prm["depth"]["range"][1], lo_hi["depth"][1] + 0.01)),
                "close_yaw_deg": (max(-90, lo_hi["close_yaw_deg"][0] - 10), min(90, lo_hi["close_yaw_deg"][1] + 10)),
                "tilt_deg": tuple(prm["tilt_deg"]["range"]),
                "grip_force": (15.0, 40.0),
            }
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        for r in rows[:3]:
            p = self._params[r["id"]]
            study.enqueue_trial({k: float(np.clip(p[k], *space[k])) for k in space})
        seeds = split_seeds("train", self.tune_seeds, start=self.family_seeds)
        batch = max(1, self.pool.workers // self.tune_seeds + 1)
        history, done, stopped = [], 0, None
        while done < self.tune_trials:
            ts = [study.ask() for _ in range(min(batch, self.tune_trials - done))]
            ps = []
            for t in ts:
                p = dict(best0)
                p.update({k: t.suggest_float(k, *space[k]) for k in space})
                ps.append(p)
            jobs = []
            for t, p in zip(ts, ps):
                jobs += self._candidate_jobs(f"tune{t.number}", p, seeds, perturb=TUNE_PERTURB, primitive=pname)
            res = self._run(jobs)
            for t, p in zip(ts, ps):
                rr = [x for (c, _), x in zip(jobs, res) if c["tag"] == f"tune{t.number}"]
                val = sum(x["success"] for x in rr) / len(rr)
                study.tell(t, val)
                history.append({"trial": t.number, "params": {k: round(p[k], 4) for k in space}, "value": val,
                                "labels": dict(collections.Counter(l for x in rr for l in x["labels"]))})
            done += len(ts)
            _log(f"tune {family['family']}: {done}/{self.tune_trials} trials, best {study.best_value:.2f}")
            if done >= 8 and sum(h["value"] == 1.0 for h in history) >= 4:
                stopped = "saturated: 4 trials succeeded on every tuning object"
                _log(f"tune {family['family']}: {stopped}")
                break
        # Confirm the leaders on fresh (perturbed) train objects; ties in tuning are common with few seeds.
        leaders = sorted(history, key=lambda h: (-h["value"], h["trial"]))[:4]
        cseeds = split_seeds("train", 8, start=self.family_seeds + self.tune_seeds)
        jobs = []
        for h in leaders:
            jobs += self._candidate_jobs(f"confirm{h['trial']}", {**best0, **h["params"]}, cseeds, perturb=TUNE_PERTURB,
                                         primitive=pname)
        res = self._run(jobs)
        for h in leaders:
            rr = [x for (c, _), x in zip(jobs, res) if c["tag"] == f"confirm{h['trial']}"]
            h["confirm"] = round(sum(x["success"] for x in rr) / len(rr), 3)
        best = max(leaders, key=lambda h: (h["confirm"], h["value"], -h["trial"]))
        params = {**best0, **best["params"]}
        self.report["phases"].setdefault("tune", []).append(
            {"family": family["family"], "seeds": seeds, "confirm_seeds": cseeds, "perturb": TUNE_PERTURB,
             "space": space, "history": history, "stopped": stopped, "leaders": leaders, "best": best,
             "seeded_from": [r["id"] for r in rows[:3]]})
        _log(f"tune {family['family']}: chose trial {best['trial']} (tuning {best['value']:.2f}, "
             f"confirmation {best['confirm']:.2f})")
        self._save()
        return params, best["confirm"]

    def make_card(self, family, params, cluster):
        name = family["family"]
        old = L.load_cards().get(name)            # versions continue from the live library, not the snapshot
        prim = self.primitives[family["primitive"]]
        prm = copy.deepcopy(prim["params"])
        for k, v in prm.items():
            v["default"] = round(float(params[k]), 4)
        shape = cluster["shape"]
        requires = {"hollow": shape["hollow"]}
        if shape["rim"]:
            requires["rim"] = True
        # Preconditions cover the shapes the card was found and tested on (perceived size of the failing objects,
        # with a margin), not every object with the same hollow/rim flags: a card must not take over objects
        # it was never checked on. The router can widen use later only through evidence.
        ws = [d["width"] for d in self._cluster_descs]
        hs = [d["height"] for d in self._cluster_descs]
        requires.update(min_width=round(max(0.0, min(ws) - 0.01), 3), max_width=round(max(ws) + 0.01, 3),
                        min_height=round(max(0.0, min(hs) - 0.01), 3), max_height=round(max(hs) + 0.01, 3))
        requires.update(elongation_bounds([d.get("elongation", 1.0) for d in self._cluster_descs]))
        return {
            "name": name, "version": (old["version"] + 1) if old else 1, "kind": "grasp", "impl": prim["name"],
            "description": (f"{family['anchor']} grasp, fingers closing "
                            f"{'across the object' if family['closing'] == 'any' else family['closing'] + 'ly'}; "
                            f"found by exploration after {cluster['skill']} failed on {cluster['obj']} "
                            f"({cluster['label']})."),
            "requires": requires, "params": prm, "check": prim["check"],
            "lineage": (old["lineage"] if old else []),
        }

    def _regression_items(self):
        """The fixed regression tasks plus every earlier repair's trigger task (with its region), so a new change
        cannot undo an earlier one."""
        items = [{"task": t, "kwargs": {}} for t in REGRESSION_TASKS]
        for c in self.cards.values():
            for e in c.get("lineage", []):
                tr = e.get("trigger") or {}
                if e.get("source") == "exploration" and tr.get("task"):
                    it = {"task": tr["task"], "kwargs": tr.get("task_kwargs") or {}}
                    if it not in items:
                        items.append(it)
        for card, groups in [*self.perf.items(), *self.patches.items()]:
            for g in groups.values():
                for src in g.get("triggers", []):
                    if src not in items:
                        items.append(src)
        mine = {"task": self.task, "kwargs": self.task_kwargs}
        return [it for it in items if it != mine]

    def gate(self, name, cluster, extra_cards=None, performance=None, patches=None, n_dev=None):
        """All gates run on the full task (every item), so a fix that breaks another step cannot pass.
        A change is new cards (extra_cards) and/or new routing evidence (performance)."""
        extra = extra_cards or {}
        perf = self.perf if performance is None else performance
        pat = self.patches if patches is None else patches
        tag = f"gate_{name.replace(':', '_')}"
        obj = cluster["obj"]
        # (a) the failure cluster: the same train seeds that failed before (paired), the repaired item's step
        fs = cluster["seeds"]
        a_res = self._run([(self._cfg(tag + "_cluster", extra_cards=extra, performance=perf, patches=pat), s)
                           for s in fs])
        a = {**_obj_rate(a_res, obj), "task": _rate(a_res),
             "before": {"success": 0, "trials": len(fs), "ci95": wilson(0, len(fs))}}
        # Paired outcomes on the same seeds: McNemar's exact test, one-sided (after better), alpha 0.05.
        a["p_value"] = mcnemar_p([False] * len(fs), [bool(r["subtask_ok"].get(obj)) for r in a_res])
        a["pass"] = a["p_value"] < ALPHA
        # (c) held-out objects (dev split), before and after
        ds = split_seeds("dev", n_dev or self.n_dev)
        c_res = self._run([(self._cfg(tag + "_dev", extra_cards=extra, performance=perf, patches=pat), s)
                           for s in ds])
        b_res = self._run([(self._cfg("dev_before", oracle=True), s) for s in ds])
        c_after, c_before = _obj_rate(c_res, obj), _obj_rate(b_res, obj)
        p_c = mcnemar_p([bool(r["subtask_ok"].get(obj)) for r in b_res], [bool(r["subtask_ok"].get(obj)) for r in c_res])
        # If held-out objects already mostly succeed (the failure is rare there), the change must HOLD: no
        # significant worsening (paired test reversed). Otherwise it must significantly improve them.
        p_worse = mcnemar_p([bool(r["subtask_ok"].get(obj)) for r in c_res], [bool(r["subtask_ok"].get(obj)) for r in b_res])
        mode = "hold" if c_before["rate"] >= 0.8 else "improve"
        c = {**c_after, "task": _rate(c_res), "before": {**c_before, "task": _rate(b_res)},
             "oracle_solved": sum(bool(r["oracle"]["subtask_ok"].get(obj)) for r in b_res), "p_value": p_c,
             "p_worse": p_worse, "mode": mode,
             "pass": (p_c < ALPHA if mode == "improve" else p_worse >= ALPHA) and c_after["rate"] >= 0.7}
        # (b) no regression on the other tasks (same seeds, library with and without the new card)
        b = {"tasks": {}, "pass": True}
        for it in self._regression_items():
            t, kw = it["task"], it["kwargs"]
            label = t + (f"[{kw['region']['template']}]" if "region" in kw else "")
            seeds = list(range(self.n_regression))
            base = self._baseline(t, seeds, kw)
            # Regression always at nominal physics (perturb={}), whatever conditions the repair targets.
            after = _rate(self._run([(self._cfg(f"{tag}_reg_{label}", task=t, task_kwargs=kw, extra_cards=extra,
                                                performance=perf, patches=pat, perturb={}), s) for s in seeds]))
            ok = after["goal_rate"] >= base["goal_rate"] - 0.1 and after["goal_ci95"][1] >= base["goal_ci95"][0]
            b["tasks"][label] = {"before": base, "after": after, "pass": ok}
            b["pass"] &= ok
        a["seeds"] = fs
        g = {"card": name, "obj": obj, "fixes_cluster": a, "no_regression": b, "held_out": c,
             "pass": a["pass"] and b["pass"] and c["pass"]}
        self.report["phases"].setdefault("gates", []).append(g)
        _log(f"gate {name}: {obj} {a['success']}/{a['trials']} on the failed trials, whole task "
             f"{a['task']['success']}/{a['trials']} ({'pass' if a['pass'] else 'FAIL'}); held-out {obj} "
             f"{c['success']}/{c['trials']} vs {c_before['success']}/{c_before['trials']} before (p={p_c:.3g}), whole task "
             f"{c['task']['success']}/{c['trials']} ({'pass' if c['pass'] else 'FAIL'}); "
             f"regression {'pass' if b['pass'] else 'FAIL'}")
        self._save()
        return g

    def _baseline(self, task, seeds, kwargs=None):
        """Regression baseline with the current library and routing, cached by library, routing evidence, task
        arguments, code version and seeds."""
        from ..version import code_version       # a code change can change behaviour: re-run the baseline
        key = (f"{task}_{_library_hash(self.cards, self.perf, self.patches, kwargs or {})}_{code_version()['code_hash']}"
               f"_{seeds[0]}-{seeds[-1]}")
        path = BASELINE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        r = _rate(self._run([(self._cfg(f"baseline_{task}", task=task, task_kwargs=kwargs, perturb={}), s)
                             for s in seeds]))
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r, indent=1))
        return r

    def _final_perf(self, perf, card_name, gate):
        """Routing evidence to keep: the change's evidence plus the gate's paired cluster trials."""
        tag = f"gate_{card_name}_cluster"
        perf = self._with(perf, card_name, self._evidence(tag, gate["fixes_cluster"]["seeds"], gate["obj"]))
        trigger = {"task": self.task, "kwargs": self.task_kwargs}
        for groups in perf.get(card_name, {}).values():
            if trigger not in groups.setdefault("triggers", []):
                groups["triggers"].append(trigger)
        return perf

    def promote_routing(self, card_name, cluster, gate, perf):
        perf = self._with(perf, card_name, self._evidence(f"gate_route_{card_name}_cluster",
                                                          gate["fixes_cluster"]["seeds"], cluster["obj"]))
        trigger = {"task": self.task, "kwargs": self.task_kwargs}
        for groups in perf.get(card_name, {}).values():
            if trigger not in groups.setdefault("triggers", []):
                groups["triggers"].append(trigger)
        save_performance(perf)
        self.report["routed"] = {"card": card_name, "instead_of": cluster["skill"],
                                 "groups": sorted(perf.get(card_name, {})), "performance_file": "skill_library/performance.json"}
        _log(f"ROUTED {cluster['obj']}: {card_name} instead of {cluster['skill']} "
             f"(groups {self.report['routed']['groups']})")
        self._save()

    def promote(self, card, family, params, cluster, gate, tune_val, perf=None):
        parent = L.load_cards().get(card["name"])
        prim = self.primitives[card["impl"]]
        diff_base = {k: v["default"] for k, v in (parent or prim)["params"].items()}
        entry = {
            "version": card["version"], "source": "exploration", "run_id": self.run_id,
            "date": time.strftime("%Y-%m-%d"),
            "parent": ({"skill": card["name"], "version": parent["version"]} if parent else
                       {"primitive": prim["name"], "version": prim["version"]}),
            "trigger": {k: cluster[k] for k in ("skill", "skill_version", "label", "labels", "obj")}
                       | {"task": self.task, "task_kwargs": self.task_kwargs},
            "family": family["family"], "trials_simulated": self.report["trials_simulated"],
            "tune_best_train_rate": tune_val,
            "results": {"cluster": [gate["fixes_cluster"]["before"]["success"], gate["fixes_cluster"]["success"],
                                    gate["fixes_cluster"]["trials"]],
                        "held_out": {"before": gate["held_out"]["before"]["rate"], "after": gate["held_out"]["rate"],
                                     "ci95": gate["held_out"]["ci95"], "trials": gate["held_out"]["trials"]}},
            "diff": {k: [diff_base[k], card["params"][k]["default"]] for k in card["params"]
                     if abs(float(diff_base[k]) - float(card["params"][k]["default"])) > 1e-9},
            "requires_diff": ([parent.get("requires"), card["requires"]]
                              if parent and parent.get("requires") != card["requires"] else None),
        }
        card["lineage"] = card["lineage"] + [entry]
        path = L.save_card(card)
        if perf is not None:
            save_performance(self._final_perf(perf, card["name"], gate))
        self.report["promoted"] = {"card": card, "path": str(path.relative_to(ROOT))}
        _log(f"PROMOTED {card['name']} v{card['version']} -> {path.relative_to(ROOT)}")
        self._save()

    # ------------------------------------------------------------------ parameter patches
    def _cluster_groups(self):
        gs = collections.Counter(group_of(d) for d in self._cluster_descs)
        n = sum(gs.values())
        return sorted(g for g, c in gs.items() if c >= max(2, 0.2 * n)) or [gs.most_common(1)[0][0]]

    def _patch_dims(self, cluster):
        """(skill, parameter) pairs a patch may change: release and transport, lift, grip, grasp depth."""
        g = self.cards[cluster["skill"]]
        dims = [("place_in", "clearance"), ("place_in", "speed"), ("lift", "speed"), (cluster["skill"], "grip_force")]
        dims += [(cluster["skill"], k) for k, v in g["params"].items() if v["range"] == [0, 1]]   # strategy switches
        depth = next((p for p in ("depth", "depth_frac") if p in g["params"]), None)
        return dims + ([(cluster["skill"], depth)] if depth else [])

    def _range(self, skill, param):
        return self.cards[skill]["params"][param]["range"]

    def _as_patches(self, values, groups):
        out = copy.deepcopy(self.patches)
        for (sk, pm), v in values.items():
            lo, hi = self._range(sk, pm)
            for g in groups:
                e = out.setdefault(sk, {}).setdefault(g, {})
                e.setdefault("params", {})[pm] = round(float(min(max(v, lo), hi)), 4)
        return out

    def _patch_rate(self, tag, patches, seeds, perturb=None):
        cfg = self._cfg(tag, patches=patches, perturb=perturb, task_kwargs={"isolate_obj": self.obj})
        return _obj_rate(self._run([(cfg, s) for s in seeds]), self.obj)

    def patch_search(self, cluster):
        """Candidates: each confirmed hint alone, and the best Optuna patch; confirmed on fresh perturbed objects."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        groups, dims = self._cluster_groups(), self._patch_dims(cluster)
        cands = []
        for h in self.hints:
            key = (h.get("skill"), h.get("param"))
            if h.get("type") == "skill_param" and key[0] in self.cards and key[1] in self.cards[key[0]]["params"]:
                c = {"name": f"hint {key[0]}.{key[1]}={h['value']}", "values": {key: float(h["value"])},
                     "source": "investigator"}
                if c["values"] not in [x["values"] for x in cands]:
                    cands.append(c)
        seeds = split_seeds("train", self.tune_seeds, start=self.family_seeds)
        cseeds = split_seeds("train", 8, start=self.family_seeds + self.tune_seeds)
        base = {d: self.cards[d[0]]["params"][d[1]]["default"] for d in dims}
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.enqueue_trial({f"{k[0]}.{k[1]}": v for k, v in base.items()})
        for c in cands:
            study.enqueue_trial({f"{k[0]}.{k[1]}": v for k, v in {**base, **c["values"]}.items() if k in base})
        history, batch = [], max(1, self.pool.workers // self.tune_seeds + 1)
        while len(history) < self.patch_trials:
            ts = [study.ask() for _ in range(min(batch, self.patch_trials - len(history)))]
            for t in ts:
                vals = {d: t.suggest_float(f"{d[0]}.{d[1]}", *self._range(*d)) for d in dims}
                r = self._patch_rate(f"patch{t.number}", self._as_patches(vals, groups), seeds, TUNE_PERTURB)
                study.tell(t, r["rate"])
                history.append({"trial": t.number, "values": {f"{k[0]}.{k[1]}": round(v, 4) for k, v in vals.items()},
                                "rate": r["rate"]})
            _log(f"patch search: {len(history)}/{self.patch_trials} trials, best {study.best_value:.2f}")
        best = max(history, key=lambda h: (h["rate"], -h["trial"]))
        cands.append({"name": "optuna", "values": {tuple(k.split(".")): v for k, v in best["values"].items()},
                      "source": "optuna", "tune_rate": best["rate"]})
        for i, c in enumerate(cands):
            c["confirm"] = self._patch_rate(f"patchconfirm{i}", self._as_patches(c["values"], groups), cseeds,
                                            TUNE_PERTURB)
            _log(f"patch candidate {c['name']}: {c['confirm']['success']}/{c['confirm']['trials']} on fresh objects")
        chosen = max(cands, key=lambda c: (c["confirm"]["rate"], -len(c["values"])))
        self.report["phases"]["patch"] = {
            "groups": groups, "dims": [f"{a}.{b}" for a, b in dims], "seeds": seeds, "confirm_seeds": cseeds,
            "history": history, "chosen": chosen["name"],
            "candidates": [{**c, "values": {f"{k[0]}.{k[1]}": v for k, v in c["values"].items()}} for c in cands]}
        self._save()
        return chosen, groups

    def promote_patch(self, chosen, groups, cluster, gate, patches):
        ev = self._evidence("gate_patch_cluster", gate["fixes_cluster"]["seeds"], cluster["obj"])
        trigger = {"task": self.task, "kwargs": self.task_kwargs}
        for (sk, pm), v in chosen["values"].items():
            default = self.cards[sk]["params"][pm]["default"]
            for g in groups:
                e = patches[sk][g]
                old = (self.patches.get(sk, {}).get(g) or {}).get("params", {}).get(pm, default)
                e.setdefault("diff", {})[pm] = [old, e["params"][pm]]
                k, n = ev.get(g, [0, 0])
                e["evidence"] = {"success": k, "trials": n}
                e.setdefault("sources", []).append(self.run_id)
                if trigger not in e.setdefault("triggers", []):
                    e["triggers"].append(trigger)
                e["source"] = chosen["source"]
                e["date"] = time.strftime("%Y-%m-%d")
        save_patches(patches)
        self.report["patched"] = {"values": {f"{k[0]}.{k[1]}": v for k, v in chosen["values"].items()},
                                  "groups": groups, "source": chosen["source"], "file": "skill_library/patches.json"}
        _log(f"PATCHED {self.report['patched']['values']} for groups {groups} (source: {chosen['source']})")
        self._save()

    # ------------------------------------------------------------------ main
    def run(self):
        try:
            cluster = self.before()
            if not cluster:
                _log("no failure cluster: nothing to repair")
                self.report["outcome"] = "no_failures"
                return self.report
            if self.report["phases"]["before"].get("rare"):
                self.report["outcome"] = "rare_failure"
                return self.report
            for row in [r for r in self.existing(cluster) if r["rate"] >= 0.75][:2]:
                perf = self._with(self._with(self.perf, cluster["skill"], cluster["evidence"]), row["card"],
                                  row["evidence"])
                g = self.gate(f"route:{row['card']}", cluster, performance=perf)
                if g["pass"]:
                    self.promote_routing(row["card"], cluster, g, perf)
                    self.report["outcome"] = "routed"
                    return self.report
            post_grasp = sum(c for l, c in cluster["labels"].items() if l in POST_GRASP) > 0.5 * sum(
                cluster["labels"].values())
            if post_grasp or self.hints:
                chosen, groups = self.patch_search(cluster)
                if chosen["confirm"]["rate"] >= 0.75:
                    pat = self._as_patches(chosen["values"], groups)
                    g = self.gate("patch", cluster, patches=pat, n_dev=N_DEV_PATCH)
                    if g["pass"]:
                        self.promote_patch(chosen, groups, cluster, g, pat)
                        self.report["outcome"] = "patched"
                        return self.report
                if post_grasp:
                    self.report["outcome"] = "no_patch_passed"
                    _log("failure is after the grasp and no patch passed; grasp exploration would not address it")
                    return self.report
            survivors = self.sample(cluster)
            if not survivors:
                self.report["outcome"] = "no_feasible_candidates"
                return self.report
            fams, trials = self.families(survivors)
            working = [f for f in fams if f["success"] > 0][:self.max_families]
            # Families the trials cannot tell apart from the leader (overlapping intervals) all go through
            # tuning and the gates; the one that holds best on held-out objects is promoted. If none of
            # them passes, the remaining families are tried in order.
            if working:
                lead_lo = working[0]["ci95"][0]
                contenders = [f for f in working if f["ci95"][1] >= lead_lo]
                rest = [f for f in working if f not in contenders]
                passed = []
                for group in (contenders, *([f] for f in rest)):
                    for fam in group:
                        params, val = self.tune(fam, trials)
                        card = self.make_card(fam, params, cluster)
                        tinfo = self.report["phases"]["tune"][-1]
                        ev = self._evidence(f"confirm{tinfo['best']['trial']}", tinfo["confirm_seeds"], cluster["obj"])
                        perf = self._with(self._with(self.perf, cluster["skill"], cluster["evidence"]),
                                          card["name"], ev)
                        g = self.gate(card["name"], cluster, extra_cards={card["name"]: card}, performance=perf)
                        if g["pass"]:
                            passed.append((g["held_out"]["rate"], val, fam["rate"], card, fam, params, g, perf))
                    if passed:
                        break
                if passed:
                    _, val, _, card, fam, params, g, perf = max(passed, key=lambda x: x[:3])
                    if len(passed) > 1:
                        _log(f"{len(passed)} variants passed the gates; promoting {card['name']} "
                             f"(best on held-out objects)")
                    self.promote(card, fam, params, cluster, g, val, perf)
                    self.report["outcome"] = "promoted"
                    return self.report
            self.report["outcome"] = "no_candidate_passed_gates"
            _log("no candidate passed the gates; nothing promoted")
            return self.report
        finally:
            self._save()
            self.pool.close()
