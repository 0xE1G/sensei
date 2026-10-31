"""Failure search: find where the current skill library fails, before anyone says where to look.

Each trial is a probe scene: one object (template + parameters) at a chosen table position, to a tray (or bin).
Three strategies spend the same simulated-trial budget:

    random   uniform over templates, tested parameter ranges and reachable positions
    optuna   TPE maximizing valid failures (template is a categorical, parameters conditional on it)
    model    the model reads the results so far and proposes regions for the next batch (job "search_proposal";
             rule stand-in, Claude or Spark through the model client)

A scenario is simulated only if it is plausible: parameters in range, position within reach, and the placer can lay
the scene out. A failure counts only if the oracle solves the same scene ("valid failure"). Outputs, in
runs/<id>/report.json: per-strategy metrics (valid failures per 100 trials, distinct failure clusters, trials to find
each), a capability map (success by template, size band and reach band) and failure clusters with their parameter
region, ready for the explorer.
"""
from __future__ import annotations

import collections
import json
import math
import time

import numpy as np

from ..runner import ROOT, TrialPool, trial_cfg, wilson
from ..sim.scene import REACH, ObjectSpec, _sample_positions
from ..sim.templates import RANGES
from ..tasks.specs import PROBE_TEMPLATES, probe

XY = {"x": list(REACH["x"]), "y": list(REACH["y"])}
YAW = [-math.pi / 2, math.pi / 2]       # item orientation on the table (matters for blocks and bars)
OBSTACLE = None                          # set by FailureSearch(obstacle=True): a bottle near every item
OB_RANGES = {"dist": [0.06, 0.14], "radius": [0.02, 0.03], "height": [0.10, 0.16]}
SPACE = {t: {k: list(v) for k, v in RANGES[t].items()} for t in PROBE_TEMPLATES}
SIZE_KEY = {t: next(k for k in RANGES[t] if k not in ("mass", "friction")) for t in PROBE_TEMPLATES}
REACH_BANDS = ((0.0, 0.48, "near"), (0.48, 0.60, "mid"), (0.60, 9.0, "far"))


def _log(msg):
    print(f"[failsearch] {msg}", flush=True)


def plausible(scn: dict, seed: int) -> tuple[bool, str]:
    t = scn["template"]
    for k, v in scn["params"].items():
        lo, hi = RANGES[t][k]
        if not lo - 1e-9 <= v <= hi + 1e-9:
            return False, f"{k} out of range"
    x, y = scn["pos"]
    if not (REACH["x"][0] <= x <= REACH["x"][1] and REACH["y"][0] <= y <= REACH["y"][1]
            and REACH["r"][0] <= math.hypot(x, y) <= REACH["r"][1]):
        return False, "out of reach"
    task = probe(seed, scn)
    try:
        _sample_positions([ObjectSpec(**o) for o in task["objects"]], np.random.default_rng(seed),
                          fixed=task["layout_fixed"])
    except RuntimeError:
        return False, "no room for the target"
    return True, ""


def _uniform(t, rng, region=None):
    region = region or {}
    params = {}
    clip = lambda ab, lo, hi: (max(lo, min(ab[0], hi)), max(lo, min(ab[1], hi)))   # proposals may round past bounds
    for k, (lo, hi) in SPACE[t].items():
        a, b = clip((region.get("params") or {}).get(k, (lo, hi)), lo, hi)
        params[k] = float(rng.uniform(a, b))
    x = float(rng.uniform(*clip(region.get("x", XY["x"]), *XY["x"])))
    y = float(rng.uniform(*clip(region.get("y", XY["y"]), *XY["y"])))
    scn = {"template": t, "params": params, "pos": [x, y], "yaw": float(rng.uniform(*clip(region.get("yaw", YAW), *YAW)))}
    if OBSTACLE:
        scn["obstacle"] = {k: float(rng.uniform(*v)) for k, v in OB_RANGES.items()}
        scn["obstacle"]["angle"] = float(rng.uniform(-math.pi, math.pi))
    return scn


class RandomStrategy:
    name = "random"

    def __init__(self, rng, **_):
        self.rng = rng

    def propose(self, n, history):
        ts = list(SPACE)
        return [_uniform(ts[self.rng.integers(len(ts))], self.rng) for _ in range(n)]

    def tell(self, scn, value):
        pass


class OptunaStrategy:
    name = "optuna"

    def __init__(self, rng, seed=0, **_):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        self.pending = {}

    def propose(self, n, history):
        out = []
        for _ in range(n):
            tr = self.study.ask()
            t = tr.suggest_categorical("template", list(SPACE))
            params = {k: tr.suggest_float(f"{t}.{k}", lo, hi) for k, (lo, hi) in SPACE[t].items()}
            scn = {"template": t, "params": params,
                   "pos": [tr.suggest_float("x", *XY["x"]), tr.suggest_float("y", *XY["y"])],
                   "yaw": tr.suggest_float("yaw", *YAW)}
            if OBSTACLE:
                scn["obstacle"] = {k: tr.suggest_float(f"obstacle.{k}", *v) for k, v in OB_RANGES.items()}
                scn["obstacle"]["angle"] = tr.suggest_float("obstacle.angle", -math.pi, math.pi)
            self.pending[id(scn)] = tr
            out.append(scn)
        return out

    def tell(self, scn, value):
        self.study.tell(self.pending.pop(id(scn)), 0.0 if value is None else float(value))


class ModelStrategy:
    name = "model"

    def __init__(self, rng, client=None, budget=0, **_):
        from ..agents.llm import ModelClient
        self.rng, self.client, self.budget = rng, client or ModelClient(), budget
        self.calls = []

    def propose(self, n, history):
        stats = {t: {"n": 0, "success": 0, "valid_failures": 0, "unsolvable": 0} for t in SPACE}
        for h in history:
            s = stats[h["scenario"]["template"]]
            s["n"] += 1
            s[{"success": "success", "valid_failure": "valid_failures", "unsolvable": "unsolvable"}[h["outcome"]]] += 1
        rnd = lambda d: {k: round(v, 4) for k, v in d.items()}
        fails = [{"template": h["scenario"]["template"], "params": rnd(h["scenario"]["params"]),
                  "pos": [round(v, 3) for v in h["scenario"]["pos"]], "yaw": round(h["scenario"].get("yaw", 0), 2),
                  "label": h["label"], "skill": h["grasp"]}
                 for h in history if h["outcome"] == "valid_failure"][-25:]
        succ = [{"template": h["scenario"]["template"], "params": rnd(h["scenario"]["params"]),
                 "pos": [round(v, 3) for v in h["scenario"]["pos"]]}
                for h in history if h["outcome"] == "success"][-10:]
        inputs = {"batch": n, "budget_left": max(0, self.budget - len(history) - n), "space": SPACE, "xy": XY,
                  "stats": stats, "failures": fails, "successes": succ}
        from ..agents.llm import ModelClient, ModelRejected
        try:
            res = self.client.complete("search_proposal", inputs)
        except ModelRejected as e:             # one bad proposal must not end a long run: stand-in for this batch
            res = ModelClient("rule").complete("search_proposal", inputs)
            res.meta.update(fallback_after_rejection=e.reasons)
        self.calls.append({"meta": res.meta, "rationale": res.output.get("rationale"),
                           "regions": res.output["regions"]})
        out = []
        for r in res.output["regions"]:
            out += [_uniform(r["template"], self.rng, r) for _ in range(r["count"])]
        return out

    def tell(self, scn, value):
        pass


STRATEGIES = {"random": RandomStrategy, "optuna": OptunaStrategy, "model": ModelStrategy}


class FailureSearch:
    def __init__(self, strategies=("random", "optuna", "model"), budget=160, batch=16, workers=None, library=None,
                 library_name="current", seed=0, run_id=None, client=None, templates=None, obstacle=False):
        """templates: restrict the search to some object types (a focused search); default all probe templates."""
        global SPACE, OBSTACLE
        OBSTACLE = obstacle
        if templates:
            SPACE = {t: v for t, v in SPACE.items() if t in templates}
            self.templates = list(templates)
        else:
            self.templates = list(PROBE_TEMPLATES)
        self.strategies, self.budget, self.batch, self.seed = strategies, budget, batch, seed
        self.library, self.library_name, self.client = library, library_name, client
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S_failsearch")
        self.out = ROOT / "runs" / self.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool = TrialPool(workers)
        self.report = {"run_id": self.run_id, "library": library_name, "budget_per_strategy": budget,
                       "obstacle": obstacle,
                       "batch": batch, "space": SPACE, "xy": XY, "strategies": {}}
        self.t0 = time.time()

    def _save(self):
        self.report["wall_time_s"] = round(time.time() - self.t0, 1)
        (self.out / "report.json").write_text(json.dumps(self.report, indent=1, default=float))

    def run_strategy(self, name, seed_base):
        rng = np.random.default_rng(self.seed + seed_base)
        strat = STRATEGIES[name](rng, seed=self.seed, client=self.client, budget=self.budget)
        history, rejected, i = [], collections.Counter(), 0
        while len(history) < self.budget:
            n = min(self.batch, self.budget - len(history))
            jobs, scns = [], []
            for scn in strat.propose(n, history):
                seed = seed_base + i
                i += 1
                ok, why = plausible(scn, seed)
                if not ok:
                    rejected[why] += 1
                    strat.tell(scn, None)
                    continue
                cfg = trial_cfg("probe", self.out, save_failures=False, success_sample=0, save_video=False,
                                save_poses=False, tag=f"{name}", library=self.library, oracle="on_failure",
                                task_spec=probe(seed, scn))
                jobs.append((cfg, seed))
                scns.append(scn)
            if not jobs:
                if sum(rejected.values()) > 50 * self.budget:
                    raise RuntimeError(f"{name}: almost every proposal is implausible: {dict(rejected)}")
                continue
            res = {r["seed"]: r for r in self.pool.map(jobs)}
            for (cfg, seed), scn in zip(jobs, scns):
                r = res[seed]
                t = scn["template"]
                if r["success"]:
                    outcome = "success"
                elif r["oracle"] and r["oracle"]["success"]:
                    outcome = "valid_failure"
                else:
                    outcome = "unsolvable"
                strat.tell(scn, 1.0 if outcome == "valid_failure" else 0.0)
                history.append({"i": len(history), "seed": seed, "scenario": scn, "outcome": outcome,
                                "label": (r["labels"] or [None])[0], "grasp": r["grasp"].get(t)})
            k = sum(h["outcome"] == "valid_failure" for h in history)
            _log(f"{name}: {len(history)}/{self.budget} trials, {k} valid failures")
        metrics = _metrics(history)
        metrics["rejected_implausible"] = dict(rejected)
        entry = {"metrics": metrics, "trials": history}
        if name == "model":
            entry["model_calls"] = strat.calls
        self.report["strategies"][name] = entry
        self._save()
        return history

    def run(self):
        try:
            all_hist = []
            for j, name in enumerate(self.strategies):
                all_hist += self.run_strategy(name, seed_base=(j + 1) * 1_000_000)
            self.report["capability_map"] = capability_map(all_hist)
            self.report["clusters"] = clusters(all_hist)
            self._save()
            return self.report
        finally:
            self.pool.close()


def _cluster_key(h):
    """(object type, grasp skill), as in the explorer; one cause often shows as several labels."""
    return f"{h['scenario']['template']}/{h['grasp']}"


def _metrics(history):
    n = len(history)
    vf = [h for h in history if h["outcome"] == "valid_failure"]
    first = {}
    for h in vf:
        first.setdefault(_cluster_key(h), h["i"] + 1)
    return {"trials": n, "success": sum(h["outcome"] == "success" for h in history), "valid_failures": len(vf),
            "unsolvable": sum(h["outcome"] == "unsolvable" for h in history),
            "valid_failures_per_100": round(100 * len(vf) / max(n, 1), 1), "valid_failure_ci95": wilson(len(vf), n),
            "distinct_clusters": len(first), "trials_to_find": first}


def _band(v, lo, hi, k=4):
    i = min(k - 1, int((v - lo) / (hi - lo + 1e-12) * k))
    step = (hi - lo) / k
    return i, f"{lo + i * step:.3g}-{lo + (i + 1) * step:.3g}"


def capability_map(history):
    """Success among solvable trials, by template x size band x reach band (all strategies pooled)."""
    cells = collections.defaultdict(lambda: {"n": 0, "success": 0, "valid_failures": 0, "unsolvable": 0})
    for h in history:
        t = h["scenario"]["template"]
        k = SIZE_KEY[t]
        _, band = _band(h["scenario"]["params"][k], *RANGES[t][k])
        r = math.hypot(*h["scenario"]["pos"])
        reach = next(name for lo, hi, name in REACH_BANDS if lo <= r < hi)
        c = cells[(t, f"{k} {band}", reach)]
        c["n"] += 1
        c[{"success": "success", "valid_failure": "valid_failures", "unsolvable": "unsolvable"}[h["outcome"]]] += 1
    out = []
    for (t, size, reach), c in sorted(cells.items()):
        solvable = c["success"] + c["valid_failures"]
        out.append({"template": t, "size": size, "reach": reach, **c,
                    "success_rate": round(c["success"] / solvable, 3) if solvable else None})
    return out


def clusters(history):
    """Valid failures grouped by (template, skill) with their labels and the parameter region they span."""
    groups = collections.defaultdict(list)
    for h in history:
        if h["outcome"] == "valid_failure":
            groups[_cluster_key(h)].append(h)
    out = []
    for key, hs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        t = hs[0]["scenario"]["template"]
        same_t = [h for h in history if h["scenario"]["template"] == t and h["outcome"] != "unsolvable"]
        region = {k: [round(min(h["scenario"]["params"][k] for h in hs), 4),
                      round(max(h["scenario"]["params"][k] for h in hs), 4)] for k in SPACE[t]}
        rs = [math.hypot(*h["scenario"]["pos"]) for h in hs]
        if hs[0]["scenario"].get("obstacle"):
            region_ob = {k: [round(min(h["scenario"]["obstacle"][k] for h in hs), 4),
                             round(max(h["scenario"]["obstacle"][k] for h in hs), 4)] for k in ("dist", "radius", "height")}
        else:
            region_ob = None
        labels = collections.Counter(h["label"] for h in hs)
        out.append({"key": key, "template": t, "skill": hs[0]["grasp"], "labels": dict(labels.most_common()),
                    "count": len(hs),
                    "template_failure_rate": round(len(hs) / max(len(same_t), 1), 3), "region": region,
                    "reach_r": [round(min(rs), 3), round(max(rs), 3)], **({"obstacle": region_ob} if region_ob else {}),
                    "examples": [{"seed": h["seed"], "scenario": h["scenario"]} for h in hs[:5]]})
    return out
