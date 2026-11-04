"""Skill packs: move items between the base library (skill_library/, shipped) and user libraries.

An item is a skill card (with its routing evidence and the patches on it), a parameter patch ("patch:<skill>/<group>")
or a verified composite. Three operations:

    export      user-library items (or named items) -> one JSON pack, with each item's lineage and evidence
    import      a pack -> the user library, but only items that re-pass the gates HERE (evidence measured on another
                machine or code version is not trusted until it is measured again)
    contribute  user-library items -> the base library, only after re-passing the gates against the base library
                (what everyone gets must be better than the base without it, and break nothing in it)

Both import and contribute use one pre-declared re-verification gate, the same rules as the explorer's gates:
    request     the item's own recorded request (lineage: the trigger task + region, or the typed request's task spec
                + plan program; a patch's triggers), on fresh dev seeds no earlier gate used (dev, start 2000), the
                library WITH the item vs WITHOUT it, paired: McNemar one-sided alpha 0.05 -> improve; if it already
                succeeds >= 80% without the item it must HOLD (no significant worsening); with the item >= 70%
    regression  the fixed regression tasks + every trigger / request recorded in the target library: goal rate not
                more than 0.1 lower and confidence intervals overlapping (explorer rule), fresh dev seeds
Composites are exported but not imported or contributed yet (their verification needs the full program check).
"""
from __future__ import annotations

import copy
import json
import pathlib
import time

from .runner import ROOT, TrialPool, trial_cfg
from .search.explore import ALPHA, REGRESSION_TASKS, _rate, mcnemar_p
from .skills import library as L
from .tasks.splits import split_seeds

FORMAT = "kintsugi-skill-pack"
GATE_SEEDS_START = 2000          # dev seeds 102000+: never used by the explorer / creator gates (start 0..~100)
N_REQUEST, N_REGRESSION = 16, 8


def _log(msg):
    print(f"[skill_packs] {msg}", flush=True)


def _read(p):
    return json.loads(pathlib.Path(p).read_text()) if pathlib.Path(p).exists() else {}


# ---------------------------------------------------------------------------------------------------- items
def user_items() -> dict:
    """What the user library holds: {"cards": {name: card}, "performance": {...}, "patches": {...},
    "composites": {name: card}}."""
    U = L.USER_DIR
    comps = {c["name"]: c for c in (_read(p) for p in sorted((U / "composites").glob("*.json")))} \
        if (U / "composites").exists() else {}
    return {"cards": L.load_cards(U) if U.exists() else {}, "performance": _read(U / "performance.json"),
            "patches": _read(U / "patches.json"), "composites": comps}


def select(items: dict, names=None) -> dict:
    """Items by name: a card name (its evidence and the patches on it come along), "patch:<skill>/<group>", or a
    composite name. None = everything."""
    if not names:
        return copy.deepcopy(items)
    out = {"cards": {}, "performance": {}, "patches": {}, "composites": {}}
    for n in names:
        if n.startswith("patch:"):
            skill, _, group = n[6:].partition("/")
            e = items["patches"].get(skill, {}).get(group)
            if e is None:
                raise KeyError(f"no patch {n}")
            out["patches"].setdefault(skill, {})[group] = e
        elif n in items["cards"]:
            out["cards"][n] = items["cards"][n]
            if n in items["performance"]:
                out["performance"][n] = items["performance"][n]
            if n in items["patches"]:
                out["patches"][n] = items["patches"][n]
        elif n in items["composites"]:
            out["composites"][n] = items["composites"][n]
        else:
            raise KeyError(f"no item {n!r} (cards: {sorted(items['cards'])}, composites: {sorted(items['composites'])})")
    return copy.deepcopy(out)


def export_pack(out_path, names=None, items=None) -> dict:
    from .version import code_version
    sel = select(items or user_items(), names)
    pack = {"format": FORMAT, "version": 1, "created": time.strftime("%Y-%m-%d %H:%M"),
            "code_version": code_version(), "items": sel}
    pathlib.Path(out_path).write_text(json.dumps(pack, indent=1))
    _log(f"exported {len(sel['cards'])} cards, {sum(len(v) for v in sel['patches'].values())} patches, "
         f"{len(sel['composites'])} composites -> {out_path}")
    return pack


def static_errors(items: dict) -> list[str]:
    """Checks that need no simulation: card structure, sandbox rules for generated code, known implementations."""
    from .primitive_code import check_primitive
    errs = []
    for n, c in items["cards"].items():
        miss = [k for k in ("name", "kind", "version", "params", "lineage") if k not in c]
        if miss:
            errs.append(f"{n}: missing {miss}")
            continue
        if c.get("impl") == "generated":
            errs += [f"{n}: {e}" for e in check_primitive(c.get("code", ""))]
        elif c["kind"] in ("grasp", "grasp_primitive") and c.get("impl", n) not in L.GRASP_SKILLS:
            errs.append(f"{n}: unknown grasp implementation {c.get('impl', n)!r}")
    return errs


# ---------------------------------------------------------------------------------------------------- requests
def _card_requests(card: dict) -> list[dict]:
    """The request(s) a card was made for, from its newest lineage entry: {"task", "kwargs", "program"?, "spec"?}."""
    e = (card.get("lineage") or [{}])[-1]
    tr = e.get("trigger") or {}
    if tr.get("task"):
        return [{"task": tr["task"], "kwargs": tr.get("task_kwargs") or {}}]
    if e.get("program") and (e.get("task_spec") or e.get("task")):
        if e.get("task_spec"):
            return [{"task": "spec", "spec": e["task_spec"], "program": e["program"]}]
        return [{"task": e["task"], "kwargs": {}, "program": e["program"]}]
    return []


def requests_for(change: dict) -> list[dict]:
    reqs = []
    for c in change["cards"].values():
        reqs += _card_requests(c)
    for groups in [*change["patches"].values(), *change["performance"].values()]:
        for g in groups.values():
            reqs += [{"task": t["task"], "kwargs": t.get("kwargs") or {}} for t in g.get("triggers", [])]
    out = []
    for r in reqs:
        if r not in out:
            out.append(r)
    return out


def regression_items(cards: dict, perf: dict, patches: dict, exclude=()) -> list[dict]:
    """Fixed regression tasks + every request recorded in a library (exploration triggers, created skills' requests,
    patch / evidence triggers), so a change cannot undo what the library already does."""
    items = [{"task": t, "kwargs": {}} for t in REGRESSION_TASKS]
    for c in cards.values():
        for it in _card_requests(c):
            if it not in items:
                items.append(it)
    for groups in [*perf.values(), *patches.values()]:
        for g in groups.values():
            for t in g.get("triggers", []):
                it = {"task": t["task"], "kwargs": t.get("kwargs") or {}}
                if it not in items:
                    items.append(it)
    return [it for it in items if it not in exclude]


def _label(req) -> str:
    if req["task"] == "spec":
        sp = req.get("spec") or (req.get("kwargs") or {}).get("task_spec") or {}
        return f"spec[{sp.get('name', '?')}]"
    kw = req.get("kwargs") or {}
    if "region" in kw:
        r = kw["region"]
        return f"{req['task']}[{'+'.join(r['templates'][:2]) if r.get('templates') else r.get('template')}]"
    return req["task"]


# ---------------------------------------------------------------------------------------------------- gate
class Reverifier:
    def __init__(self, target="base", workers=None, run_id=None):
        """target "base": the change is judged against the base library alone (contribute); "current": against the
        merged library this checkout uses (import)."""
        if target == "base":
            self.cards = L.load_cards(L.BASE_DIR)
            self.perf = _read(L.BASE_DIR / "performance.json")
            self.patches = _read(L.BASE_DIR / "patches.json")
        else:
            from .skills.router import load_patches, load_performance
            self.cards, self.perf, self.patches = L.load_cards(), load_performance(), load_patches()
        self.target = target
        self.run_id = run_id or time.strftime(f"%Y%m%d-%H%M%S_reverify_{target}")
        self.out = ROOT / "runs" / self.run_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.pool = TrialPool(workers)

    def _with(self, change):
        cards = {**self.cards, **change["cards"]}
        perf, patches = copy.deepcopy(self.perf), copy.deepcopy(self.patches)
        for k, gs in change["performance"].items():
            perf.setdefault(k, {}).update(gs)
        for k, gs in change["patches"].items():
            patches.setdefault(k, {}).update(gs)
        return cards, perf, patches

    def _run(self, tag, req, seeds, lib):
        cards, perf, patches = lib
        jobs = []
        for s in seeds:
            # created skills carry their typed request as "spec"; patch triggers keep it in kwargs["task_spec"]
            # and run like any named task below
            if req.get("program") or "spec" in req:
                from .tasks.specs import make_task
                spec = (make_task("spec", seed=s, task_spec=req["spec"]) if req["task"] == "spec"
                        else make_task(req["task"], seed=s, **(req.get("kwargs") or {})))
                if req.get("program"):
                    spec = {**spec, "program": req["program"]}
                cfg = trial_cfg("spec", self.out, save_failures=False, success_sample=0, save_video=False,
                                save_poses=False, tag=tag, library=cards, performance=perf, patches=patches,
                                task_kwargs={"task_spec": spec}, perturb={})
                cfg["task_spec"] = spec
            else:
                cfg = trial_cfg(req["task"], self.out, save_failures=False, success_sample=0, save_video=False,
                                save_poses=False, tag=tag, library=cards, performance=perf, patches=patches,
                                task_kwargs=req.get("kwargs") or {}, perturb={})
            jobs.append((cfg, s))
        res = {r["seed"]: r for r in self.pool.map(jobs)}
        return [res[s] for s in seeds]

    def gate(self, change: dict, name="change") -> dict:
        errs = static_errors(change)
        if errs:
            return {"pass": False, "reason": "static check failed", "errors": errs}
        reqs = requests_for(change)
        if not reqs:
            return {"pass": False, "reason": "no recorded request to verify on (lineage has no trigger, task or "
                                             "program): it cannot be re-verified, so it is not added"}
        without, with_ = (self.cards, self.perf, self.patches), self._with(change)
        g = {"name": name, "target": self.target, "requests": {}, "regression": {}, "pass": True}
        seeds = split_seeds("dev", N_REQUEST, start=GATE_SEEDS_START)
        for i, req in enumerate(reqs):
            lab = _label(req)
            b = self._run(f"{name}_req{i}_without", req, seeds, without)
            a = self._run(f"{name}_req{i}_with", req, seeds, with_)
            ok_b, ok_a = [bool(r["success"]) for r in b], [bool(r["success"]) for r in a]
            rb, ra = _rate(b), _rate(a)
            mode = "hold" if rb["rate"] >= 0.8 else "improve"
            p_better, p_worse = mcnemar_p(ok_b, ok_a), mcnemar_p(ok_a, ok_b)
            ok = (p_better < ALPHA if mode == "improve" else p_worse >= ALPHA) and ra["rate"] >= 0.7
            g["requests"][lab] = {"without": rb, "with": ra, "mode": mode, "p_better": p_better, "p_worse": p_worse,
                                  "pass": ok}
            g["pass"] &= ok
            _log(f"{name} request {lab}: {ra['success']}/{ra['trials']} with vs {rb['success']}/{rb['trials']} "
                 f"without ({mode}, p={p_better if mode == 'improve' else p_worse:.3g}) {'pass' if ok else 'FAIL'}")
        rseeds = split_seeds("dev", N_REGRESSION, start=GATE_SEEDS_START)
        for it in regression_items(self.cards, self.perf, self.patches, exclude=reqs):
            lab = _label(it)
            b = _rate(self._run(f"{name}_reg_without", it, rseeds, without))
            a = _rate(self._run(f"{name}_reg_with", it, rseeds, with_))
            ok = a["goal_rate"] >= b["goal_rate"] - 0.1 and a["goal_ci95"][1] >= b["goal_ci95"][0]
            g["regression"][lab] = {"without": b, "with": a, "pass": ok}
            g["pass"] &= ok
        bad = [k for k, v in g["regression"].items() if not v["pass"]]
        _log(f"{name} regression on {len(g['regression'])} tasks: {'pass' if not bad else 'FAIL ' + str(bad)}")
        (self.out / f"gate_{name}.json").write_text(json.dumps(g, indent=1, default=float))
        return g

    def close(self):
        self.pool.close()


def _summary(g):
    return {"requests": {k: {"with": [v["with"]["success"], v["with"]["trials"]],
                             "without": [v["without"]["success"], v["without"]["trials"]],
                             "mode": v["mode"], "p": v["p_better"] if v["mode"] == "improve" else v["p_worse"]}
                         for k, v in g.get("requests", {}).items()},
            "regression_pass": all(v["pass"] for v in g.get("regression", {}).values())}


def _split(change: dict) -> list[tuple[str, dict]]:
    """One gate per card (with its evidence and patches), one per stand-alone patch."""
    units = []
    for n, c in change["cards"].items():
        units.append((n, {"cards": {n: c}, "performance": {n: change["performance"][n]} if n in change["performance"] else {},
                          "patches": {n: change["patches"][n]} if n in change["patches"] else {}, "composites": {}}))
    for skill, groups in change["patches"].items():
        if skill in change["cards"]:
            continue
        for grp, e in groups.items():
            units.append((f"patch:{skill}/{grp}", {"cards": {}, "performance": {}, "patches": {skill: {grp: e}},
                                                    "composites": {}}))
    for skill, groups in change["performance"].items():
        if skill not in change["cards"]:
            units.append((f"evidence:{skill}", {"cards": {}, "performance": {skill: groups}, "patches": {},
                                                 "composites": {}}))
    return units


# ---------------------------------------------------------------------------------------------------- operations
def import_pack(path, workers=None, verify=True) -> dict:
    """Pack -> user library; each item is re-verified against the library this checkout uses (verify=False skips
    the gate and writes the items marked unverified: for moving one's own pack between checkouts of the same code)."""
    pack = json.loads(pathlib.Path(path).read_text())
    if pack.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT}")
    items = pack["items"]
    report = {"pack": str(path), "imported": [], "rejected": [], "skipped": []}
    for n in items.get("composites", {}):
        report["skipped"].append({"name": n, "reason": "composites are not imported yet"})
    rv = Reverifier("current", workers) if verify else None
    try:
        for name, unit in _split(items):
            known = L.load_cards().get(name)
            if known and known == unit["cards"].get(name):
                report["skipped"].append({"name": name, "reason": "identical item already in the library"})
                continue
            g = rv.gate(unit, name.replace(":", "_").replace("/", "_")) if rv else {"pass": True, "unverified": True}
            if not g["pass"]:
                report["rejected"].append({"name": name, "reason": g.get("reason") or "failed the gates",
                                           "gate": _summary(g), "errors": g.get("errors")})
                continue
            entry = {"source": "imported", "pack": pathlib.Path(path).name, "date": time.strftime("%Y-%m-%d"),
                     "pack_code_version": pack.get("code_version"),
                     **({"gate": _summary(g), "run_id": rv.run_id} if rv else {"verified_here": False})}
            _write(unit, L.USER_DIR, entry)
            report["imported"].append({"name": name, **({"gate": _summary(g)} if rv else {})})
    finally:
        if rv:
            rv.close()
    _log(f"imported {len(report['imported'])}, rejected {len(report['rejected'])}, skipped {len(report['skipped'])}")
    return report


def contribute(names=None, workers=None, dry_run=False) -> dict:
    """User-library items -> base library, each only after re-passing the gates against the base library. The item
    then leaves the user library (the base copy carries a "contribution" lineage entry with the gate)."""
    change = select(user_items(), names)
    report = {"contributed": [], "rejected": [], "skipped": []}
    for n in change["composites"]:
        report["skipped"].append({"name": n, "reason": "composites are not contributed yet"})
    rv = Reverifier("base", workers)
    report["run_id"] = rv.run_id
    try:
        for name, unit in _split(change):
            g = rv.gate(unit, name.replace(":", "_").replace("/", "_"))
            if not g["pass"]:
                report["rejected"].append({"name": name, "reason": g.get("reason") or "failed the gates",
                                           "gate": _summary(g), "errors": g.get("errors")})
                continue
            if not dry_run:
                entry = {"source": "contribution", "date": time.strftime("%Y-%m-%d"), "run_id": rv.run_id,
                         "gate": _summary(g)}
                _write(unit, L.BASE_DIR, entry)
                _remove_user(unit)
            report["contributed"].append({"name": name, "gate": _summary(g), "dry_run": dry_run})
    finally:
        rv.close()
    _log(f"contributed {len(report['contributed'])}, rejected {len(report['rejected'])}"
         + (" (dry run: nothing written)" if dry_run else ""))
    return report


def _write(unit, lib_dir, entry):
    """Write a unit's card(s), evidence and patches into a library; the card gets a lineage entry."""
    lib_dir = pathlib.Path(lib_dir)
    for n, c in unit["cards"].items():
        c = copy.deepcopy(c)
        c["lineage"] = list(c.get("lineage", [])) + [{"version": c["version"], **entry}]
        L.save_card(c, lib_dir)
    for fname, key in (("performance.json", "performance"), ("patches.json", "patches")):
        if unit[key]:
            data = _read(lib_dir / fname)
            for k, gs in unit[key].items():
                data.setdefault(k, {}).update(gs)
            lib_dir.mkdir(parents=True, exist_ok=True)
            (lib_dir / fname).write_text(json.dumps(data, indent=1, sort_keys=True))


def _remove_user(unit):
    U = L.USER_DIR
    for n in unit["cards"]:
        p = U / f"{n}.json"
        if p.exists():
            p.unlink()
    for fname, key in (("performance.json", "performance"), ("patches.json", "patches")):
        if unit[key] and (U / fname).exists():
            data = _read(U / fname)
            for k, gs in unit[key].items():
                for grp in gs:
                    data.get(k, {}).pop(grp, None)
                if k in data and not data[k]:
                    del data[k]
            (U / fname).write_text(json.dumps(data, indent=1, sort_keys=True))
