"""Studio: typed request -> understood task + scene preview -> plan -> run -> (if steps fail) repair -> re-run.

Two parts:
  plan_spec / plan_program   quick, called by the server per request (model calls + one scene build, no physics)
  run_job(job_dir)           the long job, run in its own process by scripts/studio_job.py; writes events.jsonl

Events (one JSON object per line in runs/<job>/events.jsonl), all with "t" (seconds since start):
  stage   {stage: run|repair|rerun|done, state: start|end, ...}
  trial   {phase: run|rerun, seed, success, labels}               one per finished episode
  log     {line, stage}                                            human-readable progress from the repair loop
  result  {before, after, repair}                                  final outcome
  error   {message}
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback

from .runner import ROOT

EXAMPLES = [
    "Sort the blocks and the bowl onto the trays of the same colour.",
    "Stack the blocks, largest at the bottom.",
    "Move the plate onto the blue zone.",
    "Put the mug on the tray, don't knock over the bottle.",
]


# ----------------------------------------------------------------------------- planning (in the server process)
def plan_spec(prompt: str, backend: str | None = None) -> dict:
    """Task spec from the request + a scene preview (the first episode's layout)."""
    from .agents.jobs.task_spec import to_task
    from .agents.llm import ModelClient, ModelRejected
    client = ModelClient(backend)
    try:
        res = client.complete("task_spec", {"prompt": prompt})
    except ModelRejected as e:
        return {"status": "rejected", "reasons": e.reasons}
    out = {"status": res.output["status"], "output": res.output, "meta": res.meta}
    if res.output["status"] == "ok":
        task = to_task(res.output, {"prompt": prompt})
        out["task"] = task
        out["preview"] = preview(task)
    return out


def plan_program(prompt: str, task: dict, backend: str | None = None) -> dict:
    """A plan program: a verified composite with exactly this spec if one exists, else written by the model.
    Falls back to the fixed plan (no program) if the model's program is rejected."""
    from .agents.llm import ModelClient, ModelRejected
    from .composites import find_composite, spec_hash
    card = find_composite(task)
    if card and card.get("verified_spec") == spec_hash(task):
        ev = card["evidence"]
        return {"program": card["program"], "source": {"composite": card["name"], "version": card["version"]},
                "explanation": f"Reusing the verified program {card['name']} v{card['version']} "
                               f"({ev['success']}/{ev['trials']} in verification); no model call."}
    try:
        from .agents.jobs.plan_program import library_capabilities
        pr = ModelClient(backend).complete("plan_program", {"prompt": prompt, "task": task,
                                                            "capabilities": library_capabilities()})
    except ModelRejected as e:
        return {"program": None, "source": None, "explanation": "The model's program was rejected "
                f"({'; '.join(e.reasons)[:300]}); the fixed plan will run instead."}
    return {"program": pr.output["program"], "source": pr.meta, "explanation": pr.output.get("explanation", "")}


# The studio does not repair failures confidently rarer than this (95% upper bound): a gate needs ~10 failed cases,
# so a 3% failure means ~300 scenes before the search even starts, and the gates rarely pass. Reported instead.
RARE_FAILURE_RATE = 0.10
HARD_PHYSICS = {"friction": [0.35, 0.7], "mass": [1.3, 2.0]}     # slippery, heavy objects (as measure_recovery)


def preview(task: dict) -> dict:
    """Static scene + one pose frame of the first layout that fits (= the job's first scene), no physics."""
    import mujoco
    from .sim.recorder import export_geoms
    from .sim.scene import build_scene
    seeds, _ = valid_seeds(task, 1, max_tries=20)
    if not seeds:
        return None
    scene = build_scene(task, seed=seeds[0])
    mujoco.mj_forward(scene.model, scene.data)
    d = scene.data
    return {"fps": 30, "scene": export_geoms(scene.model),
            "frames": [{"t": 0.0, "p": d.xpos.round(4).tolist(), "q": d.xquat.round(4).tolist()}], "events": []}


# ----------------------------------------------------------------------------- the job (its own process)
class Events:
    def __init__(self, path: pathlib.Path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, type_, **kw):
        self.f.write(json.dumps({"type": type_, "t": round(time.time() - self.t0, 2), **kw}, default=str) + "\n")
        self.f.flush()


STAGE_WORDS = [("[create_skill] before", "measure"), ("[create_skill] proposal", "propose"),
               ("[create_skill] search", "search"), ("[create_skill] no candidate", "propose"), ("[create_skill] gates failed", "propose"),
               ("[create_skill] tune", "tune"), ("[create_skill] gate", "verify"),
               ("[create_skill] PROMOTED", "learned"), ("[create_skill]", None), ("[auto_repair]", "check"), ("[investigate]", "diagnose"), ("[explore] before", "measure"),
               ("[explore] gate", "verify"), ("[explore] PROMOTED", "learned"), ("[explore] PATCHED", "learned"),
               ("[explore] ROUTED", "learned"), ("[explore]", "search")]


class LogTee:
    """stdout of the job process -> log events with a coarse repair stage."""

    def __init__(self, emit, raw):
        self.emit, self.raw, self.buf = emit, raw, ""

    def write(self, s):
        self.raw.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                stage = next((st for key, st in STAGE_WORDS if line.startswith(key)), None)
                self.emit("log", line=line, stage=stage)

    def flush(self):
        self.raw.flush()


def valid_seeds(task: dict, n: int, max_tries: int | None = None) -> tuple[list[int], list[int]]:
    """The first n seeds whose layout fits on the table (a crowded spec cannot be placed for every seed; such a
    scene is not a robot failure, so it is skipped instead of run). Returns (seeds, skipped)."""
    from .sim.scene import build_scene
    seeds, skipped = [], []
    for s in range(max_tries or 4 * n + 8):
        if len(seeds) == n:
            break
        try:
            build_scene(task, seed=s)
            seeds.append(s)
        except RuntimeError:
            skipped.append(s)
    return seeds, skipped


def _batch(job, task, seeds, run_id, phase, emit):
    from .runner import run_batch
    emit("stage", stage=phase, state="start", run_id=run_id, n=len(seeds), seeds=seeds)
    s = run_batch(task["name"], workers=job["workers"], task_spec=task, seeds=seeds, run_id=run_id,
                  save_video=False, save_poses=True, save_failures=True, success_sample=1, progress=False,
                  recovery=job["recovery"], perturb=HARD_PHYSICS if job.get("hard") else None,
                  on_result=lambda r: emit("trial", phase=phase, seed=r["seed"], success=r["success"],
                                           labels=r["labels"]))
    emit("stage", stage=phase, state="end", run_id=run_id, success=s["success"], trials=s["trials"],
         failure_labels=s["failure_labels"], wall_time_s=s["wall_time_s"])
    return s


def _missing_call(run_dir: pathlib.Path):
    """The use(capability, obj, target) call that found no skill, from the first run's trial records."""
    for p in sorted(run_dir.glob("trials/*.json")):
        rec = json.loads(p.read_text())
        if rec.get("missing_capability"):
            return rec.get("missing_call") or {"capability": rec["missing_capability"], "obj": None, "target": None}
    return None


def _create_skill(job, task, call, rel, emit):
    """Model-written primitive for the missing capability: proposals -> search -> tune -> gates -> card."""
    from .agents.llm import ModelClient
    from .create_skill import SkillCreator
    sc = SkillCreator(call["capability"], workers=job["workers"], client=ModelClient(job.get("backend")),
                      task_spec=task, program=task.get("program"), obj=call.get("obj"), target=call.get("target"),
                      run_id=f"{rel}/create_{call['capability']}")
    rep = sc.run()
    gate = rep.get("phases", {}).get("gate") or {}
    return {"changed": rep.get("outcome") == "promoted", "created": True, "capability": call["capability"],
            "run_id": rep["run_id"], "outcome": rep.get("outcome"),
            "promoted": (rep.get("promoted") or {}).get("card"),
            "proposals": [{"name": p["card"]["name"], "attempt": p["attempt"], "option": p["option"],
                           "description": p["card"]["description"][:200]}
                          for p in rep.get("phases", {}).get("proposals", [])],
            "held_out": gate.get("held_out"), "regression": (gate.get("no_regression") or {}).get("pass"),
            "trials": rep.get("trials_simulated")}


def run_job(job_dir: pathlib.Path):
    from .agents.llm import ModelClient
    from .auto_repair import auto_repair
    job_dir = pathlib.Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    emit = Events(job_dir / "events.jsonl")
    sys.stdout = LogTee(emit, sys.stdout)
    rel = job_dir.relative_to(ROOT / "runs").as_posix()
    task = {**job["task"]}
    if job.get("program"):
        task["program"], task["program_source"] = job["program"], job.get("program_source")
    result = {"before": None, "repair": None, "after": None}
    try:
        seeds, skipped = valid_seeds(task, job["n"])
        if skipped:
            emit("log", line=f"[studio] {len(skipped)} layout(s) did not fit on the table and were skipped "
                             f"(seeds {skipped[:10]})", stage=None)
        if not seeds:
            raise RuntimeError("no layout of this task fits on the table: fewer or smaller objects needed")
        before = _batch(job, task, seeds, f"{rel}/run", "run", emit)
        result["before"] = {k: before[k] for k in ("success", "trials", "failure_labels", "run_id")}
        robot_failures = sum(v for k, v in before["failure_labels"].items() if k != "episode_error")
        if robot_failures and job.get("auto_fix"):
            call = _missing_call(job_dir / "run")
            emit("stage", stage="repair", state="start", mode="create" if call else "repair",
                 capability=call and call["capability"])
            if call:
                result["repair"] = _create_skill(job, task, call, rel, emit)
            else:
                rep = auto_repair([job_dir / "run"], workers=job["workers"], client=ModelClient(job.get("backend")),
                                  min_failures=1, max_cases=4, run_id=f"{rel}/repair",
                                  explorer_kwargs={"min_fail_rate": RARE_FAILURE_RATE})
                changes = [c for c in rep["clusters"] if c.get("repair") and
                           (c["repair"].get("routed") or c["repair"].get("patched") or c["repair"].get("promoted"))]
                result["repair"] = {"changed": bool(changes), "run_id": rep["run_id"],
                                    "clusters": [{k: c.get(k) for k in ("kind", "skill", "valid_failures", "labels",
                                                                         "action", "investigation", "repair")}
                                                 for c in rep["clusters"]]}
            emit("stage", stage="repair", state="end", **result["repair"])
            if result["repair"]["changed"]:
                after = _batch(job, task, seeds, f"{rel}/rerun", "rerun", emit)
                result["after"] = {k: after[k] for k in ("success", "trials", "failure_labels", "run_id")}
        emit("result", **result)
        emit("stage", stage="done", state="end")
    except Exception as e:
        emit("error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        raise
