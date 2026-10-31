"""Kintsugi server: serves saved runs, the skill library and the robot meshes to the web app.

    uvicorn server.app:app --port 8000          (from the repo root; the web app's dev server proxies /api here)

Read-only over files on disk (except the studio, which starts one job at a time): runs/<id>/ (report.json, summary.json, trials/*.json, media/*_poses.json.gz),
skill_library/ (cards, history, primitives, composites, reference, performance.json, patches.json).
Physics never runs in the browser: the 3D view replays saved pose logs.
"""
from __future__ import annotations

import base64
import functools
import gzip
import json
import pathlib
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LIB = ROOT / "skill_library"
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Kintsugi")
app.add_middleware(GZipMiddleware, minimum_size=2000)

KINDS = [("_explore_", "explore"), ("_explore", "explore"), ("_investigate_", "investigation"), ("_failsearch", "failure_search"), ("_investigate", "investigation"),
         ("_create_", "create_skill"), ("_transfer_", "transfer"), ("_verify_program", "verify_program"),
         ("_auto_repair", "auto_repair"), ("_measure_recovery", "recovery"), ("_quick_check", "quick_check")]


def run_kind(name: str) -> str:
    """By the run folder's name; nested runs (auto_repair/explore_mug_...) by their own folder name."""
    if name.split("/")[-1] == "repair":              # a studio job's repair folder
        return "auto_repair"
    last = "_" + name.split("/")[-1]
    for probe in (last, name):
        for key, kind in KINDS:
            if key in probe:
                return kind
    return "task"


def _read(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _safe(rel: str) -> pathlib.Path:
    """A path inside runs/ (rejects anything that escapes it)."""
    p = (RUNS / rel).resolve()
    if RUNS.resolve() not in p.parents and p != RUNS.resolve():
        raise HTTPException(404)
    return p


def _run_dirs():
    """Run folders up to three levels deep (auto-repair and studio jobs nest their runs)."""
    out = []

    def walk(d, depth):
        if (d / "report.json").exists() or (d / "summary.json").exists() or (d / "trials").exists():
            out.append(d)
        if depth < 3:
            for s in d.iterdir():
                if s.is_dir() and s.name not in ("trials", "media", "live", "ledgers", "smoke"):
                    walk(s, depth + 1)

    for d in RUNS.iterdir():
        if d.is_dir() and d.name != "smoke":
            walk(d, 1)
    return out


KIND_TITLES = {"explore": "Repair search", "investigation": "Diagnosis", "failure_search": "Failure search",
               "create_skill": "Skill creation", "transfer": "Transfer", "task": "Task run", "auto_repair": "Auto-repair",
               "verify_program": "Program check", "recovery": "Recovery test", "quick_check": "Quick check"}
STUDIO_PHASES = {"run": "first run", "rerun": "after the fix", "repair": "repair"}


def run_title(rel: str, kind: str, task=None) -> str:
    """A readable name for a run folder (the id stays the key)."""
    parts = rel.split("/")
    when = re.match(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", parts[0])
    stamp = f"{when.group(3)}.{when.group(2)} {when.group(4)}:{when.group(5)}" if when else ""
    if parts[0].endswith("_studio"):
        try:
            prompt = _read(RUNS / parts[0] / "job.json")["prompt"]
        except (OSError, ValueError, KeyError):
            prompt = "studio"
        phase = STUDIO_PHASES.get(parts[1], parts[1].replace("_", " ")) if len(parts) > 1 else ""
        return f"Studio · {prompt[:60]} · {phase}" + (f" · {parts[2].replace('_', ' ')}" if len(parts) > 2 else "")
    if parts[0] == "demo":
        return "Demo · " + re.sub(r"^\d+_", "", parts[-1]).replace("_", " ")
    what = (task or re.sub(r"^\d{8}-\d{6}_?", "", parts[-1])).replace("_", " ")
    return f"{KIND_TITLES.get(kind, kind)} · {what} · {stamp}".strip(" ·")


def _summary(d: pathlib.Path) -> dict:
    rel = d.relative_to(RUNS).as_posix()
    m = re.match(r"(\d{8})-(\d{6})", d.name) or re.match(r"(\d{8})-(\d{6})", rel)
    item = {"id": rel, "kind": run_kind(rel), "time": f"{m.group(1)} {m.group(2)}" if m else None,
            "has_report": (d / "report.json").exists(), "has_summary": (d / "summary.json").exists(),
            "n_trials": len(list((d / "trials").glob("*.json"))) if (d / "trials").exists() else 0,
            "n_replays": len(list((d / "media").glob("*_poses.json.gz"))) if (d / "media").exists() else 0}
    try:
        if item["has_summary"]:
            s = _read(d / "summary.json")
            item.update(task=s.get("task"), success=s.get("success"), trials=s.get("trials"))
        if item["has_report"]:
            r = _read(d / "report.json")
            item.update(outcome=r.get("outcome"), task=item.get("task") or r.get("task") or r.get("template"))
            if item["kind"] == "failure_search":
                item["task"] = f"{r.get('library')} library" + (" · with a bottle nearby" if r.get("obstacle") else "")
            elif item["kind"] == "explore" and (r.get("task_kwargs") or {}).get("region"):
                item["task"] = f"{r['task_kwargs']['region']['template']}s (probe region)"
    except Exception:
        pass
    item["title"] = run_title(rel, item["kind"], item.get("task") if item["kind"] in ("task", "failure_search", "explore") else None)
    return item


@app.get("/api/runs")
def list_runs():
    items = [_summary(d) for d in _run_dirs()]
    items.sort(key=lambda x: x.get("time") or "", reverse=True)
    return items


@app.get("/api/run")
def get_run(id: str):
    d = _safe(id)
    if not d.is_dir():
        raise HTTPException(404)
    out = {"id": id, "kind": run_kind(id)}
    for name in ("report.json", "summary.json", "config.json"):
        if (d / name).exists():
            out[name.split(".")[0]] = _read(d / name)
    out["trials"] = sorted(p.stem for p in (d / "trials").glob("*.json")) if (d / "trials").exists() else []
    out["replays"] = sorted(p.name.replace("_poses.json.gz", "") for p in (d / "media").glob("*_poses.json.gz")) \
        if (d / "media").exists() else []
    out["title"] = run_title(id, out["kind"])
    # Outcome of each replayable episode, for the episode picker.
    out["episodes"] = []
    for name in out["replays"][:300]:
        try:
            t = _read(d / "trials" / f"{name}.json")
            out["episodes"].append({"name": name, "seed": t.get("seed"), "success": t.get("success"),
                                    "labels": [s["label"] for s in t.get("subtasks", []) if s.get("label")]})
        except (OSError, ValueError):
            out["episodes"].append({"name": name})
    return out


@app.get("/api/trial")
def get_trial(run: str, name: str):
    p = _safe(f"{run}/trials/{name}.json")
    if not p.exists():
        raise HTTPException(404)
    return _read(p)


@app.get("/api/replay")
def get_replay(run: str, name: str):
    """A pose log (every body's position and orientation at 30 fps + the static scene), already gzip-compressed."""
    p = _safe(f"{run}/media/{name}_poses.json.gz")
    if not p.exists():
        raise HTTPException(404)
    return Response(p.read_bytes(), media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/skills")
def skills():
    def cards(dirname=None):
        d = LIB / dirname if dirname else LIB
        return [c for c in (_read(p) for p in sorted(d.glob("*.json"))) if "name" in c] if d.exists() else []
    hist = {}
    for p in sorted((LIB / "history").glob("*.json")) if (LIB / "history").exists() else []:
        c = _read(p)
        hist.setdefault(c["name"], []).append(c)
    return {"active": cards(), "history": hist, "primitives": cards("primitives"), "reference": cards("reference"),
            "composites": cards("composites"),
            "performance": _read(LIB / "performance.json") if (LIB / "performance.json").exists() else {},
            "patches": _read(LIB / "patches.json") if (LIB / "patches.json").exists() else {}}


@app.get("/api/headlines")
def headlines():
    """The headline numbers, read from the latest report of each kind (labelled 'measured' in the app)."""
    runs = sorted(_run_dirs(), key=lambda d: d.as_posix())
    latest = {}
    for d in runs:
        if (d / "report.json").exists():
            latest.setdefault(run_kind(d.relative_to(RUNS).as_posix()), []).append(d)
    out = {}
    try:
        inv = [(d, _read(d / "report.json")) for d in latest.get("investigation", [])]
        inv = [(d, r) for d, r in inv if r.get("backend") == "claude_cli"]

        def pv(r):
            vs = [int(c["model_meta"]["prompt_version"]) for c in r.get("cases", []) if c.get("model_meta")]
            return max(vs) if vs else 0
        if inv:
            # Evidence with derived facts (diagnose prompt v3+); earlier runs had raw numbers only (listed per run).
            top = 3
            cur = [r["metrics"] for _, r in inv if pv(r) >= top]
            out["diagnosis"] = {"top1_correct": round(sum(x["top1_label_accuracy"] * x["diagnosed"] for x in cur)),
                                "diagnosed": sum(x["diagnosed"] for x in cur),
                                "confirmed": sum(x["confirmed"] for x in cur), "rejected": sum(x["rejected"] for x in cur),
                                "cases_with_confirmed": sum(x["cases_with_a_confirmed_cause"] for x in cur),
                                "prompt_version": top, "runs": len(cur),
                                "per_run": [{"run": d.relative_to(RUNS).as_posix(), "prompt_version": pv(r),
                                             **{k: r["metrics"][k] for k in ("diagnosed", "top1_label_accuracy",
                                                                            "confirmed", "rejected")}}
                                            for d, r in inv]}
        tr = [_read(d / "report.json") for d in latest.get("transfer", [])]
        tr = [r for r in tr if r.get("scratch")]
        if tr:
            r = tr[-1]
            out["transfer"] = {"template": r["template"], "verify_trials": r["verification"]["trials"],
                               "verify_success": r["verification"]["success"], "scratch_trials": r["scratch"]["trials_total"],
                               "scratch_search": r["scratch"]["trials_search"], "card": r["verification"]["card"]}
        repairs = []
        for d in latest.get("explore", []):
            r = _read(d / "report.json")
            if r.get("outcome") not in ("promoted", "patched", "routed"):
                continue
            name = {"promoted": ((r.get("promoted") or {}).get("card") or {}).get("name"),
                    "routed": f"route:{(r.get('routed') or {}).get('card')}", "patched": "patch"}[r["outcome"]]
            g = next((g for g in r.get("phases", {}).get("gates", []) if g.get("pass") and g.get("card") == name),
                     None)
            if g:
                repairs.append({"run": d.relative_to(RUNS).as_posix(), "outcome": r["outcome"],
                                "obj": g.get("obj") or r["phases"]["before"].get("cluster", {}).get("obj"),
                                "card": name if r["outcome"] != "patched" else (r.get("patched") or {}).get("values"),
                                "cluster": [g["fixes_cluster"]["success"], g["fixes_cluster"]["trials"]],
                                "held_out": [g["held_out"]["before"]["success"], g["held_out"]["success"],
                                             g["held_out"]["trials"]], "p": g["held_out"].get("p_value")})
        out["repairs"] = repairs
        fs = [(d, _read(d / "report.json")) for d in latest.get("failure_search", [])]
        if fs:
            out["failure_search"] = [{"run": d.relative_to(RUNS).as_posix(), "library": r.get("library"),
                                      "per100": {k: v["metrics"]["valid_failures_per_100"] for k, v in r["strategies"].items()}}
                                     for d, r in fs if r.get("strategies") and len(r["strategies"]) == 3]
        cs = [_read(d / "report.json") for d in latest.get("create_skill", [])]
        prom = [r for r in cs if r.get("outcome") == "promoted" and r.get("phases", {}).get("proposals", [{}])[0]
                .get("meta", {}).get("backend") == "claude_cli"
                and (LIB / f"{r['promoted']['card']}.json").exists()]        # only skills still in the library
        if prom:
            r = prom[-1]
            g = r["phases"]["gate"]["held_out"]
            out["created_skill"] = {"card": r["promoted"]["card"], "held_out": [g["before"], g["success"], g["trials"]],
                                    "trials": r["trials_simulated"],
                                    "rejected_proposals": sum(1 for x in cs if x.get("outcome") != "promoted")}
    except Exception as e:                           # headlines are best-effort; the panels show the details
        out["error"] = str(e)
    return out


def _when(run_id: str):
    """'20261031-013109' -> '2026-10-31 01:31' (from any component of the run id)."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", run_id)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}" if m else None


def _repair_name(r: dict) -> str:
    card = r["card"]
    if r["outcome"] == "patched" and isinstance(card, dict):
        return "patch " + ", ".join(f"{k} = {v}" for k, v in card.items())
    if r["outcome"] == "routed":
        return f"use {str(card).replace('route:', '')}"
    return f"new skill {card}"


@app.get("/api/progress")
def progress():
    """Time-ordered data for the Progress charts: failure-search rounds, gated repairs, transfer, diagnosis.
    Everything comes from saved reports; nothing here is computed by a model."""
    h = headlines()
    out = {"kpis": {}, "search": [], "repairs": [], "transfer": h.get("transfer"), "created": h.get("created_skill"),
           "diagnosis": None}
    # Failure-search rounds that are comparable: full search space, 160 trials per strategy, no obstacle.
    for d in sorted(RUNS.glob("*_failsearch")):
        try:
            r = _read(d / "report.json")
        except (OSError, ValueError):
            continue
        st = r.get("strategies", {})
        if r.get("budget_per_strategy") != 160 or r.get("obstacle") or "random" not in st or "model" not in st:
            continue
        out["search"].append({"run": d.name, "when": _when(d.name), "library": r.get("library"),
                              "per100": {k: v["metrics"]["valid_failures_per_100"] for k, v in st.items()},
                              "clusters": [c["key"] for c in r.get("clusters", [])][:4]})
    for rp in h.get("repairs", []):
        out["repairs"].append({**rp, "when": _when(rp["run"]), "name": _repair_name(rp)})
    out["repairs"].sort(key=lambda r: r["when"] or "")
    # Which repairs happened between two search rounds (annotations on the trend line).
    for i, s in enumerate(out["search"]):
        prev = out["search"][i - 1]["when"] if i else ""
        s["after_repairs"] = [r["name"] for r in out["repairs"] if prev < (r["when"] or "") <= s["when"]]
    dg = h.get("diagnosis")
    if dg:
        out["diagnosis"] = {k: dg[k] for k in ("top1_correct", "diagnosed", "confirmed", "rejected", "cases_with_confirmed")}
        out["diagnosis"]["per_run"] = [{**x, "when": _when(x["run"])} for x in dg["per_run"]]
    out["kpis"] = {
        "repairs": sum(r["outcome"] in ("promoted", "patched", "routed") for r in out["repairs"]),
        "search_first": out["search"][0]["per100"]["random"] if out["search"] else None,
        "search_last": out["search"][-1]["per100"]["random"] if out["search"] else None,
        "skills": len([p for p in LIB.glob("*.json") if p.stem not in ("performance", "patches")]),
    }
    return out


# ---------------------------------------------------------------- robot meshes (compiled, so offsets match)
@functools.lru_cache(maxsize=1)
def _meshes_payload() -> bytes:
    import sys
    sys.path.insert(0, str(ROOT))
    from kintsugi.sim.scene import load_panda_spec
    m = load_panda_spec().compile()
    meshes = {}
    for i in range(m.nmesh):
        va, vn = m.mesh_vertadr[i], m.mesh_vertnum[i]
        fa, fn = m.mesh_faceadr[i], m.mesh_facenum[i]
        v = np.asarray(m.mesh_vert[va:va + vn], np.float32)
        f = np.asarray(m.mesh_face[fa:fa + fn], np.uint32)
        meshes[m.mesh(i).name] = {"v": base64.b64encode(v.tobytes()).decode(),
                                  "f": base64.b64encode(f.tobytes()).decode()}
    return gzip.compress(json.dumps(meshes).encode())


@app.get("/api/meshes")
def meshes():
    return Response(_meshes_payload(), media_type="application/json", headers={"Content-Encoding": "gzip"})


# ---------------------------------------------------------------- built web app
# ----------------------------------------------------------------------------- studio (prompt -> run -> repair)
# The only part of the server that starts work: one job at a time, in its own process (scripts/studio_job.py).
import os
import shutil
import subprocess
import sys
import time

from fastapi import Body

_job = {"id": None, "proc": None}


def _backends():
    out = ["rule"]
    if shutil.which("claude"):
        out.append("claude_cli")
    if os.environ.get("META_API_KEY"):
        out.append("spark")
    return out


@app.get("/api/studio/info")
def studio_info():
    from kintsugi.runner import default_workers
    from kintsugi.studio import EXAMPLES
    b = _backends()
    return {"backends": b, "default_backend": os.environ.get("KINTSUGI_LLM") or b[-1], "examples": EXAMPLES,
            "max_workers": os.cpu_count() or 2, "default_workers": default_workers(),
            "running": _job["id"] if _job["proc"] is not None and _job["proc"].poll() is None else None}


@app.post("/api/studio/spec")
def studio_spec(body: dict = Body(...)):
    from kintsugi.studio import plan_spec
    return plan_spec(body["prompt"], body.get("backend"))


@app.post("/api/studio/program")
def studio_program(body: dict = Body(...)):
    from kintsugi.studio import plan_program
    return plan_program(body["prompt"], body["task"], body.get("backend"))


def _job_dir(job_id: str) -> pathlib.Path:
    if not re.fullmatch(r"\d{8}-\d{6}_studio", job_id):
        raise HTTPException(404)
    return RUNS / job_id


@app.post("/api/studio/jobs")
def studio_start(body: dict = Body(...)):
    if _job["proc"] is not None and _job["proc"].poll() is None:
        raise HTTPException(409, f"job {_job['id']} is still running")
    job_id = time.strftime("%Y%m%d-%H%M%S_studio")
    d = RUNS / job_id
    d.mkdir(parents=True)
    job = {"id": job_id, "prompt": body["prompt"], "task": body["task"], "program": body.get("program"),
           "program_source": body.get("program_source"), "n": max(1, min(int(body.get("n", 8)), 64)),
           "workers": max(1, min(int(body.get("workers", 4)), os.cpu_count() or 2)),
           "recovery": 2 if body.get("retries", True) else 0, "auto_fix": bool(body.get("auto_fix", True)), "hard": bool(body.get("hard", False)),
           "backend": body.get("backend"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "job.json").write_text(json.dumps(job, indent=1))
    env = {**os.environ, "KINTSUGI_LIVE_DIR": str(d / "live"), "PYTHONUNBUFFERED": "1"}
    if job["backend"]:
        env["KINTSUGI_LLM"] = job["backend"]
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True})
    log = open(d / "log.txt", "w")
    _job.update(id=job_id, proc=subprocess.Popen([sys.executable, str(ROOT / "scripts" / "studio_job.py"), str(d)],
                                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, **kw))
    return job


def _running(job_id):
    return _job["id"] == job_id and _job["proc"] is not None and _job["proc"].poll() is None


def _read_lines(p: pathlib.Path, offset: int, limit: int = 4_000_000):
    """Complete lines from a byte offset (a line still being written is left for the next call)."""
    if not p.exists():
        return [], offset
    with open(p, "rb") as f:
        f.seek(offset)
        data = f.read(limit)
    end = data.rfind(b"\n") + 1
    lines = []
    for raw in data[:end].splitlines():
        try:
            lines.append(json.loads(raw))
        except ValueError:
            pass
    return lines, offset + end


@app.get("/api/studio/jobs")
def studio_jobs():
    out = []
    for d in sorted(RUNS.glob("*_studio"), reverse=True)[:30]:
        try:
            job = _read(d / "job.json")
        except (OSError, ValueError):
            continue
        ev, _ = _read_lines(d / "events.jsonl", 0)
        res = next((e for e in reversed(ev) if e["type"] == "result"), None)
        out.append({"id": d.name, "prompt": job["prompt"], "n": job["n"], "started": job.get("started"),
                    "running": _running(d.name), "result": res,
                    "stopped": any(e["type"] == "stopped" for e in ev),
                    "error": next((e["message"] for e in ev if e["type"] == "error"), None)})
    return out


@app.get("/api/studio/jobs/{job_id}")
def studio_job(job_id: str, offset: int = 0):
    d = _job_dir(job_id)
    if not (d / "job.json").exists():
        raise HTTPException(404)
    events, new = _read_lines(d / "events.jsonl", offset)
    out = {"events": events, "offset": new, "running": _running(job_id)}
    if offset == 0:
        out["job"] = _read(d / "job.json")
    if not out["running"] and (d / "log.txt").exists():
        done = any(e["type"] in ("result", "error", "stopped") for e in _read_lines(d / "events.jsonl", 0)[0])
        if not done:                                   # crashed (or still starting) before writing an end event
            out["log_tail"] = (d / "log.txt").read_text(errors="replace")[-3000:]
    return out


@app.post("/api/studio/jobs/{job_id}/stop")
def studio_stop(job_id: str):
    d = _job_dir(job_id)
    if not _running(job_id):
        return {"stopped": False}
    pid = _job["proc"].pid
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        os.killpg(pid, signal.SIGTERM)
    _job["proc"].wait(timeout=10)
    with open(d / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "stopped", "t": None}) + "\n")
    return {"stopped": True}


@app.get("/api/studio/live/{job_id}")
def studio_live(job_id: str):
    """Each worker's current episode (pointer files written by kintsugi/sim/live.py)."""
    live = _job_dir(job_id) / "live"
    out, now = [], time.time()
    for p in live.glob("w*.json"):
        try:
            if _running(job_id) and now - p.stat().st_mtime > 20:     # a worker of an earlier, finished pool
                continue
            out.append(_read(p))
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda w: w["worker"])


@app.get("/api/studio/live/{job_id}/{episode}")
def studio_live_episode(job_id: str, episode: str, offset: int = 0):
    if not re.fullmatch(r"\d+_\d+", episode):
        raise HTTPException(404)
    p = _job_dir(job_id) / "live" / f"{episode}.jsonl"
    if not p.exists():
        raise HTTPException(404, "episode no longer kept")
    lines, new = _read_lines(p, offset)
    return {"lines": lines, "offset": new}


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/logo.svg")
    def logo():
        return FileResponse(WEB_DIST / "logo.svg", media_type="image/svg+xml")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIST / "index.html")
