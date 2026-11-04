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
LIB = ROOT / "skill_library"            # base library (shipped); local discoveries live in the user library
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Kintsugi")
app.add_middleware(GZipMiddleware, minimum_size=2000)

KINDS = [("_explore_", "explore"), ("_explore", "explore"), ("_investigate_", "investigation"), ("_failsearch", "failure_search"), ("_investigate", "investigation"),
         ("_create_", "create_skill"), ("_revise_", "create_skill"), ("_transfer_", "transfer"), ("_verify_program", "verify_program"),
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
                rg = r["task_kwargs"]["region"]
                item["task"] = (", ".join(t.replace("_", " ") for t in rg["templates"][:3])
                                + (f" +{len(rg['templates']) - 3}" if len(rg["templates"]) > 3 else "")
                                if rg.get("templates") else f"{rg['template']}s (probe region)")
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


@app.get("/api/image")
def get_image(path: str):
    """An evidence image (contact sheet) inside runs/."""
    p = _safe(path)
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png" if p.suffix.lower() == ".png" else "image/jpeg")


@app.get("/api/replay")
def get_replay(run: str, name: str):
    """A pose log (every body's position and orientation at 30 fps + the static scene), already gzip-compressed."""
    p = _safe(f"{run}/media/{name}_poses.json.gz")
    if not p.exists():
        raise HTTPException(404)
    return Response(p.read_bytes(), media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/skills")
def skills():
    """Base library (skill_library/, shipped) merged with the user library (what this checkout discovered); every
    card carries "origin": "base" | "user", evidence and patches are the merged view the router uses."""
    from kintsugi.skills import library as L

    def cards(dirname=None, merged=True):
        out = {}
        for origin, root in (("base", LIB), ("user", L.USER_DIR)) if merged else (("base", LIB),):
            d = root / dirname if dirname else root
            for p in sorted(d.glob("*.json")) if d.exists() else []:
                c = _read(p)
                if "name" in c:
                    out[c["name"]] = {**c, "origin": origin}
        return list(out.values())
    hist = {}
    for root in (LIB, L.USER_DIR):
        for p in sorted((root / "history").glob("*.json")) if (root / "history").exists() else []:
            c = _read(p)
            if not any(h.get("version") == c.get("version") for h in hist.get(c["name"], [])):
                hist.setdefault(c["name"], []).append(c)
    for v in hist.values():
        v.sort(key=lambda c: c.get("version", 0))
    return {"active": cards(), "history": hist, "primitives": cards("primitives", False),
            "reference": cards("reference", False), "composites": cards("composites"),
            "performance": L.load_merged("performance.json"), "patches": L.load_merged("patches.json"),
            "user_library": L.rel(L.USER_DIR)}


def _active_cards() -> dict:
    from kintsugi.skills.library import load_cards
    return load_cards()


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
        inv = [(d, r) for d, r in inv if r.get("backend") == "spark"]

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
                .get("meta", {}).get("backend") == "spark"
                and r["promoted"]["card"] in _active_cards()]                 # only skills still in the library
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
        "skills": len(_active_cards()),
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


@functools.lru_cache(maxsize=64)
def _asset_mesh_payload(name: str) -> bytes | None:
    """Visual meshes of one imported object (objects/ or the scanned catalogue), compiled like the scene compiles
    them so the exported geom offsets match: non-indexed triangles with a colour per corner sampled from the texture."""
    import sys
    sys.path.insert(0, str(ROOT))
    from kintsugi.sim.assets import _spec, assets
    a = assets().get(name)
    if a is None:
        return None
    m = _spec(a["path"], a["meta"]).compile()
    out = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != 7 or m.geom_group[g] > 2:            # 7 = mjGEOM_MESH; collision copies are group 3
            continue
        i = int(m.geom_dataid[g])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]], np.float32)
        f = np.asarray(m.mesh_face[m.mesh_faceadr[i]:m.mesh_faceadr[i] + m.mesh_facenum[i]])
        col = np.tile(np.round(255 * m.geom_rgba[g][:3]).astype(np.uint8), (f.size, 1))
        mat = int(m.geom_matid[g])
        tex = int(m.mat_texid[mat][1]) if mat >= 0 else -1           # role 1 = RGB
        if tex >= 0 and m.mesh_texcoordnum[i] > 0:
            tc = np.asarray(m.mesh_texcoord[m.mesh_texcoordadr[i]:m.mesh_texcoordadr[i] + m.mesh_texcoordnum[i]])
            ft = np.asarray(m.mesh_facetexcoord[m.mesh_faceadr[i]:m.mesh_faceadr[i] + m.mesh_facenum[i]]).reshape(-1)
            w, h, c = int(m.tex_width[tex]), int(m.tex_height[tex]), int(m.tex_nchannel[tex])
            img = np.asarray(m.tex_data[m.tex_adr[tex]:m.tex_adr[tex] + w * h * c]).reshape(h, w, c)[:, :, :3]
            uv = tc[ft] % 1.0
            col = img[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
        out[m.mesh(i).name] = {"v": base64.b64encode(v[f.reshape(-1)].tobytes()).decode(),
                               "c": base64.b64encode(np.ascontiguousarray(col, np.uint8).tobytes()).decode()}
    return gzip.compress(json.dumps(out).encode())


@app.get("/api/asset_mesh/{name}")
def asset_mesh(name: str):
    data = _asset_mesh_payload(name)
    if data is None:
        raise HTTPException(404, f"no imported object '{name}'")
    return Response(data, media_type="application/json", headers={"Content-Encoding": "gzip"})


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
    from kintsugi import settings
    out = ["rule"]
    if settings.spark_ready():
        out.append("spark")
    return out


@app.get("/api/studio/info")
def studio_info():
    from kintsugi.runner import default_workers
    from kintsugi.studio import EXAMPLES
    from kintsugi import settings
    from kintsugi.perception.sam import availability
    b = _backends()
    pref = settings.get("default_backend") or os.environ.get("KINTSUGI_LLM")
    sam_ok, sam_why = availability()
    return {"backends": b, "default_backend": pref if pref in b else b[-1], "examples": EXAMPLES,
            "masks": {"sim": {"available": True}, "sam": {"available": sam_ok, "reason": sam_why}},
            "default_masks": settings.get("default_masks") if settings.get("default_masks") == "sam" and sam_ok else "sim",
            "max_workers": os.cpu_count() or 2, "default_workers": default_workers(),
            "running": _job["id"] if _job["proc"] is not None and _job["proc"].poll() is None else None}


@app.post("/api/studio/spec")
def studio_spec(body: dict = Body(...)):
    from kintsugi.studio import plan_spec
    return plan_spec(body["prompt"], body.get("backend"))


@app.post("/api/studio/photo_spec")
def studio_photo_spec(body: dict = Body(...)):
    """A photo of a real table (data URL) + an optional request -> task spec with the photo's layout + preview."""
    from kintsugi.studio import plan_photo
    m = re.fullmatch(r"data:image/(png|jpe?g|webp);base64,(.+)", body.get("image") or "", re.S)
    if not m:
        raise HTTPException(400, "image must be a PNG, JPEG or WebP data URL")
    raw = base64.b64decode(m.group(2))
    if len(raw) > 15_000_000:
        raise HTTPException(413, "photo larger than 15 MB")
    d = RUNS / "twins" / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"upload.{'jpg' if m.group(1).startswith('jp') else m.group(1)}"
    src.write_bytes(raw)
    out = plan_photo(src, body.get("prompt") or "", body.get("backend"))
    tw = out.get("twin") or {}
    for k in ("photo", "overlay"):                   # paths for /api/image (relative to runs/)
        if tw.get(k):
            tw[k] = str(pathlib.Path(tw[k]).relative_to("runs")).replace("\\", "/")
    return out


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
    job = {"id": job_id, "prompt": body["prompt"], "task": body["task"], "program": body.get("program"),
           "program_source": body.get("program_source"), "n": max(1, min(int(body.get("n", 8)), 64)),
           "workers": max(1, min(int(body.get("workers", 4)), os.cpu_count() or 2)),
           "recovery": 2 if body.get("retries", True) else 0, "auto_fix": bool(body.get("auto_fix", True)), "hard": bool(body.get("hard", False)),
           "backend": body.get("backend"), "masks": body.get("masks") or "sim",
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if job["masks"] not in ("sim", "sam"):
        raise HTTPException(400, "masks must be sim or sam")
    if job["masks"] == "sam":
        from kintsugi.perception.sam import availability
        ok, why = availability()
        if not ok:
            raise HTTPException(400, why)
    d.mkdir(parents=True)
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


# ----------------------------------------------------------------------------- settings and assets
# Local only (the server listens on localhost): model access keys are saved on this machine (config/local.json,
# git-ignored) and never sent back to the browser; asset downloads run one at a time in a background thread.
import threading

_dl = {"which": None, "done": 0, "total": 0, "label": "", "error": None, "finished": None, "running": False}


@app.get("/api/model_calls")
def model_calls(since: float = 0.0):
    """Meta model calls that started at or after `since` (server time), live: the app shows them as they happen."""
    from kintsugi.agents.calllog import recent
    return {"now": time.time(), "calls": recent(since)}


@app.get("/api/settings")
def get_settings():
    from kintsugi import settings
    from kintsugi.perception.sam import availability, boxes_available
    ok, why = availability()
    pok, pwhy = boxes_available()                    # photos: SAM 3.1 boxes + Muse Image (Meta key, or SAM's own)
    return {**settings.public(),
            "sam": {"available": ok, "reason": why}, "photo": {"available": pok, "reason": pwhy},
            "config_file": "config/local.json"}


@app.post("/api/settings")
def set_settings(body: dict = Body(...)):
    from kintsugi import settings
    try:
        settings.save(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return get_settings()


@app.post("/api/settings/test")
def test_backend(body: dict = Body(...)):
    """One tiny call to check access; the answer says only whether it worked (never the key)."""
    from kintsugi.agents import llm
    from kintsugi import settings
    backend = body.get("backend")
    t0 = time.time()
    try:
        if backend == "spark":
            out = llm._call_openai_compatible("Reply with the single word OK.", "Say OK.",
                                              settings.get("meta_model") or llm.DEFAULT_MODEL["spark"], timeout=30)
        else:
            raise HTTPException(400, "backend must be spark")
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        for secret in (settings.get("meta_api_key"), settings.get("sam_api_key")):
            if secret:
                msg = msg.replace(secret, "…")
        return {"ok": False, "message": f"{type(e).__name__}: {msg[:300]}", "seconds": round(time.time() - t0, 1)}
    return {"ok": True, "message": f"answered: {str(out).strip()[:40]}", "seconds": round(time.time() - t0, 1)}


@app.get("/api/assets")
def assets_status():
    from kintsugi.assets_fetch import status
    return {**status(), "download": {k: v for k, v in _dl.items()}}


def _download(which):
    from kintsugi import assets_fetch as F

    def prog(i, n, label):
        _dl.update(done=i, total=n, label=label)
    try:
        if which == "robot":
            F.fetch_robot(prog)
        else:
            r = F.fetch_scanned(prog)
            if r["failed"]:
                _dl["error"] = f"{len(r['failed'])} object(s) failed: " + "; ".join(
                    f"{k} ({v[:80]})" for k, v in list(r["failed"].items())[:3])
            from kintsugi.sim.templates import refresh_assets
            refresh_assets()                     # usable at once: task generation and scenes see the new objects
    except Exception as e:
        _dl["error"] = f"{type(e).__name__}: {e}"
    finally:
        _meshes_payload.cache_clear()
        _asset_mesh_payload.cache_clear()            # a missing object was cached as None
        _dl.update(running=False, finished=time.strftime("%H:%M:%S"))


@app.post("/api/assets/download")
def assets_download(body: dict = Body(...)):
    which = body.get("which")
    if which not in ("robot", "scanned"):
        raise HTTPException(400, "which must be robot or scanned")
    if _dl["running"]:
        raise HTTPException(409, f"already downloading {_dl['which']}")
    _dl.update(which=which, done=0, total=0, label="", error=None, finished=None, running=True)
    threading.Thread(target=_download, args=(which,), daemon=True).start()
    return {k: v for k, v in _dl.items()}
