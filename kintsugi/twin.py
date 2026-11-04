"""Digital twin from a photo: a picture of a real table (+ an optional request) -> an ordinary task spec whose objects
stand where they stand in the photo. Only the start of the Studio uses it; everything after (plan, run, repair) sees a
normal task with `layout_fixed`, so the existing pipeline is unchanged.

    photo --Muse Spark (vision)--> objects as templates, a label + rough box each, real widths, camera angle, the task
          --SAM 3.1--------------> exact pixel boxes, one request per label
          --geometry-------------> table positions (metres) -> robot frame, fitted into the reachable area

Geometry (no depth, no calibration): each object's real width (Spark) over its pixel width (SAM) gives the metres per
pixel at its image row; a line fitted through these rows models perspective. Sideways position = pixels from the
image centre x local scale; depth = the scale integrated from the bottom of the photo up, divided by sin(camera
elevation) for the foreshortening of the table plane. The robot stands where the photographer stood. The layout is
centred in the reachable area, shrunk if the real table is larger than the reach, and overlaps are pushed apart; the
report says when that happened. Positions are an estimate (reported as such); `evaluate` measures them on rendered
scenes whose true layout is known.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

from .runner import ROOT

MAX_SIDE = 1280
TWIN_DIR = ROOT / "runs" / "twins"


def prepare_image(src, out_dir: pathlib.Path) -> tuple[pathlib.Path, int, int]:
    """EXIF-upright RGB JPEG, longest side <= MAX_SIDE (model and segmenter see the same pixels)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    k = MAX_SIDE / max(im.size)
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "photo.jpg"
    im.save(p, quality=92)
    return p, im.width, im.height


def _assign(objs, matches):
    """Greedy: closest (Spark box centre, SAM box centre) pairs first. Returns {object index: match}."""
    c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    pairs = sorted((math.dist(c(o["box"]), c(m["box"])), i, j) for i, o in enumerate(objs) for j, m in enumerate(matches))
    used_o, used_m, out = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_o and j not in used_m:
            out[i] = matches[j]
            used_o.add(i)
            used_m.add(j)
    return out


def locate(objects: list[dict], width: int, height: int, elevation_deg: float) -> dict:
    """Pixel boxes + real widths -> table-plane positions in metres, photographer at the origin:
    {id: (X right, Y away from the camera)}, plus the fit used."""
    th = math.radians(min(90.0, max(25.0, elevation_deg or 50.0)))
    rows, scales = [], []
    pts = {}
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        wpx = max(4.0, x2 - x1)
        vc = y2 - 0.5 * wpx * math.sin(th)          # ground centre: half a (round) footprint above the box bottom
        vc = min(max(vc, y1), y2)
        pts[o["id"]] = ((x1 + x2) / 2, vc)
        if o.get("width_m"):
            rows.append(vc)
            scales.append(float(o["width_m"]) / wpx)
    if not scales:
        raise ValueError("no object has a width estimate")
    rows, scales = np.array(rows), np.array(scales)
    if len(rows) >= 3 and np.ptp(rows) > 0.08 * height:
        # Theil-Sen slope (robust to one bad width); farther up the photo = more metres per pixel, so slope <= 0
        sl = [(scales[j] - scales[i]) / (rows[j] - rows[i]) for i in range(len(rows)) for j in range(i + 1, len(rows))
              if abs(rows[j] - rows[i]) > 1e-6]
        b = min(0.0, float(np.median(sl)))
    else:
        b = 0.0
    a = float(np.median(scales - b * rows))
    s = lambda v: max(1e-5, a + b * v)
    out = {}
    for oid, (u, v) in pts.items():
        X = (u - width / 2) * s(v)
        Y = (a * (height - v) + b * (height ** 2 - v ** 2) / 2) / math.sin(th)
        out[oid] = (X, Y)
    return {"positions": out, "fit": {"a": a, "b": b, "elevation_deg": math.degrees(th)}}


def orientations(task: dict, boxes: dict, elevation_deg: float) -> dict:
    """Yaw for rectangular objects from the shape of their photo box: a tray whose long side runs left-right in the
    photo gets its long side along the robot's y axis. Template axes: width / length along x (away from the robot)."""
    from .sim.templates import resolve_params
    th = math.radians(min(90.0, max(25.0, elevation_deg or 50.0)))
    yaw = {}
    for o in task["objects"]:
        if o["id"] not in boxes or o["template"] not in ("tray", "bin", "zone", "bar"):
            continue
        p = resolve_params(o["template"], o.get("params") or {})
        long_x = p.get("length", p.get("width", 0)) >= p.get("depth", p.get("width", 0))
        x1, y1, x2, y2 = boxes[o["id"]]
        wide_in_photo = (x2 - x1) >= (y2 - y1) / math.sin(th)       # undo the foreshortening of the table plane
        if wide_in_photo == long_x:                                  # long side left-right: turn by 90 degrees
            yaw[o["id"]] = round(math.pi / 2, 4)
    return yaw


def fit_to_reach(task: dict, plane: dict, yaw: dict | None = None) -> tuple[dict, list[str]]:
    """Table-plane positions -> robot frame {id: (x, y, yaw)} inside the reachable area, without overlaps."""
    from .sim.scene import REACH, ObjectSpec, _footprint
    from .sim.templates import ARTICULATED
    notes = []
    specs = {o["id"]: ObjectSpec(**o) for o in task["objects"]}
    ids = [i for i in plane if i in specs and specs[i].template not in ARTICULATED]   # cabinets: the placer's job
    if not ids:
        return {}, notes
    rad = {i: _footprint(specs[i]) for i in ids}
    P = np.array([(plane[i][1], -plane[i][0]) for i in ids])     # robot x = away from the camera, y = to the left
    P -= P.mean(axis=0)
    cx, cy = (REACH["x"][0] + REACH["x"][1]) / 2, 0.0
    half = np.array([(REACH["x"][1] - REACH["x"][0]) / 2, REACH["y"][1]])
    ext = np.abs(P).max(axis=0)                     # the placer bounds centres (objects may overhang the reach box)
    k = float(min(1.0, *(half / np.maximum(ext, 1e-6))))
    if k < 0.98:
        notes.append(f"the real layout is larger than the robot's reach: distances scaled by {k:.2f}")
    P = P * k + np.array([cx, cy])
    R = np.array([rad[i] for i in ids])

    def clamp(p, r):
        x = min(max(p[0], REACH["x"][0]), REACH["x"][1])
        y = min(max(p[1], REACH["y"][0]), REACH["y"][1])
        d = math.hypot(x, y)
        lo, hi = REACH["r"][0] + r * 0.3, REACH["r"][1] - r * 0.3
        if d < lo or d > hi:
            f = min(max(d, lo), hi) / max(d, 1e-6)
            x, y = x * f, y * f
        return np.array([x, y])

    moved = 0.0
    start = P.copy()
    for _ in range(300):
        changed = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dv = P[j] - P[i]
                dist = float(np.hypot(*dv))
                need = R[i] + R[j] + 0.02
                if dist < need:
                    dirv = dv / dist if dist > 1e-6 else np.array([0.0, 1.0])
                    push = (need - dist) / 2 + 1e-4
                    P[i] -= dirv * push
                    P[j] += dirv * push
                    changed = True
        P = np.array([clamp(p, r) for p, r in zip(P, R)])
        if not changed:
            break
    moved = float(np.max(np.hypot(*(P - start).T))) if len(ids) else 0.0
    if moved > 0.015:
        notes.append(f"objects nudged apart to fit the simulated templates (largest move {100 * moved:.0f} cm)")
    yaw = yaw or {}
    return {i: (round(float(p[0]), 4), round(float(p[1]), 4), float(yaw.get(i, 0.0))) for i, p in zip(ids, P)}, notes


def build(photo, prompt: str = "", backend: str | None = None, out_dir: pathlib.Path | None = None,
          use_sam: bool = True, texture: bool = True) -> dict:
    """The whole path. Returns the Studio's plan shape (status, output, meta, task) plus "twin" (detections, notes,
    image paths) and writes twin.json into out_dir."""
    from .agents.jobs.task_spec import to_task
    from .agents.llm import ModelClient, ModelRejected
    from .perception import sam as sam_mod
    out_dir = pathlib.Path(out_dir or TWIN_DIR / time.strftime("%Y%m%d-%H%M%S")).resolve()
    img, W, H = prepare_image(photo, out_dir)
    client = ModelClient(backend)
    try:
        res = client.complete("scene_photo", {"prompt": prompt or "", "width": W, "height": H}, images=[img])
    except ModelRejected as e:
        return {"status": "rejected", "reasons": e.reasons}
    out = res.output
    report = {"status": out["status"], "output": out, "meta": res.meta,
              "twin": {"photo": str(img.relative_to(ROOT)), "size": [W, H], "sam_calls": 0, "notes": [], "detections": []}}
    if out["status"] != "ok":
        _save(out_dir, report)
        return report
    objs = out["task"]["objects"]
    sam_calls, sam_t = 0, 0.0
    det = {o["id"]: {"id": o["id"], "label": o["label"], "spark_box": o["box"], "box": o["box"], "source": "spark"}
           for o in objs}
    if use_sam:
        ok, why = sam_mod.boxes_available()
        if not ok:
            report["twin"]["notes"].append(f"SAM not used: {why}")
        else:
            groups = {}
            for o in objs:
                groups.setdefault(o["label"].strip().lower(), []).append(o)
            for label, group in groups.items():
                try:
                    r = sam_mod.segment(img, label)
                except Exception as e:                 # SAM down: keep Muse Spark's boxes, say so
                    report["twin"]["notes"].append(f"SAM failed for '{label}': {e}")
                    continue
                sam_calls += 1
                sam_t += r["latency_s"]
                for i, m in _assign(group, r["matches"]).items():
                    det[group[i]["id"]].update(box=m["box"], source="sam", conf=m["conf"])
    report["twin"].update(sam_calls=sam_calls, sam_latency_s=round(sam_t, 2))
    located = [{"id": o["id"], "box": det[o["id"]]["box"], "width_m": o.get("width_m")} for o in objs]
    task = to_task(out, {"prompt": out["task"].get("prompt") or prompt})
    try:
        geo = locate(located, W, H, out.get("camera_elevation_deg", 50))
        yaw = orientations(task, {o["id"]: o["box"] for o in located}, out.get("camera_elevation_deg", 50))
        fixed, notes = fit_to_reach(task, geo["positions"], yaw)
        report["twin"]["notes"] += notes
        report["twin"]["fit"] = geo["fit"]
        report["twin"]["plane"] = {k: [round(v[0], 3), round(v[1], 3)] for k, v in geo["positions"].items()}
    except ValueError as e:
        fixed = {}
        report["twin"]["notes"].append(f"positions not estimated ({e}): the placer lays the scene out")
    if fixed:
        task["layout_fixed"] = fixed
        from .sim.scene import build_scene
        try:
            build_scene(task, seed=0)
        except Exception as e:                       # should not happen after fit_to_reach; never block the user
            task.pop("layout_fixed")
            report["twin"]["notes"].append(f"photo layout could not be built ({e}); the placer lays the scene out")
    if texture:
        try:
            tex = table_texture(img, out_dir)
            task["table_texture"] = tex["texture"]          # appearance only (scene build + browser view)
            report["twin"]["table"] = tex
        except Exception as e:                           # never block the twin on the texture
            report["twin"]["notes"].append(f"table surface not generated ({str(e)[:160]}); the default table is used")
    report["task"] = task
    report["twin"]["detections"] = list(det.values())
    report["twin"]["overlay"] = str(overlay(img, report["twin"]["detections"], out_dir / "overlay.jpg").relative_to(ROOT))
    _save(out_dir, report)
    return report


IMAGE_MODEL = "muse-image-1.0"         # Meta Model API, /v1/images/edits, $0.01 per generated image (docs, 2026-11-03)
EMPTY_TABLE_PROMPT = ("Remove every object from this table: keep only the empty tabletop, with exactly the same "
                      "surface material, colour, grain, lighting and camera view. No objects, no shadows of objects.")


def table_texture(photo, out_dir: pathlib.Path, timeout: float = 180) -> dict:
    """The photo's empty tabletop, made by Muse Image (an edit of the photo), cropped to the middle of the table and
    saved as a square texture. Appearance only: the simulated table keeps its geometry and friction."""
    import base64
    import io
    import urllib.error
    import urllib.request
    from PIL import Image
    from . import settings
    base, key = settings.get("meta_api_base"), settings.get("meta_api_key")
    if not (base and key):
        raise RuntimeError("Meta API access is not set up")
    data = base64.b64encode(pathlib.Path(photo).read_bytes()).decode()
    body = {"model": IMAGE_MODEL, "prompt": EMPTY_TABLE_PROMPT, "n": 1,
            "images": [{"image_url": f"data:image/jpeg;base64,{data}"}]}
    req = urllib.request.Request(base.rstrip("/") + "/images/edits", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    from .agents.calllog import live_call
    try:
        with live_call(IMAGE_MODEL, "empty_table"), urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:                # the API's message, never the request (holds the key)
        raise RuntimeError(f"Muse Image HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    item = res["data"][0]
    raw = base64.b64decode(item["b64_json"]) if item.get("b64_json") else urllib.request.urlopen(item["url"], timeout=60).read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.save(out_dir / "empty_table.jpg", quality=90)
    w, h = im.size                                   # middle of the table: away from its edges and the background
    s = int(0.62 * min(w, h))
    tex = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((1024, 1024), Image.LANCZOS)
    # texture u (left -> right) runs along the robot's x (away from it) = the photo's up: turn the photo clockwise
    tex = tex.transpose(Image.Transpose.ROTATE_270)
    # 2 x 2 mirrored tile: repeats without seams (MuJoCo repeats textures plainly)
    m = Image.new("RGB", (2048, 2048))
    m.paste(tex, (0, 0))
    m.paste(tex.transpose(Image.Transpose.FLIP_LEFT_RIGHT), (1024, 0))
    m.paste(tex.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0, 1024))
    m.paste(tex.transpose(Image.Transpose.ROTATE_180), (1024, 1024))
    tex = m
    tex.save(out_dir / "table_texture.png")
    return {"empty": (out_dir / "empty_table.jpg").relative_to(ROOT).as_posix(),
            "texture": (out_dir / "table_texture.png").relative_to(ROOT).as_posix(), "model": IMAGE_MODEL,
            "latency_s": round(time.time() - t0, 1), "usage": res.get("usage")}


def overlay(img, detections, path) -> pathlib.Path:
    """The photo with each object's box and id (SAM boxes solid, Muse Spark-only boxes dashed)."""
    from PIL import Image, ImageDraw
    im = Image.open(img).convert("RGB")
    dr = ImageDraw.Draw(im)
    lw = max(2, im.width // 320)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        col = (217, 172, 82) if d["source"] == "sam" else (120, 170, 255)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        tag = d["id"] + (f" {d['conf']:.2f}" if d.get("conf") is not None else " (Spark box)")
        tw = dr.textlength(tag)
        dr.rectangle([x1, max(0, y1 - 18), x1 + tw + 8, max(0, y1 - 18) + 18], fill=col)
        dr.text((x1 + 4, max(0, y1 - 18) + 3), tag, fill=(20, 20, 20))
    im.save(path, quality=90)
    return path


def _save(out_dir, report):
    (out_dir / "twin.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
