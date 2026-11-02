"""Visual evidence for failures the text evidence could not explain (escalation only).

When a failed step stays unexplained (the investigator confirmed no cause, or a new skill failed its gates), the model
gets what the robot's cameras saw, in one compact image: a contact sheet of at most five moments, two views each.

    replay   the failing episode is replayed exactly (same task, seed, layout, physics, skills: deterministic) with a
             FrameTap in the recorder slot; it renders locally (no model, no SAM call) every 0.25 s around the step
    select   the moments are chosen from OBSERVABLE signals only: the robot's own skill timeline, the finger opening
             (proprioception), the object's motion measured from the camera's depth + segmentation, and the first arm
             contact (already part of the text evidence); never from true object state
    sheet    per moment: the front camera with the perceived objects outlined and labelled by id and the hand's path
             projected so far, plus a close-up beside the hand; a caption with time and what happened
    text     the same moments as a short list, so the model can refer to them

Masks are the episode's own perception masks (simulator segmentation now; with SAM, the masks SAM already returned):
building evidence adds no SAM calls. One JPEG per case (~1600 x 520) keeps model cost to one image per call.
"""
from __future__ import annotations

import math
import pathlib

import mujoco
import numpy as np

SAMPLE_EVERY = 0.25          # s between rendered samples inside the window
PATH_EVERY = 0.05            # s between hand-path points
MAX_MOMENTS = 5
TILE = (320, 240)
COLORS = [(230, 80, 60), (60, 140, 230), (90, 190, 90), (230, 180, 40), (170, 90, 210), (40, 190, 190),
          (240, 120, 180), (150, 150, 150)]


class FrameTap:
    """Recorder stand-in that renders samples inside a time window and keeps the hand path (wraps the real one)."""

    def __init__(self, scene, monitor, obj, window, inner=None):
        from .perception.camera import Camera
        from .perception.perceive import _geom_to_object
        self.scene, self.monitor, self.obj, self.inner = scene, monitor, obj, inner
        self.t_lo, self.t_hi = window
        self.cam = Camera(scene.model, scene.data, "front", *TILE)
        self.close = mujoco.Renderer(scene.model, TILE[1], TILE[0])
        self.lut = _geom_to_object(scene.model, list(scene.objects))
        self.samples, self.path = [], []
        self._next, self._next_path = self.t_lo, 0.0
        self.events = getattr(inner, "events", [])

    def event(self, t, kind, **kw):              # the episode reports skill ends through the recorder
        if self.inner is not None and hasattr(self.inner, "event"):
            self.inner.event(t, kind, **kw)

    def on_step(self, robot):
        if self.inner is not None:
            self.inner.on_step(robot)
        t = robot.t
        if t >= self._next_path:
            self.path.append((t, robot.tcp_pose()[0].copy()))
            self._next_path = t + PATH_EVERY
        if self.t_lo <= t <= self.t_hi and t >= self._next:
            self._next = t + SAMPLE_EVERY
            self.samples.append(self._sample(robot))

    def _sample(self, robot):
        seg = self.cam.segmentation()
        depth = self.cam.depth()
        objmask = np.isin(seg, [g for g, o in self.lut.items() if o == self.obj])
        centroid = None
        if objmask.sum() > 15:                        # the object's position as the camera sees it (depth + mask)
            centroid = self.cam.points(depth, objmask).mean(axis=0)
        tcp = robot.tcp_pose()[0]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        look = tcp if centroid is None else 0.5 * (tcp + centroid)
        cam.lookat[:] = look
        cam.distance = 0.32
        cam.azimuth = math.degrees(math.atan2(tcp[1], tcp[0])) + 90.0     # from the side of the hand
        cam.elevation = -22.0
        self.close.update_scene(self.scene.data, camera=cam)
        return {"t": round(robot.t, 2), "rgb": self.cam.rgb(), "seg": seg, "close": self.close.render().copy(),
                "centroid": centroid, "finger": robot.finger_width(), "contacts": len(self.monitor.collisions)}

    def close_all(self):
        for r in (self.close,):
            r.close()


def select_moments(samples, trace_step, label_hint=None) -> list[dict]:
    """At most MAX_MOMENTS samples, each with why it was picked (observable signals only)."""
    if not samples:
        return []
    picks = []

    def nearest(t):
        return min(range(len(samples)), key=lambda i: abs(samples[i]["t"] - t))
    picks.append((nearest(trace_step["t0"]), f"{trace_step['skill']} starts"))
    fw = [s["finger"] for s in samples]
    for i in range(1, len(samples)):
        if fw[i - 1] - fw[i] > 0.005 and fw[i] < 0.07:
            picks.append((i, f"fingers closing ({1000 * fw[i]:.0f} mm apart)"))
            break
    for i in range(1, len(samples)):
        if samples[i]["contacts"] > samples[i - 1]["contacts"]:
            picks.append((i, "arm touches something"))
            break
    moves = [0.0] + [float(np.linalg.norm(samples[i]["centroid"] - samples[i - 1]["centroid"]))
                     if samples[i]["centroid"] is not None and samples[i - 1]["centroid"] is not None else 0.0
                     for i in range(1, len(samples))]
    k = int(np.argmax(moves))
    if moves[k] > 0.01:
        picks.append((k, f"object moves most ({100 * moves[k]:.1f} cm in {SAMPLE_EVERY} s, seen by the camera)"))
    for i in range(1, len(samples)):
        if fw[i - 1] > 0.004 and fw[i] < 0.003:
            picks.append((i, "fingers close on nothing (grip lost)"))
            break
    picks.append((nearest(trace_step["t1"]), f"{trace_step['skill']} ends ({trace_step.get('check') or 'done'})"))
    picks.append((len(samples) - 1, "after the step"))
    out, used = [], []
    for i, why in sorted(picks, key=lambda p: p[0]):
        if any(abs(samples[i]["t"] - samples[j]["t"]) < 0.3 for j in used):
            continue
        used.append(i)
        out.append({"i": i, "t": samples[i]["t"], "why": why})
    if len(out) > MAX_MOMENTS:                          # keep start, end and the most informative middles
        keep = [out[0], out[-1]] + [m for m in out[1:-1] if not m["why"].startswith("after")][:MAX_MOMENTS - 2]
        out = sorted(keep, key=lambda m: m["t"])
    return out


def _project(cam, pts):
    """World points -> pixel coordinates of a scene camera (inverse of Camera.points)."""
    d, cid = cam.d, cam.cam_id
    R = d.cam_xmat[cid].reshape(3, 3)
    pc = (np.asarray(pts, float) - d.cam_xpos[cid]) @ R          # camera frame: looks along -z, y up
    z = -pc[:, 2]
    ok = z > 1e-3
    u = cam.cx + cam.f * pc[:, 0] / np.where(ok, z, 1)
    v = cam.cy - cam.f * pc[:, 1] / np.where(ok, z, 1)
    return np.stack([u, v], 1), ok


def _draw_tile(tap, s, why, ids):
    from PIL import Image, ImageDraw
    img = Image.fromarray(s["rgb"])
    draw = ImageDraw.Draw(img)
    seg = s["seg"]
    obj_of = np.full(seg.shape, -1)
    for g, o in tap.lut.items():
        if o in ids:
            obj_of[seg == g] = ids.index(o)
    for k, o in enumerate(ids):                       # outlines of perceived objects + id labels
        m = obj_of == k
        if m.sum() < 10:
            continue
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        ys, xs = np.nonzero(edge)
        col = COLORS[k % len(COLORS)]
        for x, y in zip(xs[::2], ys[::2]):
            draw.point((int(x), int(y)), fill=col)
        cy, cx = np.argwhere(m).mean(0)
        draw.text((int(cx) - 10, int(cy) - 6), o, fill=(255, 255, 255), stroke_width=2, stroke_fill=col)
    pts = [p for t, p in tap.path if t <= s["t"]]
    if len(pts) > 1:                                  # the hand's path so far
        uv, ok = _project(tap.cam, pts)
        line = [tuple(map(float, q)) for q, k in zip(uv, ok) if k]
        if len(line) > 1:
            draw.line(line, fill=(255, 220, 0), width=2)
    return img


def make_sheet(tap, moments, out_path, title=""):
    from PIL import Image, ImageDraw
    ids = [tap.obj] + [o for o in tap.scene.objects if o != tap.obj]
    W, H = TILE
    cap = 34
    n = max(1, len(moments))
    sheet = Image.new("RGB", (n * W, 2 * H + cap + 18), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((6, 3), title[:190], fill=(235, 235, 235))
    for c, m in enumerate(moments):
        s = tap.samples[m["i"]]
        sheet.paste(_draw_tile(tap, s, m["why"], ids), (c * W, 18))
        sheet.paste(Image.fromarray(s["close"]), (c * W, 18 + H))
        d.text((c * W + 6, 18 + 2 * H + 3), f"t={s['t']:.2f} s", fill=(255, 220, 0))
        d.text((c * W + 6, 18 + 2 * H + 17), m["why"][:52], fill=(235, 235, 235))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)
    return out_path


def capture(task, seed, obj, out_path, *, positions=None, perturb=None, library=None, extra_cards=None,
            performance=None, patches=None, overrides=None, program=None, step=None, title="") -> dict | None:
    """Replay one episode exactly and build its evidence sheet for `obj`'s failing step. `step` = the failing step's
    trace entry from the original record ({skill, t0, t1, check}); without it the first failed step of the replay
    is used (one extra, render-free replay)."""
    from .executor import Episode
    if step is None:
        from .executor import run_episode
        rec, _ = run_episode(task, seed=seed, perturb=perturb, library=library, extra_cards=extra_cards,
                             performance=performance, patches=patches, positions=positions, overrides=overrides,
                             program=program)
        st = next((s for s in rec["subtasks"] if s.get("obj") == obj and not s.get("ok") and s.get("trace")), None)
        if st is None:
            return None
        step = next((t for t in st["trace"] if not t["ok"]), st["trace"][-1])
    ep = Episode(task, seed, "sim", None, overrides, False, False, perturb, extra_cards, library, performance,
                 positions, patches, 0, None, False)
    tap = FrameTap(ep.scene, ep.monitor, obj, (max(0.0, step["t0"] - 0.3), step["t1"] + 1.2), inner=ep.rec)
    ep.robot.recorder = tap
    try:
        if program or task.get("program"):
            from .program import run_program
            run_program(ep, program or task["program"])
        else:
            ep.run_plan()
    finally:
        ep.close()
    moments = select_moments(tap.samples, step)
    if not moments:
        return None
    path = make_sheet(tap, moments, out_path, title)
    tap.close_all()
    lines = [f"{k + 1}. t={m['t']:.2f} s: {m['why']}" for k, m in enumerate(moments)]
    return {"image": str(path), "moments": moments and [{"t": m["t"], "why": m["why"]} for m in moments],
            "text": ("Attached image: one column per moment (top: front camera, perceived objects outlined with their "
                     "ids, yellow line = the hand's path so far; bottom: close-up beside the hand). Moments:\n"
                     + "\n".join(lines))}
