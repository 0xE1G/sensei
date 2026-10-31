"""Records poses (for the Three.js replay) and optional MP4 video of an episode."""
from __future__ import annotations

import gzip
import json
import pathlib

import mujoco
import numpy as np

GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
              int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
              int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
              int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid"}


def export_geoms(model) -> dict:
    """Static description of every visible geom: enough for a browser to rebuild the scene."""
    geoms = []
    for g in range(model.ngeom):
        if model.geom_rgba[g, 3] == 0 or model.geom_group[g] > 2:
            continue
        t = GEOM_TYPES.get(int(model.geom_type[g]))
        if t is None:
            continue
        item = {"body": model.body(model.geom_bodyid[g]).name, "type": t,
                "size": model.geom_size[g].round(5).tolist(), "pos": model.geom_pos[g].round(5).tolist(),
                "quat": model.geom_quat[g].round(5).tolist(), "rgba": model.geom_rgba[g].round(3).tolist()}
        if t == "mesh":
            item["mesh"] = model.mesh(model.geom_dataid[g]).name
        geoms.append(item)
    return {"bodies": [model.body(b).name for b in range(model.nbody)], "geoms": geoms}


class Recorder:
    def __init__(self, scene, poses: bool = True, video: bool = False, fps: int = 30,
                 camera: str = "front", size=(480, 360)):
        self.scene, self.fps = scene, fps
        self.poses = poses
        self.video = video
        self.frames, self.pose_frames, self.events = [], [], []
        self._next = 0.0
        self._renderer = None
        if video:
            self._renderer = mujoco.Renderer(scene.model, size[1], size[0])
            self._cam = scene.model.camera(camera).id

    def on_step(self, robot):
        if robot.t + 1e-9 < self._next:
            return
        self._next += 1.0 / self.fps
        d = self.scene.data
        if self.poses:
            self.pose_frames.append({"t": round(robot.t, 3), "p": d.xpos.round(4).tolist(),
                                     "q": d.xquat.round(4).tolist()})
        if self._renderer is not None:
            self._renderer.update_scene(d, camera=self._cam)
            self.frames.append(self._renderer.render().copy())

    def event(self, t, kind, **info):
        self.events.append({"t": round(t, 3), "kind": kind, **info})

    def save(self, out_dir: pathlib.Path, stem: str, save_poses=True, save_video=True) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        if save_poses and self.pose_frames:
            p = out_dir / f"{stem}_poses.json.gz"
            with gzip.open(p, "wt") as f:
                json.dump({"fps": self.fps, "scene": export_geoms(self.scene.model), "frames": self.pose_frames,
                           "events": self.events}, f)
            files["poses"] = p.name
        if save_video and self.frames:
            import imageio.v2 as imageio
            p = out_dir / f"{stem}.mp4"
            imageio.mimsave(p, self.frames, fps=self.fps, macro_block_size=8)
            files["video"] = p.name
        return files

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
