"""Off-screen rendering: colour, depth and segmentation, plus depth-to-point-cloud."""
from __future__ import annotations

import mujoco
import numpy as np


class Camera:
    def __init__(self, model, data, name: str = "front", width: int = 320, height: int = 240):
        self.m, self.d, self.name = model, data, name
        self.w, self.h = width, height
        self.cam_id = model.camera(name).id
        self._rgb = mujoco.Renderer(model, height, width)
        self._depth = mujoco.Renderer(model, height, width)
        self._depth.enable_depth_rendering()
        self._seg = mujoco.Renderer(model, height, width)
        self._seg.enable_segmentation_rendering()
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        self.f = 0.5 * height / np.tan(fovy / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2

    def rgb(self) -> np.ndarray:
        self._rgb.update_scene(self.d, camera=self.cam_id)
        return self._rgb.render().copy()

    def depth(self) -> np.ndarray:
        self._depth.update_scene(self.d, camera=self.cam_id)
        return self._depth.render().copy()

    def segmentation(self) -> np.ndarray:
        """(H, W) array of geom ids, -1 for background. Stand-in for SAM masks."""
        self._seg.update_scene(self.d, camera=self.cam_id)
        s = self._seg.render()
        ids = s[..., 0].astype(np.int64)
        ids[s[..., 1] != mujoco.mjtObj.mjOBJ_GEOM] = -1
        return ids

    def points(self, depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Back-project depth pixels (optionally masked) into world-frame points, shape (N, 3)."""
        v, u = np.nonzero(mask if mask is not None else np.ones_like(depth, bool))
        z = depth[v, u]
        ok = (z > 0.01) & (z < 10)
        u, v, z = u[ok], v[ok], z[ok]
        x_c = (u - self.cx) * z / self.f
        y_c = -(v - self.cy) * z / self.f
        pc = np.stack([x_c, y_c, -z], axis=1)
        R = self.d.cam_xmat[self.cam_id].reshape(3, 3)
        return pc @ R.T + self.d.cam_xpos[self.cam_id]

    def close(self):
        for r in (self._rgb, self._depth, self._seg):
            r.close()
