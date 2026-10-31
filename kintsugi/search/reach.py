"""Reachability pre-filter: can the arm put its TCP at the pre-grasp and grasp poses?

Inverse kinematics only (mink, robot-only model, joint limits enforced), no physics, so poses the arm
cannot reach are dropped before any trial is simulated.
"""
from __future__ import annotations

import mink
import numpy as np

from ..sim.robot import down_rotation
from ..sim.scene import HOME_Q, load_panda_spec
from ..skills.library import GRASP_POSES


class ReachChecker:
    def __init__(self, tol_pos: float = 0.005, tol_rot: float = 0.05):
        self.model = load_panda_spec().compile()
        self.cfg = mink.Configuration(self.model)
        self.task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=0.6, lm_damping=1.0)
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.q0 = np.zeros(self.model.nq)
        self.q0[:7] = HOME_Q
        self.q0[7:9] = 0.04
        self.posture.set_target(self.q0)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.tol_pos, self.tol_rot = tol_pos, tol_rot

    def _solve(self, pos, R, iters=60, dt=0.05):
        T = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), np.asarray(pos, float))
        self.task.set_target(T)
        for _ in range(iters):
            v = mink.solve_ik(self.cfg, [self.task, self.posture], dt, solver="daqp", damping=1e-3,
                              limits=self.limits)
            self.cfg.integrate_inplace(v, dt)
        e = self.task.compute_error(self.cfg)
        return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def reachable(self, desc: dict, p: dict, primitive: str = "feature_pinch") -> dict:
        """Pre-grasp then grasp, starting from the home pose, as the skill moves."""
        g = GRASP_POSES[primitive](desc, p)
        R = down_rotation(g["yaw"], tilt=g["tilt"], tilt_axis="x")
        self.cfg.update(self.q0)
        worst = (0.0, 0.0)
        for pos in ([*g["point"], desc["top_z"] + p["approach_height"]], [*g["point"], g["z"]]):
            ep, er = self._solve(pos, R)
            worst = (max(worst[0], ep), max(worst[1], er))
        ok = worst[0] < self.tol_pos and worst[1] < self.tol_rot
        return {"ok": ok, "pos_err": round(worst[0], 4), "rot_err": round(worst[1], 4)}
