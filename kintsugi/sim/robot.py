"""Motion layer: Cartesian moves with mink inverse kinematics, gripper control, and stepping.

Skills only call these functions; they never touch motor commands directly.
A separate robot-only model is used for IK so object free joints never enter the solve.
"""
from __future__ import annotations

import mujoco
import mink
import numpy as np

from .scene import HOME_Q, load_panda_spec

CTRL_DT = 0.02          # control period, seconds (50 Hz)


def down_rotation(yaw: float = 0.0, tilt: float = 0.0, tilt_axis: str = "x") -> np.ndarray:
    """Rotation of the TCP frame: z points down (approach), fingers close along the TCP y axis.

    yaw rotates about world z. tilt (radians) leans the approach about the TCP x or y axis.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    if tilt:
        ct, st = np.cos(tilt), np.sin(tilt)
        if tilt_axis == "x":
            Rt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        else:
            Rt = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        R = R @ Rt
    return R


GRIP_LOST_WIDTH = 0.003      # m: fingers closer than this hold nothing (same threshold as the place check)


class Robot:
    def __init__(self, scene, recorder=None, monitor=None):
        self.scene = scene
        self.m, self.d = scene.model, scene.data
        self.recorder = recorder
        self.monitor = monitor
        m = self.m
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i + 1}").id] for i in range(7)])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(f).id] for f in ("finger_joint1", "finger_joint2")])
        self.arm_act = np.array([m.actuator(f"actuator{i + 1}").id for i in range(7)])
        self.grip_act = m.actuator("actuator8").id
        self.tcp_site = m.site("tcp").id
        self.n_sub = int(round(CTRL_DT / m.opt.timestep))
        # IK model: robot only.
        ik_spec = load_panda_spec()
        self.ik_model = ik_spec.compile()
        self.cfg = mink.Configuration(self.ik_model)
        self.ik_finger_qadr = np.array([self.ik_model.jnt_qposadr[self.ik_model.joint(f).id]
                                        for f in ("finger_joint1", "finger_joint2")])
        self.frame_task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=0.6, lm_damping=1.0)
        self.posture = mink.PostureTask(self.ik_model, cost=1e-3)
        q0 = np.zeros(self.ik_model.nq)
        q0[:7] = HOME_Q
        q0[self.ik_finger_qadr] = 0.04
        self.posture.set_target(q0)
        self.limits = [mink.ConfigurationLimit(self.ik_model)]
        self.grip_cmd = 255.0
        self.t = 0.0
        self.ik_error = 0.0

    # ---------- state ----------
    def tcp_pose(self):
        p = self.d.site_xpos[self.tcp_site].copy()
        R = self.d.site_xmat[self.tcp_site].reshape(3, 3).copy()
        return p, R

    def finger_width(self) -> float:
        return float(self.d.qpos[self.finger_qadr].sum())

    def arm_q(self):
        return self.d.qpos[self.arm_qadr].copy()

    # ---------- stepping ----------
    def step(self, n_ctrl: int = 1):
        for _ in range(n_ctrl):
            self.d.ctrl[self.grip_act] = self.grip_cmd
            mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
            self.t += CTRL_DT
            if self.monitor:
                self.monitor.on_step(self)
            if self.recorder:
                self.recorder.on_step(self)

    def wait(self, seconds: float):
        self.step(max(1, int(round(seconds / CTRL_DT))))

    # ---------- IK ----------
    def _sync_cfg(self):
        q = np.zeros(self.ik_model.nq)
        q[:7] = self.arm_q()
        q[self.ik_finger_qadr] = self.d.qpos[self.finger_qadr]
        self.cfg.update(q)

    def _ik(self, pos, R, iters=8, dt=CTRL_DT):
        T = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), np.asarray(pos, float))
        self.frame_task.set_target(T)
        for _ in range(iters):
            v = mink.solve_ik(self.cfg, [self.frame_task, self.posture], dt, solver="daqp", damping=1e-3,
                              limits=self.limits)
            self.cfg.integrate_inplace(v, dt)
        err = self.frame_task.compute_error(self.cfg)
        self.ik_error = float(np.linalg.norm(err[:3]))
        return self.cfg.q[:7].copy()

    def move_to(self, pos, R, speed: float = 0.25, settle: float = 0.25, tol: float = 0.006,
                watch_grip: bool = False, correct: bool = True) -> dict:
        """Straight-line Cartesian move of the TCP to (pos, R). Returns reached/err info.
        watch_grip: the hand carries something; if the finger gap collapses (closed on nothing: the object slipped
        out) the move stops right there and returns lost_grip=True with the time (proprioception only).
        correct=False: no sag correction (it aims past the target by the residual, which pushes harder into
        whatever is blocking the hand: wrong for motions that must notice they have hit an end stop)."""
        self._sync_cfg()
        p0, R0 = self.tcp_pose()
        pos = np.asarray(pos, float)
        dist = float(np.linalg.norm(pos - p0))
        ang = float(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
        n = max(2, int(np.ceil(max(dist / (speed * CTRL_DT), ang / (1.2 * CTRL_DT)))))
        q0 = mink.SO3.from_matrix(R0)
        q1 = mink.SO3.from_matrix(R)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)                      # smooth start and stop
            p = p0 + a * (pos - p0)
            Ri = _slerp(q0, q1, a).as_matrix()
            qa = self._ik(p, Ri, iters=3)
            self.d.ctrl[self.arm_act] = qa
            self.step()
            if watch_grip and self.finger_width() < GRIP_LOST_WIDTH:
                return {"reached": False, "lost_grip": True, "lost_at": round(self.t, 2),
                        "pos_err": float(np.linalg.norm(self.tcp_pose()[0] - pos)), "ik_err": self.ik_error}
        qa = self._ik(pos, R, iters=20)
        self.d.ctrl[self.arm_act] = qa
        self.wait(settle)
        # Joint servos sag a few mm under gravity and load: aim past the target by the residual.
        cmd = pos.copy()
        for _ in range(3 if correct else 0):
            resid = pos - self.tcp_pose()[0]
            if np.linalg.norm(resid) < tol * 0.4:
                break
            cmd = cmd + resid
            self._sync_cfg()
            qa = self._ik(cmd, R, iters=15)
            self.d.ctrl[self.arm_act] = qa
            self.wait(0.15)
        err = float(np.linalg.norm(self.tcp_pose()[0] - pos))
        return {"reached": err < tol and self.ik_error < tol, "pos_err": err, "ik_err": self.ik_error}

    # ---------- gripper ----------
    def open(self, wait: float = 0.35):
        self.grip_cmd = 255.0
        self.wait(wait)

    def close(self, force: float = 25.0, wait: float = 0.5):
        self.m.actuator_forcerange[self.grip_act] = [-float(force), float(force)]
        self.grip_cmd = 0.0
        self.wait(wait)


def _slerp(q0, q1, a):
    """Fallback slerp for mink SO3 objects."""
    d = q0.inverse() @ q1
    return q0 @ mink.SO3.exp(d.log() * a)
