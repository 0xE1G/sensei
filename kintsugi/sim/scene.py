"""Build a MuJoCo scene: table plane, Panda, camera, and objects from a task spec.

The placer is deterministic code (never the language model): it puts objects
inside the arm's reach, without overlaps, with targets spaced apart, and
randomizes positions per seed.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .templates import ObjectSpec, add_object, resolve_params

ROOT = pathlib.Path(__file__).resolve().parents[2]
PANDA_XML = ROOT / "assets" / "franka_emika_panda" / "panda.xml"

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034          # flange to fingertip-pad centre, metres
REACH = dict(x=(0.32, 0.68), y=(-0.32, 0.32), r=(0.33, 0.72))

CAMERAS = {
    # name: (position, look-at)
    "front": ((1.25, 0.0, 0.85), (0.5, 0.0, 0.0)),
    "top": ((0.5, 0.0, 1.35), (0.5, 0.0, 0.0)),
    "side": ((0.55, -0.95, 0.65), (0.5, 0.0, 0.05)),
}


def _find_body(spec, name):
    for fn in ("body", "find_body"):
        if hasattr(spec, fn):
            b = getattr(spec, fn)(name)
            if b is not None:
                return b
    raise KeyError(name)


def _lookat_quat(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = pos - target
    z /= np.linalg.norm(z)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.99 else np.array([-1.0, 0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1).reshape(-1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q.tolist()


def load_panda_spec() -> mujoco.MjSpec:
    if not PANDA_XML.exists():
        raise FileNotFoundError(f"{PANDA_XML} missing. Run: python scripts/fetch_assets.py")
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    # Drop the robot-only keyframe; we set the home pose ourselves once objects are added.
    for key in list(spec.keys):
        try:
            spec.delete(key)
        except Exception:
            key.qpos, key.ctrl = [], []
    hand = _find_body(spec, "hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0, 0], rgba=[1, 0, 1, 0.5], group=4)
    return spec


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    objects: dict                       # id -> true-geometry meta (privileged)
    seed: int
    task: dict
    cameras: list = field(default_factory=lambda: list(CAMERAS))


def _sample_positions(objs, rng, restarts=60, fixed=None):
    """Try whole layouts until one fits. Raises if the task cannot fit on the table.
    `fixed` pre-places some objects ({id: (x, y, yaw)}); the rest are laid out around them."""
    for _ in range(restarts):
        try:
            return _sample_once(objs, rng, fixed=fixed)
        except _Crowded:
            continue
    raise RuntimeError("Could not place all objects: table too crowded for this task")


class _Crowded(Exception):
    pass


def _footprint(o) -> float:
    """Radius of the circle an object needs on the table."""
    p = resolve_params(o.template, o.params)
    if o.template in ("bin", "tray", "zone"):
        return 0.5 * math.hypot(p["width"], p["depth"])
    if o.template == "mug":
        return p["radius"] + p["handle"] + 0.01
    if o.template in ("bowl", "cup"):
        return p["radius"] + (p.get("depth", 0) * math.sin(math.radians(p.get("flare_deg", 0))) if o.template == "bowl" else 0) + 0.01
    if o.template == "bar":
        return 0.5 * math.hypot(p["length"], p["width"])
    return 0.5 * max(p.get("size", 0), 2 * p.get("radius", 0)) * 1.42


def _sample_once(objs, rng, max_tries=400, fixed=None):
    """Targets first, then items. Returns {id: (x, y, yaw)}."""
    fixed = fixed or {}
    order = sorted(objs, key=lambda o: 0 if o.role == "target" else 1)
    placed = [(fixed[o.id][0], fixed[o.id][1], _footprint(o)) for o in objs if o.id in fixed]   # (x, y, radius)
    out = {oid: tuple(v) for oid, v in fixed.items()}
    for o in order:
        if o.id in fixed:
            continue
        rad = _footprint(o)
        gap = 0.05 if o.role == "target" else 0.035
        for _ in range(max_tries):
            x = rng.uniform(*REACH["x"])
            y = rng.uniform(*REACH["y"])
            r = math.hypot(x, y)
            if not (REACH["r"][0] + rad * 0.3 <= r <= REACH["r"][1] - rad * 0.3):
                continue
            if all(math.hypot(x - px, y - py) > rad + pr + gap for px, py, pr in placed):
                break
        else:
            raise _Crowded(o.id)
        yaw_range = {"mug": math.pi, "bar": math.pi / 2, "block": math.pi / 4,
                     "bin": 0.3, "tray": 0.3, "zone": 0.3}.get(o.template, 0.0)
        yaw = rng.uniform(-yaw_range, yaw_range) if yaw_range else 0.0
        placed.append((x, y, rad))
        out[o.id] = (x, y, yaw)
    return out


def build_scene(task: dict, seed: int = 0, positions: dict | None = None, perturb: dict | None = None) -> Scene:
    """Build the scene for a task spec. `perturb` lets experiments randomize friction/mass by a factor."""
    rng = np.random.default_rng(seed)
    spec = load_panda_spec()
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    try:
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
    except Exception:
        pass
    wb = spec.worldbody
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05],
                rgba=[0.82, 0.78, 0.72, 1], friction=[1.0, 0.005, 0.0001])
    wb.add_light(name="key", pos=[0.6, -0.4, 2.0], dir=[-0.1, 0.2, -1], diffuse=[0.7, 0.7, 0.7],
                 specular=[0.2, 0.2, 0.2], castshadow=True)
    wb.add_light(name="fill", pos=[1.2, 0.6, 1.5], dir=[-0.6, -0.3, -1], diffuse=[0.35, 0.35, 0.35],
                 castshadow=False)
    for name, (cpos, tgt) in CAMERAS.items():
        wb.add_camera(name=name, pos=list(cpos), quat=_lookat_quat(cpos, tgt), fovy=50)

    objs = [ObjectSpec(**o) if isinstance(o, dict) else o for o in task["objects"]]
    if perturb:
        for o in objs:
            p = dict(o.params)
            for k, f in perturb.items():
                if k in p or k in ("mass", "friction"):
                    from .templates import DEFAULTS
                    base = p.get(k, DEFAULTS[o.template].get(k))
                    if base is not None:
                        p[k] = base * rng.uniform(*f) if isinstance(f, (list, tuple)) else base * f
            o.params = p
    layout = positions or _sample_positions(objs, rng, fixed=task.get("layout_fixed"))
    metas = {}
    for o in objs:
        x, y, yaw = layout[o.id]
        metas[o.id] = add_object(spec, o, (x, y), yaw)

    spec.add_exclude(bodyname1="left_finger", bodyname2="right_finger")
    model = spec.compile()
    data = mujoco.MjData(model)
    _set_home(model, data)
    _strengthen_gripper(model)
    _stiffen_contacts(model)
    mujoco.mj_forward(model, data)
    return Scene(model=model, data=data, objects=metas, seed=seed, task=task)


def _set_home(model, data):
    for i in range(7):
        j = model.joint(f"joint{i + 1}")
        data.qpos[model.jnt_qposadr[j.id]] = HOME_Q[i]
        data.ctrl[model.actuator(f"actuator{i + 1}").id] = HOME_Q[i]
    for f in ("finger_joint1", "finger_joint2"):
        data.qpos[model.jnt_qposadr[model.joint(f).id]] = 0.04
    data.ctrl[model.actuator("actuator8").id] = 255


def _strengthen_gripper(model, kp: float = 1500.0):
    """Menagerie's finger servo is soft (kp=100 N/m), too weak to hold a bowl by its rim.

    A stiffer servo needs stiffer finger joint limits and some armature, otherwise the light
    fingers are pushed straight through their limits (MuJoCo limits are soft constraints).
    """
    a = model.actuator("actuator8").id
    model.actuator_gainprm[a, 0] = kp * 0.04 / 255.0
    model.actuator_biasprm[a, 1] = -kp
    model.actuator_biasprm[a, 2] = -40.0
    model.actuator_forcerange[a] = [-40.0, 40.0]
    for f in ("finger_joint1", "finger_joint2"):
        j = model.joint(f).id
        model.jnt_solref[j] = [0.004, 1.0]
        model.dof_armature[model.jnt_dofadr[j]] = 0.05


def _stiffen_contacts(model):
    """Stiffer, less penetrable contacts so fingers cannot squeeze through thin walls."""
    for g in range(model.ngeom):
        model.geom_solref[g] = [0.004, 1.0]
        model.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2.0]
