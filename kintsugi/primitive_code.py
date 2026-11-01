"""Model-written primitives: parameterized motion code, run in a sandbox, trusted only after search and gates.

A primitive is Python that defines

    def primitive(obj, target, p):
        ...

`obj` / `target` are perception descriptors (center, width, height, top_z, axis_yaw, ...; target may be None),
`p` the parameter values (ranges are declared next to the code and searched by the explorer). The code may call
only the motion API below and math helpers; it never sees simulator state. The same AST whitelist idea as plan
programs (kintsugi.program), plus a cap on motion calls.

Two levels of motion API. Low level: move / open_gripper / close_gripper / wait / tcp / look. Part level (fixed,
generic, defined on perceived parts and directions, never on object names): grasp_part (approach and pinch a part
such as a handle), move_linear and move_arc (carry the hand along a line or an arc, stopping by themselves when the
hand stalls against an end stop or the grip is lost: proprioception only), release, retreat, holding. A skill written
on the part level is a short composition (grasp_part, then move_linear until it stalls), so a fix to one part-level
motion helps every skill that uses it.
"""
from __future__ import annotations

import ast
import math

import numpy as np

from .program import ALLOWED_NODES, SAFE_BUILTINS

MOTION_API = ("move", "open_gripper", "close_gripper", "wait", "tcp", "look",
              "grasp_part", "move_linear", "move_arc", "release", "retreat", "holding")
MATH = {"sin": math.sin, "cos": math.cos, "atan2": math.atan2, "hypot": math.hypot, "sqrt": math.sqrt,
        "pi": math.pi, "radians": math.radians, "degrees": math.degrees,
        "clip": lambda v, lo, hi: max(lo, min(hi, v))}
MAX_MOVES = 40
MAX_SEGMENTS = 400        # small steps of the part-level line / arc motions, per primitive run
STEP = 0.01               # m per step of move_linear; move_arc uses the same arc length


class PrimitiveError(Exception):
    pass


def check_primitive(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]
    errs = []
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "primitive"]
    if not fn:
        errs.append("define a function primitive(obj, target, p)")
    elif [a.arg for a in fn[0].args.args] != ["obj", "target", "p"]:
        errs.append("primitive must take exactly (obj, target, p)")
    defined = set(MOTION_API) | set(MATH) | set(SAFE_BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            errs.append(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed")
        elif isinstance(node, ast.Attribute):
            errs.append(f"line {node.lineno}: attribute access is not allowed (use obj['center'] style)")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
            errs.append(f"line {node.lineno}: unknown name '{node.id}' (motion API: {', '.join(MOTION_API)})")
    return sorted(set(errs))


def run_primitive(robot, code: str, obj: dict, target: dict | None, p: dict, on_move=None, look=None) -> dict:
    """Execute a primitive on the robot. Returns {moves, reached, error}.
    on_move(t0, t1): called after every motion (the episode audits it); look(): a fresh perception of the object,
    so a primitive can work closed-loop (e.g. stop pushing once the object has arrived)."""
    from .sim.robot import down_rotation
    stats = {"moves": 0, "reached": 0, "error": None, "finger_width_mm_while_moving": [],
             "part_motions": [], "hand_path_mm": []}     # proprioception: what the hand did, for feedback

    def move(x, y, z, yaw=0.0, tilt=0.0, speed=0.2):
        stats["moves"] += 1
        if stats["moves"] > MAX_MOVES:
            raise PrimitiveError(f"more than {MAX_MOVES} moves")
        z = max(0.008, float(z))                      # never drive the fingertips into the table
        t0 = robot.t
        r = robot.move_to([float(x), float(y), z], down_rotation(float(yaw), tilt=float(tilt), tilt_axis="x"),
                          speed=float(min(max(speed, 0.02), 0.5)))
        if on_move:
            on_move(t0, robot.t)
        stats["reached"] += int(r["reached"])
        stats["finger_width_mm_while_moving"].append(round(1000 * robot.finger_width()))
        return bool(r["reached"])

    from .sim.robot import GRIP_LOST_WIDTH
    seg = {"n": 0}

    def _count():
        stats["moves"] += 1
        if stats["moves"] > MAX_MOVES:
            raise PrimitiveError(f"more than {MAX_MOVES} moves")
        stats["hand_path_mm"].append([round(1000 * v) for v in robot.tcp_pose()[0]])

    def _segment(pos, R, speed):
        seg["n"] += 1
        if seg["n"] > MAX_SEGMENTS:
            raise PrimitiveError(f"more than {MAX_SEGMENTS} line/arc steps")
        t0 = robot.t
        robot.move_to(np.asarray(pos, float), R, speed=float(min(max(speed, 0.02), 0.3)), settle=0.02, tol=1.0,
                      correct=False)
        if on_move:
            on_move(t0, robot.t)

    def holding():
        return bool(GRIP_LOST_WIDTH < robot.finger_width() < 0.076)

    def grasp_part(part, approach=0.08, height=0.0, depth=0.0, force=25.0, from_above=0):
        """Pinch a perceived part (e.g. obj['parts']['handle']) at its tip: fingers either side, the hand's x axis
        along the part's normal. Grasp point = tip + height (z) - depth along the normal (into the object);
        approach from `approach` m out along the normal (or above it if from_above). True if it holds something."""
        _count()
        n = np.asarray(part["normal"][:2], float)
        n = n / max(np.linalg.norm(n), 1e-6)
        tip = np.asarray(part["tip"], float)
        g = np.array([tip[0] - n[0] * depth, tip[1] - n[1] * depth, max(0.008, tip[2] + height)])
        pre = g + (np.array([0.0, 0.0, approach]) if from_above else np.array([n[0] * approach, n[1] * approach, 0.0]))
        R = down_rotation(float(math.atan2(n[1], n[0])))
        robot.open()
        for pos, sp in ((pre, 0.2), (g, 0.08)):
            t0 = robot.t
            robot.move_to(pos, R, speed=sp)
            if on_move:
                on_move(t0, robot.t)
        robot.close(force=float(min(max(force, 5), 40)))
        stats["part_motions"].append({"motion": "grasp_part", "holding": holding(),
                                      "finger_width_mm": round(1000 * robot.finger_width())})
        return holding()

    def _hold_here(R):
        """Stop pulling: command the hand where it actually is, so it no longer strains against an end stop."""
        robot.move_to(robot.tcp_pose()[0], R, speed=0.1, settle=0.15, tol=1.0, correct=False)

    def move_linear(dx, dy, dz=0.0, dist=0.1, speed=0.06, until_stall=1, stall_lag=0.012):
        """Carry the hand along direction (dx, dy, dz) for up to `dist` m in small steps, keeping its orientation.
        Stops early when the hand falls more than stall_lag behind the commanded path (it hit an end stop) or the
        grip is lost. Returns {'moved': m along the direction, 'stalled': bool, 'lost': bool}."""
        _count()
        u = np.array([dx, dy, dz], float)
        u = u / max(np.linalg.norm(u), 1e-6)
        p0, R = robot.tcp_pose()
        held = holding()
        out = {"moved": 0.0, "stalled": False, "lost": False}
        k, total = 0, float(min(max(dist, 0.0), 0.4))
        while k * STEP < total:
            k += 1
            cmd = min(k * STEP, total)
            _segment(p0 + u * cmd, R, speed)
            along = float((robot.tcp_pose()[0] - p0) @ u)
            out["moved"] = round(along, 4)
            if held and not holding():
                out["lost"] = True
                break
            if until_stall and cmd - along > stall_lag:
                out["stalled"] = True
                _hold_here(R)
                break
        stats["part_motions"].append({"motion": "move_linear", **out})
        return out

    def move_arc(cx, cy, angle=90.0, speed=0.06, until_stall=1, stall_lag=0.012, turn_hand=1):
        """Carry the hand along a horizontal arc about the vertical axis through (cx, cy): `angle` degrees
        (positive = counter-clockwise seen from above), turning the hand with the arc if turn_hand. Stops early on a
        stall or a lost grip like move_linear. Returns {'turned': degrees actually turned, 'stalled', 'lost'}."""
        _count()
        p0, R0 = robot.tcp_pose()
        r = float(math.hypot(p0[0] - cx, p0[1] - cy))
        a0 = float(math.atan2(p0[1] - cy, p0[0] - cx))
        sgn = 1.0 if angle >= 0 else -1.0
        total = math.radians(min(abs(float(angle)), 180.0))
        da = STEP / max(r, 0.02)
        held = holding()
        out = {"turned": 0.0, "stalled": False, "lost": False}
        k = 0
        while k * da < total:
            k += 1
            a = sgn * min(k * da, total)
            Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
            R = Rz @ R0 if turn_hand else R0
            cmd = np.array([cx + r * math.cos(a0 + a), cy + r * math.sin(a0 + a), p0[2]])
            _segment(cmd, R, speed)
            p = robot.tcp_pose()[0]
            d = (math.atan2(p[1] - cy, p[0] - cx) - a0 + math.pi) % (2 * math.pi) - math.pi
            out["turned"] = round(math.degrees(sgn * d), 1)
            if held and not holding():
                out["lost"] = True
                break
            if until_stall and float(np.linalg.norm(p[:2] - cmd[:2])) > stall_lag:
                out["stalled"] = True
                _hold_here(R)
                break
        stats["part_motions"].append({"motion": "move_arc", **out})
        return out

    def release():
        robot.open()

    def retreat(dz=0.08):
        _count()
        p, R = robot.tcp_pose()
        t0 = robot.t
        robot.move_to(p + np.array([0.0, 0.0, float(min(max(dz, 0.0), 0.2))]), R, speed=0.2)
        if on_move:
            on_move(t0, robot.t)

    api = {"move": move, "open_gripper": lambda: robot.open(),
           "close_gripper": lambda force=20.0: robot.close(force=float(min(max(force, 5), 40))),
           "wait": lambda s: robot.wait(float(min(max(s, 0.0), 2.0))),
           "tcp": lambda: [float(v) for v in robot.tcp_pose()[0]],
           "look": lambda: dict(look() or {}) if look else dict(obj),
           "grasp_part": grasp_part, "move_linear": move_linear, "move_arc": move_arc, "release": release,
           "retreat": retreat, "holding": holding, **MATH}
    errs = check_primitive(code)
    if errs:
        stats["error"] = "static check failed: " + "; ".join(errs)
        return stats
    env = {"__builtins__": SAFE_BUILTINS, **api}
    try:
        exec(compile(code, "<primitive>", "exec"), env)
        env["primitive"](dict(obj), dict(target) if target else None, dict(p))
    except PrimitiveError as e:
        stats["error"] = str(e)
    except Exception as e:                            # a broken proposal is a result, not a crash
        stats["error"] = f"{type(e).__name__}: {e}"
    return stats
