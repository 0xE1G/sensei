"""Model-written primitives: parameterized motion code, run in a sandbox, trusted only after search and gates.

A primitive is Python that defines

    def primitive(obj, target, p):
        ...

`obj` / `target` are perception descriptors (center, width, height, top_z, axis_yaw, ...; target may be None),
`p` the parameter values (ranges are declared next to the code and searched by the explorer). The code may call
only the motion API below and math helpers; it never sees simulator state. The same AST whitelist idea as plan
programs (kintsugi.program), plus a cap on motion calls.
"""
from __future__ import annotations

import ast
import math

from .program import ALLOWED_NODES, SAFE_BUILTINS

MOTION_API = ("move", "open_gripper", "close_gripper", "wait", "tcp", "look")
MATH = {"sin": math.sin, "cos": math.cos, "atan2": math.atan2, "hypot": math.hypot, "sqrt": math.sqrt,
        "pi": math.pi, "radians": math.radians, "degrees": math.degrees,
        "clip": lambda v, lo, hi: max(lo, min(hi, v))}
MAX_MOVES = 40


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
    stats = {"moves": 0, "reached": 0, "error": None, "finger_width_mm_while_moving": []}

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

    api = {"move": move, "open_gripper": lambda: robot.open(),
           "close_gripper": lambda force=20.0: robot.close(force=float(min(max(force, 5), 40))),
           "wait": lambda s: robot.wait(float(min(max(s, 0.0), 2.0))),
           "tcp": lambda: [float(v) for v in robot.tcp_pose()[0]],
           "look": lambda: dict(look() or {}) if look else dict(obj), **MATH}
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
