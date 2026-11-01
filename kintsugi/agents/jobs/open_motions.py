"""Stand-in motion code for opening articulated objects (rule backend of propose_primitive).

Same contract as model-written primitives: primitive(obj, target, p) over the motion API, reading only perception
(obj['parts']['handle'] / ['front']: centre, tip, normal, width). Joints are not observable: a drawer is pulled
straight out along the front's normal; a door's hinge is assumed at the edge of the front opposite the handle and
the handle is pulled along the arc around it.
"""

PULL_CODE = '''def primitive(obj, target, p):
    parts = obj['parts']
    hx, hy, hz = parts['handle']['tip']
    nx, ny = parts['front']['normal'] if 'front' in parts else parts['handle']['normal']
    yaw = atan2(ny, nx)
    zg = hz - p['grip_depth']
    open_gripper()
    move(hx, hy, hz + p['clearance'], yaw=yaw, speed=0.2)
    move(hx, hy, zg, yaw=yaw, speed=0.08)
    close_gripper(force=p['force'])
    steps = max(1, int(p['pull'] / 0.03))
    for i in range(1, steps + 1):
        d = p['pull'] * i / steps
        move(hx + nx * d, hy + ny * d, zg, yaw=yaw, speed=p['speed'])
    open_gripper()
    x, y, z = tcp()
    move(x, y, z + 0.08, yaw=yaw, speed=0.2)
    return True
'''

PULL_PARAMS = {
    "grip_depth": {"default": 0.015, "range": [0.005, 0.03], "meaning": "how far below the handle's top the fingers pinch (m)"},
    "clearance": {"default": 0.06, "range": [0.03, 0.1], "meaning": "height above the handle before descending (m)"},
    "force": {"default": 30.0, "range": [15.0, 40.0], "meaning": "grip force (N)"},
    "pull": {"default": 0.11, "range": [0.05, 0.16], "meaning": "pull distance along the front's normal (m)"},
    "speed": {"default": 0.06, "range": [0.02, 0.15], "meaning": "pull speed (m/s)"},
}

ARC_CODE = '''def primitive(obj, target, p):
    parts = obj['parts']
    hx, hy, hz = parts['handle']['tip']
    f = parts['front']
    nx, ny = f['normal']
    tx, ty = -ny, nx
    fx, fy = f['center'][0], f['center'][1]
    side = 1.0 if (hx - fx) * tx + (hy - fy) * ty >= 0 else -1.0
    half = f['width'] / 2 + p['hinge_shift']
    cx, cy = fx - side * tx * half, fy - side * ty * half
    r = hypot(hx - cx, hy - cy)
    a0 = atan2(hy - cy, hx - cx)
    turn = 1.0 if (-sin(a0)) * nx + cos(a0) * ny >= 0 else -1.0
    yaw0 = atan2(ny, nx)
    zg = hz - p['grip_depth']
    open_gripper()
    move(hx, hy, hz + p['clearance'], yaw=yaw0, speed=0.2)
    move(hx, hy, zg, yaw=yaw0, speed=0.08)
    close_gripper(force=p['force'])
    total = radians(p['angle'])
    steps = max(2, int(p['angle'] / 8))
    for i in range(1, steps + 1):
        a = turn * total * i / steps
        move(cx + r * cos(a0 + a), cy + r * sin(a0 + a), zg, yaw=yaw0 + a, speed=p['speed'])
    open_gripper()
    x, y, z = tcp()
    move(x, y, z + 0.08, yaw=yaw0 + turn * total, speed=0.2)
    return True
'''

ARC_PARAMS = {
    "grip_depth": {"default": 0.015, "range": [0.005, 0.03], "meaning": "how far below the handle's top the fingers pinch (m)"},
    "clearance": {"default": 0.06, "range": [0.03, 0.1], "meaning": "height above the handle before descending (m)"},
    "force": {"default": 30.0, "range": [15.0, 40.0], "meaning": "grip force (N)"},
    "angle": {"default": 75.0, "range": [40.0, 100.0], "meaning": "how far to swing the handle around the hinge (deg)"},
    "hinge_shift": {"default": 0.0, "range": [-0.02, 0.02], "meaning": "correction of the assumed hinge position along the front (m)"},
    "speed": {"default": 0.06, "range": [0.02, 0.15], "meaning": "pull speed (m/s)"},
}


def stand_in(capability: str) -> dict:
    """propose_primitive output for an opening capability: straight pull, or an arc for doors / hinges."""
    arc = any(w in capability for w in ("door", "hinge", "lid", "swing"))
    return {"name": "open_arc_pull" if arc else "open_straight_pull", "code": ARC_CODE if arc else PULL_CODE,
            "params": ARC_PARAMS if arc else PULL_PARAMS,
            "check": "the moving part opened by the goal amount",
            "explanation": ("grasp the handle post and pull it along the arc around the hinge (assumed at the front's "
                            "edge opposite the handle)") if arc else
                           "grasp the handle post and pull straight out along the front's normal"}
