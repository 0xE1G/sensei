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


# Part-level versions (the motion API's grasp_part / move_linear / move_arc): pull or swing until the hand stalls
# against the end stop instead of a fixed distance, so short drawers are not overpulled (they bounced back).
PULL_STALL_CODE = '''def primitive(obj, target, p):
    parts = obj['parts']
    h = parts['handle']
    n = parts['front']['normal'] if 'front' in parts else h['normal']
    if not grasp_part(h, approach=p['clearance'], height=-p['grip_depth'], force=p['force'], from_above=1):
        release()
        retreat(0.08)
        return False
    move_linear(n[0], n[1], 0.0, dist=p['max_pull'], speed=p['speed'], until_stall=1, stall_lag=p['stall_lag'])
    release()
    retreat(0.08)
    return True
'''

PULL_STALL_PARAMS = {
    "grip_depth": PULL_PARAMS["grip_depth"], "clearance": PULL_PARAMS["clearance"], "force": PULL_PARAMS["force"],
    "max_pull": {"default": 0.16, "range": [0.06, 0.2], "meaning": "longest pull along the front's normal (m); the pull "
                                                                    "stops earlier when the drawer reaches its end stop"},
    "speed": PULL_PARAMS["speed"],
    "stall_lag": {"default": 0.012, "range": [0.006, 0.025], "meaning": "how far the hand may fall behind its path "
                                                                        "before the pull counts as stopped (m)"},
}

ARC_STALL_CODE = '''def primitive(obj, target, p):
    parts = obj['parts']
    h = parts['handle']
    f = parts['front']
    hx, hy, hz = h['tip']
    nx, ny = f['normal']
    tx, ty = -ny, nx
    fx, fy = f['center'][0], f['center'][1]
    side = 1.0 if (hx - fx) * tx + (hy - fy) * ty >= 0 else -1.0
    half = f['width'] / 2 + p['hinge_shift']
    cx, cy = fx - side * tx * half, fy - side * ty * half
    a0 = atan2(hy - cy, hx - cx)
    turn = 1.0 if (-sin(a0)) * nx + cos(a0) * ny >= 0 else -1.0
    if not grasp_part(h, approach=p['clearance'], height=-p['grip_depth'], force=p['force'], from_above=1):
        release()
        retreat(0.08)
        return False
    move_arc(cx, cy, angle=turn * p['angle'], speed=p['speed'], until_stall=1, stall_lag=p['stall_lag'])
    release()
    retreat(0.08)
    return True
'''

ARC_STALL_PARAMS = {
    "grip_depth": ARC_PARAMS["grip_depth"], "clearance": ARC_PARAMS["clearance"], "force": ARC_PARAMS["force"],
    "angle": {"default": 85.0, "range": [40.0, 110.0], "meaning": "largest swing around the hinge (deg); stops "
                                                                  "earlier at the door's end stop"},
    "hinge_shift": ARC_PARAMS["hinge_shift"], "speed": ARC_PARAMS["speed"],
    "stall_lag": {"default": 0.015, "range": [0.006, 0.03], "meaning": "how far the hand may fall behind the arc "
                                                                       "before the swing counts as stopped (m)"},
}



# Closing: the same part-level motions in the other direction, stopping when the part reaches its closed stop.
PUSH_IN_CODE = PULL_STALL_CODE.replace("move_linear(n[0], n[1], 0.0, dist=p['max_pull']",
                                       "move_linear(-n[0], -n[1], 0.0, dist=p['max_pull']")
# grip the middle of the post when closing: gripping lower puts the palm on the front panel's top edge, which drags
# the closed drawer out again as the hand lifts away (measured: 4/12 -> 11/12)
CLOSE_GRIP = {"default": 0.0, "range": [-0.015, 0.03], "meaning": "how far below the handle's centre the fingers pinch (m)"}
PUSH_IN_PARAMS = {**PULL_STALL_PARAMS, "grip_depth": CLOSE_GRIP,
                  "max_pull": {"default": 0.18, "range": [0.06, 0.22],
                               "meaning": "longest push back in along the front's normal (m); stops at the closed stop"}}

ARC_CLOSE_CODE = ARC_STALL_CODE.replace(
    "    turn = 1.0 if (-sin(a0)) * nx + cos(a0) * ny >= 0 else -1.0\n",
    "    ox, oy = obj['center']\n"
    "    turn = 1.0 if (-sin(a0)) * (ox - hx) + cos(a0) * (oy - hy) >= 0 else -1.0\n")
ARC_CLOSE_PARAMS = {**ARC_STALL_PARAMS,
                    "angle": {"default": 95.0, "range": [40.0, 120.0],
                              "meaning": "largest swing back towards the cabinet (deg); stops at the closed stop"}}


def stand_in(capability: str) -> dict:
    """propose_primitive output for an opening capability: straight pull, or an arc for doors / hinges. Both are
    written on the part-level motion API and stop at the end stop (the fixed-distance versions above are kept for
    comparison: they overpull short drawers)."""
    arc = any(w in capability.replace("slide", "") for w in ("door", "hinge", "lid", "swing"))
    if "close" in capability or "shut" in capability:
        return {"name": "close_arc_push" if arc else "close_straight_push",
                "code": ARC_CLOSE_CODE if arc else PUSH_IN_CODE, "params": ARC_CLOSE_PARAMS if arc else PUSH_IN_PARAMS,
                "check": "the moving part closed to the goal amount",
                "explanation": ("grasp the handle and swing it back around the hinge towards the cabinet until it "
                                "stops") if arc else "grasp the handle and push the drawer straight back in until it stops"}
    return {"name": "open_arc_pull" if arc else "open_straight_pull", "code": ARC_STALL_CODE if arc else PULL_STALL_CODE,
            "params": ARC_STALL_PARAMS if arc else PULL_STALL_PARAMS,
            "check": "the moving part opened by the goal amount",
            "explanation": ("grasp the handle post and pull it along the arc around the hinge (assumed at the front's "
                            "edge opposite the handle)") if arc else
                           "grasp the handle post and pull straight out along the front's normal"}
