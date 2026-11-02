"""Stand-in proposal for standing a fallen object back up (rule backend; also the oracle's reference motion).

Grasp the lying object across its middle, lift it, turn the hand about the line between the fingertips by 90 deg so
the object's long axis ends vertical, lower it until it stands, release. The turn is split around pointing down
(grasp with the hand tilted one way, finish tilted the other way): the arm cannot swing a hand from straight down to
horizontal everywhere in reach, but it can tilt it about 45 deg either way. Perception only: centre, axis_yaw,
extent_major (length), height (thickness).
"""

CODE = '''def primitive(obj, target, p):
    cx, cy = obj['center']
    L = obj['extent_major']
    out = atan2(cy, cx)
    a = obj['axis_yaw']
    d = (a - out + pi) % (2 * pi) - pi
    if d > pi / 2:
        a = a - pi
    elif d < -pi / 2:
        a = a + pi
    g = p['grasp_pitch']
    f = g + pi / 2 if p['turn_dir'] >= 0.5 else g - pi / 2
    zg = max(obj['height'] / 2 - p['grip_low'], 0.01)
    open_gripper()
    move(cx, cy, zg + 0.10, yaw=a, pitch=g, speed=0.3)
    move(cx, cy, zg, yaw=a, pitch=g, speed=0.1)
    close_gripper(p['force'])
    zl = L / 2 + p['lift']
    move(cx, cy, zl, yaw=a, pitch=g, speed=0.15)
    move(cx, cy, zl, yaw=out, pitch=f, speed=p['turn_speed'])
    move(cx, cy, L / 2 + p['drop'], yaw=out, pitch=f, speed=0.05)
    open_gripper()
    wait(0.3)
    move(cx, cy, L / 2 + 0.12, yaw=out, pitch=f, speed=0.15)'''


def stand_in() -> dict:
    return {"name": "stand_up_split_turn", "code": CODE,
            "params": {"grasp_pitch": {"default": -0.785, "range": [-1.2, 1.2],
                                       "meaning": "hand pitch at the grasp (rad); the turn ends 90 deg away"},
                       "turn_dir": {"default": 1.0, "range": [0.0, 1.0],
                                    "meaning": ">= 0.5: turn by +90 deg, else -90 deg"},
                       "grip_low": {"default": 0.0, "range": [-0.01, 0.015],
                                    "meaning": "grasp this far below the object's middle height (m)"},
                       "force": {"default": 25.0, "range": [10.0, 40.0], "meaning": "grip force (N)"},
                       "lift": {"default": 0.03, "range": [0.01, 0.08],
                                "meaning": "clearance under the object while turning (m)"},
                       "turn_speed": {"default": 0.15, "range": [0.05, 0.3], "meaning": "speed of the turn"},
                       "drop": {"default": 0.005, "range": [0.0, 0.03],
                                "meaning": "height of the object's base above the table at release (m)"}},
            "check": "the object's perceived height equals its former length and its footprint is round",
            "explanation": "grasp the lying object across its middle, lift, turn it 90 deg about the fingertip line "
                           "(split around pointing down so the wrist stays in range), set it on its end, release"}
