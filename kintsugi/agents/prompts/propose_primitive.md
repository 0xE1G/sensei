version: 4
You write a new motion primitive for a simulated robot arm (Franka Panda with a parallel-jaw gripper) because its
skill library has no skill that can do what a task needs. You write the SHAPE of the motion as parameterized Python;
a search in simulation will then tune the parameter values within the ranges you declare, and statistical tests
decide whether the skill is kept. So declare wide but sensible ranges for anything you are unsure about. Answer with
ONE JSON object and nothing else.

The gripper: two fingers that open to 8 cm and close to 0; each finger pad is about 2 cm wide; the fingertips reach
about 1 cm below the tool point; the hand above the fingers is about 22 cm long along the direction the fingers
open/close and 7 cm wide. Used as a pushing surface, closed fingers make one contact about 2 cm wide; open fingers
make two contacts about 8 cm apart across the hand's y direction. The table is at z = 0. Positions are metres, the robot base is at (0, 0), +x points away
from the robot, +y to its left. Reachable: roughly 0.33 to 0.72 m from the base.

Motion API (the only functions you may call, besides math helpers sin cos atan2 hypot sqrt pi radians degrees
clip(v, lo, hi) and Python built-ins like min max abs):
- move(x, y, z, yaw=0.0, tilt=0.0, speed=0.2) -> bool: move the tool point in a straight line to (x, y, z) with the
  gripper pointing down. yaw (radians) turns the hand about vertical: the hand's x axis points along (cos yaw,
  sin yaw) and the fingers open/close along the horizontal axis perpendicular to it. tilt (radians, about the hand's
  x axis) leans the gripper. z is clamped to at least 0.008. speed in m/s.
- open_gripper(), close_gripper(force=20.0), wait(seconds), tcp() -> [x, y, z]
- look() -> a fresh perception dict of the object (same keys as obj; may be empty if it is not visible). Use it to
  work closed-loop: re-plan from where the object is now, and stop once it has reached the target. Motions after
  the goal is reached are measured as wasted and count against a proposal.

Write a module that defines exactly one entry point:
    def primitive(obj, target, p):
obj and target are dicts from perception (target may be None): "center" [x, y], "width" (footprint diameter, m),
"height", "top_z", "axis_yaw", "extent_major", "extent_minor", and for objects with parts "parts": {"handle":
{"center": [x, y, z], "tip": [x, y, z] (outermost point: a handle's graspable post), "normal": [nx, ny] (horizontal
unit vector the part faces, out of the object), "width", "height"}, "front": {... the front panel; its "width" runs
along the front}}. Joints are not observable: a drawer slides out along the front's normal; a door swings about a
vertical hinge at one side edge of its front (usually the edge opposite the handle). A handle post is about 1 cm
thick and 3.6 cm tall: grip it with fingers either side, the hand's x axis along the normal. p is a dict of your parameters: read them as
p['name']. Do not hard-code positions; compute everything from obj, target and p. No attribute access (use
obj['center']), no imports, classes, try/except, with, global. At most 40 move calls.

Answer format:
{"name": "<snake_case name for the skill, e.g. push_two_finger>",
 "code": "<python source, lines separated by \n>",
 "params": {"<param>": {"default": <number>, "range": [<lo>, <hi>], "meaning": "<what it controls>"}},
 "check": "<how to tell from perception afterwards that it worked>",
 "explanation": "<one or two sentences: the motion and why it should work>"}
---
Capability needed: {capability}
Why: {why}

Example object (perceived): {obj}
Example target (perceived): {target}
{feedback}
