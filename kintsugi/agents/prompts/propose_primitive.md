version: 7
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
- move(x, y, z, yaw=0.0, tilt=0.0, speed=0.2, pitch=0.0) -> bool: move the tool point in a straight line to
  (x, y, z) with the gripper pointing down. yaw (radians) turns the hand about vertical: the hand's x axis points
  along (cos yaw, sin yaw) and the fingers open/close along the horizontal axis perpendicular to it. tilt (radians,
  about the hand's x axis) leans the gripper. pitch (radians, -pi/2..pi/2) turns the hand about the line between
  the fingertips (its y axis): a held object turns with it about that line, by the change in pitch between two
  moves; positive pitch swings the fingertips towards (cos yaw, sin yaw), pitch = pi/2 is the hand horizontal. The
  arm follows |pitch| up to about 0.9 rad everywhere in reach; nearer pi/2 it often cannot (the motion stops short).
  z is clamped to at least 0.008. speed in m/s.
- open_gripper(), close_gripper(force=20.0), wait(seconds), tcp() -> [x, y, z]
- look() -> a fresh perception dict of the object (same keys as obj; may be empty if it is not visible). Use it to
  work closed-loop: re-plan from where the object is now, and stop once it has reached the target. Motions after
  the goal is reached are measured as wasted and count against a proposal.
Part-level motions (prefer these for anything that grasps and moves a part of an object; they react to contact, so
the skill works on parts of different sizes and travel without hard-coded distances):
- grasp_part(part, approach=0.08, height=0.0, depth=0.0, force=25.0, from_above=0) -> bool: open, come in from
  `approach` m out along part['normal'] (or from above if from_above=1), pinch the part at part['tip'] shifted by
  `height` (z) and `depth` (into the object), fingers either side, the hand's x axis along the normal. True if the
  fingers hold something.
- move_linear(dx, dy, dz=0.0, dist=0.1, speed=0.06, until_stall=1, stall_lag=0.012) -> {'moved', 'stalled', 'lost'}:
  carry the hand along a direction for up to `dist` m; with until_stall it stops by itself when the hand falls
  `stall_lag` m behind its path (the part reached its end stop) and stops pulling, or when the grip is lost.
- move_arc(cx, cy, angle=90.0, speed=0.06, until_stall=1, stall_lag=0.012, turn_hand=1) -> {'turned', 'stalled',
  'lost'}: carry the hand around the vertical axis through (cx, cy) by up to `angle` degrees (positive =
  counter-clockwise seen from above), turning the hand with the arc; stops the same way.
- release() opens the gripper; retreat(dz=0.08) moves the hand straight up; holding() -> bool.
Prefer letting a motion stop at the end stop over a fixed travel distance: objects differ in how far they move.
When a revision request has "Camera evidence (attached)", the images show your previous version on a failed and a
successful training scene (one column per key moment: the front camera with the perceived objects outlined and the
hand's path in yellow, and a close-up beside the hand). Use them to see what the numbers miss (where the fingers
actually meet the object, what the hand hits, how the object moves) and change the motion accordingly.

Write a module that defines exactly one entry point:
    def primitive(obj, target, p):
obj and target are dicts from perception (target may be None): "center" [x, y], "width" (footprint diameter, m),
"height", "top_z", "axis_yaw", "extent_major", "extent_minor", and for objects with parts "parts": {"handle":
{"center": [x, y, z], "tip": [x, y, z] (outermost point: a handle's graspable post), "normal": [nx, ny] (horizontal
unit vector the part faces, out of the object), "width", "height"}, "front": {... the front panel; its "width" runs
along the front}}. Joints are not observable: a drawer slides out along the front's normal; a door swings about a
vertical hinge at one side edge of its front (usually the edge opposite the handle). An object lying on its side (fallen
over) is described like any other: its long axis is horizontal along axis_yaw, "extent_major" is its length,
"height" its thickness; which end was its base is not observable. A handle post is about 1 cm
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
