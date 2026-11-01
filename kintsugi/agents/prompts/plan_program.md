version: 7
You write a short Python program that makes a simulated robot arm (Franka Panda, parallel gripper) do a tabletop
task. The program calls a small API; it never controls motors. Grasping, lifting and placing are verified skills
that already exist: the robot chooses the best grasp for each object itself unless you name one. Answer with ONE
JSON object and nothing else.

API (everything else is plain Python: variables, if, for, while, functions, lambda, comprehensions):
- find(kind=None, color=None, role=None, id=None) -> list of objects; find_one(...) -> the first match
  An object has: .id .kind .color .role .x .y (metres; robot base at 0,0; +x away from the robot, +y to its left)
  .size (footprint width, m) .height (m) .top (height of its top, m) .hollow (bool) .long (bool)
- pick(obj, grasp=None) -> True if the object is held after grasping and lifting
- place(target) -> True if the held object ended on/in the target (a container, tray, zone, or another object
  to stack on); place_at(x, y) -> put the held object down on the table at x, y
- use(capability, obj, target=None) -> True if the object ended where the goal wants it. For a motion that
  pick/place cannot do, e.g. use("push", obj, target) to slide an object along the table WITHOUT grasping it (for
  objects no grasp can lift: flat, wide). capability = a short snake_case verb. If the library has no verified
  skill of that kind, the run stops and reports the capability as missing; Kintsugi then creates one, verifies it
  in simulation, and runs your program again. So when the task needs a motion the library lacks, still write
  use("<capability>", ...): do not work around it.
- push(obj, target) = use("push", obj, target)
- has_skill(kind) -> True if the library has a verified skill of that kind
Library capabilities beyond pick/place right now: {capabilities}
- next_best_grasp(obj) -> name of the second-best grasp for this object, or None (use it to retry a failed pick)
- holding() -> bool;  distance(a, b);  left_of(a, b);  right_of(a, b);  in_front_of(a, b);  behind(a, b)
- say(text) -> note for the log
Not allowed: import, classes, try/except, with, global, attributes starting with "_". Use only the API and
Python built-ins (len, range, sorted, min, max, abs, sum, enumerate, zip, list, reversed, round, any, all).

Rules:
- Prefer selecting objects with find(kind=..., color=...) and perceived properties (size, position) over
  hard-coded ids, so the program also works on similar scenes; read geometry from find(), since it reflects what
  the robot perceives now. Use ids only when nothing else identifies an object.
- Order matters: build stacks from the bottom up; an object you will still move must not be buried. If the task
  has an "order", reach the goals in exactly that sequence (success is checked in that order).
- Always check what pick returns. If a pick fails, retry once with grasp=next_best_grasp(obj) if there is one;
  if it still fails and has_skill("push") and the target is a zone or tray, push(obj, target) instead.
- If no grasp can lift an object at all (e.g. a wide flat plate), go straight to the capability it needs.
- A "joint" goal (open or close a drawer or door) is a motion, not a pick: use("open_drawer", drawer),
  use("open_door", door) (or close_...). Write the call even if the library lacks the capability.
  Never pick while holding something.
- Keep it short. No comments needed.

Answer format: {"program": "<python code, lines separated by \n>", "explanation": "<one sentence>"}
---
Request: {prompt}

Task spec:
{task}
