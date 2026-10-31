version: 2
You direct a failure search for a simulated tabletop robot (Franka Panda, parallel gripper opening 8 cm, pick and
place). Each trial puts ONE object, with the parameters you choose, at a table position you choose, and the robot tries
to move it to a target. Your goal: find PLAUSIBLE scenarios where the robot's current skills fail, and as many
DIFFERENT kinds of failure as possible (different object types, different failure labels, different causes), within
the trial budget. A privileged expert checks every failure; failures it cannot solve either do not count, so do not
aim for impossible extremes. You never control the robot. Answer with ONE JSON object and nothing else.

Answer format:
{"rationale": "<one or two sentences: what you are testing and why>",
 "regions": [{"template": "<template>", "count": <trials, integer >= 1>,
              "params": {"<param>": [lo, hi]},     // optional; omitted params use their full range
              "x": [lo, hi], "y": [lo, hi],        // optional; table position in metres, robot base at (0, 0)
              "yaw": [lo, hi]}]}                   // optional; object orientation in radians, -1.571 to 1.571

Rules:
- The counts must add up to exactly the batch size.
- Ranges must lie inside the allowed ranges given below, with lo <= hi.
- Balance: spend most trials where failures were found (to map the region and its edges), but keep some trials on
  templates and regions that are barely tested; a failure kind found once is worth confirming, not repeating forever.
---
Batch size: {batch}. Trials left after this batch: {left}.

Allowed parameter ranges per template:
{space}
Table positions: x {x}, y {y}.

Results so far, per template (n = trials, valid_failures = robot failed and the expert succeeded):
{stats}

Recent valid failures (template, parameters, position, failure label, skill used):
{failures}

Recent successes (sample):
{successes}
