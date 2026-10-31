version: 4
You diagnose why a step of a simulated robot task failed. The robot is a Franka Panda arm with a parallel-jaw gripper
(opens 8 cm; the fingers are about 5 cm long below the palm). A task step is: a grasp skill, then lift, then place.
Each step lists its parameters, their allowed ranges and what each one means; read the meanings before
blaming a parameter. info.tcp_z is the height the fingertips reached; info.grasp_offset is that height above the
object's bottom. "derived" collects facts computed from those observations (e.g. how far below the top the
fingertips were aimed, how much of the descent was blocked). "nearby_objects" lists other objects close to the item
(height, gap); the hand is ~22 cm long along the direction the fingers close and ~7 cm wide, so a tall object within
~11 cm along that direction can be hit. You see only what the robot could observe: perception before and after (positions, sizes in metres), each skill's
parameters and checks, the finger width, which arm parts touched which objects, and how the grasp was chosen. You do
NOT see the simulator's own verdict. Answer with ONE JSON object and nothing else.

Failure labels (use exactly these names):
- missed_grasp: the fingers closed without anything between them
- weak_grasp: the fingers touched the object but did not hold it
- collision: the hand or fingers hit the object (or something else) before reaching the grasp pose
- unreachable: the arm could not reach the grasp pose, without touching anything
- slip: the object was held at first but fell out while lifting
- lift_check_failed: the lift check failed although the object rose
- drop: the object fell during transport or release and ended on the table off the target
- wrong_place: the object was released but did not end on/in the target, or ended tipped over
- constraint_violation: a "don't disturb / keep upright" rule on another object was broken

Each hypothesis needs ONE test: a single change to replay the very same scene with. If the hypothesis is right, the
step should then succeed. Allowed tests:
- {"type": "skill_param", "skill": "<skill in the trace>", "param": "<its parameter>", "value": <number in range>}
- {"type": "object_param", "param": "<object parameter>", "value": <number>}  (or "factor": <multiplier>)
- {"type": "alternative_skill", "card": "<another grasp card from the library list>"}

Answer format:
{"summary": "<one sentence on what happened>",
 "hypotheses": [{"cause": "<the cause, in terms of the object and the skill>",
                 "predicted_label": "<label>", "confidence": <0 to 1>,
                 "test": {...}}]}          // 1 to 3 hypotheses, most likely first
---
Evidence:
{evidence}
