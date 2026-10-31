# Kintsugi: Implementation Plan (v2)

Rewritten 2026-10-30. Merges the earlier v1 plan (online copy, now out of date:
https://claude.ai/artifact/9ogQiLNZ3fMTFziTtbNsoq), the early "Final Project Plan" draft, and ideas from related work.
CLAUDE.md holds the live status; this file holds the design and the order of work.

## Summary

Kintsugi turns a sentence into a simulated tabletop, a task list and a plan, then runs it on a Franka Panda in MuJoCo.
When a step fails it finds out why with evidence, explores new ways to handle the object, keeps only fixes that pass
held-out tests, and reuses them on objects it has never seen.

Nothing is trained. Every improvement is a skill, parameter or plan change found by search in simulation and verified
there. Laptop CPU plus Meta model API calls. Say "builds and verifies its own skill library", never "learns".

**The demo story:** "Sort the blocks and the bowl onto the trays of the same colour (don't knock over the bottle)."
The blocks go on their trays, the bowl fails, the system shows why, discovers a rim grasp, proves it on bowls it never
tuned on, then handles an unseen cup after a few verification trials.

**Why this can win.** Most robot demos show a scripted success. Kintsugi shows a failure turned into a verified, reusable
skill, and every claim is a measurement with a sample count and an interval:

1. Failure diagnosis matches the simulator's own label in X% of cases, and Y hypotheses were confirmed or rejected by
   one-change replays.
2. Success on the failure cluster goes from X% to Y%, holds on held-out objects, and nothing else regresses.
3. A new object reaches 80% success in N trials when it starts from the library, against M trials from scratch.
4. Supporting: a failure search finds the weak spots in K trials (vs random sampling), and Spark-guided repair search
   beats Optuna alone at equal budget (or honestly does not).

A modest or null result is still a valid result, because it is measured.

## Where we are (2026-10-30)

Built and tested (details in CLAUDE.md): scene builder from parametric templates, cameras and depth perception with
shape descriptors, mink motion, skills with checks, executor, failure labeler, parallel runner, frozen splits for bowls
and cups, the demo task `sort_with_bowl`, and the repair loop (`scripts/explore.py`): failure cluster, geometric grasp
sampler, IK reach filter, families, success map, Optuna, three gates, promotion with lineage. Starting from the seed
library the loop discovers `rim_pinch` from the demo failure (bowl 0/24 to 24/24, held-out 16/16 vs 0/16).

Not built yet: oracle, failure search, investigator, retrieval and transfer, model layer, constraints, web app.

## Objects: templates that keep growing

Objects are **parametric templates** built from simple shapes with `MjSpec`. The language model picks a template and fills
in parameters; code clips them to tested ranges; a deterministic placer lays them out. This keeps on-the-fly variety with
the reliability of a tested library, and lets thousands of trials rebuild scenes in milliseconds.

**Designed for many object types.** Nothing downstream may depend on a template name. Every template declares:

| Field | Meaning | Example (mug) |
| --- | --- | --- |
| parameters + tested ranges | what the model may set | radius, height, wall, handle size, mass, friction |
| features | what perception can find and skills can use | `rim`, `body` (cylinder), `handle` (loop), `base` |
| keypoints / axes (true geometry) | for the oracle, labeler and perception-error measurement only | rim circle, handle centre and axis, up axis |
| goal predicates it supports | how "inside/on/upright" is checked | upright = up axis within 25 deg |
| split bands | train / dev / test ranges | in `experiments/splits.json` |

Perception turns masks and depth into the same feature vocabulary (hollow, rim, handle, top face, elongation, flat), so a
new template that exposes known features is handled by existing skills, and one that exposes a new feature becomes an
honest "no skill fits" case for the explorer. **Adding a template** = builder + feature declaration + ranges + one smoke
check (it simulates stably at range corners) + split bands.

| Template | Status | Features | Brings |
| --- | --- | --- | --- |
| block, ball, cylinder | done | top face, body | sorting, baskets |
| bowl, cup | done | rim, hollow | the repair story, transfer |
| bin, tray, zone | done | container, surface | targets |
| bottle (cylinder + neck) | next | body, neck, tall | constraint obstacle ("don't knock it over"), neck grasp |
| mug with handle | next | rim, handle | transfer with a NEW feature: rim skill matches, handle is a choice |
| plate (thin disc) | later | flat, thin | no pinch fits from the table: needs push-to-edge (a composite skill) |
| bar (long cuboid: tool, rod, remote) | done | elongated, top face | oriented grasps: closing across the long axis; tests the general grasp sampler |
| box with lid | later | lid | lid removal |
| mesh objects (ProjectSim mugs, cans) | optional | via convex decomposition | realism test set at the end |

**Generality: how far beyond the templates it reaches.** Today the system is general within its vocabulary (skills,
search, router and gates never use object names) and rigid at its edges: the object templates, the grasp search space,
the action set and what a repair may change. These close the gaps without any training:

| Improvement | What changes | Effort |
| --- | --- | --- |
| **Richer perception** | descriptor adds principal axes, long/short extents, elongation, flat/thin, later handle detection; router shape groups add "long" | ~0.5 day |
| **General grasp sampler** | a second search primitive placed in the object's own principal frame (offset along each axis, closing angle relative to the long axis, depth, tilt), so oriented and irregular shapes get grasps, not only round and boxy ones | ~1 day |
| **Repairs beyond grasps** | the explorer's patch space adds lift and place parameters (grip force, transport speed, release height), then push and re-grasp; fixes failures after the grasp (thin cylinders tip at release) | 1-1.5 days |
| **Plans as programs** | general task logic (see below) | 1.5-2 days |
| **Unseen-object benchmark** | a frozen set of ~20 objects never used in development (new templates and parameter corners, optional ProjectSim meshes): success before vs after ONE automatic search-and-repair cycle, no object-specific code | ~0.5 day once the above exist |

The benchmark is the answer to "is it general?": "on N objects it had never seen, success went from X% to Y% after
one automatic cycle". Objects composed by the model from primitive parts are deliberately left out for now.

Visual variety (textures, lighting) stays out of scope. Physical variety (size, mass, friction, clutter, table height,
placement) is in scope and is how splits and the held-out perturbation family are made.

## Architecture

```
Browser app (React + Vite + react-three-fiber)   task & scene, run replay, investigator, experiments, results & skill graph
        |  reads saved files; WebSocket only for live progress
Server (FastAPI)                                 starts jobs, streams tallies, serves runs, reports, cards
        |
Agents (model layer)        | Perception                 | Search & evaluation
 task spec, questions,      |  masks (SAM 3 / sim)       |  failure search (capability map)
 refusal, plan programs,    |  depth -> point cloud      |  grasp sampler + IK filter
 diagnosis, patch and grasp |  features, keypoints       |  explorer, Optuna / Spark-guided search
 family proposals, re-rank  |                            |  gates, oracle, router table
        |
Simulation (MuJoCo, CPU)    scene builder, Panda + mink, skill runtime with checks, plan interpreter,
                            oracle expert + failure labeler (true state), parallel runner, recorder
        |
Files                        skill cards (+history), task specs, trial records, reports, poses, MP4s, model cache
```

Requests flow down; results come back up through saved files. Only the oracle, the labeler and the evaluator read true
simulator state; skills see perception only.

## Model layer: Meta models at the end, stand-ins while building

The Meta API key arrives late and credits are limited. Everything that calls a model goes through one client so the
switch is a configuration change:

```
kintsugi/agents/llm.py      ModelClient.complete(job, prompt, schema) -> validated JSON
  backend "rule"            deterministic stand-ins (current specs.py / planner.py); tests and smoke test
  backend "claude_cli"      `claude -p --output-format json` (Claude Code, already logged in on the laptop); dev reasoning
  backend "spark"           Muse Spark via the Meta API, enabled by KINTSUGI_LLM=spark + key
kintsugi/agents/vision.py   segmenter: "sim" masks (bulk), "sam" (SAM 3 via API, key moment frames only)
```

Rules for every model job:

- **Same prompt, same JSON schema, same validator for every backend.** Prompts live in `kintsugi/agents/prompts/` with a
  version. On a schema error the error is fed back once, then the output is rejected with a reason.
- **Cache by hash** of (job, prompt version, input, backend). Re-running an experiment costs no calls.
- **Every record stores the backend and model id**, so results produced with the stand-in are never reported as Spark
  results. Before the demo, the reported runs are re-run on Spark (cache makes this cheap to plan: a dry-run mode counts
  the calls an experiment would make, to budget credits).
- **Call per proposal, never per trial.** A full explore run needs roughly 5-15 model calls; a demo episode 3-6.

| Model | Job | Notes |
| --- | --- | --- |
| Muse Spark | task spec, clarifying question, refusal; plan programs; failure diagnosis; grasp-family names and proposals; patch proposals; failure-search region proposals; retrieval re-ranking | stand-in: Claude via CLI |
| SAM 3 | masks at key moments (start, before grasp, after place, failure frames); hand + object tracking for video imitation | stand-in: simulator masks; report the perception gap on ~50 SAM episodes |
| Muse Wonder (if available) | orchestration of multi-step jobs; repairing invalid specs from validator errors (count repairs) | verify access and capability when the key arrives |
| Muse Code | coding assistant during the build | mention in the submission |
| open-weight models | optional self-hosting if rate limits bite | only if needed |

## From a sentence to a workspace and a plan

1. **Task spec (JSON).** Goal as rules over object properties, objects as templates with parameters, **constraints**
   (`not_disturbed(obj)`, `stays_upright(obj)`, `max_tilt(obj, deg)`, `no_contact(obj)`), success predicates. One
   clarifying question if something essential is missing; refusal with a reason if unsupported (liquids, deformables).
2. **Scene.** The placer puts objects in reach without overlaps; a validator rejects unsolvable layouts.
3. **Oracle check.** A scripted expert with privileged state (true geometry, hand-written reference skills) attempts the
   task. Scenes it cannot solve are discarded, so a later failure always means the robot's own skills fell short.
4. **Plan as a program** (see next section), shown before anything runs and editable.

Success, constraint checks and failure labels are deterministic code, never the model.

### Plans as programs (Code as Policies)

What exists now is *not* code-as-policies: the planner emits a fixed JSON list of skill calls per subtask, and the skills
are hand-written Python functions whose parameters search can tune. Code as Policies means the **model writes the plan
as a short program** against a perception and skill API, with loops, conditionals and arithmetic:

```python
blocks = sorted(find(kind="block"), key=lambda o: o.size, reverse=True)
base = find_one(kind="tray", color="blue")
for b in blocks:
    grasp = choose_grasp(b)                  # router: best verified card for this shape group
    if not pick(b, grasp):
        pick(b, next_best_grasp(b))          # recovery written into the plan
    place_on(b, base if b is blocks[0] else last_placed())
```

Design:

- A restricted interpreter runs it: the AST is checked against a whitelist (our API, numbers, loops with a bound, no
  imports, no attribute access outside API objects), and every API call is recorded, so the trace is the same
  skill-call record the executor already writes. The JSON plan stays as the validated fallback.
- The API exposes perception queries (`find`, `size`, `left_of`, `distance`) and skills, never motor commands.
- Unlocks tasks the JSON planner cannot express: ordering by size, spatial relations ("left of the bowl"), counting,
  conditional recovery. Words-to-numbers edits ("a bit higher") map to parameter overrides.
- **Composite skills are programs too** (Voyager / hierarchical code generation): a repeated successful sequence, or a
  model-proposed helper (`clear_path_then_pick`), is stored as a program card and goes through the same gates.

## Skills

| Layer | Examples | Changes over time? |
| --- | --- | --- |
| Motion | move_to, approach, open, close, retreat (mink IK) | no |
| Skills (cards) | grasps, place, push, recovery; composite program cards | yes: new variants, tuned parameters, new composites |
| Plan | program per task | per task |

**Skill cards** (versioned JSON, old versions kept in `history/`): required features, parameters with ranges, the check
run at the end, lineage (trigger, parent, results, diff), and a **performance table by shape group** filled from every
trial (the router reads it). Grasp variants found by the explorer reuse the `feature_pinch` primitive with tuned values.

| Skill | Status | Purpose |
| --- | --- | --- |
| top_grasp, side_wrap | seed | solid objects |
| rim_pinch | discovered | hollow objects with a rim (bowl, cup, mug) |
| lift, place_in / drop_in | seed | transport, containers |
| place_on / stack | next | stacking tasks; check: still standing after 2 s |
| push | next | move an object into reach or out of the way; base of "push closer" recovery and push-to-edge |
| put_down_and_regrasp | next | recovery when the first grasp pose is poor |
| handle_grasp | with the mug | grasp on the handle feature (an explorer target, not hand-written if avoidable) |
| push_to_edge_then_grasp | later | composite for flat objects |

**Recovery inside a run:** when a check fails, try the next-best verified variant or a bounded recovery (re-grasp, lower
grasp point, push closer) instead of stopping. Report task success with and without recovery.

## Finding, explaining and repairing failures

1. **Failure search -> capability map.** Search over scenario parameters (size, wall, friction, mass, position, clutter)
   for regions where the current library fails, comparing random sampling, Optuna and Spark-guided proposals at equal
   trial budget. A scenario counts only if it is plausible (in ranges, reachable, no overlap) and the oracle solves it.
   Output: a map of success by region, and failure clusters handed to the explorer. This replaces the hand-chosen
   starting failure.
2. **Locate.** The step whose check failed; the simulator label (missed_grasp, collision, slip, drop, wrong_place,
   constraint_violation, unreachable).
3. **Explain.** The model reads the frames around the failure, contact log, skill parameters and percepts, and proposes
   ranked causes. Each goes into the **evidence ledger** as a hypothesis.
4. **Test.** One-factor counterfactual replays: same seed, one thing changed (grasp depth, approach side, force, object
   friction). Flip = confirmed, no flip = rejected. Diagnosis accuracy is scored against simulator labels.
5. **Explore.** If no variant can work, sample candidates on the perceived point cloud (gripper fit, finger and hand
   clearance) + IK filter, grouped into families by feature and closing direction; the model may name families and
   propose extra ones. Candidate trials run on the isolated sub-task (done).
6. **Tune.** Optuna on the best families under physical perturbation, plus a **Spark-guided evolutionary search**
   (Eureka-style reflection on trial statistics: propose K patches, keep the best, repeat). Same budget, compared.
7. **Gate.** Promote only if it (a) fixes the cluster, (b) does not regress the regression set, (c) holds on held-out
   objects. Families that cannot be told apart compete through the gates; best held-out wins (done).
8. **Non-grasp repairs.** Not every fix is a grasp: a constraint violation (hand hits the bottle) is repaired by approach
   side, plan order or a push. The patch space includes plan edits and rules ("approach from the side with most free
   space"), gated the same way.

## Reusing skills on new objects

- **Retrieval** by perceived features (hollow, rim radius, wall slope, handle, size), not names; the model re-ranks and
  explains ("rim, like the bowl").
- **Router table:** success per (variant, shape group). Report task success for best-single-variant, random choice,
  model choice without the table, router, and the hindsight oracle (best variant per object, known after the fact):
  share of the gap closed.
- **Verification:** 5-10 trials before trusting a transferred skill; on failure, the explorer starts from the transferred
  parameters.
- **Transfer metric:** trials to 80% success, transfer vs a full exploration from scratch on the new object. Note that
  rim_pinch already gets cups 16/16 untuned, so the mug (new handle feature) and test-split small cups are the
  interesting targets.

## Evaluation rules and metrics

- **Oracle filter** on every reported failure.
- **Frozen splits** (`experiments/splits.json`): train (tuning), dev (gate), test (run once). Plus one **held-out
  perturbation family** never used in tuning (e.g. very low friction, or clutter around the item), and a regression set.
- **Reproducibility:** seed, scene spec, card versions, code version (git commit), config hash, model backend + id.
- **Null results are reported**, including rejected patches and a router that does not beat the best single skill.

| Measure | Compared against | Reported as |
| --- | --- | --- |
| Diagnosis accuracy | simulator labels | % top cause matches |
| Hypothesis tests | one-change replays | confirmed vs rejected counts |
| Repair effect | starting skills, per cluster | before/after with 95% interval |
| Held-out generalization | never-tuned objects and the perturbation family | success rate |
| No regressions | regression set | change per accepted patch |
| Failure search | random vs Optuna vs Spark, equal budget | failures and distinct clusters per 100 trials |
| Repair search | random vs Optuna vs Spark-guided, equal budget | success reached per 100 trials |
| Router | best single, random, model w/o table, hindsight oracle | share of gap closed |
| Transfer | from scratch | trials to 80% |
| Generality | unseen-object benchmark, before vs after one automatic cycle | success on never-seen objects |
| Recovery | no recovery | task success |
| Perception gap | ground-truth features vs SAM | success difference; rim/size error in mm |
| Throughput | | trials per minute (laptop: ~150/min short episodes, 8 workers) |

## Web app

Physics never runs in the browser: it replays saved pose logs (30 fps) with Three.js; MP4 as fallback. Every statement is
labelled **measured / hypothesis / confirmed / rejected**.

| Panel | Shows |
| --- | --- |
| Task & scene | prompt box, clarifying question, spec with constraints, plan program (editable), 3D scene, editable object parameters + re-run |
| Run replay | 3D replay with timeline, skill trace (failed step red), camera view with masks, subtask progress (2 of 3) |
| Investigator | simulator label, frames, evidence ledger, one-change replays |
| Experiments | live per-candidate tallies while jobs run (or replayed), capability map, success map around the object |
| Results & skills | success with intervals, gates, **promotion diff** (parameters before/after + evidence), rejected patches, skill graph / lineage tree with per-group success, export JSON/CSV |

## Build order

Finish each thin path before adding breadth; keep the smoke test green and under two minutes.

| # | Build | Done when |
| --- | --- | --- |
| 1 | done: sim, perception, skills, executor, runner, splits, demo task, repair loop | rim_pinch discovered from the demo failure |
| 2 | done: **Oracle** + git + code version in records | every reported failure is oracle-filtered |
| 3 | done: **Model layer** (`agents/llm.py`, rule + claude_cli backends, cache, schemas) | one job (task spec from prompt) runs on both backends with identical JSON |
| 4 | done: **Failure search + capability map** | the bowl cluster is found by search from the seed library, random vs Optuna vs model compared |
| 4b | done: **Search -> explorer handoff + router**; repair the clusters the search found on the current library (tall cylinders: new card; narrow cups: routing change) | a second repair round, started by the search, passes the gates |
| 4c | done: **Richer perception + bar template + general grasp sampler** | the search finds the bar weakness and the explorer repairs it with the new primitive |
| 4d | done: **Repairs beyond grasps** (lift/place parameters) | thin-cylinder transport/placement failures repaired through the gates |
| 5 | done: **Investigator**: ledger, diagnosis job, one-change replays | one hypothesis confirmed and one rejected from saved logs; diagnosis accuracy on 20+ failures |
| 6 | done: **Constraints + bottle**, non-grasp repair (clear-side patch) | "don't knock over the bottle" fails repeatably (20-80%) and a repair passes the gates |
| 7 | done: **Retrieval, router, transfer** (mug with handle: 10 vs 924 trials) | transfer vs scratch numbers |
| 8 | **Plan programs** (restricted interpreter, plan job) + push + recovery | a stacking-by-size task runs from a prompt; recovery numbers |
| 9 | **Web app** over saved files | full demo replays offline |
| 10 | **Spark + SAM switch-over** when the key arrives; re-run reported experiments; perception gap on ~50 episodes | headline numbers produced with Meta models |
| 10b | **Unseen-object benchmark** (frozen set, one automatic search-and-repair cycle) | before/after on never-seen objects |
| 11 | **Test split, once**; freeze numbers | results file frozen |
| 12 | Demo video, write-up | submitted |

If time runs short: drop 8 to plan programs only, keep the router as a table, keep 6 but with the bottle optional in the
demo.

## Optional improvements, ranked

| Rank | Improvement | Adds | Start when |
| --- | --- | --- | --- |
| 1 | Learn from a human video: SAM tracks hand and object, MediaPipe pinch vs wrap, contact segments reach/grasp/lift/move/release, the model maps them to a plan program, sim retries | "watches once, then does it" | core loop done |
| 2 | Automatic curriculum: after a fix, generate harder variants (thinner, slipperier, tighter, more clutter) and show where the skill stops working | difficulty ladder | gates work |
| 3 | Self-proposed tasks (RoboGen/GenSim style): the model proposes tasks from template affordances, the oracle keeps solvable ones, they grow the regression set | breadth without hand-writing tasks | model layer + oracle |
| 4 | Composite skill mining from repeated successful sequences | new skills from the library's own history | plan programs |
| 5 | Keypoint constraint placement (ReKep-style: the model writes constraints on perceived keypoints, an optimizer solves the pose): "handle facing left", "rim up" | richer goals, no training | mug template |
| 6 | Realistic mesh objects (ProjectSim) as a final test set | realism | transfer works |

Out of scope: voice, visual randomization, any trained policy (ACT, VLA, RL), self-hosted models unless rate-limited,
real robot (future work only).

## The 3-minute demo

1. **0:00 Prompt.** "Sort the blocks and the bowl onto the trays of the same colour, don't knock over the bottle." The
   spec, the plan program and the scene appear; one clarifying question if needed.
2. **0:20 Partial success.** Blocks on their trays; the bowl fails. Run replay marks the failed step.
3. **0:40 Where and why.** Capability map shows the weak region (hollow, wide objects). The investigator shows the frames,
   the ledger, one confirmed cause ("wider than the gripper opening") and one rejected one.
4. **1:10 Repair.** Candidates around the rim, the success map fills in live, families compete, gates run. Promotion diff
   with before/after, regression and held-out numbers.
5. **1:50 Transfer.** An unseen mug: "rim, like the bowl"; a few verification trials; success. Skill graph shows the new node.
6. **2:20 Video (if built).**
7. **2:45 Close.** The headline numbers.

Replays saved runs. At most one short live run, only after three clean rehearsals. Backup video recorded.

## Risks

| Risk | Early sign | Response |
| --- | --- | --- |
| Meta key arrives late, credits limited | no key until the end | model layer + cache + call budgeting; dev on Claude via CLI; re-run reported numbers on Spark at the end, label which backend produced each number |
| Stand-in and Spark behave differently | Spark output fails validation or plans differ | same schemas and validators; feed errors back once; prompts kept model-neutral; keep rule backend as fallback |
| No repeatable failure for a new feature (bottle) | 0% or 100% | adjust placement / sizes once, then simplify |
| Optuna adds nothing | saturates at once (seen on the bowl) | report it honestly; harder families or perturbation; Spark-guided vs Optuna stays a measured comparison |
| Router shows no gain | preconditions already route | report the null result; keep it a table, not a headline |
| Physics trouble with new templates | jitter, fingers pass through | smoke check at range corners per template; thicker walls, contact settings as for the bowl |
| Parallel slowdown | more workers, fewer trials/min | measured: cap at 8 workers on the laptop, BLAS threads pinned |
| Interface takes too long | panels lag the loop | read-only views over saved files; skill graph can fall back to a list |
| Overclaiming | "learns" | "builds and verifies its own skill library" |

## Related work (check each citation before submission)

| Work | What it does | What we take / how we differ |
| --- | --- | --- |
| Code as Policies | LLM writes robot programs over perception and control APIs; hierarchical generation of helper functions | plans as restricted programs over our skill API; composite skills as generated helpers. We add verification, repair and gates |
| Voyager | lifelong agent with a code skill library, automatic curriculum, iterative prompting with environment feedback, execution errors and self-verification | skill library of verified cards and programs; curriculum (optional); our feedback = sim trace, validator errors, skill checks |
| RoboGen | proposes tasks, generates scenes, decomposes into subtasks, learns skills with RL / motion planning; success judged by humans | self-proposed tasks (optional); per-subtask choice of skill type. We use automatic checks and no learning |
| RoboTwin 2.0 | MLLM-generated expert code refined in simulation; annotated objects (functional points, axes); task-level randomization | template features/keypoints; physical randomization (clutter, table height) as held-out family. We do single-arm skill repair with held-out proof |
| GenSim | LLM-generated simulation tasks, verified by execution | oracle-filtered task generation |
| Eureka | LLM-written reward code improved by evolutionary search with reflection | the search pattern, applied to patches instead of rewards |
| AW-0 (Agentic World Action Models) | code + video + action generation; recurring routines consolidated into skill libraries (abstract only seen) | same skill-library direction without any trained model |
| ReKep, VoxPoser | training-free: LLM/VLM writes constraints or value maps solved by optimization | keypoint-constraint placement (optional) |
| Inner Monologue | closed-loop language feedback with success detectors | skill checks + diagnosis as feedback |
