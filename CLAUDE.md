# Kintsugi — context for Claude Code

Meta AI hackathon project. Goal: win. Read this whole file before changing code.

## The idea in one paragraph
A Franka Panda in MuJoCo turns a typed task ("sort the blocks by colour", "put the balls in the basket",
"put the bowl on the tray") into a generated tabletop scene, a task spec and a plan of skill calls, runs it,
and when a step fails it explains why with evidence, explores new grasps, keeps only fixes that pass
held-out tests, and reuses them on unseen objects (bowl skill -> cup). **No training, no GPU**: every
improvement is a skill or parameter change found by search in simulation and verified there. Laptop CPU +
Meta model API calls only. Say "builds and verifies its own skill library", never "learns" or "trains".

## Demo story (build everything backwards from this)
1. Prompt: "Sort the blocks and the bowl onto the trays of the same colour" (later: "...don't knock over the bottle"). Scene + subtasks + plan shown before running.
2. Blocks succeed. Bowl fails with top_grasp (bowl wider than gripper opening) -> label "missed_grasp".
3. Investigator: failed check, frames, one hypothesis CONFIRMED by a one-change replay.
4. Explore: grasp candidates around the rim, parallel trials, success map, Optuna tuning, gates (fixes failure cluster, no regression on earlier tasks, holds on held-out bowls). New variant promoted with a diff.
5. Transfer: unseen cup -> library match by shape ("rim, like the bowl") -> 5-10 verification trials -> success.
6. Optional: learn from a human video (SAM tracks hand+object, MediaPipe pinch vs wrap, Spark maps steps to skills, sim retries).
Headline numbers: diagnosis accuracy vs simulator labels; success before/after repair incl. held-out; trials to success with transfer vs from scratch.

## Key decisions (do not reverse without asking the user)
- **Robot**: Franka Panda from MuJoCo Menagerie (`scripts/fetch_assets.py`). Gripper servo stiffened in `sim/scene.py` (kp 1500, finger joint solref 0.004, armature 0.05, left/right finger contact excluded, all contacts solref 0.004) — without this, fingers pass through thin walls.
- **Objects**: parametric templates built with MjSpec from simple shapes (block, ball, cylinder, bowl = disc + ring of tilted boxes, cup, bin, tray, zone). The language model picks templates + parameters; code clips them to tested ranges and a deterministic placer lays them out. ProjectSim assets (USD, needs conversion) only as an optional late realism test set.
- **Skills are defined on object FEATURES (rim, top face, walls), never object names.** Three layers: motion (mink IK, fixed) / grasp+place skills (versioned skill cards in `skill_library/*.json`, parameters + ranges + check + lineage) / task plan (Spark).
- **Privilege split**: skills see only perception descriptors + proprioception (TCP pose, finger width). True sim state is used only by the scripted expert, goal checks and failure labeler (`sim/monitor.py`, `Perception(mode="ground_truth")`).
- **SAM is called only at key moments** (~3-5 per episode): tag objects at start, optionally before a grasp, after a place, on failure frames. Between calls, track by depth clusters. Bulk exploration trials use simulator masks (`mode="sim"`); real SAM on demo episodes + a measured sample (~50) to report the perception gap.
- **Meta models**: Muse Spark = task spec / clarifying question / refusal, plan programs over the skill API, failure diagnosis, grasp-family proposals, patch proposals, failure-search proposals, retrieval re-ranking. SAM 3 = masks. **The Meta key arrives only at the end (limited credits).** All model calls go through one client (`kintsugi/agents/llm.py`, planned) with backends `rule` (deterministic stand-ins, tests), `claude_cli` (dev reasoning via `claude -p --output-format json`, installed and logged in on the laptop) and `spark` (enabled by `KINTSUGI_LLM=spark` + key). Same prompts, schemas and validators for every backend; switching = config only. Cache every call by hash of (job, prompt version, input, backend); store backend + model id in every record and never report stand-in results as Spark results. Call models per proposal, never per trial.
- **Many object types later**: nothing downstream may depend on a template name. Each template declares parameters + ranges, the features perception can find (rim, handle, top face, body, flat...), true keypoints (privileged, for oracle/labeler only), supported goal predicates and split bands. A new feature with no skill is an explorer case, not a hand-written special case.
- **Plans are programs (Code as Policies, built)**: the model writes a short Python program over a perception + skill API, run by a restricted AST-whitelisted interpreter on the same Episode step methods as the fixed plan. The fixed plan stays as the bulk-experiment engine (no model call per scene), the fallback, and the baseline.
- **Who writes what (do not blur)**: the model writes task specs, plans/programs, diagnoses and proposals, and may PROPOSE new primitives (parameterized motion code + ranges + check). It never writes trusted skills directly: simulation search finds the numbers and the gates decide. Composition helps only when the motion exists in some primitive; a missing motion (push; later tilt-while-holding for pouring) is a model-written primitive, searched and gated like feature_pinch/axis_pinch.
- **Platform vision**: reuse what is known, fix what fails, create and verify what is missing; limited by simulation fidelity, code-checkable success, the motion API and search cost. Liquids/pouring out of scope for the hackathon (mechanism planned, particle simulation not).
- **Web app**: React + Vite + Three.js (react-three-fiber). Physics never runs in the browser: it replays saved pose logs (`media/*_poses.json.gz`, every body pose at 30 fps + static geom description; Panda meshes from Menagerie). MP4 as fallback. Four panels: Task & scene / Run replay with skill trace (failed step red) / Failure investigator (measured vs hypothesis vs confirmed vs rejected) / Results & skill lineage tree. FastAPI backend serving saved files.
- **Run settings**: parallel workers default = CPU cores - 1; JSON record for every trial; poses + MP4 for failures and 1 in N successes (flags).
- **Out of scope**: voice, lighting/texture randomization, any trained policy (ACT, VLA), real robot.

## Current status (2026-10-30)
Done and tested (smoke test 8/8, ~8 s on the Windows laptop):
- Scene build, cameras (front/top/side), colour/depth/segmentation, depth -> point cloud -> descriptor (centre, size, hollow via deep-centre or ring test, rim radius, wall slope, yaw). Centre error 3-7 mm.
- Motion with mink + gravity-sag correction (< 2 mm). Skills: top_grasp, rim_pinch, side_wrap (untested), lift, place_in/drop_in with container slots.
- Executor (perceive -> plan -> skills -> checks -> labels), parallel runner, recorder.
- Results: sort_by_color 8/8 (48/48 blocks); balls_in_basket 15/16; bowl_on_tray rim_pinch 16/16, top_grasp 0/8 (missed_grasp); cup_on_tray rim_pinch 8/8. ~1.5-5 s per episode without video.
- **Demo task** `sort_with_bowl` ("Sort the blocks and the bowl onto the trays of the same colour": red/green
  blocks, blue bowl from the bowl split, three trays). Seed library: 0/24 tasks, blocks 48/48, bowl fails every time
  (missed_grasp 22, collision 2). Current library: 24/24 train, 24/24 dev.
- **Repair loop (step 1) done, starting from the demo**: `python scripts/explore.py sort_with_bowl --from-seed`
  (~4-5 min, 392 trials, 8 workers). Seed library = top_grasp, side_wrap, lift, place_in. The hand-written rim
  pinch lives in `skill_library/reference/rim_pinch_handwritten.json` (not active; comparison only).
  Loop (`kintsugi/search/`): before run on the full task -> failure cluster = (item, grasp skill) with its labels
  (top_grasp on bowl_blue, 24/24: missed_grasp 14, collision 10) -> candidate trials on the ISOLATED sub-task
  (`make_task(..., isolate_obj=)`: same bowl and its tray only; planner `overrides["grasp_for"]` forces the candidate
  for that item) -> 800 `feature_pinch` candidates (half uniform, half anchored on perceived surface points;
  `feature_pinch` = generic grasp in feature coords: edge_offset, angle, depth, close_yaw, tilt) -> gripper fit on
  the point cloud + mink IK reach -> ~128 survive, families rim_pinch / rim_oblique / rim_clamp -> 4 candidates x 4
  seeds per family (rim_clamp 0/16) -> Optuna per family under friction/mass perturbation, early stop when
  saturated, top 4 confirmed on fresh seeds -> gates on the FULL task. Families whose intervals overlap the
  leader's all go through tuning + gates; best held-out wins (rim_pinch 16/16 beat rim_oblique 14/16).
  Promoted `skill_library/rim_pinch.json` v1 (impl feature_pinch, close_yaw -5 deg, depth 13 mm; lineage has
  trigger, parent, results, diff). Gates: bowl 24/24 on the failed trials (was 0/24), held-out dev bowls 16/16 vs
  0/16, no regression on sort_by_color / balls_in_basket / single_block.
  Report for the UI: `runs/20261030-214545_explore_sort_with_bowl/report.json`.
  Also works on `bowl_on_tray` directly. Verification: dev bowls 24/24; cups 16/16 with the bowl card, untuned.
- Executor fixes found via the demo: `go_ready` retreats straight up before opening when the fingers closed on
  nothing (opening inside a bowl wedged the fingers against its walls and carried it into the next block); a grasp
  that stops short while touching the object is labelled `collision`, not `unreachable`. Known rare cascade: a
  bowl at the edge of reach can still get dragged into a block after a failed top grasp (1 seed in 16).
  Honest caveat: Optuna saturates at once on this bowl (the best geometric candidate already succeeds), so tuning
  mostly confirms; it matters only if a harder object family is used.
- **Frozen splits (most of step 2)**: `experiments/splits.json` (bowl + cup parameter bands for train / dev / test,
  seed blocks 0 / 100000 / 200000). `bowl_on_tray` and `cup_on_tray` draw objects from the split their seed belongs
  to. dev = held-out objects for the gate; test = final evaluation, run once (not run yet).
- Windows: more than 8 parallel renderers slows everything down; default workers = min(cores-1, 8), BLAS threads
  pinned to 1 per worker. Scripts that use the pool must be real files with `if __name__ == "__main__"` (spawn).
  Conda env on this PC: `C:/Users/yahya/anaconda3/envs/kintsugi` (Python 3.11, MuJoCo 3.17, Optuna 5).

- **Oracle done** (`kintsugi/oracle.py`): same scene, ground-truth perception, seed + reference skills (never
  discovered cards). `run_batch(..., oracle=True)` / `run_task.py --oracle` adds `record["oracle"]` and a summary
  (solved / valid_failures / unsolvable). The explorer's failure cluster keeps only valid failures (item step failed,
  oracle solved it); dev gate scenes are oracle-checked. Caveat: on the bottle variant the oracle itself knocks the
  bottle over in ~2/16 scenes (those are excluded as unsolvable); give the oracle retries before constraint
  results become a headline.
- **Trial records** now carry `code_version` (content hash of source + cards + splits, git commit when one exists)
  and `config_hash`. `git init` done with `.gitignore` (runs/, cache/, assets/, baselines); nothing committed yet.
- **Model layer done** (`kintsugi/agents/`): `ModelClient(backend)` with `rule` / `claude_cli` / `spark`; jobs are
  registered (`agents/jobs/`), prompts versioned in `agents/prompts/*.md`, JSON schema (jsonschema) + semantic
  validator (also checks the scene can be placed), errors fed back once then `ModelRejected`, cache + call log in
  `cache/llm/`, `KINTSUGI_LLM_DRYRUN=1` counts calls a paid backend would get. First job `task_spec` (ok / clarify /
  refuse, constraints, obstacle role); the template catalogue in the prompt is generated from `templates.py`.
  `python scripts/prompt_task.py "<request>" [--backend claude_cli] [-n 8]`. Spark backend assumes an
  OpenAI-compatible `/chat/completions` (META_API_BASE, META_API_KEY, META_MODEL): verify when the key arrives.
  claude_cli (logged in once with `claude` + `/login` in a normal terminal; the desktop session's env vars are
  stripped when spawning it) tested: bowl/tray spec, pouring refused, "tidy up" -> clarifying question, demo with
  bottle -> 7 objects + constraint; 3-8 s per call (sonnet). Prompt v2 added sizing guidance after v1 needed one
  repair ("table too crowded"); meta records `repair_errors`. A novel request ("three balls in the basket, cup on the
  tray, leave the can alone") ran end to end: 7/8, 2 scenes unsolvable by the oracle.
- **Constraints** `stays_upright`, `not_disturbed` checked by the monitor after every sub-task (label
  `constraint_violation`, pinned to the step) and in final success. With rim_pinch the bottle variant succeeds 15/16:
  the bottle is rarely hit yet; step 5 must place it where it matters.

- **Failure search done** (`kintsugi/search/failure_search.py`, `scripts/failure_search.py`): probe scenes (`probe`
  task: one item of any template at a chosen position via `layout_fixed`, to a tray / bin for balls), plausibility
  gate (ranges, reach, placer), oracle on failures only (`oracle="on_failure"`), strategies random / Optuna TPE /
  model (job `search_proposal`, rule bandit stand-in or Claude). Clusters = (template, grasp skill) with labels.
  Results, 160 trials per strategy, model = Claude (10-11 calls, 0 repairs, ~11 s each):
  seed library: valid failures per 100 random 52.5 / Optuna 90.6 / model 78.8; distinct clusters 5 / 4 / 6; bowl,
  cup, tall cylinder found within the first 2-14 trials by every strategy (report runs/20261030-225954_failsearch).
  current library: bowls/cups no longer fail (independent confirmation of the rim_pinch repair); remaining clusters
  cylinder/top_grasp (92% of solvable cylinder trials, palm hits the top of objects taller than ~9 cm) and
  cup/side_wrap (49%: narrow cups are routed to side_wrap before rim_pinch), plus rare block/ball cases (1-4
  failures, maybe noise); per 100: 18.1 / 60.0 / 63.1 (report runs/20261030-230646_failsearch). Optuna maximizes
  yield, the model finds more distinct clusters; one run each, variance not measured yet.
- Oracle now retries: default grasps, a shallow top grasp (privileged expert answer to tall objects), then each other
  grasp preferred; solvable if any attempt succeeds.
- Explorer regression baselines are cached by library + code_hash + seeds.

- **Router** (`kintsugi/skills/router.py`): among cards whose preconditions fit, the planner picks the best
  measured success on the object's shape group (hollow/solid, narrow/wide for the gripper, short/tall at 8.5 cm),
  score (k+1)/(n+2), no evidence = 0.5, ties keep preference order (seed first). Evidence lives in
  `skill_library/performance.json` (not in cards; `load_cards` skips non-card JSON). Each subtask records its
  `routing` (group + scores). The oracle routes with empty evidence (stays independent).
- **Search -> explorer handoff + second repair round done**: `explore.py --from-search <report> --cluster <key>` runs
  the explorer on region-sampled probe scenes (`probe(region=...)`; held-out = interleaved bands of the region's first
  shape parameter, train bands 1+3, dev bands 2+4). New phase `existing`: other fitting cards are tried first; if one
  works the repair is a gated ROUTING change (evidence written), else exploration proposes a new card whose gate
  carries routing evidence. Regression set = fixed tasks + every earlier repair's trigger (named task or probe
  region), so repairs protect each other.
  Results: narrow cups -> routed to rim_pinch (existing 10/10; gate 16/17 on failed trials, held-out 16/16 vs 7/16;
  89 s, runs/20261030-232517_explore_probe_cup). Tall cylinders -> side_wrap 0/10, explored: centre_grasp 16/16 vs
  edge_pinch 5/16, edge_oblique 3/16, edge_clamp 0/16; promoted `skill_library/centre_grasp.json` v1 (grips 17 mm
  below the top instead of half height); gate 20/22, held-out 16/16 vs 0/16, regression incl. sort_with_bowl and the
  cup routing region (188 s, runs/20261030-232707_explore_probe_cylinder).
  A follow-up failure search caught a regression the gates missed: v1 required only hollow=False, so rotated 6 cm
  blocks (perceived wider than top_grasp's 7.2 cm limit, which used to fall back to top_grasp) were routed to it and
  failed (block/centre_grasp 36 failures). Fix: promoted cards now get preconditions from the perceived size of the
  cluster's objects (min/max width and height + 1 cm); `--exclude centre_grasp` redid the repair -> v2 (width
  2.4-7.2 cm, height 7.8-16.8 cm; runs/20261030-234603_explore_probe_cylinder); lineage records `requires_diff`.
  Search after the fix (runs/20261030-234922_failsearch): valid failures per 100 random / Optuna / model
  4.4 / 18.8 / 17.5 (before round 2: 18.1 / 60.0 / 63.1). Remaining: cylinder/centre_grasp 29% (thin cylinders,
  radius 1.5-2.5 cm: grasp holds, then `drop` in transport or `wrong_place` = tips over at release) -> needs
  non-grasp patches (lift/place parameters: grip force, transport speed, release height); block/top_grasp ~9%
  (collision, large blocks), rare cup/ball cases.

## Next steps, in order (build order in docs/implementation_plan.md)
Done: repair loop, splits, demo task, oracle, model layer, constraint checks, failure search, router, second repair
round started by the search (see status).
3b. **Generality**, first half done: perception descriptor adds `axis_yaw`, `extent_major/minor`, `elongation`,
   `flat` (bar ~6, round objects ~1.0); router groups add "-long" (elongation > 2). New `bar` template (long cuboid,
   any yaw; in PROBE_TEMPLATES; probe scenarios carry `yaw`). New search primitive `skill_library/primitives/
   axis_pinch.json` (`axis_grasp_pose`: along/across fractions in the principal frame, closing angle vs the short
   axis, only flipped by 180 deg); primitive cards list `search_params`; the explorer samples, filters, names
   families (`axis_<centre|end>_<across|along|oblique>`) and tunes across ALL primitives. Fixes found on the way:
   the oracle could not grasp rotated bars (top_grasp folds yaw to +-45 deg) -> extra oracle attempt with the
   axis_pinch defaults; the gripper-fit check rejected grasps the skill would clamp to MIN_TCP_Z (flat objects) ->
   threshold aligned; region held-out bands now interleave inside the tuned span (5 bands: train 1,3,5 / dev 2,4).
   Result: focused search (`--templates bar`) found bar/top_grasp at 63%; the explorer promoted
   `axis_centre_across` v1 (axis_pinch; axis families 15/16 or better vs feature_pinch centre_grasp 3/8; gate 10/11,
   held-out 16/16 vs 9/16; runs/20261031-003922_explore_probe_bar). Known quirk: an item's step can be marked failed
   (a step check) while the final goal holds, so item vs whole-task counts can differ slightly.
   Verification search on all 6 templates (runs/20261031-004905_failsearch): bars 47/47 solvable trials succeed;
   valid failures per 100 random / Optuna / model 5.0 / 21.9 / 13.1; remaining clusters cylinder/centre_grasp 24%
   (drop 35, wrong_place 15: after the grasp), block/top_grasp 12% (collision), rare bowl/cup rim_pinch (2 each).
   The model strategy now falls back to the rule stand-in for a batch if a proposal is rejected twice (recorded as
   `fallback_after_rejection`); proposal ranges get 1% rounding tolerance and are clipped when sampled.
   Still to do: repairs beyond grasps (lift/place parameters) for thin cylinders; unseen-object benchmark (10b).
   Objects composed from parts: not now (user).
4. ~~**Investigator**~~ done: `kintsugi/investigate.py`, `scripts/investigate.py --from-search <report> --library
   seed|current [--backend claude_cli]`. Exact replay (`run_episode(positions=...)`; records now carry `task_spec`,
   per-step `observed` = perception before/after, moved, height_ratio, distance to target, and `contacts` = arm part
   -> object) -> evidence (observable only: perception, skill params with ranges AND meanings (cards now have a
   `meaning` per param), checks, finger width, contacts, routing, plus `derived` facts like fingertips aimed below
   top / descent stopped short / tipped over; never the simulator label or true state) -> job `diagnose` (prompt
   v3; rule stand-in) -> up to 3 causes each with ONE test (skill_param / object_param / alternative_skill) ->
   replays on the same layout: confirmed / rejected / inconclusive (change clipped by a range; stays a hypothesis
   in the ledger) -> ledger (measured / hypothesis / confirmed / rejected) in runs/<id>/ledgers/.
   Results (Claude sonnet): seed-library failures (16): top-1 label 87.5% (rule stand-in 62.5%), 6 causes confirmed
   (tall cylinder "fingertips aimed 7 cm below the top, fingers are 5 cm" 3/3), bowl/cup "wider than the opening"
   untestable with one change (structural); runs/20261031-011504_investigate. Current-library failures (19): top-1
   68%, 14/19 cases with a confirmed cause: thin cylinders "released too high, tips over" confirmed with
   place_in.clearance = 0 in 6/9; bowl rim_pinch slip "grasp too shallow" 2/2; large blocks' cause still unknown;
   runs/20261031-011712_investigate. Evidence quality mattered more than the model: without param meanings and
   derived facts Claude scored 56% and blamed hover height.
4b. ~~**Repairs beyond grasps**~~ done: parameter PATCHES per (skill, shape group) in `skill_library/patches.json`
   (`router.load_patches/patch_for`; planner applies them to the item's group, explicit overrides win; the oracle
   uses none). Explorer phase `patch` (after `existing`, when most cluster labels are post-grasp or hints are given):
   candidates = each confirmed skill_param fix from the investigator (`explore.py --hints <investigate report>`)
   plus the best Optuna patch over place_in.clearance/speed, lift.speed, grasp grip_force/depth; all re-checked on 8
   fresh perturbed objects; smallest best wins; gated (held-out n=48 for patches). Post-grasp clusters skip grasp
   exploration if no patch passes. Gate statistics changed (pre-declared, applied to all gates): before keeps
   collecting until >= 10 valid failures (max 96 trials); cluster and held-out gates use McNemar's exact paired
   test, one-sided alpha 0.05 (held-out also needs >= 70% success), p-values in reports. The old rule
   (non-overlapping Wilson intervals, ~p<0.005) rejected the cylinder patch at 43/48 vs 32/48.
   Result: thin cylinders -> patch place_in.clearance 0.02 -> 0.0 for solid-narrow-tall, source = investigator
   (8/8 on fresh objects vs Optuna's 7/8); gate cluster 11/12, held-out 43/48 vs 32/48 (p = 0.0017), regression
   pass (runs/20261031-013109_explore_probe_cylinder). Full chain: search -> investigate -> patch -> gates.
   Slipping bowls (rim_pinch, 2 failures in ~50 bowl trials) are too rare to gate at this budget: not repaired.
4c. **Constraints ("don't knock over the bottle") done**: grasp descriptors now carry `neighbors` (other perceived
   objects); `top_grasp` has a strategy switch `clear_side` (0/1, default 0): on a square top pick the 90-deg
   orientation whose ~22 cm hand stays farther from tall neighbours, then slide the grasp point up to 1.5 cm away.
   Probe scenarios may carry `obstacle` {dist, angle, radius, height} -> a cylinder "bottle" pre-placed near the
   item + constraints stays_upright/not_disturbed (`failure_search.py --obstacle`; clusters carry obstacle ranges;
   region probes reproduce them). Oracle got a clear_side attempt. `isolate()` now keeps constraint obstacles
   (it used to drop the bottle while keeping its constraints, so every candidate trial crashed). Grasp labeler: any
   arm contact while short of the pose = collision (was "unreachable" when the bottle, not the item, was hit).
   Explorer patch dims include a grasp card's [0, 1] strategy switches. Held-out gate rule (pre-declared, all gates):
   if held-out before >= 80% the change must HOLD (no significant worsening, reversed McNemar), else IMPROVE.
   Chain: focused search (blocks + bottle) 29% failure (runs/20261031-015434_failsearch) -> investigator (after
   adding `nearby_objects` and fixing the misleading "wider than opening" fact to use the narrow side) top-1 label
   9/10, 8/10 confirmed "hand hit the bottle", fix clear_side=1 (runs/20261031-015813_investigate) -> patch
   top_grasp.clear_side=1 for solid-narrow-short + solid-wide-short; gate cluster 11/14, held-out 44/48 vs 35/48
   (p = 0.002), regression pass (runs/20261031-021559_explore_probe_block). Known perception limit: a bottle right
   in front of the item can hide it (object_not_perceived).
4d. **Transfer done**: new `mug` template (cup + C-handle of 2 arms and a bar, any yaw; in PROBE_TEMPLATES).
   Perception fix found on the way: the object centre came from the bounding box of ALL points, so the handle
   dragged it sideways and the hollow test failed for some handle directions (3/8 mugs treated as solid -> top_grasp);
   `describe()` now takes the centre (and the radius for the hollow tests) from the top band (rim / top face):
   mugs 8/8, named tasks 8/8 each, smoke OK. Job `retrieve` (prompt + rule stand-in = router evidence) picks a
   library skill from perceived features, candidate descriptions, preconditions, evidence per group and
   "found_on". `scripts/transfer.py <template>` = perceive 4 new objects -> retrieval -> verification (10 trials,
   test-split seeds, chosen skill forced) -> from-scratch baseline (explorer from the seed library, dry run).
   Mug result (Claude): retrieval "rim_pinch - rim and hollow-wide-tall match, like the bowl_blue where this skill
   was found"; verification 9/10 -> trusted; from scratch: promoted after 924 trials (580 search).
   HEADLINE: 10 trials with the library vs 580 search trials (924 total) from scratch
   (runs/20261031-022103_transfer_mug).
4e. **Plan programs (Code as Policies) done**: `kintsugi/executor.py` is now an `Episode` with step methods
   (begin / grasp / lift / place / finish); the fixed plan (`run_plan`) and programs share every skill call, check,
   label and evidence. `kintsugi/program.py`: API find / find_one (Obj: id kind color role x y size height top
   hollow long), pick(obj, grasp=None) (router + patches), place(target) (containers, trays, or another object =
   stacking; shared-container spots planned from the goals like the planner), place_at(x, y), next_best_grasp,
   holding, distance / left_of / right_of / in_front_of / behind, say. Safety: AST whitelist + known-name check
   (`check_program`, also the job validator), line budget 50k, 40 pick/place calls; unattempted goals recorded as
   `not_attempted`. `run_episode(program=...)` or task["program"]; records carry program, program_log,
   program_error. Job `plan_program` (prompt v1; rule stand-in = one pick/place per goal, bases first, one retry).
   Stacking: monitor "on" a block/bar/cylinder (centred over its top face, resting on it); task `stack_by_size`;
   task_spec validator allows stacking (bottom of a stack has no goal; prompt v3). The oracle got the stand-in
   program as a last attempt (ordered tasks were "unsolvable" before).
   `prompt_task.py "<request>" --program [--compare]`. Results: "Stack the blocks, largest at the bottom": fixed plan
   0/8 (distance order + targets perceived only at the start), program 8/8 with both the rule stand-in and Claude
   (Claude wrote spec + program, no repairs). Demo task with the bottle: Claude program 6/8 = fixed plan 6/8 (after
   fixing shared-tray spots: first version put the bowl on the block, 0/8).
   Verification search after the patch (runs/20261031-013441_failsearch): cylinder/centre_grasp 24% -> 5.8%;
   valid failures per 100 random / Optuna / model 5.0 / 6.2 / 10.6. Largest remaining cluster: block/top_grasp
   collision 23% (large blocks; the investigator's "too wide" was rejected, cause still unknown); rare rim_pinch,
   ball and short-cylinder cases (2-3 each).
   Large blocks fixed in PERCEPTION (fixed infrastructure, not a skill): the top-face yaw came from the principal
   axis, undefined for a square, so 5.7-6.5 cm blocks were perceived 21-45 deg off and the gripper closed across the
   diagonal (9.2 cm > 8 cm opening). `describe()` now takes yaw from the minimum-area bounding rectangle of the top
   face (along its longer side): error <= 4 deg, 15/15 previously failing blocks succeed with the unchanged
   top_grasp; named tasks 8/8 each. Found by comparing perceived vs true yaw (perception gap), which the investigator
   cannot do: true orientation is not observable and its tests only change skills/objects. Smoke check added.
   Verification search (runs/20261031-014357_failsearch): block cluster gone; valid failures per 100 random /
   Optuna / model 3.8 / 5.6 / 5.0. Trend over the rounds (Optuna): seed 90.6 -> bowl repair 60.0 -> cups+tall
   cylinders 18.8 -> thin-cylinder patch 6.2 -> perception fix 5.6. Remaining: cylinder/centre_grasp drops 9%,
   short cylinders 3.5%, rare rim_pinch/ball cases.

4f. **Program verification done** (`kintsugi/composites.py`): composite cards in `skill_library/composites/` (kind
   composite: signature = goal structure by object kind incl. constraints, program, evidence, lineage, verified_spec
   hash; history/ keeps old versions). `verify_program`: program AND fixed plan on the same 32 fresh scenes (train
   seeds 500+, dev 100500+); pass = >= 80% overall and on held-out, and not significantly worse than the fixed plan
   (reversed McNemar). Reuse = retrieve then verify: `find_composite` by signature + hard-coded ids present; if the
   task spec differs from the verified one, `quick_check` (4 fresh scenes, >= 3) before reuse, else the model writes
   a new program. Plan-program prompt v2 prefers find(kind/colour/properties) over ids. `prompt_task.py --program
   [--verify] [--fresh]`; records carry `program_source` (model meta or composite name/version).
   Results: Claude's id-free stacking program verified 31/32 (held-out 15/16) vs fixed plan 0/32 -> composite
   `stack_blocks_largest_bottom` v1 (runs/20261031-032054_verify_program); "Build a tower ... biggest at the bottom"
   -> quick check 4/4 -> reused, no plan model call, 4/4; "smallest at the bottom" (same signature, opposite goals)
   -> quick check 0/4 -> rejected -> Claude wrote a new program, 4/4.

4g. **Model-written primitives done (push)**: new `plate` template (wide thin disc: no grasp fits) and task
   `slide_plate` (plate onto a zone 15-25 cm away). Perception now describes flat markers (zones were invisible:
   every point under 2 mm was dropped as table). `kintsugi/primitive_code.py`: sandbox with motion API move(x, y, z,
   yaw, tilt, speed) / open_gripper / close_gripper / wait / tcp + math; code defines primitive(obj, target, p); AST
   whitelist, no attribute access, 40 moves max; records finger width while moving. Cards kind "push", impl
   "generated", code stored in the card; `Episode.push`; program API push(obj, target) and has_skill(kind); a
   program asking for push without a push card fails with "missing capability: ..." (record field
   missing_capability). Job `propose_primitive` (prompt v2 incl. gripper-as-pusher geometry; validator: code +
   every p['...'] declared; rule stand-in = open two-finger push). `kintsugi/create_skill.py` /
   `scripts/create_skill.py push --task slide_plate`: before -> round 1: 3 different strategies (each told the
   others), each searched (41 candidates x 3 scenes) -> best -> Optuna (perturbed) -> gate (held-out 16 vs before,
   McNemar, >= 70%; regression) -> if it fails, feedback (own code, tuned values, held-out failures split into
   along/sideways error, finger opening) -> one revision. `drop` label now requires the object to have been lifted.
   Results: rule stand-in promoted in 140 s (held-out 16/16). Claude single proposals (Sonnet x4, Opus x2) all chose
   a closed-gripper / single-contact push and failed the gates (11, 5, 3, 4, 6, 5 of 16): the gates rejected six
   plausible primitives. With 3 strategies + feedback: Claude's revision `push_from_behind` (open gripper, segmented
   push from behind) -> held-out 16/16 vs 0/16 (p = 1.5e-05) -> PROMOTED (skill_library/push_from_behind.json,
   789 trials, 10 min, runs/20261031-040016_create_push). End to end: "Move the plate onto the blue zone." ->
   Claude spec + program (pick, next grasp, else push; plan prompt v4 rule) -> 8/8 vs fixed plan 0/8.
   Caveat: the oracle has no push skill, so it labels these scenes unsolvable.

4h. **Recovery, fixed-plan ordering, auto-repair (2026-10-31)**:
   - Fixed plan: goals whose target is another item (stack base) go first; movable targets (block/bar/cylinder) are
     re-perceived at placing time (`Episode.target_desc`). stack_by_size with the fixed plan 0/32 -> 24/24.
   - Recovery inside a run (`Episode(recovery=N)`, `run_task.py --recovery N`, `prompt_task.py` default 2,
     experiments 0 so searches/gates measure raw skills): after a failed step, if `should_retry` (observable only:
     retryable label, perception does NOT see it on the target, still perceived, not tipped: height ratio >= 0.7)
     re-perceive and retry with the next verified grasp (`retry_grasp`), same in program pick/place. First version
     decided from the simulator's goal check and retried tipped objects: no gain (148 -> 147/160 hard). Fixed:
     nominal 242/244 either way (nothing to recover); hard physics (friction x0.35-0.7, mass x1.3-2) probe 148 ->
     150/160 (p = 0.25, not significant), faster on average. Remaining failures are mostly tipped objects (no
     righting skill). `scripts/measure_recovery.py [--hard] [--tasks ...]`.
   - Run-time grasp-pose ranking NOT built: remaining failures are physics (tipping, slips), only 2/12 collisions.
   - Auto-repair orchestrator (`kintsugi/auto_repair.py`, `scripts/auto_repair.py <run dirs> [--tag] [--min-failures]`):
     failed steps from ordinary runs -> clusters (source, kind, skill, physics) -> exact oracle replays under the
     recorded physics (records now carry `perturb`) -> investigator -> explorer (probe failures: region around the
     failing objects; named task; typed-request spec via task "spec" in make_task) with the confirmed fixes as hints
     and the same physics. Regression checks and baselines always at nominal physics (perturb={}).
     Result on the hard probe runs: 12 failures, all oracle-solvable; mug/rim_pinch (6) repaired: investigator 9
     confirmed fixes -> patch rim_pinch.depth 0.0129 -> 0.035 for hollow-wide-tall (source investigator); gate
     cluster 8/10, held-out 46/48 vs 41/48 (p = 0.03); at nominal physics mugs 45 -> 47/48, cups and bowls 24/24
     unchanged (runs/20261031-050348_auto_repair). run_batch now accepts patches.

4i. **Web app (first version) done**: `server/app.py` (FastAPI, read-only): /api/runs (kind by folder name,
   nested runs by their own name), /api/run, /api/trial, /api/replay (gzip pose logs), /api/skills (cards, history,
   primitives, composites, reference, performance, patches), /api/headlines (diagnosis over prompt v3+ runs,
   transfer, created skill, gated repairs, failure-search trend), /api/meshes (Panda meshes from the COMPILED model:
   MuJoCo recentres meshes, raw .obj files would be offset; base64 float32/uint32, gzip). Serves web/dist at / when
   built. Recorder now exports material colours for meshes. `web/` (React 18 + Vite 5 + three 0.169 +
   react-three-fiber 8 + drei 9, pinned): tabs Overview / Task & plan / Replay / Investigator / Experiments /
   Skills; chips measured / hypothesis / confirmed / rejected; light + dark. Replay keeps MuJoCo z-up (camera
   up = z), w-first quaternions converted; Canvas needs preserveDrawingBuffer for screenshots/video capture.
   `scripts/record_demo.py` -> runs/demo/1..6 (bowl before/after repair, stack program, plate push, mug, bar) with
   pose logs. `.claude/launch.json` has "api" (uvicorn :8000) and "web" (vite :5173). fastapi + uvicorn added to
   requirements; httpx2 for TestClient was NOT installed (unknown package): test endpoints by calling functions.
   Checked in the browser pane: all tabs load, no console errors; 3D replay renders the Panda, objects, shadows.
4j. **Studio (web app front door) done**: prompt -> task spec + 3D preview (first layout that fits) -> plan program
   (verified composite with the same spec hash, else the model; fixed plan if rejected) -> settings (scenes,
   parallel workers, fix automatically, retry a failed step, harder physics = friction x0.35-0.7, mass x1.3-2) -> one
   job at a time: `kintsugi/studio.py` run_job (run -> auto_repair on the run if robot failures -> re-run the SAME
   seeds if the library changed), started by the server as `scripts/studio_job.py runs/<id>_studio` in its own
   process group (Stop = taskkill /T), progress in events.jsonl (stage / trial / log with a coarse repair stage
   parsed from the [auto_repair]/[investigate]/[explore] log prefixes / result / error). Seeds whose layout does not
   fit are skipped (`valid_seeds`; Claude's crowded specs failed 2/8 placements), episode_error never triggers
   repair. Live 3D: `kintsugi/sim/live.py` LiveWriter, enabled by env KINTSUGI_LIVE_DIR (spawned pool workers
   inherit it, so every TrialPool trial streams, incl. investigator/explorer trials): per-episode jsonl (meta, scene,
   frames, events, done) + per-worker pointer, last 3 kept; ~300 KB-1.5 MB per episode. The browser buffers and plays
   one episode at real time, then the freshest unseen one; simulation is never slowed. A hidden browser pane pauses
   requestAnimationFrame (looks frozen in tests; not a bug). Tested: demo prompt (Claude) 6/8 with 2 unplaceable
   layouts (fixed since), mug + harder physics 8/8, thin cylinder + harder 16/16, balls + bowl + harder without
   retries 14/16 -> auto_repair ran from the UI (expert check, investigator, explorer patch search).
   That last repair took 17 min / 1679 trials for a 3/96 failure and the gates rejected everything, so the studio
   now passes `Explorer(min_fail_rate=0.10)` (via `auto_repair(explorer_kwargs=...)`): the before phase stops as
   outcome "rare_failure" once the Wilson 95% upper bound of the valid-failure rate is < 10% (pre-declared; off by
   default, so experiments and earlier reports are unchanged). The UI reports it as "too rare to fix" with the
   measured count.
4k. **Missing skills are created from the studio**: program API `use(capability, obj, target=None)` runs a verified
   generated card whose kind == capability (push(...) = use("push", ...)); with none it stops with "missing
   capability: <kind>" and records `missing_call` {capability, obj, target} in the trial record. Plan prompt v5 tells
   the model to still write use("<capability>") when the library lacks it and lists the library's capabilities
   (input `capabilities`, part of the cache key). `SkillCreator` takes a typed request's task_spec + the user's
   program + obj/target + description (named tasks still work: scripts/create_skill.py); card kind = capability.
   Studio job: missing_call -> SkillCreator -> re-run the same seeds if promoted; UI shows a creation checklist
   (confirm missing, write candidates, try, tune, verify, add). Tested with push_from_behind moved out of the
   library: "Move the plate onto the blue zone." (Claude) -> 0/8 program_error -> 3 strategies, straddle push won
   (39/41 settings succeed once), tuned 8/8, gate 16/16 vs 0/16 (p 1.5e-05), regression pass -> 8/8 re-run, 6.5 min,
   522 trials. The original card was restored; the test card is kept in
   runs/20261031-193322_studio/create_push/. Limit: success must be expressible with existing goal predicates
   (on/in/upright), so new capabilities are ones whose result the monitor can already check.

4l. **UI redesign**: tabs Studio / Progress / Episodes / Investigator / Experiments / Skills (Task & plan + Replay
   merged into Episodes; Overview -> Progress). `web/src/charts.jsx`: LineChart, Dumbbell, HBars, Funnel, Heatmap,
   RateBar (SVG, hover tooltips, palette roles --s0 neutral / --s1 blue / --s2 orange from the dataviz reference,
   one blue ramp for heatmaps). `web/src/viewer.jsx`: shared 3D viewer. `ui.jsx`: `Guard` error boundary inside every
   Card and around every tab (shows the raw data instead of blanking the app), status chips without text render as
   coloured dots with a hover meaning, `Details` for raw data. Server: `/api/progress` (comparable failure-search
   rounds = 160/strategy, no obstacle, with the repairs between rounds; repairs with time and readable names;
   diagnosis per run), run titles (`run_title`: studio prompt + phase, demo names, kind + subject + time),
   `/api/run` returns `episodes` (outcome per replayable trial); created-skill headline only counts cards still in
   the library. Program interpreter records `program_lines` (source line + sim time of every pick/place/use call), so
   Episodes highlights the running line; demos re-recorded with it. Model outputs never produce UI: reports are
   written by fixed code, views are fixed per report kind, Generic fallback shows raw JSON.
4m. **Ordered goals + polished shell**: task spec may carry `"order": [item ids]` ("first red, then green...";
   task_spec prompt v4, validator: goal items only, no repeats, >= 2). Episode records the sequence in which goals are
   first reached (`completion_order`); an item reached before an earlier one is labelled `wrong_order` (with
   `order_missing`); success also needs `order_ok`. The fixed plan sorts by the order (overrides distance); plan
   prompt v6 + the rule stand-in follow it. wrong_order is a plan mistake, so the studio never sends it to skill
   repair. Tested: correct order 2/2, reversed 0/2 (blue, green labelled wrong_order), fixed plan 2/2; Claude spec
   emits the order; studio run 8/8 with order_ok. Fixed a validator bug: plan programs using only find_one(...)
   were rejected ("never refers to the task's objects") and silently replaced by the fixed plan.
   UI: left sidebar (Work: Studio, Episodes / Insight: Progress, Investigator, Experiments / Library: Skills; inline
   SVG icons; label legend in the footer), page header per tab, quieter tokens, skeleton loading, previous run list
   kept while refreshing. Studio: prompt bar with the model as a pill, numbered goal rows ("in this order"),
   one-line settings summary, banner with a progress bar, repair/learning steps as an icon row with one activity
   line, outcome = big numbers + one verdict, details behind "why?". Fixed run_kind: demo folders ending in
   "_repair" were classified as auto-repair runs (only a studio job's "repair" folder is).
4n. **Theme v3 + robot cameras**: fonts Space Grotesk (headings) / Inter (UI) / JetBrains Mono (code, numbers) from
   Google Fonts (system fallbacks offline); near-black dark palette, Kintsugi gold for primary buttons and the brand,
   uppercase tracked section labels, KPI strip as one divided panel, digit-tile `Clock` (Studio job time, Episodes
   episode time). Viewer draws two robot cameras into the same canvas (scissored viewports, `RobotCameras` in
   viewer.jsx): a wrist camera on the "hand" body looking along its z axis (fingers) and a fixed front camera;
   toggle "cameras on/off" in the corner; used by Studio (preview + live) and Episodes. Layout stability: every
   grid child min-width 0, code blocks scroll, side columns fixed (300 / 290 / 400 px) and sticky, main columns
   minmax(0, 1fr): opening details no longer resizes neighbouring cards (measured: widths unchanged).
4o. **UI pass (user feedback)**: gold only for the logo and primary buttons (icons, dots white/ink). Robot cameras
   moved out of the main view into a strip of rounded tiles below it (`Regions` in viewer.jsx renders the main
   camera and each robot camera into regions measured from placeholder elements; rounded corners via rings in the
   page colour). Episodes picker: run row + episode-tile row (scrolls past two rows). Headings shortened to
   professional titles, explanatory sentences removed or reduced to one-line captions. Skills: card header,
   facts row (works on / success check / evidence), success by shape, patches, versions; "Library timeline" strip
   at the top. **Label rule as applied in the UI (user decision 2026-10-31)**: "measured" is the default for
   everything shown and is not marked; hypothesis / confirmed / rejected are always tagged in words where they
   appear; no separate legend.
   Follow-up pass: robot cameras are small translucent rounded tiles inside the main view (bottom left): each
   camera renders to an offscreen target, drawn as a quad with a rounded alpha mask at 0.9 opacity
   (`RobotCameras` in viewer.jsx); camera toggle = icon at bottom right. Step rows, hypothesis cards and gate
   cards have no coloured side bars (status icon or tag instead). One tag style for every status chip (18 px,
   uppercase, tinted). Experiments sidebar = "Experiment log" (grouped rows: time · title · short outcome).
   Skills timeline: scrollbar hidden, arrow buttons + edge fades appear only when it overflows.
   Studio: prompt box (auto-growing textarea, model pill and round send button inside, keyboard hint),
   suggestion chips; "Recent runs" strip always visible, the running job or else the newest run is selected on
   open, the selected run is highlighted. Logo (user pick 2026-11-01): striped diamond, white horizontal stripes clipped to a
   diamond on a near-black tile (earlier: white ring crossed by a gold seam)
   (`web/public/logo.svg`, favicon, inline in the sidebar; server route /logo.svg for the built app).
   Navigation moved to a sticky translucent top bar (logo + wordmark, nav groups Work / Insight / Library split
   by thin separators); content uses the full width; sticky side columns sit at top 76 px. `ScrollRow` (ui.jsx):
   horizontal row with arrows + edge fades instead of a scrollbar (recent runs). The studio job poll retries
   failed requests (5 in a row ends following) instead of staying "running" after one failed request.
   Studio playback: a finished run plays scene 1 of its latest phase and then the next scenes in order (playlist,
   also after a live job ends); the scene being viewed is outlined in the tiles; past runs are attached as not
   running (the live player only follows jobs known to run); a started job is added to Recent runs at once and
   highlighted. Prompt box v2: 3 lines tall, "New task" label, gold gradient border + glow, suggestion icons.

**Remaining, in order** (plan build order 8c onwards):
6. done: Model-written primitives (8c), see 4g.
7. old text kept for reference: **Model-written primitives** (8c), shown on push: gap detected (flat plate no grasp fits / item out of reach) ->
   model proposes a primitive (motion-API code + parameter ranges + check) -> AST sandbox -> explorer search -> gates
   -> card `source: model-written primitive`; plan program push-to-edge-then-grasp.
7. Recovery inside a run (8d).
8. **Web app** over saved files (task & scene with program, replay, investigator ledger, experiments monitor +
   capability map, results with promotion diff and gate p-values, skill graph / lineage, export).
8f. **Execution checks (user-reported oddities, 2026-10-31)**:
   - Grip monitoring: `Robot.move_to(watch_grip=True)` stops the motion when the finger gap collapses
     (< GRIP_LOST_WIDTH 3 mm, proprioception only); place_in carries and lowers with it and returns
     `object_lost_in_transport` (+ lost_at) instead of lowering and opening over an empty target; the episode
     labels it `drop`, recovery re-grasps. Cause found: under harder physics a bowl slipped while carried and
     place_in still opened over the tray (scene 11, balls+bowl run); replay now stops at t=40.1 s.
   - Wasted-motion audit for model-written primitives: run_primitive calls on_move after every motion; the
     episode records (simulator, evaluation only) object displacement and whether the goal was already met;
     `motion_audit`: idle motion after the goal (moved nothing, final lift-away excluded) = wasted_frac, and
     overshoot (still moved the object) separately. Runner results carry wasted_frac; SkillCreator search ranks
     equal success by less waste, tuning objective = success - 0.1 x waste, gate reports it, gate feedback tells
     the model; Episodes shows "redundant motion N%", Experiments shows idle motion per proposal and gate.
     Cause found: Claude's straddle push did two open-loop passes from the start pose, the second through empty air
     (48-50% idle); push_from_behind 0% idle (12 cm overshoot inside the zone).
   - Motion API `look()`: fresh perception of the object, so primitives can work closed-loop (prompt v3).
8g. **Objects with moving parts, composable success, part perception (2026-11-01, built generic)**:
   - Templates `drawer` (fixed cabinet + drawer on a slide joint, travel 10-15 cm) and `door` (cabinet + panel on a
     hinge at one side, 0-109 deg), `ARTICULATED` in templates.py (part, joint type, unit m/deg). Geom names
     <id>__front, <id>__handle_arm/_post (a post the gripper pinches from above), moving body <id>__<part>. Placer puts
     cabinets at x 0.60-0.67 facing the robot (yaw = atan2(y, x)) with room in front to open.
   - Monitor: child bodies map to their object; `joint_value` (m / deg); `moving_pos`; check_goal adds composable
     measured terms `joint` / `distance` / `height` with op (>=, <=) + value (records carry `measured`); on / inside /
     upright unchanged. task_spec schema/validator accept them (joint only on articulated templates, value within the
     object's range); prompt v6 lists the forms and says to clarify/refuse when none expresses the request; the old
     refusal of drawers/doors is narrowed to unscrewing jars/lids/bottles.
   - Perception: geoms of moving parts belong to their object; parts "front" / "handle" described from depth
     (`describe_part`: centre, tip = outermost points (the handle post), normal, width, height); the front's normal is
     perpendicular to the panel's long axis, a handle takes its front's normal. Part points also use one extra camera
     "near" on the robot's side; object descriptors keep using front + top only. Handle tip error 5-6 mm.
   - Skill creation: propose prompt v4 explains parts and that joints are not observable; inputs carry the goal;
     rule stand-in `agents/jobs/open_motions.py` (straight pull along the front normal; arc around the hinge assumed at
     the front's edge opposite the handle). CAPABILITIES open_drawer / open_door, named tasks open_drawer (>= 8 cm),
     open_door (>= 50 deg); plan prompt v7 + rule stand-in write use("open_<template>", obj); executor labels missed
     measured goals `goal_not_met`, records `parts_moved_cm`; motion audit follows the moving part.
   - Results (rule backend, dry runs, no card written): open_drawer from nothing: before 0/8, search 29-32/41
     settings work, tune 8/8, gate held-out 16/16 vs 0/16 (p 1.5e-05), regression pass; open_door: 22-31/41, tune 8/8,
     gate 15/16 vs 0/16 (p 3.1e-05), regression pass. Claude (Studio): "Open the drawer of the cabinet." -> goal
     joint >= 0.10 m, program use("open_drawer", drawer). Existing tasks unchanged: 5 tasks x 6 fixed seeds, new code
     vs last commit, 30/30 identical (success, sim time, labels). Smoke 24/24 (new drawer check).
   - Not built yet: importer for user MJCF/URDF models (doors etc. come from the templates for now); closing
     motions (the goal form supports op "<=").
8h. **Feasibility check + oracle for motions (2026-11-01)**:
   - Oracle: `reference_motions()` in oracle.py = hand-written push + one opening motion per ARTICULATED template
     (slide -> straight pull, hinge -> arc; the rule stand-ins at default parameters, never tuned or discovered);
     ground-truth perception now has true parts (`_truth_parts`: front centre/normal/width, handle tip = middle of
     the post, matching what depth reports). Oracle runs act on feasibility. Oracle before: push/open scenes always
     "unsolvable"; now drawer 16/16, door 14/16, plate 14/16, so their failures count as valid and can be repaired.
     Bug found on the way: stand_in("slide") picked the arc ("slide" contains "lid").
   - `kintsugi/feasibility.py` `assess(task, percepts, cards)`: per goal, from perception only: grasp (cards whose
     preconditions fit) / motion (no grasp fits but a verified push card covers "on" a tray/zone, or a joint goal has
     its open_<template> card) / fallback (no grasp fits, nothing else: default grasp is still tried, so the seed
     demo's bowl still fails with top_grasp) / missing / unseen / unreachable, each with a reason (e.g. "15.6 cm
     across at its narrowest, the gripper opens 8 cm"). Every record carries `feasibility`. `Episode(feasibility=
     True)` (run_episode / run_batch / trial_cfg `feasibility`; on in the Studio and prompt_task.py, off in
     experiments like recovery): the fixed plan runs motion subtasks (`Episode.run_motion`), reports missing
     capabilities up front (label missing_capability + `missing_call`, so the Studio creates the skill from a fixed
     plan too), and program pick() skips a grasp when the verdict is motion. Studio goal rows show the verdict
     (skill name, or "no grasp fits" / "skill missing" with the reason on hover).
   - Precondition fix found by it: axis_centre_across v1 required only width/height, so plates (round) were routed to
     the bar grasp and collided. The explorer now also derives elongation bounds (`elongation_bounds`: found only on
     long objects -> min_elongation = 0.8 x smallest; only on round -> max 1.25 x largest); card v2 adds
     min_elongation 1.73 (bars perceive 2.16-9.39, plates ~1.0; history/axis_centre_across_v1.json, lineage
     "precondition audit"); dev bars 32/32 before and after.
   - Results: slide_plate fixed plan 0/16 -> 16/16 with feasibility; pick-else-push program 8/8 both ways, 25% less
     robot time (no doomed grasp); sort_with_bowl / stack_by_size / balls_in_basket 16/16 both ways; standard tasks
     new code vs last commit (feasibility off) 36/36 identical. Smoke 25/25.
8i. **One lifecycle for every skill + part-level motions (2026-11-01)**:
   - Part-level motion API (primitive_code.py, fixed, generic, on perceived parts/directions): grasp_part(part,
     approach, height, depth, force, from_above), move_linear(dx, dy, dz, dist, speed, until_stall, stall_lag),
     move_arc(cx, cy, angle, ...), release, retreat, holding. Line/arc motions step 1 cm and stop by themselves when
     the hand falls stall_lag behind its path (end stop) or the grip is lost (proprioception), then stop pulling
     (`Robot.move_to(correct=False)`: no sag correction that would push harder into the stop). Trials record
     `part_motions` and `hand_path_mm`. propose prompt v5 documents them and prefers stopping at the end stop.
     Stand-ins (open_motions) and the oracle's reference motions now use them: 9 cm drawers 12/12 (fixed pull 11/12),
     drawers 12/12, doors 10-12/12 and 3 s faster.
   - `goal_lost`: a goal met right after its step but not at the end (the drawer slid back after release, an object
     fell off later) now fails that step with this label (+ `final_measured`); before, such episodes failed with no
     label and the Studio saw nothing to repair. Standard tasks: 36/36 identical vs the last commit.
   - `kintsugi/revise_skill.py` SkillReviser (SkillCreator subclass): auto_repair sends clusters whose failing skill
     is a model-written card here (grasp skills still go to the explorer). before (rare/not reproduced stop) ->
     settings (same code, search + Optuna) -> rewrite (model gets the code, observable evidence, the scenes where it
     works; 3 strategies, search, tune, one more revision with gate feedback) -> gate: FIX (failing request, dev
     seeds, McNemar improve, >= 70%), HOLD (the card's original request from its lineage: not significantly worse,
     >= 70%), REGRESSION -> promote a NEW card beside the old one (lineage parent + why + results; task_spec and
     program now stored in every created card's lineage) + evidence for both on the object's shape group. Router
     for motion cards (`router.pick_motion`): measured score, then `generation` (newest revision) on ties.
     Candidates run with every card of that kind removed, so the candidate is the one used.
   - Drawer travel range now 0.08-0.15 m (short drawers can be requested).
   - Result (Studio job, Claude, "Open the short drawer (9 cm)"): 0/8 (goal_lost: overpulled, bounced back) -> expert
     solves all 8 -> revision: settings alone best 2/3 -> Claude rewrote it with grasp_part + move_linear until stall
     (41/41 settings work) -> tune 8/8 -> gate fix 16/16 vs 1/16 (p 3.1e-05), hold 16/16 vs 16/16, regression pass ->
     `open_drawer_stall_detect` beside `open_drawer` -> re-run 8/8; 17.5 min. Test-split check (16 fresh scenes each):
     normal drawers old 16/16, now 16/16; 9 cm old 1/16, now 16/16; 8.5 cm (never seen, 5 mm margin) old 1/16, now
     11/16. Smoke 25/25 (drawer check uses a 9 cm drawer).
   - Note: skill_library/open_drawer.json (learned in a Studio run) and open_drawer_stall_detect.json are untracked.
8e. **Base + user skill libraries** (user decision 2026-10-31, implement before the final commit): shipped
   `skill_library/` read-only; discoveries go to a local git-ignored user library merged by the loader; export/import
   skill packs; contribute back only after re-passing the gates. Today every discovery is written into the tracked
   `skill_library/`, so local learning and library updates can collide.
9. **Spark + SAM switch-over**, re-run reported experiments on Meta models, perception gap on ~50 episodes.
10. Unseen-object benchmark, test split once, freeze numbers, demo video. Post-hackathon: pouring (tilt-while-holding
   primitive + particle liquid), video imitation, curriculum, self-proposed tasks.

## How to run
```
pip install -r requirements.txt
python scripts/fetch_assets.py
python scripts/smoke_test.py                       # run after every change
python scripts/run_task.py sort_with_bowl -n 8 --oracle         # the demo task, failures checked by the oracle
python scripts/prompt_task.py "Put the bowl on the tray." -n 4  # prompt -> task spec (rule backend) -> run
python scripts/failure_search.py --from-seed --budget 64        # where does the library fail? (random/Optuna/model)
python scripts/explore.py sort_with_bowl --from-seed --dry-run  # demo repair loop from the seed library, no card written
python scripts/explore.py --from-search runs/<id>_failsearch/report.json --cluster cylinder/top_grasp --dry-run
python scripts/run_task.py bowl_on_tray -n 16 --split dev       # held-out bowls with the current library
python scripts/run_task.py bowl_on_tray -n 8 --reference --grasps rim_pinch_handwritten
python scripts/run_task.py sort_by_color -n 4 --workers 4 --no-video
```
Headless Linux: `MUJOCO_GL=egl`, or `xvfb-run -a` with `MUJOCO_GL=glfw` if EGL is missing.

## Rules
- Build a thin end-to-end path before breadth; keep the smoke test green and under ~2 minutes.
- Every model output is validated against a schema; on failure feed the error back once, then reject with a reason.
- Simulator is the source of truth for outcomes; success predicates and failure labels are deterministic code.
- Log every trial with seed, scene, skill card version, code version, outcome. Never tune on held-out objects.
- Every statement in the UI is labelled measured / hypothesis / confirmed / rejected (measured is the unmarked
  default; the other three are always tagged in words).

## Plan documents
- **Full implementation plan: `docs/implementation_plan.md`** (read it before starting a new step: templates, task generation, skills and repair loop, web app, build order, optional improvements, demo script, metrics, risks).
- Online copy (v1, out of date since the v2 rewrite on 2026-10-30): https://claude.ai/artifact/9ogQiLNZ3fMTFziTtbNsoq
- Earlier concept plan (superseded; kept for history): https://claude.ai/artifact/4xDz5bF8uk2NiiQwJhZE2j
