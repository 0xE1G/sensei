# Kintsugi: notes for contributors and coding agents

Read this file and the [README](README.md) before changing code.

## What it is
A Franka Panda in MuJoCo turns a typed task (or a photo of a table) into a scene, a task spec and a plan of skill calls,
runs it, and when a step fails it explains why with evidence, searches for a fix, keeps only fixes that pass held-out
statistical gates, and reuses verified skills on unseen objects. No training and no GPU: every improvement is a skill,
a parameter patch or a routing change found by search in simulation and verified there. Say "builds and verifies its
own skill library", never "learns" or "trains".

## Design decisions (ask before reversing one)
- **Robot**: Franka Panda from MuJoCo Menagerie (`scripts/fetch_assets.py`). The gripper servo is stiffened in
  `kintsugi/sim/scene.py`; without it the fingers pass through thin walls.
- **Objects**: parametric templates built with MjSpec (`kintsugi/sim/templates.py`), imported MJCF/URDF files in
  `objects/`, and optional Google Scanned Objects. The model picks templates and parameters; code clips them to tested
  ranges and a deterministic placer lays them out. Nothing downstream may depend on a template name.
- **Skills are defined on perceived features** (rim, top face, long axis, parts), never object names. Three layers:
  motion (mink IK, fixed) / grasp, place and motion skills (versioned cards in `skill_library/`) / task plan.
- **Privilege split**: skills see only perception descriptors and proprioception. True simulator state is used only by
  the expert (oracle), goal checks and the failure labeler.
- **Models**: every model call goes through `kintsugi/agents/llm.py` with backends `rule` (deterministic stand-ins,
  tests) and `spark` (Meta Model API). Every job has a versioned prompt (`agents/prompts/`), a JSON schema and a
  semantic validator; an invalid answer gets its errors fed back once, then is rejected with the reasons. Calls are
  cached by (job, prompt version, input, backend, model) and logged; call models per proposal, never per trial.
- **Who writes what**: models write task specs, plan programs, diagnoses and proposals, and may propose new motion
  code. They never write trusted skills directly: search finds the numbers and the gates decide.
- **Plans are programs**: model-written Python over the perception and skill API, run by an AST-whitelisted
  interpreter on the same episode step methods as the fixed plan. The fixed plan stays as the bulk-experiment engine,
  the fallback and the baseline.
- **Libraries**: `skill_library/` is the base library (read-only at run time); every discovery goes to the user library
  `user_library/` (git-ignored). Packs move between them only through the re-verification gate (`scripts/skills.py`).
- **Web app**: React + Vite + react-three-fiber over a read-only FastAPI server. Physics never runs in the browser; it
  replays recorded poses and live frames.

## Rules
- Keep the smoke test green and under about two minutes: `python scripts/smoke_test.py` after every change.
- The simulator is the source of truth for outcomes; success predicates and failure labels are deterministic code.
- Log every trial with seed, scene, skill card version, code version and outcome. Never tune on held-out objects.
- Gates are pre-declared: fix the failure (paired McNemar), hold on held-out objects, no regression on earlier tasks.
- Every statement in the UI is labelled measured / hypothesis / confirmed / rejected (measured is the unmarked default).
- Scripts that use the worker pool must be real files with an `if __name__ == "__main__"` guard (spawn on Windows).

## How to run
```
pip install -r requirements.txt
python scripts/fetch_assets.py
python scripts/smoke_test.py
python scripts/run_task.py sort_with_bowl -n 8 --oracle
python scripts/explore.py sort_with_bowl --from-seed --dry-run
python -m uvicorn server.app:app --port 8000      # after: cd web && npm install && npm run build
```
Headless Linux: `MUJOCO_GL=egl`, or `xvfb-run -a` with `MUJOCO_GL=glfw` if EGL is missing.
