# Kintsugi

A Franka Panda in MuJoCo that turns a task into a scene, runs it, explains its failures, and builds a verified library of skills. No training: every improvement is a skill or parameter change found and checked in simulation.

## Setup (Windows, macOS or Linux)

```bash
conda create -n kintsugi python=3.11 -y
conda activate kintsugi
cd C:\Users\yahya\source\kintsugi
pip install -r requirements.txt
python scripts/fetch_assets.py          # downloads the Panda model from MuJoCo Menagerie
```

On Windows and macOS, rendering works out of the box. On a headless Linux server, set `MUJOCO_GL=egl`, or run under `xvfb-run` with `MUJOCO_GL=glfw`.

## Test it

```bash
python scripts/smoke_test.py                                    # ~10 s, 8 checks, images in runs/smoke/
python scripts/run_task.py bowl_on_tray -n 8 --grasps top_grasp # expected to FAIL: bowl is too wide
python scripts/run_task.py sort_with_bowl -n 8 --oracle         # the demo task; --oracle checks each scene was solvable
python scripts/prompt_task.py "Put the bowl on the tray." -n 4  # typed request -> task spec -> run
python scripts/prompt_task.py "..." --backend claude_cli        # same, with Claude as the reasoning stand-in (run `claude` /login once)
python scripts/prompt_task.py "Stack the blocks, largest at the bottom." --program --compare -n 8   # model-written plan program vs fixed plan
python scripts/failure_search.py --from-seed                    # where does the library fail? random vs Optuna vs model, capability map
python scripts/investigate.py --from-search runs/<id>_failsearch/report.json --library current  # why? tested causes + ledger
python scripts/transfer.py mug --backend claude_cli               # unseen object type: library retrieval + verification vs from scratch
python scripts/create_skill.py push --task slide_plate --backend claude_cli   # the model writes a missing skill; search + gates decide
python scripts/explore.py sort_with_bowl --from-seed            # repair loop: bowl fails, rim grasp found, tuned, gated, promoted
python scripts/run_task.py bowl_on_tray -n 8                    # uses the promoted rim_pinch card
python scripts/run_task.py bowl_on_tray -n 16 --split dev       # held-out bowls
python scripts/run_task.py balls_in_basket -n 8
python scripts/run_task.py sort_by_color -n 4
```

Every run writes to `runs/<time>_<task>/`:

| File | What |
| --- | --- |
| `summary.json` | success rate with 95% interval, failure label counts, trials per minute |
| `trials/00007.json` | one record per trial: seed, plan, skill trace with checks, failure label, percepts |
| `media/00007.mp4` | video, saved for failures and for 1 in 20 successes |
| `media/00007_poses.json.gz` | every body's pose at 30 fps, for the Three.js replay |

`scripts/explore.py` writes `runs/<time>_explore_<task>/report.json`: the failure cluster, every sampled grasp candidate with the reason it was kept or rejected, families and their success map, tuning history, gate results and the promoted card.

Useful options: `--workers N` (parallel simulations, default CPU cores minus 1, at most 8), `--split train|dev|test`, `--reference` (also load hand-written reference cards), `--no-video`, `--no-poses`, `--success-sample 0` (no media for successes), `--set rim_pinch.depth=0.025` (override a skill parameter).

## What is in here

| Folder | Contents |
| --- | --- |
| `kintsugi/sim/` | templates (block, ball, cylinder, bowl, cup, bin, tray, zone), scene builder and placer, Panda motion with `mink` IK, privileged monitor (goal checks, failure labels), recorder |
| `kintsugi/perception/` | colour, depth and segmentation rendering, depth to point cloud, shape descriptors (size, hollow, rim radius, wall slope) |
| `kintsugi/skills/` | top_grasp, rim_pinch, side_wrap, lift, place_in / drop_in, each with an observable check |
| `kintsugi/tasks/` | task specs (sort by colour, balls in basket, bowl on tray, cup on tray, the demo `sort_with_bowl`), frozen splits, and the planner |
| `kintsugi/executor.py` | one episode: perceive, plan, run skills, check, label |
| `kintsugi/runner.py` | parallel trials and saved results |
| `kintsugi/search/` | grasp sampler (gripper fit on the point cloud), IK reachability filter, exploration, tuning, gates, promotion |
| `skill_library/` | active skill cards: parameters, allowed ranges, checks, version lineage. `history/` keeps old versions, `primitives/` holds the explorer's search space, `reference/` holds hand-written cards used only for comparison |
| `experiments/` | frozen object splits (`splits.json`) and cached regression baselines |

## Not wired yet (needs the Meta API key)

- **Muse Spark** for prompt to task spec, planning and failure diagnosis. Every model job goes through `kintsugi/agents/llm.py`, which has a `rule` backend (deterministic stand-ins), a `claude_cli` backend (Claude via the Claude Code CLI, for development) and a `spark` backend (set `KINTSUGI_LLM=spark`, `META_API_BASE`, `META_API_KEY`). Same prompts, schemas and validators for all three; calls are cached in `cache/llm/`.
- **SAM 3** for masks. `perception/perceive.py` uses the simulator's own segmentation for now (`mode="sim"`); `mode="sam"` is the slot for it.

`.sync.tgz` and `.sync_stamp` are leftovers from how files were copied to this PC during development; they are safe to delete.
