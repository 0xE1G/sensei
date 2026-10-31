"""Create a skill the library does not have: model-written primitive -> search -> tune -> gates -> card.

    python scripts/create_skill.py push --task slide_plate                         # rule stand-in proposal
    python scripts/create_skill.py push --task slide_plate --backend claude_cli    # Claude writes the primitive

Writes runs/<time>_create_<capability>/report.json (proposals, success map, tuning, gate) and, if the gates pass,
skill_library/<name>.json with the code and lineage "model-written primitive".
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.llm import BACKENDS, ModelClient  # noqa: E402
from kintsugi.create_skill import CAPABILITIES, SkillCreator  # noqa: E402
from kintsugi.runner import default_workers  # noqa: E402
from kintsugi.tasks.specs import TASKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capability", choices=sorted(CAPABILITIES))
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--backend", choices=BACKENDS, default=None)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--dry-run", action="store_true", help="everything except writing the card")
    a = ap.parse_args()
    sc = SkillCreator(a.capability, a.task, a.workers, ModelClient(a.backend))
    if a.dry_run:
        sc.promote = lambda card, *rest: sc.report.update(promoted={"card": card["name"], "dry_run": True})
    rep = sc.run()
    print(json.dumps({k: rep.get(k) for k in ("run_id", "outcome", "promoted", "trials_simulated", "wall_time_s")},
                     indent=1))


if __name__ == "__main__":
    main()
