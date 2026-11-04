"""From a typed request to a validated task spec, and optionally run it.

    python scripts/prompt_task.py "Put the bowl on the tray."                      # rule stand-in
    python scripts/prompt_task.py "Stack the blocks, largest at the bottom." --program --verify -n 4
                                        # model writes a program, verified on fresh scenes -> composite card
    python scripts/prompt_task.py "Sort the blocks and the bowl onto the trays of the same colour, don't knock over the bottle" --backend spark -n 8
    KINTSUGI_LLM=spark python scripts/prompt_task.py "..."                         # Muse Spark, once the key is set

Prints the status (ok / clarify / refuse), the spec and the call metadata (backend, model, cached, repairs).
With -n, runs the spec n times (placement varies by seed) with the oracle and prints the summary.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.jobs.task_spec import to_task  # noqa: E402
from kintsugi.agents.llm import BACKENDS, ModelClient, ModelRejected  # noqa: E402
from kintsugi.composites import find_composite, quick_check, spec_hash, verify_program  # noqa: E402
from kintsugi.runner import default_workers, run_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--backend", choices=BACKENDS, default=None, help="default: KINTSUGI_LLM or 'rule'")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-n", type=int, default=0, help="run the spec n times (0 = only print it)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--program", action="store_true", help="the model also writes a plan program; run that")
    ap.add_argument("--compare", action="store_true", help="with --program: also run the fixed plan, same seeds")
    ap.add_argument("--verify", action="store_true",
                    help="with --program: verify a newly written program on fresh scenes; promote it to a composite card")
    ap.add_argument("--fresh", action="store_true", help="with --program: ignore verified composites, ask the model")
    ap.add_argument("--recovery", type=int, default=2,
                    help="retries per item after an observed failure (re-perceive, next verified grasp); 0 = off")
    a = ap.parse_args()
    client = ModelClient(a.backend, a.model, use_cache=not a.no_cache)
    try:
        res = client.complete("task_spec", {"prompt": a.prompt})
    except ModelRejected as e:
        print(json.dumps({"status": "rejected", "reasons": e.reasons}, indent=1))
        return 1
    print(json.dumps(res.output, indent=1))
    print("meta:", json.dumps(res.meta))
    if res.output["status"] != "ok" or a.n <= 0:
        return 0
    task = to_task(res.output, {"prompt": a.prompt})
    runs = [("fixed plan", task)]
    if a.program:
        card = None if a.fresh else find_composite(task)
        if card and card.get("verified_spec") != spec_hash(task):
            qc = quick_check(task, card["program"], a.workers)
            print(f"\ncomposite {card['name']} matches the task structure; quick check on this task: "
                  f"{qc['success']}/{qc['trials']} -> {'reuse' if qc['pass'] else 'rejected, asking the model'}")
            if not qc["pass"]:
                card = None
        if card:
            # A verified composite applies: reuse it, no model call.
            program, source = card["program"], {"composite": card["name"], "version": card["version"]}
            ev = card["evidence"]
            print(f"\nreusing composite {card['name']} v{card['version']} (verified {ev['success']}/{ev['trials']}, "
                  f"held-out {ev['heldout']['success']}/{ev['heldout']['trials']}); no model call\n{program}")
        else:
            try:
                pr = client.complete("plan_program", {"prompt": a.prompt, "task": task})
            except ModelRejected as e:
                print(json.dumps({"program": "rejected", "reasons": e.reasons}, indent=1))
                return 1
            program, source = pr.output["program"], pr.meta
            print("\nprogram:\n" + program + "\n\n" + pr.output.get("explanation", ""))
            print("meta:", json.dumps(pr.meta))
            if a.verify:
                rep = verify_program(task, program, a.workers, source=source)
                g = rep["gate"]
                print(f"verification: program {g['program']['success']}/{g['program']['trials']} "
                      f"(held-out {g['heldout']['success']}/{g['heldout']['trials']}), fixed plan "
                      f"{g['fixed_plan']['success']}/{g['fixed_plan']['trials']}, p(better)={g['p_better_than_fixed']:.3g} "
                      f"-> {'PROMOTED ' + rep['card']['path'] if rep.get('card') else 'not promoted: ' + g['reason']}")
        runs = ([("fixed plan", task)] if a.compare else []) + [("program", {**task, "program": program,
                                                                            "program_source": source})]
    for label, spec in runs:
        s = run_batch(spec["name"], n=a.n, workers=a.workers, task_spec=spec, oracle=True, save_video=False,
                      save_poses=False, success_sample=0, save_failures=False, progress=False, recovery=a.recovery,
                      feasibility=True)
        print(label + ":", json.dumps({k: s[k] for k in ("run_id", "success", "trials", "success_ci95",
                                                         "failure_labels", "oracle")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
