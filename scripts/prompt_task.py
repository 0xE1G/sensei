"""From a typed request to a validated task spec, and optionally run it.

    python scripts/prompt_task.py "Put the bowl on the tray."                      # rule stand-in
    python scripts/prompt_task.py "Sort the blocks and the bowl onto the trays of the same colour, don't knock over the bottle" --backend claude_cli -n 8
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
from kintsugi.runner import default_workers, run_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--backend", choices=BACKENDS, default=None, help="default: KINTSUGI_LLM or 'rule'")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-n", type=int, default=0, help="run the spec n times (0 = only print it)")
    ap.add_argument("--workers", type=int, default=default_workers())
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
    s = run_batch(task["name"], n=a.n, workers=a.workers, task_spec=task, oracle=True, save_video=False,
                  save_poses=False, success_sample=0, save_failures=False)
    print(json.dumps({k: s[k] for k in ("run_id", "success", "trials", "success_ci95", "failure_labels", "oracle")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
