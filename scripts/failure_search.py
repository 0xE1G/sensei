"""Failure search: where does the skill library fail? Compares random, Optuna and model-guided search at equal budget.

    python scripts/failure_search.py --from-seed                    # the hand-written starting library
    python scripts/failure_search.py                                # the current library (after repairs)
    python scripts/failure_search.py --budget 64 --strategies random model --backend spark

Writes runs/<time>_failsearch/report.json: per-strategy metrics, every trial, a capability map, and failure clusters.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.llm import BACKENDS, ModelClient  # noqa: E402
from kintsugi.runner import default_workers  # noqa: E402
from kintsugi.search.failure_search import STRATEGIES, FailureSearch  # noqa: E402
from kintsugi.skills.library import seed_library  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--budget", type=int, default=160, help="simulated trials per strategy")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--from-seed", action="store_true", help="search the seed library instead of the current one")
    ap.add_argument("--backend", choices=BACKENDS, default=None, help="model backend for the model strategy")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--templates", nargs="+", default=None, help="focused search on some object types")
    ap.add_argument("--obstacle", action="store_true",
                    help="every scene gets a bottle near the item that must not be knocked over or moved")
    a = ap.parse_args()
    fs = FailureSearch(a.strategies, a.budget, a.batch, a.workers, library=seed_library() if a.from_seed else None,
                       library_name="seed" if a.from_seed else "current", seed=a.seed, client=ModelClient(a.backend),
                       templates=a.templates, obstacle=a.obstacle)
    rep = fs.run()

    print(f"\nStrategy comparison ({a.budget} simulated trials each, library: {rep['library']})")
    print(f"  {'strategy':<8} {'valid fail':>10} {'per 100':>8} {'clusters':>9} {'unsolvable':>11}  first found (trial)")
    for name, s in rep["strategies"].items():
        m = s["metrics"]
        firsts = ", ".join(f"{k} @{v}" for k, v in sorted(m["trials_to_find"].items(), key=lambda kv: kv[1]))
        print(f"  {name:<8} {m['valid_failures']:>10} {m['valid_failures_per_100']:>8} {m['distinct_clusters']:>9} "
              f"{m['unsolvable']:>11}  {firsts}")
    weak = sorted([c for c in rep["capability_map"] if c["success_rate"] is not None and c["n"] >= 3],
                  key=lambda c: c["success_rate"])[:10]
    print("\nWeakest regions (all strategies pooled; success among solvable trials):")
    for c in weak:
        print(f"  {c['template']:<9} {c['size']:<22} {c['reach']:<5} {c['success']}/{c['success'] + c['valid_failures']}"
              f"  ({c['unsolvable']} unsolvable)")
    print("\nFailure clusters:")
    for c in rep["clusters"]:
        print(f"  {c['key']:<24} {c['count']:>3} failures  failure rate {c['template_failure_rate']:<6} {c['labels']}")
    print(f"\nreport: runs/{rep['run_id']}/report.json  ({rep['wall_time_s']} s)")


if __name__ == "__main__":
    main()
