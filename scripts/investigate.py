"""Investigate failures: reproduce, diagnose (model job), test each cause with a one-change replay, write a ledger.

    python scripts/investigate.py --from-search runs/<id>_failsearch/report.json --library current
    python scripts/investigate.py --from-search runs/<id>_failsearch/report.json --library seed --backend claude_cli
    python scripts/investigate.py --from-search ... --per-cluster 3 --clusters cylinder/centre_grasp

Cases are valid failures from a failure-search report, stratified by cluster and label. --library must be the
library that report searched (seed or current). Prints top-1 label accuracy (vs the simulator's labels, which the
model never sees) and confirmed / rejected counts; writes runs/<time>_investigate/report.json and ledgers/.
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.llm import BACKENDS, ModelClient  # noqa: E402
from kintsugi.investigate import Investigator  # noqa: E402
from kintsugi.runner import default_workers  # noqa: E402
from kintsugi.skills.library import seed_library  # noqa: E402
from kintsugi.tasks.specs import probe  # noqa: E402


def pick_cases(report, per_cluster, clusters=None):
    """Valid failures, round-robin over labels within each cluster, up to per_cluster each."""
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for strat in report["strategies"].values():
        for h in strat["trials"]:
            if h["outcome"] == "valid_failure":
                key = f"{h['scenario']['template']}/{h['grasp']}"
                if not clusters or key in clusters:
                    groups[key][h["label"]].append(h)
    cases = []
    for key, by_label in sorted(groups.items()):
        queues, chosen = [list(v) for v in by_label.values()], []
        while len(chosen) < per_cluster and any(queues):
            for q in queues:
                if q and len(chosen) < per_cluster:
                    chosen.append(q.pop(0))
        cases += [{"task_spec": probe(h["seed"], h["scenario"]), "seed": h["seed"],
                   "source": {"cluster": key, "search_label": h["label"]}} for h in chosen]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-search", required=True)
    ap.add_argument("--library", choices=["seed", "current"], required=True,
                    help="the library the failure search ran with")
    ap.add_argument("--per-cluster", type=int, default=4)
    ap.add_argument("--clusters", nargs="*", default=None)
    ap.add_argument("--backend", choices=BACKENDS, default=None)
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    rep = json.loads(pathlib.Path(a.from_search).read_text())
    cases = pick_cases(rep, a.per_cluster, a.clusters)
    print(f"{len(cases)} cases from {sorted({c['source']['cluster'] for c in cases})}")
    inv = Investigator(a.workers, library=seed_library() if a.library == "seed" else None,
                       performance={} if a.library == "seed" else None, library_name=a.library,
                       client=ModelClient(a.backend))
    out = inv.run(cases)
    m = out["metrics"]
    print(json.dumps(m, indent=1))
    for c in out["cases"]:
        if not c.get("hypotheses"):
            continue
        h = c["hypotheses"][0]
        print(f"- {c['cluster']:<24} sim {c['simulator_label']:<14} top {c['top_label']:<14} "
              f"{h.get('verdict', '-'):<9} {h['cause'][:70]}")
    print(f"report: runs/{out['run_id']}/report.json  ({out['wall_time_s']} s, backend {out['backend']})")


if __name__ == "__main__":
    main()
