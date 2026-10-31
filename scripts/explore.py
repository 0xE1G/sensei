"""Repair loop: find the failure cluster of a task, explore grasps, tune, gate, and promote a new skill card.

    python scripts/explore.py bowl_on_tray
    python scripts/explore.py bowl_on_tray --quick          # smaller budget, for a fast check
    python scripts/explore.py bowl_on_tray --dry-run        # everything except writing the card
    python scripts/explore.py bowl_on_tray --from-seed      # start from the hand-written seed library
    python scripts/explore.py --from-search runs/<id>_failsearch/report.json --cluster cylinder/top_grasp
                                                            # repair a cluster the failure search found
    python scripts/explore.py --from-search ... --cluster cylinder/centre_grasp --hints runs/<id>_investigate/report.json
                                                            # start from the investigator's confirmed fixes

Writes runs/<time>_explore_<task>/report.json (success map, families, tuning history, gates) and, if the
gates pass, skill_library/<family>.json with its lineage. The hand-written rim pinch in
skill_library/reference/ is never used here.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.runner import default_workers  # noqa: E402
from kintsugi.search.explore import Explorer  # noqa: E402
from kintsugi.skills.library import load_cards, seed_library  # noqa: E402
from kintsugi.skills.router import load_performance  # noqa: E402
from kintsugi.tasks.specs import TASKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", choices=sorted(TASKS), help="task to repair (omit with --from-search)")
    ap.add_argument("--from-search", help="failure-search report.json to take the cluster from")
    ap.add_argument("--cluster", help="cluster key from that report, e.g. cylinder/top_grasp")
    ap.add_argument("--hints", help="investigate report.json: confirmed one-change fixes for this cluster seed the "
                                    "parameter-patch search")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="cards left out of the snapshot (with their routing evidence), e.g. to redo a repair")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--quick", action="store_true", help="small budget (fewer candidates, trials and gate seeds)")
    ap.add_argument("--dry-run", action="store_true", help="run all phases but do not write the promoted card")
    ap.add_argument("--seed", type=int, default=0, help="sampler seed")
    ap.add_argument("--from-seed", action="store_true",
                    help="trials use the seed library (as before any exploration), not the current one")
    a = ap.parse_args()
    task, task_kwargs = a.task, None
    if a.from_search:
        rep = json.loads(pathlib.Path(a.from_search).read_text())
        cl = next((c for c in rep["clusters"] if c["key"] == a.cluster), None)
        if cl is None:
            ap.error(f"cluster '{a.cluster}' not in report; have: {[c['key'] for c in rep['clusters']]}")
        task, task_kwargs = "probe", {"region": {"template": cl["template"], "params": cl["region"],
                                                 **({"obstacle": cl["obstacle"]} if cl.get("obstacle") else {})}}
        print(f"repairing {cl['key']}: {cl['count']} failures ({cl['labels']}) in region {cl['region']}")
    elif not task:
        ap.error("give a task or --from-search with --cluster")
    kw = dict(n_before=12, n_candidates=400, per_family=3, family_seeds=3, tune_trials=12, tune_seeds=4,
              n_dev=8, n_regression=4) if a.quick else {}
    library = seed_library() if a.from_seed else load_cards()
    perf = {} if a.from_seed else load_performance()
    hints = []
    if a.hints:
        inv = json.loads(pathlib.Path(a.hints).read_text())
        for c in inv["cases"]:
            if a.cluster and c.get("cluster") != a.cluster:
                continue
            hints += [h["test"] for h in c.get("hypotheses", []) if h.get("verdict") == "confirmed"]
        print(f"{len(hints)} confirmed fixes from the investigator: "
              f"{sorted({str(h.get('skill')) + '.' + str(h.get('param')) + '=' + str(h.get('value')) for h in hints})}")
    library = {n: c for n, c in library.items() if n not in a.exclude}
    perf = {n: g for n, g in perf.items() if n not in a.exclude}
    ex = Explorer(task, workers=a.workers, seed=a.seed, library=library, performance=perf, task_kwargs=task_kwargs,
                  patches={} if a.from_seed else None, hints=hints, **kw)
    if a.dry_run:
        ex.promote = lambda card, *rest: ex.report.update(promoted={"card": card, "path": None, "dry_run": True})
        ex.promote_routing = lambda name, *rest: ex.report.update(routed={"card": name, "dry_run": True})
        ex.promote_patch = lambda chosen, *rest: ex.report.update(patched={"name": chosen["name"], "dry_run": True})
    rep = ex.run()
    print(json.dumps({k: rep.get(k) for k in ("run_id", "outcome", "trials_simulated", "wall_time_s")}, indent=1))
    print(f"report: runs/{rep['run_id']}/report.json")


if __name__ == "__main__":
    main()
