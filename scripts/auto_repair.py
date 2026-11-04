"""Repair what ordinary runs got wrong: cluster failures, keep oracle-solvable ones, investigate, explore, gate.

    python scripts/auto_repair.py runs/<id1> runs/<id2> --backend spark
    python scripts/auto_repair.py runs/<id> --min-failures 5 --dry-run

Writes runs/<time>_auto_repair/report.json (clusters, investigations, repair outcomes).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.llm import BACKENDS, ModelClient  # noqa: E402
from kintsugi.auto_repair import auto_repair  # noqa: E402
from kintsugi.runner import default_workers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories with trials/*.json")
    ap.add_argument("--backend", choices=BACKENDS, default=None)
    ap.add_argument("--min-failures", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default=None, help="only trial records whose tag contains this (e.g. probe_off)")
    ap.add_argument("--min-fail-rate", type=float, default=None,
                    help="skip clusters whose failure rate is surely below this (as the Studio does with 0.10)")
    a = ap.parse_args()
    runs = [str(d) for r in a.runs for d in ([pathlib.Path(r)] + sorted(pathlib.Path(r).glob("*/")))
            if any(pathlib.Path(d).glob("trials/*.json"))]          # a benchmark run nests one run per object
    rep = auto_repair(runs, a.workers, ModelClient(a.backend), a.min_failures, a.max_cases, a.dry_run, tag=a.tag,
                      explorer_kwargs={"min_fail_rate": a.min_fail_rate} if a.min_fail_rate else None)
    print(json.dumps(rep["clusters"], indent=1, default=str))
    print(f"report: runs/{rep['run_id']}/report.json")


if __name__ == "__main__":
    main()
