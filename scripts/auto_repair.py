"""Repair what ordinary runs got wrong: cluster failures, keep oracle-solvable ones, investigate, explore, gate.

    python scripts/auto_repair.py runs/<id1> runs/<id2> --backend claude_cli
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
    a = ap.parse_args()
    rep = auto_repair(a.runs, a.workers, ModelClient(a.backend), a.min_failures, a.max_cases, a.dry_run, tag=a.tag)
    print(json.dumps(rep["clusters"], indent=1, default=str))
    print(f"report: runs/{rep['run_id']}/report.json")


if __name__ == "__main__":
    main()
