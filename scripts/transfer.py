"""Transfer experiment: library retrieval + verification on an unseen object type vs a from-scratch repair.

    python scripts/transfer.py mug --backend spark
    python scripts/transfer.py cup --no-scratch

Writes runs/<time>_transfer_<template>/report.json (retrieval choice and explanation, verification trials,
from-scratch explorer cost).
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.agents.llm import BACKENDS, ModelClient  # noqa: E402
from kintsugi.runner import default_workers  # noqa: E402
from kintsugi.sim.templates import DEFAULTS  # noqa: E402
from kintsugi.transfer import run_transfer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template", choices=sorted(DEFAULTS))
    ap.add_argument("--backend", choices=BACKENDS, default=None)
    ap.add_argument("--n-verify", type=int, default=10)
    ap.add_argument("--no-scratch", action="store_true", help="skip the from-scratch baseline")
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    rep = run_transfer(a.template, a.workers, ModelClient(a.backend), a.n_verify, not a.no_scratch)
    v, s = rep["verification"], rep.get("scratch")
    print(json.dumps({"retrieval": rep["retrieval"]["output"], "verification": {k: v[k] for k in ("card", "success", "trials", "trusted", "labels")},
                      "scratch": s}, indent=1))
    if s:
        print(f"\nTransfer: {v['trials']} verification trials ({v['success']}/{v['trials']}) vs from scratch: "
              f"{s['trials_total']} trials ({s['trials_search']} search), outcome {s['outcome']}")
    print(f"report: runs/{rep['run_id']}/report.json")


if __name__ == "__main__":
    main()
