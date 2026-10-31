"""Frozen object splits (experiments/splits.json): train for tuning, dev for the promotion gate, test run once.

A split is a set of parameter bands per template plus its own block of seeds, so the same seed number
never produces the same object in two splits.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

SPLITS_FILE = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "splits.json"
SPLITS = json.loads(SPLITS_FILE.read_text())
NAMES = ("train", "dev", "test")


def split_seed(split: str, i: int) -> int:
    return SPLITS["seed_base"][split] + i


def split_seeds(split: str, n: int, start: int = 0) -> list[int]:
    return [split_seed(split, start + i) for i in range(n)]


def which_split(seed: int) -> str:
    return max((b, s) for s, b in SPLITS["seed_base"].items() if seed >= b)[1]


def sample_params(template: str, split: str, rng: np.random.Generator) -> dict:
    """Draw one object's parameters from a split's bands (uniform over the union, by band width)."""
    out = {}
    for k, bands in SPLITS[template][split].items():
        w = np.array([hi - lo for lo, hi in bands], float)
        lo, hi = bands[rng.choice(len(bands), p=w / w.sum())]
        out[k] = float(rng.uniform(lo, hi))
    return out
