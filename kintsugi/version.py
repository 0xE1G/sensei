"""Code and config identity for trial records (reproducibility)."""
from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def code_version() -> dict:
    """git commit (if any) plus a content hash of the source and skill library, so uncommitted work is identified."""
    h = hashlib.sha1()
    for p in sorted([*ROOT.glob("kintsugi/**/*.py"), *ROOT.glob("skill_library/**/*.json"),
                     ROOT / "experiments" / "splits.json"]):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    out = {"code_hash": h.hexdigest()[:12], "git": None, "dirty": None}
    try:
        out["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, timeout=5).stdout.strip() or None
        if out["git"]:
            out["dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                               capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
