"""Live pose stream: each worker writes the episode it is simulating, so the web app can show it in 3D while it runs.

Enabled by the environment variable KINTSUGI_LIVE_DIR (set by the studio job; spawned workers inherit it).
Layout of the live directory:
  w<pid>.json          pointer: the worker's current episode (id, meta, done, success)
  <pid>_<n>.jsonl      one episode: {"type": "meta"} / {"type": "scene"} / {"type": "f"} frames / {"type": "e"}
                       events / {"type": "done"}; flushed every few frames
A worker keeps its last KEEP episode files, so a viewer that is still playing one does not lose its tail.
Simulation is never slowed down: the browser buffers frames and plays them at real time.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import time

KEEP = 3
FLUSH_EVERY = 8
_counter = itertools.count()


def live_dir() -> pathlib.Path | None:
    d = os.environ.get("KINTSUGI_LIVE_DIR")
    return pathlib.Path(d) if d else None


def _write_atomic(path: pathlib.Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    for _ in range(5):                 # Windows: the server may be reading the old file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.01)


class LiveWriter:
    def __init__(self, root: pathlib.Path, meta: dict):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.id = f"{self.pid}_{next(_counter)}"
        self.meta = {**meta, "episode": self.id, "worker": self.pid, "started": round(time.time(), 2)}
        self.f = open(self.root / f"{self.id}.jsonl", "w", encoding="utf-8")
        self._n = 0
        self._line({"type": "meta", **self.meta})
        _write_atomic(self.root / f"w{self.pid}.json", {**self.meta, "done": False})

    def _line(self, obj):
        self.f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def scene(self, scene_desc: dict, fps: int):
        self._line({"type": "scene", "fps": fps, **scene_desc})
        self.f.flush()

    def frame(self, fr: dict):
        self._line({"type": "f", **fr})
        self._n += 1
        if self._n % FLUSH_EVERY == 0:
            self.f.flush()

    def event(self, ev: dict):
        self._line({"type": "e", **ev})

    def finish(self, result: dict):
        self._line({"type": "done", **result})
        self.f.close()
        _write_atomic(self.root / f"w{self.pid}.json", {**self.meta, "done": True, **result})
        mine = sorted(self.root.glob(f"{self.pid}_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
        for p in mine[:-KEEP]:
            try:
                p.unlink()
            except OSError:
                pass
