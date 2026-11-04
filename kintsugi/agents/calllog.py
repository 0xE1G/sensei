"""Live log of every Meta model call (Muse Spark, SAM 3.1, Muse Image), so the app can show calls while they happen.

    with live_call("muse-spark-1.3", "task_spec") as rec:
        ...                      # the HTTP request
        rec["tokens"] = 4210     # optional

Each call writes a "start" line and an "end" line (ok, latency, tokens or error) to cache/llm/live_calls.jsonl.
Studio jobs run in their own processes and append to the same file; the server reads its tail (/api/model_calls).
Cached answers make no call and write nothing here.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIVE_LOG = ROOT / "cache" / "llm" / "live_calls.jsonl"
MAX_BYTES = 2_000_000


def _write(rec: dict) -> None:
    try:
        LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
        if LIVE_LOG.exists() and LIVE_LOG.stat().st_size > MAX_BYTES:       # keep it small: start a fresh file
            LIVE_LOG.replace(LIVE_LOG.with_suffix(".old.jsonl"))
        with open(LIVE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass                                     # the log is for display only; never break a call over it


@contextlib.contextmanager
def live_call(model: str, job: str, detail: str = "", endpoint: str = ""):
    """endpoint: the URL the request goes to (no key). The block may set rec["request_id"] (the API's response id)
    and rec["tokens"]."""
    cid, t0 = uuid.uuid4().hex[:12], time.time()
    base = {"id": cid, "model": model, "job": job, "detail": str(detail)[:80], "pid": os.getpid(),
            **({"endpoint": endpoint} if endpoint else {})}
    _write({**base, "event": "start", "t": t0})
    rec: dict = {}
    try:
        yield rec
    except BaseException as e:
        _write({**base, "event": "end", "t": time.time(), "ok": False, "latency_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"[:160]})
        raise
    _write({**base, "event": "end", "t": time.time(), "ok": True, "latency_s": round(time.time() - t0, 2),
            **({"tokens": rec["tokens"]} if rec.get("tokens") else {}),
            **({"request_id": str(rec["request_id"])[:80]} if rec.get("request_id") else {})})


def recent(since: float = 0.0, limit: int = 12) -> list[dict]:
    """Calls that started at or after `since` (server time), newest last; each merged from its start and end lines."""
    if not LIVE_LOG.exists():
        return []
    with open(LIVE_LOG, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    calls: dict[str, dict] = {}
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = calls.setdefault(r["id"], {"id": r["id"], "model": r.get("model"), "job": r.get("job"),
                                       "detail": r.get("detail", ""), "endpoint": r.get("endpoint")})
        if r.get("event") == "start":
            c["start"] = r["t"]
        else:
            c.update(end=r["t"], ok=r.get("ok"), latency_s=r.get("latency_s"), tokens=r.get("tokens"),
                     error=r.get("error"), request_id=r.get("request_id"))
    out = [c for c in calls.values() if c.get("start", 0) >= since]
    out.sort(key=lambda c: c["start"])
    now = time.time()
    for c in out:                                # a start with no end after 15 min: the process died
        if "end" not in c and now - c["start"] > 900:
            c.update(end=None, ok=False, error="no answer recorded")
    return out[-limit:]
