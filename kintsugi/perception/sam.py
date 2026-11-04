"""SAM 3.1 (Meta) through the Meta Model API: used to find objects in a real photo (kintsugi/twin.py).

Planned use in the simulation loop (CLAUDE.md, key decisions): SAM at key moments only; bulk experiments keep the
simulator's exact masks. Per-pixel masks are not decoded here: their compressed encoding is documented only through
Meta's parser package (meta-sam-parser), which is not installed. The photo path uses SAM's boxes, which come as plain
text; the raw mask tokens are kept in the result. So Perception(mode="sam") still refuses to start
(`availability()`), instead of silently falling back to simulator masks.

API (dev.meta.ai/docs/sam, checked 2026-11-03): POST {base}/responses, model "sam-3.1", one short noun phrase per
request plus the image (data URL). Output text: "<0f>" then records "ID <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|>
<|mask;...|>", box in source pixels (x2, y2 inclusive), c = confidence when metadata.include_confidence = "true".
"""
from __future__ import annotations

import base64
import json
import pathlib
import re
import time
import urllib.error
import urllib.request

MODEL = "sam-3.1"
_BOX = re.compile(r"(\d+)\s*<\|box;([^|]*)\|>")


def _access():
    from .. import settings
    base = settings.get("sam_api_base") or settings.get("meta_api_base")
    key = settings.get("sam_api_key") or settings.get("meta_api_key")
    return base, key


def boxes_available() -> tuple[bool, str]:
    """SAM boxes for photos: needs Meta API access (its own SAM key, or the Meta key)."""
    base, key = _access()
    return (True, "") if base and key else (False, "Meta API access is not set up (add the address and key in Settings)")


def availability() -> tuple[bool, str]:
    """SAM masks in the simulation loop (job setting masks = "sam")."""
    ok, why = boxes_available()
    if not ok:
        return False, why
    return False, ("SAM is used for photos (boxes); per-pixel masks inside the simulation loop are not connected: "
                   "the simulator's own masks are exact there")


def parse(text: str) -> list[dict]:
    """Records of the first frame: [{id, box: [x1, y1, x2, y2], frame: (w, h), conf}]."""
    out = []
    for oid, fields in _BOX.findall(text or ""):
        kv = dict(f.split("=", 1) for f in fields.split(";") if "=" in f)
        try:
            box = [int(kv["x1"]), int(kv["y1"]), int(kv["x2"]), int(kv["y2"])]
        except (KeyError, ValueError):
            continue
        out.append({"id": int(oid), "box": box, "frame": (int(kv.get("w", 0)), int(kv.get("h", 0))),
                    "conf": float(kv["c"]) if "c" in kv else None})
    return out


def segment(image_path, phrase: str, min_conf: float = 0.3, timeout: float = 120, retries: int = 2) -> dict:
    """One SAM call: every match of `phrase` in the image. Returns {phrase, matches: [...], latency_s, raw}."""
    base, key = _access()
    if not (base and key):
        raise RuntimeError(boxes_available()[1])
    p = pathlib.Path(image_path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    body = {"model": MODEL, "stream": False, "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"}]}]}
    req = urllib.request.Request(base.rstrip("/") + "/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:            # the API's message, never the request (holds the key)
            msg = f"SAM HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(msg) from None
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(3 * 2 ** attempt)
                continue
            raise RuntimeError(f"SAM unreachable: {type(e).__name__}: {e}") from None
    text = "".join(c.get("text", "") for o in res.get("output", []) for c in o.get("content", []) if c.get("type") == "output_text")
    matches = [m for m in parse(text) if m["conf"] is None or m["conf"] >= min_conf]
    return {"phrase": phrase, "matches": matches, "latency_s": round(time.time() - t0, 2), "raw": text[:4000]}
