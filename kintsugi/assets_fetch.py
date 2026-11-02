"""Download the third-party assets this project uses (never committed; assets/ is git-ignored).

    robot    Franka Emika Panda from MuJoCo Menagerie (Apache-2.0) -> assets/franka_emika_panda. Required.
    scanned  the curated Google Scanned Objects in objects/scanned_objects.json (meshes CC-BY 4.0, MJCF MIT)
             -> assets/scanned_objects/<folder>/. Optional: only the files each model.xml references.

Used by scripts/fetch_assets.py and by the web app's download buttons (server/app.py), with a progress callback
progress(done, total, label). Needs api.github.com (robot file list) and raw.githubusercontent.com.
"""
from __future__ import annotations

import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "google-deepmind/mujoco_menagerie"
FOLDER = "franka_emika_panda"
ROBOT_DIR = ROOT / "assets" / FOLDER
SCANNED_DIR = ROOT / "assets" / "scanned_objects"
CATALOGUE = ROOT / "objects" / "scanned_objects.json"
SOURCES = {"robot": {"what": "Franka Emika Panda robot model", "from": f"github.com/{REPO}", "license": "Apache-2.0",
                     "size_mb": 33, "required": True},
           "scanned": {"what": "Household objects (Google Scanned Objects)", "license": "CC-BY 4.0 meshes, MIT MJCF",
                       "size_mb": 50, "required": False}}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kintsugi-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _walk(path: str):
    items = json.loads(_get(f"https://api.github.com/repos/{REPO}/contents/{path}"))
    for it in items:
        if it["type"] == "dir":
            yield from _walk(it["path"])
        else:
            yield it


def status() -> dict:
    cat = json.loads(CATALOGUE.read_text()) if CATALOGUE.exists() else {"objects": {}}
    have = [n for n, e in cat["objects"].items() if (SCANNED_DIR / e["folder"] / "model.xml").exists()]
    return {"robot": {**SOURCES["robot"], "present": (ROBOT_DIR / "panda.xml").exists()},
            "scanned": {**SOURCES["scanned"], "from": f"github.com/{cat.get('repo', '?')}", "present": len(have),
                        "total": len(cat["objects"])}}


def fetch_robot(progress=None, force: bool = False) -> int:
    if (ROBOT_DIR / "panda.xml").exists() and not force:
        return 0
    files = [it for it in _walk(FOLDER)
             if not (pathlib.PurePosixPath(it["path"]).suffix.lower() in {".png", ".jpg"}
                     and pathlib.PurePosixPath(it["path"]).parent.name != "assets")]      # skip README images
    for i, it in enumerate(files):
        rel = pathlib.PurePosixPath(it["path"]).relative_to(FOLDER)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unexpected path {it['path']}")
        out = ROBOT_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(it["download_url"]))
        if progress:
            progress(i + 1, len(files), str(rel))
    return len(files)


def fetch_scanned(progress=None, force: bool = False) -> dict:
    """Returns {"objects": downloaded, "bytes": n, "failed": {name: reason}}; one failed object does not stop the rest."""
    import xml.etree.ElementTree as ET
    cat = json.loads(CATALOGUE.read_text())
    base = f"https://raw.githubusercontent.com/{cat['repo']}/{cat['branch']}/models"
    out = {"objects": 0, "bytes": 0, "failed": {}}
    items = list(cat["objects"].items())
    for i, (name, e) in enumerate(items):
        dest = SCANNED_DIR / e["folder"]
        if not ((dest / "model.xml").exists() and not force):
            try:
                xml = _get(f"{base}/{e['folder']}/model.xml")
                files = sorted({el.get("file") for el in ET.fromstring(xml).iter() if el.get("file")})
                if any(pathlib.PurePosixPath(f).is_absolute() or ".." in f for f in files):
                    raise ValueError("model.xml references files outside its folder")
                dest.mkdir(parents=True, exist_ok=True)
                for f in files:
                    data = _get(f"{base}/{e['folder']}/{f}")
                    (dest / f).write_bytes(data)
                    out["bytes"] += len(data)
                (dest / "model.xml").write_bytes(xml)       # written last: its presence means the object is complete
                out["objects"] += 1
            except Exception as ex:
                out["failed"][name] = f"{type(ex).__name__}: {ex}"
        if progress:
            progress(i + 1, len(items), name)
    return out
