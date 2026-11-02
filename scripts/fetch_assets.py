"""Download the Franka Emika Panda model from MuJoCo Menagerie into assets/franka_emika_panda.

Usage:  python scripts/fetch_assets.py
        python scripts/fetch_assets.py --scanned    # optional: curated Google Scanned Objects (objects/scanned_objects.json)
Needs internet access to api.github.com and raw.githubusercontent.com.
Menagerie is Apache-2.0 licensed: https://github.com/google-deepmind/mujoco_menagerie
"""
import json
import pathlib
import sys
import urllib.request

REPO = "google-deepmind/mujoco_menagerie"
FOLDER = "franka_emika_panda"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / FOLDER


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


def fetch_scanned(force: bool = False) -> int:
    """The curated Google Scanned Objects in objects/scanned_objects.json -> assets/scanned_objects/<folder>/.
    Only the files model.xml references are fetched (raw URLs, no API calls)."""
    import xml.etree.ElementTree as ET
    cat = json.loads((ROOT / "objects" / "scanned_objects.json").read_text())
    base = f"https://raw.githubusercontent.com/{cat['repo']}/{cat['branch']}/models"
    dest_root = ROOT / "assets" / "scanned_objects"
    n_obj = n_bytes = 0
    for name, e in cat["objects"].items():
        dest = dest_root / e["folder"]
        if (dest / "model.xml").exists() and not force:
            continue
        try:
            xml = _get(f"{base}/{e['folder']}/model.xml")
            files = sorted({el.get("file") for el in ET.fromstring(xml).iter() if el.get("file")})
            if any(pathlib.PurePosixPath(f).is_absolute() or ".." in f for f in files):
                raise ValueError("model.xml references files outside its folder")
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                data = _get(f"{base}/{e['folder']}/{f}")
                (dest / f).write_bytes(data)
                n_bytes += len(data)
            (dest / "model.xml").write_bytes(xml)
            n_obj += 1
            print(f"  {name}: {e['folder']} ({len(files)} files)")
        except Exception as ex:                       # one missing object must not stop the rest
            print(f"  {name}: FAILED ({type(ex).__name__}: {ex})")
    print(f"Downloaded {n_obj} objects, {n_bytes / 1e6:.1f} MB, to {dest_root}")
    return 0


def main() -> int:
    if "--scanned" in sys.argv:
        return fetch_scanned("--force" in sys.argv)
    if (DEST / "panda.xml").exists() and "--force" not in sys.argv:
        print(f"Already present: {DEST}  (use --force to re-download)")
        return 0
    n = 0
    for it in _walk(FOLDER):
        rel = pathlib.Path(it["path"]).relative_to(FOLDER)
        if rel.suffix.lower() in {".png", ".jpg"} and rel.parent.name != "assets":
            continue  # skip README images
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(it["download_url"]))
        n += 1
        print(f"  {rel}")
    print(f"Downloaded {n} files to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
