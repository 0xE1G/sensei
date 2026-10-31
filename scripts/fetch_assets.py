"""Download the Franka Emika Panda model from MuJoCo Menagerie into assets/franka_emika_panda.

Usage:  python scripts/fetch_assets.py
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


def main() -> int:
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
