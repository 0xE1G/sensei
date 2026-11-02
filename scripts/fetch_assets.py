"""Download the Franka Emika Panda model from MuJoCo Menagerie into assets/franka_emika_panda.

Usage:  python scripts/fetch_assets.py
        python scripts/fetch_assets.py --scanned    # optional: curated Google Scanned Objects (objects/scanned_objects.json)
Needs internet access to api.github.com and raw.githubusercontent.com. The web app has the same downloads as
buttons (kintsugi/assets_fetch.py).
Menagerie is Apache-2.0 licensed: https://github.com/google-deepmind/mujoco_menagerie
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi import assets_fetch as F  # noqa: E402


def main() -> int:
    force = "--force" in sys.argv
    if "--scanned" in sys.argv:
        r = F.fetch_scanned(lambda i, n, name: print(f"  {i}/{n} {name}", flush=True), force)
        for name, why in r["failed"].items():
            print(f"  {name}: FAILED ({why})")
        print(f"Downloaded {r['objects']} objects, {r['bytes'] / 1e6:.1f} MB, to {F.SCANNED_DIR}")
        return 0
    if (F.ROBOT_DIR / "panda.xml").exists() and not force:
        print(f"Already present: {F.ROBOT_DIR}  (use --force to re-download)")
        return 0
    n = F.fetch_robot(lambda i, n, rel: print(f"  {rel}", flush=True), force)
    print(f"Downloaded {n} files to {F.ROBOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
