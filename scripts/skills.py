"""Skill libraries: list them, export / import skill packs, contribute local discoveries to the base library.

    python scripts/skills.py list
    python scripts/skills.py export pack.json [names ...]         # default: everything in the user library
    python scripts/skills.py import pack.json [--no-verify]       # items must re-pass the gates here
    python scripts/skills.py contribute [names ...] [--dry-run]   # re-verify against the base library, then move

The base library (skill_library/) ships with the code and is never written at run time; discoveries go to the user
library (user_library/, git-ignored; KINTSUGI_USER_LIBRARY picks another folder). See kintsugi/skill_packs.py.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi import skill_packs as SP  # noqa: E402
from kintsugi.skills import library as L  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    e = sub.add_parser("export")
    e.add_argument("out")
    e.add_argument("names", nargs="*")
    i = sub.add_parser("import")
    i.add_argument("pack")
    i.add_argument("--no-verify", action="store_true", help="write the items unverified (your own pack, same code)")
    i.add_argument("--workers", type=int)
    c = sub.add_parser("contribute")
    c.add_argument("names", nargs="*")
    c.add_argument("--dry-run", action="store_true", help="run the gates, write nothing")
    c.add_argument("--workers", type=int)
    a = ap.parse_args()

    if a.cmd == "list":
        origins = L.card_origins()
        cards = L.load_cards()
        print(f"base library: {L.rel(L.BASE_DIR)}    user library: {L.rel(L.USER_DIR)}")
        for n in sorted(cards):
            c = cards[n]
            print(f"  {origins[n]:4s}  {n:40s} v{c['version']}  {c['kind']:12s} {c['lineage'][-1].get('source', '')}")
        u = SP.user_items()
        for skill, gs in u["patches"].items():
            for g in gs:
                print(f"  user  patch:{skill}/{g}")
        for n in u["composites"]:
            print(f"  user  composite {n}")
        return 0
    if a.cmd == "export":
        SP.export_pack(a.out, a.names or None)
        return 0
    if a.cmd == "import":
        rep = SP.import_pack(a.pack, workers=a.workers, verify=not a.no_verify)
    else:
        rep = SP.contribute(a.names or None, workers=a.workers, dry_run=a.dry_run)
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
