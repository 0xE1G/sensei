"""SAM 3 masks (Meta), the alternative to the simulator's own segmentation masks.

Planned use (CLAUDE.md, key decisions): SAM is called only at key moments (tag objects at the start, before a grasp,
after a place, on failure frames, ~3-5 calls per episode) and objects are tracked by depth clusters in between; bulk
experiments keep simulator masks. The client is written once access arrives and its API format is known; until then
`availability()` says why the option cannot be used, and Perception(mode="sam") refuses to start instead of silently
falling back to simulator masks.
"""
from __future__ import annotations


def availability() -> tuple[bool, str]:
    from .. import settings
    if not (settings.get("sam_api_key") and settings.get("sam_api_base")):
        return False, "SAM 3 access is not set up (add its address and key in Settings)"
    return False, ("SAM 3 access is saved, but its client is not connected yet: it is written and checked once the "
                   "API format is confirmed")
