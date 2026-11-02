"""Grasp, transport and place skills, defined relative to perceived object features.

Each skill reads only the perception descriptor and the robot's own sensors
(TCP pose, finger width). Each ends with an observable check. Parameters and
their allowed ranges live in skill cards (skill_library/*.json).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np

from ..sim.robot import down_rotation

# Two libraries with the same layout. The BASE library (skill_library/, tracked) ships with the code and is read-only
# at run time; everything this checkout discovers (cards, new versions, routing evidence, patches, composites) goes to
# the USER library (user_library/, git-ignored; KINTSUGI_USER_LIBRARY overrides the folder). Reads merge both, the
# user entry winning (a card by name; evidence and patches per (skill, shape group)). So local learning never edits
# shipped files and a library update never overwrites local learning. Skill packs move items between libraries
# (kintsugi.skill_packs: export / import; contribute = re-verify through the gates, then add to the base).
BASE_DIR = pathlib.Path(__file__).resolve().parents[2] / "skill_library"
LIB_DIR = BASE_DIR                       # reference/ and primitives/ (hand-written, never discovered) live only here
USER_DIR = pathlib.Path(os.environ.get("KINTSUGI_USER_LIBRARY") or BASE_DIR.parent / "user_library")
SAFE_Z = 0.28           # transport height for the TCP
MIN_TCP_Z = 0.012       # keep fingertips off the table
OPEN_WIDTH = 0.08


@dataclass
class SkillResult:
    skill: str
    ok: bool
    check: str = ""
    info: dict = field(default_factory=dict)


def _load_dir(lib_dir) -> dict:
    cards = {}
    for p in sorted(pathlib.Path(lib_dir).glob("*.json")):
        c = json.loads(p.read_text())
        if "name" in c and "kind" in c:            # skip non-card files such as performance.json
            cards[c["name"]] = c
    return cards


def load_cards(lib_dir=None) -> dict:
    """Active skill cards (top level of a library folder; history/, reference/, primitives/ are not active).
    Without lib_dir: the base library merged with the user library (a user card replaces the base card of that name)."""
    if lib_dir is not None:
        return _load_dir(lib_dir)
    return {**_load_dir(BASE_DIR), **_load_dir(USER_DIR)}


def rel(path) -> str:
    """A library path for reports: relative to the repository when it lies inside it."""
    path = pathlib.Path(path)
    return path.relative_to(BASE_DIR.parent).as_posix() if path.is_relative_to(BASE_DIR.parent) else str(path)


def card_origins() -> dict:
    """{card name: "base" | "user"} for the merged library."""
    return {**{n: "base" for n in _load_dir(BASE_DIR)}, **{n: "user" for n in _load_dir(USER_DIR)}}


def load_merged(filename: str, user_dir=None) -> dict:
    """A two-level evidence file (performance.json, patches.json): {skill: {group: entry}}, base merged with user."""
    out = {}
    for d in (BASE_DIR, pathlib.Path(user_dir or USER_DIR)):
        p = d / filename
        if p.exists():
            for k, groups in json.loads(p.read_text()).items():
                out.setdefault(k, {}).update(groups)
    return out


def save_merged(filename: str, data: dict, user_dir=None):
    """Write `data` (a full merged view) as the user library's file: only the entries that differ from the base."""
    base_p = BASE_DIR / filename
    base = json.loads(base_p.read_text()) if base_p.exists() else {}
    user = {}
    for k, groups in data.items():
        for g, e in groups.items():
            if base.get(k, {}).get(g) != e:
                user.setdefault(k, {})[g] = e
    d = pathlib.Path(user_dir or USER_DIR)
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps(user, indent=1, sort_keys=True))


def seed_library(lib_dir=None) -> dict:
    """The hand-written starting library: active cards whose first version was a seed, at that version."""
    lib = pathlib.Path(lib_dir or BASE_DIR)
    out = {}
    for name, c in load_cards(lib).items():
        if c["lineage"][0]["source"] != "seed":
            continue
        if c["version"] > 1 and (lib / "history" / f"{name}_v1.json").exists():
            c = json.loads((lib / "history" / f"{name}_v1.json").read_text())
        out[name] = c
    return out


def load_card_file(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def save_card(card: dict, lib_dir=None) -> pathlib.Path:
    """Write an active card (default: into the user library). The previous version, if any (also a base card of the
    same name), is kept in that library's history/ (never deleted)."""
    lib = pathlib.Path(lib_dir or USER_DIR)
    lib.mkdir(parents=True, exist_ok=True)
    path = lib / f"{card['name']}.json"
    old = json.loads(path.read_text()) if path.exists() else (None if lib_dir else load_cards().get(card["name"]))
    if old:
        (lib / "history").mkdir(exist_ok=True)
        (lib / "history" / f"{old['name']}_v{old['version']}.json").write_text(json.dumps(old, indent=2))
    path.write_text(json.dumps(card, indent=2))
    return path


def grasp_impl(card: dict):
    """The function that runs a grasp card. Discovered cards reuse a primitive via "impl"."""
    return GRASP_SKILLS[card.get("impl", card["name"])]


def card_params(card: dict, overrides: dict | None = None) -> dict:
    out = {k: v["default"] for k, v in card["params"].items()}
    for k, v in (overrides or {}).items():
        if k in card["params"]:
            lo, hi = card["params"][k]["range"]
            out[k] = float(np.clip(v, lo, hi))
    return out


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _gripper_yaw(yaw, symmetry=math.pi):
    """Fold a grasp yaw into the range closest to the arm's neutral wrist angle."""
    y = _wrap_pi(yaw)
    while y > symmetry / 2:
        y -= symmetry
    while y < -symmetry / 2:
        y += symmetry
    return y


def _grasp_close_check(robot, before_width):
    w = robot.finger_width()
    holding = 0.003 < w < OPEN_WIDTH - 0.004
    return holding, w


HAND_HALF = (0.035, 0.11)      # Panda hand footprint: half width across the fingers, half length along them (m)


def _hand_clearance(c, yaw, z_g, neighbors):
    """Smallest gap between the hand's footprint at grasp height and any neighbour tall enough to reach the palm."""
    along = np.array([math.sin(yaw), -math.cos(yaw)])          # closing direction = the hand's long side
    across = np.array([math.cos(yaw), math.sin(yaw)])
    gap = 1.0
    for n in neighbors or []:
        if n["height"] < z_g + 0.04:                           # below the palm: the fingers pass over it
            continue
        d = np.asarray(n["center"]) - c
        dx = max(abs(d @ across) - HAND_HALF[0], 0.0)
        dy = max(abs(d @ along) - HAND_HALF[1], 0.0)
        gap = min(gap, math.hypot(dx, dy) - n["width"] / 2)
    return gap


# ---------------------------------------------------------------- grasp skills
def top_grasp(robot, desc, p):
    """Fingers come straight down over the centre and close across the object's top.
    clear_side >= 0.5: on a square top either finger orientation works; take the one whose hand stays clear of
    tall neighbours (perceived objects around the target)."""
    c = np.array(desc["center"])
    top, h = desc["top_z"], desc["height"]
    yaw = _gripper_yaw(desc.get("yaw", 0.0) + math.radians(p["yaw_offset_deg"]), math.pi / 2)
    if p.get("clear_side", 0) >= 0.5 and desc.get("extent_major", 1) < 1.3 * desc.get("extent_minor", 1):
        z0 = max(MIN_TCP_Z, top - p["depth_frac"] * h)
        alt = _gripper_yaw(yaw + math.pi / 2, math.pi)
        if _hand_clearance(c, alt, z0, desc.get("neighbors")) > _hand_clearance(c, yaw, z0, desc.get("neighbors")):
            yaw = alt
        # Then slide the grasp point (across the fingers, still on the top) away from the nearest tall neighbour.
        tall = [n for n in desc.get("neighbors") or [] if n["height"] >= z0 + 0.04]
        if tall:
            n = min(tall, key=lambda n: np.linalg.norm(np.asarray(n["center"]) - c))
            across = np.array([math.cos(yaw), math.sin(yaw)])
            side = -np.sign((np.asarray(n["center"]) - c) @ across) or 1.0
            c = c + side * across * max(0.0, min(0.015, desc.get("extent_minor", 0.04) / 2 - 0.012))
    R = down_rotation(yaw)
    z_g = max(MIN_TCP_Z, top - p["depth_frac"] * h)
    robot.open()
    r1 = robot.move_to([*c, top + p["approach_height"]], R, speed=0.35)
    r2 = robot.move_to([*c, z_g], R, speed=p["descend_speed"])
    robot.close(force=p["grip_force"])
    ok, w = _grasp_close_check(robot, OPEN_WIDTH)
    reached = r1["reached"] and r2["reached"]
    return SkillResult("top_grasp", ok and reached, "fingers_closed_on_object" if ok else "fingers_fully_closed",
                       {"finger_width": w, "reached": reached, "pos_err": r2["pos_err"],
                        "grasp_offset": z_g - (top - h), "tcp_z": z_g})


def rim_pinch(robot, desc, p):
    """One finger inside the rim, one outside, closing across the wall."""
    if not desc.get("hollow") or "rim_radius" not in desc:
        return SkillResult("rim_pinch", False, "precondition_no_rim", {})
    c = np.array(desc["center"])
    base = np.zeros(2)
    to_base = base - c
    phi = math.atan2(to_base[1], to_base[0]) + math.radians(p["angle_deg"])
    r_hat = np.array([math.cos(phi), math.sin(phi)])
    # Aim at the middle of the wall at grasp depth (walls of bowls flare outward).
    r = desc["rim_radius"] - p["wall_guess"] / 2 - p["depth"] * desc.get("wall_slope", 0.0)
    pt = c + r * r_hat
    z_g = max(MIN_TCP_Z, desc["rim_z"] - p["depth"])
    R = down_rotation(_gripper_yaw(phi + math.pi / 2), tilt=math.radians(p["tilt_deg"]), tilt_axis="x")
    robot.open()
    r1 = robot.move_to([*pt, desc["rim_z"] + p["approach_height"]], R, speed=0.35)
    r2 = robot.move_to([*pt, z_g], R, speed=p["descend_speed"])
    robot.close(force=p["grip_force"])
    ok, w = _grasp_close_check(robot, OPEN_WIDTH)
    reached = r1["reached"] and r2["reached"]
    return SkillResult("rim_pinch", ok and reached, "fingers_closed_on_object" if ok else "fingers_fully_closed",
                       {"finger_width": w, "reached": reached, "pos_err": r2["pos_err"],
                        "grasp_offset": z_g - (desc["top_z"] - desc["height"]), "tcp_z": z_g,
                        "rim_point": pt.tolist()})


def feature_grasp_pose(desc, p):
    """Where a feature_pinch puts the TCP, from perceived features only. Shared by the skill and the sampler.

    The grasp point sits `edge_offset` inside the object's outer edge at grasp height (for flared walls the
    edge moves in by depth * wall_slope), at azimuth `angle_deg` from the side facing the robot, `depth`
    below the top. The fingers close along the radial direction turned by `close_yaw_deg`
    (0 = across a wall), and the approach leans by `tilt_deg` about the axis normal to the closing direction.
    """
    c = np.array(desc["center"], float)
    slope = desc.get("wall_slope", 0.0) if desc.get("hollow") else 0.0
    r_edge = desc["width"] / 2 - p["depth"] * slope
    r = max(0.0, r_edge - p["edge_offset"])
    phi = math.atan2(-c[1], -c[0]) + math.radians(p["angle_deg"])
    pt = c + r * np.array([math.cos(phi), math.sin(phi)])
    z_g = max(MIN_TCP_Z, desc["top_z"] - p["depth"])
    close_dir = phi + math.radians(p["close_yaw_deg"])
    yaw = _wrap_pi(close_dir + math.pi / 2)
    tilt = math.radians(p["tilt_deg"])
    if abs(yaw) > math.pi / 2:                 # the gripper is symmetric: turn it round, keep the same lean
        yaw = _wrap_pi(yaw + math.pi)
        tilt = -tilt
    return {"point": pt, "z": z_g, "yaw": yaw, "tilt": tilt, "close_dir": close_dir, "radius": r, "phi": phi}


def feature_pinch(robot, desc, p):
    """Generic parallel-jaw grasp placed relative to perceived features. The explorer searches over its
    parameters; promoted variants (for example a rim pinch) are cards that reuse it with tuned values."""
    g = feature_grasp_pose(desc, p)
    R = down_rotation(g["yaw"], tilt=g["tilt"], tilt_axis="x")
    robot.open()
    r1 = robot.move_to([*g["point"], desc["top_z"] + p["approach_height"]], R, speed=0.35)
    r2 = robot.move_to([*g["point"], g["z"]], R, speed=p["descend_speed"])
    robot.close(force=p["grip_force"])
    ok, w = _grasp_close_check(robot, OPEN_WIDTH)
    reached = r1["reached"] and r2["reached"]
    return SkillResult("feature_pinch", ok and reached, "fingers_closed_on_object" if ok else "fingers_fully_closed",
                       {"finger_width": w, "reached": reached, "pos_err": r2["pos_err"],
                        "grasp_offset": g["z"] - (desc["top_z"] - desc["height"]), "tcp_z": g["z"],
                        "grasp_point": g["point"].tolist()})


def axis_grasp_pose(desc, p):
    """Where an axis_pinch puts the TCP: a point `along` / `across` (fractions of the half extents) in the object's
    principal frame, `depth` below the top, fingers closing at `close_deg` from the short axis (0 = squeezing across
    the long axis). The gripper is only turned by 180 deg (which keeps the closing direction), never by 90."""
    c = np.array(desc["center"], float)
    ay = desc.get("axis_yaw", desc.get("yaw", 0.0))
    major, minor = np.array([math.cos(ay), math.sin(ay)]), np.array([-math.sin(ay), math.cos(ay)])
    em, en = desc.get("extent_major", desc["width"]), desc.get("extent_minor", desc["width"])
    pt = c + p["along"] * em / 2 * major + p["across"] * en / 2 * minor
    z_g = max(MIN_TCP_Z, desc["top_z"] - p["depth"])
    close_dir = ay + math.pi / 2 + math.radians(p["close_deg"])
    yaw = _wrap_pi(close_dir + math.pi / 2)
    tilt = math.radians(p["tilt_deg"])
    if abs(yaw) > math.pi / 2:
        yaw = _wrap_pi(yaw + math.pi)
        tilt = -tilt
    return {"point": pt, "z": z_g, "yaw": yaw, "tilt": tilt, "close_dir": close_dir}


def axis_pinch(robot, desc, p):
    """Generic parallel-jaw grasp in the object's principal frame (oriented and elongated shapes)."""
    g = axis_grasp_pose(desc, p)
    R = down_rotation(g["yaw"], tilt=g["tilt"], tilt_axis="x")
    robot.open()
    r1 = robot.move_to([*g["point"], desc["top_z"] + p["approach_height"]], R, speed=0.35)
    r2 = robot.move_to([*g["point"], g["z"]], R, speed=p["descend_speed"])
    robot.close(force=p["grip_force"])
    ok, w = _grasp_close_check(robot, OPEN_WIDTH)
    reached = r1["reached"] and r2["reached"]
    return SkillResult("axis_pinch", ok and reached, "fingers_closed_on_object" if ok else "fingers_fully_closed",
                       {"finger_width": w, "reached": reached, "pos_err": r2["pos_err"],
                        "grasp_offset": g["z"] - (desc["top_z"] - desc["height"]), "tcp_z": g["z"],
                        "grasp_point": g["point"].tolist()})


GRASP_POSES = {"feature_pinch": feature_grasp_pose, "axis_pinch": axis_grasp_pose}


def side_wrap(robot, desc, p):
    """Approach horizontally from the robot's side and close around the body."""
    if desc["width"] > OPEN_WIDTH - 0.008:
        return SkillResult("side_wrap", False, "precondition_too_wide", {"width": desc["width"]})
    c = np.array(desc["center"])
    d = c / (np.linalg.norm(c) + 1e-9)
    z_axis = np.array([d[0], d[1], 0.0])
    y_axis = np.cross([0, 0, 1.0], z_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    z_g = max(0.03, desc["top_z"] - (1 - p["height_frac"]) * desc["height"])
    robot.open()
    pre = c - d * p["approach_dist"]
    robot.move_to([*(c - d * 0.02), SAFE_Z], down_rotation(0.0), speed=0.35)
    r1 = robot.move_to([*pre, z_g], R, speed=0.3)
    r2 = robot.move_to([*c, z_g], R, speed=p["descend_speed"])
    robot.close(force=p["grip_force"])
    ok, w = _grasp_close_check(robot, OPEN_WIDTH)
    reached = r1["reached"] and r2["reached"]
    return SkillResult("side_wrap", ok and reached, "fingers_closed_on_object" if ok else "fingers_fully_closed",
                       {"finger_width": w, "reached": reached, "grasp_offset": z_g - (desc["top_z"] - desc["height"]),
                        "tcp_z": z_g})


# ---------------------------------------------------------------- transport and place
def lift(robot, p, width_at_grasp):
    pos, R = robot.tcp_pose()
    r = robot.move_to([pos[0], pos[1], max(pos[2] + p["height"], SAFE_Z)], R, speed=p["speed"])
    robot.wait(0.3)
    w = robot.finger_width()
    # Still holding if the fingers have not closed to (nearly) nothing since the grasp.
    ok = w > 0.003 and w > 0.4 * width_at_grasp
    return SkillResult("lift", ok and r["reached"], "still_holding" if ok else "fingers_collapsed",
                       {"finger_width": w, "reached": r["reached"]})


def place_in(robot, target_desc, p, grasp_offset, mode="place", slot=(0.0, 0.0)):
    """Carry over the target (plus a slot offset), lower, release, retreat."""
    c = np.array(target_desc["center"]) + np.asarray(slot, float)
    pos, R = robot.tcp_pose()
    yaw = math.atan2(R[1, 0], R[0, 0])
    R_down = down_rotation(_gripper_yaw(yaw))
    top = target_desc.get("rim_z", target_desc["top_z"])
    if mode == "drop":
        z_rel = top + p["drop_height"] + grasp_offset
    else:
        z_rel = top + p["clearance"] + grasp_offset
    r1 = robot.move_to([*c, SAFE_Z], R, speed=p["speed"], watch_grip=True)
    if r1.get("lost_grip"):
        # The object slipped out on the way: stop here (no lowering, no opening over an empty target); the
        # episode labels it and recovery can re-perceive and grasp again.
        return SkillResult("place_in" if mode == "place" else "drop_in", False, "object_lost_in_transport",
                           {"reached": False, "lost_at": r1["lost_at"], "release_z": None})
    r2 = robot.move_to([*c, max(z_rel, MIN_TCP_Z)], R, speed=0.15, watch_grip=True)
    if r2.get("lost_grip"):
        return SkillResult("place_in" if mode == "place" else "drop_in", False, "object_lost_in_transport",
                           {"reached": False, "lost_at": r2["lost_at"], "release_z": None})
    holding = robot.finger_width() > 0.003
    robot.open(wait=0.5)
    robot.move_to([*c, SAFE_Z], R_down, speed=0.3)
    return SkillResult("place_in" if mode == "place" else "drop_in", r1["reached"] and r2["reached"] and holding,
                       "released_over_target" if holding else "nothing_held_at_release",
                       {"reached": r1["reached"] and r2["reached"], "release_z": z_rel})


def go_ready(robot):
    # After a grasp that closed on nothing the fingers may sit inside a hollow object. Opening there
    # would wedge them against its inner walls and lift it, so retreat straight up first, then open.
    pos, R = robot.tcp_pose()
    closed_empty = robot.finger_width() < 0.003
    if not closed_empty:
        robot.open(wait=0.2)
    if pos[2] < SAFE_Z:
        robot.move_to([pos[0], pos[1], SAFE_Z], R, speed=0.3)
    if closed_empty:
        robot.open(wait=0.2)
    return robot.move_to([0.45, 0.0, 0.40], down_rotation(0.0), speed=0.4)


GRASP_SKILLS = {"top_grasp": top_grasp, "rim_pinch": rim_pinch, "side_wrap": side_wrap, "feature_pinch": feature_pinch,
                "axis_pinch": axis_pinch}
