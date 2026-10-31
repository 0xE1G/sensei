"""Geometric grasp sampler: which feature_pinch poses fit the gripper on a perceived object.

Candidates are parameter sets of the feature_pinch primitive (positions relative to perceived
features, never object names). For each candidate the perceived point cloud is moved into the
TCP frame and tested against a simple model of the open Panda gripper:

  * opposite surfaces between the finger pads, closer together than the opening (minus a margin)
    and roughly centred, so both fingers touch when they close;
  * nothing where the open fingers sweep down, nothing under the hand, fingertips above the table.

Nothing here runs physics; it only decides which candidates are worth simulating.
"""
from __future__ import annotations

import math

import numpy as np

from ..sim.robot import down_rotation
from ..skills.library import GRASP_POSES, OPEN_WIDTH, feature_grasp_pose

# Panda hand, in the TCP frame: a = closing axis, b = across the pads, h = height above the pad centre.
GRIP = dict(half_open=OPEN_WIDTH / 2, margin=0.008, pad_half_w=0.011, pad_h=(-0.010, 0.015),
            finger_t=0.013, finger_h=(-0.012, 0.045), hand_h=(0.045, 0.13), hand_half=(0.11, 0.035),
            min_contacts=6, max_asym=0.02, max_sweep_pts=2)

SEARCH_PARAMS = ("edge_offset", "angle_deg", "depth", "close_yaw_deg", "tilt_deg")


def search_params(primitive: dict) -> list[str]:
    return list(primitive.get("search_params", SEARCH_PARAMS))


def sample_candidates(primitive: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Uniform samples over the primitive's search parameters; the rest stay at their defaults."""
    base = {k: v["default"] for k, v in primitive["params"].items()}
    out = []
    for _ in range(n):
        p = dict(base)
        for k in search_params(primitive):
            lo, hi = primitive["params"][k]["range"]
            p[k] = float(rng.uniform(lo, hi))
        out.append(p)
    return out


def surface_candidates(primitive: dict, points: np.ndarray, desc: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Candidates whose grasp point lies on a perceived surface point (written in feature coordinates,
    so the same candidate means the same place on another object of the shape group).
    Closing direction and tilt are sampled freely."""
    if primitive["name"] == "axis_pinch":
        return _surface_candidates_axis(primitive, points, desc, n, rng)
    base = {k: v["default"] for k, v in primitive["params"].items()}
    rng_ = {k: primitive["params"][k]["range"] for k in SEARCH_PARAMS}
    c = np.array(desc["center"])
    slope = desc.get("wall_slope", 0.0) if desc.get("hollow") else 0.0
    toward_robot = math.atan2(-c[1], -c[0])
    out = []
    for q in points[rng.choice(len(points), size=n)]:
        depth = float(np.clip(desc["top_z"] - q[2], *rng_["depth"]))
        r = float(np.linalg.norm(q[:2] - c))
        phi = math.atan2(q[1] - c[1], q[0] - c[0])
        p = dict(base)
        p.update(edge_offset=float(np.clip(desc["width"] / 2 - depth * slope - r, *rng_["edge_offset"])),
                 angle_deg=math.degrees((phi - toward_robot + math.pi) % (2 * math.pi) - math.pi), depth=depth,
                 close_yaw_deg=float(rng.uniform(*rng_["close_yaw_deg"])),
                 tilt_deg=float(rng.uniform(*rng_["tilt_deg"])))
        out.append(p)
    return out


def _surface_candidates_axis(primitive, points, desc, n, rng):
    base = {k: v["default"] for k, v in primitive["params"].items()}
    rng_ = {k: primitive["params"][k]["range"] for k in search_params(primitive)}
    c, ay = np.array(desc["center"]), desc.get("axis_yaw", 0.0)
    major, minor = np.array([math.cos(ay), math.sin(ay)]), np.array([-math.sin(ay), math.cos(ay)])
    em, en = desc.get("extent_major", desc["width"]), desc.get("extent_minor", desc["width"])
    out = []
    for q in points[rng.choice(len(points), size=n)]:
        p = dict(base)
        p.update(along=float(np.clip((q[:2] - c) @ major / (em / 2), *rng_["along"])),
                 across=float(np.clip((q[:2] - c) @ minor / (en / 2), *rng_["across"])),
                 depth=float(np.clip(desc["top_z"] - q[2], *rng_["depth"])),
                 close_deg=float(rng.uniform(*rng_["close_deg"])), tilt_deg=float(rng.uniform(*rng_["tilt_deg"])))
        out.append(p)
    return out


def geometric_fit(points: np.ndarray, desc: dict, p: dict, primitive: str = "feature_pinch") -> dict:
    """Test one candidate on one perceived object. Returns ok, the reason if not, and a quality score."""
    g = GRASP_POSES[primitive](desc, p)
    R = down_rotation(g["yaw"], tilt=g["tilt"], tilt_axis="x")
    tcp = np.array([*g["point"], g["z"]])
    # Lowest fingertip corner must not go into the table. The skill clamps the TCP height (MIN_TCP_Z) so an upright
    # gripper's fingertips stop just above it; only tilt can push a corner below.
    corners = [tcp + R @ np.array([sb * GRIP["pad_half_w"], sa * (GRIP["half_open"] + GRIP["finger_t"]),
                                   -GRIP["finger_h"][0]]) for sa in (-1, 1) for sb in (-1, 1)]
    if min(c[2] for c in corners) < -0.002:
        return {"ok": False, "reason": "fingertip_below_table"}
    local = (points - tcp) @ R                 # columns: TCP x (b), y (a, closing), z (approach, down)
    b, a, h = local[:, 0], local[:, 1], -local[:, 2]
    pad_b = np.abs(b) < GRIP["pad_half_w"]
    in_pad = pad_b & (h > GRIP["pad_h"][0]) & (h < GRIP["pad_h"][1]) & (np.abs(a) < GRIP["half_open"])
    n_c = int(in_pad.sum())
    if n_c < GRIP["min_contacts"]:
        return {"ok": False, "reason": "nothing_between_fingers", "n_contacts": n_c}
    a_lo, a_hi = float(a[in_pad].min()), float(a[in_pad].max())
    span, asym = a_hi - a_lo, abs(a_hi + a_lo) / 2
    if span > OPEN_WIDTH - GRIP["margin"]:
        return {"ok": False, "reason": "wider_than_opening", "span": span}
    if asym > GRIP["max_asym"]:
        return {"ok": False, "reason": "off_centre", "asym": asym}
    sweep = (np.abs(b) < GRIP["pad_half_w"] + 0.003) & (h > GRIP["finger_h"][0]) & (h < GRIP["finger_h"][1]) & \
            (np.abs(a) > GRIP["half_open"] - 0.002) & (np.abs(a) < GRIP["half_open"] + GRIP["finger_t"])
    if int(sweep.sum()) > GRIP["max_sweep_pts"]:
        return {"ok": False, "reason": "finger_collision", "sweep_pts": int(sweep.sum())}
    hand = (h > GRIP["hand_h"][0]) & (h < GRIP["hand_h"][1]) & (np.abs(a) < GRIP["hand_half"][0]) & \
           (np.abs(b) < GRIP["hand_half"][1])
    if int(hand.sum()) > GRIP["max_sweep_pts"]:
        return {"ok": False, "reason": "hand_collision"}
    score = min(1.0, n_c / 30) * (1 - asym / GRIP["max_asym"] * 0.5) * (1 - 0.5 * span / OPEN_WIDTH)
    return {"ok": True, "reason": "fits", "n_contacts": n_c, "span": round(span, 4), "asym": round(asym, 4),
            "score": round(float(score), 4)}


def family_of(desc: dict, p: dict, primitive: str = "feature_pinch") -> dict:
    """Describe a candidate by the object feature it uses and how the fingers close.

    anchor: centre / mid / edge (within 2 cm of the outer edge at grasp height)
    closing: radial (across a wall) / tangential (along a wall) / oblique
    The family name says the feature, never the object: an edge grasp on a hollow object is a rim grasp.
    """
    if primitive == "axis_pinch":
        # Place along the long axis (centre or end) and closing relative to it (across, along, oblique).
        pos = "centre" if abs(p["along"]) < 0.35 else "end"
        cd = abs(p["close_deg"])
        closing = "across" if cd < 30 else ("along" if cd > 60 else "oblique")
        return {"anchor": f"axis_{pos}", "closing": closing, "name": f"axis_{pos}_{closing}"}
    g = feature_grasp_pose(desc, p)
    slope = desc.get("wall_slope", 0.0) if desc.get("hollow") else 0.0
    r_edge = desc["width"] / 2 - p["depth"] * slope
    if g["radius"] < 0.25 * desc["width"] / 2:
        anchor = "centre"
    elif r_edge - g["radius"] < 0.02:
        anchor = "rim" if desc.get("hollow") and "rim_radius" in desc else "edge"
    else:
        anchor = "mid"
    cy = abs(p["close_yaw_deg"])
    closing = "radial" if cy < 30 else ("tangential" if cy > 60 else "oblique")
    if anchor == "centre":
        closing = "any"                     # at the centre the closing direction has no meaning wrt the edge
    verb = {"radial": "pinch", "tangential": "clamp", "oblique": "oblique", "any": "grasp"}[closing]
    return {"anchor": anchor, "closing": closing, "name": f"{anchor}_{verb}"}


def shape_group(desc: dict) -> dict:
    """Coarse shape features a card can require (the planner matches these)."""
    return {"hollow": bool(desc.get("hollow")), "rim": "rim_radius" in desc, "width": round(desc["width"], 3)}
