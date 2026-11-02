"""Objects from files: drop a MuJoCo MJCF (.xml) or URDF (.urdf) model into objects/ and it becomes a template.

    objects/<name>.xml | <name>.urdf     the model: everything inside ONE top-level body (no free joint needed)
    objects/<name>.json (optional)       {"description": "...", "fixed": false, "hollow": false,
                                          "articulated": {"part": "drawer", "joint": "<joint name in the file>"},
                                          "parts": {"handle": "<geom name prefix>", "front": "<geom name prefix>"}}

Nothing downstream depends on the template name: the object is described by perception like any other, its size
comes from the model's own geometry, and a moving part is described by its joint (slide -> metres, hinge ->
degrees) and its part name. Skills that open or close are named after the part ("open_drawer"), so an imported
cabinet with a drawer reuses the library's drawer skills. Parts are found by geom names: geoms whose names start with
"handle" / "front" (or the prefixes given in "parts") become the handle / front perception describes. Articulated
objects are placed like cabinets: at the back of the reach, their local -x side (the front) facing the robot.
"""
from __future__ import annotations

import functools
import json
import math
import pathlib

import mujoco
import numpy as np

ASSET_DIR = pathlib.Path(__file__).resolve().parents[2] / "objects"
SCANNED_DIR = ASSET_DIR.parent / "assets" / "scanned_objects"   # downloaded by scripts/fetch_assets.py --scanned
CATALOGUE = ASSET_DIR / "scanned_objects.json"
MAX_OPEN_DEG = 95.0          # a hinged part starting "fully open" is opened this far (if its range allows)


@functools.lru_cache(maxsize=1)
def assets() -> dict:
    """name -> {path, meta (from the json sidecar), size (measured from the model), joint (if articulated)}."""
    out = {}
    if not ASSET_DIR.exists():
        return out
    files = []
    for path in sorted(list(ASSET_DIR.glob("*.xml")) + list(ASSET_DIR.glob("*.urdf"))):
        side = path.with_suffix(".json")
        files.append((path.stem.lower().replace("-", "_").replace(" ", "_"), path,
                      json.loads(side.read_text()) if side.exists() else {}))
    if CATALOGUE.exists():                          # curated scanned objects, only those already downloaded
        for name, e in json.loads(CATALOGUE.read_text()).get("objects", {}).items():
            path = SCANNED_DIR / e["folder"] / "model.xml"
            if path.exists():
                files.append((name, path, {k: v for k, v in e.items() if k != "folder"}))
    for name, path, meta in files:
        try:
            out[name] = {"path": str(path), "meta": meta, **_measure(path, meta)}
        except Exception as e:                      # a broken file must not break the built-in templates
            print(f"[assets] skipped {path.name}: {type(e).__name__}: {e}")
    return out


def _spec(path, meta=None) -> mujoco.MjSpec:
    """The file's spec with asset files made absolute (they must resolve after attaching into the scene) and, for a
    mesh model with a visual-only copy (contype = conaffinity = 0), mass carried by the collision geoms only."""
    spec = mujoco.MjSpec.from_file(str(path))
    spec.compiler.fusestatic = False                 # URDF default: compiling would merge the joint-less body away
    base = pathlib.Path(path).parent
    mdir = base / (spec.meshdir or "")
    tdir = base / (spec.texturedir or "")
    for m_ in spec.meshes:
        if m_.file and not pathlib.Path(m_.file).is_absolute():
            m_.file = str((mdir / m_.file).resolve())
    for t_ in spec.textures:
        if t_.file and not pathlib.Path(t_.file).is_absolute():
            t_.file = str((tdir / t_.file).resolve())
    spec.meshdir, spec.texturedir = "", ""
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            g.mass = 0.0
    return spec


def _scale_mass(spec, target: float):
    """Set the collision geoms' density so the object weighs `target` kg (scanned meshes carry no real mass)."""
    m = spec.compile()
    total = float(m.body_subtreemass[0])
    if total <= 0:
        return
    for g in spec.geoms:
        if not (g.contype == 0 and g.conaffinity == 0):
            g.density = g.density * target / total


def _texture_color(m) -> str | None:
    """Nearest named colour to the median texel of the model's first texture (for the object's 'color')."""
    if m.ntex == 0:
        return None
    from .templates import COLORS
    a, n = m.tex_adr[0], m.tex_width[0] * m.tex_height[0] * m.tex_nchannel[0]
    px = np.asarray(m.tex_data[a:a + n], float).reshape(-1, m.tex_nchannel[0])[:, :3] / 255.0
    px = px[(px.max(1) > 0.03)]                      # unused atlas area is black
    if len(px) == 0:
        return None
    c = np.median(px, 0)
    return min(COLORS, key=lambda k: float(np.sum((np.asarray(COLORS[k][:3]) - c) ** 2)))


def _top_body(spec):
    bodies = [b for b in spec.worldbody.bodies]
    if len(bodies) != 1:
        raise ValueError(f"the model must have exactly one top-level body (found {len(bodies)})")
    return bodies[0]


def _measure(path, meta) -> dict:
    """Size from the compiled model alone: bounding box of all geoms (in the top body's frame) and the moving joint."""
    spec = _spec(path, meta)
    if meta.get("mass"):
        _scale_mass(spec, float(meta["mass"]))
    _top_body(spec).add_freejoint()                  # loose here (as in a scene), so a static body is not fused away
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:   # exact: a mesh's box is loose once rotated
            mid = m.geom_dataid[g]
            w = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ R.T + p
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
            continue
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = p + R @ (c + h * np.array([sx, sy, sz]))
                    lo, hi = np.minimum(lo, w), np.maximum(hi, w)
    info = {"lo": lo.tolist(), "hi": hi.tolist(), "height": float(hi[2] - lo[2]),
            "width": float(max(hi[0] - lo[0], hi[1] - lo[1])),
            "footprint": float(0.5 * math.hypot(hi[0] - lo[0], hi[1] - lo[1])),
            "mass": float(m.body_subtreemass[0]), "color": meta.get("color") or _texture_color(m)}
    art = meta.get("articulated")
    if art:
        j = m.joint(art["joint"])
        typ = {int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}.get(int(j.type[0]))
        if typ is None:
            raise ValueError(f"joint {art['joint']} must be a slide or hinge joint")
        r = [float(v) for v in m.jnt_range[j.id]]
        info["joint"] = {"part": art.get("part", "part"), "joint": art["joint"], "body": m.body(j.bodyid[0]).name,
                         "type": typ, "unit": "m" if typ == "slide" else "deg", "range": r,
                         "max": r[1] if typ == "slide" else math.degrees(r[1])}
    return info


def add_asset(body, oid: str, name: str, params: dict) -> dict:
    """Attach the file's model under the object's body (names prefixed "<id>__") and return true-geometry meta."""
    a = assets()[name]
    spec = _spec(a["path"], a["meta"])
    top = _top_body(spec)
    if params.get("mass"):                           # catalogue mass, possibly perturbed by an experiment
        _scale_mass(spec, float(params["mass"]))
    if "friction" in params:                         # same contact model as the built-in objects
        for g in spec.geoms:
            if not (g.contype == 0 and g.conaffinity == 0):
                g.friction = [float(params["friction"]), 0.005, 0.0001]
                g.condim = 4
    prefixes = {"handle": "handle", "front": "front", **(a["meta"].get("parts") or {})}
    for g in spec.geoms:                         # the file's own part names -> the perception convention
        for part, pre in prefixes.items():
            if g.name and g.name.startswith(pre) and not g.name.startswith(part):
                g.name = part + g.name[len(pre):]
    frame = body.add_frame()
    frame.pos = [0.0, 0.0, -a["lo"][2]]          # rest on the table
    frame.attach_body(top, f"{oid}__", "")
    meta = {"height": a["height"], "width": a["width"], "hollow": bool(a["meta"].get("hollow", False)),
            "bottom_offset": 0.0, "half_x": (a["hi"][0] - a["lo"][0]) / 2, "half_y": (a["hi"][1] - a["lo"][1]) / 2,
            "asset": name}
    if meta["hollow"]:                           # an imported container (basket, bin, bowl): things go inside it
        meta.update(rim_z=a["height"], floor_z=min(0.02, 0.2 * a["height"]))
    j = a.get("joint")
    if j:
        full = j["range"][1] if j["type"] == "slide" else min(j["range"][1], math.radians(MAX_OPEN_DEG))
        meta["articulated"] = {"part": j["part"], "joint": f"{oid}__{j['joint']}", "type": j["type"], "unit": j["unit"],
                               "range": j["range"], "body": f"{oid}__{j['body']}",
                               "initial": float(params.get("open", 0.0)) * full}
    return meta
