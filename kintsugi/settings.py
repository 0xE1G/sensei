"""Local settings: model access keys and defaults, kept on this machine only.

Stored in config/local.json (git-ignored). An environment variable with the same meaning always wins, so scripts and
headless runs keep working (META_API_KEY, META_API_BASE, META_MODEL, SAM_API_KEY, SAM_API_BASE, KINTSUGI_LLM).
Secrets never leave the machine: `public()` (what the web app sees) shows only whether a key is set and its last four
characters; keys are never logged, cached or written to run records.
"""
from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "local.json"

# name -> (environment variable, secret?, allowed values or None)
FIELDS = {
    "meta_api_key": ("META_API_KEY", True, None),
    "meta_api_base": ("META_API_BASE", False, None),
    "meta_model": ("META_MODEL", False, None),
    "sam_api_key": ("SAM_API_KEY", True, None),
    "sam_api_base": ("SAM_API_BASE", False, None),
    "default_backend": (None, False, ("rule", "spark")),   # the Studio's default only, never scripts
    "default_masks": (None, False, ("sim", "sam")),
}


def _file() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        return {}


# Used when neither the environment nor the settings file sets a value, so a new user only has to paste a key.
DEFAULTS = {"meta_api_base": "https://api.meta.ai/v1", "meta_model": "muse-spark-1.3"}


def get(name: str, default=None):
    env, _, _ = FIELDS[name]
    if env and os.environ.get(env):
        return os.environ[env]
    return _file().get(name) or default or DEFAULTS.get(name)


def source(name: str) -> str | None:
    env, _, _ = FIELDS[name]
    if env and os.environ.get(env):
        return "environment"
    return "settings" if _file().get(name) else None


def save(updates: dict) -> dict:
    """Set fields (an empty string or None removes one). Unknown fields or values are rejected."""
    data = _file()
    for k, v in updates.items():
        if k not in FIELDS:
            raise ValueError(f"unknown setting {k!r}")
        allowed = FIELDS[k][2]
        if v in ("", None):
            data.pop(k, None)
            continue
        v = str(v).strip()
        if allowed and v not in allowed:
            raise ValueError(f"{k} must be one of {allowed}")
        if k.endswith("_base") and not v.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError(f"{k} must be an https:// address")
        data[k] = v
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(data, indent=1))
    try:
        os.chmod(CONFIG, 0o600)                     # owner only, where the OS supports it
    except OSError:
        pass
    return public()


def public() -> dict:
    """What the web app may see: values of plain fields, and for secrets only whether they are set."""
    out = {}
    for k, (env, secret, allowed) in FIELDS.items():
        v = get(k)
        out[k] = ({"set": bool(v), "hint": f"…{v[-4:]}" if v and len(v) > 8 else ("set" if v else None)} if secret
                  else v)
        out[k + "_source"] = source(k)
    return out


def spark_ready() -> bool:
    return bool(get("meta_api_key") and get("meta_api_base"))
