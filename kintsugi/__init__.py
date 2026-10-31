"""Kintsugi: a robot that turns task failures into verified, reusable skills."""
import os as _os
import platform as _platform

# Off-screen OpenGL must be chosen before mujoco is first imported.
# Windows/macOS: nothing to do. Headless Linux: EGL unless the user chose otherwise
# (e.g. run under `xvfb-run` with MUJOCO_GL=glfw when EGL is not available).
if _platform.system() == "Linux" and "MUJOCO_GL" not in _os.environ and not _os.environ.get("DISPLAY"):
    _os.environ["MUJOCO_GL"] = "egl"
