"""Runs one studio job (started by the server): python scripts/studio_job.py runs/<job_id>_studio

The job directory holds job.json (request, task, program, settings); progress goes to events.jsonl, live poses to
live/ (KINTSUGI_LIVE_DIR is set by the server).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kintsugi.studio import run_job  # noqa: E402

if __name__ == "__main__":
    run_job(pathlib.Path(sys.argv[1]).resolve())
