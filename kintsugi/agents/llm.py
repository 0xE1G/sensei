"""One client for every language-model job, with interchangeable backends.

    client = ModelClient()                       # backend from KINTSUGI_LLM (default "rule")
    res = client.complete("task_spec", {"prompt": "Put the bowl on the tray."})
    res.output   # validated JSON          res.meta   # backend, model, prompt version, cached, repairs, latency

Backends (same prompts, same schemas, same validators, so switching is configuration only):
  rule        deterministic stand-ins written in code; used by tests and the smoke test
  claude_cli  Claude through the Claude Code CLI (`claude -p`), for development while the Meta key is pending
  spark       Muse Spark through the Meta API (OpenAI-compatible chat endpoint assumed; verify when the key arrives)

Every call is cached by hash of (job, prompt version, inputs, backend, model) in cache/llm/, and logged (without the
output) to cache/llm/calls.jsonl. An invalid output gets the validator's errors fed back once; a second invalid
output is rejected with the reasons. With KINTSUGI_LLM_DRYRUN=1, calls that would reach a paid backend are counted and
answered by the rule stand-in instead (to budget credits before running an experiment on Spark).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import jsonschema

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "llm"
PROMPT_DIR = pathlib.Path(__file__).resolve().parent / "prompts"
BACKENDS = ("rule", "claude_cli", "spark")
# Spark reasoning effort per job where it was measured (scripts/eval_models.py, 2026-11-02): low effort kept accuracy
# and was 2-3x faster for task specs (20/20 vs 19/20) and plan programs (48/56 = 48/56); diagnosis keeps the model's own
# level (first look 11/14 vs 8/14 at low). Jobs not listed: the model decides until measured.
SPARK_EFFORT = {"task_spec": "low", "plan_program": "low"}
DEFAULT_MODEL = {"rule": "rule-v1", "claude_cli": os.environ.get("KINTSUGI_CLAUDE_MODEL", "sonnet"),
                 "spark": "muse-spark"}     # spark: META_MODEL / the model in the local settings wins (kintsugi.settings)


class ModelRejected(Exception):
    """The model's output failed validation twice (or could not be parsed). `reasons` says why."""

    def __init__(self, job, reasons, raw=None):
        super().__init__(f"{job}: output rejected: {'; '.join(reasons)}")
        self.job, self.reasons, self.raw = job, reasons, raw


@dataclasses.dataclass
class Job:
    """A model job: prompt template, JSON schema, semantic validator, and the rule-based stand-in."""
    name: str
    schema: dict
    render: callable            # inputs -> (system, user)
    validate: callable          # output, inputs -> list of error strings (empty = valid)
    rule: callable              # inputs -> output (deterministic stand-in, same format)
    prompt_version: str = "1"


JOBS: dict[str, Job] = {}


def register(job: Job) -> Job:
    JOBS[job.name] = job
    return job


def load_prompt(name: str) -> tuple[str, str]:
    """Prompt file: first line 'version: N', then the system part, a line '---', then the user template."""
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    head, body = text.split("\n", 1)
    version = head.split(":", 1)[1].strip()
    return version, body


@dataclasses.dataclass
class Result:
    output: dict
    meta: dict


class ModelClient:
    def __init__(self, backend: str | None = None, model: str | None = None, dry_run: bool | None = None,
                 use_cache: bool = True, fallback: bool | None = None, effort=None, schema_output: bool | None = None):
        """spark only:
        fallback       a Spark call that fails (error, timeout, or an answer rejected twice) is re-sent to Claude; the
                       result says so (backend claude_cli, requested_backend spark) and is cached under Claude's key, so
                       it is never counted or reused as a Spark answer. None: the setting spark_fallback (default on);
                       experiments that measure Spark pass False.
        effort         reasoning_effort per request (minimal / low / medium / high / xhigh): one value, or {job: value};
                       None = SPARK_EFFORT (measured per job; else the model decides). Part of the cache key.
        schema_output  ask the API to constrain the answer to the job's JSON schema (response_format json_schema).
                       None: KINTSUGI_SPARK_SCHEMA (default off until measured). Part of the cache key."""
        from .. import settings
        self.backend = backend or os.environ.get("KINTSUGI_LLM", "rule")    # the Studio's default comes from settings
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend '{self.backend}'. Known: {BACKENDS}")
        self.model = model or (settings.get("meta_model") if self.backend == "spark" else None) or DEFAULT_MODEL[self.backend]
        self.dry_run = bool(int(os.environ.get("KINTSUGI_LLM_DRYRUN", "0"))) if dry_run is None else dry_run
        self.use_cache = use_cache
        self.fallback = (settings.get("spark_fallback", "1") != "0") if fallback is None else bool(fallback)
        if effort is None and os.environ.get("KINTSUGI_SPARK_EFFORT"):
            effort = os.environ["KINTSUGI_SPARK_EFFORT"]
        self.effort = effort
        self.schema_output = (os.environ.get("KINTSUGI_SPARK_SCHEMA", "0") == "1") if schema_output is None else schema_output
        self.stats = {"calls": 0, "cached": 0, "would_call": 0, "repairs": 0, "rejected": 0, "fallbacks": 0}

    def _effort(self, job_name):
        if self.backend != "spark":
            return None
        if not self.effort:
            return SPARK_EFFORT.get(job_name)
        return self.effort.get(job_name) if isinstance(self.effort, dict) else self.effort

    # ------------------------------------------------------------------ public
    def complete(self, job_name: str, inputs: dict, images=None) -> Result:
        """images: local image files sent with the request (backends that read images: spark, claude_cli); the
        rule stand-in ignores them. Their content is part of the cache key."""
        job = _job(job_name)
        system, user = job.render(inputs)
        images = [str(i) for i in (images or [])]
        img_hash = [hashlib.sha1(pathlib.Path(i).read_bytes()).hexdigest()[:16] for i in images]
        effort = self._effort(job.name)
        schema = self.backend == "spark" and bool(self.schema_output)
        key = hashlib.sha1(json.dumps([job.name, job.prompt_version, inputs, self.backend, self.model]
                                      + ([img_hash] if img_hash else [])
                                      + ([f"effort={effort}"] if effort else []) + (["schema"] if schema else []),
                                      sort_keys=True, default=str).encode()).hexdigest()
        meta = {"job": job.name, "backend": self.backend, "model": self.model, "prompt_version": job.prompt_version,
                "input_hash": key[:12], "cached": False, "repairs": 0,
                **({"images": len(images)} if images and self.backend != "rule" else {}),
                **({"effort": effort} if effort else {}), **({"schema_output": True} if schema else {})}
        path = CACHE_DIR / f"{job.name}_{key[:20]}.json"
        if self.use_cache and path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))
            self.stats["cached"] += 1
            return Result(hit["output"], {**hit["meta"], "cached": True})
        t0 = time.time()
        if self.backend == "rule" or self.dry_run:
            if self.dry_run and self.backend != "rule":
                self.stats["would_call"] += 1
                meta.update(dry_run=True, answered_by="rule")
            out = job.rule(inputs)
            errs = self._errors(job, out, inputs)
            if errs:                                     # a broken stand-in is a bug, not a model failure
                raise AssertionError(f"rule stand-in for {job.name} is invalid: {errs}")
            meta["latency_s"] = round(time.time() - t0, 3)
            if not meta.get("dry_run"):
                self._store(path, out, meta)
            return Result(out, meta)
        try:
            return self._complete_remote(job, system, user, inputs, images, meta, path, t0, effort, schema)
        except Exception as e:
            if not (self.backend == "spark" and self.fallback and shutil.which("claude")):
                raise
            # Safe fallback: the same request to Claude. Marked, and cached under Claude's own key.
            self.stats["fallbacks"] += 1
            reason = (f"{type(e).__name__}: {e}")[:300]
            claude = ModelClient("claude_cli", use_cache=self.use_cache, fallback=False)
            res = claude.complete(job.name, inputs, images=images)
            meta2 = {**res.meta, "requested_backend": "spark", "fallback": reason}
            self._log({**meta2, "note": "spark failed; answered by claude_cli"})
            return Result(res.output, meta2)

    def _complete_remote(self, job, system, user, inputs, images, meta, path, t0, effort, schema):
        out, raw = None, None
        messages_user = user
        for attempt in range(2):
            raw, usage = self._call(system, messages_user, images, effort=effort,
                                    schema=(job.name, job.schema) if schema else None)
            self.stats["calls"] += 1
            if usage:
                meta["usage"] = {k: meta.get("usage", {}).get(k, 0) + int(usage.get(k) or 0)
                                 for k in ("prompt_tokens", "completion_tokens")}
                for k in ("prompt_tokens", "completion_tokens"):
                    self.stats[k] = self.stats.get(k, 0) + int(usage.get(k) or 0)
            out, errs = _parse_json(raw), None
            errs = ["output is not a JSON object"] if out is None else self._errors(job, out, inputs)
            if not errs:
                break
            if attempt == 0:
                meta["repairs"] += 1
                meta["repair_errors"] = errs
                self.stats["repairs"] += 1
                messages_user = (f"{user}\n\nYour previous answer was:\n{raw}\n\nIt was rejected by the validator:\n- "
                                 + "\n- ".join(errs) + "\n\nReply again with only the corrected JSON object.")
        else:
            self.stats["rejected"] += 1
            self._log({**meta, "rejected": errs, "latency_s": round(time.time() - t0, 2)})
            raise ModelRejected(job.name, errs, raw)
        meta["latency_s"] = round(time.time() - t0, 2)
        self._store(path, out, meta)
        return Result(out, meta)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _errors(job, out, inputs):
        errs = [f"{'/'.join(map(str, e.absolute_path)) or 'root'}: {e.message}"
                for e in jsonschema.Draft202012Validator(job.schema).iter_errors(out)]
        return errs or job.validate(out, inputs)

    def _store(self, path, out, meta):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"output": out, "meta": meta}, indent=1), encoding="utf-8")
        self._log(meta)

    def _log(self, meta):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**meta, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")

    def _call(self, system: str, user: str, images=None, effort=None, schema=None) -> tuple[str, dict | None]:
        if self.backend == "claude_cli":
            return _call_claude_cli(system, user, self.model, images=images), None
        if self.backend == "spark":
            usage = {}
            return _call_openai_compatible(system, user, self.model, images=images, usage_out=usage, effort=effort,
                                           schema=schema), usage
        raise ValueError(self.backend)


def _job(name):
    if name not in JOBS:
        from . import jobs  # noqa: F401  (registers the jobs)
    return JOBS[name]


def _parse_json(text: str):
    if text is None:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    s = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def _call_claude_cli(system: str, user: str, model: str, timeout: float = 240, images=None) -> str:
    """Claude Code in print mode, no session saved, run outside the repo so no project context leaks in. No tools,
    except with images: then only Read, in a temporary folder that holds nothing but copies of those images."""
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found on PATH (needed for KINTSUGI_LLM=claude_cli)")
    # Drop variables inherited from a parent Claude Code session (desktop app, IDE): they point the CLI at that
    # session's proxy instead of its own login.
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC_BASE_URL"))}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tools = ""
        if images:
            names = []
            for k, src in enumerate(images):
                name = f"image_{k + 1}{pathlib.Path(src).suffix.lower() or '.jpg'}"
                shutil.copy(src, pathlib.Path(tmp) / name)
                names.append(name)
            tools = "Read"
            user = (f"{user}\n\nAttached images (read each with the Read tool before answering; they are files in the "
                    f"current folder): {', '.join(names)}")
        p = subprocess.run([exe, "-p", "--output-format", "json", "--tools", tools, "--no-session-persistence",
                            "--model", model, "--system-prompt", system], input=user, cwd=tmp, capture_output=True,
                           text=True, encoding="utf-8", timeout=timeout, env=env)
    try:
        res = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude CLI failed (exit {p.returncode}): {p.stderr.strip()[:500] or p.stdout[:500]}")
    if res.get("is_error"):
        msg = str(res.get("result"))
        if "login" in msg.lower() or "api key" in msg.lower():
            raise RuntimeError("the claude CLI is not logged in: open a terminal, run `claude`, type /login once, "
                               f"then retry ({msg})")
        raise RuntimeError(f"claude CLI error: {msg}")
    return res.get("result", "")


def _call_openai_compatible(system: str, user: str, model: str, timeout: float = 420, images=None,
                            usage_out=None, retries: int = 2, effort=None, schema=None) -> str:
    """Meta Model API (https://api.meta.ai/v1, OpenAI-compatible /chat/completions, verified 2026-11-02). Images go
    as base64 data URLs in the user message (at most 50 per request); token usage is copied into usage_out."""
    from .. import settings
    base, key = settings.get("meta_api_base"), settings.get("meta_api_key")
    if not (base and key):
        raise RuntimeError("Meta API not configured: add the API address and key in the app's Settings "
                           "(or set META_API_BASE and META_API_KEY)")
    content = user
    if images:
        import base64
        content = [{"type": "text", "text": user}]
        for src in images[:50]:
            mime = "image/png" if str(src).lower().endswith(".png") else "image/jpeg"
            data = base64.b64encode(pathlib.Path(src).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    req_body = {"model": model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
    if effort:
        req_body["reasoning_effort"] = effort           # minimal / low / medium / high / xhigh (docs/reasoning)
    if schema:
        name, sch = schema                              # answer constrained to the job's schema (docs/structured-output)
        req_body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": sch}}
    body = json.dumps(req_body)
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body.encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:          # the API's own message, without the request (no key)
            msg = f"Meta API HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(5 * 3 ** attempt)         # transient: back off and retry
                continue
            raise RuntimeError(msg) from None
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(5 * 3 ** attempt)
                continue
            raise RuntimeError(f"Meta API unreachable after {retries + 1} attempts: {type(e).__name__}: {e}") from None
    if usage_out is not None:
        usage_out.update(res.get("usage") or {})
    return res["choices"][0]["message"]["content"]
