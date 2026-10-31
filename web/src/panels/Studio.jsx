import { useEffect, useRef, useState } from "react";
import { forget, getReplay, studio } from "../api.js";
import { Chip, Code, useAsync } from "../ui.jsx";
import { loadMeshGeometries, Viewer } from "./Replay.jsx";

// ---------------------------------------------------------------------------------------------- words
const nice = (id) => (id || "").replace(/_/g, " ");
const LABELS = {
  missed_grasp: "missed the grasp", weak_grasp: "grasp too weak", collision: "hit something", drop: "dropped it",
  wrong_place: "ended in the wrong place", unreachable: "out of reach", constraint_violation: "broke a constraint",
  not_attempted: "not attempted", episode_error: "scene error", precondition_failed: "no skill fits",
  object_not_perceived: "could not see the object", tipped_over: "tipped over",
};
const label = (l) => LABELS[l] || nice(l);
const OUTCOMES = {
  no_failures: "it did not fail again when measured on more scenes (too rare to fix)",
  rare_failure: "too rare to fix: it fails in fewer than 1 in 10 scenes (95% confidence), and the checks need about 10 failed cases to prove a fix",
  no_patch_passed: "the failure happens after the grasp and no parameter patch passed the gates",
  no_feasible_candidates: "no grasp candidate fits the gripper and is reachable",
  no_candidate_passed_gates: "candidates were found, but none passed every gate (held-out objects, no regressions)",
};
const COLORS = { red: "#d0453b", green: "#3f9d4f", blue: "#3b6fd0", yellow: "#e2b93b", orange: "#e5813a",
  purple: "#8a5cc7", white: "#eeeeee", black: "#333", grey: "#999", gray: "#999", brown: "#8a5a3c", pink: "#e58fb5" };
const BACKEND_NAMES = { rule: "rule stand-in (offline)", claude_cli: "Claude (development stand-in)", spark: "Muse Spark" };
const STAGES = [
  ["check", "Check the failures are real", "an expert with full knowledge replays each failed scene"],
  ["diagnose", "Find out why", "hypotheses from the evidence, each tested by a one-change replay"],
  ["measure", "Measure how often it fails", "same object type, more scenes"],
  ["search", "Search for a fix", "existing skills, parameter patches, new grasps in simulation"],
  ["verify", "Verify on held-out objects", "statistical gates: fixes the failures, no regressions"],
  ["learned", "Add it to the skill library", "only if every gate passed"],
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function sourceChip(src) {
  if (!src) return <Chip status="measured">fixed plan</Chip>;
  if (src.composite) return <Chip status="confirmed">verified program {src.composite} v{src.version}</Chip>;
  return <Chip status="hypothesis">written by {BACKEND_NAMES[src.backend] || src.backend} · not verified yet</Chip>;
}

// ---------------------------------------------------------------------------------------------- live player
// Workers simulate at full speed and write pose frames; this buffers one episode and plays it at real time,
// then jumps to the freshest episode it has not shown yet. Clicking a worker or an episode tile switches.
function useLivePlayer(jobId, running) {
  const [workers, setWorkers] = useState([]);
  const [, setTick] = useState(0);
  const buf = useRef(null);
  const watched = useRef(new Set());
  const workersRef = useRef([]);
  const hold = useRef(0);

  const watch = (episode, saved = null) => {
    watched.current.add(episode);
    hold.current = 0;
    buf.current = saved
      ? { episode, saved: true, meta: saved.meta, scene: saved.scene, frames: saved.frames, events: saved.events || [],
          done: true, result: saved.result, offset: 0, idx: 0 }
      : { episode, meta: null, scene: null, frames: [], events: [], done: false, result: null, offset: 0, idx: 0 };
    setTick((t) => t + 1);
  };
  const pickNext = () => {
    const fresh = workersRef.current.filter((w) => !watched.current.has(w.episode));
    const live = fresh.filter((w) => !w.done).sort((a, b) => b.started - a.started);
    const next = live[0] || fresh.sort((a, b) => b.started - a.started)[0];
    if (next) watch(next.episode);
  };

  useEffect(() => { buf.current = null; watched.current = new Set(); setWorkers([]); }, [jobId]);

  useEffect(() => {                                        // worker pointers
    if (!jobId) return;
    let stop = false;
    (async () => {
      do {
        try { const w = await studio.workers(jobId); workersRef.current = w; if (!stop) setWorkers(w); } catch {}
        await sleep(1000);
      } while (!stop && running);
    })();
    return () => { stop = true; };
  }, [jobId, running]);

  useEffect(() => {                                        // frames of the watched episode
    if (!jobId) return;
    let stop = false;
    (async () => {
      while (!stop) {
        const b = buf.current;
        if (b && !b.done && !b.saved) {
          try {
            const r = await studio.episode(jobId, b.episode, b.offset);
            if (buf.current === b) {
              for (const l of r.lines) {
                if (l.type === "f") b.frames.push(l);
                else if (l.type === "scene") b.scene = { bodies: l.bodies, geoms: l.geoms };
                else if (l.type === "meta") b.meta = l;
                else if (l.type === "e") b.events.push(l);
                else if (l.type === "done") { b.done = true; b.result = l; }
              }
              b.offset = r.offset;
            }
          } catch { b.done = true; }
        }
        await sleep(300);
      }
    })();
    return () => { stop = true; };
  }, [jobId]);

  useEffect(() => {                                        // real-time playback clock
    let raf, last = performance.now(), shown = -1;
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      const b = buf.current;
      if (!b) {
        if (running) pickNext();
      } else if (b.frames.length) {
        b.idx = Math.min(b.idx + dt * 30, b.frames.length - 1);
        const i = Math.floor(b.idx);
        if (i !== shown) { shown = i; setTick((t) => t + 1); }
        if (b.done && i >= b.frames.length - 1 && !b.saved && running) {
          if (!hold.current) hold.current = now + 1500;
          else if (now > hold.current) pickNext();
        }
      } else if (b.done && running) pickNext();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const b = buf.current;
  const pose = b && b.frames.length ? b.frames[Math.floor(b.idx)] : null;
  const t = pose ? pose.t : 0;
  const lastSkill = b ? [...b.events].reverse().find((e) => e.kind === "skill_end" && e.t <= t) : null;
  return { workers, watch, b, pose, t, lastSkill, replayFromStart: () => b && ((b.idx = 0), (hold.current = 0)) };
}

// ---------------------------------------------------------------------------------------------- pieces
function Composer({ info, prompt, setPrompt, backend, setBackend, onSubmit, busy, locked }) {
  return (
    <section className="card composer">
      <h2>What should the robot do?</h2>
      <form onSubmit={(e) => { e.preventDefault(); if (prompt.trim() && !busy && !locked) onSubmit(); }}>
        <textarea rows={2} value={prompt} placeholder="e.g. Put the bowl on the blue tray" disabled={locked}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); e.currentTarget.form.requestSubmit(); } }} />
        <div className="composer-row">
          <div className="examples">
            {(info?.examples || []).map((x) => (
              <button type="button" key={x} disabled={locked} onClick={() => setPrompt(x)}>{x}</button>
            ))}
          </div>
          <div className="composer-actions">
            {info && (
              <select value={backend || ""} onChange={(e) => setBackend(e.target.value)} title="Model that reads the request and writes the plan">
                {info.backends.map((b) => <option key={b} value={b}>{BACKEND_NAMES[b] || b}</option>)}
              </select>
            )}
            <button className="btn primary" disabled={!prompt.trim() || busy || locked}>{busy ? "Thinking…" : "Plan it"}</button>
          </div>
        </div>
        {locked && <p className="sub">A job is running. Stop it or wait for it to finish to plan a new one.</p>}
      </form>
    </section>
  );
}

function Understood({ spec, program, programLoading }) {
  const out = spec.output;
  const t = spec.task;
  return (
    <div className="understood">
      <div className="card-head"><h3>Here is what I understood</h3><Chip status="hypothesis">model output, validated</Chip></div>
      <div className="objs">
        {t.objects.map((o) => (
          <span key={o.id} className={`obj role-${o.role || "item"}`}>
            <i style={{ background: COLORS[o.params?.color] || "#bbb" }} />
            {nice(o.id)}{o.role && o.role !== "item" ? <em>{o.role}</em> : null}
          </span>
        ))}
      </div>
      <ol className="goals">
        {t.goal.map((g, i) => (
          <li key={i}>Put <b>{nice(g.obj)}</b> {g.pred === "in" ? "into" : "onto"} <b>{nice(g.target)}</b>{g.upright ? ", upright" : ""}</li>
        ))}
        {(t.constraints || []).map((c, i) => (
          <li key={`c${i}`} className="constraint">Keep <b>{nice(c.obj)}</b> {c.type === "stays_upright" ? "upright" : c.type === "not_disturbed" ? "undisturbed" : nice(c.type)}</li>
        ))}
      </ol>
      {out.explanation && <p className="sub">{out.explanation}</p>}
      <div className="plan-line">
        <span>Plan:</span>
        {programLoading ? <span className="muted">writing the plan…</span> : program && sourceChip(program.source)}
      </div>
      {program?.program && (
        <details><summary className="sub">show the plan program</summary><Code>{program.program}</Code>
          {program.explanation && <p className="sub">{program.explanation}</p>}</details>
      )}
      {program && !program.program && <p className="sub">{program.explanation}</p>}
      <p className="sub">Model: {BACKEND_NAMES[spec.meta?.backend] || spec.meta?.backend}{spec.meta?.cached ? " · cached" : ""}
        {spec.meta?.latency_s != null ? ` · ${spec.meta.latency_s.toFixed(1)} s` : ""}</p>
    </div>
  );
}

function Settings({ info, s, set }) {
  const maxW = Math.max(1, Math.min((info?.max_workers || 2) - 1, 16));
  return (
    <div className="settings">
      <label>Scenes
        <select value={s.n} onChange={(e) => set({ ...s, n: +e.target.value })}>
          {[1, 4, 8, 16, 32].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
      <label title="Simulations run in parallel processes. Above ~8 a laptop gets slower, not faster.">In parallel
        <input type="range" min={1} max={maxW} value={s.workers} onChange={(e) => set({ ...s, workers: +e.target.value })} />
        <span className="pill">{s.workers}{s.workers === info?.default_workers ? " (recommended)" : ""}</span>
      </label>
      <label className="toggle"><input type="checkbox" checked={s.autoFix} onChange={(e) => set({ ...s, autoFix: e.target.checked })} />
        Fix failures automatically</label>
      <label className="toggle"><input type="checkbox" checked={s.retries} onChange={(e) => set({ ...s, retries: e.target.checked })} />
        Retry a failed step</label>
      <label className="toggle" title="Friction x0.35-0.7 and mass x1.3-2 per scene: a stress test that makes the current skills fail sometimes, so there is something to fix">
        <input type="checkbox" checked={s.hard} onChange={(e) => set({ ...s, hard: e.target.checked })} />
        Harder physics (slippery, heavy objects)</label>
    </div>
  );
}

function Stepper({ steps }) {
  return (
    <ol className="stepper">
      {steps.map(([key, text, state]) => <li key={key} className={state}>{text}</li>)}
    </ol>
  );
}

function Tiles({ trials, seeds, runningSeeds, onPick, title }) {
  const bySeed = Object.fromEntries(trials.map((t) => [t.seed, t]));
  const k = trials.filter((t) => t.success).length;
  const n = seeds.length;
  return (
    <div className="tiles-block">
      <div className="card-head"><h3>{title}</h3><span className="sub">{trials.length}/{n} done · {k} succeeded</span></div>
      <div className="tiles">
        {seeds.map((s, i) => {
          const t = bySeed[s];
          const cls = t ? (t.success ? "ok" : "fail") : runningSeeds.has(s) ? "running" : "";
          return (
            <button key={s} className={`tile ${cls}`} disabled={!t} onClick={() => onPick(s)}
                    title={t ? (t.success ? "succeeded · click to replay" : `${t.labels.map(label).join(", ")} · click to replay`) : runningSeeds.has(s) ? "running" : "waiting"}>
              {i + 1}
            </button>
          );
        })}
      </div>
      {trials.some((t) => !t.success) && (
        <div className="sub">Failures: {Object.entries(trials.flatMap((t) => t.labels).reduce((a, l) => ({ ...a, [l]: (a[l] || 0) + 1 }), {}))
          .map(([l, c]) => `${label(l)} ×${c}`).join(", ")}</div>
      )}
    </div>
  );
}

function RepairProgress({ logs, stageDone, current }) {
  const order = STAGES.map(([k]) => k);
  const reached = order.indexOf(current);
  const seen = new Set(order.filter((k, i) => i < reached || logs.some((l) => l.stage === k)));
  const last = logs.length ? logs[logs.length - 1].line.replace(/^\[\w+\]\s*/, "") : "starting…";
  return (
    <ol className="repair">
      {STAGES.filter(([k]) => k !== "learned" || seen.has("learned")).map(([k, title, how]) => {
        const state = current === k && !stageDone ? "active" : seen.has(k) ? "done" : "";
        return (
          <li key={k} className={state}>
            <b>{title}</b> <span className="sub">{how}</span>
            {state === "active" && <div className="sub mono">{last}</div>}
          </li>
        );
      })}
    </ol>
  );
}

function Outcome({ result, open }) {
  const { before, repair, after } = result;
  return (
    <div className="outcome">
      <div className="nums">
        <div><span className="sub">first run</span><b className={before.success === before.trials ? "ok" : "bad"}>{before.success}/{before.trials}</b></div>
        {after && <><span className="arrow">→</span>
          <div><span className="sub">after the fix</span><b className={after.success === after.trials ? "ok" : ""}>{after.success}/{after.trials}</b></div></>}
        <Chip status="measured" />
      </div>
      {repair?.missing_capability && <p>{repair.message}</p>}
      {repair?.clusters && (
        <ul className="clusters">
          {repair.clusters.map((c, i) => {
            const r = c.repair || {};
            const what = r.promoted ? `new skill ${r.promoted}` : r.patched ? `parameter patch ${r.patched.name || JSON.stringify(r.patched.values || "")}`
              : r.routed ? `use ${r.routed.card} instead of ${r.routed.instead_of || c.skill}` : null;
            return (
              <li key={i}>
                <b>{nice(c.kind)}</b> with {c.skill} ({c.valid_failures} real failure{c.valid_failures === 1 ? "" : "s"}): {" "}
                {what ? <><Chip status="confirmed">learned</Chip> {what}</>
                  : <><Chip status={r.outcome === "rare_failure" ? "measured" : "rejected"}>{r.outcome === "rare_failure" ? "rare" : "no fix"}</Chip>{" "}
                    {c.action || OUTCOMES[r.outcome] || nice(r.outcome) || "nothing passed the gates"}</>}
                {r.rare && <div className="sub">measured: {r.rare.valid_failures} real failures in {r.rare.trials} scenes</div>}
                {c.investigation && <div className="sub">diagnosis: {c.investigation.confirmed} cause(s) confirmed, {c.investigation.rejected} rejected</div>}
                {r.run_id && <button className="link" onClick={() => open({ tab: "experiments", run: r.run_id })}>see the search and gates →</button>}
              </li>
            );
          })}
          {repair.clusters.length === 0 && <li>None of the failures could be solved by the expert either (unsolvable scenes): nothing to learn.</li>}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- page
export default function Studio({ open }) {
  const [info, setInfo] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [backend, setBackend] = useState(null);
  const [settings, setSettings] = useState({ n: 8, workers: 4, autoFix: true, retries: true, hard: false });
  const [plan, setPlan] = useState(null);
  const [job, setJob] = useState(null);
  const [history, setHistory] = useState([]);
  const meshes = useAsync(loadMeshGeometries, []);

  const refreshHistory = () => studio.jobs().then(setHistory).catch(() => {});
  useEffect(() => {
    studio.info().then((i) => {
      setInfo(i);
      setBackend(i.default_backend);
      setSettings((s) => ({ ...s, workers: i.default_workers }));
      if (i.running) attach(i.running);
    }).catch(() => {});
    refreshHistory();
  }, []);

  const attach = (id) => setJob({ id, events: [], offset: 0, running: true, meta: null });

  useEffect(() => {                                        // follow the job's events
    if (!job?.id) return;
    let stop = false;
    (async () => {
      let offset = 0, running = true;
      while (!stop) {
        try {
          const r = await studio.job(job.id, offset);
          offset = r.offset;
          running = r.running;
          if (!stop) setJob((j) => j && j.id === job.id ? { ...j, events: [...j.events, ...r.events], running, meta: r.job || j.meta, logTail: r.log_tail } : j);
        } catch { running = false; }
        if (!running) break;
        await sleep(700);
      }
      if (!stop) { forget("runs"); forget("skills"); forget("headlines"); refreshHistory(); }
    })();
    return () => { stop = true; };
  }, [job?.id]);

  const player = useLivePlayer(job?.id, !!job?.running);

  const doPlan = async () => {
    setJob((j) => (j?.running ? j : null));
    setPlan({ phase: "spec" });
    try {
      const spec = await studio.spec(prompt, backend);
      if (spec.status !== "ok") return setPlan({ phase: "answer", spec });
      setPlan({ phase: "program", spec });
      const program = await studio.program(prompt, spec.task, backend);
      setPlan({ phase: "ready", spec, program });
    } catch (e) {
      setPlan({ phase: "error", error: String(e.message || e) });
    }
  };
  const run = async () => {
    try {
      const j = await studio.start({ prompt, task: plan.spec.task, program: plan.program?.program,
        program_source: plan.program?.source, n: settings.n, workers: settings.workers, retries: settings.retries,
        auto_fix: settings.autoFix, hard: settings.hard, backend });
      attach(j.id);
    } catch (e) { setPlan({ ...plan, runError: String(e.message || e) }); }
  };

  // derived job state
  const ev = job?.events || [];
  const stageEv = (st, state) => ev.find((e) => e.type === "stage" && e.stage === st && e.state === state);
  const trials = (phase) => ev.filter((e) => e.type === "trial" && e.phase === phase);
  const logs = ev.filter((e) => e.type === "log");
  const result = ev.find((e) => e.type === "result");
  const error = ev.find((e) => e.type === "error");
  const stopped = ev.find((e) => e.type === "stopped");
  const seeds = stageEv("run", "start")?.seeds || Array.from({ length: job?.meta?.n || settings.n }, (_, i) => i);
  const n = seeds.length;
  const sceneNo = (meta) => (meta.tag ? meta.seed : seeds.indexOf(meta.seed) + 1);
  const order = STAGES.map(([k]) => k);                    // the furthest stage reached (log lines can repeat earlier ones)
  const repairStage = logs.reduce((best, l) => (l.stage && order.indexOf(l.stage) > order.indexOf(best) ? l.stage : best), null);
  const phase = !job ? null : result || error || stopped || !job.running ? "done"
    : stageEv("rerun", "start") ? "rerun" : stageEv("repair", "start") ? "repair" : "run";
  const runningSeeds = (ph) => new Set(player.workers.filter((w) => !w.done && w.run === ph && !w.tag).map((w) => w.seed));
  const runDir = (ph) => `${job.id}/${ph}`;
  const pickTile = async (ph, seed) => {
    try {
      const r = await getReplay(runDir(ph), String(seed).padStart(5, "0"));
      const t = trials(ph).find((x) => x.seed === seed);
      player.watch(`saved:${ph}:${seed}`, { ...r, meta: { seed, run: ph, saved: true }, result: t });
    } catch {}
  };

  const runTrials = trials("run");
  useEffect(() => {
    if (job && !job.running && !player.b && runTrials.length) {
      const t = runTrials.find((x) => !x.success && !x.labels.includes("episode_error")) || runTrials[0];
      pickTile("run", t.seed);
    }
  }, [job?.id, job?.running, runTrials.length]);

  const steps = [
    ["understand", plan?.phase === "answer" && !job ? "Understand · needs your input" : "Understand",
      plan?.phase === "spec" || (plan?.phase === "answer" && !job) ? "active" : plan?.spec || job ? "done" : ""],
    ["plan", "Plan", plan?.phase === "program" ? "active" : plan?.phase === "ready" || job ? "done" : ""],
    ["run", "Run", phase === "run" ? "active" : phase ? "done" : ""],
    ["fix", "Fix", phase === "repair" ? "active" : stageEv("repair", "end") ? "done" : phase === "done" ? "skipped" : ""],
    ["rerun", "Run again", phase === "rerun" ? "active" : stageEv("rerun", "end") ? "done" : phase === "done" ? "skipped" : ""],
  ];

  let headline = null;
  if (job) {
    const t1 = trials("run"), ok1 = t1.filter((t) => t.success).length;
    if (phase === "run") headline = `Running ${n} scene${n > 1 ? "s" : ""} on ${job.meta?.workers || settings.workers} parallel workers · ${t1.length}/${n} done, ${ok1} succeeded`;
    else if (phase === "repair") headline = `${t1.length - ok1} of ${t1.length} failed. Kintsugi is finding out why and searching for a fix in simulation…`;
    else if (phase === "rerun") headline = `A fix passed every check and was added to the library. Running the same ${n} scenes again…`;
    else if (error) headline = `The job stopped with an error: ${error.message}`;
    else if (stopped) headline = "Stopped. Nothing after this point was kept.";
    else if (result) {
      const { before, after, repair } = result;
      headline = after ? `Fixed: ${before.success}/${before.trials} → ${after.success}/${after.trials} on the same scenes.`
        : before.success === before.trials ? `All ${before.trials} scenes succeeded.`
        : repair?.missing_capability ? `${before.success}/${before.trials} succeeded: the plan needs a skill the library does not have.`
        : repair && !repair.clusters?.length ? `${before.success}/${before.trials} succeeded. The expert could not solve the failed scenes either, so there is nothing to learn from them.`
        : repair?.clusters?.every((c) => c.repair?.outcome === "rare_failure") ? `${before.success}/${before.trials} succeeded. The failures are too rare to fix reliably, so the library is unchanged.`
        : repair ? `${before.success}/${before.trials} succeeded. No fix passed the checks, so the library is unchanged.`
        : `${before.success}/${before.trials} succeeded.`;
    } else if (job.logTail) headline = "The job ended without a result.";
    else headline = "Starting…";
  }

  const preview = plan?.spec?.preview;
  const b = player.b;
  const showLive = job && (b || job.running);
  return (
    <div className="studio grid">
      <Composer info={info} prompt={prompt} setPrompt={setPrompt} backend={backend} setBackend={setBackend}
                onSubmit={doPlan} busy={plan?.phase === "spec" || plan?.phase === "program"} locked={!!job?.running} />

      {(plan || job) && <Stepper steps={steps} />}

      {!job && plan?.phase === "error" && <p className="bad">{plan.error}</p>}
      {!job && plan?.phase === "answer" && (
        <section className="card">
          {plan.spec.status === "clarify" && <><h3>I need one detail</h3><p>{plan.spec.output.question}</p>
            <p className="sub">Add the answer to your request and plan again.</p></>}
          {plan.spec.status === "refuse" && <><h3>I can't do this one</h3><p>{plan.spec.output.reason || plan.spec.output.explanation}</p></>}
          {plan.spec.status === "rejected" && <><h3>I could not turn this into a valid task</h3><Code>{(plan.spec.reasons || []).join("\n")}</Code></>}
        </section>
      )}

      {!job && plan?.spec?.task && !preview && <p className="bad">No layout of this task fits on the table: try fewer or smaller objects.</p>}
      {!job && plan?.spec?.task && preview && (
        <div className="split studio-split">
          <Viewer scene={preview.scene} meshes={meshes.data} pose={preview.frames[0]} height={440} zoom={false}>
            <span className="badge">scene preview · first of {settings.n}</span>
          </Viewer>
          <section className="card">
            <Understood spec={plan.spec} program={plan.program} programLoading={plan.phase === "program"} />
            <Settings info={info} s={settings} set={setSettings} />
            <button className="btn primary big-btn" disabled={plan.phase !== "ready"} onClick={run}>
              Run {settings.n} scene{settings.n > 1 ? "s" : ""}</button>
            <p className="sub">Each scene places the same objects differently. {settings.autoFix ? "If steps fail, Kintsugi diagnoses them, searches for a fix in simulation, and keeps it only if it passes held-out checks." : ""}</p>
            {plan.runError && <p className="bad">{plan.runError}</p>}
          </section>
        </div>
      )}

      {job && (
        <>
          <section className={`banner ${phase}`}>
            <span>{headline}</span>
            {job.running ? <button className="btn" onClick={() => studio.stop(job.id)}>Stop</button>
              : <button className="btn" onClick={() => setJob(null)}>Close</button>}
          </section>
          <div className="split studio-split">
            <div>
              <Viewer scene={b?.scene} meshes={meshes.data} pose={player.pose} height={460} zoom={false}>
                {showLive && b?.meta && (
                  <span className={`badge ${b.saved ? "" : "live"}`}>
                    {b.saved ? "replay" : "live"} · {b.meta.run === "rerun" ? "after the fix" : b.meta.run === "run" ? "first run" : "repair trial"}
                    {b.meta.tag ? ` · ${nice(b.meta.tag)}` : ""} · scene {sceneNo(b.meta)} · {player.t.toFixed(1)} s
                    {player.lastSkill ? ` · ${player.lastSkill.skill} ${player.lastSkill.ok ? "✓" : "✗"}` : ""}
                    {b.done && b.result && Math.floor(b.idx) >= b.frames.length - 1 ? (b.result.success ? " · succeeded" : ` · failed: ${(b.result.labels || []).map(label).join(", ")}`) : ""}
                  </span>
                )}
                {!b && job.running && <span className="badge">waiting for the first frames…</span>}
                {b?.saved && <button className="btn overlay-btn" onClick={player.replayFromStart}>replay</button>}
              </Viewer>
              {job.running && player.workers.length > 0 && (
                <div className="workers">
                  <span className="sub">workers</span>
                  {player.workers.map((w) => (
                    <button key={w.worker} className={`wdot ${w.done ? (w.success ? "ok" : "fail") : "running"} ${b?.episode === w.episode ? "on" : ""}`}
                            title={`${w.run}${w.tag ? " · " + w.tag : ""} · scene ${w.seed}`} onClick={() => player.watch(w.episode)} />
                  ))}
                  <span className="sub">simulation runs at full speed; the view plays one scene at real time</span>
                </div>
              )}
            </div>
            <section className="card">
              <Tiles title="First run" trials={trials("run")} seeds={seeds} runningSeeds={runningSeeds("run")} onPick={(s) => pickTile("run", s)} />
              {stageEv("repair", "start") && (
                <div className="tiles-block">
                  <div className="card-head"><h3>Fixing the failures</h3>{phase === "repair" && <span className="pulse">working</span>}</div>
                  {result?.repair?.missing_capability ? <p className="sub">{result.repair.message}</p>
                    : <RepairProgress logs={logs} stageDone={!!stageEv("repair", "end")} current={repairStage} />}
                </div>
              )}
              {stageEv("rerun", "start") && (
                <Tiles title="Same scenes after the fix" trials={trials("rerun")} seeds={seeds} runningSeeds={runningSeeds("rerun")} onPick={(s) => pickTile("rerun", s)} />
              )}
              {result && <Outcome result={result} open={open} />}
              {job.logTail && <details><summary className="bad">job log</summary><Code>{job.logTail}</Code></details>}
              {logs.length > 0 && (
                <details><summary className="sub">activity ({logs.length} lines)</summary>
                  <Code>{logs.map((l) => l.line).join("\n")}</Code></details>
              )}
            </section>
          </div>
        </>
      )}

      {!plan && !job && history.length > 0 && (
        <section className="card">
          <h3>Recent studio runs</h3>
          <table>
            <tbody>
              {history.map((h) => (
                <tr key={h.id} className="click" onClick={() => { setPrompt(h.prompt); attach(h.id); }}>
                  <td>{h.prompt}</td>
                  <td className="sub">{h.started}</td>
                  <td className="num">{h.running ? "running" : h.error ? "error" : h.stopped ? "stopped" : h.result ?
                    `${h.result.before.success}/${h.result.before.trials}${h.result.after ? ` → ${h.result.after.success}/${h.result.after.trials}` : ""}` : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
