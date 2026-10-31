import { useEffect, useRef, useState } from "react";
import { getReplay, getRun, getTrial } from "../api.js";
import { Card, Chip, Clock, Details, Guard, Loading, useAsync } from "../ui.jsx";
import { loadMeshGeometries, Viewer } from "../viewer.jsx";

const nice = (s) => (s || "").replace(/_/g, " ");

// One episode: the 3D replay, a skill timeline under it, the plan with the line that is running now, the steps.
export default function Episodes({ runs, sel, open }) {
  const withReplays = runs.filter((r) => r.n_replays > 0);
  const run = useAsync(() => (sel.run ? getRun(sel.run) : Promise.resolve(null)), [sel.run]);
  const ep = useAsync(async () => {
    if (!sel.run || !sel.trial) return null;
    const [replay, meshes, trial] = await Promise.all([getReplay(sel.run, sel.trial), loadMeshGeometries(),
                                                       getTrial(sel.run, sel.trial).catch(() => null)]);
    return { replay, meshes, trial };
  }, [sel.run, sel.trial]);

  // default episode: the first failure, else the first one
  useEffect(() => {
    const r = run.data;
    if (r && r.episodes?.length && !r.episodes.some((e) => e.name === sel.trial)) {
      const f = r.episodes.find((e) => e.success === false) || r.episodes[0];
      open({ trial: f.name });
    }
  }, [run.data]);

  return (
    <div className="grid">
      <div className="toolbar ep-toolbar">
        <div className="ep-row">
          <span className="sub">Run</span>
          <select value={sel.run || ""} onChange={(e) => open({ run: e.target.value, trial: null })}>
            <option value="" disabled>choose a run ({withReplays.length})</option>
            {withReplays.map((r) => <option key={r.id} value={r.id}>{r.title || r.id}{r.trials != null ? ` · ${r.success}/${r.trials}` : ""}</option>)}
          </select>
        </div>
        <Loading state={run}>
          {(r) => r && (
            <div className="ep-row">
              <span className="sub">Episodes</span>
              <div className="ep-tiles" title="Red = failed">
                {r.episodes.map((e, i) => (
                  <button key={e.name} className={`tile ${e.success ? "ok" : e.success === false ? "fail" : ""} ${e.name === sel.trial ? "on" : ""}`}
                          title={e.success ? "succeeded" : (e.labels || []).map(nice).join(", ")} onClick={() => open({ trial: e.name })}>{i + 1}</button>
                ))}
              </div>
            </div>
          )}
        </Loading>
      </div>
      <Loading state={ep}>
        {(d) => d ? <Guard resetKey={`${sel.run}/${sel.trial}`} data={d.trial}><Episode d={d} /></Guard>
          : <p className="muted">Choose a run and an episode. Saved replays: demos, Studio runs, and failures of task runs.</p>}
      </Loading>
    </div>
  );
}

function Episode({ d }) {
  const frames = d.replay.frames;
  const T = frames[frames.length - 1]?.t || 1;
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const track = useRef(null);

  useEffect(() => { setFrame(0); setPlaying(true); }, [d]);
  useEffect(() => {
    if (!playing) return;
    const fps = d.replay.fps || 30;
    let last = performance.now(), acc = 0, raf;
    const tick = (now) => {
      acc += ((now - last) / 1000) * fps * speed;
      last = now;
      if (acc >= 1) {
        const step = Math.floor(acc);
        acc -= step;
        setFrame((f) => (f + step >= frames.length ? (setPlaying(false), frames.length - 1) : f + step));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, d]);

  const t = frames[frame]?.t || 0;
  const seek = (time) => {
    setPlaying(false);
    let i = frames.findIndex((f) => f.t >= time);
    setFrame(i < 0 ? frames.length - 1 : i);
  };
  const trial = d.trial || {};
  const subtasks = trial.subtasks || [];
  const segs = subtasks.flatMap((st, si) => (st.trace || []).map((s) => ({ ...s, obj: st.obj, si })));
  const active = segs.find((s) => t >= s.t0 && t <= s.t1);
  const activeStep = active ? active.si : -1;
  const lines = trial.program_lines || [];
  const activeLine = lines.find((l) => t >= l.t0 && t <= l.t1)?.line;
  const failedSeg = segs.find((s) => !s.ok);

  return (
    <div className="split episode-split">
      <div>
        {trial.prompt && <div className="ep-request">"{trial.prompt}"</div>}
        <Viewer scene={d.replay.scene} meshes={d.meshes} pose={frames[frame]} height={470}>
          <span className="badge">{active ? `${active.skill} · ${nice(active.obj)}` : t < 0.5 ? "start" : "between skills"} · {t.toFixed(1)} s</span>
          <span className={`badge right ${trial.success ? "ok" : "fail"}`}>{trial.success ? "task succeeded" : "task failed"}</span>
        </Viewer>
        <div className="timeline2">
          <button className="btn icon" onClick={() => { if (frame >= frames.length - 1) setFrame(0); setPlaying((p) => !p); }}
                  title={playing ? "pause" : "play"}>{playing ? "❚❚" : "▶"}</button>
          <Clock seconds={t} label="episode time" />
          <div className="seg-track" ref={track}
               onClick={(e) => { const r = track.current.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * T); }}>
            {segs.map((s, i) => (
              <span key={i} className={`seg ${s.ok ? "" : "fail"} ${active === s ? "on" : ""}`}
                    style={{ left: `${(100 * s.t0) / T}%`, width: `${Math.max(0.4, (100 * (s.t1 - s.t0)) / T)}%` }}
                    title={`${s.skill} · ${nice(s.obj)} · ${s.ok ? "ok" : "failed: " + s.check}`} />
            ))}
            <span className="playhead" style={{ left: `${(100 * t) / T}%` }} />
          </div>
          <select value={speed} onChange={(e) => setSpeed(+e.target.value)} title="speed">
            {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
        </div>
        <div className="sub">Skill timeline: each block is one skill call; red = its check failed. Click to jump.{failedSeg ? " " : ""}
          {failedSeg && <button className="link" onClick={() => seek(Math.max(0, failedSeg.t0))}>jump to the failure</button>}</div>
      </div>
      <div className="grid">
        {trial.program && (
          <Card title="Plan program" actions={<Chip status={trial.program_error ? "rejected" : "measured"}>{trial.program_error ? "stopped" : "ran"}</Chip>}>
            <pre className="code program">
              {trial.program.split("\n").map((ln, i) => (
                <div key={i} className={`pline ${activeLine === i + 1 ? "on" : ""} ${lines.some((l) => l.line === i + 1) ? "ran" : ""}`}>
                  <span className="lno">{i + 1}</span>{ln || " "}
                </div>
              ))}
            </pre>
            {!lines.length && <div className="sub">This episode was recorded before line tracking; the steps below show progress.</div>}
            {trial.program_error && <p className="bad">{trial.program_error}</p>}
          </Card>
        )}
        <Card title="Steps">
          <ol className="steps">
            {subtasks.map((st, i) => (
              <li key={i} className={`${st.ok ? "ok" : "fail"} ${i === activeStep ? "on" : ""}`}>
                <span className={`step-icon ${st.ok ? "ok" : "fail"}`} aria-label={st.ok ? "succeeded" : "failed"}>
                  <svg viewBox="0 0 16 16">{st.ok ? <path d="M3.5 8.5l3 3 6-7" /> : <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />}</svg>
                </span>
                <div className="step-body">
                  <div className="step-head">{nice(st.obj)} <span className="step-arrow">→</span> {nice(st.target) || "–"}</div>
                  <div className="step-skills">
                    {(st.trace || []).map((s, j) => (
                      <button key={j} className={`skill-pill ${s.ok ? "" : "fail"}`} onClick={() => seek(s.t0)} title={s.check}>{s.skill}</button>
                    ))}
                    {st.label && <span className="step-label">{nice(st.label)}</span>}
                  </div>
                </div>
              </li>
            ))}
          </ol>
          <Details label="Task details">
            <ul className="sub">
              {(trial.task_spec?.objects || []).map((o) => <li key={o.id}>{o.id}: {o.template} ({o.role})</li>)}
              {(trial.task_spec?.goal || []).map((g, i) => <li key={`g${i}`}>goal: {g.obj} {g.pred} {g.target}{g.upright ? " (upright)" : ""}</li>)}
              <li>seed {trial.seed}{trial.recovery ? ` · retries on (${trial.recovery_attempts || 0} used)` : ""}</li>
            </ul>
          </Details>
        </Card>
      </div>
    </div>
  );
}
