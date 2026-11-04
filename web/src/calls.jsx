// Live Meta model calls (Muse Spark, SAM 3.1, Muse Image) made since this page opened: each call appears when it
// starts and shows its time and tokens when it answers. Data: /api/model_calls (cache/llm/live_calls.jsonl, written
// by the server and by Studio jobs). Cached answers make no call, so they never appear here.
import { useEffect, useRef, useState } from "react";

const JOB = {
  task_spec: "turns the request into a task",
  plan_program: "writes the plan program",
  scene_photo: "reads the photo",
  diagnose: "diagnoses a failure",
  propose_primitive: "writes new skill code",
  search_proposal: "proposes where to look for failures",
  retrieve: "picks a skill for a new object",
  segment: "finds",
  empty_table: "recreates the empty table",
};
const MAX_ROWS = 5;

function label(c) {
  const j = JOB[c.job] || c.job.replace(/_/g, " ");
  if (c.job === "segment") return `${j} “${c.detail}”`;
  return c.detail === "repair" ? `${j} (corrected answer)` : j;
}

function tokens(n) {
  if (!n) return null;
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k tokens` : `${n} tokens`;
}

export default function ModelCalls() {
  const [calls, setCalls] = useState([]);
  const [hidden, setHidden] = useState(false);
  const [open, setOpen] = useState(null);       // the call whose details are shown
  const since = useRef(null);

  useEffect(() => {
    let alive = true, timer = null;
    async function poll() {
      try {
        const r = await fetch(`/api/model_calls?since=${since.current ?? 0}`);
        if (r.ok) {
          const d = await r.json();
          if (since.current == null) since.current = d.now;        // first answer: only calls from now on
          else if (alive) setCalls(d.calls);
        }
      } catch { /* the server may be restarting */ }
      if (alive) timer = setTimeout(poll, 1000);
    }
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const last = calls[calls.length - 1];
  useEffect(() => { if (last && !last.end) setHidden(false); }, [last?.id, last?.end]);
  if (!calls.length || hidden) return null;
  const busy = calls.some((c) => !("end" in c));
  const total = calls.length;
  return (
    <aside className="mc" aria-live="polite" aria-label="Meta model calls">
      <header className="mc-head">
        <span className={`mc-dot ${busy ? "busy" : ""}`} />
        <span className="mc-title">Meta model calls</span>
        <span className="mc-count">{busy ? "live" : `${total} this session`}</span>
        <button className="mc-x" onClick={() => setHidden(true)} aria-label="Hide">×</button>
      </header>
      <ol className="mc-list">
        {calls.slice(-MAX_ROWS).map((c) => {
          const state = !("end" in c) ? "run" : c.ok ? "ok" : "bad";
          return (
            <li key={c.id} className={`mc-item ${open === c.id ? "open" : ""}`}>
              <button type="button" className={`mc-row ${state}`} onClick={() => setOpen(open === c.id ? null : c.id)}
                      title="Show the endpoint and request ID">
              <span className="mc-icon">{state === "run" ? <span className="mc-spin" /> : state === "ok" ? "✓" : "✗"}</span>
              <span className="mc-main">
                <span className="mc-model">{c.model}</span>
                <span className="mc-job">{label(c)}</span>
              </span>
              <span className="mc-meta">
                {state === "run" ? "calling…" : state === "bad" ? <span title={c.error}>failed</span> :
                  <>{c.latency_s != null && <b>{c.latency_s.toFixed(1)} s</b>}{tokens(c.tokens) && <em>{tokens(c.tokens)}</em>}</>}
              </span>
              </button>
              {open === c.id && (
                <dl className="mc-detail">
                  <dt>endpoint</dt><dd>{c.endpoint || "–"}</dd>
                  <dt>request ID</dt><dd>{c.request_id || (state === "run" ? "waiting for the answer" : "–")}</dd>
                  <dt>started</dt><dd>{new Date(c.start * 1000).toLocaleTimeString()}</dd>
                  {c.error && <><dt>error</dt><dd>{c.error}</dd></>}
                </dl>
              )}
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
