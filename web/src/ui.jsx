import { Component, useEffect, useRef, useState } from "react";

// Every statement in the app carries one of these labels.
export const STATUS = {
  measured: "measured",
  hypothesis: "hypothesis",
  confirmed: "confirmed",
  rejected: "rejected",
};

const STATUS_HELP = {
  measured: "measured: counted in simulation by deterministic code",
  hypothesis: "hypothesis: proposed by the model, not yet tested",
  confirmed: "confirmed: a test in simulation supported it",
  rejected: "rejected: a test in simulation contradicted it",
};

// Status tags. "measured" is the default for everything the app shows (counted by code in simulation), so it is
// not marked; hypothesis / confirmed / rejected are always shown in words.
export function Chip({ status, children }) {
  if (!children) {
    if (status === "measured") return null;
    return <span className={`chip chip-${status} status-tag`} title={STATUS_HELP[status] || ""}>{status}</span>;
  }
  return <span className={`chip chip-${status}`} title={STATUS_HELP[status] || ""}>{children}</span>;
}

// A part of the page that fails to render shows a notice (and its data, if given) instead of blanking the app.
export class Guard extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidUpdate(prev) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="guard">
        <b>This part could not be displayed.</b> <span className="sub">{String(this.state.error.message || this.state.error).slice(0, 200)}</span>
        {this.props.data !== undefined && (
          <details><summary className="sub">raw data</summary><pre className="code">{JSON.stringify(this.props.data, null, 1)?.slice(0, 20000)}</pre></details>
        )}
      </div>
    );
  }
}

export function Card({ title, status, children, actions, data }) {
  return (
    <section className="card">
      {(title || status) && (
        <header className="card-head">
          <h3>{title}</h3>
          <div className="card-actions">
            {actions}
            {status && <Chip status={status} />}
          </div>
        </header>
      )}
      <Guard data={data}>{children}</Guard>
    </section>
  );
}

export function Details({ label = "details", children }) {
  return <details className="more"><summary>{label}</summary>{children}</details>;
}

export function Code({ children }) {
  return <pre className="code">{children}</pre>;
}

export function Ok({ ok }) {
  return <span className={ok ? "ok" : "bad"}>{ok ? "✓" : "✗"}</span>;
}

export function useAsync(fn, deps) {
  const [state, set] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    set({ data: null, error: null, loading: true });
    fn()
      .then((data) => live && set({ data, error: null, loading: false }))
      .catch((error) => live && set({ data: null, error, loading: false }));
    return () => {
      live = false;
    };
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

export function Loading({ state, children }) {
  if (state.loading) return <div className="skeleton" aria-label="loading" />;
  if (state.error) return <p className="bad">Could not load: {String(state.error.message || state.error)}</p>;
  return children(state.data);
}

export function RunPicker({ runs, kinds, value, onChange, label = "Run" }) {
  const list = runs.filter((r) => kinds.includes(r.kind));
  return (
    <label className="picker">
      <span>{label}</span>
      <select value={value || ""} onChange={(e) => onChange(e.target.value)}>
        <option value="" disabled>
          choose ({list.length})
        </option>
        {list.map((r) => (
          <option key={r.id} value={r.id}>
            {r.title || r.id}
            {r.outcome ? ` · ${r.outcome.replace(/_/g, " ")}` : ""}
            {r.trials != null ? ` · ${r.success}/${r.trials}` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

export function KV({ rows }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

// mm:ss as digit tiles (job time, episode time)
export function Clock({ seconds, label }) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const d = `${String(Math.floor(s / 60)).padStart(2, "0")}${String(s % 60).padStart(2, "0")}`;
  return (
    <span className="clock" title={label} aria-label={`${label || "time"} ${d.slice(0, 2)}:${d.slice(2)}`}>
      <b>{d[0]}</b><b>{d[1]}</b><i>:</i><b>{d[2]}</b><b>{d[3]}</b>
    </span>
  );
}

// A horizontal row that scrolls with arrow buttons and edge fades instead of a visible scrollbar.
export function ScrollRow({ className = "", children }) {
  const ref = useRef(null);
  const [edge, setEdge] = useState({ left: false, right: false });
  const update = () => {
    const el = ref.current;
    if (!el) return;
    const left = el.scrollLeft > 4, right = el.scrollLeft + el.clientWidth < el.scrollWidth - 4;
    // same object back when nothing changed, so the effect after each render does not re-render forever
    setEdge(e => (e.left === left && e.right === right ? e : { left, right }));
  };
  useEffect(() => {
    update();
    const ro = new ResizeObserver(update);
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  useEffect(update);
  const scroll = (dir) => ref.current?.scrollBy({ left: dir * ref.current.clientWidth * 0.7, behavior: "smooth" });
  return (
    <div className={`tl-wrap ${edge.left ? "fade-l" : ""} ${edge.right ? "fade-r" : ""}`}>
      {edge.left && <button type="button" className="tl-arrow left" onClick={() => scroll(-1)} aria-label="scroll left">‹</button>}
      {edge.right && <button type="button" className="tl-arrow right" onClick={() => scroll(1)} aria-label="scroll right">›</button>}
      <div className={`scroll-row ${className}`} ref={ref} onScroll={update}>{children}</div>
    </div>
  );
}
