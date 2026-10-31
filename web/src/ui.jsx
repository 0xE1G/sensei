import { Component, useEffect, useState } from "react";

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

// With text: a labelled pill. Without: a small coloured dot whose hover says what the label means.
export function Chip({ status, children }) {
  if (!children) return <span className={`dot-label dot-${status}`} title={STATUS_HELP[status] || status} aria-label={status} />;
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
