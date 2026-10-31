import { useEffect, useState } from "react";

// Every statement in the app carries one of these labels.
export const STATUS = {
  measured: "measured",
  hypothesis: "hypothesis",
  confirmed: "confirmed",
  rejected: "rejected",
};

export function Chip({ status, children }) {
  return <span className={`chip chip-${status}`}>{children || status}</span>;
}

export function Card({ title, status, children, actions }) {
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
      {children}
    </section>
  );
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
  if (state.loading) return <p className="muted">Loading…</p>;
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
            {r.id}
            {r.outcome ? ` · ${r.outcome}` : ""}
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
