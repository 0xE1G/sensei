// Small SVG charts in one visual system (see styles.css "charts"): thin marks, recessive grid, text in ink tokens,
// a hover tooltip on every mark. Data always comes from saved reports; charts never compute or guess values.
import { useRef, useState } from "react";

function useTip() {
  const box = useRef(null);
  const [tip, setTip] = useState(null);
  const show = (e, content) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, content });
  };
  const el = tip && (
    <div className="tip" style={{ left: Math.min(tip.x + 12, tip.w - 230), top: tip.y + 12 }}>{tip.content}</div>
  );
  return { box, show, hide: () => setTip(null), el };
}

// ------------------------------------------------------------------------------------------- line chart
// series: [{name, cls ("s1" | "s2" | "s3"), values: [number|null per x]}]; xs: [{label, note, tip}]
export function LineChart({ xs, series, height = 220, yMax, yLabel, annotate }) {
  const t = useTip();
  const W = 640, H = height, L = 36, R = 92, T = 14, B = 34;
  const max = yMax ?? Math.max(1, ...series.flatMap((s) => s.values.filter((v) => v != null))) * 1.1;
  const x = (i) => L + (xs.length < 2 ? 0 : (i * (W - L - R)) / (xs.length - 1));
  const y = (v) => T + (H - T - B) * (1 - v / max);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(max * f));
  // end-of-line labels: keep them >= 13px apart so they never overlap
  const ends = series.map((s) => {
    const i = s.values.map((v, j) => (v == null ? -1 : j)).filter((j) => j >= 0).pop();
    return { name: s.name, y: i == null ? null : y(s.values[i]) };
  }).filter((e) => e.y != null).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
  const labelY = Object.fromEntries(ends.map((e) => [e.name, e.y]));
  return (
    <div className="chart" ref={t.box} onMouseLeave={t.hide}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel}>
        {ticks.map((v) => (
          <g key={v}>
            <line className="grid-line" x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
            <text className="axis-text" x={L - 6} y={y(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        {xs.map((p, i) => (
          <text key={i} className="axis-text" x={x(i)} y={H - B + 16} textAnchor="middle">{p.label}</text>
        ))}
        {annotate && xs.map((p, i) => p.note && (
          <g key={`n${i}`}>
            <line className="note-line" x1={x(i)} x2={x(i)} y1={T} y2={H - B} />
          </g>
        ))}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v)])).filter(Boolean);
          const last = pts[pts.length - 1];
          return (
            <g key={s.name} className={s.cls}>
              <polyline className="line" points={pts.map((p) => p.join(",")).join(" ")} />
              {pts.map((p, i) => <circle key={i} className="dot" cx={p[0]} cy={p[1]} r={4} />)}
              {last && <text className="direct-label" x={last[0] + 8} y={labelY[s.name] + 4}>{s.name}</text>}
            </g>
          );
        })}
        {xs.map((p, i) => (
          <rect key={`h${i}`} className="hit" x={x(i) - 18} y={T} width={36} height={H - T - B}
                onMouseMove={(e) => t.show(e, <>
                  <b>{p.tipTitle || p.label}</b>
                  {series.map((s) => s.values[i] != null && <div key={s.name}><i className={`key ${s.cls}`} />{s.name}: <b>{s.values[i]}</b></div>)}
                  {p.tip && <div className="tip-note">{p.tip}</div>}
                </>)} />
        ))}
      </svg>
      {t.el}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- dumbbell (before -> after)
// rows: [{label, before, after, n, sub, tip}] as counts out of n; shown as % of n
export function Dumbbell({ rows, beforeName = "before", afterName = "after" }) {
  const t = useTip();
  const W = 640, rowH = 34, L = 210, R = 56, T = 22;
  const H = T + rows.length * rowH + 8;
  const x = (f) => L + f * (W - L - R);
  return (
    <div className="chart" ref={t.box} onMouseLeave={t.hide}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="before and after">
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line className="grid-line" x1={x(f)} x2={x(f)} y1={T - 6} y2={H - 6} />
            <text className="axis-text" x={x(f)} y={T - 10} textAnchor="middle">{f * 100}%</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = T + i * rowH + rowH / 2;
          const b = r.before / r.n, a = r.after / r.n;
          return (
            <g key={i} onMouseMove={(e) => t.show(e, <>
              <b>{r.label}</b>
              <div><i className="key s0" />{beforeName}: <b>{r.before}/{r.n}</b></div>
              <div><i className="key s1" />{afterName}: <b>{r.after}/{r.n}</b></div>
              {r.tip && <div className="tip-note">{r.tip}</div>}
            </>)}>
              <rect className="hit" x={0} y={cy - rowH / 2} width={W} height={rowH} />
              <text className="row-label" x={L - 10} y={cy - 1} textAnchor="end">{r.label}</text>
              {r.sub && <text className="row-sub" x={L - 10} y={cy + 12} textAnchor="end">{r.sub}</text>}
              <line className="connector" x1={x(b)} x2={x(a)} y1={cy} y2={cy} />
              <circle className="dot s0" cx={x(b)} cy={cy} r={5} />
              <circle className="dot s1" cx={x(a)} cy={cy} r={6} />
              <text className="value-text" x={W - R + 8} y={cy + 4}>{Math.round(100 * a)}%</text>
            </g>
          );
        })}
      </svg>
      <div className="legend-row"><span><i className="key s0" />{beforeName}</span><span><i className="key s1" />{afterName}</span></div>
      {t.el}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- horizontal bars
// rows: [{label, value, cls, sub, tip}]
export function HBars({ rows, unit = "", max }) {
  const t = useTip();
  const m = max ?? Math.max(1, ...rows.map((r) => r.value));
  return (
    <div className="hbars" ref={t.box} onMouseLeave={t.hide}>
      {rows.map((r, i) => (
        <div key={i} className="hbar-row" onMouseMove={(e) => t.show(e, <><b>{r.label}</b><div>{r.value}{unit}</div>{r.tip && <div className="tip-note">{r.tip}</div>}</>)}>
          <span className="hbar-label">{r.label}{r.sub && <small>{r.sub}</small>}</span>
          <span className="hbar-track"><span className={`hbar-fill ${r.cls || "s1"}`} style={{ width: `${Math.max(0.6, (100 * r.value) / m)}%` }} /></span>
          <span className="hbar-value">{r.value}{unit}</span>
        </div>
      ))}
      {t.el}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- funnel (counts that narrow)
export function Funnel({ steps }) {
  const max = Math.max(1, ...steps.map((s) => s.value));
  return (
    <div className="funnel">
      {steps.map((s, i) => (
        <div key={i} className="funnel-step" title={s.tip || ""}>
          <div className="funnel-bar" style={{ width: `${Math.max(8, (100 * s.value) / max)}%` }} />
          <div className="funnel-text"><b>{s.value}</b> {s.label}</div>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- heatmap (success rate cells)
// rows: [{label, cells: [{label, k, n, tip}]}]; one sequential hue, light (low) -> dark (high)
export function Heatmap({ rows }) {
  const t = useTip();
  return (
    <div className="heatmap" ref={t.box} onMouseLeave={t.hide}>
      {rows.map((r) => (
        <div key={r.label} className="heat-row">
          <span className="heat-label">{r.label}</span>
          <div className="heat-cells">
            {r.cells.map((c, i) => {
              const rate = c.n ? c.k / c.n : null;
              const step = rate == null ? "none" : Math.min(6, Math.floor(rate * 6.999));
              return (
                <span key={i} className={`heat-cell q${step}`}
                      onMouseMove={(e) => t.show(e, <><b>{r.label} · {c.label}</b><div>{c.n ? `${c.k}/${c.n} succeeded` : "no solvable trials"}</div>{c.tip && <div className="tip-note">{c.tip}</div>}</>)}>
                  {c.n ? `${Math.round(100 * rate)}` : "–"}
                </span>
              );
            })}
          </div>
        </div>
      ))}
      <div className="legend-row"><span>success %:</span>{[0, 2, 4, 6].map((q) => <span key={q}><i className={`key heat-cell q${q}`} />{Math.round((q / 6) * 100)}</span>)}</div>
      {t.el}
    </div>
  );
}

// ------------------------------------------------------------------------------------------- tiny rate bar
export function RateBar({ k, n }) {
  const f = n ? k / n : 0;
  return (
    <span className="ratebar" title={`${k}/${n}`}>
      <span className="ratebar-track"><span className="ratebar-fill" style={{ width: `${100 * f}%` }} /></span>
      <span className="ratebar-text">{k}/{n}</span>
    </span>
  );
}
