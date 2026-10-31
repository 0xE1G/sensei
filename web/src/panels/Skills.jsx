import { useState } from "react";
import { getSkills } from "../api.js";
import { RateBar } from "../charts.jsx";
import { Card, Chip, Code, Details, Guard, Loading, useAsync } from "../ui.jsx";

const nice = (s) => String(s ?? "").replace(/_/g, " ");
const cm = (m) => `${(100 * m).toFixed(1)}`;
const round = (v) => (typeof v === "number" ? +v.toPrecision(3) : v);
// held-out evidence comes as [before, after, n] counts or {before, after (rates), trials}
function heldOut(h) {
  if (Array.isArray(h)) return `${h[0]} → ${h[1]}/${h[2]}`;
  if (h && h.trials != null) {
    const c = (v) => (v <= 1 ? Math.round(v * h.trials) : v);
    return `${c(h.before)} → ${c(h.after)}/${h.trials}`;
  }
  return JSON.stringify(h);
}

function origin(card) {
  const src = card.lineage?.[card.lineage.length - 1]?.source || "seed";
  if (src === "seed") return ["measured", "hand-written seed"];
  if (src.includes("model")) return ["confirmed", "written by the model · gated"];
  if (card.kind === "composite") return ["confirmed", "verified program"];
  return ["confirmed", "found by search · gated"];
}

function preconditions(req) {
  if (!req) return "any object";
  const out = [];
  if (req.hollow === true) out.push("hollow");
  if (req.hollow === false) out.push("solid");
  if (req.rim) out.push("with a rim");
  if (req.min_width != null || req.max_width != null) out.push(`width ${req.min_width != null ? cm(req.min_width) : "0"}–${req.max_width != null ? cm(req.max_width) : "∞"} cm`);
  if (req.min_height != null || req.max_height != null) out.push(`height ${req.min_height != null ? cm(req.min_height) : "0"}–${req.max_height != null ? cm(req.max_height) : "∞"} cm`);
  return out.join(" · ") || "any object";
}

// shape groups like "hollow-wide-tall" or "solid-narrow-short-long" -> readable
const group = (g) => g.replace("-long", ", long").replace(/-/g, " ");

function CardDetail({ card, history, performance, patches }) {
  const [st, label] = origin(card);
  const perf = Object.entries(performance[card.name] || {});
  const myPatches = Object.entries(patches[card.name] || {});
  const versions = [...(history || []), card];
  const events = versions.flatMap((v) => (v.lineage || []).filter((e) => e.version === v.version || versions.length === 1));
  return (
    <div className="grid">
      <Card title={`${nice(card.name)} · v${card.version}`} actions={<Chip status={st}>{label}</Chip>}>
        <p className="lead">{card.description}</p>
        <div className="kv-line"><span>works on</span><span>{preconditions(card.requires)}</span></div>
        {card.check && <div className="kv-line"><span>success check</span><span className="sub">{nice(card.check)}</span></div>}
        {perf.length > 0 && (
          <div className="where">
            <div className="sub">where it works (measured)</div>
            {perf.map(([g, e]) => <div key={g} className="kv-line"><span>{group(g)}</span><RateBar k={e.success} n={e.trials} /></div>)}
          </div>
        )}
        {myPatches.length > 0 && (
          <div className="where">
            <div className="sub">parameter patches for specific shapes</div>
            {myPatches.map(([g, e]) => (
              <div key={g} className="kv-line"><span>{group(g)}</span>
                <span>{Object.entries(e.diff || e.params || {}).map(([k, v]) => Array.isArray(v) ? `${k}: ${v[0]} → ${v[1]}` : `${k} = ${v}`).join(", ")}
                  <span className="sub"> · {e.source}{e.evidence ? ` · ${e.evidence.success}/${e.evidence.trials}` : ""}</span></span></div>
            ))}
          </div>
        )}
        <Details label={`parameters (${Object.keys(card.params || {}).length})`}>
          <table>
            <tbody>{Object.entries(card.params || {}).map(([k, v]) => (
              <tr key={k}><td>{k}</td><td className="num">{v.default}</td><td className="sub">{v.meaning || v.unit || ""} {v.range ? `[${v.range.join(", ")}]` : ""}</td></tr>
            ))}</tbody>
          </table>
        </Details>
        {card.code && <Details label="code (model-written, passed the gates)"><Code>{card.code}</Code></Details>}
        {card.program && <Details label="program"><Code>{card.program}</Code></Details>}
      </Card>
      {events.length > 0 && (
        <Card title="History of this skill">
          <div className="timelineL">
            {events.map((e, i) => (
              <div key={i}>
                <b>v{e.version}</b> · {e.source}{e.date ? ` · ${e.date}` : ""}
                {e.trigger && <div className="sub">because {e.trigger.skill} failed on {nice(e.trigger.obj)} ({nice(e.trigger.label)})</div>}
                {e.written_by && <div className="sub">written by {e.written_by.backend}, {e.attempts} attempt(s)</div>}
                {e.results?.held_out && <div className="sub">held-out objects: {heldOut(e.results.held_out)}</div>}
                {e.diff && Object.keys(e.diff).length > 0 && <div className="sub">changed: {Object.entries(e.diff).map(([k, [a, b]]) => `${k} ${round(a)} → ${round(b)}`).join(", ")}</div>}
                {e.requires_diff && <div className="sub">preconditions: {preconditions(e.requires_diff[0])} → {preconditions(e.requires_diff[1])}</div>}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

// Every change to the library in time order: new cards (lineage), patches, routing evidence sources.
function Growth({ d, select }) {
  const ev = [];
  for (const c of d.active) for (const e of c.lineage || []) if (e.source !== "seed") ev.push({ date: e.date, what: `${nice(c.name)} v${e.version}`, how: e.source, card: c.name });
  for (const [sk, gs] of Object.entries(d.patches)) for (const [g, e] of Object.entries(gs)) ev.push({ date: e.date, what: `${nice(sk)} patch for ${group(g)}`, how: e.source || "search", card: sk });
  for (const c of d.composites) ev.push({ date: c.lineage?.[0]?.date, what: `program ${nice(c.name)}`, how: "verified program", card: c.name });
  ev.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const seeds = d.active.filter((c) => (c.lineage?.[0]?.source || "seed") === "seed").map((c) => nice(c.name));
  return (
    <Card title="How the library grew" status="confirmed">
      <div className="growth">
        <div className="growth-item seed"><b>seed</b><span>{seeds.join(", ")}</span></div>
        {ev.map((e, i) => (
          <button key={i} className="growth-item" onClick={() => select(e.card)}>
            <b>{e.what}</b><span>{e.how}{e.date ? ` · ${e.date}` : ""}</span>
          </button>
        ))}
      </div>
    </Card>
  );
}

const GROUPS = [
  ["Grasps", (c) => c.kind === "grasp"],
  ["Model-written skills", (c) => c.impl === "generated"],
  ["Moving and placing", (c) => c.kind !== "grasp" && c.impl !== "generated"],
];

export default function Skills() {
  const s = useAsync(getSkills, []);
  const [sel, setSel] = useState(null);
  return (
    <Loading state={s}>
      {(d) => {
        const all = [...d.active, ...d.composites, ...d.primitives, ...d.reference];
        const card = all.find((c) => c.name === sel) || d.active.find((c) => c.lineage?.[0]?.source !== "seed") || d.active[0];
        const list = (title, items) => items.length > 0 && (
          <div key={title}>
            <div className="exp-group">{title}</div>
            {items.map((c) => (
              <button key={c.name} className={card?.name === c.name ? "on" : ""} onClick={() => setSel(c.name)}>
                <span className="exp-title">{nice(c.name)}</span>
                <span className="sub">v{c.version}</span>
                <Chip status={origin(c)[0]} />
              </button>
            ))}
          </div>
        );
        return (
          <div className="grid">
            <div className="split">
              <Card title="Skill library">
                <div className="exp-list">
                  {GROUPS.map(([t, f]) => list(t, d.active.filter(f)))}
                  {list("Verified programs", d.composites)}
                  {list("Search primitives (used by the repair search)", d.primitives)}
                  {list("Reference (hand-written, comparison only)", d.reference)}
                </div>
              </Card>
              {card && <Guard resetKey={card.name} data={card}><CardDetail card={card} history={d.history[card.name]} performance={d.performance} patches={d.patches} /></Guard>}
            </div>
            <Guard data={d}><Growth d={d} select={setSel} /></Guard>
          </div>
        );
      }}
    </Loading>
  );
}
