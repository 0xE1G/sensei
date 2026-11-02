import { useEffect, useRef, useState } from "react";
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

function shortOrigin(card) {
  const src = card.lineage?.[card.lineage.length - 1]?.source || "seed";
  if (card.kind === "composite") return "program";
  return src === "seed" ? "seed" : src.includes("model") ? "model" : src.includes("handwritten") ? "manual" : "search";
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
  const evidence = perf.reduce((a, [, e]) => [a[0] + e.success, a[1] + e.trials], [0, 0]);
  return (
    <div className="grid">
      <section className="card">
        <div className="skill-head">
          <h3>{nice(card.name)}</h3><span className="ver">v{card.version}</span>
          {card.origin === "user" && <span className="ver" title="In this checkout's user library (user_library/), not in the shipped library">local</span>}
          <Chip status={st === "measured" ? "hypothesis" : st}>{label}</Chip>
        </div>
        <p className="lead">{card.description}</p>
        <div className="facts-row">
          <div><div className="k">Works on</div><div className="v">{preconditions(card.requires)}</div></div>
          <div><div className="k">Success check</div><div className="v">{nice(card.check) || "–"}</div></div>
          <div><div className="k">Evidence</div><div className="v">{evidence[1] ? <RateBar k={evidence[0]} n={evidence[1]} /> : "–"}</div></div>
        </div>
        {perf.length > 0 && (
          <>
            <div className="section-label">Success by shape</div>
            {perf.map(([g, e]) => <div key={g} className="perf-row"><span>{group(g)}</span><RateBar k={e.success} n={e.trials} /></div>)}
          </>
        )}
        {myPatches.length > 0 && (
          <>
            <div className="section-label">Patches</div>
            {myPatches.map(([g, e]) => (
              <div key={g} className="perf-row"><span>{group(g)}</span>
                <span className="sub">{Object.entries(e.diff || e.params || {}).map(([k, v]) => Array.isArray(v) ? `${k} ${round(v[0])} → ${round(v[1])}` : `${k} = ${round(v)}`).join(", ")}
                  {e.evidence ? ` · ${e.evidence.success}/${e.evidence.trials}` : ""}</span></div>
            ))}
          </>
        )}
        <Details label={`Parameters (${Object.keys(card.params || {}).length})`}>
          <table>
            <tbody>{Object.entries(card.params || {}).map(([k, v]) => (
              <tr key={k}><td>{k}</td><td className="num">{round(v.default)}</td><td className="sub">{v.meaning || v.unit || ""} {v.range ? `[${v.range.join(", ")}]` : ""}</td></tr>
            ))}</tbody>
          </table>
        </Details>
        {card.code && <Details label="Code"><Code>{card.code}</Code></Details>}
        {card.program && <Details label="Program"><Code>{card.program}</Code></Details>}
      </section>
      {events.length > 0 && (
        <Card title="Versions">
          <ol className="versions">
            {events.map((e, i) => (
              <li key={i}>
                <span className="vtag">v{e.version}</span>
                <div>
                  <div>{e.source}{e.date ? <span className="sub"> · {e.date}</span> : null}</div>
                  {e.trigger && <div className="sub">Trigger: {e.trigger.skill} failed on {nice(e.trigger.obj)} ({nice(e.trigger.label)})</div>}
                  {e.written_by && <div className="sub">Written by {e.written_by.backend}, {e.attempts} attempt(s)</div>}
                  {e.results?.held_out && <div className="sub">Held-out: {heldOut(e.results.held_out)}</div>}
                  {e.diff && Object.keys(e.diff).length > 0 && <div className="sub">Changed: {Object.entries(e.diff).map(([k, [a, b]]) => `${k} ${round(a)} → ${round(b)}`).join(", ")}</div>}
                  {e.requires_diff && <div className="sub">Preconditions: {preconditions(e.requires_diff[0])} → {preconditions(e.requires_diff[1])}</div>}
                </div>
              </li>
            ))}
          </ol>
        </Card>
      )}
    </div>
  );
}

// Every change to the library in time order: seed skills, new cards, patches, verified programs.
function Timeline({ d, select }) {
  const strip = useRef(null);
  const [edge, setEdge] = useState({ left: false, right: false });
  const update = () => {
    const el = strip.current;
    if (el) setEdge({ left: el.scrollLeft > 4, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 4 });
  };
  useEffect(() => {
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  const scroll = (dir) => strip.current?.scrollBy({ left: dir * strip.current.clientWidth * 0.7, behavior: "smooth" });
  const ev = [];
  for (const c of d.active) for (const e of c.lineage || []) if (e.source !== "seed") ev.push({ date: e.date, what: `${nice(c.name)} v${e.version}`, how: e.source, card: c.name, kind: "new" });
  for (const [sk, gs] of Object.entries(d.patches)) for (const [g, e] of Object.entries(gs)) ev.push({ date: e.date, what: `${nice(sk)} patch`, how: group(g), card: sk, kind: "patch" });
  for (const c of d.composites) ev.push({ date: c.lineage?.[0]?.date, what: nice(c.name), how: "verified program", card: c.name, kind: "new" });
  ev.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  const seeds = d.active.filter((c) => (c.lineage?.[0]?.source || "seed") === "seed");
  return (
    <Card title="Library timeline" actions={<span className="sub">{seeds.length} seed skills · {ev.length} verified changes</span>}>
      <div className={`tl-wrap ${edge.left ? "fade-l" : ""} ${edge.right ? "fade-r" : ""}`}>
      {edge.left && <button className="tl-arrow left" onClick={() => scroll(-1)} aria-label="earlier">‹</button>}
      {edge.right && <button className="tl-arrow right" onClick={() => scroll(1)} aria-label="later">›</button>}
      <div className="timeline-strip" ref={strip} onScroll={update}>
        <button className="tl-node seed" onClick={() => select(seeds[0]?.name)} title={seeds.map((c) => nice(c.name)).join(", ")}>
          <b>Seed library</b><span>{seeds.length} hand-written skills</span>
        </button>
        {ev.map((e, i) => (
          <button key={i} className={`tl-node ${e.kind}`} onClick={() => select(e.card)} title={`${e.what} · ${e.how}${e.date ? ` · ${e.date}` : ""}`}>
            <b>{e.what}</b><span>{e.how}</span>
          </button>
        ))}
      </div>
      </div>
    </Card>
  );
}

const GROUPS = [
  ["Grasps", (c) => c.kind === "grasp"],
  ["Generated skills", (c) => c.impl === "generated"],
  ["Transport and placing", (c) => c.kind !== "grasp" && c.impl !== "generated"],
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
                <span className="origin-word">{shortOrigin(c)}{c.origin === "user" ? " · local" : ""}</span>
              </button>
            ))}
          </div>
        );
        return (
          <div className="grid">
            <Guard data={d}><Timeline d={d} select={setSel} /></Guard>
            <div className="split">
              <Card title="Library">
                <div className="exp-list">
                  {GROUPS.map(([t, f]) => list(t, d.active.filter(f)))}
                  {list("Verified programs", d.composites)}
                  {list("Search primitives", d.primitives)}
                  {list("Reference", d.reference)}
                </div>
              </Card>
              {card && <Guard resetKey={card.name} data={card}><CardDetail card={card} history={d.history[card.name]} performance={d.performance} patches={d.patches} /></Guard>}
            </div>
          </div>
        );
      }}
    </Loading>
  );
}
