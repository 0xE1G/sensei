import { useState } from "react";
import { getSkills } from "../api.js";
import { Card, Chip, Code, Loading, useAsync } from "../ui.jsx";

function Lineage({ card, history }) {
  const versions = [...(history || []), card];
  return (
    <div className="timelineL">
      {versions.flatMap((v) => (v.lineage || []).filter((e) => e.version === v.version || versions.length === 1)).map((e, i) => (
        <div key={i}>
          <b>v{e.version}</b> · {e.source}{e.date ? ` · ${e.date}` : ""}
          {e.trigger && <div className="sub">trigger: {e.trigger.skill} failed on {e.trigger.obj} ({e.trigger.label}) in {e.trigger.task}</div>}
          {e.parent && <div className="sub">parent: {JSON.stringify(e.parent)}</div>}
          {e.results && <div className="sub">results: {JSON.stringify(e.results)}</div>}
          {e.written_by && <div className="sub">written by {e.written_by.backend} ({e.written_by.model}), {e.attempts} attempt(s)</div>}
          {e.diff && Object.keys(e.diff).length > 0 && (
            <table style={{ marginTop: 4 }}><thead><tr><th>parameter</th><th className="num">before</th><th className="num">after</th></tr></thead>
              <tbody>{Object.entries(e.diff).map(([k, [a, b]]) => <tr key={k}><td>{k}</td><td className="num">{a}</td><td className="num">{b}</td></tr>)}</tbody></table>
          )}
          {e.requires_diff && <div className="sub">preconditions: {JSON.stringify(e.requires_diff[0])} → {JSON.stringify(e.requires_diff[1])}</div>}
        </div>
      ))}
    </div>
  );
}

function CardDetail({ card, history }) {
  return (
    <div className="grid">
      <Card title={`${card.name} v${card.version}`}
            actions={<Chip status={card.lineage?.[0]?.source === "seed" ? "measured" : "confirmed"}>
              {card.lineage?.[0]?.source === "seed" ? "hand-written seed" : `${card.lineage?.[card.lineage.length - 1]?.source} · gated`}</Chip>}>
        <p>{card.description}</p>
        <div className="sub">kind {card.kind}{card.impl ? ` · runs ${card.impl}` : ""} · check: {card.check}</div>
        {card.requires && <div className="sub">preconditions: {JSON.stringify(card.requires)}</div>}
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>parameter</th><th className="num">value</th><th>range</th><th>meaning</th></tr></thead>
          <tbody>{Object.entries(card.params || {}).map(([k, v]) => (
            <tr key={k}><td>{k}</td><td className="num">{v.default}</td><td className="sub">{JSON.stringify(v.range)}</td>
              <td className="sub">{v.meaning || v.unit || ""}</td></tr>
          ))}</tbody>
        </table>
      </Card>
      {card.code && <Card title="Code (model-written primitive)" actions={<Chip status="confirmed">passed the gates</Chip>}><Code>{card.code}</Code></Card>}
      {card.program && <Card title="Program (verified composite)"><Code>{card.program}</Code></Card>}
      {card.lineage && <Card title="Lineage"><Lineage card={card} history={history} /></Card>}
    </div>
  );
}

export default function Skills() {
  const s = useAsync(getSkills, []);
  const [sel, setSel] = useState(null);
  return (
    <Loading state={s}>
      {(d) => {
        const all = [...d.active, ...d.composites, ...d.primitives, ...d.reference];
        const card = all.find((c) => c.name === sel) || d.active.find((c) => c.lineage?.[0]?.source !== "seed") || d.active[0];
        const group = (title, list) => list.length > 0 && (
          <div key={title}>
            <div className="sub" style={{ margin: "8px 0 2px" }}>{title}</div>
            <div className="list" style={{ maxHeight: "none" }}>
              {list.map((c) => (
                <button key={c.name} className={card?.name === c.name ? "on" : ""} onClick={() => setSel(c.name)}>
                  <span>{c.name}</span><span className="sub">v{c.version}{c.lineage?.[0]?.source && c.lineage[0].source !== "seed" ? ` · ${c.lineage[c.lineage.length - 1].source}` : ""}</span>
                </button>
              ))}
            </div>
          </div>
        );
        return (
          <div className="grid">
            <div className="split">
              <Card title="Skill library">
                {group("Active skills", d.active)}
                {group("Verified programs (composites)", d.composites)}
                {group("Search primitives", d.primitives)}
                {group("Reference (hand-written, comparison only)", d.reference)}
              </Card>
              {card && <CardDetail card={card} history={d.history[card.name]} />}
            </div>
            <div className="grid grid-2">
              <Card title="Routing evidence: which skill for which shape" status="measured">
                <table>
                  <thead><tr><th>skill</th><th>shape group</th><th className="num">success</th></tr></thead>
                  <tbody>{Object.entries(d.performance).flatMap(([c, gs]) => Object.entries(gs).map(([g, e]) => (
                    <tr key={c + g}><td>{c}</td><td>{g}</td><td className="num">{e.success}/{e.trials}</td></tr>
                  )))}</tbody>
                </table>
              </Card>
              <Card title="Parameter patches (per shape group)" status="measured">
                <table>
                  <thead><tr><th>skill</th><th>shape group</th><th>change</th><th className="num">evidence</th><th>source</th></tr></thead>
                  <tbody>{Object.entries(d.patches).flatMap(([sk, gs]) => Object.entries(gs).map(([g, e]) => (
                    <tr key={sk + g}><td>{sk}</td><td>{g}</td>
                      <td>{Object.entries(e.diff || e.params || {}).map(([k, v]) => Array.isArray(v) ? `${k}: ${v[0]} → ${v[1]}` : `${k} = ${v}`).join(", ")}</td>
                      <td className="num">{e.evidence ? `${e.evidence.success}/${e.evidence.trials}` : "–"}</td><td className="sub">{e.source}</td></tr>
                  )))}</tbody>
                </table>
              </Card>
            </div>
          </div>
        );
      }}
    </Loading>
  );
}
