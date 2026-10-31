import { useState } from "react";
import { fmtP, getRun } from "../api.js";
import { Card, Chip, Code, KV, Loading, RunPicker, useAsync } from "../ui.jsx";

const KINDS = ["failure_search", "explore", "create_skill", "transfer", "verify_program", "auto_repair", "recovery"];

function heatColor(rate) {
  if (rate == null) return "transparent";
  const h = Math.round(120 * rate); // red (0) -> green (120)
  return `hsla(${h}, 60%, 50%, 0.28)`;
}

function Spark({ values, height = 40 }) {
  if (!values.length) return null;
  const w = 220, max = Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${height - (v / max) * (height - 4) - 2}`);
  return (
    <svg width={w} height={height} style={{ display: "block" }}>
      <polyline points={pts.join(" ")} fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.8" />
    </svg>
  );
}

function FailureSearch({ rep }) {
  const sizes = {};
  for (const c of rep.capability_map || []) {
    sizes[c.template] = sizes[c.template] || {};
    const k = c.size;
    const cell = (sizes[c.template][k] = sizes[c.template][k] || { s: 0, f: 0, u: 0 });
    cell.s += c.success; cell.f += c.valid_failures; cell.u += c.unsolvable;
  }
  return (
    <div className="grid">
      <Card title={`Strategy comparison (${rep.budget_per_strategy} trials each, library: ${rep.library})`} status="measured">
        <table>
          <thead><tr><th>strategy</th><th className="num">valid failures</th><th className="num">per 100</th><th className="num">clusters</th><th className="num">unsolvable</th><th>first found (trial)</th></tr></thead>
          <tbody>
            {Object.entries(rep.strategies).map(([k, s]) => (
              <tr key={k}><td>{k}</td><td className="num">{s.metrics.valid_failures}</td><td className="num">{s.metrics.valid_failures_per_100}</td>
                <td className="num">{s.metrics.distinct_clusters}</td><td className="num">{s.metrics.unsolvable}</td>
                <td className="sub">{Object.entries(s.metrics.trials_to_find).map(([c, n]) => `${c} @${n}`).join(", ")}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div className="grid grid-2">
        <Card title="Capability map: success among solvable trials" status="measured">
          {Object.entries(sizes).map(([t, cells]) => (
            <div key={t} style={{ marginBottom: 8 }}>
              <b>{t}</b>
              <table className="heat"><tbody><tr>
                {Object.entries(cells).sort().map(([k, c]) => {
                  const n = c.s + c.f;
                  return <td key={k} style={{ background: heatColor(n ? c.s / n : null) }}>
                    <div className="sub">{k}</div>{n ? `${c.s}/${n}` : "–"}</td>;
                })}
              </tr></tbody></table>
            </div>
          ))}
        </Card>
        <Card title="Failure clusters (handed to the explorer)" status="measured">
          <table>
            <thead><tr><th>object / skill</th><th className="num">failures</th><th>labels</th></tr></thead>
            <tbody>{(rep.clusters || []).map((c) => (
              <tr key={c.key}><td>{c.key}</td><td className="num">{c.count}</td><td className="sub">{JSON.stringify(c.labels)}</td></tr>
            ))}</tbody>
          </table>
          {rep.strategies.model?.model_calls && (
            <>
              <h3 style={{ marginTop: 12 }}>Model-guided search: the model's reasoning per batch <Chip status="hypothesis" /></h3>
              <ol className="sub">{rep.strategies.model.model_calls.map((c, i) => <li key={i}>{c.rationale}</li>)}</ol>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}

function Gates({ gates }) {
  return (
    <table>
      <thead><tr><th>change</th><th>failed cases fixed</th><th>held-out before → after</th><th>regression</th><th>result</th></tr></thead>
      <tbody>{gates.map((g, i) => (
        <tr key={i}>
          <td>{g.card}</td>
          <td>{g.fixes_cluster.success}/{g.fixes_cluster.trials} <span className="sub">{fmtP(g.fixes_cluster.p_value)}</span></td>
          <td>{g.held_out.before.success} → {g.held_out.success} / {g.held_out.trials} <span className="sub">{fmtP(g.held_out.p_value)}{g.held_out.mode ? ` · must ${g.held_out.mode}` : ""}</span></td>
          <td>{g.no_regression.pass ? <span className="ok">none</span> : <span className="bad">regressed</span>}
            <div className="sub">{Object.keys(g.no_regression.tasks).join(", ")}</div></td>
          <td><Chip status={g.pass ? "confirmed" : "rejected"}>{g.pass ? "passed" : "failed"}</Chip></td>
        </tr>
      ))}</tbody>
    </table>
  );
}

function Explore({ rep, open }) {
  const ph = rep.phases || {};
  return (
    <div className="grid">
      <div className="grid grid-3">
        <Card title="Failure cluster" status="measured">
          <p>{ph.before?.cluster?.statement || "no failure cluster"}</p>
          <div className="sub">before: {ph.before?.success}/{ph.before?.trials} tasks succeed</div>
        </Card>
        <Card title="Outcome" status="measured">
          <div className="big" style={{ fontSize: 22 }}>{rep.outcome}</div>
          <div className="sub">{rep.trials_simulated} simulated trials · {rep.wall_time_s} s</div>
          {rep.promoted && <div>card <b>{rep.promoted.card?.name}</b> v{rep.promoted.card?.version}</div>}
          {rep.patched && <div>patch {JSON.stringify(rep.patched.values)} for {rep.patched.groups?.join(", ")}</div>}
          {rep.routed && <div>routing: {rep.routed.card} instead of {rep.routed.instead_of}</div>}
        </Card>
        {ph.existing && (
          <Card title="Existing skills tried first" status="measured">
            {ph.existing.candidates.length ? ph.existing.candidates.map((c) => (
              <div key={c.card}>{c.card}: {c.success}/{c.trials}</div>
            )) : <span className="muted">no other skill fits</span>}
          </Card>
        )}
      </div>
      {ph.sample && (
        <Card title="Grasp candidates: sampled → fit the gripper → reachable" status="measured">
          <p>{ph.sample.n_sampled} → {ph.sample.n_geometric} → {ph.sample.n_survivors}
            <span className="sub"> · rejected: {Object.entries(ph.sample.reject_reasons).map(([k, v]) => `${k} ${v}`).join(", ")}</span></p>
          {ph.families && (
            <table>
              <thead><tr><th>family</th><th>primitive</th><th className="num">success</th><th style={{ width: "40%" }}></th></tr></thead>
              <tbody>{ph.families.families.map((f) => (
                <tr key={f.family}><td>{f.family}</td><td className="sub">{f.primitive || "feature_pinch"}</td>
                  <td className="num">{f.success}/{f.trials}</td>
                  <td><div className="bar"><span style={{ width: `${100 * f.rate}%` }} /></div></td></tr>
              ))}</tbody>
            </table>
          )}
        </Card>
      )}
      <div className="grid grid-2">
        {(ph.tune || []).map((t, i) => (
          <Card key={i} title={`Tuning: ${t.family}`} status="measured">
            <Spark values={t.history.map((h) => h.value)} />
            <div className="sub">{t.history.length} Optuna trials under perturbed physics{t.stopped ? ` · ${t.stopped}` : ""}; chosen
              trial {t.best.trial} (confirmation {t.best.confirm})</div>
          </Card>
        ))}
        {ph.patch && (
          <Card title="Parameter patch candidates" status="measured">
            <table><tbody>{ph.patch.candidates.map((c, i) => (
              <tr key={i} className={c.name === ph.patch.chosen ? "sel" : ""}><td>{c.name}</td><td className="sub">{c.source}</td>
                <td className="num">{c.confirm.success}/{c.confirm.trials}</td></tr>
            ))}</tbody></table>
            <div className="sub">groups: {ph.patch.groups.join(", ")}</div>
          </Card>
        )}
      </div>
      {ph.gates && <Card title="Gates" status="measured"><Gates gates={ph.gates} /></Card>}
    </div>
  );
}

function CreateSkill({ rep }) {
  const ph = rep.phases || {};
  return (
    <div className="grid">
      <Card title="A capability the library did not have" status="measured">
        <KV rows={[["capability", rep.capability], ["task", rep.task],
          ["before", `${ph.before?.success}/${ph.before?.trials} (${JSON.stringify(ph.before?.labels)}); program: ${ph.before?.missing_capability || ph.before?.program_error}`],
          ["outcome", rep.outcome], ["trials", rep.trials_simulated]]} />
      </Card>
      {(ph.proposals || []).map((p, i) => {
        const s = (ph.search || []).find((x) => String(x.attempt).startsWith(String(p.attempt)) && (ph.proposals.length === 1 || String(x.attempt).endsWith("abc"[p.option] || String(p.attempt))));
        return (
          <Card key={i} title={`Proposal ${p.attempt}${p.option != null ? "abc"[p.option] || "" : ""}: ${p.card.name}`}
                actions={<Chip status="hypothesis">written by {p.meta.backend}</Chip>}>
            <p className="sub">{p.card.description}</p>
            <Code>{p.card.code}</Code>
            {s && <div className="sub">search: best {s.candidates[0].success}/{s.candidates[0].trials}; {s.candidates.filter((c) => c.success > 0).length}/{s.candidates.length} parameter sets succeed at least once</div>}
            {p.feedback && <details><summary className="sub">feedback it received</summary><Code>{p.feedback}</Code></details>}
          </Card>
        );
      })}
      {ph.gate && (
        <Card title="Gate" status={ph.gate.pass ? "confirmed" : "rejected"}>
          <p>held-out {ph.gate.held_out.success}/{ph.gate.held_out.trials} vs {ph.gate.held_out.before}/{ph.gate.held_out.trials} before ({fmtP(ph.gate.held_out.p_value)}); regression {ph.gate.no_regression.pass ? "none" : "found"}</p>
        </Card>
      )}
    </div>
  );
}

function Generic({ rep }) {
  return <Card title="Report"><Code>{JSON.stringify(rep, null, 1).slice(0, 20000)}</Code></Card>;
}

function Transfer({ rep }) {
  return (
    <div className="grid grid-2">
      <Card title="Retrieval" actions={<Chip status="hypothesis">model's choice</Chip>}>
        <p><b>{rep.retrieval.output.choice}</b>: {rep.retrieval.output.explanation}</p>
        <div className="sub">candidates: {rep.candidates.map((c) => c.card).join(", ")}</div>
      </Card>
      <Card title="Verification vs from scratch" status="measured">
        <p>{rep.verification.card}: {rep.verification.success}/{rep.verification.trials} on new {rep.template}s →{" "}
          <Chip status={rep.verification.trusted ? "confirmed" : "rejected"}>{rep.verification.trusted ? "trusted" : "not trusted"}</Chip></p>
        {rep.scratch && <p>from scratch: {rep.scratch.trials_total} trials ({rep.scratch.trials_search} search), outcome {rep.scratch.outcome}</p>}
      </Card>
    </div>
  );
}

const VIEWS = { failure_search: FailureSearch, explore: Explore, create_skill: CreateSkill, transfer: Transfer };

export default function Experiments({ runs, open, sel }) {
  const fromSel = runs.find((r) => r.id === sel?.run && KINDS.includes(r.kind));
  const first = fromSel || runs.find((r) => r.kind === "explore");
  const [id, setId] = useState(first?.id || "");
  const run = useAsync(() => (id ? getRun(id) : Promise.resolve(null)), [id]);
  return (
    <div className="grid">
      <div className="toolbar">
        <RunPicker runs={runs} kinds={KINDS} value={id} onChange={setId} label="Experiment" />
      </div>
      <Loading state={run}>
        {(r) => {
          if (!r?.report) return <p className="muted">Choose an experiment run.</p>;
          const View = VIEWS[r.kind] || Generic;
          return <View rep={r.report} open={open} />;
        }}
      </Loading>
    </div>
  );
}
