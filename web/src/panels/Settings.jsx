import { useEffect, useState } from "react";
import { assetsApi, settingsApi } from "../api.js";

// Settings: model access (keys stay on this machine; the server only ever returns "set" + the last four characters),
// photo tools and the third-party assets with download buttons, as one dialog with three sections. AssetGate blocks
// the app until the robot model is present.

export function useAssets() {
  const [st, setSt] = useState(null);
  useEffect(() => {
    let stop = false, timer;
    const tick = async () => {
      try {
        const s = await assetsApi.status();
        if (!stop) setSt(s);
        timer = setTimeout(tick, s.download.running ? 800 : 10000);
      } catch {
        if (!stop) setSt((cur) => cur ?? { error: true });       // no status: do not block the app on it
        timer = setTimeout(tick, 3000);
      }
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, []);
  const download = async (which) => {
    try { await assetsApi.download(which); setSt(await assetsApi.status()); } catch (e) { alert(e.message); }
  };
  return [st, download, setSt];
}

function Progress({ dl, which }) {
  if (dl?.which !== which) return null;
  if (dl.running) {
    const pct = dl.total ? Math.round((100 * dl.done) / dl.total) : 0;
    return (
      <div className="dl-progress">
        <div className={`banner-progress${dl.total ? "" : " indeterminate"}`}><span style={{ width: dl.total ? `${pct}%` : "30%" }} /></div>
        <span className="sub">{dl.total ? `${dl.done}/${dl.total} · ${dl.label}` : "listing files…"}</span>
      </div>
    );
  }
  if (dl.error) return <p className="sub warn">{dl.error}</p>;
  return dl.finished ? <p className="sub">Done at {dl.finished}.</p> : null;
}

function AssetRow({ item, which, dl, onDownload }) {
  const complete = which === "robot" ? item.present : item.present >= item.total && item.total > 0;
  const busy = dl?.running;
  return (
    <div className="asset-row">
      <div>
        <div className="asset-title">{item.what}{item.required ? <span className="pill">required</span> : <span className="pill">optional</span>}</div>
        <div className="sub">{item.from} · {item.license} · about {item.size_mb} MB
          {which === "scanned" ? ` · ${item.present}/${item.total} downloaded` : item.present ? " · downloaded" : " · not downloaded"}</div>
        <Progress dl={dl} which={which} />
      </div>
      {complete ? <Status tone="ok">Installed</Status> : (
        <button className="btn primary" disabled={busy} onClick={() => onDownload(which)}>
          {busy && dl.which === which ? "Downloading…" : which === "scanned" && item.present ? "Download the rest" : "Download"}
        </button>
      )}
    </div>
  );
}

export function AssetGate({ children }) {
  const [st, download] = useAssets();
  if (!st) return null;                                          // wait for the status (one quick request)
  if (st.error || st.robot.present) return children;
  return (
    <div className="asset-gate">
      <div className="card asset-gate-card">
        <h3>One download before the first run</h3>
        <p className="sub">The robot model is not on this machine yet. It comes from its public repository and is
          stored in <code>assets/</code> (never committed). Nothing is downloaded until you press the button.</p>
        <AssetRow item={st.robot} which="robot" dl={st.download} onDownload={download} />
        {st.download.finished && st.robot.present === false && !st.download.running && st.download.error &&
          <p className="sub warn">The download did not finish. Check the internet connection and try again.</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------- settings dialog
const Icon = ({ d, size = 18 }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8"
       strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{d}</svg>
);
const I = {
  models: <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />,
  photo: <><path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></>,
  download: <path d="M12 4v11M7 10l5 5 5-5M5 20h14" />,
  close: <path d="M6 6l12 12M18 6L6 18" />,
  eye: <><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>,
  eyeOff: <path d="M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3.2 4.2M6.6 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7a9.7 9.7 0 0 0 5.4-1.6" />,
  lock: <><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></>,
};

const MetaMark = () => (
  <svg viewBox="0 0 42 24" width="30" height="17" aria-hidden="true" className="sx-brand">
    <path d="M5 15.5C5 9.5 7.8 5 11.4 5c3.2 0 5.4 3.4 8.4 8.6S25.6 21 29.2 21c3.4 0 6-2.6 6-6.4C35.2 9.2 32.6 5 29 5c-3.3 0-5.6 3.6-8.6 8.6S14.8 21 11.6 21C7.8 21 5 18.8 5 15.5z"
          fill="none" stroke="#1877f2" strokeWidth="3.6" strokeLinejoin="round" />
  </svg>
);

function Status({ tone, children }) {                 // tone: ok | warn | off
  return <span className={`st-pill ${tone}`}><span className="st-dot" />{children}</span>;
}

function KeyField({ label, field, s, edit, setEdit }) {
  const [show, setShow] = useState(false);
  const cur = s[field];
  if (s[`${field}_source`] === "environment") {
    return <div className="sx-field"><label>{label}</label><div className="sx-static">Set by an environment variable ({cur?.hint})</div></div>;
  }
  return (
    <div className="sx-field">
      <label htmlFor={field}>{label}</label>
      <div className="sx-input-wrap">
        <input id={field} type={show ? "text" : "password"} autoComplete="off" spellCheck={false}
               placeholder={cur?.set ? `Saved (${cur.hint}) · paste a new key to replace it` : "Paste your API key"}
               value={edit[field] ?? ""} onChange={(e) => setEdit({ ...edit, [field]: e.target.value })} />
        <button type="button" className="sx-eye" onClick={() => setShow(!show)} aria-label={show ? "Hide key" : "Show key"}
                title={show ? "Hide" : "Show"}><Icon d={show ? I.eyeOff : I.eye} size={16} /></button>
      </div>
    </div>
  );
}

function TextField({ label, field, s, edit, setEdit, placeholder }) {
  if (s[`${field}_source`] === "environment") {
    return <div className="sx-field"><label>{label}</label><div className="sx-static">{s[field]} (environment variable)</div></div>;
  }
  return (
    <div className="sx-field">
      <label htmlFor={field}>{label}</label>
      <input id={field} type="text" autoComplete="off" spellCheck={false} placeholder={placeholder}
             value={edit[field] ?? s[field] ?? ""} onChange={(e) => setEdit({ ...edit, [field]: e.target.value })} />
    </div>
  );
}

function TestResult({ r }) {
  if (!r) return null;
  if (r === "testing") return <div className="sx-result"><span className="spinner" /> Testing the connection…</div>;
  return <div className={`sx-result ${r.ok ? "ok" : "bad"}`}>{r.ok ? "✓" : "✗"} {r.message}{r.seconds != null ? ` · ${r.seconds} s` : ""}</div>;
}

const BACKENDS = [
  ["spark", "Muse Spark", "Meta's model: plans tasks, diagnoses failures, writes new skills"],
  ["rule", "Offline rules", "No model: built-in rules for the standard tasks"],
];

function ModelsPane({ s, edit, setEdit, save, forget, setDefault, tests, test }) {
  const metaReady = s.meta_api_key?.set && s.meta_api_base;
  const dirty = ["meta_api_base", "meta_model", "meta_api_key"].some((f) => f in edit);
  return (
    <>
      <div className="sx-section-head"><h4>Default model</h4><p>Used by the Studio. Scripts and experiments keep the model they are given.</p></div>
      <div className="sx-choices two" role="radiogroup" aria-label="Default model">
        {BACKENDS.map(([k, name, desc]) => {
          const on = (s.default_backend || "") === k;
          const unavailable = k === "spark" && !metaReady;
          return (
            <button key={k} type="button" role="radio" aria-checked={on} className={`sx-choice ${on ? "on" : ""}`}
                    onClick={() => setDefault("default_backend", k)}>
              <span className="sx-radio" />
              <span className="sx-choice-text"><b>{name}</b><span>{desc}</span>{unavailable && <em>not set up yet</em>}</span>
            </button>
          );
        })}
      </div>
      <button type="button" className={`sx-auto ${!s.default_backend ? "on" : ""}`} onClick={() => setDefault("default_backend", "")}>
        <span className="sx-radio" />Pick the best available automatically</button>

      <div className="sx-card">
        <div className="sx-card-head">
          <MetaMark />
          <div><h5>Meta Model API</h5><p>Muse Spark, SAM 3.1 and Muse Image</p></div>
          {metaReady ? <Status tone="ok">Key saved {s.meta_api_key.hint}</Status> : <Status tone="off">Not set</Status>}
        </div>
        <KeyField label="API key" field="meta_api_key" s={s} edit={edit} setEdit={setEdit} />
        <details className="sx-adv">
          <summary>Advanced: address and model</summary>
          <TextField label="API address" field="meta_api_base" s={s} edit={edit} setEdit={setEdit} placeholder="https://api.meta.ai/v1" />
          <TextField label="Model" field="meta_model" s={s} edit={edit} setEdit={setEdit} placeholder="muse-spark-1.3" />
        </details>
        <div className="sx-actions">
          <button className="btn primary" disabled={!dirty} onClick={() => save(["meta_api_base", "meta_model", "meta_api_key"])}>Save</button>
          <button className="btn" disabled={!metaReady || dirty} onClick={() => test("spark")}>Test connection</button>
          {(s.meta_api_key_source === "settings" || s.meta_api_base_source === "settings") &&
            <button className="btn ghost danger" onClick={() => forget(["meta_api_base", "meta_model", "meta_api_key"])}>Remove key</button>}
        </div>
        <TestResult r={tests.spark} />
      </div>

    </>
  );
}

function PhotoPane({ s, edit, setEdit, save, forget }) {
  const dirty = ["sam_api_base", "sam_api_key"].some((f) => f in edit);
  return (
    <>
      <div className="sx-section-head"><h4>Photo tools</h4>
        <p>Start a task from a photo of a real table: Muse Spark reads the scene, SAM 3.1 finds each object and Muse Image
          recreates the tabletop. They use your Meta API key.</p></div>
      <div className="sx-card">
        <div className="sx-card-head">
          <MetaMark />
          <div><h5>SAM 3.1 + Muse Image</h5><p>Object boxes and the empty-table surface</p></div>
          {s.photo?.available ? <Status tone="ok">Ready</Status> : <Status tone="off">Needs the Meta key</Status>}
        </div>
        {!s.photo?.available && <p className="sx-hint">Add your Meta API key under Models.</p>}
        <details className="sx-adv">
          <summary>Advanced: a separate SAM endpoint</summary>
          <p className="sx-hint">Only needed if SAM runs somewhere other than the Meta Model API.</p>
          <TextField label="SAM address" field="sam_api_base" s={s} edit={edit} setEdit={setEdit} placeholder="https://…" />
          <KeyField label="SAM key" field="sam_api_key" s={s} edit={edit} setEdit={setEdit} />
          <div className="sx-actions">
            <button className="btn primary" disabled={!dirty} onClick={() => save(["sam_api_base", "sam_api_key"])}>Save</button>
            {(s.sam_api_key_source === "settings" || s.sam_api_base_source === "settings") &&
              <button className="btn ghost danger" onClick={() => forget(["sam_api_base", "sam_api_key"])}>Remove</button>}
          </div>
        </details>
      </div>
      <div className="sx-note"><Icon d={I.lock} size={15} />Inside the simulation the robot keeps the simulator's exact object
        masks; SAM is used on real photos.</div>
    </>
  );
}

function DownloadsPane({ assets, download }) {
  return (
    <>
      <div className="sx-section-head"><h4>Downloads</h4><p>Third-party assets, stored in <code>assets/</code> and never
        committed. Nothing downloads until you press a button.</p></div>
      {assets && !assets.error ? (<>
        <div className="sx-card"><AssetRow item={assets.robot} which="robot" dl={assets.download} onDownload={download} /></div>
        <div className="sx-card"><AssetRow item={assets.scanned} which="scanned" dl={assets.download} onDownload={download} /></div>
      </>) : <p className="sx-hint">Checking…</p>}
    </>
  );
}

const SECTIONS = [["models", "Models", I.models], ["photo", "Photo tools", I.photo], ["downloads", "Downloads", I.download]];

export function SettingsDrawer({ onClose, onChanged }) {
  const [s, setS] = useState(null);
  const [edit, setEdit] = useState({});
  const [note, setNote] = useState(null);
  const [tests, setTests] = useState({});
  const [tab, setTab] = useState("models");
  const [assets, download] = useAssets();
  useEffect(() => { settingsApi.get().then(setS).catch((e) => setNote({ bad: true, text: e.message })); }, []);
  const finished = assets?.download?.finished;
  useEffect(() => { if (finished) onChanged?.(); }, [finished]);   // the Studio's "not downloaded" hint goes away
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const flash = (text, bad = false) => { setNote({ text, bad }); setTimeout(() => setNote(null), 3500); };
  const reload = async () => { try { setS(await settingsApi.get()); } catch { /* keep the last view */ } };
  const save = async (fields) => {
    const body = Object.fromEntries(fields.filter((f) => f in edit).map((f) => [f, edit[f]]));
    try { await settingsApi.save(body); await reload(); setEdit({}); flash("Saved on this computer."); onChanged?.(); }
    catch (e) { flash(e.message, true); }
  };
  const forget = async (fields) => {
    try { await settingsApi.save(Object.fromEntries(fields.map((f) => [f, ""]))); await reload(); flash("Removed."); onChanged?.(); }
    catch (e) { flash(e.message, true); }
  };
  const test = async (backend) => {
    setTests((t) => ({ ...t, [backend]: "testing" }));
    try { const r = await settingsApi.test(backend); setTests((t) => ({ ...t, [backend]: r })); }
    catch (e) { setTests((t) => ({ ...t, [backend]: { ok: false, message: e.message } })); }
  };
  const setDefault = async (k, v) => {
    try { await settingsApi.save({ [k]: v }); await reload(); onChanged?.(); } catch (e) { flash(e.message, true); }
  };
  return (
    <div className="sx-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="sx-dialog" role="dialog" aria-modal="true" aria-label="Settings">
        <nav className="sx-nav">
          <div className="sx-title">Settings</div>
          {SECTIONS.map(([k, label, icon]) => (
            <button key={k} className={`sx-tab ${tab === k ? "on" : ""}`} onClick={() => setTab(k)} aria-current={tab === k}>
              <Icon d={icon} size={17} />{label}
            </button>
          ))}
          <div className="sx-nav-foot"><Icon d={I.lock} size={14} /><span>Keys are stored only on this computer
            {s?.config_file ? <> in <code>{s.config_file}</code></> : null} and never committed.</span></div>
        </nav>
        <div className="sx-body">
          <button className="sx-close" onClick={onClose} aria-label="Close settings"><Icon d={I.close} /></button>
          {!s ? <p className="sx-hint">{note?.text || "Loading…"}</p> : (
            tab === "models" ? <ModelsPane s={s} edit={edit} setEdit={setEdit} save={save} forget={forget} setDefault={setDefault} tests={tests} test={test} />
              : tab === "photo" ? <PhotoPane s={s} edit={edit} setEdit={setEdit} save={save} forget={forget} />
                : <DownloadsPane assets={assets} download={download} />
          )}
          {note && <div className={`sx-toast ${note.bad ? "bad" : ""}`}>{note.text}</div>}
        </div>
      </div>
    </div>
  );
}
