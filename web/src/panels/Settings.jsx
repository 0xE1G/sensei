import { useEffect, useState } from "react";
import { assetsApi, settingsApi } from "../api.js";

// Settings drawer: model access (keys stay on this machine; the server never sends them back), mask source,
// and the third-party assets with download buttons. AssetGate blocks the app until the robot model is present.

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
      {!complete && (
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
    <div className="gate">
      <div className="card gate-card">
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

function Secret({ label, field, s, edit, setEdit }) {
  const cur = s[field];
  const fromEnv = s[`${field}_source`] === "environment";
  return (
    <label className="set-field">
      <span>{label}</span>
      {fromEnv ? <span className="sub">set by an environment variable ({cur?.hint})</span> : (
        <input type="password" autoComplete="off" placeholder={cur?.set ? `saved (${cur.hint}) · type to replace` : "paste the key"}
          value={edit[field] ?? ""} onChange={(e) => setEdit({ ...edit, [field]: e.target.value })} />
      )}
    </label>
  );
}

function Plain({ label, field, s, edit, setEdit, placeholder }) {
  const fromEnv = s[`${field}_source`] === "environment";
  return (
    <label className="set-field">
      <span>{label}</span>
      {fromEnv ? <span className="sub">{s[field]} (environment variable)</span> : (
        <input type="text" autoComplete="off" placeholder={placeholder}
          value={edit[field] ?? s[field] ?? ""} onChange={(e) => setEdit({ ...edit, [field]: e.target.value })} />
      )}
    </label>
  );
}

export function SettingsDrawer({ onClose, onChanged }) {
  const [s, setS] = useState(null);
  const [edit, setEdit] = useState({});
  const [msg, setMsg] = useState({});
  const [assets, download] = useAssets();
  useEffect(() => { settingsApi.get().then(setS).catch((e) => setMsg({ save: e.message })); }, []);
  const finished = assets?.download?.finished;
  useEffect(() => { if (finished) onChanged?.(); }, [finished]);   // the Studio's "not downloaded" hint goes away
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  const save = async (fields) => {
    const body = Object.fromEntries(fields.filter((f) => f in edit).map((f) => [f, edit[f]]));
    try {
      const r = await settingsApi.save(body);
      setS(r); setEdit({}); setMsg({ save: "Saved on this machine." }); onChanged?.();
    } catch (e) { setMsg({ save: e.message }); }
  };
  const forget = async (fields) => {
    try { setS(await settingsApi.save(Object.fromEntries(fields.map((f) => [f, ""])))); setMsg({ save: "Removed." }); onChanged?.(); }
    catch (e) { setMsg({ save: e.message }); }
  };
  const test = async (backend) => {
    setMsg({ ...msg, [backend]: "testing…" });
    try {
      const r = await settingsApi.test(backend);
      setMsg((m) => ({ ...m, [backend]: `${r.ok ? "works" : "failed"}: ${r.message} (${r.seconds} s)` }));
    } catch (e) { setMsg((m) => ({ ...m, [backend]: e.message })); }
  };
  const setDefault = async (k, v) => {
    try { setS(await settingsApi.save({ [k]: v })); onChanged?.(); } catch (e) { setMsg({ save: e.message }); }
  };
  return (
    <div className="drawer-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <aside className="drawer" role="dialog" aria-label="Settings">
        <div className="drawer-head"><h3>Settings</h3><button className="btn" onClick={onClose}>Close</button></div>
        {!s ? <p className="sub">{msg.save || "Loading…"}</p> : (
          <>
            <section>
              <div className="exp-group">Language model</div>
              <label className="set-field"><span>Default in Studio</span>
                <select value={s.default_backend || ""} onChange={(e) => setDefault("default_backend", e.target.value)}>
                  <option value="">automatic (best available)</option>
                  <option value="rule">Rule stand-in (offline, no model)</option>
                  <option value="claude_cli">Claude (claude CLI)</option>
                  <option value="spark">Meta Muse Spark</option>
                </select>
              </label>
              <p className="sub">Scripts and experiments are not affected: they use the backend they are given.</p>
              <div className="set-block">
                <div className="asset-title">Meta Muse Spark</div>
                <Plain label="API address" field="meta_api_base" s={s} edit={edit} setEdit={setEdit} placeholder="https://…/v1" />
                <Plain label="Model" field="meta_model" s={s} edit={edit} setEdit={setEdit} placeholder="muse-spark" />
                <Secret label="API key" field="meta_api_key" s={s} edit={edit} setEdit={setEdit} />
                <label className="toggle" title="A Spark call that fails or times out is sent to Claude instead; the answer is marked as Claude's and never counted as Spark's">
                  <input type="checkbox" checked={s.spark_fallback !== "0"} onChange={(e) => setDefault("spark_fallback", e.target.checked ? "1" : "0")} />
                  {" "}If Spark fails, ask Claude (marked as a fallback)</label>
                <div className="set-actions">
                  <button className="btn primary" onClick={() => save(["meta_api_base", "meta_model", "meta_api_key"])}>Save</button>
                  <button className="btn" disabled={!s.meta_api_key?.set || !s.meta_api_base} onClick={() => test("spark")}>Test connection</button>
                  {(s.meta_api_key_source === "settings" || s.meta_api_base_source === "settings") &&
                    <button className="btn" onClick={() => forget(["meta_api_base", "meta_model", "meta_api_key"])}>Remove</button>}
                </div>
                {msg.spark && <p className="sub">{msg.spark}</p>}
              </div>
              <div className="set-block">
                <div className="asset-title">Claude (development stand-in)</div>
                <p className="sub">{s.claude_cli_found ? "claude CLI found on this machine." : "claude CLI not found on PATH."}</p>
                <div className="set-actions"><button className="btn" disabled={!s.claude_cli_found} onClick={() => test("claude_cli")}>Test</button></div>
                {msg.claude_cli && <p className="sub">{msg.claude_cli}</p>}
              </div>
            </section>
            <section>
              <div className="exp-group">Object masks</div>
              <label className="set-field"><span>Default in Studio</span>
                <select value={s.default_masks || "sim"} onChange={(e) => setDefault("default_masks", e.target.value)}>
                  <option value="sim">Simulator masks</option>
                  <option value="sam" disabled={!s.sam.available}>SAM 3{s.sam.available ? "" : " (not available)"}</option>
                </select>
              </label>
              {!s.sam.available && <p className="sub">{s.sam.reason}.</p>}
              <div className="set-block">
                <div className="asset-title">SAM 3 access</div>
                <Plain label="API address" field="sam_api_base" s={s} edit={edit} setEdit={setEdit} placeholder="https://…" />
                <Secret label="API key" field="sam_api_key" s={s} edit={edit} setEdit={setEdit} />
                <div className="set-actions">
                  <button className="btn primary" onClick={() => save(["sam_api_base", "sam_api_key"])}>Save</button>
                  {(s.sam_api_key_source === "settings" || s.sam_api_base_source === "settings") &&
                    <button className="btn" onClick={() => forget(["sam_api_base", "sam_api_key"])}>Remove</button>}
                </div>
              </div>
            </section>
            <section>
              <div className="exp-group">Downloads</div>
              {assets ? (<>
                <AssetRow item={assets.robot} which="robot" dl={assets.download} onDownload={download} />
                <AssetRow item={assets.scanned} which="scanned" dl={assets.download} onDownload={download} />
              </>) : <p className="sub">Checking…</p>}
            </section>
            {msg.save && <p className="sub">{msg.save}</p>}
            <p className="sub foot">Keys are stored only on this computer ({s.config_file}, not committed) and are never shown again in full.</p>
          </>
        )}
      </aside>
    </div>
  );
}
