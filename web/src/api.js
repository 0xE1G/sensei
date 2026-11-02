// Client for the Kintsugi server (server/app.py). Everything shown comes from saved files; the studio
// endpoints (uncached) start one job at a time and stream its progress.
const cache = new Map();

export async function api(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch(`/api/${path}`).then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return r.json();
  });
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

export const q = (o) => new URLSearchParams(o).toString();
export const getRuns = () => api("runs");
export const getRun = (id) => api(`run?${q({ id })}`);
export const getTrial = (run, name) => api(`trial?${q({ run, name })}`);
export const getReplay = (run, name) => api(`replay?${q({ run, name })}`);
export const getSkills = () => api("skills");
export const getHeadlines = () => api("headlines");
export const getProgress = () => api("progress");
export const getMeshes = () => api("meshes");
export const getAssetMesh = (name) => api(`asset_mesh/${encodeURIComponent(name)}`);

export const pct = (k, n) => (n ? `${Math.round((100 * k) / n)}%` : "–");
export const fmtP = (p) => (p == null ? "–" : p < 0.001 ? "p < 0.001" : `p = ${p.toPrecision(2)}`);

export const forget = (path) => cache.delete(path);

async function send(method, path, body) {
  const r = await fetch(`/api/${path}`, {
    method, headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.json();
}
export const studio = {
  info: () => send("GET", "studio/info"),
  spec: (prompt, backend) => send("POST", "studio/spec", { prompt, backend }),
  program: (prompt, task, backend) => send("POST", "studio/program", { prompt, task, backend }),
  start: (body) => send("POST", "studio/jobs", body),
  jobs: () => send("GET", "studio/jobs"),
  job: (id, offset) => send("GET", `studio/jobs/${id}?offset=${offset}`),
  stop: (id) => send("POST", `studio/jobs/${id}/stop`),
  workers: (id) => send("GET", `studio/live/${id}`),
  episode: (id, ep, offset) => send("GET", `studio/live/${id}/${ep}?offset=${offset}`),
};
// Settings (keys are sent once and never come back in full) and third-party asset downloads.
export const settingsApi = {
  get: () => send("GET", "settings"),
  save: (body) => send("POST", "settings", body),
  test: (backend) => send("POST", "settings/test", { backend }),
};
export const assetsApi = {
  status: () => send("GET", "assets"),
  download: (which) => send("POST", "assets/download", { which }),
};
