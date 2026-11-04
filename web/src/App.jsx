import { useRef, useState } from "react";
import { getRuns } from "./api.js";
import { Guard, Loading, useAsync } from "./ui.jsx";
import Studio from "./panels/Studio.jsx";
import Progress from "./panels/Progress.jsx";
import Episodes from "./panels/Episodes.jsx";
import Investigator from "./panels/Investigator.jsx";
import Experiments from "./panels/Experiments.jsx";
import Skills from "./panels/Skills.jsx";
import { AssetGate, SettingsDrawer } from "./panels/Settings.jsx";

// Small inline icons (24px grid, stroke = currentColor); no icon library.
const ICONS = {
  studio: <path d="M5 4l14 8-14 8z" />,
  episodes: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M7 5v14M17 5v14M3 12h4M17 12h4" /></>,
  progress: <path d="M4 19V5M4 19h16M8 15l4-5 3 3 5-6" />,
  investigator: <><circle cx="11" cy="11" r="6" /><path d="M20 20l-4.5-4.5" /></>,
  experiments: <path d="M9 3h6M10 3v6l-5 9a2 2 0 002 3h10a2 2 0 002-3l-5-9V3M7.5 15h9" />,
  skills: <path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5" />,
  // a cog (Lucide "settings", ISC): the old circle with rays read as a light-mode sun
  settings: <><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" /><circle cx="12" cy="12" r="3" /></>,
};

const SECTIONS = [
  ["Work", [["studio", "Studio", "Describe a task, run it in simulation, and repair what fails."],
            ["episodes", "Episodes", "Recorded episodes with the plan, skill timeline and robot cameras."]]],
  ["Insight", [["progress", "Progress", "Repairs, transfer and diagnosis, measured across runs."],
               ["investigator", "Investigator", "Failure diagnosis with evidence and one-change tests."],
               ["experiments", "Experiments", "Failure searches, repairs and skill creation, with their gates."]]],
  ["Library", [["skills", "Skills", "Verified skills, where they work, and how the library changed."]]],
];
const TAB = Object.fromEntries(SECTIONS.flatMap(([, items]) => items.map(([k, title, desc]) => [k, { title, desc }])));

export default function App() {
  const [tab, setTab] = useState("studio");
  const [showSettings, setShowSettings] = useState(false);
  const [settingsVersion, setSettingsVersion] = useState(0);   // the Studio re-reads backends / masks when it changes
  const [sel, setSel] = useState({ run: "demo/1_bowl_before_repair", trial: "00001" });
  const runs = useAsync(getRuns, [tab]);   // re-read on tab change: studio jobs add runs
  const lastRuns = useRef(null);           // keep showing the previous list while it refreshes (no flash)
  if (runs.data) lastRuns.current = runs.data;
  const runsState = lastRuns.current ? { data: lastRuns.current, loading: false, error: null } : runs;
  const open = (next) => {
    setSel((s) => ({ ...s, ...next }));
    if (next.tab) setTab(next.tab);
  };
  return (
    <div className="shell top">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <svg className="logo-mark" viewBox="0 0 32 32" aria-label="Kintsugi">
              <defs><clipPath id="logo-diamond"><path d="M16 6L26 16L16 26L6 16Z" /></clipPath></defs>
              <rect width="32" height="32" rx="8" />
              <path clipPath="url(#logo-diamond)" d="M4 8H28M4 11.5H28M4 15H28M4 18.5H28M4 22H28M4 25.5H28" />
            </svg>
            <h1>KINTSUGI</h1>
          </div>
          <nav className="top-nav">
            {SECTIONS.map(([section, items], si) => (
              <div key={section} className="nav-group" title={section}>
                {si > 0 && <span className="nav-sep" aria-hidden="true" />}
                {items.map(([k, label]) => (
                  <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)} aria-current={tab === k ? "page" : undefined}>
                    <svg viewBox="0 0 24 24" aria-hidden="true">{ICONS[k]}</svg>{label}
                  </button>
                ))}
              </div>
            ))}
          </nav>
          <button className="top-settings" onClick={() => setShowSettings(true)} title="Settings: model access, masks, downloads" aria-label="Settings">
            <svg viewBox="0 0 24 24" aria-hidden="true">{ICONS.settings}</svg><span>Settings</span>
          </button>
        </div>
      </header>
      {showSettings && <SettingsDrawer onClose={() => setShowSettings(false)} onChanged={() => setSettingsVersion((v) => v + 1)} />}
      <AssetGate>
      <main className="content">
        <header className="page-head">
          <h2>{TAB[tab].title}</h2>
          <p>{TAB[tab].desc}</p>
        </header>
        {/* the studio stays mounted so a running job keeps streaming while other tabs are open */}
        <div style={{ display: tab === "studio" ? "block" : "none" }}><Guard><Studio open={open} settingsVersion={settingsVersion} openSettings={() => setShowSettings(true)} /></Guard></div>
        <Loading state={runsState}>
          {(list) => (
            <Guard resetKey={tab}>
              {tab === "progress" && <Progress open={open} />}
              {tab === "episodes" && <Episodes runs={list} sel={sel} open={open} />}
              {tab === "investigator" && <Investigator runs={list} open={open} />}
              {tab === "experiments" && <Experiments key={sel.run} runs={list} open={open} sel={sel} />}
              {tab === "skills" && <Skills open={open} />}
            </Guard>
          )}
        </Loading>
      </main>
      </AssetGate>
    </div>
  );
}
