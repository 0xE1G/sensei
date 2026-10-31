import { useRef, useState } from "react";
import { getRuns } from "./api.js";
import { Guard, Loading, useAsync } from "./ui.jsx";
import Studio from "./panels/Studio.jsx";
import Progress from "./panels/Progress.jsx";
import Episodes from "./panels/Episodes.jsx";
import Investigator from "./panels/Investigator.jsx";
import Experiments from "./panels/Experiments.jsx";
import Skills from "./panels/Skills.jsx";

// Small inline icons (24px grid, stroke = currentColor); no icon library.
const ICONS = {
  studio: <path d="M5 4l14 8-14 8z" />,
  episodes: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M7 5v14M17 5v14M3 12h4M17 12h4" /></>,
  progress: <path d="M4 19V5M4 19h16M8 15l4-5 3 3 5-6" />,
  investigator: <><circle cx="11" cy="11" r="6" /><path d="M20 20l-4.5-4.5" /></>,
  experiments: <path d="M9 3h6M10 3v6l-5 9a2 2 0 002 3h10a2 2 0 002-3l-5-9V3M7.5 15h9" />,
  skills: <path d="M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5" />,
};

const SECTIONS = [
  ["Work", [["studio", "Studio", "Tell the robot what to do, watch it run, and let it fix what fails."],
            ["episodes", "Episodes", "Replay any saved episode in 3D with its plan and every skill call."]]],
  ["Insight", [["progress", "Progress", "How the system improved: repairs, transfer, diagnosis, measured in simulation."],
               ["investigator", "Investigator", "Why a step failed: evidence, possible causes, and the one-change test of each."],
               ["experiments", "Experiments", "Every search and repair, with the candidates tried and the gates they faced."]]],
  ["Library", [["skills", "Skills", "What the robot knows, where each skill works, and how the library grew."]]],
];
const TAB = Object.fromEntries(SECTIONS.flatMap(([, items]) => items.map(([k, title, desc]) => [k, { title, desc }])));

export default function App() {
  const [tab, setTab] = useState("studio");
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
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">金</span>
          <div><h1>Kintsugi</h1><p>self-repairing robot skills</p></div>
        </div>
        <nav className="side-nav">
          {SECTIONS.map(([section, items]) => (
            <div key={section} className="nav-section">
              <div className="nav-label">{section}</div>
              {items.map(([k, label]) => (
                <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
                  <svg viewBox="0 0 24 24" aria-hidden="true">{ICONS[k]}</svg>{label}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="side-foot" title="Every statement in the app carries one of these labels">
          <div className="nav-label">Labels</div>
          <span><i className="dot-label dot-measured" />measured in simulation</span>
          <span><i className="dot-label dot-hypothesis" />model's hypothesis</span>
          <span><i className="dot-label dot-confirmed" />confirmed by a test</span>
          <span><i className="dot-label dot-rejected" />rejected by a test</span>
        </div>
      </aside>
      <main className="content">
        <header className="page-head">
          <h2>{TAB[tab].title}</h2>
          <p>{TAB[tab].desc}</p>
        </header>
        {/* the studio stays mounted so a running job keeps streaming while other tabs are open */}
        <div style={{ display: tab === "studio" ? "block" : "none" }}><Guard><Studio open={open} /></Guard></div>
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
    </div>
  );
}
