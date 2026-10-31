import { useState } from "react";
import { getRuns } from "./api.js";
import { Loading, useAsync } from "./ui.jsx";
import Overview from "./panels/Overview.jsx";
import Studio from "./panels/Studio.jsx";
import TaskPanel from "./panels/TaskPanel.jsx";
import Replay from "./panels/Replay.jsx";
import Investigator from "./panels/Investigator.jsx";
import Experiments from "./panels/Experiments.jsx";
import Skills from "./panels/Skills.jsx";

const TABS = [
  ["studio", "Studio"],
  ["overview", "Results"],
  ["task", "Task & plan"],
  ["replay", "Replay"],
  ["investigator", "Investigator"],
  ["experiments", "Experiments"],
  ["skills", "Skills"],
];

export default function App() {
  const [tab, setTab] = useState("studio");
  const [sel, setSel] = useState({ run: "demo/1_bowl_before_repair", trial: "00001" });
  const runs = useAsync(getRuns, [tab]);   // re-read on tab change: studio jobs add runs
  const open = (next) => {
    setSel((s) => ({ ...s, ...next }));
    if (next.tab) setTab(next.tab);
  };
  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo">金</span>
          <div>
            <h1>Kintsugi</h1>
            <p>builds and verifies its own skill library · every statement labelled</p>
          </div>
        </div>
        <nav>
          {TABS.map(([k, label]) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {/* the studio stays mounted so a running job keeps streaming while other tabs are open */}
        <div style={{ display: tab === "studio" ? "block" : "none" }}><Studio open={open} /></div>
        <Loading state={runs}>
          {(list) => (
            <>
              {tab === "overview" && <Overview runs={list} open={open} />}
              {tab === "task" && <TaskPanel runs={list} sel={sel} open={open} />}
              {tab === "replay" && <Replay runs={list} sel={sel} open={open} />}
              {tab === "investigator" && <Investigator runs={list} />}
              {tab === "experiments" && <Experiments key={sel.run} runs={list} open={open} sel={sel} />}
              {tab === "skills" && <Skills />}
            </>
          )}
        </Loading>
      </main>
    </div>
  );
}
