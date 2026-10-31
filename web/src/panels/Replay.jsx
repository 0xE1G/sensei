import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { getMeshes, getReplay, getRun, getTrial } from "../api.js";
import { Card, Chip, Loading, RunPicker, useAsync } from "../ui.jsx";
import { Trace } from "./TaskPanel.jsx";

// Compiled Panda meshes (server decodes the MuJoCo model, so vertex offsets match the geoms).
let meshGeoms = null;
export async function loadMeshGeometries() {
  if (meshGeoms) return meshGeoms;
  const raw = await getMeshes();
  const out = {};
  for (const [name, m] of Object.entries(raw)) {
    const v = new Float32Array(Uint8Array.from(atob(m.v), (c) => c.charCodeAt(0)).buffer);
    const f = new Uint32Array(Uint8Array.from(atob(m.f), (c) => c.charCodeAt(0)).buffer);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(v, 3));
    g.setIndex(new THREE.BufferAttribute(f, 1));
    g.computeVertexNormals();
    out[name] = g;
  }
  meshGeoms = out;
  return out;
}

const mjQuat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // MuJoCo is w-first

function geomGeometry(g, meshes) {
  const s = g.size;
  switch (g.type) {
    case "box": return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "sphere": return new THREE.SphereGeometry(s[0], 24, 16);
    case "ellipsoid": return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case "cylinder": return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 28).rotateX(Math.PI / 2);
    case "capsule": return new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 16).rotateX(Math.PI / 2);
    case "plane": return new THREE.PlaneGeometry(2 * (s[0] || 1.5), 2 * (s[1] || 1.5));
    case "mesh": return meshes[g.mesh] || null;
    default: return null;
  }
}

// scene: static geoms per body (recorder.export_geoms); pose: one frame {p: body positions, q: body quaternions}.
function SceneObjects({ scene, meshes, pose }) {
  const groups = useRef([]);
  const built = useMemo(() => {
    return scene.bodies.map((body, b) =>
      scene.geoms
        .filter((g) => g.body === body)
        .map((g, i) => {
          const geo = geomGeometry(g, meshes);
          if (!geo) return null;
          const plane = g.type === "plane";
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
            transparent: g.rgba[3] < 1, opacity: g.rgba[3], roughness: plane ? 0.95 : 0.55, metalness: 0.05,
          });
          return { key: `${b}-${i}`, geo, mat, pos: g.pos, quat: mjQuat(g.quat), plane };
        })
        .filter(Boolean)
    );
  }, [scene, meshes]);

  useEffect(() => {
    if (!pose) return;
    groups.current.forEach((grp, b) => {
      if (!grp || !pose.p[b]) return;
      grp.position.set(...pose.p[b]);
      grp.quaternion.copy(mjQuat(pose.q[b]));
    });
  }, [pose, built]);

  return built.map((items, b) => (
    <group key={b} ref={(el) => (groups.current[b] = el)}>
      {items.map((it) => (
        <mesh key={it.key} geometry={it.geo} material={it.mat} position={it.pos} quaternion={it.quat}
              receiveShadow={it.plane} castShadow={!it.plane} />
      ))}
    </group>
  ));
}

// zoom=false: the mouse wheel scrolls the page instead of zooming (rotate and pan still work)
export function Viewer({ scene, meshes, pose, height = 520, zoom = true, children }) {
  return (
    <div className="viewer" style={{ height }}>
      <Canvas shadows gl={{ preserveDrawingBuffer: true }}
              camera={{ position: [1.75, -1.2, 1.2], up: [0, 0, 1], fov: 38, near: 0.01, far: 20 }}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[1.2, -0.8, 2.2]} intensity={1.2} castShadow shadow-mapSize={[2048, 2048]} />
        <directionalLight position={[-1, 1, 1.5]} intensity={0.35} />
        {scene && meshes && <SceneObjects scene={scene} meshes={meshes} pose={pose} />}
        <OrbitControls target={[0.4, 0, 0.2]} enableZoom={zoom} makeDefault />
      </Canvas>
      {children}
    </div>
  );
}

export default function Replay({ runs, sel, open }) {
  const run = useAsync(() => (sel.run ? getRun(sel.run) : Promise.resolve(null)), [sel.run]);
  const name = sel.trial;
  const data = useAsync(async () => {
    if (!sel.run || !name) return null;
    const [replay, meshes] = await Promise.all([getReplay(sel.run, name), loadMeshGeometries()]);
    const trial = await getTrial(sel.run, name).catch(() => null);
    return { replay, meshes, trial };
  }, [sel.run, name]);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  useEffect(() => { setFrame(0); setPlaying(true); }, [sel.run, name]);
  useEffect(() => {
    if (!playing || !data.data) return;
    const n = data.data.replay.frames.length;
    const fps = data.data.replay.fps || 30;
    let last = performance.now(), acc = 0, raf;
    const tick = (now) => {
      acc += ((now - last) / 1000) * fps * speed;
      last = now;
      if (acc >= 1) {
        const step = Math.floor(acc);
        acc -= step;
        setFrame((f) => (f + step >= n ? (setPlaying(false), n - 1) : f + step));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, data.data]);

  const runsWithReplays = runs.filter((r) => r.n_replays > 0);
  return (
    <div className="grid">
      <div className="toolbar">
        <RunPicker runs={runsWithReplays} kinds={["task"]} value={sel.run}
                   onChange={(id) => open({ run: id, trial: null })} label="Run with saved replays" />
        <Loading state={run}>
          {(r) => r && (
            <label className="picker">
              <span>Episode</span>
              <select value={name || ""} onChange={(e) => open({ trial: e.target.value })}>
                <option value="" disabled>choose ({r.replays.length})</option>
                {r.replays.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </label>
          )}
        </Loading>
      </div>
      <Loading state={data}>
        {(d) => {
          if (!d) return <p className="muted">Choose a run and an episode with a saved replay.</p>;
          const frames = d.replay.frames;
          const T = frames[frames.length - 1]?.t || 1;
          const t = frames[frame]?.t || 0;
          const subtasks = d.trial?.subtasks || [];
          const active = subtasks.flatMap((st) => (st.trace || []).map((s) => ({ ...s, obj: st.obj })))
            .find((s) => t >= s.t0 && t <= s.t1);
          return (
            <div className="split" style={{ gridTemplateColumns: "1fr minmax(280px, 360px)" }}>
              <div>
                <Viewer scene={d.replay.scene} meshes={d.meshes} pose={frames[frame]} />
                <div className="timeline">
                  <button className="btn" onClick={() => {
                    if (frame >= frames.length - 1) setFrame(0);
                    setPlaying((p) => !p);
                  }}>{playing ? "Pause" : "Play"}</button>
                  <input type="range" min={0} max={frames.length - 1} value={frame}
                         onChange={(e) => { setPlaying(false); setFrame(+e.target.value); }} />
                  <span className="pill">{t.toFixed(1)} / {T.toFixed(1)} s</span>
                  <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>
                    {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
                  </select>
                </div>
                <div className="ticks">
                  {(d.replay.events || []).filter((e) => e.kind === "skill_end").map((e, i) => (
                    <span key={i} className={`tick ${e.ok ? "" : "fail"}`} title={`${e.skill} ${e.ok ? "ok" : "failed"} (${e.obj})`}
                          style={{ left: `${(100 * e.t) / T}%` }} />
                  ))}
                </div>
                <p className="sub">
                  Simulator poses replayed at 30 fps; no physics runs in the browser. Marks: end of each skill (red = its
                  check failed).
                </p>
              </div>
              <Card title="Skill trace" status="measured">
                <p className="sub">now: {active ? `${active.skill} on ${active.obj}` : "between skills"}</p>
                {d.trial ? <Trace subtasks={subtasks} /> : <p className="muted">No trial record.</p>}
                {d.trial && (
                  <p style={{ marginTop: 10 }}>
                    <Chip status={d.trial.success ? "confirmed" : "rejected"}>{d.trial.success ? "task succeeded" : "task failed"}</Chip>{" "}
                    <span className="sub">checked by code against the goals</span>
                  </p>
                )}
              </Card>
            </div>
          );
        }}
      </Loading>
    </div>
  );
}
