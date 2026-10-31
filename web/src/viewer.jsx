import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { getMeshes } from "./api.js";



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

// Robot cameras as small translucent tiles inside the main view (bottom left): a wrist camera that rides on the hand
// and looks along the fingers, and a fixed front camera. Each is rendered into an offscreen target, then drawn as a
// rounded, slightly transparent quad over the main view.
const TILE_MAX = 200, TILE_FRAC = 0.22, TILE_PAD = 12, TILE_GAP = 8, TILE_OPACITY = 0.9;
const tileSize = (width) => {
  const w = Math.round(Math.min(TILE_MAX, (width - TILE_PAD) * TILE_FRAC));
  return [w, Math.round(w * 0.75)];
};
const _v = new THREE.Vector3(), _f = new THREE.Vector3(), _u = new THREE.Vector3(), _q = new THREE.Quaternion();

function roundedAlpha() {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 192;
  const g = c.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, 256, 192);
  g.fillStyle = "#fff"; g.beginPath(); g.roundRect(0, 0, 256, 192, 18); g.fill();
  return new THREE.CanvasTexture(c);
}

function RobotCameras({ scene: desc, pose, show }) {
  const { gl, scene, camera, size } = useThree();
  const wrist = useMemo(() => new THREE.PerspectiveCamera(75, 4 / 3, 0.005, 10), []);
  const front = useMemo(() => {
    const c = new THREE.PerspectiveCamera(46, 4 / 3, 0.01, 20);
    c.position.set(1.12, 0, 0.5);
    c.up.set(0, 0, 1);
    c.lookAt(0.45, 0, 0.06);
    return c;
  }, []);
  // offscreen targets + one quad scene per tile
  const tiles = useMemo(() => {
    const alpha = roundedAlpha();
    const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    return [0, 1].map(() => {
      const rt = new THREE.WebGLRenderTarget(4, 3, { samples: 4 });
      rt.texture.colorSpace = THREE.SRGBColorSpace;
      const mat = new THREE.MeshBasicMaterial({ map: rt.texture, alphaMap: alpha, transparent: true, opacity: TILE_OPACITY,
                                                toneMapped: false, depthTest: false });
      const quad = new THREE.Scene();
      quad.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
      return { rt, quad, ortho };
    });
  }, []);
  useEffect(() => () => tiles.forEach((t) => t.rt.dispose()), [tiles]);
  const hand = desc ? desc.bodies.indexOf("hand") : -1;

  useFrame(() => {
    gl.setRenderTarget(null);
    gl.autoClear = true;
    gl.render(scene, camera);
    if (!show || !desc) return;
    const cams = [];
    if (pose && hand >= 0) {
      _q.set(pose.q[hand][1], pose.q[hand][2], pose.q[hand][3], pose.q[hand][0]);
      _f.set(0, 0, 1).applyQuaternion(_q);              // the hand's z axis points along the fingers
      _u.set(1, 0, 0).applyQuaternion(_q);
      wrist.position.set(...pose.p[hand]).addScaledVector(_f, 0.035).addScaledVector(_u, -0.055);
      wrist.up.copy(_u).negate();
      wrist.lookAt(_v.copy(wrist.position).addScaledVector(_f, 1).addScaledVector(_u, 0.25));
      cams.push(wrist);
    }
    cams.push(front);
    const [w, h] = tileSize(size.width);
    const dpr = gl.getPixelRatio();
    cams.forEach((cam, i) => {
      const t = tiles[i];
      if (t.rt.width !== Math.round(w * dpr)) t.rt.setSize(Math.round(w * dpr), Math.round(h * dpr));
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
      gl.setRenderTarget(t.rt);
      gl.setClearColor(0x1c1c1f, 1);
      gl.clear();
      gl.render(scene, cam);
    });
    gl.setRenderTarget(null);
    gl.setClearColor(0x000000, 0);
    gl.autoClear = false;
    gl.setScissorTest(true);
    cams.forEach((_, i) => {
      const x = TILE_PAD + i * (w + TILE_GAP), y = TILE_PAD;
      gl.setViewport(x, y, w, h);
      gl.setScissor(x, y, w, h);
      gl.render(tiles[i].quad, tiles[i].ortho);
    });
    gl.setScissorTest(false);
    gl.setViewport(0, 0, size.width, size.height);
    gl.autoClear = true;
  }, 1);
  return null;
}

// zoom=false: the mouse wheel scrolls the page instead of zooming (rotate and pan still work).
// cameras: show the robot camera tiles (toggle in the corner).
export function Viewer({ scene, meshes, pose, height = 520, zoom = true, cameras = true, children }) {
  const [showCams, setShowCams] = useState(cameras);
  const box = useRef(null);
  const [width, setWidth] = useState(800);
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const hasHand = !!scene && scene.bodies.includes("hand");
  const [tw, th] = tileSize(width);
  const labels = [...(hasHand ? ["Wrist"] : []), "Front"];
  return (
    <div className="viewer" ref={box} style={{ height }}>
      <Canvas shadows gl={{ preserveDrawingBuffer: true }}
              camera={{ position: [1.75, -1.2, 1.2], up: [0, 0, 1], fov: 38, near: 0.01, far: 20 }}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[1.2, -0.8, 2.2]} intensity={1.2} castShadow shadow-mapSize={[2048, 2048]} />
        <directionalLight position={[-1, 1, 1.5]} intensity={0.35} />
        {scene && meshes && <SceneObjects scene={scene} meshes={meshes} pose={pose} />}
        <OrbitControls target={[0.4, 0, 0.2]} enableZoom={zoom} makeDefault />
        <RobotCameras scene={scene} pose={pose} show={showCams && !!scene} />
      </Canvas>
      {showCams && scene && labels.map((l, i) => (
        <span key={l} className="cam-tile" style={{ left: TILE_PAD + i * (tw + TILE_GAP), bottom: TILE_PAD, width: tw, height: th }}>
          <b>{l}</b>
        </span>
      ))}
      {children}
      {scene && (
        <button type="button" className={`cam-toggle ${showCams ? "on" : ""}`} onClick={() => setShowCams((s) => !s)}
                title={showCams ? "Hide robot cameras" : "Show robot cameras"} aria-label="robot cameras">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h3l2-2h6l2 2h3v12H4z" /><circle cx="12" cy="13" r="3.5" /></svg>
        </button>
      )}
    </div>
  );
}
