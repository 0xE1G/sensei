import { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
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

