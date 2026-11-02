"""Build the self-contained Three.js 3D flight viewer (an HTML string) for one flight.

The simulation runs in Python; this page only replays the recorded trajectory in the
visitor's browser. Three.js is loaded from the jsDelivr CDN; nothing else is fetched.
"""

from __future__ import annotations

import json

import numpy as np

from sim.world import World

THEMES = {
    "forest": {
        "skyTop": "#1d2b52",
        "skyHorizon": "#f2a26b",
        "fog": "#d99a72",
        "ground": "#45552f",
        "grid": "#ffcf99",
        "sun": "#ffd7a3",
        "sunPos": [-40, 22, -30],
        "hemiSky": "#ffd2a6",
        "hemiGround": "#2a3324",
        "trunk": "#4a3222",
        "foliage": "#2f5a36",
        "foliage2": "#3d6b43",
        "block": "#6b5a4a",
        "wall": "#7a6a58",
        "exposure": 1.0,
    },
    "city": {
        "skyTop": "#141b33",
        "skyHorizon": "#c76b5a",
        "fog": "#7d5a63",
        "ground": "#45484f",
        "grid": "#9fb3c8",
        "sun": "#ffc4a8",
        "sunPos": [35, 18, 25],
        "hemiSky": "#e7b7a8",
        "hemiGround": "#1d2026",
        "trunk": "#8d939c",
        "foliage": "#8d939c",
        "foliage2": "#8d939c",
        "block": "#3b4250",
        "wall": "#596273",
        "exposure": 1.05,
    },
    "warehouse": {
        "skyTop": "#05070c",
        "skyHorizon": "#1b2333",
        "fog": "#111723",
        "ground": "#33373e",
        "grid": "#4b5d78",
        "sun": "#cfe1ff",
        "sunPos": [10, 30, 10],
        "hemiSky": "#9fb8de",
        "hemiGround": "#15181d",
        "trunk": "#c9a227",
        "foliage": "#c9a227",
        "foliage2": "#c9a227",
        "block": "#4a5263",
        "wall": "#6b7385",
        "exposure": 1.25,
    },
}


def _level_sectors(ray_dirs: list[list[float]]) -> list[int]:
    """Indices of the sectors close to horizontal (these see obstacles, not the ground)."""
    return [k for k, d in enumerate(ray_dirs) if abs(d[2]) < np.sin(np.deg2rad(12.0))]


def _nearest(rays: list[float], level: list[int]) -> float | None:
    vals = [rays[k] for k in level if rays[k] >= 0]
    return min(vals) if vals else None


def _status(
    frames: list[dict], n_wp: int, final: np.ndarray, outcome: str, level: list[int]
) -> list[str]:
    out = []
    for i, f in enumerate(frames):
        near = _nearest(f["rays"], level)
        near = 99.0 if near is None else near
        d_final = float(np.linalg.norm(np.asarray(f["pos"]) - final))
        if i == len(frames) - 1 and outcome != "success":
            out.append(outcome.upper())
        elif f["wp"] == n_wp - 1 and d_final < 0.6 and f["speed"] < 0.6:
            out.append("HOVERING")
        elif near < 1.2:
            out.append("AVOIDING")
        else:
            out.append("CRUISING")
    return out


def build_payload(
    result: dict, world: World, theme: str, show_rays: bool, frame_stride: int = 2
) -> dict:
    frames = result["frames"][::frame_stride]
    if result["frames"] and frames[-1] is not result["frames"][-1]:
        frames.append(result["frames"][-1])
    if result["success"]:
        outcome = "success"
    elif result["collision"]:
        outcome = "collision"
    elif result["out_of_bounds"]:
        outcome = "lost control"
    else:
        outcome = "timeout"
    n_wp = int(world.n_waypoints)
    final = world.waypoints[n_wp - 1]
    level = _level_sectors(result["ray_dirs"])
    nearest = []
    for f in frames:
        near = _nearest(f["rays"], level)
        nearest.append(None if near is None else round(near, 2))
    return {
        "theme": THEMES.get(theme, THEMES["forest"]),
        "world": {
            "cyl": world.cyl[world.cyl_mask].round(3).tolist(),
            "box": world.box[world.box_mask].round(3).tolist(),
            "waypoints": world.waypoints[:n_wp].round(3).tolist(),
            "start": world.start.round(3).tolist(),
        },
        "frames": {
            "t": [f["t"] for f in frames],
            "pos": [f["pos"] for f in frames],
            "quat": [f["quat"] for f in frames],
            "speed": [f["speed"] for f in frames],
            "motors": [f["motors"] for f in frames],
            "wp": [f["wp"] for f in frames],
            "rays": [f["rays"] for f in frames],
            "nearest": nearest,
            "status": _status(frames, n_wp, final, outcome, level),
        },
        "rayDirs": result["ray_dirs"],
        "showRays": bool(show_rays),
        "outcome": outcome,
        "hoverThrust": 1.844,
    }


def viewer_html(payload: dict, height: int = 580) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data).replace("__HEIGHT__", str(height))


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html,body{margin:0;background:#0d1117;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#e6edf3}
  #wrap{position:relative;width:100%;height:__HEIGHT__px;border-radius:14px;overflow:hidden;border:1px solid #222b36}
  canvas{display:block}
  .panel{position:absolute;background:rgba(9,12,17,.72);border:1px solid rgba(255,255,255,.08);border-radius:10px}
  #hud{left:14px;top:14px;padding:10px 12px;font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;min-width:180px}
  #hud .st{display:flex;gap:8px;align-items:center;font-weight:600;margin-bottom:6px}
  #hud .dot{width:8px;height:8px;border-radius:50%}
  #hud .g{display:grid;grid-template-columns:auto auto;gap:3px 14px}
  #hud .k{color:#9aa7b4}
  #cams{right:14px;top:14px;display:flex;gap:4px;padding:4px}
  button{font:inherit;font-size:13px;min-height:34px;padding:0 11px;border:none;border-radius:7px;background:transparent;color:#e6edf3;cursor:pointer}
  button.on{background:#f5b041;color:#1a1205;font-weight:600}
  #bar{left:14px;right:14px;bottom:14px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:7px 10px}
  #bar .tm{font:12.5px ui-monospace,Menlo,monospace;color:#c9d3dd}
  #scrub{flex:1 1 200px;min-width:120px;accent-color:#38bdf8}
  .sp{border:1px solid #2c3744;border-radius:6px;font:12px ui-monospace,Menlo,monospace;min-height:30px}
  .sp.on{border-color:#38bdf8;background:#0f2a3a;color:#e6edf3;font-weight:400}
  #play{background:#1c2530;width:38px;padding:0}
  #raysBtn{border:1px solid #2c3744}
  #banner{left:50%;top:50%;transform:translate(-50%,-50%);padding:14px 22px;font-weight:700;font-size:18px;display:none}
  #err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#9aa7b4;font-size:14px;text-align:center;padding:20px}
</style></head>
<body>
<div id="wrap">
  <div id="hud" class="panel">
    <div class="st"><span class="dot" id="dot"></span><span id="status">CRUISING</span></div>
    <div class="g">
      <span class="k">SPEED</span><span id="hSpeed">0.0 m/s</span>
      <span class="k">ALT</span><span id="hAlt">0.0 m</span>
      <span class="k">NEAREST</span><span id="hNear">–</span>
      <span class="k">WAYPT</span><span id="hWp">1 / 1</span>
    </div>
  </div>
  <div id="cams" class="panel" role="group" aria-label="Camera">
    <button data-cam="chase" class="on">Chase</button><button data-cam="orbit">Orbit</button>
    <button data-cam="top">Top</button><button data-cam="fpv">FPV</button>
  </div>
  <div id="bar" class="panel">
    <button id="play" aria-label="Play or pause">❚❚</button>
    <span class="tm" id="tNow">00:00.0</span>
    <label for="scrub" style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">Playback position</label>
    <input id="scrub" type="range" min="0" max="1000" value="0">
    <span class="tm" id="tEnd">00:00.0</span>
    <div role="group" aria-label="Playback speed" style="display:flex;gap:4px">
      <button class="sp" data-sp="0.5">0.5×</button><button class="sp on" data-sp="1">1×</button><button class="sp" data-sp="2">2×</button>
    </div>
    <button id="raysBtn">Rays</button>
  </div>
  <div id="banner" class="panel"></div>
  <div id="err">The 3D viewer needs WebGL and access to cdn.jsdelivr.net.</div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const D = JSON.parse(document.getElementById('data').textContent);
const T = D.theme, F = D.frames, N = F.t.length;
// simulator frame (x fwd, y left, z up) -> three.js frame (y up)
const V = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
const Q = (q) => new THREE.Quaternion(q[0], q[2], -q[1], q[3]);

const wrap = document.getElementById('wrap');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) { document.getElementById('err').style.display = 'flex'; throw e; }
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = T.exposure;
wrap.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(T.fog, 22, 110);
const camera = new THREE.PerspectiveCamera(60, 1, 0.03, 600);

// ---------- sky dome ----------
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: { top: { value: new THREE.Color(T.skyTop) }, horizon: { value: new THREE.Color(T.skyHorizon) } },
  vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }',
  fragmentShader: 'uniform vec3 top; uniform vec3 horizon; varying vec3 vP; void main(){ float h = clamp(vP.y*1.6+0.05,0.0,1.0); gl_FragColor = vec4(mix(horizon, top, pow(h,0.7)),1.0); }'
});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(400, 32, 16), skyMat));

// ---------- lights ----------
scene.add(new THREE.HemisphereLight(T.hemiSky, T.hemiGround, 1.25));
const sun = new THREE.DirectionalLight(T.sun, 2.4);
sun.position.set(...T.sunPos);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -22, right: 22, top: 22, bottom: -22, near: 1, far: 120 });
scene.add(sun);
const sunGlow = new THREE.Mesh(new THREE.SphereGeometry(6, 24, 12), new THREE.MeshBasicMaterial({ color: T.sun, fog: false }));
sunGlow.position.set(...T.sunPos).normalize().multiplyScalar(320);
scene.add(sunGlow);

// ---------- ground ----------
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: T.ground, roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(30, 30, T.grid, T.grid);
grid.material.transparent = true; grid.material.opacity = 0.08; grid.position.y = 0.01;
scene.add(grid);

// ---------- obstacles ----------
const mat = (c, r = 0.9) => new THREE.MeshStandardMaterial({ color: c, roughness: r });
const trunkM = mat(T.trunk), folM = mat(T.foliage), fol2M = mat(T.foliage2), blockM = mat(T.block, 0.8);
for (const [x, y, r, h] of D.world.cyl) {
  const g = new THREE.Group();
  g.position.copy(V([x, y, 0]));
  if (T.trunk !== T.foliage) {           // forest: slim conifer that never exceeds radius r
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.35, r * 0.45, h * 0.45, 10), trunkM);
    trunk.position.y = h * 0.225; g.add(trunk);
    for (let k = 0; k < 3; k++) {
      const cone = new THREE.Mesh(new THREE.ConeGeometry(r * (1.0 - 0.18 * k), h * 0.34, 12), k % 2 ? fol2M : folM);
      cone.position.y = h * (0.42 + 0.19 * k); g.add(cone);
    }
  } else {                                // city / warehouse: pillars
    const p = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 20), trunkM);
    p.position.y = h / 2; g.add(p);
  }
  g.traverse((m) => { m.castShadow = true; m.receiveShadow = true; });
  scene.add(g);
}
for (const [x0, y0, z0, x1, y1, z1] of D.world.box) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0, z1 - z0, y1 - y0), blockM);
  m.position.copy(V([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]));
  m.castShadow = true; m.receiveShadow = true;
  scene.add(m);
}

// ---------- waypoints ----------
const wps = D.world.waypoints, rings = [];
let prev = D.world.start;
wps.forEach((w, i) => {
  const isFinal = i === wps.length - 1;
  const ringMat = new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: '#ffffff', emissiveIntensity: 0.6, transparent: true, opacity: 0.5 });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(isFinal ? 0.45 : 0.6, 0.05, 12, 56), ringMat);
  ring.position.copy(V(w));
  const dir = V(w).sub(V(prev));
  if (!isFinal && dir.length() > 1e-3) ring.lookAt(ring.position.clone().add(dir));
  if (isFinal) ring.rotation.x = Math.PI / 2;
  scene.add(ring); rings.push(ring);
  prev = w;
});
const final = wps[wps.length - 1];
const pad = new THREE.Mesh(new THREE.RingGeometry(0.55, 0.75, 48), new THREE.MeshBasicMaterial({ color: '#f5b041', transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
pad.rotation.x = -Math.PI / 2; pad.position.copy(V([final[0], final[1], 0.02]));
scene.add(pad);
const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, final[2], 6), new THREE.MeshBasicMaterial({ color: '#f5b041', transparent: true, opacity: 0.35 }));
pole.position.copy(V([final[0], final[1], final[2] / 2]));
scene.add(pole);

// ---------- drone (Kingfisher geometry, drawn 2.5x enlarged so it is visible) ----------
const DRONE_SCALE = 2.5;
const drone = new THREE.Group();
const dark = mat('#20262f', 0.5), carbon = mat('#11161d', 0.6);
const body = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.045, 0.09), dark);
drone.add(body);
const cam = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.025, 0.09), carbon);
cam.position.set(0.09, 0, 0); drone.add(cam);
const rotorXY = [[0.075, -0.10], [-0.075, 0.10], [-0.075, -0.10], [0.075, 0.10]];
const props = [];
for (const [x, y] of rotorXY) {
  const p = V([x, y, 0]);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(x, y) * 2, 0.012, 0.018), carbon);
  arm.position.copy(p.clone().multiplyScalar(0.5));
  arm.rotation.y = Math.atan2(-p.z, p.x);
  drone.add(arm);
  const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.022, 12), dark);
  motor.position.copy(p).add(new THREE.Vector3(0, 0.015, 0)); drone.add(motor);
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.002, 32), new THREE.MeshBasicMaterial({ color: '#cfe8ff', transparent: true, opacity: 0.18 }));
  disc.position.copy(p).add(new THREE.Vector3(0, 0.028, 0)); drone.add(disc);
  const blade = new THREE.Group();
  for (let k = 0; k < 3; k++) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.003, 0.012), mat('#e6edf3', 0.4));
    b.position.x = 0.03; const h = new THREE.Group(); h.rotation.y = (k * 2 * Math.PI) / 3; h.add(b); blade.add(h);
  }
  blade.position.copy(disc.position); drone.add(blade); props.push(blade);
}
const led = (c, x) => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.008, 8, 8), new THREE.MeshBasicMaterial({ color: c })); m.position.set(x, 0.02, 0); drone.add(m); };
led('#4ade80', 0.08); led('#ef4444', -0.08);
drone.traverse((m) => { m.castShadow = true; });
drone.scale.setScalar(DRONE_SCALE);
scene.add(drone);

// ---------- trail, coloured by speed ----------
const trailGeo = new THREE.BufferGeometry();
const tp = new Float32Array(N * 3), tc = new Float32Array(N * 3);
const slow = new THREE.Color('#38bdf8'), fast = new THREE.Color('#f5b041'), tmp = new THREE.Color();
for (let i = 0; i < N; i++) {
  const v = V(F.pos[i]); tp.set([v.x, v.y, v.z], i * 3);
  tmp.copy(slow).lerp(fast, Math.min(F.speed[i] / 6, 1)); tc.set([tmp.r, tmp.g, tmp.b], i * 3);
}
trailGeo.setAttribute('position', new THREE.BufferAttribute(tp, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(tc, 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
scene.add(trail);

// ---------- sensor rays ----------
const nR = D.rayDirs.length;
const rayGeo = new THREE.BufferGeometry();
const rp = new Float32Array(nR * 6), rc = new Float32Array(nR * 6);
rayGeo.setAttribute('position', new THREE.BufferAttribute(rp, 3));
rayGeo.setAttribute('color', new THREE.BufferAttribute(rc, 3));
const rays = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.3 }));
let showRays = D.showRays;
scene.add(rays);
const dirsLocal = D.rayDirs.map((d) => V(d));
// sector distances are horizontal ranges: scale each direction so its horizontal part is 1
const horiz = dirsLocal.map((v) => Math.max(Math.hypot(v.x, v.z), 1e-6));
const yawQ = new THREE.Quaternion(), upY = new THREE.Vector3(0, 1, 0);
const cRay = new THREE.Color('#38bdf8'), cNear = new THREE.Color('#ef4444');

// ---------- controls & cameras ----------
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.enabled = false;
let camMode = 'chase';
const chaseFwd = new THREE.Vector3(1, 0, 0);
camera.position.copy(V(F.pos[0])).add(new THREE.Vector3(-1.7, 0.65, 0));

document.querySelectorAll('#cams button').forEach((b) => b.onclick = () => {
  camMode = b.dataset.cam;
  document.querySelectorAll('#cams button').forEach((x) => x.classList.toggle('on', x === b));
  controls.enabled = camMode === 'orbit';
  if (camMode === 'orbit') { controls.target.copy(drone.position); camera.position.copy(drone.position).add(new THREE.Vector3(-6, 5, 6)); }
});

// ---------- playback ----------
const dur = F.t[N - 1] - F.t[0];
let tNow = 0, playing = true, speed = 1;
const fmt = (s) => { const m = Math.floor(s / 60); return String(m).padStart(2, '0') + ':' + (s - 60 * m).toFixed(1).padStart(4, '0'); };
document.getElementById('tEnd').textContent = fmt(dur);
const playBtn = document.getElementById('play');
playBtn.onclick = () => { if (!playing && tNow >= dur) tNow = 0; playing = !playing; playBtn.textContent = playing ? '❚❚' : '▶'; };
const scrub = document.getElementById('scrub');
scrub.oninput = () => { tNow = (scrub.value / 1000) * dur; };
document.querySelectorAll('.sp').forEach((b) => b.onclick = () => {
  speed = parseFloat(b.dataset.sp);
  document.querySelectorAll('.sp').forEach((x) => x.classList.toggle('on', x === b));
});
const raysBtn = document.getElementById('raysBtn');
const syncRaysBtn = () => { raysBtn.style.background = showRays ? '#0f2a3a' : 'transparent'; raysBtn.style.borderColor = showRays ? '#38bdf8' : '#2c3744'; };
raysBtn.onclick = () => { showRays = !showRays; syncRaysBtn(); };
syncRaysBtn();

const statusColors = { CRUISING: '#38bdf8', AVOIDING: '#f5b041', HOVERING: '#4ade80', COLLISION: '#ef4444', TIMEOUT: '#9aa7b4', 'LOST CONTROL': '#ef4444' };
const el = (id) => document.getElementById(id);
const banner = el('banner');

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap); resize();

const clock = new THREE.Clock();
const qA = new THREE.Quaternion(), qB = new THREE.Quaternion();
const fpvOffset = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2);
let propAngle = [0, 0, 0, 0];

function frameAt(t) {
  const x = Math.max(0, Math.min(t + F.t[0], F.t[N - 1]));
  let lo = 0, hi = N - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (F.t[m] <= x) lo = m; else hi = m; }
  const span = F.t[hi] - F.t[lo];
  return [lo, hi, span > 0 ? (x - F.t[lo]) / span : 0];
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  if (playing) { tNow += dt * speed; if (tNow >= dur) { tNow = dur; playing = false; playBtn.textContent = '▶'; } }
  scrub.value = String(Math.round((tNow / Math.max(dur, 1e-6)) * 1000));
  el('tNow').textContent = fmt(tNow);

  const [i, j, a] = frameAt(tNow);
  drone.position.copy(V(F.pos[i])).lerp(V(F.pos[j]), a);
  qA.copy(Q(F.quat[i])); qB.copy(Q(F.quat[j]));
  drone.quaternion.copy(qA.slerp(qB, a));

  // propellers: visual spin speed grows with motor thrust
  props.forEach((p, k) => {
    const f = F.motors[i][k];
    propAngle[k] += dt * 45 * Math.sqrt(Math.max(f, 0) / D.hoverThrust) * (k < 2 ? 1 : -1);
    p.rotation.y = propAngle[k];
  });

  trailGeo.setDrawRange(0, i + 1);

  // waypoint states
  const wpi = F.wp[i];
  rings.forEach((r, k) => {
    const m = r.material;
    const done = k < wpi || (k === wps.length - 1 && F.status[i] === 'HOVERING' && i === N - 1);
    if (done) { m.color.set('#4ade80'); m.emissive.set('#4ade80'); m.opacity = 0.9; r.scale.setScalar(1); }
    else if (k === wpi) { m.color.set('#f5b041'); m.emissive.set('#f5b041'); m.opacity = 1; r.scale.setScalar(1 + 0.08 * Math.sin(performance.now() / 180)); }
    else { m.color.set('#ffffff'); m.emissive.set('#ffffff'); m.opacity = 0.45; r.scale.setScalar(1); }
  });

  // rays
  rays.visible = showRays;
  if (showRays) {
    const o = drone.position;
    // the scan is level: it turns with the heading only, not with roll and pitch
    const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.quaternion);
    yawQ.setFromAxisAngle(upY, Math.atan2(-nose.z, nose.x));
    for (let k = 0; k < nR; k++) {
      const d = F.rays[i][k];
      const end = d < 0 ? o : dirsLocal[k].clone().applyQuaternion(yawQ).multiplyScalar(d / horiz[k]).add(o);
      rp.set([o.x, o.y, o.z, end.x, end.y, end.z], k * 6);
      const c = d >= 0 && d < 1.0 ? cNear : cRay;
      rc.set([c.r, c.g, c.b, c.r, c.g, c.b], k * 6);
    }
    rayGeo.attributes.position.needsUpdate = true; rayGeo.attributes.color.needsUpdate = true;
  }

  // HUD
  const st = F.status[i];
  el('status').textContent = st;
  el('dot').style.background = statusColors[st] || '#38bdf8';
  el('hSpeed').textContent = F.speed[i].toFixed(1) + ' m/s';
  el('hAlt').textContent = F.pos[i][2].toFixed(1) + ' m';
  el('hNear').textContent = F.nearest[i] == null ? '–' : F.nearest[i].toFixed(1) + ' m';
  el('hNear').style.color = F.nearest[i] != null && F.nearest[i] < 1.0 ? '#fca5a5' : '#e6edf3';
  el('hWp').textContent = Math.min(wpi + 1, wps.length) + ' / ' + wps.length;
  if (tNow >= dur) {
    banner.style.display = 'block';
    banner.textContent = D.outcome === 'success' ? 'Route complete · hover held' : 'Flight ended: ' + D.outcome;
    banner.style.color = D.outcome === 'success' ? '#4ade80' : '#fca5a5';
  } else banner.style.display = 'none';

  // cameras
  const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.quaternion); fwd.y = 0;
  if (fwd.lengthSq() > 1e-6) chaseFwd.lerp(fwd.normalize(), 0.06).normalize();
  if (camMode === 'chase') {
    const want = drone.position.clone().addScaledVector(chaseFwd, -1.7).add(new THREE.Vector3(0, 0.65, 0));
    camera.position.lerp(want, 0.08);
    camera.lookAt(drone.position.clone().addScaledVector(chaseFwd, 1.2));
  } else if (camMode === 'top') {
    camera.position.lerp(drone.position.clone().add(new THREE.Vector3(0.01, 16, 0)), 0.1);
    camera.lookAt(drone.position);
  } else if (camMode === 'fpv') {
    camera.position.copy(drone.position).add(new THREE.Vector3(0.1, 0.02, 0).applyQuaternion(drone.quaternion));
    camera.quaternion.copy(drone.quaternion).multiply(fpvOffset);
  } else {
    controls.target.lerp(drone.position, 0.1); controls.update();
  }
  renderer.render(scene, camera);
}
animate();
</script>
</body></html>
"""