import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// Config & Palette
const API_BASE_URL = (window.location.port === '5173' || !window.location.origin || window.location.origin === 'null' || window.location.origin.startsWith('file:')) ? 'http://127.0.0.1:8000' : window.location.origin;
const AGENT_COLORS = [0x0284c7, 0x10b981, 0xf59e0b, 0xef4444, 0x8b5cf6];
const AGENT_COLOR_STRS = ['#0284c7', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

// State
let replayData = null;
let scene, camera, renderer, controls, transformControls;
let clock = new THREE.Clock();

// Interactive City Layout Editing State
let isEditLayoutMode = false;
const editableModelGroups = [];
let selectedModelObject = null;
const btnEditLayout = document.getElementById('btn-edit-layout');

// Default Locked Base City Layout (4-Quadrant Tiling: NW [25,25], NE [75,25], SW [25,75], SE [75,75])
const DEFAULT_LOCKED_CITY_LAYOUT = [
  { "id": "statue", "pos": [50.0, 0.0, 50.0], "rot": [0, 0, 0, "XYZ"], "scale": [3.5990544254536285, 3.5990544254536285, 3.5990544254536285] },
  { "id": "tile_nw", "pos": [25.0, 0.0, 25.0], "rot": [0, 0, 0, "XYZ"], "scale": [3.898772613290013, 3.898772613290013, 3.898772613290013] },
  { "id": "tile_ne", "pos": [75.0, 0.0, 25.0], "rot": [3.141592653589793, -0.017884550951215317, 3.141592653589793, "XYZ"], "scale": [2.7477067827630943, 2.7477067827630943, 2.7477067827630943] },
  { "id": "tile_sw", "pos": [25.0, 0.0, 75.0], "rot": [0, 0, 0, "XYZ"], "scale": [2.0092326469098416, 2.0092326469098416, 2.0092326469098416] },
  { "id": "tile_se", "pos": [75.0, 0.0, 75.0], "rot": [3.141592653589793, -1.5556825245831971, 3.141592653589793, "XYZ"], "scale": [2.1407560107685226, 2.1407560107685226, 2.1407560107685226] }
];

// Environment & Models
let cityMeshA = null;
let cityMeshB = null;
let hongkongMesh = null;
let seoulMesh = null;
let brutalistMesh = null;
let statueMesh = null;
let residentialMesh = null;
let ruinedMesh = null;
let envGroup = null;
let droneTemplate = null;
let personTemplate = null;

const droneInstances = [];
const personInstances = [];

// Replay vs Live Mode State
let isLiveMode = false;
let liveIntervalId = null;
let isLiveRunning = false;

let isPlayingReplay = false;
let playbackSpeed = 1.0;
let totalDurationSeconds = 60.0;
let currentTime = 0.0;
let currentStepIndex = 0;

let liveSummaryData = {
  open_rescued: 0, open_total: 6,
  hidden_rescued: 0, hidden_total: 4,
  total_rescued: 0, total_survivors: 10,
  collisions: 0, total_steps: 0, coverage_pct: 0.0
};

// Follow Drone & Camera Transition State
let followedDroneId = null;
let isFollowMode = false;
let savedFreeCamPos = new THREE.Vector3();
let savedFreeCamTarget = new THREE.Vector3();
let isCameraTransitioning = false;
let transitionProgress = 1.0;
let transitionStartCamPos = new THREE.Vector3();
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const _targetCamVec = new THREE.Vector3();
const _targetLookVec = new THREE.Vector3();
const _tmpHeightColor = new THREE.Color();
const MAP_TYPE_COLORS = {
  0: new THREE.Color(0x34d399),
  1: new THREE.Color(0x38bdf8),
  2: new THREE.Color(0xf59e0b),
  3: new THREE.Color(0xec4899),
  4: new THREE.Color(0xa855f7)
};

// UI Elements
const loadingScreen = document.getElementById('loading-screen');
const loaderBar = document.getElementById('loader-bar');
const loaderStatus = document.getElementById('loader-status');
const assetWarnings = document.getElementById('asset-warnings');

const hudStep = document.getElementById('hud-step');
const hudTime = document.getElementById('hud-time');
const modeBadge = document.getElementById('mode-badge');
const statSurvivors = document.getElementById('stat-survivors');
const statCoverage = document.getElementById('stat-coverage');
const statCollisions = document.getElementById('stat-collisions');
const statAgents = document.getElementById('stat-agents');
const statGrid = document.getElementById('stat-grid');
const statScale = document.getElementById('stat-scale');
const statBuildings = document.getElementById('stat-buildings');
const agentList = document.getElementById('agent-list');

const btnRandomizeHumans = document.getElementById('btn-randomize-humans');
const btnStartLive = document.getElementById('btn-start-live');
const selectKillDrone = document.getElementById('select-kill-drone');
const btnKillDrone = document.getElementById('btn-kill-drone');
const btnPlay = document.getElementById('btn-play');
const btnReset = document.getElementById('btn-reset');
const btnReplay = document.getElementById('btn-replay');
const progressBar = document.getElementById('progress-bar');
const progressSlider = document.getElementById('progress-slider');
const speedBtns = document.querySelectorAll('.speed-btn');

const followHudCard = document.getElementById('follow-hud-card');
const followDot = document.getElementById('follow-dot');
const followTitle = document.getElementById('follow-title');
const followStatus = document.getElementById('follow-status');
const followPos = document.getElementById('follow-pos');
const btnExitFollow = document.getElementById('btn-exit-follow');

const statsModal = document.getElementById('stats-modal');
const finalSurvivors = document.getElementById('final-survivors');
const finalOpenSurvivors = document.getElementById('final-open-survivors');
const finalHiddenSurvivors = document.getElementById('final-hidden-survivors');
const finalCoverage = document.getElementById('final-coverage');
const finalCollisions = document.getElementById('final-collisions');
const finalSteps = document.getElementById('final-steps');
const cardFailedRecovery = document.getElementById('card-failed-recovery');
const finalFailedRecovery = document.getElementById('final-failed-recovery');
const toastContainer = document.getElementById('event-toast-container');

const triggeredEvents = new Set();
const failedDrones = new Set();
const failedZoneMeshes = [];

// Helper for safe DOM text updates
function setTxt(el, val) {
  if (el) el.innerText = val;
}
function setHtml(el, val) {
  if (el) el.innerHTML = val;
}

// ==============================================================================
// INITIALIZATION (LOAD EXACT GLTF CITY DIRECTLY ON STARTUP)
// ==============================================================================
async function init() {
  try {
    setupThreeScene();
    setupEventListeners();

    replayData = {
      metadata: { grid_size: 10, cell_size_meters: 10.0, num_agents: 5 },
      initial_state: {
        agent_positions: { agent_0: [0, 0], agent_1: [2, 2], agent_2: [1, 3], agent_3: [4, 0], agent_4: [5, 2] },
        agent_altitudes: { agent_0: 0, agent_1: 0, agent_2: 0, agent_3: 0, agent_4: 0 },
        survivor_positions: [[2, 3], [4, 5], [7, 2], [8, 8], [1, 7], [6, 4], [3, 8], [8, 3], [5, 6], [2, 8]],
        hidden_survivors: [[8, 8], [6, 4], [8, 3], [2, 8]],
        open_survivors: [[2, 3], [4, 5], [7, 2], [1, 7], [3, 8], [5, 6]],
        obstacle_positions: []
      }
    };

    // Load exact GLTF 3D city models FIRST before rendering so no rough boxes ever appear
    await loadAssets();

    setupEnvironmentComposition();
    setupSurvivors();
    setupDrones();
    setupUI();
    animate();
  } catch (err) {
    console.error('Initialization error caught safely:', err);
    setupEnvironmentComposition();
    setupSurvivors();
    setupDrones();
    setupUI();
    animate();
  }

  // Connect to Live API or fallback log asynchronously in background
  (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/start`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        replayData = data;
        setupSurvivors();
        setupDrones();
      }
    } catch (e) {
      try {
        const resLog = await fetch('replay_log.json');
        if (resLog.ok) {
          replayData = await resLog.json();
          setupSurvivors();
          setupDrones();
        }
      } catch (e2) {}
    }
  })();
}

function updateLoader(pct, text) {
  if (loaderBar) loaderBar.style.width = `${pct}%`;
  setTxt(loaderStatus, text);
}

// ==============================================================================
// THREE.JS SCENE SETUP & REALISTIC LIGHTING (WHITE BACKGROUND)
// ==============================================================================
function setupThreeScene() {
  const canvas = document.getElementById('scene-canvas');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf8fafc);
  scene.fog = new THREE.FogExp2(0xf8fafc, 0.002);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.02;

  transformControls = new TransformControls(camera, renderer.domElement);
  transformControls.size = 0.85;
  transformControls.addEventListener('dragging-changed', (event) => {
    controls.enabled = !event.value;
  });
  scene.add(transformControls);

  const hemiLight = new THREE.HemisphereLight(0xffffff, 0xe2e8f0, 0.8);
  hemiLight.position.set(0, 200, 0);
  scene.add(hemiLight);

  const sunLight = new THREE.DirectionalLight(0xfffbeb, 1.4);
  sunLight.position.set(80, 150, 60);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 1024;
  sunLight.shadow.mapSize.height = 1024;
  const d = 120;
  sunLight.shadow.camera.left = -d; sunLight.shadow.camera.right = d;
  sunLight.shadow.camera.top = d; sunLight.shadow.camera.bottom = -d;
  scene.add(sunLight);

  const fillLight = new THREE.DirectionalLight(0xbae6fd, 0.5);
  fillLight.position.set(-80, 100, -60);
  scene.add(fillLight);

  window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// ==============================================================================
// FAST PARALLEL ASSET LOADING WITH TIMEOUT FALLBACKS
// ==============================================================================
function loadGLTFWithTimeout(loader, url, timeoutMs = 90000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      console.warn(`Asset load timed out: ${url}`);
      reject(new Error(`Timeout ${url}`));
    }, timeoutMs);
    loader.load(url, (gltf) => {
      clearTimeout(timer);
      resolve(gltf);
    }, (progress) => {
      if (progress.total > 0) {
        const pct = Math.round((progress.loaded / progress.total) * 100);
        const el = document.getElementById('loader-status');
        if (el) el.innerText = `Loading 3D assets... (${pct}%)`;
      }
    }, (err) => {
      clearTimeout(timer);
      console.warn(`Failed to load asset: ${url}`, err);
      reject(err);
    });
  });
}

async function loadAssets() {
  const loader = new GLTFLoader();

  const [resHK, resSeoul, resRes, resRuin, resStatue, resDrone, resPerson] = await Promise.allSettled([
    loadGLTFWithTimeout(loader, 'assets/hongkong/scene.gltf', 25000),
    loadGLTFWithTimeout(loader, 'assets/seoul/scene.gltf', 25000),
    loadGLTFWithTimeout(loader, 'assets/residential/scene.gltf', 25000),
    loadGLTFWithTimeout(loader, 'assets/ruined/scene.gltf', 25000),
    loadGLTFWithTimeout(loader, 'assets/statue/scene.gltf', 25000),
    loadGLTFWithTimeout(loader, 'models/drone_design/scene.gltf', 10000),
    loadGLTFWithTimeout(loader, 'assets/person/scene.gltf', 10000),
  ]);

  if (resHK.status === 'fulfilled') hongkongMesh = resHK.value.scene;
  if (resSeoul.status === 'fulfilled') seoulMesh = resSeoul.value.scene;
  if (resRes.status === 'fulfilled') residentialMesh = resRes.value.scene;
  if (resRuin.status === 'fulfilled') ruinedMesh = resRuin.value.scene;
  if (resStatue.status === 'fulfilled') statueMesh = resStatue.value.scene;

  if (resDrone.status === 'fulfilled') {
    droneTemplate = resDrone.value.scene;
  } else {
    droneTemplate = createDroneFallback();
  }

  if (resPerson.status === 'fulfilled') {
    personTemplate = resPerson.value.scene;
    personTemplate.updateMatrixWorld(true);
    let boxP = new THREE.Box3().setFromObject(personTemplate);
    let sizeP = boxP.getSize(new THREE.Vector3());
    const maxDimP = Math.max(sizeP.x, sizeP.y, sizeP.z);
    if (maxDimP > 0) {
      const scaleP = 2.4 / maxDimP;
      personTemplate.scale.set(scaleP, scaleP, scaleP);
    }
    personTemplate.updateMatrixWorld(true);
    boxP = new THREE.Box3().setFromObject(personTemplate);
    personTemplate.position.y = -boxP.min.y;
  } else {
    personTemplate = createPersonFallback();
  }
}

function createDroneFallback() {
  const group = new THREE.Group();

  // Central Sleek Fuselage Body
  const bodyGeo = new THREE.BoxGeometry(1.4, 0.45, 1.4);
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.9, roughness: 0.2 });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  group.add(body);

  // Top Status LED Dome
  const domeGeo = new THREE.SphereGeometry(0.35, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2);
  const domeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.9, roughness: 0.1 });
  const dome = new THREE.Mesh(domeGeo, domeMat);
  dome.position.y = 0.22;
  group.add(dome);

  // Front Camera Sensor & Gimbal
  const gimbalGeo = new THREE.SphereGeometry(0.32, 16, 16);
  const gimbalMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8, roughness: 0.3 });
  const gimbal = new THREE.Mesh(gimbalGeo, gimbalMat);
  gimbal.position.set(0, -0.2, 0.7);
  group.add(gimbal);

  const lensGeo = new THREE.CylinderGeometry(0.16, 0.16, 0.1, 16);
  const lensMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee });
  const lens = new THREE.Mesh(lensGeo, lensMat);
  lens.rotation.x = Math.PI / 2;
  lens.position.set(0, -0.2, 0.86);
  group.add(lens);

  // 4 Diagonal Rotor Arms + Motors + Rotor Blades
  const armAngles = [Math.PI / 4, 3 * Math.PI / 4, 5 * Math.PI / 4, 7 * Math.PI / 4];
  const armLength = 1.8;

  armAngles.forEach((angle, idx) => {
    // Carbon Arm
    const armGeo = new THREE.CylinderGeometry(0.1, 0.1, armLength, 8);
    const armMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.4 });
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.rotation.z = Math.PI / 2;
    arm.rotation.y = angle;
    arm.position.set(Math.cos(angle) * (armLength / 2), 0, Math.sin(angle) * (armLength / 2));
    group.add(arm);

    // Motor Hub
    const motorX = Math.cos(angle) * armLength;
    const motorZ = Math.sin(angle) * armLength;
    const motorGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.3, 16);
    const motorMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.9, roughness: 0.2 });
    const motor = new THREE.Mesh(motorGeo, motorMat);
    motor.position.set(motorX, 0.1, motorZ);
    group.add(motor);

    // Propeller Blade Guard Ring
    const guardGeo = new THREE.TorusGeometry(0.7, 0.04, 8, 32);
    guardGeo.rotateX(Math.PI / 2);
    const guardMat = new THREE.MeshStandardMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.75 });
    const guard = new THREE.Mesh(guardGeo, guardMat);
    guard.position.set(motorX, 0.2, motorZ);
    group.add(guard);

    // Dual Blade Propeller
    const propGeo = new THREE.BoxGeometry(1.2, 0.02, 0.12);
    const propMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.5 });
    const prop = new THREE.Mesh(propGeo, propMat);
    prop.position.set(motorX, 0.22, motorZ);
    prop.rotation.y = idx * (Math.PI / 4);
    group.add(prop);
  });

  // Landing Skids
  [-0.6, 0.6].forEach(x => {
    const skidGeo = new THREE.CylinderGeometry(0.06, 0.06, 1.8, 8);
    const skidMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8 });
    const skid = new THREE.Mesh(skidGeo, skidMat);
    skid.rotation.x = Math.PI / 2;
    skid.position.set(x, -0.4, 0);
    group.add(skid);
  });

  return group;
}

function createPersonFallback() {
  const group = new THREE.Group();
  // Body torso
  const torsoGeo = new THREE.CylinderGeometry(0.35, 0.3, 1.1, 12);
  const matVest = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.3 });
  const torso = new THREE.Mesh(torsoGeo, matVest);
  torso.position.y = 0.85;
  group.add(torso);

  // Head
  const headGeo = new THREE.SphereGeometry(0.25, 12, 12);
  const matHead = new THREE.MeshStandardMaterial({ color: 0xfed7aa, roughness: 0.5 });
  const head = new THREE.Mesh(headGeo, matHead);
  head.position.y = 1.6;
  group.add(head);

  return group;
}

// ==============================================================================
// ENVIRONMENT COMPOSITION: STRICT NON-OVERLAPPING PUZZLE-PIECE 10x10 CITY
// ==============================================================================
// ==============================================================================
// ENVIRONMENT COMPOSITION: STRICT NON-OVERLAPPING PUZZLE-PIECE 10x10 CITY
// ==============================================================================
function placeTileModel(model, minX, maxX, minZ, maxZ, modelId = 'tile') {
  if (!model) return null;
  const clone = model.clone(true);
  clone.name = modelId;
  clone.updateMatrixWorld(true);

  // Traverse materials to ensure high visibility & proper lighting
  clone.traverse(child => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      if (child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach(m => {
          m.side = THREE.DoubleSide;
          if (m.roughness !== undefined) m.roughness = 0.6;
          if (m.metalness !== undefined) m.metalness = 0.1;
        });
      }
    }
  });

  let box = new THREE.Box3().setFromObject(clone);
  let size = box.getSize(new THREE.Vector3());

  if (size.x > 0 && size.z > 0) {
    const tileW = maxX - minX;
    const tileD = maxZ - minZ;

    const scaleX = tileW / size.x;
    const scaleZ = tileD / size.z;
    const scale = Math.min(scaleX, scaleZ) * 0.95;
    clone.scale.set(scale, scale, scale);
    clone.updateMatrixWorld(true);

    box = new THREE.Box3().setFromObject(clone);
    const center = box.getCenter(new THREE.Vector3());

    const targetX = (minX + maxX) / 2;
    const targetZ = (minZ + maxZ) / 2;

    clone.position.x = targetX - center.x;
    clone.position.z = targetZ - center.z;
    clone.position.y = -box.min.y;
  }
  envGroup.add(clone);
  editableModelGroups.push({ id: modelId, object: clone });
  return clone;
}

function toggleEditLayoutMode() {
  isEditLayoutMode = !isEditLayoutMode;
  const editToolbar = document.getElementById('edit-layout-toolbar');

  if (isEditLayoutMode) {
    if (btnEditLayout) {
      btnEditLayout.innerHTML = '<span class="btn-icon">🔒</span> Lock & Save City';
      btnEditLayout.style.background = 'linear-gradient(135deg, #10b981, #059669)';
      btnEditLayout.style.borderColor = '#10b981';
    }
    if (editToolbar) editToolbar.classList.remove('hidden');
    showToast('🛠️ EDIT MODE ACTIVE: Click any building/statue to select & drag handles! Use Size slider or S key to scale.', 'status');
  } else {
    if (transformControls) transformControls.detach();
    selectedModelObject = null;

    if (btnEditLayout) {
      btnEditLayout.innerHTML = '<span class="btn-icon">🛠️</span> Edit City Layout';
      btnEditLayout.style.background = 'linear-gradient(135deg, #8b5cf6, #6366f1)';
      btnEditLayout.style.borderColor = '#a855f7';
    }
    if (editToolbar) editToolbar.classList.add('hidden');

    // Save custom transforms to localStorage
    const savedLayout = editableModelGroups.map(m => ({
      id: m.id,
      pos: m.object.position.toArray(),
      rot: m.object.rotation.toArray(),
      scale: m.object.scale.toArray()
    }));
    try {
      localStorage.setItem('vihang_custom_city_layout', JSON.stringify(savedLayout));
      showToast('🔒 Custom City Layout Locked & Saved! Persists automatically.', 'status');
    } catch (e) {
      console.warn('Could not save custom layout:', e);
    }
  }
}

// ==============================================================================
// HIGH-FIDELITY PROCEDURAL TEXTURE & ARCHITECTURE GENERATOR (0MS INSTANT LOAD)
// ==============================================================================
function createSkyscraperTexture(baseColorHex = '#1e293b', glassColorHex = '#38bdf8', illuminatedRatio = 0.45) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = baseColorHex;
  ctx.fillRect(0, 0, 512, 512);

  // Vertical structural columns
  ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
  for (let x = 0; x < 512; x += 64) {
    ctx.fillRect(x, 0, 8, 512);
  }

  // Window grid with illuminated interior offices
  const rows = 16;
  const cols = 8;
  const w = 46;
  const h = 20;

  for (let r = 0; r < rows; r++) {
    const y = r * 32 + 6;
    for (let c = 0; c < cols; c++) {
      const x = c * 64 + 10;
      const isLit = Math.random() < illuminatedRatio;
      if (isLit) {
        const warm = Math.random() < 0.65;
        ctx.fillStyle = warm ? 'rgba(253, 224, 71, 0.95)' : 'rgba(56, 189, 248, 0.9)';
      } else {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
      }
      ctx.fillRect(x, y, w, h);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, w, h);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function createResidentialFacadeTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#475569';
  ctx.fillRect(0, 0, 512, 512);

  for (let r = 0; r < 8; r++) {
    const y = r * 64;
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, y, 512, 10);

    for (let c = 0; c < 6; c++) {
      const x = c * 85 + 10;
      ctx.fillStyle = Math.random() < 0.5 ? 'rgba(254, 240, 138, 0.9)' : 'rgba(30, 41, 59, 0.95)';
      ctx.fillRect(x + 5, y + 16, 55, 34);

      ctx.fillStyle = '#334155';
      ctx.fillRect(x, y + 42, 65, 14);
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y + 42, 65, 14);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function createRubbleDisasterTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#475569';
  ctx.fillRect(0, 0, 512, 512);

  ctx.fillStyle = '#f59e0b';
  for (let i = -512; i < 512; i += 40) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i + 20, 0);
    ctx.lineTo(i + 20 + 512, 512);
    ctx.lineTo(i + 512, 512);
    ctx.closePath();
    ctx.fill();
  }

  ctx.fillStyle = 'rgba(30, 41, 59, 0.85)';
  ctx.fillRect(0, 0, 512, 512);

  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2;
  for (let i = 20; i < 512; i += 55) {
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i + (Math.random() - 0.5) * 20);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function createHelipadMarkingTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, 512, 512);

  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 24;
  ctx.beginPath();
  ctx.arc(256, 256, 210, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 12;
  ctx.beginPath();
  ctx.arc(256, 256, 170, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 200px Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('H', 256, 256);

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

function createAsphaltRoadTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  // Asphalt base
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, 1024, 1024);

  // Outer border road & cross grid lines
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 8;
  for (let i = 0; i <= 1024; i += 102.4) {
    ctx.beginPath();
    ctx.moveTo(i, 0); ctx.lineTo(i, 1024);
    ctx.moveTo(0, i); ctx.lineTo(1024, i);
    ctx.stroke();
  }

  // Yellow dashed lane dividers
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 3;
  ctx.setLineDash([14, 14]);
  ctx.beginPath();
  ctx.moveTo(512, 0); ctx.lineTo(512, 1024);
  ctx.moveTo(0, 512); ctx.lineTo(1024, 512);
  ctx.stroke();
  ctx.setLineDash([]);

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

// Pre-create shared procedural textures
const texSkyscraperTech = createSkyscraperTexture('#0f172a', '#38bdf8', 0.6);
const texSkyscraperCommercial = createSkyscraperTexture('#1e293b', '#0284c7', 0.45);
const texResidential = createResidentialFacadeTexture();
const texRubble = createRubbleDisasterTexture();
const texHelipad = createHelipadMarkingTexture();
const texAsphalt = createAsphaltRoadTexture();

function createProceduralTile(type, minX, maxX, minZ, maxZ, modelId = 'tile') {
  const group = new THREE.Group();
  group.name = modelId;

  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;

  if (type === 'nw') {
    // Tech District: 4 High-Rise Skyscraper Towers with illuminated office windows & glass crowns
    const configs = [
      { dx: -11, dz: -11, w: 13, d: 13, h: 28, em: 0x0284c7 },
      { dx: 11, dz: -9, w: 14, d: 11, h: 22, em: 0x06b6d4 },
      { dx: -9, dz: 11, w: 11, d: 14, h: 24, em: 0x38bdf8 },
      { dx: 11, dz: 11, w: 13, d: 13, h: 18, em: 0x0ea5e9 },
    ];
    configs.forEach(c => {
      const geo = new THREE.BoxGeometry(c.w, c.h, c.d);
      const mat = new THREE.MeshStandardMaterial({
        map: texSkyscraperTech,
        metalness: 0.35,
        roughness: 0.3,
        side: THREE.DoubleSide
      });
      const b = new THREE.Mesh(geo, mat);
      b.position.set(cx + c.dx, c.h / 2, cz + c.dz);
      b.castShadow = true; b.receiveShadow = true;
      group.add(b);

      // Rooftop HVAC unit
      const hvacGeo = new THREE.BoxGeometry(c.w * 0.5, 1.8, c.d * 0.5);
      const hvacMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.4 });
      const hvac = new THREE.Mesh(hvacGeo, hvacMat);
      hvac.position.set(cx + c.dx, c.h + 0.9, cz + c.dz);
      group.add(hvac);

      // Antenna mast with blinking warning light
      const antGeo = new THREE.CylinderGeometry(0.18, 0.3, 7, 8);
      const antMat = new THREE.MeshBasicMaterial({ color: c.em });
      const ant = new THREE.Mesh(antGeo, antMat);
      ant.position.set(cx + c.dx, c.h + 4.5, cz + c.dz);
      group.add(ant);
    });
  } else if (type === 'ne') {
    // Commercial District: Stepped Glass Towers & Heli-deck
    const configs = [
      { dx: 0, dz: 0, w: 18, d: 18, h: 24 },
      { dx: -12, dz: 12, w: 11, d: 11, h: 18 },
      { dx: 12, dz: -12, w: 13, d: 11, h: 16 },
    ];
    configs.forEach(c => {
      const geo = new THREE.BoxGeometry(c.w, c.h, c.d);
      const mat = new THREE.MeshStandardMaterial({
        map: texSkyscraperCommercial,
        metalness: 0.4,
        roughness: 0.3,
        side: THREE.DoubleSide
      });
      const b = new THREE.Mesh(geo, mat);
      b.position.set(cx + c.dx, c.h / 2, cz + c.dz);
      b.castShadow = true; b.receiveShadow = true;
      group.add(b);
    });

    // Detailed Rooftop Helipad
    const heliGeo = new THREE.CylinderGeometry(6.5, 6.5, 0.5, 32);
    const heliMat = new THREE.MeshStandardMaterial({ map: texHelipad, roughness: 0.4, metalness: 0.2 });
    const heli = new THREE.Mesh(heliGeo, heliMat);
    heli.position.set(cx, 24.25, cz);
    heli.receiveShadow = true;
    group.add(heli);
  } else if (type === 'sw') {
    // Residential Blocks: 5-Storey Complex with Balconies & Facades
    const offsets = [
      { dx: -11, dz: -10, w: 16, d: 9, h: 15 },
      { dx: 11, dz: -10, w: 15, d: 9, h: 15 },
      { dx: -11, dz: 10, w: 16, d: 9, h: 13 },
      { dx: 11, dz: 10, w: 15, d: 9, h: 13 },
    ];
    offsets.forEach(c => {
      const geo = new THREE.BoxGeometry(c.w, c.h, c.d);
      const mat = new THREE.MeshStandardMaterial({
        map: texResidential,
        roughness: 0.5,
        metalness: 0.1,
        side: THREE.DoubleSide
      });
      const b = new THREE.Mesh(geo, mat);
      b.position.set(cx + c.dx, c.h / 2, cz + c.dz);
      b.castShadow = true; b.receiveShadow = true;
      group.add(b);

      // Rooftop water tank
      const tankGeo = new THREE.CylinderGeometry(1.8, 1.8, 3.2, 16);
      const tankMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.6, roughness: 0.3 });
      const tank = new THREE.Mesh(tankGeo, tankMat);
      tank.position.set(cx + c.dx - 3, c.h + 1.6, cz + c.dz - 1);
      group.add(tank);
    });
  } else if (type === 'se') {
    // Disaster / Ruined Zone: Collapsed Angular Pillars & Hazard Rubble Blocks
    const ruins = [
      { dx: -9, dz: -9, w: 13, d: 11, h: 15, rotZ: 0.14 },
      { dx: 11, dz: -6, w: 11, d: 13, h: 11, rotX: -0.16 },
      { dx: -6, dz: 11, w: 15, d: 9, h: 9, rotZ: -0.09 },
      { dx: 11, dz: 11, w: 9, d: 9, h: 17, rotX: 0.12 },
    ];
    ruins.forEach(r => {
      const geo = new THREE.BoxGeometry(r.w, r.h, r.d);
      const mat = new THREE.MeshStandardMaterial({
        map: texRubble,
        roughness: 0.7,
        metalness: 0.15,
        side: THREE.DoubleSide
      });
      const b = new THREE.Mesh(geo, mat);
      b.position.set(cx + r.dx, r.h / 2, cz + r.dz);
      if (r.rotX) b.rotation.x = r.rotX;
      if (r.rotZ) b.rotation.z = r.rotZ;
      b.castShadow = true; b.receiveShadow = true;
      group.add(b);

      // Red hazard alert beacon
      const hazardGeo = new THREE.SphereGeometry(0.6, 12, 12);
      const hazardMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
      const hazard = new THREE.Mesh(hazardGeo, hazardMat);
      hazard.position.set(cx + r.dx, r.h + 0.6, cz + r.dz);
      group.add(hazard);
    });
  }

  envGroup.add(group);
  editableModelGroups.push({ id: modelId, object: group });
  return group;
}

function setupEnvironmentComposition() {
  const gridSize = replayData?.metadata?.grid_size || 10;
  const cellSize = replayData?.metadata?.cell_size_meters || 10.0;
  const worldSize = gridSize * cellSize; // 100m x 100m

  envGroup = new THREE.Group();
  editableModelGroups.length = 0;

  // 1. Full 10x10 Area Asphalt Ground Base with Road Grid Textures
  const zoneGeo = new THREE.PlaneGeometry(worldSize, worldSize);
  const zoneMat = new THREE.MeshStandardMaterial({ map: texAsphalt, roughness: 0.75, metalness: 0.2 });
  const zonePlane = new THREE.Mesh(zoneGeo, zoneMat);
  zonePlane.rotation.x = -Math.PI / 2;
  zonePlane.position.set(worldSize / 2, 0.0, worldSize / 2);
  zonePlane.receiveShadow = true;
  envGroup.add(zonePlane);

  // 2. Center SciFi Plaza (Center of 10x10 Grid: [50, 0, 50])
  const plazaGeo = new THREE.CylinderGeometry(8.5, 9.0, 0.6, 32);
  const plazaMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8, roughness: 0.3 });
  const plaza = new THREE.Mesh(plazaGeo, plazaMat);
  plaza.position.set(worldSize / 2, 0.3, worldSize / 2);
  plaza.receiveShadow = true;
  envGroup.add(plaza);

  const plazaRingGeo = new THREE.RingGeometry(7.5, 8.3, 32);
  plazaRingGeo.rotateX(-Math.PI / 2);
  const plazaRingMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
  const plazaRing = new THREE.Mesh(plazaRingGeo, plazaRingMat);
  plazaRing.position.set(worldSize / 2, 0.62, worldSize / 2);
  envGroup.add(plazaRing);

  // Central Hologram Pillar
  const holPillarGeo = new THREE.CylinderGeometry(1.2, 1.8, 14, 16);
  const holPillarMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.8, roughness: 0.2 });
  const holPillar = new THREE.Mesh(holPillarGeo, holPillarMat);
  holPillar.position.set(worldSize / 2, 7.0, worldSize / 2);
  envGroup.add(holPillar);

  // Center Statue Spotlight
  const statueLight = new THREE.PointLight(0x06b6d4, 6.0, 45);
  statueLight.position.set(worldSize / 2, 12.0, worldSize / 2);
  envGroup.add(statueLight);

  if (statueMesh) {
    const sClone = statueMesh.clone(true);
    sClone.name = 'statue';
    sClone.updateMatrixWorld(true);
    let boxS = new THREE.Box3().setFromObject(sClone);
    let sizeS = boxS.getSize(new THREE.Vector3());
    const maxDimS = Math.max(sizeS.x, sizeS.y, sizeS.z);
    if (maxDimS > 0) {
      const scaleS = 22.0 / maxDimS;
      sClone.scale.set(scaleS, scaleS, scaleS);
      sClone.updateMatrixWorld(true);
    }
    boxS = new THREE.Box3().setFromObject(sClone);
    const centerS = boxS.getCenter(new THREE.Vector3());
    sClone.position.x = (worldSize / 2) - centerS.x;
    sClone.position.z = (worldSize / 2) - centerS.z;
    sClone.position.y = 0.6 - boxS.min.y;
    envGroup.add(sClone);
    editableModelGroups.push({ id: 'statue', object: sClone });
  } else {
    // Procedural SciFi Monument Fallback
    const monGeo = new THREE.CylinderGeometry(1.2, 2.5, 16, 8);
    const monMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.5, roughness: 0.3 });
    const mon = new THREE.Mesh(monGeo, monMat);
    mon.position.set(worldSize / 2, 8.6, worldSize / 2);
    mon.name = 'statue';
    mon.castShadow = true;
    envGroup.add(mon);
    editableModelGroups.push({ id: 'statue', object: mon });
  }

  // 3. NON-OVERLAPPING 4-QUADRANT JIGSAW PUZZLE TILING
  // Quadrant 1: North-West [0..50, 0..50] (HongKong / Tech District)
  if (hongkongMesh) placeTileModel(hongkongMesh, 0.0, 50.0, 0.0, 50.0, 'tile_nw');
  else createProceduralTile('nw', 0.0, 50.0, 0.0, 50.0, 'tile_nw');

  // Quadrant 2: North-East [50..100, 0..50] (Seoul / Commercial District)
  if (seoulMesh) placeTileModel(seoulMesh, 50.0, 100.0, 0.0, 50.0, 'tile_ne');
  else createProceduralTile('ne', 50.0, 100.0, 0.0, 50.0, 'tile_ne');

  // Quadrant 3: South-West [0..50, 50..100] (Residential Complex)
  if (residentialMesh) placeTileModel(residentialMesh, 0.0, 50.0, 50.0, 100.0, 'tile_sw');
  else createProceduralTile('sw', 0.0, 50.0, 50.0, 100.0, 'tile_sw');

  // Quadrant 4: South-East [50..100, 50..100] (Ruined Disaster District)
  if (ruinedMesh) placeTileModel(ruinedMesh, 50.0, 100.0, 50.0, 100.0, 'tile_se');
  else createProceduralTile('se', 50.0, 100.0, 50.0, 100.0, 'tile_se');



  envGroup.traverse(child => {
    if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; }
  });
  scene.add(envGroup);

  const gridHelper = new THREE.GridHelper(worldSize, gridSize, 0x0891b2, 0x475569);
  gridHelper.position.set(worldSize / 2, 0.05, worldSize / 2);
  scene.add(gridHelper);

  camera.position.set(worldSize * 0.5, worldSize * 1.1, worldSize * 1.55);
  controls.target.set(worldSize / 2, 0, worldSize / 2);
  controls.update();
}

// Pre-cached shared geometries for high-performance survivor rendering
const survivorRingGeo = new THREE.RingGeometry(0.8, 1.3, 24);
survivorRingGeo.rotateX(-Math.PI / 2);
const survivorBeamGeo = new THREE.CylinderGeometry(0.1, 0.5, 10, 12, 1, true);

// ==============================================================================
// SURVIVORS & RED GLOW SYSTEM (OPTIMIZED FOR ZERO-LAG RANDOMIZATION)
// ==============================================================================
function setupSurvivors(customLayout = null) {
  clearSurvivors();

  const gridSize = replayData.metadata?.grid_size || 10;
  const cellSize = replayData.metadata?.cell_size_meters || 10.0;

  const initSurv = customLayout ? customLayout.survivors : (replayData.initial_state?.survivor_positions || []);
  const hiddenSurv = new Set((customLayout ? customLayout.hidden_survivors : (replayData.initial_state?.hidden_survivors || [])).map(p => `${p[0]},${p[1]}`));

  // Map found step from log if present
  const foundStepMap = new Map();
  (replayData.steps || []).forEach(s => {
    (s.events || []).forEach(e => {
      if (e.type === 'survivor_found') {
        const k = `${e.survivor_position[0]},${e.survivor_position[1]}`;
        if (!foundStepMap.has(k)) foundStepMap.set(k, s.step);
      }
    });
  });

  initSurv.forEach(pos => {
    const template = personTemplate || createPersonFallback();
    const clone = template.clone(true);
    const [r, c] = pos;
    const worldX = (c + 0.5) * cellSize;
    const worldZ = (r + 0.5) * cellSize;

    clone.position.set(worldX, 0, worldZ);
    scene.add(clone);

    const isHidden = hiddenSurv.has(`${r},${c}`);

    // Red/Green Ring
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xef4444, side: THREE.DoubleSide, transparent: true, opacity: 0.85
    });
    const ring = new THREE.Mesh(survivorRingGeo, ringMat);
    ring.position.set(worldX, 0.1, worldZ);
    scene.add(ring);

    // Red/Green Beacon Beam
    const beamMat = new THREE.MeshBasicMaterial({
      color: 0xef4444, transparent: true, opacity: 0.45, side: THREE.DoubleSide
    });
    const beam = new THREE.Mesh(survivorBeamGeo, beamMat);
    beam.position.set(worldX, 5, worldZ);
    scene.add(beam);

    const light = new THREE.PointLight(0xef4444, 1.5, 12);
    light.position.set(worldX, 2, worldZ);
    scene.add(light);

    const foundStep = foundStepMap.has(`${r},${c}`) ? foundStepMap.get(`${r},${c}`) : Infinity;

    personInstances.push({
      mesh: clone, ring, beam, light, pos,
      isHidden, foundStep, revealed: false
    });
  });
}

function clearSurvivors() {
  personInstances.forEach(p => {
    if (p.mesh) {
      scene.remove(p.mesh);
      p.mesh.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
          else child.material.dispose();
        }
      });
    }
    if (p.ring) {
      scene.remove(p.ring);
      if (p.ring.material) p.ring.material.dispose();
    }
    if (p.beam) {
      scene.remove(p.beam);
      if (p.beam.material) p.beam.material.dispose();
    }
    if (p.light) {
      scene.remove(p.light);
      p.light.dispose();
    }
  });
  personInstances.length = 0;
}

function updateSurvivorGlowState(stepIdx, remainingSurvList = null) {
  personInstances.forEach(person => {
    let isFound = false;

    if (remainingSurvList !== null) {
      const isRemaining = remainingSurvList.some(s => s[0] === person.pos[0] && s[1] === person.pos[1]);
      isFound = !isRemaining;
    } else {
      isFound = stepIdx >= person.foundStep;
    }

    if (isFound) {
      person.ring.material.color.setHex(0x10b981);
      person.beam.material.color.setHex(0x10b981);
      person.light.color.setHex(0x10b981);
      person.light.intensity = 2.0;
      person.ring.material.opacity = 0.9;
      person.beam.material.opacity = 0.6;
    } else {
      person.ring.material.color.setHex(0xef4444);
      person.beam.material.color.setHex(0xef4444);
      person.light.color.setHex(0xef4444);
      person.light.intensity = 1.2;
      person.ring.material.opacity = 0.7;
      person.beam.material.opacity = 0.35;
    }
  });
}

// ==============================================================================
// DRONES SETUP
// ==============================================================================
function setupDrones() {
  droneInstances.forEach(d => {
    scene.remove(d.mesh); scene.remove(d.detCone);
    scene.remove(d.groundRing); scene.remove(d.flashSphere);
    if (d.groundShadow) scene.remove(d.groundShadow);
    if (d.verticalTether) scene.remove(d.verticalTether);
    if (d.avoidanceMesh) scene.remove(d.avoidanceMesh);
    if (d.targetGlowGroup) scene.remove(d.targetGlowGroup);
  });
  droneInstances.length = 0;

  const numAgents = replayData.metadata?.num_agents || 5;
  const cellSize = replayData.metadata?.cell_size_meters || 10.0;
  const detRadiusMeters = (replayData.metadata?.detection_radius_thermal || 1.8) * cellSize;

  for (let i = 0; i < numAgents; i++) {
    const agentName = `agent_${i}`;
    const colorHex = AGENT_COLORS[i % AGENT_COLORS.length];

    const template = droneTemplate || createDroneFallback();
    const clone = template.clone(true);
    const bBox = new THREE.Box3().setFromObject(clone);
    const sz = bBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(sz.x, sz.y, sz.z);
    if (maxDim > 0) clone.scale.setScalar(3.0 / maxDim);

    clone.traverse(child => {
      if (child.isMesh && child.material) {
        child.material = child.material.clone();
        const matName = (child.name || '').toLowerCase();
        // Accent lighting / guard rings get agent team color
        if (matName.includes('dome') || matName.includes('guard') || matName.includes('led') || matName.includes('light') || matName.includes('accent')) {
          child.material.color.setHex(colorHex);
          if (child.material.emissive) child.material.emissive.setHex(colorHex);
        }
      }
    });

    const initPos = replayData.initial_state?.agent_positions[agentName] || [0, 0];
    const initAlt = (replayData.initial_state?.agent_altitudes && replayData.initial_state.agent_altitudes[agentName] !== undefined) ? replayData.initial_state.agent_altitudes[agentName] : 0;
    const worldX = (initPos[1] + 0.5) * cellSize;
    const worldZ = (initPos[0] + 0.5) * cellSize;
    const altitude = 10.0 + initAlt * 4.0;

    clone.position.set(worldX, altitude, worldZ);
    scene.add(clone);

    const detGeo = new THREE.CylinderGeometry(detRadiusMeters, detRadiusMeters, altitude, 16, 1, true);
    const detMat = new THREE.MeshBasicMaterial({
      color: colorHex, transparent: true, opacity: 0.15, side: THREE.DoubleSide, depthWrite: false
    });
    const detCone = new THREE.Mesh(detGeo, detMat);
    detCone.position.set(worldX, altitude / 2, worldZ);
    scene.add(detCone);

    // Ground Ring (Detection footprint)
    const groundRingGeo = new THREE.RingGeometry(detRadiusMeters - 0.3, detRadiusMeters, 16);
    groundRingGeo.rotateX(-Math.PI / 2);
    const groundRingMat = new THREE.MeshBasicMaterial({
      color: colorHex, transparent: true, opacity: 0.45, side: THREE.DoubleSide
    });
    const groundRing = new THREE.Mesh(groundRingGeo, groundRingMat);
    groundRing.position.set(worldX, 0.15, worldZ);
    scene.add(groundRing);

    // Soft Circular Ground Shadow
    const shadowGeo = new THREE.RingGeometry(0.0, 1.8, 16);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false
    });
    const groundShadow = new THREE.Mesh(shadowGeo, shadowMat);
    groundShadow.position.set(worldX, 0.04, worldZ);
    scene.add(groundShadow);

    // Vertical Spatial Tether Line connecting drone to ground shadow
    const tetherGeo = new THREE.CylinderGeometry(0.12, 0.12, 1.0, 12, 1, true);
    const tetherMat = new THREE.MeshBasicMaterial({
      color: colorHex, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false
    });
    const verticalTether = new THREE.Mesh(tetherGeo, tetherMat);
    verticalTether.position.set(worldX, altitude / 2, worldZ);
    scene.add(verticalTether);

    // Avoidance Highlight Ring/Sphere
    const avGeo = new THREE.TorusGeometry(2.2, 0.2, 12, 20);
    avGeo.rotateX(Math.PI / 2);
    const avMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8, transparent: true, opacity: 0.0, side: THREE.DoubleSide
    });
    const avoidanceMesh = new THREE.Mesh(avGeo, avMat);
    avoidanceMesh.position.set(worldX, altitude, worldZ);
    scene.add(avoidanceMesh);

    const flashGeo = new THREE.SphereGeometry(3.0, 12, 12);
    const flashMat = new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.0 });
    const flashSphere = new THREE.Mesh(flashGeo, flashMat);
    flashSphere.position.set(worldX, altitude, worldZ);
    scene.add(flashSphere);

    // Self-Destruct Selection Target Glow Group
    const targetGlowGroup = new THREE.Group();

    const targetRingGeo = new THREE.RingGeometry(3.2, 4.0, 16);
    targetRingGeo.rotateX(-Math.PI / 2);
    const targetRingMat = new THREE.MeshBasicMaterial({
      color: 0xef4444, transparent: true, opacity: 0.85, side: THREE.DoubleSide
    });
    const targetRingMesh = new THREE.Mesh(targetRingGeo, targetRingMat);
    targetRingMesh.position.y = 0.18;
    targetGlowGroup.add(targetRingMesh);

    const targetInnerRingGeo = new THREE.RingGeometry(1.6, 2.2, 16);
    targetInnerRingGeo.rotateX(-Math.PI / 2);
    const targetInnerRingMat = new THREE.MeshBasicMaterial({
      color: 0xff3300, transparent: true, opacity: 0.95, side: THREE.DoubleSide
    });
    const targetInnerRingMesh = new THREE.Mesh(targetInnerRingGeo, targetInnerRingMat);
    targetInnerRingMesh.position.y = 0.20;
    targetGlowGroup.add(targetInnerRingMesh);

    const targetBeaconGeo = new THREE.CylinderGeometry(3.6, 3.6, altitude, 16, 1, true);
    const targetBeaconMat = new THREE.MeshBasicMaterial({
      color: 0xef4444, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false
    });
    const targetBeaconMesh = new THREE.Mesh(targetBeaconGeo, targetBeaconMat);
    targetBeaconMesh.position.y = altitude / 2;
    targetGlowGroup.add(targetBeaconMesh);

    const targetLight = new THREE.PointLight(0xff2200, 4.0, 20);
    targetLight.position.y = altitude;
    targetGlowGroup.add(targetLight);

    targetGlowGroup.position.set(worldX, 0, worldZ);
    targetGlowGroup.visible = false;
    scene.add(targetGlowGroup);

    droneInstances.push({
      id: agentName, mesh: clone, detCone, groundRing, groundShadow, verticalTether, avoidanceMesh, flashSphere,
      targetGlowGroup, targetRingMesh, targetInnerRingMesh, targetBeaconMesh, targetLight,
      colorHex, colorStr: AGENT_COLOR_STRS[i % AGENT_COLOR_STRS.length],
      isSelectedForDestruct: false, burstTriggered: false,
      currentRenderPos: new THREE.Vector3(worldX, altitude, worldZ),
      targetPos: new THREE.Vector3(worldX, altitude, worldZ),
      currentRotationY: 0,
      targetRotationY: 0,
      avoidanceTimer: 0.0,
      avoidanceDirection: 'ascend',
      avoidanceReason: 'inter_drone'
    });
  }
}

// ==============================================================================
// SELF-DESTRUCT SELECTION GLOW & BURST ANIMATIONS
// ==============================================================================
const activeBurstAnimations = [];

function updateTargetSelectionGlow() {
  if (!selectKillDrone) return;
  let targetId = selectKillDrone.value;

  if (failedDrones.has(targetId)) {
    const nextActive = droneInstances.find(d => !failedDrones.has(d.id));
    if (nextActive) {
      selectKillDrone.value = nextActive.id;
      targetId = nextActive.id;
    }
  }

  const killGroup = document.querySelector('.kill-drone-group');
  const hasActiveTarget = droneInstances.some(d => !failedDrones.has(d.id));

  if (btnKillDrone) {
    btnKillDrone.disabled = !isLiveRunning || !hasActiveTarget;
    if (hasActiveTarget && isLiveRunning) {
      btnKillDrone.classList.add('glowing-target');
    } else {
      btnKillDrone.classList.remove('glowing-target');
    }
  }

  if (selectKillDrone) {
    if (hasActiveTarget && isLiveRunning) {
      selectKillDrone.classList.add('glowing-select');
    } else {
      selectKillDrone.classList.remove('glowing-select');
    }
  }

  if (killGroup) {
    if (hasActiveTarget && isLiveRunning) killGroup.classList.add('has-target');
    else killGroup.classList.remove('has-target');
  }

  droneInstances.forEach(drone => {
    const isSelected = (drone.id === targetId) && !failedDrones.has(drone.id);
    drone.isSelectedForDestruct = isSelected;
    if (drone.targetGlowGroup) {
      drone.targetGlowGroup.visible = isSelected;
    }

    const itemEl = document.getElementById(`agent-item-${drone.id}`);
    if (itemEl) {
      if (isSelected) {
        itemEl.classList.add('selected-for-kill');
      } else {
        itemEl.classList.remove('selected-for-kill');
      }
    }
  });
}

function triggerSelfDestructBurst(position, colorHex = 0xef4444) {
  const group = new THREE.Group();
  group.position.copy(position);
  scene.add(group);

  // 1. Flash Light
  const flashLight = new THREE.PointLight(0xff5500, 35.0, 60);
  group.add(flashLight);

  // 2. Expanding Shockwave Sphere
  const shockGeo = new THREE.SphereGeometry(1.5, 32, 32);
  const shockMat = new THREE.MeshBasicMaterial({
    color: 0xff3300, transparent: true, opacity: 0.95, wireframe: true
  });
  const shockMesh = new THREE.Mesh(shockGeo, shockMat);
  group.add(shockMesh);

  // 3. Core Flash Sphere
  const coreGeo = new THREE.SphereGeometry(2.5, 16, 16);
  const coreMat = new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 1.0
  });
  const coreMesh = new THREE.Mesh(coreGeo, coreMat);
  group.add(coreMesh);

  // 4. Burst Particles (Optimized count)
  const particleCount = 25;
  const particles = [];
  const pGeo = new THREE.BoxGeometry(0.35, 0.35, 0.35);
  const sparkColors = [0xffffff, 0xffd700, 0xff4500, 0xef4444, 0xd97706];

  for (let i = 0; i < particleCount; i++) {
    const pMat = new THREE.MeshBasicMaterial({
      color: sparkColors[Math.floor(Math.random() * sparkColors.length)],
      transparent: true, opacity: 1.0
    });
    const pMesh = new THREE.Mesh(pGeo, pMat);

    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos((Math.random() * 2) - 1);
    const speed = 15.0 + Math.random() * 28.0;

    const vx = speed * Math.sin(phi) * Math.cos(theta);
    const vy = speed * Math.sin(phi) * Math.sin(theta) + 7.0;
    const vz = speed * Math.cos(phi);

    pMesh.position.set(0, 0, 0);
    pMesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, 0);
    group.add(pMesh);

    particles.push({
      mesh: pMesh,
      velocity: new THREE.Vector3(vx, vy, vz),
      rotSpeed: new THREE.Vector3((Math.random() - 0.5) * 22, (Math.random() - 0.5) * 22, (Math.random() - 0.5) * 22),
      decay: 0.85 + Math.random() * 0.5
    });
  }

  // 5. Smoke Cloud Particles (Optimized count)
  const smokeParticles = [];
  const smokeGeo = new THREE.SphereGeometry(1.4, 8, 8);
  for (let s = 0; s < 6; s++) {
    const sMat = new THREE.MeshBasicMaterial({
      color: 0x334155, transparent: true, opacity: 0.6
    });
    const sMesh = new THREE.Mesh(smokeGeo, sMat);
    sMesh.position.set((Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2, (Math.random() - 0.5) * 2);
    group.add(sMesh);
    smokeParticles.push({
      mesh: sMesh,
      velocity: new THREE.Vector3((Math.random() - 0.5) * 3.5, 3.5 + Math.random() * 4.5, (Math.random() - 0.5) * 3.5),
      growth: 3.5 + Math.random() * 3.5
    });
  }

  activeBurstAnimations.push({
    group, flashLight, shockMesh, coreMesh, particles, smokeParticles,
    age: 0, duration: 1.4
  });
}

// ==============================================================================
// FOLLOW DRONE CAMERA MODE CONTROLLER
// ==============================================================================
function enterFollowMode(droneId) {
  const drone = droneInstances.find(d => d.id === droneId);
  if (!drone || failedDrones.has(droneId)) return;

  if (!isFollowMode) {
    savedFreeCamPos.copy(camera.position);
    savedFreeCamTarget.copy(controls.target);
    isFollowMode = true;
  }

  followedDroneId = droneId;
  isCameraTransitioning = true;
  transitionProgress = 0.0;
  transitionStartCamPos.copy(camera.position);
  transitionStartTargetPos.copy(controls.target);

  if (followHudCard) {
    followHudCard.classList.remove('hidden');
    setTxt(followTitle, `FOLLOWING ${drone.id.toUpperCase()}`);
    if (followDot) {
      followDot.style.background = drone.colorStr;
      followDot.style.color = drone.colorStr;
    }
  }
  updateFollowHudCard();
}

function exitFollowMode() {
  if (!isFollowMode) return;

  isFollowMode = false;
  followedDroneId = null;
  isCameraTransitioning = true;
  transitionProgress = 0.0;
  transitionStartCamPos.copy(camera.position);
  transitionStartTargetPos.copy(controls.target);

  if (followHudCard) {
    followHudCard.classList.add('hidden');
  }
  showToast('Exited Follow View — Returned to Free Orbit Camera', 'survivor');
}

function updateFollowHudCard() {
  if (!isFollowMode || !followedDroneId) return;
  const drone = droneInstances.find(d => d.id === followedDroneId);
  if (!drone) return;

  const isFailed = failedDrones.has(drone.id);
  const posLabel = document.getElementById(`pos-${drone.id}`);
  const currentPosStr = posLabel ? posLabel.innerText : '[0, 0]';

  if (followPos) setTxt(followPos, currentPosStr);
  if (followStatus) {
    if (isFailed) {
      setTxt(followStatus, 'SYSTEM FAILURE / OFFLINE');
      followStatus.style.color = '#ef4444';
    } else {
      setTxt(followStatus, 'THERMAL SEARCH ACTIVE');
      followStatus.style.color = '#34d399';
    }
  }
}



// ==============================================================================
// UI SETUP & EVENT LISTENERS
// ==============================================================================
function setupUI() {
  const meta = replayData.metadata || {};
  setTxt(statGrid, `${meta.grid_size || 10} x ${meta.grid_size || 10}`);
  setTxt(statScale, `${meta.cell_size_meters || 10} m`);
  setTxt(statBuildings, meta.num_buildings || 6);
  setTxt(statAgents, meta.num_agents || 5);

  if (agentList) {
    agentList.innerHTML = '';
    droneInstances.forEach((drone, idx) => {
      const item = document.createElement('div');
      item.className = 'agent-item';
      item.id = `agent-item-${drone.id}`;
      item.title = `Click to follow ${drone.id.toUpperCase()}`;
      item.innerHTML = `
        <div class="agent-dot" style="background:${drone.colorStr}; color:${drone.colorStr}"></div>
        <div class="agent-name">AGENT_${idx}</div>
        <div class="agent-pos" id="pos-${drone.id}">[0, 0]</div>
      `;
      item.addEventListener('click', () => {
        if (!failedDrones.has(drone.id)) {
          if (isFollowMode && followedDroneId === drone.id) {
            exitFollowMode();
          } else {
            enterFollowMode(drone.id);
            showToast(`🎥 FOLLOW VIEW ACTIVE: Tracking ${drone.id.toUpperCase()}`, 'survivor');
          }
        }
      });
      agentList.appendChild(item);
    });
  }

  updateStepUI(0);
  updateTargetSelectionGlow();
}

function setupEventListeners() {
  if (btnPlay) btnPlay.addEventListener('click', toggleReplayPlay);
  if (btnReset) btnReset.addEventListener('click', resetAll);
  if (btnReplay) btnReplay.addEventListener('click', startLiveSimulation);

  if (btnEditLayout) {
    btnEditLayout.addEventListener('click', toggleEditLayoutMode);
  }

  if (btnExitFollow) {
    btnExitFollow.addEventListener('click', exitFollowMode);
  }

  const btnGizmoTranslate = document.getElementById('gizmo-mode-translate');
  const btnGizmoRotate = document.getElementById('gizmo-mode-rotate');
  const btnGizmoScale = document.getElementById('gizmo-mode-scale');
  const sliderModelSize = document.getElementById('model-size-slider');
  const valModelSize = document.getElementById('model-size-val');

  function updateGizmoUI(mode) {
    if (transformControls) transformControls.setMode(mode);
    if (btnGizmoTranslate) btnGizmoTranslate.style.background = mode === 'translate' ? '#3b82f6' : 'rgba(30,41,59,0.8)';
    if (btnGizmoRotate) btnGizmoRotate.style.background = mode === 'rotate' ? '#3b82f6' : 'rgba(30,41,59,0.8)';
    if (btnGizmoScale) btnGizmoScale.style.background = mode === 'scale' ? '#3b82f6' : 'rgba(30,41,59,0.8)';
  }

  if (btnGizmoTranslate) btnGizmoTranslate.addEventListener('click', () => updateGizmoUI('translate'));
  if (btnGizmoRotate) btnGizmoRotate.addEventListener('click', () => updateGizmoUI('rotate'));
  if (btnGizmoScale) btnGizmoScale.addEventListener('click', () => updateGizmoUI('scale'));

  if (sliderModelSize) {
    sliderModelSize.addEventListener('input', (e) => {
      const factor = parseFloat(e.target.value);
      if (valModelSize) valModelSize.innerText = `${factor.toFixed(2)}x`;

      if (selectedModelObject) {
        if (!selectedModelObject.userData.baseScale) {
          selectedModelObject.userData.baseScale = selectedModelObject.scale.clone();
        }
        const base = selectedModelObject.userData.baseScale;
        selectedModelObject.scale.set(base.x * factor, base.y * factor, base.z * factor);
      }
    });
  }

  // Keyboard Shortcuts for Transform Controls in Edit Mode (T = Move, R = Rotate, S = Scale)
  window.addEventListener('keydown', (event) => {
    if (!isEditLayoutMode || !transformControls) return;
    const k = event.key.toLowerCase();
    if (k === 't') updateGizmoUI('translate');
    if (k === 'r') updateGizmoUI('rotate');
    if (k === 's') updateGizmoUI('scale');
  });

  const canvas = document.getElementById('scene-canvas');
  if (canvas) {
    canvas.addEventListener('click', (event) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);

      // Handle Model Selection when Edit Mode is Active
      if (isEditLayoutMode) {
        const selectableObjects = editableModelGroups.map(m => m.object);
        const intersects = raycaster.intersectObjects(selectableObjects, true);
        if (intersects.length > 0) {
          let hitObj = intersects[0].object;
          while (hitObj.parent && hitObj.parent !== envGroup && !editableModelGroups.some(m => m.object === hitObj)) {
            hitObj = hitObj.parent;
          }
          const item = editableModelGroups.find(m => m.object === hitObj);
          if (item && transformControls) {
            selectedModelObject = item.object;
            transformControls.attach(item.object);
            showToast(`Selected model [${item.id.toUpperCase()}]! Drag 3D handles to move/rotate/scale. (Keys: T, R, S)`, 'status');
          }
        }
        return;
      }

      const droneMeshes = droneInstances.map(d => d.mesh);
      const intersects = raycaster.intersectObjects(droneMeshes, true);

      if (intersects.length > 0) {
        let hitObj = intersects[0].object;
        while (hitObj.parent && !droneInstances.some(d => d.mesh === hitObj)) {
          hitObj = hitObj.parent;
        }
        const clickedDrone = droneInstances.find(d => d.mesh === hitObj);
        if (clickedDrone && !failedDrones.has(clickedDrone.id)) {
          if (isFollowMode && followedDroneId === clickedDrone.id) {
            exitFollowMode();
          } else {
            enterFollowMode(clickedDrone.id);
            showToast(`🎥 FOLLOW VIEW ACTIVE: Tracking ${clickedDrone.id.toUpperCase()}`, 'survivor');
          }
        }
      }
    });
  }

  if (selectKillDrone) {
    selectKillDrone.addEventListener('change', () => {
      updateTargetSelectionGlow();
    });
  }

  // Helper function to generate random human positions (fixed 10 count for lag-free performance)
  function generateRandomHumans(count = 10) {
    const gridSize = (replayData && replayData.metadata && replayData.metadata.grid_size) ? replayData.metadata.grid_size : 10;
    const used = new Set();
    const survivors = [];
    const hiddenSurvivors = [];
    let attempts = 0;
    while (survivors.length < count && attempts < 200) {
      attempts++;
      const r = Math.floor(Math.random() * gridSize);
      const c = Math.floor(Math.random() * gridSize);
      const key = `${r},${c}`;
      if (!used.has(key)) {
        used.add(key);
        survivors.push([r, c]);
        if (Math.random() < 0.35) {
          hiddenSurvivors.push([r, c]);
        }
      }
    }
    return { survivors, hidden_survivors: hiddenSurvivors, count: survivors.length };
  }

  // 1. RANDOMIZE HUMAN POSITIONS (INSTANT 0MS RESPONSE, FIXED 10 HUMANS FOR OPTIMAL PERFORMANCE)
  if (btnRandomizeHumans) {
    btnRandomizeHumans.addEventListener('click', () => {
      const randomSet = generateRandomHumans(10);

      // Instant local 3D rendering (0ms lag!)
      setupSurvivors({ survivors: randomSet.survivors, hidden_survivors: randomSet.hidden_survivors });
      updateSurvivorGlowState(0);
      showToast('Randomized 10 human target placements', 'survivor');

      // Non-blocking background API sync with 500ms timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 500);
      fetch(`${API_BASE_URL}/randomize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 10, survivors: randomSet.survivors, hidden_survivors: randomSet.hidden_survivors }),
        signal: controller.signal
      }).then(() => clearTimeout(timeoutId)).catch(() => {});
    });
  }

  // 2. START LIVE SIMULATION (Drives real-time PPO model inference)
  if (btnStartLive) {
    btnStartLive.addEventListener('click', startLiveSimulation);
  }

  // 3. MANUAL SELF-DESTRUCT / KILL DRONE TRIGGER
  if (btnKillDrone) {
    btnKillDrone.addEventListener('click', async () => {
      if (!isLiveRunning) return;
      const targetAgent = selectKillDrone ? selectKillDrone.value : 'agent_2';
      const drone = droneInstances.find(d => d.id === targetAgent);

      if (drone && !failedDrones.has(targetAgent)) {
        triggerSelfDestructBurst(drone.mesh.position, drone.colorHex);
        drone.burstTriggered = true;
      }

      try {
        const res = await fetch(`${API_BASE_URL}/kill_drone`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ drone_id: targetAgent })
        });
        if (res.ok) {
          const data = await res.json();
          showToast(`💥 SELF-DESTRUCT: ${targetAgent.toUpperCase()} destroyed!`, 'collision');
          failedDrones.add(targetAgent);
          updateTargetSelectionGlow();
        }
      } catch (err) {
        console.error('Failed to trigger kill_drone API:', err);
      }
    });
  }

  if (progressSlider) {
    progressSlider.addEventListener('input', (e) => {
      if (isLiveMode) return;
      const pct = parseFloat(e.target.value) / 100;
      currentTime = pct * totalDurationSeconds;
      updateSimulationTime(currentTime);
    });
  }

  if (speedBtns && speedBtns.length > 0) {
    speedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        playbackSpeed = parseFloat(btn.dataset.speed);
      });
    });
  }
}

// ==============================================================================
// LIVE INFERENCE CONTROLLER
// ==============================================================================
async function startLiveSimulation() {
  if (isLiveRunning) return;

  btnRandomizeHumans.disabled = true;
  btnStartLive.disabled = true;
  if (btnKillDrone) btnKillDrone.disabled = false;
  statsModal.classList.remove('visible');
  triggeredEvents.clear();
  failedDrones.clear();
  droneInstances.forEach(d => { d.burstTriggered = false; });
  clearFailedZoneHighlights();

  isLiveMode = true;
  isLiveRunning = true;
  modeBadge.innerText = 'LIVE MODEL INFERENCE';
  modeBadge.style.background = 'rgba(16, 185, 129, 0.15)';
  modeBadge.style.color = '#10b981';

  try {
    const res = await fetch(`${API_BASE_URL}/start`, { method: 'POST' });
    if (!res.ok) throw new Error('Start request failed');
    const data = await res.json();

    replayData = data;
    setupSurvivors({
      survivors: data.initial_state.survivor_positions,
      hidden_survivors: data.initial_state.hidden_survivors
    });
    setupDrones();
    updateSurvivorGlowState(0);
    updateTargetSelectionGlow();

    showToast('LIVE PPO SIMULATION STARTED (Step-by-step model inference)', 'survivor');

    // Poll /step every 300ms
    liveIntervalId = setInterval(executeLiveStep, 300 / playbackSpeed);
  } catch (err) {
    console.error('Failed to start live simulation:', err);
    showToast(`Failed to connect to Live API server at ${API_BASE_URL}`, 'collision');
    btnRandomizeHumans.disabled = false;
    btnStartLive.disabled = false;
    if (btnKillDrone) btnKillDrone.disabled = true;
    isLiveRunning = false;
  }
}

function clearFailedZoneHighlights() {
  failedZoneMeshes.forEach(m => {
    if (m.parent) m.parent.remove(m);
    if (m.geometry) m.geometry.dispose();
    if (m.material) m.material.dispose();
  });
  failedZoneMeshes.length = 0;
}

function addFailedZoneHighlight(zoneBounds, agentName, reassignedTo) {
  const cellSize = replayData.metadata?.cell_size_meters || 10.0;
  const startRow = zoneBounds[0];
  const endRow = zoneBounds[1];
  const numRows = endRow - startRow;

  const width = 10 * cellSize;
  const depth = numRows * cellSize;

  const geom = new THREE.PlaneGeometry(width, depth);
  const mat = new THREE.MeshBasicMaterial({
    color: 0xef4444,
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide
  });

  const plane = new THREE.Mesh(geom, mat);
  plane.rotation.x = -Math.PI / 2;
  plane.position.x = width / 2;
  plane.position.y = 0.2;
  plane.position.z = (startRow + numRows / 2) * cellSize;

  scene.add(plane);
  failedZoneMeshes.push(plane);
}

function lerpAngle(start, end, t) {
  let diff = (end - start) % (Math.PI * 2);
  if (diff < -Math.PI) diff += Math.PI * 2;
  if (diff > Math.PI) diff -= Math.PI * 2;
  return start + diff * t;
}

let liveErrorCount = 0;

async function executeLiveStep() {
  try {
    const res = await fetch(`${API_BASE_URL}/step`, { method: 'POST' });
    if (!res.ok) throw new Error('Step request failed');
    const data = await res.json();
    liveErrorCount = 0;

    const sData = data.step;
    const cellSize = replayData.metadata?.cell_size_meters || 10.0;

    // Track failed agents
    (sData.failed_agents || []).forEach(fa => failedDrones.add(fa));

    // 1. Set Drone Target Positions & Rotations
    droneInstances.forEach(drone => {
      const isFailed = failedDrones.has(drone.id);
      const pos = sData.agent_positions[drone.id] || [0, 0];
      const targetX = (pos[1] + 0.5) * cellSize;
      const targetZ = (pos[0] + 0.5) * cellSize;
      const alt = (sData.agent_altitudes && sData.agent_altitudes[drone.id] !== undefined) ? sData.agent_altitudes[drone.id] : 0;
      const targetY = isFailed ? 1.0 : (10.0 + alt * 4.0);

      // Calculate direction to new target for smooth heading turn
      const dx = targetX - drone.currentRenderPos.x;
      const dz = targetZ - drone.currentRenderPos.z;
      if (dx * dx + dz * dz > 0.05) {
        drone.targetRotationY = Math.atan2(dx, dz);
      }

      drone.targetPos.set(targetX, targetY, targetZ);

      if (isFailed) {
        if (!drone.burstTriggered) {
          drone.burstTriggered = true;
          triggerSelfDestructBurst(drone.currentRenderPos, drone.colorHex);
        }
        drone.mesh.traverse(child => {
          if (child.isMesh && child.material) {
            if (!child.userData.originalColor) child.userData.originalColor = true;
            child.material = child.material.clone();
            child.material.color.setHex(0x475569);
          }
        });
        drone.detCone.visible = false;
        drone.groundRing.visible = false;
      }

      const posLabel = document.getElementById(`pos-${drone.id}`);
      const altStr = alt > 0 ? ` L${alt}` : '';
      if (posLabel) posLabel.innerText = isFailed ? `[${pos[0]}, ${pos[1]}] (FAILED)` : `[${pos[0]}, ${pos[1]}]${altStr}`;
    });

    updateTargetSelectionGlow();

    // 2. Update Red/Green Glow based on remaining survivors
    updateSurvivorGlowState(sData.step, sData.survivors_remaining);

    // 3. Process Events
    (sData.events || []).forEach(evt => {
      const evtKey = `live-${sData.step}-${evt.type}-${evt.agent || ''}-${(evt.position || evt.survivor_position || []).join(',')}`;
      if (!triggeredEvents.has(evtKey)) {
        triggeredEvents.add(evtKey);
        triggerEventCue(evt);
        if (evt.type === 'drone_failed' && evt.zone) {
          addFailedZoneHighlight(evt.zone, evt.agent, evt.reassigned_to);
        }
      }
    });

    // 4. Update HUD
    hudStep.innerText = `LIVE STEP ${sData.step} / ${replayData.metadata.max_steps}`;
    const survivorsFound = (replayData?.metadata?.num_survivors || 10) - sData.survivors_remaining.length;
    statSurvivors.innerText = `${survivorsFound} Located`;
    statCoverage.innerText = `${sData.coverage_pct}%`;

    let collCount = 0;
    triggeredEvents.forEach(k => { if (k.includes('collision')) collCount++; });
    statCollisions.innerText = collCount;

    // Track summary for final stats
    liveSummaryData = {
      open_rescued: sData.open_rescued, open_total: sData.open_total,
      hidden_rescued: sData.hidden_rescued, hidden_total: sData.hidden_total,
      total_rescued: survivorsFound, total_survivors: replayData.metadata.num_survivors,
      collisions: collCount, total_steps: sData.step, coverage_pct: sData.coverage_pct,
      kill_event_occurred: sData.kill_event_occurred || failedDrones.size > 0,
      failed_zone_recovery: sData.failed_zone_recovery || {}
    };

    // 5. Episode Termination
    if (data.is_done) {
      clearInterval(liveIntervalId);
      isLiveRunning = false;
      btnRandomizeHumans.disabled = false;
      btnStartLive.disabled = false;
      if (btnKillDrone) btnKillDrone.disabled = true;
      fetch(`${API_BASE_URL}/reset`, { method: 'POST' });
      showPostSimulationStatsLive();
    }
  } catch (err) {
    liveErrorCount++;
    console.warn(`Error during live step (attempt ${liveErrorCount}/3):`, err);
    if (liveErrorCount >= 3) {
      clearInterval(liveIntervalId);
      isLiveRunning = false;
      btnRandomizeHumans.disabled = false;
      btnStartLive.disabled = false;
      if (btnKillDrone) btnKillDrone.disabled = true;
      showToast('Live step failed after 3 retries.', 'collision');
    }
  }
}

function resetAll() {
  if (isLiveRunning) {
    clearInterval(liveIntervalId);
    isLiveRunning = false;
    fetch(`${API_BASE_URL}/reset`, { method: 'POST' });
  }

  btnRandomizeHumans.disabled = false;
  btnStartLive.disabled = false;
  if (btnKillDrone) btnKillDrone.disabled = true;
  isPlayingReplay = false;
  currentTime = 0.0;
  currentStepIndex = 0;
  btnPlay.innerHTML = '&#9654;';
  triggeredEvents.clear();
  failedDrones.clear();
  droneInstances.forEach(d => { d.burstTriggered = false; });
  clearFailedZoneHighlights();
  statsModal.classList.remove('visible');

  exitFollowMode();
  updateSurvivorGlowState(0);
  updateSimulationTime(0.0);
  updateTargetSelectionGlow();
}

function toggleReplayPlay() {
  if (isLiveRunning) return;
  isPlayingReplay = !isPlayingReplay;
  btnPlay.innerHTML = isPlayingReplay ? '&#10074;&#10074;' : '&#9654;';
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();

  if (isPlayingReplay && !isLiveRunning) {
    currentTime += delta * playbackSpeed;
    if (currentTime >= totalDurationSeconds) {
      currentTime = totalDurationSeconds;
      isPlayingReplay = false;
      btnPlay.innerHTML = '&#9654;';
      showPostSimulationStatsReplay();
    }
    updateSimulationTime(currentTime);
  }

  const timeSec = clock.getElapsedTime();

  // Smoothly interpolate drone positions and rotations
  droneInstances.forEach((drone, idx) => {
    const isFailed = failedDrones.has(drone.id);

    // Continuous smooth horizontal lerp towards targetPos
    const posLerpFactor = Math.min(1.0, delta * 9.0);
    drone.currentRenderPos.x += (drone.targetPos.x - drone.currentRenderPos.x) * posLerpFactor;
    drone.currentRenderPos.z += (drone.targetPos.z - drone.currentRenderPos.z) * posLerpFactor;

    // Smooth elevation (Y-axis) lerp with hovering motion
    if (!isFailed) {
      drone.avoidanceTimer = Math.max(0.0, drone.avoidanceTimer - delta);
      const hoverOffset = Math.sin(timeSec * 3 + idx) * 0.2;
      const desiredY = drone.targetPos.y + hoverOffset;
      const yLerpFactor = Math.min(1.0, delta * 6.0);
      drone.currentRenderPos.y += (desiredY - drone.currentRenderPos.y) * yLerpFactor;

      // Smooth heading turn
      const rotLerpFactor = Math.min(1.0, delta * 7.5);
      drone.currentRotationY = lerpAngle(drone.currentRotationY, drone.targetRotationY, rotLerpFactor);
      drone.mesh.rotation.y = drone.currentRotationY;
    } else {
      drone.currentRenderPos.y = 1.0;
    }

    const worldX = drone.currentRenderPos.x;
    const worldY = drone.currentRenderPos.y;
    const worldZ = drone.currentRenderPos.z;

    drone.mesh.position.set(worldX, worldY, worldZ);
    drone.detCone.position.set(worldX, worldY / 2, worldZ);
    drone.groundRing.position.set(worldX, 0.15, worldZ);
    drone.flashSphere.position.set(worldX, worldY, worldZ);

    // Dynamic Ground Shadow scale & opacity based on altitude
    if (drone.groundShadow) {
      drone.groundShadow.position.set(worldX, 0.04, worldZ);
      const currentAlt = Math.max(0, (worldY - 10.0) / 4.0);
      const shadowScale = 1.0 + currentAlt * 0.35;
      drone.groundShadow.scale.set(shadowScale, shadowScale, shadowScale);
      drone.groundShadow.material.opacity = isFailed ? 0.0 : Math.max(0.12, 0.55 - currentAlt * 0.12);
    }

    // Vertical Tether Line connecting drone to ground shadow
    if (drone.verticalTether) {
      const tetherHeight = Math.max(0.1, worldY - 0.1);
      drone.verticalTether.scale.set(1, tetherHeight, 1);
      drone.verticalTether.position.set(worldX, 0.1 + tetherHeight / 2, worldZ);

      if (drone.avoidanceTimer > 0) {
        const pulse = 0.5 + 0.5 * Math.sin(timeSec * 14.0);
        const avColor = drone.avoidanceDirection === 'ascend' ? 0x38bdf8 : 0xf59e0b;
        drone.verticalTether.material.color.setHex(avColor);
        drone.verticalTether.material.opacity = 0.75 + 0.25 * pulse;

        if (drone.avoidanceMesh) {
          drone.avoidanceMesh.position.set(worldX, worldY, worldZ);
          drone.avoidanceMesh.visible = true;
          drone.avoidanceMesh.material.color.setHex(avColor);
          drone.avoidanceMesh.material.opacity = (drone.avoidanceTimer / 2.0) * (0.6 + 0.4 * pulse);
          const avScale = 1.0 + (2.0 - drone.avoidanceTimer) * 0.8;
          drone.avoidanceMesh.scale.setScalar(avScale);
        }
      } else {
        drone.verticalTether.material.color.setHex(drone.colorHex);
        drone.verticalTether.material.opacity = isFailed ? 0.0 : 0.35;
        if (drone.avoidanceMesh) drone.avoidanceMesh.visible = false;
      }
    }

    if (drone.targetGlowGroup) {
      drone.targetGlowGroup.position.set(worldX, 0, worldZ);

      if (drone.targetGlowGroup.visible && drone.isSelectedForDestruct && !isFailed) {
        const pulse = 0.5 + 0.5 * Math.sin(timeSec * 7.5);
        drone.targetRingMesh.material.opacity = 0.5 + 0.5 * pulse;
        drone.targetBeaconMesh.material.opacity = 0.15 + 0.25 * pulse;
        drone.targetLight.intensity = 2.0 + 4.0 * pulse;
        drone.targetInnerRingMesh.rotation.z += delta * 2.5;
      }
    }
  });

  // Camera Follow Mode & Camera Transition Control
  if (isFollowMode && followedDroneId) {
    const drone = droneInstances.find(d => d.id === followedDroneId);
    if (drone) {
      const heading = drone.currentRotationY;
      const distBehind = 22.0;
      const heightAbove = 12.0;

      const targetCamX = drone.currentRenderPos.x - Math.sin(heading) * distBehind;
      const targetCamZ = drone.currentRenderPos.z - Math.cos(heading) * distBehind;
      const targetCamY = drone.currentRenderPos.y + heightAbove;

      const targetLookAtX = drone.currentRenderPos.x + Math.sin(heading) * 4.0;
      const targetLookAtZ = drone.currentRenderPos.z + Math.cos(heading) * 4.0;
      const targetLookAtY = drone.currentRenderPos.y + 1.0;

      _targetCamVec.set(targetCamX, targetCamY, targetCamZ);
      _targetLookVec.set(targetLookAtX, targetLookAtY, targetLookAtZ);

      if (isCameraTransitioning) {
        transitionProgress += delta * 2.5;
        if (transitionProgress >= 1.0) {
          transitionProgress = 1.0;
          isCameraTransitioning = false;
        }
        const t = 0.5 - 0.5 * Math.cos(transitionProgress * Math.PI);
        camera.position.lerpVectors(transitionStartCamPos, _targetCamVec, t);
        controls.target.lerpVectors(transitionStartTargetPos, _targetLookVec, t);
      } else {
        camera.position.lerp(_targetCamVec, Math.min(1.0, delta * 6.0));
        controls.target.lerp(_targetLookVec, Math.min(1.0, delta * 8.0));
      }
      controls.update();
      updateFollowHudCard();
    }
  } else if (isCameraTransitioning) {
    transitionProgress += delta * 2.2;
    if (transitionProgress >= 1.0) {
      transitionProgress = 1.0;
      isCameraTransitioning = false;
    }
    const t = 0.5 - 0.5 * Math.cos(transitionProgress * Math.PI);
    camera.position.lerpVectors(transitionStartCamPos, savedFreeCamPos, t);
    controls.target.lerpVectors(transitionStartTargetPos, savedFreeCamTarget, t);
    controls.update();
  } else {
    controls.update();
  }

  // Update active burst animations
  for (let i = activeBurstAnimations.length - 1; i >= 0; i--) {
    const anim = activeBurstAnimations[i];
    anim.age += delta;
    const progress = anim.age / anim.duration;

    if (progress >= 1.0) {
      scene.remove(anim.group);
      anim.group.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
          else child.material.dispose();
        }
      });
      activeBurstAnimations.splice(i, 1);
      continue;
    }

    anim.flashLight.intensity = Math.max(0, 35.0 * (1 - progress * 1.8));

    const shockScale = 1.0 + progress * 16.0;
    anim.shockMesh.scale.setScalar(shockScale);
    anim.shockMesh.material.opacity = Math.max(0, 0.95 - progress * 1.4);

    const coreScale = Math.max(0.01, 1.0 - progress * 2.2);
    anim.coreMesh.scale.setScalar(coreScale);
    anim.coreMesh.material.opacity = Math.max(0, 1.0 - progress * 2.5);

    anim.particles.forEach(p => {
      p.mesh.position.x += p.velocity.x * delta;
      p.mesh.position.y += p.velocity.y * delta;
      p.mesh.position.z += p.velocity.z * delta;

      p.velocity.y -= 18.0 * delta;
      p.velocity.multiplyScalar(Math.max(0, 1 - 0.8 * delta));

      p.mesh.rotation.x += p.rotSpeed.x * delta;
      p.mesh.rotation.y += p.rotSpeed.y * delta;

      const pScale = Math.max(0, 1.0 - progress * p.decay);
      p.mesh.scale.setScalar(pScale);
      p.mesh.material.opacity = Math.max(0, 1.0 - progress * (p.decay * 1.1));
    });

    anim.smokeParticles.forEach(sp => {
      sp.mesh.position.x += sp.velocity.x * delta;
      sp.mesh.position.y += sp.velocity.y * delta;
      sp.mesh.position.z += sp.velocity.z * delta;
      const sScale = 1.0 + progress * sp.growth;
      sp.mesh.scale.setScalar(sScale);
      sp.mesh.material.opacity = Math.max(0, 0.6 * (1 - progress));
    });
  }

  renderer.render(scene, camera);
}

function updateSimulationTime(t) {
  const stepsCount = replayData.steps ? replayData.steps.length : 1;
  const progressPct = Math.min(1.0, t / totalDurationSeconds);

  progressBar.style.width = `${progressPct * 100}%`;
  progressSlider.value = progressPct * 100;

  const exactStepFloat = progressPct * (stepsCount - 1);
  const stepIdx = Math.floor(exactStepFloat);
  const alpha = exactStepFloat - stepIdx;

  currentStepIndex = stepIdx;
  const currentStepData = replayData.steps ? (replayData.steps[stepIdx] || replayData.steps[0]) : null;
  if (!currentStepData) return;

  const nextStepData = replayData.steps[Math.min(stepIdx + 1, stepsCount - 1)] || currentStepData;
  const cellSize = replayData.metadata?.cell_size_meters || 10.0;

  droneInstances.forEach(drone => {
    const posCurr = currentStepData.agent_positions[drone.id] || [0, 0];
    const posNext = nextStepData.agent_positions[drone.id] || posCurr;

    const altCurr = (currentStepData.agent_altitudes && currentStepData.agent_altitudes[drone.id] !== undefined) ? currentStepData.agent_altitudes[drone.id] : 0;
    const altNext = (nextStepData.agent_altitudes && nextStepData.agent_altitudes[drone.id] !== undefined) ? nextStepData.agent_altitudes[drone.id] : altCurr;
    const altInterp = altCurr + (altNext - altCurr) * alpha;

    const rInterp = posCurr[0] + (posNext[0] - posCurr[0]) * alpha;
    const cInterp = posCurr[1] + (posNext[1] - posCurr[1]) * alpha;

    const worldX = (cInterp + 0.5) * cellSize;
    const worldZ = (rInterp + 0.5) * cellSize;
    const isFailed = failedDrones.has(drone.id);
    const targetY = isFailed ? 1.0 : (10.0 + altInterp * 4.0);

    const dx = (posNext[1] - posCurr[1]) * cellSize;
    const dz = (posNext[0] - posCurr[0]) * cellSize;
    if (dx * dx + dz * dz > 0.01) {
      drone.targetRotationY = Math.atan2(dx, dz);
    }

    drone.targetPos.set(worldX, targetY, worldZ);

    const posLabel = document.getElementById(`pos-${drone.id}`);
    const altStr = Math.round(altInterp) > 0 ? ` L${Math.round(altInterp)}` : '';
    if (posLabel) posLabel.innerText = isFailed ? `[${Math.round(rInterp)}, ${Math.round(cInterp)}] (FAILED)` : `[${Math.round(rInterp)}, ${Math.round(cInterp)}]${altStr}`;
  });

  updateSurvivorGlowState(stepIdx);
  updateStepUI(stepIdx);
}

function updateStepUI(stepIdx) {
  if (!replayData.steps) return;
  const stepsCount = replayData.steps.length;
  setTxt(hudStep, `STEP ${stepIdx} / ${stepsCount - 1}`);
  setTxt(hudTime, `${formatTime(Math.floor(currentTime))} / ${formatTime(Math.floor(totalDurationSeconds))}`);

  const stepData = replayData.steps[stepIdx] || replayData.steps[0];
  const survivorsFound = (replayData?.metadata?.num_survivors || 10) - (stepData.survivors_remaining ? stepData.survivors_remaining.length : 0);
  setTxt(statSurvivors, `${survivorsFound} Located`);
  setTxt(statCoverage, `${((stepIdx / (stepsCount - 1)) * (replayData.summary?.final_coverage_pct || 40)).toFixed(1)}%`);
}

function triggerEventCue(evt) {
  if (evt.type === 'survivor_found') {
    const hiddenTag = evt.is_building_hidden ? ' (THERMAL BUILDING-HIDDEN)' : ' (OPEN-AREA)';
    showToast(`SURVIVOR LOCATED${hiddenTag} at [${evt.survivor_position.join(', ')}] by ${evt.found_by.toUpperCase()}`, 'survivor');
  } else if (evt.type === 'collision') {
    showToast(`AGENT COLLISION WARNING: ${evt.agent.toUpperCase()} at [${evt.position.join(', ')}]`, 'collision');
    const drone = droneInstances.find(d => d.id === evt.agent);
    if (drone) {
      drone.flashSphere.material.opacity = 0.8;
      setTimeout(() => { drone.flashSphere.material.opacity = 0.0; }, 600);
    }
  } else if (evt.type === 'drone_failed') {
    const reassignedStr = (evt.reassigned_to || []).map(a => a.toUpperCase()).join(' & ') || 'Nearby drones';
    showToast(`DRONE FAILURE ALERT: ${evt.agent.toUpperCase()} is offline! ${reassignedStr} reassigned to cover zone [rows ${evt.zone.join('-')}]`, 'collision');
  } else if (evt.type === 'elevation_avoidance') {
    const drone = droneInstances.find(d => d.id === evt.agent);
    if (drone) {
      drone.avoidanceTimer = 2.0;
      drone.avoidanceDirection = evt.direction || 'ascend';
      drone.avoidanceReason = evt.reason || 'inter_drone';
    }
    const dirIcon = evt.direction === 'ascend' ? '⬆️' : '⬇️';
    const reasonStr = evt.reason === 'inter_drone' ? 'Drone-to-Drone Avoidance' : (evt.reason === 'obstacle' ? 'Building / Obstacle Clearance' : 'Tactical Height Adjustment');
    showToast(`${dirIcon} ${evt.agent.toUpperCase()} ${evt.direction.toUpperCase()}ING [L${evt.from_alt} → L${evt.to_alt}] — ${reasonStr}`, 'elevation');
  }
}

function showToast(message, type) {
  if (!toastContainer) return;
  const toast = document.createElement('div');
  toast.className = `event-toast ${type}`;
  toast.innerText = message;
  toastContainer.appendChild(toast);
  setTimeout(() => { toast.remove(); }, 4000);
}

function showPostSimulationStatsLive() {
  setTxt(finalSurvivors, `${liveSummaryData.total_rescued} / ${liveSummaryData.total_survivors}`);
  setTxt(finalOpenSurvivors, `${liveSummaryData.open_rescued} / ${liveSummaryData.open_total}`);
  setTxt(finalHiddenSurvivors, `${liveSummaryData.hidden_rescued} / ${liveSummaryData.hidden_total}`);
  setTxt(finalCoverage, `${liveSummaryData.coverage_pct}%`);
  setTxt(finalCollisions, liveSummaryData.collisions);
  setTxt(finalSteps, liveSummaryData.total_steps);

  if (cardFailedRecovery && finalFailedRecovery) {
    if (liveSummaryData.kill_event_occurred && liveSummaryData.failed_zone_recovery) {
      cardFailedRecovery.style.display = 'block';
      const parts = Object.entries(liveSummaryData.failed_zone_recovery).map(([fa, pct]) => `${fa.toUpperCase()}: ${pct}%`);
      finalFailedRecovery.innerText = parts.join(', ') || '0%';
    } else {
      cardFailedRecovery.style.display = 'none';
    }
  }

  if (statsModal) statsModal.classList.add('visible');
}

function showPostSimulationStatsReplay() {
  setTxt(finalSurvivors, `${replayData.summary?.survivors_found || 0} / ${replayData.summary?.survivors_total || 10}`);
  setTxt(finalOpenSurvivors, `6 / 6`);
  setTxt(finalHiddenSurvivors, `3 / 4`);
  setTxt(finalCoverage, `${replayData.summary?.final_coverage_pct || 40}%`);
  setTxt(finalCollisions, `2`);
  setTxt(finalSteps, replayData.summary?.total_steps || 100);

  if (statsModal) statsModal.classList.add('visible');
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// ==============================================================================
// AREA MAPPING MODE
// ==============================================================================
let currentMode = 'rescue'; // 'rescue' or 'mapping'
let isMappingRunning = false;
let mappingIntervalId = null;
let mappingMeta = null;
const mappingDroneInstances = [];

// Mode Tab Elements
const tabRescue = document.getElementById('tab-rescue');
const tabMapping = document.getElementById('tab-mapping');
const rescueControls = document.getElementById('rescue-controls');
const mappingControls = document.getElementById('mapping-controls');
const sidebar = document.getElementById('sidebar');
const sidebarMapping = document.getElementById('sidebar-mapping');
const btnStartMapping = document.getElementById('btn-start-mapping');
const btnRestartMapping = document.getElementById('btn-restart-mapping');
const mappingVerifyModal = document.getElementById('mapping-verify-modal');
const mappingAgentList = document.getElementById('mapping-agent-list');

// Mapping stat elements
const statScanPct = document.getElementById('stat-scan-pct');
const statMapStep = document.getElementById('stat-map-step');

function switchMode(mode) {
  currentMode = mode;

  // Update tabs
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
  if (mode === 'rescue') {
    tabRescue.classList.add('active');
    rescueControls.classList.remove('hidden');
    mappingControls.classList.add('hidden');
    sidebar.classList.remove('hidden');
    sidebarMapping.classList.add('hidden');
    modeBadge.innerText = 'LIVE SIMULATION';
    modeBadge.style.background = 'rgba(8, 145, 178, 0.1)';
    modeBadge.style.color = '#0891b2';

    if (envGroup) envGroup.visible = true;
    personInstances.forEach(p => {
      if (p.mesh) p.mesh.visible = true;
      if (p.ring) p.ring.visible = true;
      if (p.beam) p.beam.visible = true;
      if (p.light) p.light.visible = true;
    });
    droneInstances.forEach(d => {
      if (d.mesh) d.mesh.visible = true;
      if (d.detCone) d.detCone.visible = true;
      if (d.groundRing) d.groundRing.visible = true;
      if (d.flashSphere) d.flashSphere.visible = true;
    });
    mappingDroneInstances.forEach(d => {
      if (d.mesh) d.mesh.visible = false;
      if (d.detCone) d.detCone.visible = false;
      if (d.groundShadow) d.groundShadow.visible = false;
      if (d.verticalTether) d.verticalTether.visible = false;
    });
    if (pointCloudGroup) pointCloudGroup.visible = false;
  } else {
    tabMapping.classList.add('active');
    rescueControls.classList.add('hidden');
    mappingControls.classList.remove('hidden');
    sidebar.classList.add('hidden');
    sidebarMapping.classList.remove('hidden');
    modeBadge.innerText = '3D AREA MAPPING';
    modeBadge.style.background = 'rgba(139, 92, 246, 0.15)';
    modeBadge.style.color = '#8b5cf6';

    // KEEP 3D CITY ENVIRONMENT VISIBLE DURING MAPPING
    if (envGroup) envGroup.visible = true;

    personInstances.forEach(p => {
      if (p.mesh) p.mesh.visible = false;
      if (p.ring) p.ring.visible = false;
      if (p.beam) p.beam.visible = false;
      if (p.light) p.light.visible = false;
    });
    droneInstances.forEach(d => {
      if (d.mesh) d.mesh.visible = false;
      if (d.detCone) d.detCone.visible = false;
      if (d.groundRing) d.groundRing.visible = false;
      if (d.flashSphere) d.flashSphere.visible = false;
    });
    mappingDroneInstances.forEach(d => {
      if (d.mesh) d.mesh.visible = true;
      if (d.detCone) d.detCone.visible = true;
      if (d.groundShadow) d.groundShadow.visible = true;
      if (d.verticalTether) d.verticalTether.visible = true;
    });
    if (pointCloudGroup) pointCloudGroup.visible = true;
  }
}

// Tab click handlers
if (tabRescue) tabRescue.addEventListener('click', () => switchMode('rescue'));
if (tabMapping) tabMapping.addEventListener('click', () => switchMode('mapping'));

// ==============================================================================
// 3D AREA MAPPING MODE — Bioluminescent Spectral LiDAR Point Cloud Scan
// ==============================================================================
let pointCloudGroup = null;
let pointCloudPointsMesh = null;
const pointCloudPositions = [];
const pointCloudColors = [];
const mappingLidarRays = [];

function getLidarSpectralColor(y, typeId = 0) {
  const c = new THREE.Color();
  if (y < 0.6) {
    // Terrain / ground level: golden amber forest-floor luminescence
    c.setHSL(0.12, 0.95, 0.52);
  } else if (y < 6.0) {
    // Low architecture & vegetation: vibrant lime to emerald green
    const t = (y - 0.6) / 5.4;
    c.setHSL(0.26 + t * 0.12, 0.95, 0.55);
  } else if (y < 15.0) {
    // Mid structure & canopy: vivid cyan & electric blue
    const t = (y - 6.0) / 9.0;
    c.setHSL(0.50 + t * 0.08, 0.98, 0.58);
  } else {
    // High towers / rooftops / spires: fiery radiant orange & golden yellow
    const t = Math.min(1.0, (y - 15.0) / 12.0);
    c.setHSL(0.04 + (1.0 - t) * 0.06, 0.98, 0.60);
  }
  return c;
}

function clearPointCloud() {
  if (pointCloudGroup) {
    scene.remove(pointCloudGroup);
    pointCloudGroup.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
  }
  pointCloudGroup = new THREE.Group();
  scene.add(pointCloudGroup);

  pointCloudPositions.length = 0;
  pointCloudColors.length = 0;
  mappingLidarRays.forEach(r => scene.remove(r));
  mappingLidarRays.length = 0;
}

function addDiscovered3DPoints(hits) {
  if (!hits || hits.length === 0) return;
  if (!pointCloudGroup) clearPointCloud();

  hits.forEach(([hx, hy, hz, typeId]) => {
    // Add primary hit point
    pointCloudPositions.push(hx, hy, hz);
    const col = getLidarSpectralColor(hy, typeId);
    pointCloudColors.push(col.r, col.g, col.b);

    // Micro-scatter for dense LiDAR survey-grade point cloud effect
    const scatterCount = typeId > 0 ? 3 : 1;
    for (let s = 0; s < scatterCount; s++) {
      const rx = hx + (Math.random() - 0.5) * 0.8;
      const ry = Math.max(0.0, hy + (Math.random() - 0.5) * 0.5);
      const rz = hz + (Math.random() - 0.5) * 0.8;
      pointCloudPositions.push(rx, ry, rz);
      const sCol = getLidarSpectralColor(ry, typeId);
      pointCloudColors.push(sCol.r, sCol.g, sCol.b);
    }
  });

  // Rebuild main point cloud geometry
  if (pointCloudPointsMesh) {
    pointCloudGroup.remove(pointCloudPointsMesh);
    pointCloudPointsMesh.geometry.dispose();
    pointCloudPointsMesh.material.dispose();
  }

  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.Float32BufferAttribute(pointCloudPositions, 3));
  pGeo.setAttribute('color', new THREE.Float32BufferAttribute(pointCloudColors, 3));

  const pMat = new THREE.PointsMaterial({
    size: 2.4,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  pointCloudPointsMesh = new THREE.Points(pGeo, pMat);
  pointCloudGroup.add(pointCloudPointsMesh);
}

function setupMappingDrones(agentPositions, lidarRange = 22.0) {
  // Remove existing mapping drone instances
  mappingDroneInstances.forEach(d => {
    scene.remove(d.mesh);
    if (d.detCone) scene.remove(d.detCone);
    if (d.groundShadow) scene.remove(d.groundShadow);
    if (d.verticalTether) scene.remove(d.verticalTether);
  });
  mappingDroneInstances.length = 0;

  const numAgents = Object.keys(agentPositions).length;

  for (let i = 0; i < numAgents; i++) {
    const agentName = `agent_${i}`;
    const colorHex = AGENT_COLORS[i % AGENT_COLORS.length];
    const pos = agentPositions[agentName] || [50, 12, 50];

    const clone = droneTemplate.clone(true);
    const bBox = new THREE.Box3().setFromObject(clone);
    const sz = bBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(sz.x, sz.y, sz.z);
    if (maxDim > 0) clone.scale.setScalar(3.2 / maxDim);

    clone.traverse(child => {
      if (child.isMesh && child.material) {
        child.material = child.material.clone();
        child.material.color.setHex(colorHex);
      }
    });

    const worldX = pos[0];
    const worldY = pos[1];
    const worldZ = pos[2];
    clone.position.set(worldX, worldY, worldZ);
    scene.add(clone);

    // LiDAR cone beam
    const detGeo = new THREE.CylinderGeometry(0.3, lidarRange * 0.75, worldY, 32, 1, true);
    const detMat = new THREE.MeshBasicMaterial({
      color: colorHex, transparent: true, opacity: 0.15, side: THREE.DoubleSide, depthWrite: false
    });
    const detCone = new THREE.Mesh(detGeo, detMat);
    detCone.position.set(worldX, worldY / 2, worldZ);
    scene.add(detCone);

    // Ground Shadow
    const shadowGeo = new THREE.RingGeometry(0.0, 2.2, 32);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false
    });
    const groundShadow = new THREE.Mesh(shadowGeo, shadowMat);
    groundShadow.position.set(worldX, 0.05, worldZ);
    scene.add(groundShadow);

    // Vertical Tether
    const tetherGeo = new THREE.CylinderGeometry(0.12, 0.12, 1.0, 12, 1, true);
    const tetherMat = new THREE.MeshBasicMaterial({
      color: colorHex, transparent: true, opacity: 0.4, side: THREE.DoubleSide, depthWrite: false
    });
    const verticalTether = new THREE.Mesh(tetherGeo, tetherMat);
    verticalTether.position.set(worldX, worldY / 2, worldZ);
    scene.add(verticalTether);

    mappingDroneInstances.push({
      id: agentName, mesh: clone, detCone, groundShadow, verticalTether, colorHex,
      colorStr: AGENT_COLOR_STRS[i % AGENT_COLOR_STRS.length]
    });
  }

  // Update mapping sidebar list
  if (mappingAgentList) {
    mappingAgentList.innerHTML = '';
    mappingDroneInstances.forEach((drone, idx) => {
      const item = document.createElement('div');
      item.className = 'agent-item';
      item.innerHTML = `
        <div class="agent-dot" style="background:${drone.colorStr}; color:${drone.colorStr}"></div>
        <div class="agent-name">DRONE_${idx}</div>
        <div class="agent-pos" id="mpos-${drone.id}">[0, 0, 0]</div>
      `;
      mappingAgentList.appendChild(item);
    });
  }
}

function updateMappingDronePositions(agentPositions) {
  mappingDroneInstances.forEach(drone => {
    const pos = agentPositions[drone.id] || [50, 12, 50];
    const worldX = pos[0];
    const worldY = pos[1];
    const worldZ = pos[2];

    drone.mesh.position.set(worldX, worldY, worldZ);
    if (drone.detCone) {
      drone.detCone.position.set(worldX, worldY / 2, worldZ);
      drone.detCone.scale.set(1, worldY / 12.0, 1);
    }
    if (drone.groundShadow) {
      drone.groundShadow.position.set(worldX, 0.05, worldZ);
    }
    if (drone.verticalTether) {
      const tetherHeight = Math.max(0.1, worldY - 0.1);
      drone.verticalTether.scale.set(1, tetherHeight, 1);
      drone.verticalTether.position.set(worldX, 0.1 + tetherHeight / 2, worldZ);
    }

    const posLabel = document.getElementById(`mpos-${drone.id}`);
    if (posLabel) posLabel.innerText = `[${Math.round(worldX)}, ${Math.round(worldY)}, ${Math.round(worldZ)}]`;
  });
}

// Start 3D Area Mapping
async function startAreaMapping() {
  if (isMappingRunning) return;

  btnStartMapping.disabled = true;
  mappingVerifyModal.classList.remove('visible');

  try {
    const res = await fetch(`${API_BASE_URL}/area_mapping/start`, { method: 'POST' });
    if (!res.ok) throw new Error('Start 3D mapping failed');
    const data = await res.json();

    mappingMeta = data.metadata;
    isMappingRunning = true;

    modeBadge.innerText = '3D MAPPING — LIVE SCANNING';
    modeBadge.style.background = 'rgba(139, 92, 246, 0.15)';
    modeBadge.style.color = '#8b5cf6';

    // Clear and reset 3D Point Cloud
    clearPointCloud();

    // Setup 3D drones
    setupMappingDrones(data.agent_positions, mappingMeta.lidar_3d_range || 22.0);

    // Render initial point cloud hits
    if (data.initial_point_cloud) {
      addDiscovered3DPoints(data.initial_point_cloud);
    }

    setTxt(statScanPct, `${data.scanned_pct}%`);
    setTxt(document.getElementById('stat-3d-pts'), `${(data.initial_point_cloud || []).length}`);
    setTxt(statMapStep, `0 (100% Target)`);
    setTxt(hudStep, `3D MAPPING STEP 0 (Target: 100% Mapped)`);

    showToast('3D AREA MAPPING STARTED — 3D LiDAR raycasting active', 'survivor');

    // Poll /area_mapping/step
    mappingIntervalId = setInterval(executeMappingStep, 250);
  } catch (err) {
    console.error('Failed to start 3D mapping:', err);
    showToast('Failed to connect to Area Mapping API', 'collision');
    btnStartMapping.disabled = false;
    isMappingRunning = false;
  }
}

let mappingErrorCount = 0;

async function executeMappingStep() {
  try {
    const res = await fetch(`${API_BASE_URL}/area_mapping/step`, { method: 'POST' });
    if (!res.ok) throw new Error('Mapping step failed');
    const data = await res.json();
    mappingErrorCount = 0;

    // Update 3D drone positions
    updateMappingDronePositions(data.agent_positions);

    // Append newly discovered 3D hit points to Point Cloud
    if (data.newly_discovered_points) {
      addDiscovered3DPoints(data.newly_discovered_points);
    }

    // Update HUD stats
    setTxt(statScanPct, `${data.scanned_pct}%`);
    setTxt(document.getElementById('stat-3d-pts'), `${data.total_point_cloud_count}`);
    setTxt(statMapStep, `${data.step} (100% Target)`);
    setTxt(hudStep, `3D MAPPING STEP ${data.step} (Target: 100% Mapped / Safety: ${mappingMeta.max_steps})`);

    // Termination check
    if (data.is_done) {
      clearInterval(mappingIntervalId);
      isMappingRunning = false;
      btnStartMapping.disabled = false;
      showToast(`3D MAPPING COMPLETE — ${data.total_point_cloud_count} points captured (${data.scanned_pct}% surface coverage)`, 'survivor');
      // Run 3D verification
      await showMappingVerification();
    }
  } catch (err) {
    mappingErrorCount++;
    if (mappingErrorCount >= 3) {
      clearInterval(mappingIntervalId);
      isMappingRunning = false;
      btnStartMapping.disabled = false;
      showToast('3D Mapping step failed after 3 retries', 'collision');
    }
  }
}

// ==============================================================================
// DEDICATED INTERACTIVE 3D MAP VIEWER MANAGER (WebGL + OrbitControls)
// ==============================================================================
let viewer3DScene = null;
let viewer3DCamera = null;
let viewer3DRenderer = null;
let viewer3DControls = null;
let viewer3DPointCloudMesh = null;
let viewer3DVoxelGroup = null;
let viewer3DGTGroup = null;
let viewer3DPointsData = [];
let viewer3DGTData = [];
let viewer3DColorMode = 'height'; // 'height' | 'type'
let viewer3DShowGT = true;
let viewer3DShowVoxels = false;
let isViewer3DInitialized = false;

function init3DMapViewer() {
  const container = document.getElementById('viewer-3d-container');
  const canvas = document.getElementById('canvas-3d-viewer');
  if (!container || !canvas) return;

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 440;

  viewer3DScene = new THREE.Scene();
  viewer3DScene.background = new THREE.Color(0x090d16);

  // 3D Grid helper & Axes
  const gridHelper = new THREE.GridHelper(100, 20, 0x38bdf8, 0x1e293b);
  gridHelper.position.set(50, 0, 50);
  viewer3DScene.add(gridHelper);

  // Lighting
  const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
  viewer3DScene.add(ambLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight.position.set(80, 120, 80);
  viewer3DScene.add(dirLight);

  // Camera
  viewer3DCamera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  viewer3DCamera.position.set(135, 85, 135);

  // WebGL Renderer
  viewer3DRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  viewer3DRenderer.setSize(width, height);
  viewer3DRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // OrbitControls for free rotation, zoom, pan
  viewer3DControls = new OrbitControls(viewer3DCamera, viewer3DRenderer.domElement);
  viewer3DControls.target.set(50, 10, 50);
  viewer3DControls.enableDamping = true;
  viewer3DControls.dampingFactor = 0.05;
  viewer3DControls.maxPolarAngle = Math.PI / 2 + 0.1; // allow viewing ground
  viewer3DControls.update();

  // Resize handler
  window.addEventListener('resize', () => {
    if (!container || !viewer3DRenderer || !viewer3DCamera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w > 0 && h > 0) {
      viewer3DCamera.aspect = w / h;
      viewer3DCamera.updateProjectionMatrix();
      viewer3DRenderer.setSize(w, h);
    }
  });

  // Setup toolbar event listeners
  setupViewerToolbarEvents();

  // Animation Loop (ONLY runs when modal is active)
  let isViewerAnimating = false;
  function animateViewer() {
    if (!mappingVerifyModal || !mappingVerifyModal.classList.contains('visible')) {
      isViewerAnimating = false;
      return;
    }
    isViewerAnimating = true;
    requestAnimationFrame(animateViewer);
    if (viewer3DControls) viewer3DControls.update();
    if (viewer3DRenderer && viewer3DScene && viewer3DCamera) {
      viewer3DRenderer.render(viewer3DScene, viewer3DCamera);
    }
  }
  window._startViewerAnim = () => {
    if (!isViewerAnimating) animateViewer();
  };

  isViewer3DInitialized = true;
}

function setupViewerToolbarEvents() {
  const btn3D = document.getElementById('v3d-view-3d');
  const btnTop = document.getElementById('v3d-view-top');
  const btnFront = document.getElementById('v3d-view-front');
  const btnSide = document.getElementById('v3d-view-side');

  const btnColorHeight = document.getElementById('v3d-color-height');
  const btnColorType = document.getElementById('v3d-color-type');

  const btnToggleGT = document.getElementById('v3d-toggle-gt');
  const btnToggleVoxels = document.getElementById('v3d-toggle-voxels');
  const btnResetCam = document.getElementById('v3d-reset-cam');

  function setActiveCamBtn(activeBtn) {
    [btn3D, btnTop, btnFront, btnSide].forEach(b => b && b.classList.remove('active'));
    if (activeBtn) activeBtn.classList.add('active');
  }

  if (btn3D) btn3D.addEventListener('click', () => {
    setActiveCamBtn(btn3D);
    viewer3DCamera.position.set(135, 85, 135);
    viewer3DControls.target.set(50, 10, 50);
    viewer3DControls.update();
  });

  if (btnTop) btnTop.addEventListener('click', () => {
    setActiveCamBtn(btnTop);
    viewer3DCamera.position.set(50, 160, 50.1);
    viewer3DControls.target.set(50, 0, 50);
    viewer3DControls.update();
  });

  if (btnFront) btnFront.addEventListener('click', () => {
    setActiveCamBtn(btnFront);
    viewer3DCamera.position.set(50, 20, 160);
    viewer3DControls.target.set(50, 10, 50);
    viewer3DControls.update();
  });

  if (btnSide) btnSide.addEventListener('click', () => {
    setActiveCamBtn(btnSide);
    viewer3DCamera.position.set(160, 20, 50);
    viewer3DControls.target.set(50, 10, 50);
    viewer3DControls.update();
  });

  if (btnColorHeight) btnColorHeight.addEventListener('click', () => {
    btnColorHeight.classList.add('active');
    if (btnColorType) btnColorType.classList.remove('active');
    viewer3DColorMode = 'height';
    render3DViewerPointCloud();
  });

  if (btnColorType) btnColorType.addEventListener('click', () => {
    btnColorType.classList.add('active');
    if (btnColorHeight) btnColorHeight.classList.remove('active');
    viewer3DColorMode = 'type';
    render3DViewerPointCloud();
  });

  if (btnToggleGT) btnToggleGT.addEventListener('click', () => {
    viewer3DShowGT = !viewer3DShowGT;
    btnToggleGT.classList.toggle('active', viewer3DShowGT);
    render3DViewerGroundTruth();
  });

  if (btnToggleVoxels) btnToggleVoxels.addEventListener('click', () => {
    viewer3DShowVoxels = !viewer3DShowVoxels;
    btnToggleVoxels.classList.toggle('active', viewer3DShowVoxels);
    render3DViewerVoxels();
  });

  if (btnResetCam) btnResetCam.addEventListener('click', () => {
    setActiveCamBtn(btn3D);
    viewer3DCamera.position.set(135, 85, 135);
    viewer3DControls.target.set(50, 10, 50);
    viewer3DControls.update();
  });
}

function update3DViewerData(pointCloud, gtObjects) {
  if (!isViewer3DInitialized) {
    init3DMapViewer();
  }
  viewer3DPointsData = pointCloud || [];
  viewer3DGTData = gtObjects || [];

  render3DViewerPointCloud();
  render3DViewerGroundTruth();
  render3DViewerVoxels();

  if (window._startViewerAnim) window._startViewerAnim();

  setTimeout(() => {
    const container = document.getElementById('viewer-3d-container');
    if (container && viewer3DRenderer && viewer3DCamera) {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0) {
        viewer3DCamera.aspect = w / h;
        viewer3DCamera.updateProjectionMatrix();
        viewer3DRenderer.setSize(w, h);
      }
    }
  }, 50);
}

function render3DViewerPointCloud() {
  if (!viewer3DScene) return;
  if (viewer3DPointCloudMesh) {
    viewer3DScene.remove(viewer3DPointCloudMesh);
    viewer3DPointCloudMesh.geometry.dispose();
    viewer3DPointCloudMesh.material.dispose();
    viewer3DPointCloudMesh = null;
  }

  if (!viewer3DPointsData || viewer3DPointsData.length === 0) return;

  const positions = [];
  const colors = [];

  viewer3DPointsData.forEach(([px, py, pz, typeId]) => {
    positions.push(px, py, pz);
    let col = (viewer3DColorMode === 'height')
      ? getLidarSpectralColor(py, typeId)
      : (MAP_TYPE_COLORS[typeId] || MAP_TYPE_COLORS[0]);
    colors.push(col.r, col.g, col.b);

    // Micro-scatter for dense survey point cloud inspection
    const scatterCount = typeId > 0 ? 3 : 1;
    for (let s = 0; s < scatterCount; s++) {
      const rx = px + (Math.random() - 0.5) * 0.8;
      const ry = Math.max(0.0, py + (Math.random() - 0.5) * 0.5);
      const rz = pz + (Math.random() - 0.5) * 0.8;
      positions.push(rx, ry, rz);
      const sCol = (viewer3DColorMode === 'height')
        ? getLidarSpectralColor(ry, typeId)
        : (MAP_TYPE_COLORS[typeId] || MAP_TYPE_COLORS[0]);
      colors.push(sCol.r, sCol.g, sCol.b);
    }
  });

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 2.8,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  viewer3DPointCloudMesh = new THREE.Points(geo, mat);
  viewer3DScene.add(viewer3DPointCloudMesh);
}

function render3DViewerGroundTruth() {
  if (!viewer3DScene) return;
  if (viewer3DGTGroup) {
    viewer3DScene.remove(viewer3DGTGroup);
    viewer3DGTGroup.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    viewer3DGTGroup = null;
  }

  if (!viewer3DShowGT || !viewer3DGTData || viewer3DGTData.length === 0) return;

  viewer3DGTGroup = new THREE.Group();

  const gtColors = {
    "building": 0x38bdf8,
    "tower": 0xf59e0b,
    "ruined_block": 0xec4899,
    "wall": 0xa855f7
  };

  viewer3DGTData.forEach(obj => {
    const [xmin, ymin, zmin, xmax, ymax, zmax] = obj.bounds;
    const dx = Math.max(0.5, xmax - xmin);
    const dy = Math.max(0.5, ymax - ymin);
    const dz = Math.max(0.5, zmax - zmin);
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;
    const cz = (zmin + zmax) / 2;

    const colorHex = gtColors[obj.type] || 0x38bdf8;

    // Solid semi-transparent bounding box
    const boxGeo = new THREE.BoxGeometry(dx, dy, dz);
    const boxMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.22,
      roughness: 0.3
    });
    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    boxMesh.position.set(cx, cy, cz);
    viewer3DGTGroup.add(boxMesh);

    // Wireframe outline
    const wfGeo = new THREE.WireframeGeometry(boxGeo);
    const wfMat = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 2 });
    const wfLine = new THREE.LineSegments(wfGeo, wfMat);
    wfLine.position.set(cx, cy, cz);
    viewer3DGTGroup.add(wfLine);
  });

  viewer3DScene.add(viewer3DGTGroup);
}

function render3DViewerVoxels() {
  if (!viewer3DScene) return;
  if (viewer3DVoxelGroup) {
    viewer3DScene.remove(viewer3DVoxelGroup);
    viewer3DVoxelGroup.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    viewer3DVoxelGroup = null;
  }

  if (!viewer3DShowVoxels || !viewer3DPointsData || viewer3DPointsData.length === 0) return;

  viewer3DVoxelGroup = new THREE.Group();

  const voxelMap = new Map();
  viewer3DPointsData.forEach(([px, py, pz, typeId]) => {
    if (py < 0.5) return; // skip ground
    const vx = Math.floor(px / 2.0) * 2.0 + 1.0;
    const vy = Math.floor(py / 2.0) * 2.0 + 1.0;
    const vz = Math.floor(pz / 2.0) * 2.0 + 1.0;
    const key = `${vx}_${vy}_${vz}`;
    if (!voxelMap.has(key)) {
      voxelMap.set(key, { vx, vy, vz, typeId });
    }
  });

  if (voxelMap.size === 0) return;

  const boxGeo = new THREE.BoxGeometry(1.85, 1.85, 1.85);
  const voxMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.55,
    roughness: 0.4
  });

  const instancedMesh = new THREE.InstancedMesh(boxGeo, voxMat, voxelMap.size);
  const dummy = new THREE.Object3D();
  let idx = 0;
  voxelMap.forEach(({ vx, vy, vz }) => {
    dummy.position.set(vx, vy, vz);
    dummy.updateMatrix();
    instancedMesh.setMatrixAt(idx++, dummy.matrix);
  });
  instancedMesh.instanceMatrix.needsUpdate = true;
  viewer3DVoxelGroup.add(instancedMesh);

  viewer3DScene.add(viewer3DVoxelGroup);
}

async function showMappingVerification() {
  try {
    const res = await fetch(`${API_BASE_URL}/area_mapping/verify`);
    if (!res.ok) throw new Error('3D Verify failed');
    const data = await res.json();

    setTxt(document.getElementById('verify-accuracy'), `${data.accuracy_pct}%`);
    setTxt(document.getElementById('verify-coverage'), `${data.scan_coverage_pct}%`);
    setTxt(document.getElementById('verify-steps'), data.total_steps);
    setTxt(document.getElementById('verify-points'), data.total_discovered_points);
    setTxt(document.getElementById('verify-matched'), data.matched_surface_points);
    setTxt(document.getElementById('verify-total-gt'), data.total_gt_surface_points);

    // Update Interactive WebGL 3D Point-Cloud Inspector
    update3DViewerData(data.discovered_point_cloud || [], data.ground_truth_3d_objects || []);

    mappingVerifyModal.classList.add('visible');
  } catch (err) {
    console.error('3D Verification failed:', err);
    showToast('Failed to fetch verification results', 'collision');
  }
}

// Open Inspector manually button
const btnOpen3DViewer = document.getElementById('btn-open-3d-viewer');
if (btnOpen3DViewer) {
  btnOpen3DViewer.addEventListener('click', showMappingVerification);
}

// Close Inspector buttons
const btnCloseModal = document.getElementById('btn-close-mapping-modal');
const btnInspectClose = document.getElementById('btn-inspect-3d-map-close');
if (btnCloseModal) btnCloseModal.addEventListener('click', () => mappingVerifyModal.classList.remove('visible'));
if (btnInspectClose) btnInspectClose.addEventListener('click', () => mappingVerifyModal.classList.remove('visible'));

// Event listeners for mapping mode
if (btnStartMapping) btnStartMapping.addEventListener('click', startAreaMapping);
if (btnRestartMapping) btnRestartMapping.addEventListener('click', () => {
  mappingVerifyModal.classList.remove('visible');
  fetch(`${API_BASE_URL}/area_mapping/reset`, { method: 'POST' });
  startAreaMapping();
});

init();

