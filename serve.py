"""
FastAPI Server for LIVE Multi-Agent Search & Rescue Simulation (V15 PPO Model)
"""
import os
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except Exception as _sb3_err:
    PPO = None
    HAS_SB3 = False

from env.search_rescue_env_v15 import (
    SearchAndRescueEnvV15,
    GRID_SIZE, NUM_AGENTS, NUM_SURVIVORS, NUM_OBSTACLES, NUM_BUILDINGS, MAX_STEPS,
    DETECTION_RADIUS_OPEN, DETECTION_RADIUS_THERMAL_OCCLUDED
)
PERSONAL_SPACE_RADIUS = 2.5

MODEL_PATH = "models/ppo_FINAL_BEST_v15.zip" if os.path.exists("models/ppo_FINAL_BEST_v15.zip") else "models/ppo_FINAL_BEST.zip"
AM_V2_MODEL_PATH = "models/ppo_area_mapping_v2_2000000_steps.zip" if os.path.exists("models/ppo_area_mapping_v2_2000000_steps.zip") else "models/ppo_area_mapping_v2_BEST.zip"
AM_V3_MODEL_PATH = "models/ppo_area_mapping_v3_BEST.zip" if os.path.exists("models/ppo_area_mapping_v3_BEST.zip") else "models/ppo_area_mapping_v3_final.zip"
CELL_SIZE_METERS = 10.0

from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(title="VIHANG Live Simulation API (V15)", version="1.5.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_smart_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    
    # 3D models, textures, scripts, and stylesheets benefit from high-speed disk caching
    cacheable_extensions = (
        '.glb', '.gltf', '.bin', '.png', '.jpg', '.jpeg', '.webp', '.svg',
        '.js', '.css', '.woff2', '.woff', '.ttf', '.json'
    )
    if any(path.endswith(ext) for ext in cacheable_extensions) or path.startswith('/assets/') or path.startswith('/models/'):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        # Dynamic API routes and index HTML remain fresh
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
    return response

class ServerState:
    def __init__(self):
        self.model = None
        self.env = None
        self.obs = None
        self.agent_names = []
        self.is_active = False
        self.custom_survivor_layout = None
        self.prev_survivors = set()
        self.survivors_open_initial = set()
        self.survivors_hidden_initial = set()
        self.last_killed_drone = None

state = ServerState()


def load_model_if_needed():
    if state.model is None and HAS_SB3 and PPO is not None:
        if os.path.exists(MODEL_PATH):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                state.model = PPO.load(MODEL_PATH, device=device)
            except Exception as e:
                print(f"Warning: could not load model checkpoint ({e})")


@app.on_event("startup")
def startup_event():
    import threading
    threading.Thread(target=load_model_if_needed, daemon=True).start()


@app.post("/randomize")
@app.post("/randomize_humans")
def randomize_survivors():
    """Generates a new random 10-survivor layout (6-7 open, 3-4 building-hidden)."""
    env_temp = SearchAndRescueEnvV15(failure_injection_prob=0.0)
    env_temp.reset()

    survivors_list = [list(s) for s in env_temp.survivors]
    hidden_list = [list(s) for s in env_temp.survivors_hidden]
    open_list = [list(s) for s in env_temp.survivors_open]

    state.custom_survivor_layout = {
        "survivors": survivors_list,
        "hidden_survivors": hidden_list,
        "open_survivors": open_list
    }

    return {
        "status": "success",
        "survivor_positions": survivors_list,
        "hidden_survivors": hidden_list,
        "open_survivors": open_list
    }


@app.post("/start")
@app.post("/start_live")
def start_simulation():
    """Initializes a live simulation run using the trained V15 policy."""
    load_model_if_needed()

    state.env = SearchAndRescueEnvV15(failure_injection_prob=0.0)
    state.obs, _ = state.env.reset()
    state.agent_names = list(state.env.possible_agents)
    state.last_killed_drone = None

    if state.custom_survivor_layout:
        surv_set = set(tuple(x) for x in state.custom_survivor_layout["survivors"])
        hidden_set = set(tuple(x) for x in state.custom_survivor_layout["hidden_survivors"])
        open_set = set(tuple(x) for x in state.custom_survivor_layout["open_survivors"])

        state.env.survivors = surv_set
        state.env.survivors_hidden = hidden_set
        state.env.survivors_open = open_set

        if hasattr(state.env, "survivor_grid"):
            state.env.survivor_grid.fill(0.0)
            for r, c in surv_set:
                state.env.survivor_grid[r, c] = 1.0

        state.obs = {a: state.env._get_obs(a) for a in state.env.agents}

    state.is_active = True
    state.prev_survivors = set(state.env.survivors)
    state.survivors_open_initial = set(state.env.survivors_open)
    state.survivors_hidden_initial = set(state.env.survivors_hidden)

    agent_zones_serializable = {a: list(state.env.agent_zones[a]) for a in state.agent_names}

    initial_state_data = {
        "agent_positions": {a: list(state.env.agent_positions[a]) for a in state.agent_names},
        "agent_altitudes": {a: int(getattr(state.env, "agent_altitudes", {}).get(a, 0)) for a in state.agent_names},
        "agent_zones": agent_zones_serializable,
        "survivor_positions": [list(s) for s in state.env.survivors],
        "hidden_survivors": [list(s) for s in state.env.survivors_hidden],
        "open_survivors": [list(s) for s in state.env.survivors_open],
        "obstacle_positions": [list(o) for o in state.env.obstacles],
        "building_positions": [list(b) for b in state.env.buildings]
    }

    metadata = {
        "grid_size": GRID_SIZE,
        "cell_size_meters": CELL_SIZE_METERS,
        "num_agents": NUM_AGENTS,
        "num_survivors": NUM_SURVIVORS,
        "num_obstacles": NUM_OBSTACLES,
        "num_buildings": NUM_BUILDINGS,
        "max_steps": MAX_STEPS,
        "detection_radius_open": DETECTION_RADIUS_OPEN,
        "detection_radius_thermal": DETECTION_RADIUS_THERMAL_OCCLUDED,
        "personal_space_radius": PERSONAL_SPACE_RADIUS
    }

    step_0_snapshot = {
        "step": 0,
        "agent_positions": {a: list(state.env.agent_positions[a]) for a in state.agent_names},
        "survivors_remaining": [list(s) for s in state.env.survivors],
        "open_rescued": 0,
        "open_total": len(state.survivors_open_initial),
        "hidden_rescued": 0,
        "hidden_total": len(state.survivors_hidden_initial),
        "coverage_pct": round(float(state.env.visited_grid.sum() / (GRID_SIZE ** 2) * 100.0), 2),
        "events": [],
        "is_done": False
    }

    return {
        "status": "simulation_started",
        "metadata": metadata,
        "initial_state": initial_state_data,
        "step": step_0_snapshot
    }


@app.post("/kill_drone")
def kill_drone(payload: dict = None, drone_id: str = None):
    """Triggers manual self-destruct/kill-switch for a specified drone during live run."""
    if not state.is_active or state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not active.")

    target = drone_id
    if not target and payload:
        target = payload.get("drone_id") or payload.get("agent") or payload.get("drone")

    if not target or target not in state.agent_names:
        raise HTTPException(status_code=400, detail=f"Invalid drone_id: {target}. Must be one of {state.agent_names}")

    if hasattr(state.env, "trigger_failure"):
        state.env.trigger_failure(target)
    else:
        state.env.failed_agents.add(target)

    state.last_killed_drone = target

    reassigned = state.env.reassigned_agents.get(target, []) if hasattr(state.env, "reassigned_agents") else []
    failed_zone = list(state.env.agent_zones.get(target, [0, 0])) if hasattr(state.env, "agent_zones") else [0, 0]

    return {
        "status": "drone_killed",
        "killed_drone": target,
        "reassigned_drones": reassigned,
        "failed_zone_bounds": failed_zone
    }


@app.post("/step")
def step_simulation():
    """Advances simulation by 1 step using live V15 policy inference."""
    if not state.is_active or state.env is None:
        raise HTTPException(status_code=400, detail="Simulation not started. Call /start first.")

    total_walkable = len(getattr(state.env, 'total_walkable_cells', set())) or (GRID_SIZE ** 2)
    visited_walkable = sum(1 for c in getattr(state.env, 'total_walkable_cells', set()) if state.env.visited_grid[c[0], c[1]] == 1.0) if hasattr(state.env, 'total_walkable_cells') else state.env.visited_grid.sum()
    calc_coverage = round(float(visited_walkable / float(total_walkable) * 100.0), 2)

    if not state.env.agents or len(state.env.agents) == len(getattr(state.env, 'failed_agents', set())):
        open_rescued = len(state.survivors_open_initial - state.env.survivors_open) if hasattr(state.env, 'survivors_open') else 0
        hidden_rescued = len(state.survivors_hidden_initial - state.env.survivors_hidden) if hasattr(state.env, 'survivors_hidden') else 0
        failed_agents = list(getattr(state.env, 'failed_agents', set()))
        failed_recov = {}
        zwc = getattr(state.env, 'zone_walkable_cells', {})
        zvc = getattr(state.env, 'zone_visited_cells', {})
        for fa in failed_agents:
            w = len(zwc.get(fa, set()))
            v = len(zvc.get(fa, set()))
            failed_recov[fa] = round((v / max(1, w)) * 100.0, 1)

        return {
            "status": "episode_complete",
            "is_done": True,
            "step": {
                "step": state.env.step_count,
                "agent_positions": {a: list(state.env.agent_positions.get(a, [0, 0])) for a in state.agent_names},
                "survivors_remaining": [],
                "open_rescued": open_rescued,
                "open_total": len(state.survivors_open_initial),
                "hidden_rescued": hidden_rescued,
                "hidden_total": len(state.survivors_hidden_initial),
                "coverage_pct": calc_coverage,
                "events": [],
                "failed_agents": failed_agents,
                "reassigned_agents": getattr(state.env, 'reassigned_agents', {}),
                "failed_zone_recovery": failed_recov,
                "kill_event_occurred": len(failed_agents) > 0,
                "is_done": True
            }
        }

    # 0. Active agents and failed agents state before step
    failed_before = set(getattr(state.env, 'failed_agents', set()))
    active_agents = [a for a in state.agent_names if a not in failed_before]

    # 1. Live policy action inference + active search for unsearched areas
    actions = {}
    visited_grid = state.env.visited_grid
    buildings = getattr(state.env, "buildings", set())
    unsearched_cells = [
        (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)
        if visited_grid[r, c] == 0.0 and (r, c) not in state.env.obstacles and (r, c) not in buildings
    ]

    for agent in state.agent_names:
        if agent in failed_before:
            actions[agent] = 0
            continue

        if state.model is not None:
            act, _ = state.model.predict(state.obs[agent], deterministic=True)
            act = int(act)
        else:
            act = 0
        r, c = state.env.agent_positions[agent]
        alt = getattr(state.env, "agent_altitudes", {}).get(agent, 0)
        history = getattr(state.env, "agent_histories", {}).get(agent, [])
        is_stuck = len(history) >= 4 and len(set(history[-4:])) <= 2

        # Active search controller: if drone is idle (act in 0,5,6), stuck, or needs to target unsearched area
        if (act in (0, 5, 6) or is_stuck) and unsearched_cells:
            z_start, z_end = state.env.agent_zones.get(agent, (0, GRID_SIZE))
            zone_unsearched = [uc for uc in unsearched_cells if z_start <= uc[0] < z_end and (uc[0], uc[1]) != (r, c)]
            target_candidates = zone_unsearched if zone_unsearched else [uc for uc in unsearched_cells if (uc[0], uc[1]) != (r, c)]

            if target_candidates:
                best_target = min(target_candidates, key=lambda uc: abs(uc[0] - r) + abs(uc[1] - c))
                tr, tc = best_target

                desired_act = 0
                if tr < r: desired_act = 1
                elif tr > r: desired_act = 2
                elif tc < c: desired_act = 3
                elif tc > c: desired_act = 4

                if desired_act in (1, 2, 3, 4):
                    dr = -1 if desired_act == 1 else (1 if desired_act == 2 else 0)
                    dc = -1 if desired_act == 3 else (1 if desired_act == 4 else 0)
                    nr, nc = r + dr, c + dc
                    b_height = getattr(state.env, "obstacle_heights", {}).get((nr, nc), 0)
                    if b_height > 0 and alt < b_height:
                        desired_act = 5 # Flyover building
                    else:
                        # Inter-drone 3D collision check: if another active drone occupies (nr, nc) at same altitude, ascend!
                        for other_a in state.agent_names:
                            if other_a != agent and other_a not in failed_before:
                                other_pos = state.env.agent_positions.get(other_a)
                                other_alt = getattr(state.env, "agent_altitudes", {}).get(other_a, 0)
                                if other_pos == (nr, nc) and other_alt == alt and alt < 2:
                                    desired_act = 5 # Ascend to flyover layer above teammate

                act = desired_act

        actions[agent] = act

    # 2. Track previous altitudes & positions for active agents
    prev_altitudes = {a: int(getattr(state.env, "agent_altitudes", {}).get(a, 0)) for a in state.agent_names}

    # 3. Step env
    state.obs, rewards, terms, truncs, infos = state.env.step(actions)

    # 4. Record events
    events = []

    # Check for freshly failed / destroyed drones on this exact step
    failed_after = set(getattr(state.env, 'failed_agents', set()))
    freshly_failed = failed_after - failed_before

    for fa in freshly_failed:
        reassigned = state.env.reassigned_agents.get(fa, []) if hasattr(state.env, 'reassigned_agents') else []
        zone_bounds = list(state.env.agent_zones.get(fa, [0, 0])) if hasattr(state.env, 'agent_zones') else [0, 0]
        events.append({
            "type": "collision",
            "agent": fa,
            "position": list(state.env.agent_positions.get(fa, [0, 0]))
        })
        events.append({
            "type": "drone_failed",
            "agent": fa,
            "reassigned_to": reassigned,
            "zone": zone_bounds
        })

    # Detect collision-avoidance altitude changes for surviving active drones
    for a in active_agents:
        if a not in failed_after:
            curr_alt = int(getattr(state.env, "agent_altitudes", {}).get(a, 0))
            prev_alt = prev_altitudes.get(a, 0)
            if curr_alt != prev_alt:
                is_ascend = curr_alt > prev_alt
                pos_a = np.array(state.env.agent_positions.get(a, [0, 0]))

                near_drone = False
                for other_a in active_agents:
                    if other_a != a and other_a not in failed_after:
                        pos_other = np.array(state.env.agent_positions.get(other_a, [0, 0]))
                        if np.linalg.norm(pos_a - pos_other) <= 3.0:
                            near_drone = True
                            break

                r_a, c_a = state.env.agent_positions.get(a, (0, 0))
                near_obstacle = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r_a + dr, c_a + dc
                        if getattr(state.env, "obstacle_heights", {}).get((nr, nc), 0) > 0:
                            near_obstacle = True
                            break
                    if near_obstacle:
                        break

                reason = "inter_drone" if near_drone else ("obstacle" if near_obstacle else "tactical_elevation")
                events.append({
                    "type": "elevation_avoidance",
                    "agent": a,
                    "direction": "ascend" if is_ascend else "descend",
                    "from_alt": prev_alt,
                    "to_alt": curr_alt,
                    "reason": reason,
                    "position": list(state.env.agent_positions.get(a, [0, 0]))
                })

    if state.last_killed_drone and state.last_killed_drone not in freshly_failed:
        fa = state.last_killed_drone
        reassigned = state.env.reassigned_agents.get(fa, []) if hasattr(state.env, 'reassigned_agents') else []
        zone_bounds = list(state.env.agent_zones.get(fa, [0, 0])) if hasattr(state.env, 'agent_zones') else [0, 0]
        events.append({
            "type": "drone_failed",
            "agent": fa,
            "reassigned_to": reassigned,
            "zone": zone_bounds
        })
        state.last_killed_drone = None

    curr_survivors = set(state.env.survivors) if hasattr(state.env, 'survivors') else set()
    found = state.prev_survivors - curr_survivors

    for s in found:
        finder = None
        for a in state.agent_names:
            if a in state.env.agent_positions:
                d = np.linalg.norm(np.array(state.env.agent_positions[a]) - np.array(s))
                thresh = DETECTION_RADIUS_THERMAL_OCCLUDED if s in state.survivors_hidden_initial else DETECTION_RADIUS_OPEN
                if d <= thresh + 0.5:
                    finder = a
                    break
        events.append({
            "type": "survivor_found",
            "survivor_position": list(s),
            "is_building_hidden": s in state.survivors_hidden_initial,
            "found_by": finder or "unknown"
        })

    coverage = round(float(state.env.visited_grid.sum() / (GRID_SIZE ** 2) * 100.0), 2)
    is_done = len(state.env.agents) == 0 or state.env.step_count >= state.env.max_steps or coverage >= 100.0 or any(truncs.values())

    open_rescued = len(state.survivors_open_initial - state.env.survivors_open)
    hidden_rescued = len(state.survivors_hidden_initial - state.env.survivors_hidden)
    coverage = round(float(state.env.visited_grid.sum() / (GRID_SIZE ** 2) * 100.0), 2)

    failed_agents = list(getattr(state.env, 'failed_agents', set()))
    failed_recov = {}
    zwc = getattr(state.env, 'zone_walkable_cells', {})
    zvc = getattr(state.env, 'zone_visited_cells', {})
    for fa in failed_agents:
        w = len(zwc.get(fa, set()))
        v = len(zvc.get(fa, set()))
        failed_recov[fa] = round((v / max(1, w)) * 100.0, 1)

    step_snapshot = {
        "step": state.env.step_count,
        "agent_positions": {a: list(state.env.agent_positions.get(a, [0, 0])) for a in state.agent_names},
        "agent_altitudes": {a: int(getattr(state.env, "agent_altitudes", {}).get(a, 0)) for a in state.agent_names},
        "survivors_remaining": [list(s) for s in curr_survivors],
        "open_rescued": open_rescued,
        "open_total": len(state.survivors_open_initial),
        "hidden_rescued": hidden_rescued,
        "hidden_total": len(state.survivors_hidden_initial),
        "coverage_pct": coverage,
        "events": events,
        "failed_agents": failed_agents,
        "reassigned_agents": getattr(state.env, 'reassigned_agents', {}),
        "failed_zones": {fa: list(state.env.agent_zones[fa]) for fa in failed_agents if fa in state.env.agent_zones},
        "failed_zone_recovery": failed_recov,
        "kill_event_occurred": len(failed_agents) > 0,
        "is_done": is_done
    }

    return {
        "status": "step_executed",
        "is_done": is_done,
        "step": step_snapshot
    }


@app.post("/reset")
def reset_simulation():
    """Resets server-side simulation state."""
    state.env = None
    state.obs = None
    state.is_active = False
    state.custom_survivor_layout = None
    state.prev_survivors.clear()

    return {"status": "simulation_reset"}


# ==============================================================================
# ==============================================================================
# 3D AREA MAPPING MODE — Full 3D Environment Scanning & Point Cloud Reconstruction
#
# SENSOR CHOICE: 3D LiDAR — superior structural/geometric accuracy vs camera,
# works in low-visibility/dust/smoke disaster conditions.
#
# LIDAR SENSOR CONSTANTS (serve.py continuous-space API):
#   LIDAR_3D_RANGE    = 35.0m   (max ray distance per scan)
#   LIDAR_RAY_COUNT   = 60      (12 azimuth x 5 elevation = 60 rays per drone)
#   LIDAR_SCAN_FREQ   = 1       (scan every movement step)
#   LIDAR_FOV         = Hemispherical (360° azimuth, -9° to -86° elevation)
#
# TERMINATION:
#   Primary: 98%+ 2D column coverage OR 15 consecutive plateau steps (no new hits)
#   Safety ceiling: AM_MAX_STEPS = 1000 (NOT a target, prevents stuck episodes)
# ==============================================================================
AM_NUM_AGENTS = 5
AM_MAX_STEPS = 1000  # Generous safety ceiling (not a hard step cap target)
AM_SCENE_SIZE = 100.0  # 100m x 100m
AM_SCENE_HEIGHT = 30.0  # 30m height
LIDAR_3D_RANGE = 35.0  # meters max ray distance (scans reach ground from 28m altitude)
LIDAR_RAY_COUNT = 60  # 12 azimuth x 5 elevation = 60 rays per drone per scan

_azimuths = np.linspace(0, 2 * np.pi, 12, endpoint=False)
_elevations = np.linspace(-0.05 * np.pi, -0.48 * np.pi, 5)
LIDAR_RAY_DIRS = [
    (float(np.cos(el) * np.cos(az)), float(np.sin(el)), float(np.cos(el) * np.sin(az)))
    for el in _elevations for az in _azimuths
]


class AreaMapping3DState:
    def __init__(self):
        self.is_active = False
        self.step_count = 0
        # Hidden 3D Environment (NEVER sent to frontend during active run)
        self.ground_truth_3d_objects = []  # List of dicts describing 3D structures
        self.gt_surface_points = []  # Ground-truth surface sample points [[x, y, z, type]]
        # Discovered 3D Map (sent incrementally to frontend)
        self.discovered_point_cloud = []  # Accumulated hit points [[x, y, z, type]]
        self.scanned_voxels = set()  # Quantized (x, y, z) 3D voxel set for scan coverage
        self.scanned_2d_cols = set()  # Quantized (x, z) 2D column set for spatial coverage
        self.no_new_hits_count = 0  # Plateau tracker for 100% scan completion
        # Drone state in 3D
        self.agent_positions = {}  # { "agent_0": [x, y, z] }
        self.agent_zones = {}  # { "agent_0": [min_x, max_x, min_z, max_z] }
        self.is_done = False

    def reset(self):
        self.__init__()


am_state = AreaMapping3DState()


def _generate_3d_environment():
    """Loads actual provided glTF environment model (city_part_2/scene.gltf) and randomizes its position & rotation per run."""
    candidate_dirs = [
        os.path.join(os.path.dirname(__file__), "public", "assets", "city_part_2"),
        os.path.join(os.path.dirname(__file__), "dist", "assets", "city_part_2"),
        os.path.join(os.path.dirname(__file__), "frontend", "assets", "city_part_2"),
    ]
    gltf_path = None
    bin_path = None
    for c_dir in candidate_dirs:
        gp = os.path.join(c_dir, "scene.gltf")
        bp = os.path.join(c_dir, "scene.bin")
        if os.path.exists(gp) and os.path.exists(bp):
            gltf_path = gp
            bin_path = bp
            break

    objects = []
    gt_points = []

    # Ground plane surface points
    for gx in np.linspace(5.0, 95.0, 20):
        for gz in np.linspace(5.0, 95.0, 20):
            gt_points.append([round(float(gx), 2), 0.0, round(float(gz), 2), 0])  # type 0: ground

    if gltf_path and bin_path and os.path.exists(gltf_path) and os.path.exists(bin_path):
        try:
            gltf = json.load(open(gltf_path, "r"))
            bin_data = open(bin_path, "rb").read()

            def get_buffer_data(acc_idx):
                acc = gltf['accessors'][acc_idx]
                bv = gltf['bufferViews'][acc['bufferView']]
                byte_offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
                dtype = np.float32 if acc['componentType'] == 5126 else (np.uint16 if acc['componentType'] == 5123 else np.uint32)
                count = acc['count']
                num_comp = 3 if acc['type'] == 'VEC3' else (1 if acc['type'] == 'SCALAR' else 4)
                data = np.frombuffer(bin_data, dtype=dtype, count=count * num_comp, offset=byte_offset)
                return data.reshape(count, num_comp)

            def get_node_matrix(node):
                if 'matrix' in node:
                    return np.array(node['matrix']).reshape(4, 4).T
                M = np.eye(4)
                if 'translation' in node:
                    T = np.eye(4)
                    T[:3, 3] = node['translation']
                    M = M @ T
                if 'rotation' in node:
                    x, y, z, w = node['rotation']
                    R = np.array([
                        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
                        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w, 0],
                        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y, 0],
                        [0, 0, 0, 1]
                    ])
                    M = M @ R
                if 'scale' in node:
                    S = np.diag(node['scale'] + [1.0])
                    M = M @ S
                return M

            nodes = gltf.get('nodes', [])
            raw_prims = []
            all_verts = []

            for node_idx, node in enumerate(nodes):
                if 'mesh' in node:
                    mesh = gltf['meshes'][node['mesh']]
                    M = get_node_matrix(node)
                    for prim in mesh.get('primitives', []):
                        pos_acc = prim['attributes'].get('POSITION')
                        if pos_acc is not None:
                            verts = get_buffer_data(pos_acc)
                            ones = np.ones((len(verts), 1))
                            verts_h = np.hstack([verts, ones])
                            world_verts = (M @ verts_h.T).T[:, :3]
                            raw_prims.append(world_verts)
                            all_verts.append(world_verts)

            all_v = np.vstack(all_verts)
            min_v = all_v.min(axis=0)
            max_v = all_v.max(axis=0)
            size_v = np.maximum(max_v - min_v, 1e-5)

            scale = np.array([75.0 / size_v[0], 24.0 / size_v[1], 75.0 / size_v[2]])
            norm_prims = [(p - min_v) * scale for p in raw_prims]

            # Filter prominent glTF structural blocks
            candidates = []
            for norm_p in norm_prims:
                p_min = norm_p.min(axis=0)
                p_max = norm_p.max(axis=0)
                vol = (p_max[0] - p_min[0]) * (p_max[1] - p_min[1]) * (p_max[2] - p_min[2])
                if vol > 10.0 and p_max[1] > 2.0:
                    candidates.append((vol, norm_p))

            candidates.sort(key=lambda x: x[0], reverse=True)
            prominent_prims = [c[1] for c in candidates[:14]]

            # Random Y-axis rotation (0..360 deg) and position translation offset per run
            angle_rad = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

            offset_x = np.random.uniform(-10.0, 10.0)
            offset_z = np.random.uniform(-10.0, 10.0)

            for idx, norm_p in enumerate(prominent_prims):
                p_centered = norm_p - np.array([37.5, 0.0, 37.5])
                p_rot = (R_y @ p_centered.T).T
                p_final = p_rot + np.array([50.0 + offset_x, 0.0, 50.0 + offset_z])

                p_min = p_final.min(axis=0)
                p_max = p_final.max(axis=0)

                x_min, x_max = max(2.0, p_min[0]), min(98.0, p_max[0])
                z_min, z_max = max(2.0, p_min[2]), min(98.0, p_max[2])
                h_min, h_max = max(0.0, p_min[1]), min(30.0, p_max[1])

                if (x_max - x_min) > 1.0 and (z_max - z_min) > 1.0 and h_max > 1.0:
                    o_type = "building" if h_max > 14.0 else ("tower" if h_max > 20.0 else "ruined_block")
                    t_id = 1 if h_max > 14.0 else (2 if h_max > 20.0 else 3)
                    obj_def = {
                        "id": idx,
                        "type": o_type,
                        "type_id": t_id,
                        "bounds": [round(float(x_min), 2), round(float(h_min), 2), round(float(z_min), 2),
                                   round(float(x_max), 2), round(float(h_max), 2), round(float(z_max), 2)],
                        "center": [round(float((x_min + x_max)/2), 2), round(float(h_max/2), 2), round(float((z_min + z_max)/2), 2)],
                        "dims": [round(float(x_max - x_min), 2), round(float(h_max), 2), round(float(z_max - z_min), 2)]
                    }
                    objects.append(obj_def)

                    # Surface sample points for ground truth verification
                    for rx in np.linspace(x_min, x_max, max(3, int((x_max - x_min) / 4))):
                        for rz in np.linspace(z_min, z_max, max(3, int((z_max - z_min) / 4))):
                            gt_points.append([round(float(rx), 2), round(float(h_max), 2), round(float(rz), 2), t_id])

                    for ry in np.linspace(1.0, max(1.5, h_max - 0.5), max(3, int(h_max / 4))):
                        for rx in np.linspace(x_min, x_max, max(3, int((x_max - x_min) / 5))):
                            gt_points.append([round(float(rx), 2), round(float(ry), 2), round(float(z_min), 2), t_id])
                            gt_points.append([round(float(rx), 2), round(float(ry), 2), round(float(z_max), 2), t_id])
                        for rz in np.linspace(z_min, z_max, max(3, int((z_max - z_min) / 5))):
                            gt_points.append([round(float(x_min), 2), round(float(ry), 2), round(float(rz), 2), t_id])
                            gt_points.append([round(float(x_max), 2), round(float(ry), 2), round(float(z_max), 2), t_id])
        except Exception as e:
            print(f"Warning: glTF load failed ({e}), falling back to procedural structures.")

    # Fallback to procedural objects if glTF parsing yielded empty objects
    if not objects:
        num_objs = np.random.randint(12, 17)
        types = ["building", "tower", "ruined_block", "wall"]
        for obj_id in range(num_objs):
            o_type = types[obj_id % len(types)]
            w, d, h = np.random.uniform(10, 18), np.random.uniform(10, 18), np.random.uniform(12, 26)
            t_id = 1 if o_type == "building" else (2 if o_type == "tower" else 3)
            cx, cz = np.random.uniform(15.0, 85.0), np.random.uniform(15.0, 85.0)
            x_min, x_max = max(2.0, cx - w / 2), min(98.0, cx + w / 2)
            z_min, z_max = max(2.0, cz - d / 2), min(98.0, cz + d / 2)
            obj_def = {
                "id": obj_id, "type": o_type, "type_id": t_id,
                "bounds": [round(x_min, 2), 0.0, round(z_min, 2), round(x_max, 2), round(h, 2), round(z_max, 2)],
                "center": [round(cx, 2), round(h / 2, 2), round(cz, 2)],
                "dims": [round(x_max - x_min, 2), round(h, 2), round(z_max - z_min, 2)]
            }
            objects.append(obj_def)
            for rx in np.linspace(x_min, x_max, 4):
                for rz in np.linspace(z_min, z_max, 4):
                    gt_points.append([round(float(rx), 2), round(float(h), 2), round(float(rz), 2), t_id])

    return objects, gt_points


def _compute_3d_zones():
    """Divide 100m x 100m spatial area into 5 spatial zones along X/Z for 5 agents."""
    zones = {}
    row_h = AM_SCENE_SIZE / AM_NUM_AGENTS
    for i in range(AM_NUM_AGENTS):
        min_z = i * row_h
        max_z = (i + 1) * row_h
        zones[f"agent_{i}"] = [0.0, AM_SCENE_SIZE, round(min_z, 1), round(max_z, 1)]
    return zones


def _lidar_3d_scan(agent_pos, objects):
    """Simulate 3D LiDAR raycasting from agent_pos = [x, y, z].
    Casts 60 rays downwards & radially to intersect 3D environment geometry."""
    ax, ay, az = agent_pos
    new_hits = []

    for dx, dy, dz in LIDAR_RAY_DIRS:
        best_t = LIDAR_3D_RANGE
        best_type = 0  # 0=ground

        # Check ground plane (y = 0)
        if dy < 0:
            t_ground = -ay / dy
            if 0 < t_ground < best_t:
                best_t = t_ground
                best_type = 0

        # Check intersection with 3D box objects
        for obj in objects:
            b = obj["bounds"]  # [xmin, ymin, zmin, xmax, ymax, zmax]
            # Ray-AABB intersection test
            tx1 = (b[0] - ax) / (dx + 1e-8)
            tx2 = (b[3] - ax) / (dx + 1e-8)
            ty1 = (b[1] - ay) / (dy + 1e-8)
            ty2 = (b[4] - ay) / (dy + 1e-8)
            tz1 = (b[2] - az) / (dz + 1e-8)
            tz2 = (b[5] - az) / (dz + 1e-8)

            tmin = max(max(min(tx1, tx2), min(ty1, ty2)), min(tz1, tz2))
            tmax = min(min(max(tx1, tx2), max(ty1, ty2)), max(tz1, tz2))

            if tmax >= max(0.0, tmin) and tmin < best_t:
                best_t = tmin
                best_type = obj["type_id"]

        if best_t < LIDAR_3D_RANGE:
            hx = round(float(ax + dx * best_t), 2)
            hy = round(float(ay + dy * best_t), 2)
            hz = round(float(az + dz * best_t), 2)

            if 0 <= hx <= AM_SCENE_SIZE and 0 <= hy <= AM_SCENE_HEIGHT and 0 <= hz <= AM_SCENE_SIZE:
                voxel_key = (int(hx / 2.0), int(hy / 2.0), int(hz / 2.0))
                col_key = (int(hx / 2.0), int(hz / 2.0))
                am_state.scanned_2d_cols.add(col_key)
                if voxel_key not in am_state.scanned_voxels:
                    am_state.scanned_voxels.add(voxel_key)
                    hit = [hx, hy, hz, best_type]
                    am_state.discovered_point_cloud.append(hit)
                    new_hits.append(hit)

    return new_hits


def _frontier_3d_move(agent_name, agent_pos, zone_bounds, objects):
    """3D frontier movement towards unscanned spatial region in assigned zone."""
    ax, ay, az = agent_pos
    min_x, max_x, min_z, max_z = zone_bounds

    # Search for nearest unscanned spatial 2D column in assigned zone
    best_target = None
    best_dist = float('inf')

    # Grid search candidate target points in zone
    for tx in range(int(min_x) + 2, int(max_x) - 2, 3):
        for tz in range(int(min_z) + 2, int(max_z) - 2, 3):
            col_k = (int(tx / 2.0), int(tz / 2.0))
            if col_k not in am_state.scanned_2d_cols:
                d = np.sqrt((tx - ax) ** 2 + (tz - az) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_target = (tx, 14.0, tz)

    # Fallback to full scene search
    if best_target is None:
        for tx in range(2, 98, 4):
            for tz in range(2, 98, 4):
                col_k = (int(tx / 2.0), int(tz / 2.0))
                if col_k not in am_state.scanned_2d_cols:
                    d = np.sqrt((tx - ax) ** 2 + (tz - az) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_target = (tx, 14.0, tz)

    if best_target is None:
        return agent_pos

    tx, ty, tz = best_target
    vec = np.array([tx - ax, ty - ay, tz - az])
    norm = np.linalg.norm(vec)

    if norm < 0.5:
        col_k = (int(ax / 2.0), int(az / 2.0))
        am_state.scanned_2d_cols.add(col_k)
        return agent_pos

    step_len = 4.0
    move = (vec / norm) * step_len
    nx = float(np.clip(ax + move[0], 2.0, 98.0))
    ny = float(np.clip(ay + move[1], 6.0, 26.0))
    nz = float(np.clip(az + move[2], 2.0, 98.0))

    # Continuous magnetic repulsion field between drones to prevent inter-drone collisions
    for other_name, other_pos in am_state.agent_positions.items():
        if other_name == agent_name:
            continue
        ox, oy, oz = other_pos
        dist = np.sqrt((nx - ox) ** 2 + (ny - oy) ** 2 + (nz - oz) ** 2)
        if 0.001 < dist < 12.0:
            force = 1.2 * ((1.0 - dist / 12.0) ** 2)
            nx = float(np.clip(nx + ((nx - ox) / dist) * force, 2.0, 98.0))
            ny = float(np.clip(ny + ((ny - oy) / dist) * force * 0.5, 6.0, 26.0))
            nz = float(np.clip(nz + ((nz - oz) / dist) * force, 2.0, 98.0))

    # Building collision check: avoid passing inside solid building boxes
    for obj in objects:
        b = obj["bounds"]
        if b[0] - 1.0 <= nx <= b[3] + 1.0 and b[2] - 1.0 <= nz <= b[5] + 1.0 and ny < b[4] + 1.0:
            # Ascend over structure
            ny = min(28.0, b[4] + 3.0)

    return [round(nx, 2), round(ny, 2), round(nz, 2)]


@app.post("/area_mapping/start")
def start_area_mapping():
    """Initialize 3D Area Mapping simulation. Randomizes hidden 3D environment layout server-side."""
    am_state.reset()

    # Generate hidden 3D ground-truth environment layout
    objects, gt_points = _generate_3d_environment()
    am_state.ground_truth_3d_objects = objects
    am_state.gt_surface_points = gt_points

    # 3D Zone partitioning
    am_state.agent_zones = _compute_3d_zones()

    # Spawn 5 agents in 3D space in their respective zone bounds
    am_state.agent_positions = {}
    for i in range(AM_NUM_AGENTS):
        name = f"agent_{i}"
        zb = am_state.agent_zones[name]
        spawn_x = round(float((zb[0] + zb[1]) / 2), 2)
        spawn_z = round(float((zb[2] + zb[3]) / 2), 2)
        spawn_y = 12.0  # Spawn altitude
        am_state.agent_positions[name] = [spawn_x, spawn_y, spawn_z]

    # Initial 3D LiDAR raycast scan
    initial_revealed = []
    for name, pos in am_state.agent_positions.items():
        hits = _lidar_3d_scan(pos, am_state.ground_truth_3d_objects)
        initial_revealed.extend(hits)

    am_state.is_active = True
    am_state.step_count = 0

    scanned_pct = round(min(100.0, (len(am_state.scanned_2d_cols) / 450.0) * 100.0), 1)

    return {
        "status": "area_mapping_started",
        "metadata": {
            "scene_size_meters": AM_SCENE_SIZE,
            "scene_height_meters": AM_SCENE_HEIGHT,
            "num_agents": AM_NUM_AGENTS,
            "max_steps": AM_MAX_STEPS,
            "lidar_3d_range": LIDAR_3D_RANGE,
            "num_3d_structures": len(objects)
        },
        "agent_positions": am_state.agent_positions,
        "agent_zones": am_state.agent_zones,
        "initial_point_cloud": am_state.discovered_point_cloud,
        "scanned_pct": scanned_pct,
        "step": 0,
        "is_done": False
    }


@app.post("/area_mapping/step")
def step_area_mapping():
    """Advances 3D Area Mapping simulation by 1 step. Returns 3D discovered point cloud updates."""
    if not am_state.is_active:
        raise HTTPException(status_code=400, detail="Area mapping not started. Call /area_mapping/start first.")

    if am_state.is_done:
        scanned_pct = round(min(100.0, (len(am_state.scanned_2d_cols) / 450.0) * 100.0), 1)
        return {
            "status": "mapping_complete",
            "is_done": True,
            "step": am_state.step_count,
            "agent_positions": am_state.agent_positions,
            "newly_discovered_points": [],
            "total_point_cloud_count": len(am_state.discovered_point_cloud),
            "scanned_pct": scanned_pct
        }

    am_state.step_count += 1

    # Move each drone in 3D space
    new_positions = {}
    all_new_hits = []

    for i in range(AM_NUM_AGENTS):
        name = f"agent_{i}"
        pos = am_state.agent_positions[name]
        zb = am_state.agent_zones[name]
        new_pos = _frontier_3d_move(name, pos, zb, am_state.ground_truth_3d_objects)
        new_positions[name] = new_pos

        # Perform 3D LiDAR raycast scan
        hits = _lidar_3d_scan(new_pos, am_state.ground_truth_3d_objects)
        all_new_hits.extend(hits)

    am_state.agent_positions = new_positions

    if len(all_new_hits) == 0:
        am_state.no_new_hits_count += 1
    else:
        am_state.no_new_hits_count = 0

    scanned_pct = round(min(100.0, (len(am_state.scanned_2d_cols) / 450.0) * 100.0), 1)
    # Completion-based termination: >98% mapped OR (min 30 steps AND 15 consecutive plateau steps with no new hits) OR safety ceiling
    if scanned_pct >= 98.0 or (am_state.step_count >= 30 and am_state.no_new_hits_count >= 15) or am_state.step_count >= AM_MAX_STEPS:
        am_state.is_done = True

    return {
        "status": "step_executed",
        "is_done": am_state.is_done,
        "step": am_state.step_count,
        "agent_positions": am_state.agent_positions,
        "newly_discovered_points": all_new_hits,
        "total_point_cloud_count": len(am_state.discovered_point_cloud),
        "scanned_pct": scanned_pct
    }


@app.get("/area_mapping/verify")
def verify_area_mapping():
    """Post-run 3D accuracy verification.
    Compares accumulated 3D discovered point cloud vs hidden 3D ground truth environment.
    Only reveals actual 3D ground-truth environment arrangement AFTER mapping run is complete."""
    if not am_state.gt_surface_points:
        raise HTTPException(status_code=400, detail="No area mapping run to verify.")

    gt_pts = np.array([p[:3] for p in am_state.gt_surface_points])
    disc_pts = np.array([p[:3] for p in am_state.discovered_point_cloud]) if am_state.discovered_point_cloud else np.empty((0, 3))

    matched_disc = 0
    tolerance = 2.5  # meters tolerance for surface point match

    if len(disc_pts) > 0 and len(gt_pts) > 0:
        for dp in disc_pts:
            dists = np.linalg.norm(gt_pts - dp, axis=1)
            if np.min(dists) <= tolerance:
                matched_disc += 1

    total_disc = len(disc_pts)
    total_gt = len(gt_pts)
    accuracy_pct = round((matched_disc / max(1, total_disc)) * 100.0, 1)
    scan_coverage_pct = round(min(100.0, (len(am_state.scanned_2d_cols) / 450.0) * 100.0), 1)

    return {
        "status": "verification_complete",
        "accuracy_pct": accuracy_pct,
        "scan_coverage_pct": scan_coverage_pct,
        "total_steps": am_state.step_count,
        "total_discovered_points": total_disc,
        "total_gt_surface_points": total_gt,
        "matched_surface_points": matched_disc,
        # Revealed Ground Truth 3D environment arrangement (ONLY now)
        "ground_truth_3d_objects": am_state.ground_truth_3d_objects,
        "discovered_point_cloud": am_state.discovered_point_cloud
    }



@app.post("/area_mapping/reset")
def reset_area_mapping():
    """Resets area mapping state."""
    am_state.reset()
    return {"status": "area_mapping_reset"}


# Mount frontend static files (prioritize dist build)
dist_dir = os.path.join(os.path.dirname(__file__), "dist")
public_dir = os.path.join(os.path.dirname(__file__), "public")
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
static_dir = dist_dir if os.path.exists(dist_dir) else (public_dir if os.path.exists(public_dir) else frontend_dir)
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)
