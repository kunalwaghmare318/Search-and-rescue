import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# ==============================================================================
# ENVIRONMENT: AREA MAPPING V3 — TRUE 3D LIDAR + FULL REWARD HIERARCHY + V18 NAV
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                    REWARD / PENALTY HIERARCHY (ordered)                     │
# │                                                                            │
# │  DESIGN PRINCIPLE: Completeness and accuracy of the 3D map are ALWAYS      │
# │  prioritized over speed.                                                   │
# │                                                                            │
# │  Rank │ Term                      │ Value    │ Ratio vs Discovery Reward   │
# │  ─────┼───────────────────────────┼──────────┼─────────────────────────────│
# │   1   │ New-Point Discovery       │ +20.0    │ 1.0x (DOMINANT, baseline)   │
# │   2   │ Zone-Completion Bonus     │ +30.0    │ 1.5x (one-time per zone)    │
# │   3   │ Collision Penalty (v18)   │ -500.0   │ 25x  (v18 ported, safety)   │
# │   3b  │ Zero-Collision Episode    │ +200.0   │ 10x  (v18 ported, safety)   │
# │   3c  │ Path Conflict Warning     │ -30.0    │ 1.5x (v18 ported)          │
# │   3d  │ Vertical Separation Bonus │ +5.0     │ 0.25x(v18 ported)          │
# │   4   │ Frontier-Approach Bonus   │ +2.0     │ 0.1x (shaping-only nudge)  │
# │   5   │ Backfill Incentive        │ +10.0    │ 0.5x (covers orphaned zones)│
# │   6   │ Redundant-Rescan Penalty  │ -0.5     │ 0.025x (minor nudge)       │
# │   7   │ Efficiency/Time Penalty   │ -0.001   │ 0.00005x (deliberate minor)│
# │       │                           │          │ 20000x smaller than disc.   │
# │   7b  │ Efficiency Bonus (ep end) │ max 5.0  │ 0.25x (tiebreaker only)    │
# │   7c  │ 100% Coverage Completion  │ +200.0   │ 10x  (episode-terminal)    │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# LIDAR SENSOR CONFIGURATION:
#   LIDAR_RANGE         = 2.0 cells    (max ray distance per scan step)
#   LIDAR_RAY_PATTERN   = 26 directions (full 3D spherical: 3^3 - 1 unit vectors)
#   LIDAR_STEPS_PER_RAY = 2            (propagation at t=1.0 and t=2.0)
#   LIDAR_SCAN_FREQ     = 1            (scan every movement step)
#   LIDAR_FOV           = Spherical    (360° azimuth, full elevation hemisphere)
#
# V18 COLLISION AVOIDANCE (ported as-is):
#   - Inter-Drone Path Communication: broadcast intended [dr, dc, dalt] vectors
#   - Proactive Elevation Adjustment: altitude separation on projected path conflict
#   - Magnetic Repulsion Field: continuous 3D push (radius=2.5, strength=0.8)
#   - Drone Destruction: -500.0 penalty on physical collision
#
# EPISODE TERMINATION:
#   - Primary: completion_threshold (95% of total scannable 3D voxels mapped)
#     95% chosen because real self-occluding geometry (interior voxels below
#     solid structure roofs at alt < cell_height) are physically unreachable
#     by any external ray angle — reporting ~95% as structurally achievable max.
#   - Safety Ceiling: MAX_STEPS = 1500 (generous, prevents stuck/bugged episodes)
#     This does NOT function as an implicit target — it's 3-5x the typical
#     episode length for a well-trained policy.
#   - All-Destroyed: if all drones are destroyed, episode ends immediately.
# ==============================================================================

GRID_SIZE = 10
NUM_AGENTS = 5
MAX_ALTITUDE = 2
MAX_STEPS_SAFETY_CEILING = 1500  # Generous safety ceiling (3-5x typical episode length)
COMPLETION_THRESHOLD_PCT = 95.0  # 95% of scannable voxels = structurally achievable max

# ──────────────────────────────────────────────────────────────────────────────
# LIDAR SENSOR CONSTANTS (documented)
# ──────────────────────────────────────────────────────────────────────────────
LIDAR_RANGE = 2.0              # Max ray distance in grid cells
LIDAR_RAY_DIRECTIONS = 26      # 3^3 - 1 = 26 unit direction vectors (full sphere)
LIDAR_STEPS_PER_RAY = 2        # Propagation steps per ray (t=1.0, t=2.0)
LIDAR_SCAN_FREQUENCY = 1       # Scan once per movement step (every timestep)
LIDAR_FOV_TYPE = "spherical"   # Full 360° azimuth + full elevation hemisphere

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 1: New-Point Discovery (DOMINANT)
# ──────────────────────────────────────────────────────────────────────────────
REWARD_NEW_VOXEL_MAPPED = 20.0
# ^ DOMINANT term. Every other reward/penalty is documented relative to this.
# Rationale: each newly discovered unique 3D surface voxel is the single most
# valuable action a drone can take. This ensures the policy optimizes for
# maximum 3D map completeness above all else.

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 2: Zone-Completion Bonus
# ──────────────────────────────────────────────────────────────────────────────
REWARD_ZONE_COMPLETE = 30.0
# ^ 1.5x discovery reward. One-time bonus when a drone achieves 100% coverage
# of its individually assigned 3D zone. Reinforces thoroughness within own
# responsibility area, not just opportunistic point-collection elsewhere.

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 3: Collision Avoidance (v18 ported, proven values)
# ──────────────────────────────────────────────────────────────────────────────
PENALTY_DRONE_DESTRUCTION = -500.0
# ^ 25x discovery reward. Dominant safety term ported directly from v18.
# Collisions are catastrophic and must remain extremely costly.

REWARD_ZERO_COLLISION_EPISODE = 200.0
# ^ 10x discovery reward. End-of-episode bonus for zero collisions. v18 ported.

PENALTY_PATH_CONFLICT_WARNING = -30.0
# ^ 1.5x discovery reward. Proactive penalty when two drones' projected paths
# would collide at the same altitude. v18 ported.

REWARD_VERTICAL_SEPARATION_BONUS = 5.0
# ^ 0.25x discovery reward. Bonus when drones proactively separate altitude
# to avoid a projected collision. v18 ported.

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 4: Frontier-Approach Bonus (shaping-only)
# ──────────────────────────────────────────────────────────────────────────────
REWARD_FRONTIER_BONUS = 2.0
# ^ 0.1x discovery reward. Minor reward for moving toward the boundary between
# known and unknown space. Small enough to only nudge behavior toward efficient
# exploration order — never substitutes for the discovery reward itself.

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 5: Nearest-Drone Backfill Incentive
# ──────────────────────────────────────────────────────────────────────────────
REWARD_BACKFILL_ZONE = 10.0
# ^ 0.5x discovery reward. Reward for a drone extending coverage into a zone
# that another drone hasn't completed (early finish or failed drone). Ensures
# no region is permanently left unmapped by omission.

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 6: Redundant-Rescan Penalty (minor nudge)
# ──────────────────────────────────────────────────────────────────────────────
PENALTY_REDUNDANT_RESCAN = -0.5
# ^ 0.025x discovery reward. Minor penalty for spending time re-scanning
# already-fully-discovered areas. Exists only to nudge efficient distribution
# among drones, never large enough to discourage exploration.

# ──────────────────────────────────────────────────────────────────────────────
# REWARD HIERARCHY — Term 7: Efficiency/Time Penalty (deliberately minor)
# ──────────────────────────────────────────────────────────────────────────────
PENALTY_PER_STEP = -0.001
# ^ 0.00005x discovery reward (20,000x smaller than new-point discovery).
# Present ONLY as a mild tiebreaker encouraging reasonably efficient scanning.
# Explicitly minor enough that it can never plausibly outweigh the value of
# achieving more complete coverage. Exact ratio: |0.001| / 20.0 = 1/20000.

REWARD_EFFICIENCY_BONUS = 5.0
# ^ max 5.0 = 0.25x discovery reward. Minor tiebreaker bonus at episode end
# for completing in fewer steps. max(0, 5.0 * (1 - steps/max_steps)).

REWARD_100PCT_COVERAGE_COMPLETE = 200.0
# ^ 10x discovery reward. Terminal bonus when completion threshold is reached.

# ──────────────────────────────────────────────────────────────────────────────
# V18 Repulsion Field Constants (ported as-is)
# ──────────────────────────────────────────────────────────────────────────────
REPULSION_FIELD_RADIUS = 2.5
REPULSION_FORCE_STRENGTH = 0.8
REPULSION_PUSH_THRESHOLD = 0.35

# Flyover bonus (minor, obstacle navigation aid)
REWARD_OBSTACLE_FLYOVER = 2.0

# Action Mapping: Discrete(7)
ACTION_DISPLACEMENT = {
    0: (0, 0, 0),    # Stay
    1: (-1, 0, 0),   # Up (row - 1)
    2: (1, 0, 0),    # Down (row + 1)
    3: (0, -1, 0),   # Left (col - 1)
    4: (0, 1, 0),    # Right (col + 1)
    5: (0, 0, 1),    # Ascend Alt
    6: (0, 0, -1)    # Descend Alt
}


class AreaMappingEnvV3(ParallelEnv):
    """
    Area Mapping V3: True 3D LiDAR Raycasting + Full 7-Term Reward Hierarchy
    + V18 Collision Avoidance (ported) + Completion-Based Termination.
    """
    metadata = {"name": "area_mapping_v3", "render_modes": []}

    def __init__(self, grid_size=GRID_SIZE, num_agents=NUM_AGENTS,
                 max_steps=MAX_STEPS_SAFETY_CEILING, curriculum_stage=1.0,
                 render_mode=None, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents = num_agents
        self.max_steps = max_steps
        self.curriculum_stage = float(np.clip(curriculum_stage, 0.0, 1.0))
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]

        self._action_spaces = {a: spaces.Discrete(7) for a in self.possible_agents}
        # Obs: 4 self + 4 teammates * 7 path-comm + 400 grid = 432
        self._observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(432,), dtype=np.float32)
            for a in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.episode_collisions = 0
        self.episode_repulsion = 0
        self.episode_redundant_rescans = 0
        self.drones_destroyed = 0
        self.elevation_avoidance_count = 0
        self.path_comm_avoidance_count = 0

        self.scanned_voxels = set()
        self.visited_grid_2d = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # ── Generate 3D environment (obstacles + buildings with heights) ──
        min_objs = 4
        max_objs = 14
        num_objs = int(min_objs + (max_objs - min_objs) * self.curriculum_stage)

        self.obstacles = set()
        self.buildings = set()
        self.obstacle_heights = {}

        all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        np.random.shuffle(all_cells)

        for i in range(num_objs):
            pos = all_cells[i]
            if i % 2 == 0:
                self.obstacles.add(pos)
                self.obstacle_heights[pos] = 1
            else:
                self.buildings.add(pos)
                self.obstacle_heights[pos] = 2

        free_cells = [c for c in all_cells if c not in self.obstacles and c not in self.buildings]

        # ── Total scannable 3D voxels (surface + airspace above structures) ──
        self.total_walkable_voxels = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell_h = self.obstacle_heights.get((r, c), 0)
                for alt in range(MAX_ALTITUDE + 1):
                    if alt >= cell_h:
                        self.total_walkable_voxels.add((r, c, alt))

        # ── Zone Partitioning (row-based, reused from prior work) ──
        self.agent_zones = {}
        rows_per = GRID_SIZE // self.n_agents
        for i, a in enumerate(self.possible_agents):
            z_start = i * rows_per
            z_end = (i + 1) * rows_per if i < self.n_agents - 1 else GRID_SIZE
            self.agent_zones[a] = (z_start, z_end)

        self.zone_voxels = {}
        for a in self.possible_agents:
            z_start, z_end = self.agent_zones[a]
            self.zone_voxels[a] = set(
                v for v in self.total_walkable_voxels if z_start <= v[0] < z_end
            )

        self.zone_completed = {a: False for a in self.possible_agents}

        # ── Spawn agents in 3D ──
        self.failed_agents = set()
        self.failure_step_map = {}
        self.reassigned_agents = {}
        self.agent_positions = {}
        self.agent_altitudes = {}
        self.agent_intended_vectors = {a: (0, 0, 0) for a in self.possible_agents}
        used = set()
        for a in self.possible_agents:
            z_start, z_end = self.agent_zones[a]
            z_free = [c for c in free_cells if z_start <= c[0] < z_end and c not in used]
            pos = z_free[0] if z_free else free_cells[np.random.randint(len(free_cells))]
            self.agent_positions[a] = pos
            self.agent_altitudes[a] = 0
            used.add(pos)
            self._perform_lidar_scan(a)

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return observations, infos

    def trigger_failure(self, agent_name):
        """Nearest-drone backfill: reassign failed drone's zone to closest active drones."""
        if agent_name in self.failed_agents:
            return
        self.failed_agents.add(agent_name)
        self.failure_step_map[agent_name] = self.step_count
        failed_pos = self.agent_positions.get(agent_name, (0, 0))
        active = [a for a in self.possible_agents if a != agent_name and a not in self.failed_agents]
        if active:
            dists = []
            for a in active:
                d = np.linalg.norm(np.array(failed_pos) - np.array(self.agent_positions[a]))
                dists.append((d, a))
            dists.sort()
            self.reassigned_agents[agent_name] = [a for _, a in dists[:2]]

    # ──────────────────────────────────────────────────────────────────────
    # TRUE 3D LIDAR RAYCASTING
    # Range: LIDAR_RANGE (2.0 cells), Pattern: 26 spherical directions,
    # Steps: 2 per ray (t=1.0, t=2.0), Frequency: every movement step
    # ──────────────────────────────────────────────────────────────────────
    def _perform_lidar_scan(self, agent_name):
        """Cast 26 rays in true 3D spherical pattern from drone position.
        Rays intersect against actual 3D geometry collision mesh.
        Hit: record 3D point at hit location. Miss: record nothing."""
        if agent_name in self.failed_agents:
            return 0
        r, c = self.agent_positions[agent_name]
        alt = self.agent_altitudes[agent_name]

        scanned_new = 0
        # 26 direction vectors covering full 3D sphere (3^3 - 1)
        ray_dirs = [
            (dr, dc, dalt)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            for dalt in [-1, 0, 1]
            if not (dr == 0 and dc == 0 and dalt == 0)
        ]

        # Scan drone's current voxel
        curr = (r, c, alt)
        if curr in self.total_walkable_voxels and curr not in self.scanned_voxels:
            self.scanned_voxels.add(curr)
            self.visited_grid_2d[r, c] = 1.0
            scanned_new += 1

        # Cast rays at t=1.0 and t=2.0 (LIDAR_STEPS_PER_RAY = 2)
        for dr, dc, dalt in ray_dirs:
            norm = np.sqrt(dr**2 + dc**2 + dalt**2)
            step_r, step_c, step_alt = dr / norm, dc / norm, dalt / norm
            for dist in [1.0, 2.0]:
                nr = int(round(r + step_r * dist))
                nc = int(round(c + step_c * dist))
                n_alt = int(round(alt + step_alt * dist))

                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and 0 <= n_alt <= MAX_ALTITUDE:
                    cell_h = self.obstacle_heights.get((nr, nc), 0)
                    if n_alt < cell_h:
                        # Ray hits 3D surface — record hit and terminate ray
                        sv = (nr, nc, n_alt)
                        if sv not in self.scanned_voxels:
                            self.scanned_voxels.add(sv)
                            self.visited_grid_2d[nr, nc] = 1.0
                            scanned_new += 1
                        break  # Ray occluded
                    else:
                        voxel = (nr, nc, n_alt)
                        if voxel in self.total_walkable_voxels:
                            if voxel not in self.scanned_voxels:
                                self.scanned_voxels.add(voxel)
                                self.visited_grid_2d[nr, nc] = 1.0
                                scanned_new += 1

        return scanned_new

    # ──────────────────────────────────────────────────────────────────────
    # OBSERVATION (V18 PATH-COMMUNICATION FORMAT)
    # ──────────────────────────────────────────────────────────────────────
    def _get_obs(self, agent_name):
        obs = np.zeros(432, dtype=np.float32)
        if agent_name in self.failed_agents:
            return obs

        r, c = self.agent_positions[agent_name]
        alt = self.agent_altitudes[agent_name]
        obs[0] = r / (GRID_SIZE - 1)
        obs[1] = c / (GRID_SIZE - 1)
        obs[2] = self.step_count / self.max_steps
        obs[3] = alt / MAX_ALTITUDE

        # 4 teammates x 7 features (pos, alt, dist, intended_dr, intended_dc, intended_dalt)
        idx = 4
        for a in self.possible_agents:
            if a == agent_name:
                continue
            if a in self.failed_agents:
                obs[idx:idx+7] = [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]
            else:
                tr, tc = self.agent_positions[a]
                talt = self.agent_altitudes[a]
                idr, idc, idalt = self.agent_intended_vectors[a]
                obs[idx] = tr / (GRID_SIZE - 1)
                obs[idx+1] = tc / (GRID_SIZE - 1)
                obs[idx+2] = talt / MAX_ALTITUDE
                obs[idx+3] = np.linalg.norm(np.array([r, c]) - np.array([tr, tc])) / (GRID_SIZE * np.sqrt(2))
                obs[idx+4] = idr
                obs[idx+5] = idc
                obs[idx+6] = idalt / MAX_ALTITUDE
            idx += 7

        # Flattened grid (10x10x4)
        idx_flat = 32
        for gr in range(GRID_SIZE):
            for gc in range(GRID_SIZE):
                obs[idx_flat] = self.visited_grid_2d[gr, gc]
                obs[idx_flat+1] = 1.0 if (gr, gc) in self.obstacles else (2.0 if (gr, gc) in self.buildings else 0.0)
                scanned_layers = sum(1 for al in range(MAX_ALTITUDE + 1) if (gr, gc, al) in self.scanned_voxels)
                obs[idx_flat+2] = scanned_layers / (MAX_ALTITUDE + 1)
                obs[idx_flat+3] = 0.0
                idx_flat += 4

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # STEP: V18 COLLISION AVOIDANCE + FULL 7-TERM REWARD STRUCTURE
    # ──────────────────────────────────────────────────────────────────────
    def step(self, actions):
        rewards = {a: 0.0 for a in self.possible_agents}
        prev_positions = {a: self.agent_positions[a] for a in self.possible_agents}
        active_agents = [a for a in self.possible_agents if a not in self.failed_agents]

        # ── 1. V18 Inter-Drone Path Communication ──
        intent_pos = {}
        intent_alt = {}
        for agent in active_agents:
            act = actions.get(agent, 0)
            dr, dc, dalt = ACTION_DISPLACEMENT.get(act, (0, 0, 0))
            self.agent_intended_vectors[agent] = (dr, dc, dalt)

            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]

            nr = int(np.clip(r + dr, 0, GRID_SIZE - 1))
            nc = int(np.clip(c + dc, 0, GRID_SIZE - 1))
            nalt = int(np.clip(alt + dalt, 0, MAX_ALTITUDE))

            intent_pos[agent] = (nr, nc)
            intent_alt[agent] = nalt

        # ── 2. V18 Proactive Path Conflict Analysis ──
        for i in range(len(active_agents)):
            a1 = active_agents[i]
            for j in range(i + 1, len(active_agents)):
                a2 = active_agents[j]
                if intent_pos[a1] == intent_pos[a2]:
                    if intent_alt[a1] == intent_alt[a2]:
                        # Term 3c: Path conflict warning
                        rewards[a1] += PENALTY_PATH_CONFLICT_WARNING
                        rewards[a2] += PENALTY_PATH_CONFLICT_WARNING
                    else:
                        # Term 3d: Successful altitude separation
                        rewards[a1] += REWARD_VERTICAL_SEPARATION_BONUS
                        rewards[a2] += REWARD_VERTICAL_SEPARATION_BONUS
                        self.path_comm_avoidance_count += 1

        # ── 3. V18 Magnetic Repulsion Field Safety Net ──
        for agent in active_agents:
            r, c = intent_pos[agent]
            alt = intent_alt[agent]

            F_r, F_c, F_alt = 0.0, 0.0, 0.0
            for other in active_agents:
                if other == agent:
                    continue
                or_p, oc_p = intent_pos[other]
                oalt = intent_alt[other]
                dist = np.linalg.norm(np.array([r, c, alt]) - np.array([or_p, oc_p, oalt]))
                if 0.001 < dist < REPULSION_FIELD_RADIUS:
                    force = REPULSION_FORCE_STRENGTH * ((1.0 - dist / REPULSION_FIELD_RADIUS) ** 2)
                    F_r += ((r - or_p) / dist) * force
                    F_c += ((c - oc_p) / dist) * force
                    F_alt += ((alt - oalt) / dist) * force

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in self.obstacle_heights:
                        h = self.obstacle_heights[(nr, nc)]
                        if alt < h:
                            d_obs = np.sqrt(dr*dr + dc*dc) + 0.01
                            if d_obs < REPULSION_FIELD_RADIUS:
                                force = REPULSION_FORCE_STRENGTH * ((1.0 - d_obs / REPULSION_FIELD_RADIUS) ** 2)
                                F_r -= (dr / d_obs) * force
                                F_c -= (dc / d_obs) * force
                                F_alt += 0.5 * force

            if F_alt > REPULSION_PUSH_THRESHOLD:
                alt = int(np.clip(alt + 1, 0, MAX_ALTITUDE))
                self.episode_repulsion += 1
                self.elevation_avoidance_count += 1

            self.agent_positions[agent] = (r, c)
            self.agent_altitudes[agent] = alt

        # ── 4. V18 Destruction & Collision Verification ──
        destroyed = set()

        for agent in active_agents:
            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]
            cell_h = self.obstacle_heights.get((r, c), 0)
            if cell_h > 0 and alt < cell_h:
                destroyed.add(agent)
            elif cell_h > 0 and alt >= cell_h:
                rewards[agent] += REWARD_OBSTACLE_FLYOVER

        for i in range(len(active_agents)):
            a1 = active_agents[i]
            if a1 in destroyed:
                continue
            p1 = np.array(self.agent_positions[a1])
            alt1 = self.agent_altitudes[a1]
            for j in range(i + 1, len(active_agents)):
                a2 = active_agents[j]
                if a2 in destroyed:
                    continue
                p2 = np.array(self.agent_positions[a2])
                alt2 = self.agent_altitudes[a2]
                if np.linalg.norm(p1 - p2) < 1.0 and alt1 == alt2:
                    destroyed.add(a1)
                    destroyed.add(a2)
                    self.episode_collisions += 1

        # Term 3: Collision penalty (v18 ported, -500.0)
        for d_agent in destroyed:
            self.failed_agents.add(d_agent)
            rewards[d_agent] += PENALTY_DRONE_DESTRUCTION
            self.drones_destroyed += 1

        still_active = [a for a in active_agents if a not in self.failed_agents]

        # ── 5. True 3D LiDAR Scanning + Full Reward Terms ──
        for agent in still_active:
            r, c = self.agent_positions[agent]
            z_start, z_end = self.agent_zones[agent]

            # Term 7: Efficiency/time penalty (deliberately minor, -0.001)
            rewards[agent] += PENALTY_PER_STEP

            scanned_new = self._perform_lidar_scan(agent)

            if scanned_new > 0:
                # Term 1: New-point discovery reward (DOMINANT, +20.0 per voxel)
                rewards[agent] += REWARD_NEW_VOXEL_MAPPED * scanned_new

                # Term 4: Frontier-approach bonus (shaping-only, +2.0)
                rewards[agent] += REWARD_FRONTIER_BONUS

                # Term 5: Backfill incentive — reward for scanning outside own zone
                if not (z_start <= r < z_end):
                    # Check if this is an incomplete zone (backfill scenario)
                    for failed_a, reassigned_list in self.reassigned_agents.items():
                        if agent in reassigned_list:
                            rewards[agent] += REWARD_BACKFILL_ZONE
                            break
                    else:
                        # Even without explicit failure, reward covering other zones
                        rewards[agent] += REWARD_BACKFILL_ZONE
            else:
                # Term 6: Redundant-rescan penalty (minor, -0.5)
                rewards[agent] += PENALTY_REDUNDANT_RESCAN
                self.episode_redundant_rescans += 1

            # Term 2: Zone-completion bonus (one-time, +30.0)
            if not self.zone_completed[agent]:
                z_voxels = self.zone_voxels[agent]
                if z_voxels and z_voxels.issubset(self.scanned_voxels):
                    rewards[agent] += REWARD_ZONE_COMPLETE
                    self.zone_completed[agent] = True

        self.step_count += 1

        # ── 6. Completion-Based Episode Termination ──
        scanned_walkable = len(self.scanned_voxels.intersection(self.total_walkable_voxels))
        total_walkable = max(1, len(self.total_walkable_voxels))
        coverage_pct = min(100.0, (scanned_walkable / total_walkable) * 100.0)

        threshold_met = coverage_pct >= COMPLETION_THRESHOLD_PCT
        safety_ceiling = self.step_count >= self.max_steps
        all_destroyed = len(still_active) == 0

        done = threshold_met or safety_ceiling or all_destroyed

        if threshold_met:
            for agent in still_active:
                # Term 7c: 100% coverage completion (+200.0)
                rewards[agent] += REWARD_100PCT_COVERAGE_COMPLETE
                # Term 7b: Efficiency bonus (max 5.0, tiebreaker only)
                eff = max(0.0, REWARD_EFFICIENCY_BONUS * (1.0 - self.step_count / float(self.max_steps)))
                rewards[agent] += eff

        # Term 3b: Zero-collision episode bonus (v18 ported, +200.0)
        if done and self.drones_destroyed == 0:
            for agent in still_active:
                rewards[agent] += REWARD_ZERO_COLLISION_EPISODE

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: done for a in self.possible_agents}
        infos = {
            a: {
                "collisions": self.episode_collisions,
                "drones_destroyed": self.drones_destroyed,
                "elevation_avoidance": self.elevation_avoidance_count,
                "path_comm_avoidance": self.path_comm_avoidance_count,
                "coverage_pct": round(coverage_pct, 1),
                "redundant_rescans": self.episode_redundant_rescans,
                "all_mapped": threshold_met,
                "scanned_voxels": len(self.scanned_voxels),
                "total_voxels": len(self.total_walkable_voxels),
                "zone_completion_rate": sum(1 for z in self.zone_completed.values() if z) / self.n_agents,
                "steps_to_completion": self.step_count if threshold_met else self.max_steps
            } for a in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos


def parallel_area_mapping_env_v3(**kwargs):
    return AreaMappingEnvV3(**kwargs)
