import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# ==============================================================================
# ENVIRONMENT CONFIGURATION: AREA MAPPING V2 (TRUE 3D LIDAR RAYCASTING & V18 NAV)
#
# DOMINANT REWARD PHILOSOPHY:
# 3D Area Coverage Accuracy is the single dominant, near-exclusive reward priority.
# - REWARD_NEW_VOXEL_MAPPED = 20.0 (Primary exploration signal per 3D surface voxel)
# - REWARD_100PCT_COVERAGE_COMPLETE = 200.0 (Dominant episode completion bonus)
# - PENALTY_PER_STEP = -0.001 (Minimal step penalty; magnitude is 20,000x smaller than 100% coverage bonus)
# - REWARD_EFFICIENCY_BONUS = 5.0 (Minor tiebreaker nudge only; max 5.0 << 200.0 coverage bonus)
#
# REUSED V18 NAVIGATION & SAFETY MECHANICS:
# - Proactive Path Communication: Teammates broadcast intended displacement vectors [dr, dc, dalt]
# - Altitude Separation & Magnetic Repulsion Field: Continuous 3D push to avoid inter-drone collisions
# - Dominant Collision Penalties: PENALTY_DRONE_DESTRUCTION = -500.0
# ==============================================================================
GRID_SIZE = 10
NUM_AGENTS = 5
MAX_ALTITUDE = 2
MAX_STEPS_SAFETY_CEILING = 1000  # Generous safety ceiling net to allow thorough time for 100% 3D scanning

# 1. Dominant 3D Coverage Reward Hierarchy
REWARD_NEW_VOXEL_MAPPED          = 20.0   # Dominant reward per newly discovered 3D surface voxel
REWARD_100PCT_COVERAGE_COMPLETE   = 200.0  # Dominant completion bonus when 100% scannable 3D voxels are mapped
REWARD_ZONE_100PCT_COMPLETE      = 30.0   # Bonus when drone's assigned 3D zone reaches 100% coverage
REWARD_FRONTIER_BONUS            = 5.0    # Bonus for expanding active 3D scanning frontier
REWARD_FAILED_ZONE_CELL_COVERED  = 10.0   # Backfill reward for scanning leftover/unmapped zones

# 2. Minor Secondary Efficiency & Penalty Terms (Explicitly Non-Dominant)
PENALTY_PER_STEP                 = -0.001 # Minimal step penalty (20,000x smaller than 100% coverage bonus)
PENALTY_REVISIT_VOXEL            = -0.01  # Minor penalty for redundant voxel rescans
REWARD_EFFICIENCY_BONUS          = 5.0    # Minor tiebreaker bonus for completing scan in fewer steps (max 5.0 << 200.0)

# 3. Reused V18 Safety & Navigation Mechanics
PENALTY_DRONE_DESTRUCTION         = -500.0 # High penalty for physical collision/destruction
PENALTY_PATH_CONFLICT_WARNING     = -30.0  # Proactive path conflict penalty
REWARD_VERTICAL_SEPARATION_BONUS  = 5.0    # Proactive altitude separation bonus
REWARD_OBSTACLE_FLYOVER_BONUS    = 2.0    # Altitude flyover bonus

# Repulsion Field Constants
REPULSION_FIELD_RADIUS   = 2.5
REPULSION_FORCE_STRENGTH = 0.8
REPULSION_PUSH_THRESHOLD = 0.35

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


class SearchAndRescueAreaMappingEnvV2(ParallelEnv):
    """
    Area Mapping V2: True 3D LiDAR Raycasting Environment with Reused V18 Collision Avoidance & Path Comm.
    """
    metadata = {"name": "search_rescue_area_mapping_v2", "render_modes": []}

    def __init__(self, grid_size=GRID_SIZE, num_agents=NUM_AGENTS,
                 max_steps=MAX_STEPS_SAFETY_CEILING, curriculum_stage=1.0,
                 render_mode=None, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents_cfg = num_agents
        self.max_steps = max_steps
        self.curriculum_stage = float(np.clip(curriculum_stage, 0.0, 1.0))
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents_cfg)]
        self.agents = self.possible_agents[:]

        self._action_spaces = {a: spaces.Discrete(7) for a in self.possible_agents}
        # Obs space: 432 dims (4 self + 4 teammates * 7 path comm + 400 flattened 10x10x4 grid)
        self._observation_spaces = {a: spaces.Box(low=-1.0, high=1.0, shape=(432,), dtype=np.float32) for a in self.possible_agents}

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
        self.episode_collisions_count = 0
        self.episode_repulsion_count = 0
        self.episode_redundant_rescans = 0
        self.drones_destroyed_count = 0

        self.scanned_voxels = set()
        self.visited_grid_2d = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Generate 3D Environment Obstacles & Buildings
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
                self.obstacle_heights[pos] = 1  # 10m height structure
            else:
                self.buildings.add(pos)
                self.obstacle_heights[pos] = 2  # 20m height building

        free_cells = [c for c in all_cells if c not in self.obstacles and c not in self.buildings]

        # Calculate Total Scannable 3D Voxels (Surface & Air spatial voxels above structures)
        self.total_walkable_voxels = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell_h = self.obstacle_heights.get((r, c), 0)
                for alt in range(MAX_ALTITUDE + 1):
                    if alt >= cell_h:
                        self.total_walkable_voxels.add((r, c, alt))

        # Zone Partitioning
        self.agent_zones = {}
        rows_per = GRID_SIZE // self.n_agents_cfg
        for i, a in enumerate(self.possible_agents):
            z_start = i * rows_per
            z_end = (i + 1) * rows_per if i < self.n_agents_cfg - 1 else GRID_SIZE
            self.agent_zones[a] = (z_start, z_end)

        self.zone_voxels = {}
        for a in self.possible_agents:
            z_start, z_end = self.agent_zones[a]
            self.zone_voxels[a] = set(v for v in self.total_walkable_voxels if z_start <= v[0] < z_end)

        # Spawn Agents in 3D
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
            self._perform_3d_lidar_scan(a)

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return observations, infos

    def trigger_failure(self, agent_name):
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

    def _perform_3d_lidar_scan(self, agent_name):
        """Simulate true 3D LiDAR raycasting from drone's current (X, Y, Z) position across spherical field of view."""
        if agent_name in self.failed_agents:
            return 0
        r, c = self.agent_positions[agent_name]
        alt = self.agent_altitudes[agent_name]

        scanned_new = 0
        # True 3D Spherical Raycasting directions (26 direction vectors covering 360 deg azimuth and full elevation)
        ray_dirs = [
            (dr, dc, dalt)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            for dalt in [-1, 0, 1]
            if not (dr == 0 and dc == 0 and dalt == 0)
        ]

        # Scan drone's current voxel location first
        curr_voxel = (r, c, alt)
        if curr_voxel in self.total_walkable_voxels and curr_voxel not in self.scanned_voxels:
            self.scanned_voxels.add(curr_voxel)
            self.visited_grid_2d[r, c] = 1.0
            scanned_new += 1

        # Cast rays in full 3D spherical pattern up to range R = 2.0 cells
        for dr, dc, dalt in ray_dirs:
            # Ray propagation steps (t = 1.0, 2.0)
            norm = np.sqrt(dr**2 + dc**2 + dalt**2)
            step_r, step_c, step_alt = dr / norm, dc / norm, dalt / norm
            for dist in [1.0, 2.0]:
                nr = int(round(r + step_r * dist))
                nc = int(round(c + step_c * dist))
                n_alt = int(round(alt + step_alt * dist))

                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and 0 <= n_alt <= MAX_ALTITUDE:
                    cell_h = self.obstacle_heights.get((nr, nc), 0)
                    # Check 3D geometry intersection: ray blocked if n_alt < cell_h (hits wall/roof surface)
                    if n_alt < cell_h:
                        # Ray hits 3D surface geometry, record surface hit voxel & terminate ray
                        surface_voxel = (nr, nc, n_alt)
                        if surface_voxel not in self.scanned_voxels:
                            self.scanned_voxels.add(surface_voxel)
                            self.visited_grid_2d[nr, nc] = 1.0
                            scanned_new += 1
                        break  # Ray occluded by structure
                    else:
                        voxel = (nr, nc, n_alt)
                        if voxel in self.total_walkable_voxels:
                            if voxel not in self.scanned_voxels:
                                self.scanned_voxels.add(voxel)
                                self.visited_grid_2d[nr, nc] = 1.0
                                scanned_new += 1

        return scanned_new

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

        # 4 Teammates x 7 features (includes path communication intended movement vector)
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

        # Flattened Grid (10x10x4)
        idx_flat = 32
        for gr in range(GRID_SIZE):
            for gc in range(GRID_SIZE):
                obs[idx_flat] = self.visited_grid_2d[gr, gc]
                obs[idx_flat+1] = 1.0 if (gr, gc) in self.obstacles else (2.0 if (gr, gc) in self.buildings else 0.0)
                scanned_layers = sum(1 for a_l in range(MAX_ALTITUDE + 1) if (gr, gc, a_l) in self.scanned_voxels)
                obs[idx_flat+2] = scanned_layers / (MAX_ALTITUDE + 1)
                obs[idx_flat+3] = 0.0
                idx_flat += 4

        return obs

    def step(self, actions):
        rewards = {a: 0.0 for a in self.possible_agents}
        prev_positions = {a: self.agent_positions[a] for a in self.possible_agents}
        active_agents = [a for a in self.possible_agents if a not in self.failed_agents]

        # 1. Update Intended Movement Vectors (Inter-Drone Path Communication)
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

        # 2. Proactive Path Communication & Elevation Conflict Analysis
        for i in range(len(active_agents)):
            a1 = active_agents[i]
            p1_intent = intent_pos[a1]
            alt1_intent = intent_alt[a1]

            for j in range(i + 1, len(active_agents)):
                a2 = active_agents[j]
                p2_intent = intent_pos[a2]
                alt2_intent = intent_alt[a2]

                if p1_intent == p2_intent:
                    if alt1_intent == alt2_intent:
                        rewards[a1] += PENALTY_PATH_CONFLICT_WARNING
                        rewards[a2] += PENALTY_PATH_CONFLICT_WARNING
                    else:
                        rewards[a1] += REWARD_VERTICAL_SEPARATION_BONUS
                        rewards[a2] += REWARD_VERTICAL_SEPARATION_BONUS

        # 3. Continuous Magnetic Repulsion Field Safety Net
        for agent in active_agents:
            r, c = intent_pos[agent]
            alt = intent_alt[agent]

            F_r, F_c, F_alt = 0.0, 0.0, 0.0
            for other_a in active_agents:
                if other_a == agent:
                    continue
                or_pos, oc_pos = intent_pos[other_a]
                oalt = intent_alt[other_a]

                dist = np.linalg.norm(np.array([r, c, alt]) - np.array([or_pos, oc_pos, oalt]))
                if 0.001 < dist < REPULSION_FIELD_RADIUS:
                    force_mag = REPULSION_FORCE_STRENGTH * ((1.0 - dist / REPULSION_FIELD_RADIUS) ** 2)
                    F_r += ((r - or_pos) / dist) * force_mag
                    F_c += ((c - oc_pos) / dist) * force_mag
                    F_alt += ((alt - oalt) / dist) * force_mag

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in self.obstacle_heights:
                        h = self.obstacle_heights[(nr, nc)]
                        if alt < h:
                            d_obs = np.sqrt(dr*dr + dc*dc) + 0.01
                            if d_obs < REPULSION_FIELD_RADIUS:
                                force_mag = REPULSION_FORCE_STRENGTH * ((1.0 - d_obs / REPULSION_FIELD_RADIUS) ** 2)
                                F_r -= (dr / d_obs) * force_mag
                                F_c -= (dc / d_obs) * force_mag
                                F_alt += 0.5 * force_mag

            if F_alt > REPULSION_PUSH_THRESHOLD:
                alt = int(np.clip(alt + 1, 0, MAX_ALTITUDE))
                self.episode_repulsion_count += 1

            self.agent_positions[agent] = (r, c)
            self.agent_altitudes[agent] = alt

        # 4. Destruction & Collision Verification
        destroyed_this_step = set()

        for agent in active_agents:
            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]
            cell_height = self.obstacle_heights.get((r, c), 0)
            if cell_height > 0 and alt < cell_height:
                destroyed_this_step.add(agent)
            elif cell_height > 0 and alt >= cell_height:
                rewards[agent] += REWARD_OBSTACLE_FLYOVER_BONUS

        for i in range(len(active_agents)):
            a1 = active_agents[i]
            if a1 in destroyed_this_step: continue
            p1 = np.array(self.agent_positions[a1])
            alt1 = self.agent_altitudes[a1]

            for j in range(i + 1, len(active_agents)):
                a2 = active_agents[j]
                if a2 in destroyed_this_step: continue
                p2 = np.array(self.agent_positions[a2])
                alt2 = self.agent_altitudes[a2]

                if np.linalg.norm(p1 - p2) < 1.0:
                    if alt1 == alt2:
                        destroyed_this_step.add(a1)
                        destroyed_this_step.add(a2)
                        self.episode_collisions_count += 1

        for d_agent in destroyed_this_step:
            self.failed_agents.add(d_agent)
            rewards[d_agent] += PENALTY_DRONE_DESTRUCTION
            self.drones_destroyed_count += 1

        still_active = [a for a in active_agents if a not in self.failed_agents]

        # 5. True 3D LiDAR Scanning & Dominant Coverage Rewards
        for agent in still_active:
            r, c = self.agent_positions[agent]
            z_start, z_end = self.agent_zones[agent]
            rewards[agent] += PENALTY_PER_STEP

            scanned_new = self._perform_3d_lidar_scan(agent)
            if scanned_new > 0:
                rewards[agent] += REWARD_NEW_VOXEL_MAPPED * scanned_new
                rewards[agent] += REWARD_FRONTIER_BONUS
                if not (z_start <= r < z_end):
                    rewards[agent] += REWARD_FAILED_ZONE_CELL_COVERED
            else:
                rewards[agent] += PENALTY_REVISIT_VOXEL
                self.episode_redundant_rescans += 1

            # Check 100% Zone Completion Bonus
            z_voxels = self.zone_voxels[agent]
            if z_voxels and z_voxels.issubset(self.scanned_voxels):
                rewards[agent] += REWARD_ZONE_100PCT_COMPLETE

        self.step_count += 1

        # Completion-Based Episode Termination
        all_mapped = self.total_walkable_voxels.issubset(self.scanned_voxels)
        safety_ceiling = self.step_count >= self.max_steps
        all_destroyed = len(still_active) == 0

        done = all_mapped or safety_ceiling or all_destroyed

        if all_mapped:
            for agent in still_active:
                rewards[agent] += REWARD_100PCT_COVERAGE_COMPLETE
                # Minor secondary tiebreaker bonus for completing in fewer steps (max 5.0 << 200.0)
                efficiency_bonus = max(0.0, REWARD_EFFICIENCY_BONUS * (1.0 - self.step_count / float(self.max_steps)))
                rewards[agent] += efficiency_bonus

        scanned_walkable = len(self.scanned_voxels.intersection(self.total_walkable_voxels))
        coverage_pct = min(100.0, (scanned_walkable / max(1, len(self.total_walkable_voxels))) * 100.0)

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: done for a in self.possible_agents}
        infos = {
            a: {
                "collisions": self.episode_collisions_count,
                "drones_destroyed": self.drones_destroyed_count,
                "coverage_pct": round(coverage_pct, 1),
                "redundant_rescans": self.episode_redundant_rescans,
                "all_mapped": all_mapped,
                "scanned_voxels": len(self.scanned_voxels),
                "total_voxels": len(self.total_walkable_voxels)
            } for a in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos


def parallel_area_mapping_env_v2(**kwargs):
    return SearchAndRescueAreaMappingEnvV2(**kwargs)
