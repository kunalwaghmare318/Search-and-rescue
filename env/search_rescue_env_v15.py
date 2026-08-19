import numpy as np
from gymnasium import spaces
try:
    from pettingzoo import ParallelEnv
except ImportError:
    class ParallelEnv:
        pass

# ==============================================================================
# ENVIRONMENT CONFIGURATION: V15 (3D ELEVATION / ALTITUDE COLLISION AVOIDANCE)
# Extends V14 with 3D elevation module:
#   - Action space: Discrete(7) -> [Stay, Up, Down, Left, Right, Ascend, Descend]
#   - Altitude levels: 0 (Ground/Search), 1 (Mid/Flyover), 2 (High)
#   - Inter-drone 3D collision avoidance: zero collision when vertically separated
#   - Obstacle flyover: drones can ascend over buildings/obstacles
#   - Extended obs space (434 dims): includes self altitude, teammate altitudes, 
#     relative elevation, and 3D proximity sensor
# ==============================================================================
GRID_SIZE = 10
NUM_AGENTS = 5
NUM_SURVIVORS = 10
NUM_OBSTACLES = 8
NUM_BUILDINGS = 6
MAX_STEPS = 100
MAX_ALTITUDE = 2

DETECTION_RADIUS_OPEN = 1.5
DETECTION_RADIUS_THERMAL_OCCLUDED = 1.8

# Reward & Penalty Constants (inherited from V12/V14)
REWARD_NEW_CELL_COVERED           = 1.0
REWARD_SURVIVOR_FOUND             = 50.0
PENALTY_PER_STEP                  = -0.01
PENALTY_REVISIT_CELL              = -0.1

# Zone Coverage
REWARD_ZONE_COVERAGE_BONUS        = 0.5
REWARD_ZONE_100PCT_COMPLETE       = 5.0
REWARD_NEAREST_UNCHECKED_COVERED  = 1.5
REWARD_ZONE_PRESENCE              = 0.3
PENALTY_OUT_OF_ZONE               = -0.10

# V14 Failure Recovery
REWARD_FAILED_ZONE_CELL_COVERED   = 2.0
REWARD_FAILED_ZONE_COMPLETE       = 8.0
FAILURE_INJECTION_PROB            = 0.3
FAILURE_EARLIEST_STEP_FRAC        = 0.15
FAILURE_LATEST_STEP_FRAC          = 0.50

# V15 3D Elevation Avoidance Rewards & Penalties
REWARD_VERTICAL_SEPARATION_BONUS  = 0.5   # Awarded when drones pass horizontally close but separated vertically
REWARD_OBSTACLE_FLYOVER_BONUS     = 0.4   # Awarded when flying over obstacle/building at safe altitude
REWARD_ZERO_COLLISION_EPISODE     = 10.0  # Episode end bonus if 0 collisions occurred
PENALTY_ELEVATION_COLLISION       = -2.0  # Penalty for same-altitude horizontal collision

# Smooth Repulsion (Active only when at same altitude)
PERSONAL_SPACE_RADIUS = 2.5
REPULSION_MAX_PENALTY = 2.0

# Anti-Stall & Anti-Stuck Loop
PENALTY_ANTI_STALL       = -0.2
STALL_THRESHOLD          = 2
STALL_WINDOW             = 8
STUCK_WINDOW             = 10
STUCK_UNIQUE_THRESHOLD   = 2
PENALTY_STUCK_LOOP_BASE  = -0.5
PENALTY_STUCK_LOOP_ESCAL = -0.3
STUCK_ESCAL_CAP          = 5


class SearchAndRescueEnvV15(ParallelEnv):
    """
    V15: Swarm Search & Rescue with 3D Elevation & Altitude Collision Avoidance.
    Extends V14 with 7-Discrete action space, 3D elevation sensors, and vertical avoidance rewards.
    """
    metadata = {"name": "search_and_rescue_v15", "render_modes": []}

    def __init__(self, grid_size=GRID_SIZE, num_agents=NUM_AGENTS,
                 num_survivors=NUM_SURVIVORS, num_obstacles=NUM_OBSTACLES,
                 num_buildings=NUM_BUILDINGS, max_steps=MAX_STEPS,
                 render_mode=None,
                 failure_injection_prob=FAILURE_INJECTION_PROB,
                 forced_failure_agent=None, forced_failure_step=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents_cfg = num_agents
        self.num_survivors = num_survivors
        self.num_obstacles = num_obstacles
        self.num_buildings = num_buildings
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.failure_injection_prob = failure_injection_prob
        self.forced_failure_agent = forced_failure_agent
        self.forced_failure_step = forced_failure_step

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents_cfg)]
        self.agents = self.possible_agents[:]

        # V15: Discrete(7) Action Space [0:Stay, 1:Up, 2:Down, 3:Left, 4:Right, 5:Ascend, 6:Descend]
        self._action_spaces = {a: spaces.Discrete(7) for a in self.possible_agents}

        # Obs dim: 
        # ID(5) + SelfPos(2) + SelfAlt(1) + OthersPos(8) + OthersAlt(4) + RelAlt(4) + ElevSensor(5) + FailedFlags(5) + 4 Grids(400) = 434
        self.obs_dim = (self.n_agents_cfg + 2 + 1 + 
                        (self.n_agents_cfg - 1) * 2 + 
                        (self.n_agents_cfg - 1) + 
                        (self.n_agents_cfg - 1) + 5 + 
                        self.n_agents_cfg + 
                        (4 * GRID_SIZE * GRID_SIZE))

        self._observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

        self.agent_zones = {}
        for i, a in enumerate(self.possible_agents):
            self.agent_zones[a] = (i * 2, (i + 1) * 2)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _compute_obstacle_aware_zones(self):
        walkable_per_row = np.zeros(GRID_SIZE)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) not in self.obstacles and (r, c) not in self.buildings:
                    walkable_per_row[r] += 1
        total_walkable = walkable_per_row.sum()
        if total_walkable == 0:
            return
        target_per_agent = total_walkable / self.n_agents_cfg
        remaining_agents = self.n_agents_cfg
        zones = {}
        current_row = 0
        for i, a in enumerate(self.possible_agents):
            remaining_agents -= 1
            start_row = current_row
            if i == self.n_agents_cfg - 1:
                end_row = GRID_SIZE
            else:
                accumulated = 0.0
                max_row = GRID_SIZE - remaining_agents
                while current_row < max_row and accumulated < target_per_agent:
                    accumulated += walkable_per_row[current_row]
                    current_row += 1
                if current_row == start_row:
                    current_row = min(start_row + 1, GRID_SIZE - remaining_agents)
                end_row = current_row
            zones[a] = (start_row, end_row)
            current_row = end_row
        self.agent_zones = zones

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

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.failed_agents = set()
        self.failure_step_map = {}
        self.reassigned_agents = {}
        self.episode_collisions_count = 0

        # Episode failure plan
        self.episode_failure_agent = None
        self.episode_failure_step = None
        if self.forced_failure_agent and self.forced_failure_step:
            self.episode_failure_agent = self.forced_failure_agent
            self.episode_failure_step = self.forced_failure_step
        elif np.random.rand() < self.failure_injection_prob:
            self.episode_failure_agent = np.random.choice(self.possible_agents)
            self.episode_failure_step = int(np.random.randint(
                int(MAX_STEPS * FAILURE_EARLIEST_STEP_FRAC),
                int(MAX_STEPS * FAILURE_LATEST_STEP_FRAC) + 1
            ))

        # Generate obstacles and buildings
        all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        np.random.shuffle(all_cells)

        idx = 0
        self.obstacles = set(all_cells[idx:idx + self.num_obstacles])
        idx += self.num_obstacles

        self.buildings = set(all_cells[idx:idx + self.num_buildings])
        idx += self.num_buildings

        # Obstacle/Building Heights: Level 1 for obstacles/buildings
        self.obstacle_heights = {cell: 1 for cell in self.obstacles.union(self.buildings)}

        self._compute_obstacle_aware_zones()

        # Place survivors
        valid_surv_cells = [c for c in all_cells if c not in self.obstacles]
        surv_indices = np.random.choice(len(valid_surv_cells), self.num_survivors, replace=False)
        self.survivors = set(valid_surv_cells[i] for i in surv_indices)

        self.survivors_hidden = set(s for s in self.survivors if s in self.buildings)
        self.survivors_open = self.survivors - self.survivors_hidden

        self.initial_survivors_count = len(self.survivors)
        self.initial_open_count = len(self.survivors_open)
        self.initial_hidden_count = len(self.survivors_hidden)

        # Agent initial positions (start in assigned zone) & altitudes
        self.agent_positions = {}
        self.agent_altitudes = {}
        for a in self.possible_agents:
            z_start, z_end = self.agent_zones[a]
            zone_cells = [(r, c) for r in range(z_start, z_end) for c in range(GRID_SIZE)
                          if (r, c) not in self.obstacles and (r, c) not in self.buildings]
            if not zone_cells:
                zone_cells = [(r, c) for r in range(z_start, z_end) for c in range(GRID_SIZE)]
            start_pos = zone_cells[np.random.choice(len(zone_cells))]
            self.agent_positions[a] = start_pos
            self.agent_altitudes[a] = 0  # Start at ground search altitude

        # Grids
        self.visited_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for a, pos in self.agent_positions.items():
            self.visited_grid[pos[0], pos[1]] = 1.0

        self.obstacle_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r, c in self.obstacles.union(self.buildings):
            self.obstacle_grid[r, c] = 1.0

        self.building_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r, c in self.buildings:
            self.building_grid[r, c] = 1.0

        self.agent_histories = {a: [self.agent_positions[a]] for a in self.possible_agents}
        self.agent_stuck_escalation = {a: 0 for a in self.possible_agents}

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return observations, infos

    def _get_obs(self, agent_name):
        agent_idx = int(agent_name.split("_")[1])

        # 1. One-hot agent ID (5)
        one_hot = np.zeros(self.n_agents_cfg, dtype=np.float32)
        one_hot[agent_idx] = 1.0

        # 2. Self pos & altitude (3)
        self_pos = self.agent_positions[agent_name]
        self_alt = self.agent_altitudes[agent_name]
        self_norm = np.array([self_pos[0] / 10.0, self_pos[1] / 10.0, self_alt / 2.0], dtype=np.float32)

        # 3. Teammate pos & altitudes (8 + 4 + 4 = 16)
        others_pos = []
        others_alt = []
        rel_alt = []
        for a in self.possible_agents:
            if a != agent_name:
                p = self.agent_positions[a]
                alt = self.agent_altitudes[a]
                others_pos.extend([p[0] / 10.0, p[1] / 10.0])
                others_alt.append(alt / 2.0)
                rel_alt.append((self_alt - alt) / 2.0)

        others_pos_arr = np.array(others_pos, dtype=np.float32)
        others_alt_arr = np.array(others_alt, dtype=np.float32)
        rel_alt_arr = np.array(rel_alt, dtype=np.float32)

        # 4. Elevation sensor: obstacle heights in current cell + 4 adjacent cells (5)
        r, c = self_pos
        adjacent_cells = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        elev_sensor = []
        for cr, cc in adjacent_cells:
            if 0 <= cr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                height = self.obstacle_heights.get((cr, cc), 0)
                elev_sensor.append(height / 2.0)
            else:
                elev_sensor.append(1.0)  # Boundary acting as high obstacle
        elev_sensor_arr = np.array(elev_sensor, dtype=np.float32)

        # 5. Failed agent flags (5)
        failed_flags = np.zeros(self.n_agents_cfg, dtype=np.float32)
        for i, a in enumerate(self.possible_agents):
            if a in self.failed_agents:
                failed_flags[i] = 1.0

        # 6. Grid maps (400)
        survivor_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r, c in self.survivors:
            survivor_grid[r, c] = 1.0

        visited_flat = self.visited_grid.flatten()
        obstacle_flat = self.obstacle_grid.flatten()
        building_flat = self.building_grid.flatten()
        survivor_flat = survivor_grid.flatten()

        return np.concatenate([
            one_hot, self_norm, others_pos_arr, others_alt_arr, rel_alt_arr,
            elev_sensor_arr, failed_flags,
            visited_flat, obstacle_flat, building_flat, survivor_flat
        ])

    def step(self, actions):
        self.step_count += 1

        # Check random failure injection
        if self.episode_failure_agent and self.step_count == self.episode_failure_step:
            self.trigger_failure(self.episode_failure_agent)

        rewards = {a: 0.0 for a in self.possible_agents}

        # 1. Update Positions & Altitudes for Active Drones
        prev_positions = {a: self.agent_positions[a] for a in self.possible_agents}
        prev_altitudes = {a: self.agent_altitudes[a] for a in self.possible_agents}

        for agent, action in actions.items():
            if agent in self.failed_agents:
                continue  # Failed drone freeze

            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]

            # Actions: 0:Stay, 1:Up, 2:Down, 3:Left, 4:Right, 5:Ascend, 6:Descend
            if action == 1:
                r = max(0, r - 1)
            elif action == 2:
                r = min(GRID_SIZE - 1, r + 1)
            elif action == 3:
                c = max(0, c - 1)
            elif action == 4:
                c = min(GRID_SIZE - 1, c + 1)
            elif action == 5:
                alt = min(MAX_ALTITUDE, alt + 1)
            elif action == 6:
                alt = max(0, alt - 1)

            # Check obstacle/building collision at current altitude
            cell = (r, c)
            cell_height = self.obstacle_heights.get(cell, 0)

            if cell_height > 0 and alt < cell_height:
                # Collision with building/obstacle at low altitude -> blocked & penalized
                rewards[agent] += PENALTY_ELEVATION_COLLISION
                self.episode_collisions_count += 1
                # Movement blocked
                r, c = prev_positions[agent]
            elif cell_height > 0 and alt >= cell_height:
                # Successful flyover bonus
                rewards[agent] += REWARD_OBSTACLE_FLYOVER_BONUS

            self.agent_positions[agent] = (r, c)
            self.agent_altitudes[agent] = alt
            self.visited_grid[r, c] = 1.0

        # 2. 3D Inter-Drone Collision & Vertical Separation Rewards
        active_agents = [a for a in self.possible_agents if a not in self.failed_agents]
        for i in range(len(active_agents)):
            a1 = active_agents[i]
            p1 = np.array(self.agent_positions[a1])
            alt1 = self.agent_altitudes[a1]

            for j in range(i + 1, len(active_agents)):
                a2 = active_agents[j]
                p2 = np.array(self.agent_positions[a2])
                alt2 = self.agent_altitudes[a2]

                horiz_dist = np.linalg.norm(p1 - p2)
                alt_diff = abs(alt1 - alt2)

                if horiz_dist < 1.0:
                    if alt_diff == 0:
                        # Same altitude horizontal collision
                        rewards[a1] += PENALTY_ELEVATION_COLLISION
                        rewards[a2] += PENALTY_ELEVATION_COLLISION
                        self.episode_collisions_count += 1
                        # Resolve collision: ascend a2 to a safe altitude layer to break deadlock
                        self.agent_altitudes[a2] = min(MAX_ALTITUDE, self.agent_altitudes[a2] + 1)
                    else:
                        # Vertically separated flyover bonus!
                        rewards[a1] += REWARD_VERTICAL_SEPARATION_BONUS
                        rewards[a2] += REWARD_VERTICAL_SEPARATION_BONUS
                elif horiz_dist < PERSONAL_SPACE_RADIUS and alt_diff == 0:
                    # Same altitude smooth repulsion penalty
                    penalty = REPULSION_MAX_PENALTY * (1.0 - horiz_dist / PERSONAL_SPACE_RADIUS) ** 2
                    rewards[a1] -= penalty
                    rewards[a2] -= penalty

        # 3. Survivor Detection & Coverage Rewards
        for agent in active_agents:
            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]

            # Step penalty
            rewards[agent] += PENALTY_PER_STEP

            # Zone presence & search efficiency at search altitude (alt == 0)
            z_start, z_end = self.agent_zones[agent]
            if z_start <= r < z_end:
                rewards[agent] += REWARD_ZONE_PRESENCE
                if alt == 0:
                    rewards[agent] += 0.2  # Search efficiency bonus at ground altitude

            # Survivor Detection (Active when at alt == 0 or alt == 1)
            if alt <= 1:
                det_rad = DETECTION_RADIUS_OPEN if alt == 0 else 1.0
                found_now = set()
                for s in self.survivors:
                    if (s in self.buildings and np.linalg.norm(np.array([r, c]) - np.array(s)) <= DETECTION_RADIUS_THERMAL_OCCLUDED) or \
                       (s not in self.buildings and np.linalg.norm(np.array([r, c]) - np.array(s)) <= det_rad):
                        found_now.add(s)

                if found_now:
                    rewards[agent] += REWARD_SURVIVOR_FOUND * len(found_now)
                    self.survivors -= found_now
                    self.survivors_open -= found_now
                    self.survivors_hidden -= found_now

        # 4. Anti-Stall / Anti-Stuck Loop
        for agent in active_agents:
            self.agent_histories[agent].append(self.agent_positions[agent])
            if len(self.agent_histories[agent]) > STUCK_WINDOW:
                self.agent_histories[agent].pop(0)

            if len(self.agent_histories[agent]) == STUCK_WINDOW:
                unique_pos = len(set(self.agent_histories[agent]))
                if unique_pos <= STUCK_UNIQUE_THRESHOLD:
                    self.agent_stuck_escalation[agent] += 1
                    escal = min(self.agent_stuck_escalation[agent], STUCK_ESCAL_CAP)
                    rewards[agent] += PENALTY_STUCK_LOOP_BASE + (PENALTY_STUCK_LOOP_ESCAL * escal)

        # Check termination: scan until 100% grid coverage or max steps
        all_covered = float(self.visited_grid.sum()) >= (GRID_SIZE ** 2)
        terminated = all_covered
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        if done and self.episode_collisions_count == 0:
            for agent in active_agents:
                rewards[agent] += REWARD_ZERO_COLLISION_EPISODE

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: done for a in self.possible_agents}
        infos = {a: {"collisions": self.episode_collisions_count} for a in self.possible_agents}

        return observations, rewards, terminations, truncations, infos


def parallel_env_v15(**kwargs):
    return SearchAndRescueEnvV15(**kwargs)
