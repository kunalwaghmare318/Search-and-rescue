import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# ==============================================================================
# ENVIRONMENT CONFIGURATION: V17 UNKNOWN SURVIVOR COUNT & ONE-LIFE DESTRUCTION
# Key Innovations:
#   1. Unknown Survivor Count: No survivor count/position signal in obs space.
#      Framed strictly around Search Completeness (area coverage).
#   2. One-Life Collision Destruction: Collision permanently destroys drone.
#   3. Delayed Reassignment: Priority ordering via reward shaping (own zone first, then backfill).
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

# Reward & Penalty Constants
REWARD_NEW_CELL_COVERED           = 4.0
REWARD_SURVIVOR_FOUND             = 50.0
PENALTY_PER_STEP                  = -0.01
PENALTY_REVISIT_CELL              = -0.1
PENALTY_IDLE                      = -1.0
REWARD_UNCHECKED_PROGRESS         = 0.5

# Zone Coverage
REWARD_ZONE_COVERAGE_BONUS        = 1.0
REWARD_ZONE_100PCT_COMPLETE       = 10.0
REWARD_ZONE_PRESENCE              = 0.2
PENALTY_OUT_OF_ZONE               = -0.05
REWARD_EXHAUSTIVE_SEARCH_COMPLETE = 50.0

# Failure & Reassignment Priorities
REWARD_FAILED_ZONE_CELL_COVERED   = 4.0
REWARD_FAILED_ZONE_COMPLETE       = 10.0

# 3D Elevation & One-Life Destruction Penalties
REWARD_VERTICAL_SEPARATION_BONUS  = 1.0
REWARD_OBSTACLE_FLYOVER_BONUS     = 0.4
REWARD_ZERO_COLLISION_EPISODE     = 20.0
PENALTY_DRONE_DESTRUCTION         = -25.0

# Repulsion Field Constants
REPULSION_FIELD_RADIUS   = 2.5
REPULSION_FORCE_STRENGTH = 0.8
REPULSION_PUSH_THRESHOLD = 0.35

# Anti-Stall & Anti-Stuck Loop
PENALTY_ANTI_STALL       = -0.2
STALL_THRESHOLD          = 2
STALL_WINDOW             = 8
STUCK_WINDOW             = 8
STUCK_UNIQUE_THRESHOLD   = 2
PENALTY_STUCK_LOOP_BASE  = -0.3
PENALTY_STUCK_LOOP_ESCAL  = -0.2
STUCK_ESCAL_CAP          = 5


class SearchAndRescueEnvV17(ParallelEnv):
    """
    V17 Environment: Exhaustive Search under Unknown Survivor Count, One-Life Destruction,
    and Delayed Zone Reassignment.
    """
    metadata = {"name": "search_and_rescue_v17", "render_modes": []}

    def __init__(self, grid_size=GRID_SIZE, num_agents=NUM_AGENTS,
                 num_survivors=NUM_SURVIVORS, num_obstacles=NUM_OBSTACLES,
                 num_buildings=NUM_BUILDINGS, max_steps=MAX_STEPS,
                 render_mode=None, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents_cfg = num_agents
        self.num_survivors = num_survivors
        self.num_obstacles = num_obstacles
        self.num_buildings = num_buildings
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents_cfg)]
        self.agents = self.possible_agents[:]

        self._action_spaces = {a: spaces.Discrete(7) for a in self.possible_agents}
        # Dim: 434 (same feature shape, survivor grid replaced with found survivors only)
        self._observation_spaces = {a: spaces.Box(low=-1.0, high=1.0, shape=(434,), dtype=np.float32) for a in self.possible_agents}

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
        self.drones_destroyed_count = 0

        self.visited_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.found_survivor_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.obstacles = set()
        self.buildings = set()
        self.obstacle_heights = {}

        all_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        np.random.shuffle(all_cells)

        idx = 0
        for _ in range(self.num_obstacles):
            pos = all_cells[idx]
            self.obstacles.add(pos)
            self.obstacle_heights[pos] = 1
            idx += 1

        for _ in range(self.num_buildings):
            pos = all_cells[idx]
            self.buildings.add(pos)
            self.obstacle_heights[pos] = 2
            idx += 1

        free_cells = [c for c in all_cells[idx:] if c not in self.obstacles and c not in self.buildings]

        self.survivors = set(free_cells[:self.num_survivors])
        hidden_count = np.random.randint(3, 5)
        surv_list = list(self.survivors)
        self.survivors_hidden = set(surv_list[:hidden_count])
        self.survivors_open = set(surv_list[hidden_count:])
        self.found_survivors = set()
        self.initial_survivors_count = len(self.survivors)

        self.agent_zones = {}
        rows_per_agent = GRID_SIZE // self.n_agents_cfg
        for i, a in enumerate(self.possible_agents):
            z_start = i * rows_per_agent
            z_end = (i + 1) * rows_per_agent if i < self.n_agents_cfg - 1 else GRID_SIZE
            self.agent_zones[a] = (z_start, z_end)

        self.agent_positions = {}
        self.agent_altitudes = {}
        used = set()
        for a in self.possible_agents:
            z_start, z_end = self.agent_zones[a]
            zone_free = [c for c in free_cells if z_start <= c[0] < z_end and c not in used]
            if zone_free:
                pos = zone_free[np.random.randint(len(zone_free))]
            else:
                pos = [c for c in free_cells if c not in used][0]
            self.agent_positions[a] = pos
            self.agent_altitudes[a] = 0
            used.add(pos)
            self.visited_grid[pos[0], pos[1]] = 1.0

        self.failed_agents = set()  # Permanently destroyed drones
        self.failure_step_map = {}
        self.reassigned_agents = {}
        self.zone_walkable_cells = {}
        self.zone_visited_cells = {}
        self.zone_completed = {a: False for a in self.possible_agents}

        # Calculate total walkable cells for search completeness calculation
        self.total_walkable_cells = set(c for c in all_cells if c not in self.obstacles and c not in self.buildings)

        for a in self.possible_agents:
            z_start, z_end = self.agent_zones[a]
            walkable = set((r, c) for r in range(z_start, z_end) for c in range(GRID_SIZE) if (r, c) not in self.obstacles and (r, c) not in self.buildings)
            self.zone_walkable_cells[a] = walkable
            self.zone_visited_cells[a] = set([self.agent_positions[a]])

        self.agent_histories = {a: [self.agent_positions[a]] for a in self.possible_agents}
        self.agent_stuck_escalation = {a: 0 for a in self.possible_agents}
        self.exhaustive_search_reward_given = False

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

    def _get_obs(self, agent_name):
        obs = np.zeros(434, dtype=np.float32)
        if agent_name in self.failed_agents:
            return obs

        r, c = self.agent_positions[agent_name]
        alt = self.agent_altitudes[agent_name]
        obs[0] = r / (GRID_SIZE - 1)
        obs[1] = c / (GRID_SIZE - 1)
        obs[2] = self.step_count / self.max_steps
        obs[3] = alt / MAX_ALTITUDE

        idx = 4
        for a in self.possible_agents:
            if a == agent_name:
                continue
            if a in self.failed_agents:
                obs[idx:idx+4] = [-1.0, -1.0, -1.0, -1.0]
            else:
                tr, tc = self.agent_positions[a]
                talt = self.agent_altitudes[a]
                obs[idx] = tr / (GRID_SIZE - 1)
                obs[idx+1] = tc / (GRID_SIZE - 1)
                obs[idx+2] = talt / MAX_ALTITUDE
                obs[idx+3] = np.linalg.norm(np.array([r, c]) - np.array([tr, tc])) / (GRID_SIZE * np.sqrt(2))
            idx += 4

        idx_flat = 20
        for gr in range(GRID_SIZE):
            for gc in range(GRID_SIZE):
                obs[idx_flat] = self.visited_grid[gr, gc]
                obs[idx_flat+1] = 1.0 if (gr, gc) in self.obstacles else (2.0 if (gr, gc) in self.buildings else 0.0)
                # V17 Key Change: NO unfound survivor locations/count! Only found survivors.
                obs[idx_flat+2] = 1.0 if (gr, gc) in self.found_survivors else 0.0
                obs[idx_flat+3] = 1.0 if (gr, gc) in self.buildings else 0.0
                idx_flat += 4

        return obs

    def step(self, actions):
        rewards = {a: 0.0 for a in self.possible_agents}
        prev_positions = {a: self.agent_positions[a] for a in self.possible_agents}
        prev_altitudes = {a: self.agent_altitudes[a] for a in self.possible_agents}

        # 1. Action Intent Calculation for Active Drones
        intent_pos = {}
        intent_alt = {}
        active_agents = [a for a in self.possible_agents if a not in self.failed_agents]

        for agent, action in actions.items():
            if agent in self.failed_agents:
                continue
            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]

            if action == 1: r = max(0, r - 1)
            elif action == 2: r = min(GRID_SIZE - 1, r + 1)
            elif action == 3: c = max(0, c - 1)
            elif action == 4: c = min(GRID_SIZE - 1, c + 1)
            elif action == 5: alt = min(MAX_ALTITUDE, alt + 1)
            elif action == 6: alt = max(0, alt - 1)

            intent_pos[agent] = (r, c)
            intent_alt[agent] = alt

        # 2. Continuous Magnetic Repulsion Field Movement Shaping
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

            # Continuous Magnetic Repulsion Field Movement Shaping (3D Altitude & Safety)
            if F_alt > REPULSION_PUSH_THRESHOLD:
                alt = int(np.clip(alt + 1, 0, MAX_ALTITUDE))
                self.episode_repulsion_count += 1

            self.agent_positions[agent] = (r, c)
            self.agent_altitudes[agent] = alt

        # 3. One-Life Collision Destruction Logic
        destroyed_this_step = set()

        # A. Building/Obstacle Collision Destruction (moving into high building at low alt)
        for agent in active_agents:
            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]
            cell_height = self.obstacle_heights.get((r, c), 0)

            if cell_height > 0 and alt < cell_height:
                destroyed_this_step.add(agent)

        # B. Inter-drone 3D Collision Destruction (same position & same altitude)
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
                    else:
                        rewards[a1] += REWARD_VERTICAL_SEPARATION_BONUS
                        rewards[a2] += REWARD_VERTICAL_SEPARATION_BONUS

        # Apply Destruction
        for d_agent in destroyed_this_step:
            self.failed_agents.add(d_agent)
            rewards[d_agent] += PENALTY_DRONE_DESTRUCTION
            self.drones_destroyed_count += 1

        still_active = [a for a in active_agents if a not in self.failed_agents]

        # 4. Search, Coverage & Delayed Reassignment Rewards for Surviving Drones
        unvisited_remaining = np.sum(1.0 - self.visited_grid) > 0

        for agent in still_active:
            r, c = self.agent_positions[agent]
            alt = self.agent_altitudes[agent]
            act = actions.get(agent, 0)
            rewards[agent] += PENALTY_PER_STEP

            # Anti-Idle Penalty: Penalize staying still (Action 0) if area is not fully searched
            if act == 0 and unvisited_remaining:
                rewards[agent] += PENALTY_IDLE

            # Cell visited update
            new_cell = self.visited_grid[r, c] == 0.0
            self.visited_grid[r, c] = 1.0

            z_start, z_end = self.agent_zones[agent]
            in_own_zone = z_start <= r < z_end

            # Update zone coverage tracker
            if in_own_zone:
                self.zone_visited_cells[agent].add((r, c))
                if len(self.zone_visited_cells[agent]) >= max(1, len(self.zone_walkable_cells[agent])):
                    if not self.zone_completed[agent]:
                        self.zone_completed[agent] = True
                        rewards[agent] += REWARD_ZONE_100PCT_COMPLETE

                # Reward zone presence ONLY if agent is actively moving
                if act != 0:
                    rewards[agent] += REWARD_ZONE_PRESENCE
                if new_cell:
                    rewards[agent] += REWARD_NEW_CELL_COVERED
                elif act != 0:
                    rewards[agent] += PENALTY_REVISIT_CELL
            else:
                # Reassignment & Backfill Order:
                # If own zone is complete or target zone belongs to destroyed drone, NO out-of-zone penalty
                is_destroyed_zone = any(
                    self.agent_zones[d_agent][0] <= r < self.agent_zones[d_agent][1]
                    for d_agent in self.failed_agents
                )

                if self.zone_completed[agent] or is_destroyed_zone:
                    if new_cell:
                        rewards[agent] += REWARD_FAILED_ZONE_CELL_COVERED if is_destroyed_zone else REWARD_NEW_CELL_COVERED
                    elif act != 0:
                        rewards[agent] += PENALTY_REVISIT_CELL
                else:
                    if new_cell:
                        rewards[agent] += REWARD_NEW_CELL_COVERED
                    else:
                        rewards[agent] += PENALTY_OUT_OF_ZONE

            # Continuous Distance Progress reward toward nearest unvisited cell
            unvisited_cells = [(gr, gc) for (gr, gc) in self.total_walkable_cells if self.visited_grid[gr, gc] == 0.0]
            if unvisited_cells and act != 0:
                pr, pc = prev_positions[agent]
                min_dist_prev = min(abs(pr - ur) + abs(pc - uc) for ur, uc in unvisited_cells)
                min_dist_curr = min(abs(r - ur) + abs(c - uc) for ur, uc in unvisited_cells)
                if min_dist_curr < min_dist_prev:
                    rewards[agent] += REWARD_UNCHECKED_PROGRESS

            # Anti-Stuck Loop Detection
            self.agent_histories[agent].append((r, c))
            if len(self.agent_histories[agent]) > STUCK_WINDOW:
                self.agent_histories[agent].pop(0)
                recent_pos = self.agent_histories[agent]
                if len(set(recent_pos)) <= STUCK_UNIQUE_THRESHOLD and act != 0:
                    rewards[agent] += PENALTY_STUCK_LOOP_BASE

            # Survivor Detection (thermal & optical sensors scan ground level from any altitude <= MAX_ALTITUDE)
            det_rad = DETECTION_RADIUS_OPEN
            found_now = set()
            for s in self.survivors:
                dist = np.linalg.norm(np.array([r, c]) - np.array(s))
                if s in self.buildings:
                    if dist <= DETECTION_RADIUS_THERMAL_OCCLUDED:
                        found_now.add(s)
                else:
                    if dist <= det_rad:
                        found_now.add(s)

            if found_now:
                rewards[agent] += REWARD_SURVIVOR_FOUND * len(found_now)
                self.found_survivors.update(found_now)
                self.found_survivor_grid.fill(0.0)
                for fs in self.found_survivors:
                    self.found_survivor_grid[fs[0], fs[1]] = 1.0
                self.survivors -= found_now

        self.step_count += 1

        # Exhaustive Search Completion Check
        total_visited_cells = int(np.sum(self.visited_grid))
        total_possible = float(len(self.total_walkable_cells))
        if total_visited_cells >= total_possible and not self.exhaustive_search_reward_given:
            self.exhaustive_search_reward_given = True
            for agent in still_active:
                rewards[agent] += REWARD_EXHAUSTIVE_SEARCH_COMPLETE

        # Search completeness calculation (% of total walkable cells visited)
        visited_walkable = np.sum(self.visited_grid * (1.0 - self.obstacle_heights.get(None, np.zeros((GRID_SIZE, GRID_SIZE)))))
        all_visited_pct = len(np.argwhere(self.visited_grid == 1.0)) / float(GRID_SIZE * GRID_SIZE)
        search_completeness = (len(self.zone_visited_cells) / float(self.n_agents_cfg)) * 100.0

        truncated = self.step_count >= self.max_steps
        terminated = truncated or len(still_active) == 0  # End if max steps or all drones destroyed
        done = terminated or truncated

        if done and self.drones_destroyed_count == 0:
            for agent in still_active:
                rewards[agent] += REWARD_ZERO_COLLISION_EPISODE

        observations = {a: self._get_obs(a) for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: done for a in self.possible_agents}

        total_visited_cells = int(np.sum(self.visited_grid))
        total_possible = float(len(self.total_walkable_cells))
        actual_search_completeness_pct = round(min(100.0, (total_visited_cells / max(1.0, total_possible)) * 100.0), 1)

        infos = {
            a: {
                "collisions": self.episode_collisions_count,
                "drones_destroyed": self.drones_destroyed_count,
                "search_completeness_pct": actual_search_completeness_pct,
                "survivors_found_count": len(self.found_survivors),
                "total_survivors": self.initial_survivors_count
            } for a in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos


def parallel_env_v17(**kwargs):
    return SearchAndRescueEnvV17(**kwargs)


def apply_active_unsearched_navigation(env, actions):
    """Overrides idle (0) or stuck drone actions with dynamic pathfinding to nearest unsearched cell."""
    grid_size = getattr(env, "grid_size", 10)
    visited_grid = getattr(env, "visited_grid", None)
    if visited_grid is None:
        return actions

    obstacles = getattr(env, "obstacles", set())
    unsearched_cells = [
        (r, c) for r in range(grid_size) for c in range(grid_size)
        if visited_grid[r, c] == 0.0 and (r, c) not in obstacles
    ]
    if not unsearched_cells:
        return actions

    failed_agents = getattr(env, "failed_agents", set())
    agent_positions = getattr(env, "agent_positions", {})
    agent_altitudes = getattr(env, "agent_altitudes", {})
    agent_zones = getattr(env, "agent_zones", {})
    agent_histories = getattr(env, "agent_histories", {})
    obstacle_heights = getattr(env, "obstacle_heights", {})

    new_actions = actions.copy()
    for agent, act in actions.items():
        if agent in failed_agents or agent not in agent_positions:
            continue

        r, c = agent_positions[agent]
        alt = agent_altitudes.get(agent, 0)
        history = agent_histories.get(agent, [])
        is_stuck = len(history) >= 4 and len(set(history[-4:])) <= 2

        if act == 0 or is_stuck:
            z_start, z_end = agent_zones.get(agent, (0, grid_size))
            zone_unsearched = [uc for uc in unsearched_cells if z_start <= uc[0] < z_end]
            target_pool = zone_unsearched if zone_unsearched else unsearched_cells

            best_target = min(target_pool, key=lambda uc: abs(uc[0] - r) + abs(uc[1] - c))
            tr, tc = best_target

            desired_act = act
            if tr < r: desired_act = 1
            elif tr > r: desired_act = 2
            elif tc < c: desired_act = 3
            elif tc > c: desired_act = 4
            elif alt > 0: desired_act = 6 # Descend for close scanning

            if desired_act in (1, 2, 3, 4):
                dr = -1 if desired_act == 1 else (1 if desired_act == 2 else 0)
                dc = -1 if desired_act == 3 else (1 if desired_act == 4 else 0)
                nr, nc = r + dr, c + dc
                b_height = obstacle_heights.get((nr, nc), 0)
                if b_height > 0 and alt < b_height:
                    desired_act = 5 # Flyover building

            new_actions[agent] = desired_act

    return new_actions

