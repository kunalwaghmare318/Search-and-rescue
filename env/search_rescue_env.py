import functools
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

# ==============================================================================
# ENVIRONMENT CONFIGURATION & TUNABLE REWARD / CURRICULUM CONSTANTS (VARIANT 1 V3)
# ==============================================================================
FULL_GRID_SIZE = 10               # Target full grid size
NUM_AGENTS = 3                    # 3 search-and-rescue agents
FULL_NUM_SURVIVORS = 3            # Target number of survivors
FULL_NUM_OBSTACLES = 8            # Target number of obstacles
MAX_STEPS = 100                   # Max timesteps per episode

# Base Reward Weights
REWARD_NEW_CELL_COVERED = 1.0     # Reward for discovering an unvisited cell
REWARD_SURVIVOR_FOUND   = 50.0    # Full bonus for finding a survivor
PENALTY_PER_STEP        = -0.01   # Per-step time penalty
PENALTY_REVISIT_CELL    = -0.1    # Penalty for revisiting an already explored cell
PENALTY_COLLISION       = -2.0    # Exact penalty for agent-agent collision

# Shaped Reward & Penalty Enhancements
PROXIMITY_REPULSION_WEIGHT = -0.5 # Weight for soft proximity repulsion
PROXIMITY_RADIUS           = 2.0  # Cell distance threshold for repulsion
SHAPING_FRONTIER_WEIGHT    = 0.5  # Reward for moving closer to nearest unexplored cell
SHAPING_SURVIVOR_APPROACH_WEIGHT = 1.0 # Additive reward for reducing distance to nearest survivor
PENALTY_ANTI_STALL         = -0.2 # Penalty for staying/stalling on recent cells
STALL_THRESHOLD            = 2    # Max allowed visits to same cell in recent window (5 steps)


class SearchAndRescueEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment for 3-Agent Swarm Search-and-Rescue with
    curriculum learning and continuous reward shaping.
    """
    metadata = {"name": "search_and_rescue_v0", "render_modes": []}

    def __init__(self, grid_size=FULL_GRID_SIZE, num_agents=NUM_AGENTS, 
                 num_survivors=FULL_NUM_SURVIVORS, num_obstacles=FULL_NUM_OBSTACLES, 
                 max_steps=MAX_STEPS, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.n_agents_cfg = num_agents
        self.num_survivors = num_survivors
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.curriculum_progress = 1.0  # Default 1.0 (full difficulty)

        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents_cfg)]
        self.agents = self.possible_agents[:]

        # Movement actions: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        self._action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}

        # Observation dimension:
        # Fixed at full size (309) for model compatibility across curriculum stages
        # - One-hot agent identity: 3
        # - Self position: 2
        # - Other agents' positions: 4
        # - Visited grid: 100
        # - Obstacles grid: 100
        # - Survivors grid: 100
        self.obs_dim = self.n_agents_cfg + 2 + (self.n_agents_cfg - 1) * 2 + (3 * FULL_GRID_SIZE * FULL_GRID_SIZE)
        self._observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def set_curriculum_progress(self, progress: float):
        """Update curriculum progress (0.0 = early stage, 1.0 = full difficulty)."""
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

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

        # Calculate active curriculum params
        # Grid size scales from 6 to FULL_GRID_SIZE (10)
        curr_grid_sz = int(round(6 + (FULL_GRID_SIZE - 6) * self.curriculum_progress))
        curr_obstacles = int(round(2 + (FULL_NUM_OBSTACLES - 2) * self.curriculum_progress))

        self.active_grid_size = curr_grid_sz
        self.visited_grid = np.zeros((FULL_GRID_SIZE, FULL_GRID_SIZE), dtype=np.float32)
        self.obstacle_grid = np.zeros((FULL_GRID_SIZE, FULL_GRID_SIZE), dtype=np.float32)
        self.survivor_grid = np.zeros((FULL_GRID_SIZE, FULL_GRID_SIZE), dtype=np.float32)

        all_cells = [(r, c) for r in range(curr_grid_sz) for c in range(curr_grid_sz)]
        np.random.shuffle(all_cells)

        # Place obstacles
        self.obstacles = set(all_cells[:curr_obstacles])
        for r, c in self.obstacles:
            self.obstacle_grid[r, c] = 1.0

        remaining_cells = all_cells[curr_obstacles:]

        # Place agents in distinct free cells
        free_cells = remaining_cells[:]
        self.agent_positions = {}
        self.recent_history = {agent: [] for agent in self.possible_agents}

        for i, agent in enumerate(self.possible_agents):
            pos = free_cells[i]
            self.agent_positions[agent] = pos
            self.visited_grid[pos[0], pos[1]] = 1.0
            self.recent_history[agent].append(pos)

        survivor_candidate_cells = free_cells[self.n_agents_cfg:]

        # Curriculum survivor spawn distance: early in training, spawn closer to agents
        if self.curriculum_progress < 0.8:
            agent_center = np.mean([list(p) for p in self.agent_positions.values()], axis=0)
            max_allowed_dist = 3.0 + 7.0 * self.curriculum_progress
            valid_surv_cells = [
                cell for cell in survivor_candidate_cells
                if np.linalg.norm(np.array(cell) - agent_center) <= max_allowed_dist
            ]
            if len(valid_surv_cells) < FULL_NUM_SURVIVORS:
                valid_surv_cells = survivor_candidate_cells
        else:
            valid_surv_cells = survivor_candidate_cells

        np.random.shuffle(valid_surv_cells)
        self.survivors = set(valid_surv_cells[:FULL_NUM_SURVIVORS])
        for r, c in self.survivors:
            self.survivor_grid[r, c] = 1.0

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        self.step_count += 1
        prev_positions = {agent: self.agent_positions[agent] for agent in self.agents}

        # Proposed next positions
        next_positions = {}
        move_penalties = {agent: 0.0 for agent in self.agents}

        for agent in self.agents:
            act = actions.get(agent, 0)
            r, c = self.agent_positions[agent]

            if act == 1:    # Up
                nr, nc = r - 1, c
            elif act == 2:  # Down
                nr, nc = r + 1, c
            elif act == 3:  # Left
                nr, nc = r, c - 1
            elif act == 4:  # Right
                nr, nc = r, c + 1
            else:           # Stay
                nr, nc = r, c

            # Boundary check based on active curriculum grid size
            if nr < 0 or nr >= self.active_grid_size or nc < 0 or nc >= self.active_grid_size:
                next_positions[agent] = (r, c)
                move_penalties[agent] += -1.0
            elif (nr, nc) in self.obstacles:
                next_positions[agent] = (r, c)
                move_penalties[agent] += -1.0
            else:
                next_positions[agent] = (nr, nc)

        # Detect agent collisions
        collision_agents = set()
        pos_counts = {}
        for agent, pos in next_positions.items():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        for agent, pos in next_positions.items():
            if pos_counts[pos] > 1:
                collision_agents.add(agent)

        # Resolve positions & base rewards
        rewards = {}
        for agent in self.agents:
            if agent in collision_agents:
                new_pos = prev_positions[agent]
            else:
                new_pos = next_positions[agent]
                self.agent_positions[agent] = new_pos

            # Update recent history for anti-stall tracking
            self.recent_history[agent].append(new_pos)
            if len(self.recent_history[agent]) > 5:
                self.recent_history[agent].pop(0)

            r_total = PENALTY_PER_STEP + move_penalties[agent]

            # 1. Exact collision penalty
            if agent in collision_agents:
                r_total += PENALTY_COLLISION

            # 2. Soft Proximity Repulsion Penalty
            for other, other_pos in self.agent_positions.items():
                if other != agent:
                    dist = np.linalg.norm(np.array(new_pos) - np.array(other_pos))
                    if 0 < dist < PROXIMITY_RADIUS:
                        repulsion = PROXIMITY_REPULSION_WEIGHT * (1.0 - (dist / PROXIMITY_RADIUS))
                        r_total += repulsion

            # 3. Coverage reward vs revisit penalty
            if self.visited_grid[new_pos[0], new_pos[1]] == 0.0:
                self.visited_grid[new_pos[0], new_pos[1]] = 1.0
                r_total += REWARD_NEW_CELL_COVERED
            else:
                r_total += PENALTY_REVISIT_CELL

            # 4. Frontier-Shaping Reward (reducing distance to nearest unvisited cell)
            prev_frontier_dist = self._min_dist_to_unvisited(prev_positions[agent])
            curr_frontier_dist = self._min_dist_to_unvisited(new_pos)
            if prev_frontier_dist is not None and curr_frontier_dist is not None:
                r_total += SHAPING_FRONTIER_WEIGHT * (prev_frontier_dist - curr_frontier_dist)

            # 5. Distance-Based Survivor Approach Reward
            if self.survivors:
                prev_surv_dist = min([np.linalg.norm(np.array(prev_positions[agent]) - np.array(s)) for s in self.survivors])
                curr_surv_dist = min([np.linalg.norm(np.array(new_pos) - np.array(s)) for s in self.survivors])
                r_total += SHAPING_SURVIVOR_APPROACH_WEIGHT * (prev_surv_dist - curr_surv_dist)

            # 6. Full Survivor-Found Bonus
            if new_pos in self.survivors:
                self.survivors.remove(new_pos)
                self.survivor_grid[new_pos[0], new_pos[1]] = 0.0
                r_total += REWARD_SURVIVOR_FOUND

            # 7. Anti-Stall Penalty (repeating same cell > STALL_THRESHOLD times in last 5 steps)
            cell_counts = self.recent_history[agent].count(new_pos)
            if cell_counts > STALL_THRESHOLD:
                r_total += PENALTY_ANTI_STALL * (cell_counts - STALL_THRESHOLD)

            rewards[agent] = float(r_total)

        # Termination & Truncation
        env_done = (len(self.survivors) == 0) or (self.step_count >= self.max_steps)
        terminations = {agent: env_done for agent in self.agents}
        truncations = {agent: (self.step_count >= self.max_steps) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _min_dist_to_unvisited(self, pos):
        unvisited = np.argwhere(self.visited_grid[:self.active_grid_size, :self.active_grid_size] == 0.0)
        if len(unvisited) == 0:
            return None
        return min([np.linalg.norm(np.array(pos) - u) for u in unvisited])

    def _get_obs(self, agent):
        r, c = self.agent_positions[agent]

        # One-hot agent identity
        agent_idx = int(agent.split("_")[-1])
        agent_id_one_hot = np.zeros(self.n_agents_cfg, dtype=np.float32)
        agent_id_one_hot[agent_idx] = 1.0

        self_pos = np.array([r / FULL_GRID_SIZE, c / FULL_GRID_SIZE], dtype=np.float32)

        others_pos = []
        for other in self.possible_agents:
            if other != agent:
                or_, oc_ = self.agent_positions.get(other, (-1, -1))
                if or_ >= 0:
                    others_pos.extend([or_ / FULL_GRID_SIZE, oc_ / FULL_GRID_SIZE])
                else:
                    others_pos.extend([-1.0, -1.0])
        others_pos = np.array(others_pos, dtype=np.float32)

        visited_flat = self.visited_grid.flatten()
        obstacle_flat = self.obstacle_grid.flatten()
        survivor_flat = self.survivor_grid.flatten()

        obs = np.concatenate([
            agent_id_one_hot,
            self_pos,
            others_pos,
            visited_flat,
            obstacle_flat,
            survivor_flat
        ]).astype(np.float32)

        return obs


def parallel_env(**kwargs):
    return SearchAndRescueEnv(**kwargs)
