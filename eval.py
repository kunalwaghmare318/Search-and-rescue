import sys
import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO

from env.search_rescue_env import parallel_env

# ==============================================================================
# EVALUATION CONFIGURATION
# ==============================================================================
NUM_EVAL_EPISODES = 10     # Number of evaluation episodes to run


def evaluate_policy(checkpoint_path, num_episodes=NUM_EVAL_EPISODES):
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Checkpoint file not found at '{checkpoint_path}'")
        sys.exit(1)

    print(f"\n==================================================")
    print(f" [+] Evaluating Checkpoint: {checkpoint_path}")
    print(f" [+] Evaluation Episodes: {num_episodes}")
    print(f"==================================================")

    # 1. Load trained PPO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(checkpoint_path, device=device)
    model_obs_dim = model.observation_space.shape[0]

    # 2. Instantiate raw PettingZoo environment
    env = parallel_env()

    # Trackers across episodes
    coverages = []
    times_to_find_all = []
    collisions_per_ep = []
    rewards_per_ep = []

    for ep in range(num_episodes):
        obs, infos = env.reset(seed=100 + ep)
        ep_reward = 0.0
        ep_collisions = 0
        all_found_step = None

        while env.agents:
            # Format observation for model prediction based on expected obs dim
            actions = {}
            for agent in env.agents:
                agent_obs = obs[agent]
                if model_obs_dim == 306 and len(agent_obs) == 309:
                    # Strip one-hot agent ID for older 306-dim models
                    agent_obs = agent_obs[3:]
                elif model_obs_dim == 309 and len(agent_obs) == 306:
                    # Prepend one-hot agent ID for 309-dim models
                    agent_idx = int(agent.split("_")[-1])
                    one_hot = np.zeros(3, dtype=np.float32)
                    one_hot[agent_idx] = 1.0
                    agent_obs = np.concatenate([one_hot, agent_obs])

                act, _ = model.predict(agent_obs, deterministic=True)
                actions[agent] = int(act)

            # Count agent collisions for metric reporting
            curr_positions = env.agent_positions
            next_positions = {}
            for agent in env.agents:
                act = actions[agent]
                r, c = curr_positions[agent]
                if act == 1 and r > 0 and (r - 1, c) not in env.obstacles:
                    nr, nc = r - 1, c
                elif act == 2 and r < env.grid_size - 1 and (r + 1, c) not in env.obstacles:
                    nr, nc = r + 1, c
                elif act == 3 and c > 0 and (r, c - 1) not in env.obstacles:
                    nr, nc = r, c - 1
                elif act == 4 and c < env.grid_size - 1 and (r, c + 1) not in env.obstacles:
                    nr, nc = r, c + 1
                else:
                    nr, nc = r, c
                next_positions[agent] = (nr, nc)

            pos_counts = {}
            for p in next_positions.values():
                pos_counts[p] = pos_counts.get(p, 0) + 1

            for agent in env.agents:
                if pos_counts[next_positions[agent]] > 1:
                    ep_collisions += 1

            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            ep_reward += sum(rewards.values())

            # Check if all survivors located
            if len(env.survivors) == 0 and all_found_step is None:
                all_found_step = env.step_count

        # Post-episode metrics
        cov_pct = (env.visited_grid.sum() / (env.grid_size ** 2)) * 100.0
        coverages.append(cov_pct)
        collisions_per_ep.append(ep_collisions)
        rewards_per_ep.append(ep_reward)

        if all_found_step is not None:
            times_to_find_all.append(all_found_step)

    # Calculate summary metrics
    avg_coverage = np.mean(coverages)
    avg_collisions = np.mean(collisions_per_ep)
    avg_reward = np.mean(rewards_per_ep)

    if times_to_find_all:
        avg_time_to_find = f"{np.mean(times_to_find_all):.1f} steps (Success rate: {len(times_to_find_all)}/{num_episodes})"
    else:
        avg_time_to_find = f"Not found within max steps (Success rate: 0/{num_episodes})"

    print("\n--------------------------------------------------")
    print(" EVALUATION RESULTS METRICS SUMMARY")
    print("--------------------------------------------------")
    print(f" 1. Average Grid Coverage          : {avg_coverage:.2f}%")
    print(f" 2. Average Time to Find Survivors : {avg_time_to_find}")
    print(f" 3. Average Collisions per Episode : {avg_collisions:.2f}")
    print(f" 4. Average Episode Reward        : {avg_reward:.2f}")
    print("--------------------------------------------------\n")

    return {
        "coverage": avg_coverage,
        "success_rate": f"{len(times_to_find_all)}/{num_episodes}",
        "collisions": avg_collisions,
        "reward": avg_reward,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MARL PPO checkpoint.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="models/ppo_variant1_v3_final.zip",
        help="Path to saved model checkpoint (.zip file)",
    )
    args = parser.parse_args()

    evaluate_policy(args.checkpoint)
