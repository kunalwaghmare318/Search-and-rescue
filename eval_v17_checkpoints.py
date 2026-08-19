import os
import glob
import shutil
import torch
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO

from env.search_rescue_env_v17 import SearchAndRescueEnvV17, apply_active_unsearched_navigation


def evaluate_v17_checkpoint(model_path, num_episodes=20):
    """Evaluates a V17 checkpoint on deterministic test episodes."""
    raw_env = SearchAndRescueEnvV17()
    env = ss.black_death_v3(raw_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    device = "cpu"
    model = PPO.load(model_path, device=device)

    ep_rewards = []
    ep_destructions = []
    ep_drones_lost = []
    ep_completeness = []
    ep_survivor_rates = []

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False
        step_count = 0

        # Reset raw env for tracking internal metrics directly
        raw_env.reset(seed=1000 + ep)
        obs_dict = {a: raw_env._get_obs(a) for a in raw_env.possible_agents}

        total_reward = 0.0

        while step_count < raw_env.max_steps:
            step_count += 1
            actions = {}
            for a in raw_env.possible_agents:
                if a in raw_env.failed_agents:
                    actions[a] = 0
                else:
                    o = obs_dict[a]
                    action, _ = model.predict(o, deterministic=True)
                    actions[a] = int(action)

            actions = apply_active_unsearched_navigation(raw_env, actions)
            obs_dict, rewards, terminations, truncations, infos = raw_env.step(actions)
            total_reward += sum(rewards.values())

            if all(terminations.values()) or all(truncations.values()):
                break

        total_walkable = len(raw_env.total_walkable_cells)
        visited_count = int(np.sum(raw_env.visited_grid))
        completeness_pct = min(100.0, (visited_count / max(1.0, float(total_walkable))) * 100.0)

        found_count = len(raw_env.found_survivors)
        total_surv = raw_env.initial_survivors_count
        surv_rate_pct = (found_count / max(1.0, float(total_surv))) * 100.0

        drones_lost = raw_env.drones_destroyed_count
        has_destruction = 1.0 if drones_lost > 0 else 0.0

        ep_rewards.append(total_reward)
        ep_destructions.append(has_destruction)
        ep_drones_lost.append(drones_lost)
        ep_completeness.append(completeness_pct)
        ep_survivor_rates.append(surv_rate_pct)

    vec_env.close()

    return {
        "model_path": model_path,
        "mean_reward": float(np.mean(ep_rewards)),
        "destruction_rate": float(np.mean(ep_destructions)) * 100.0,
        "drones_lost_per_ep": float(np.mean(ep_drones_lost)),
        "search_completeness": float(np.mean(ep_completeness)),
        "survivor_detection_rate": float(np.mean(ep_survivor_rates))
    }


def batch_evaluate_v17_checkpoints(ckpt_dir="./models", pattern="v17_search_ckpt_*.zip"):
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    if not ckpt_files:
        print(f"[!] No checkpoints found matching {os.path.join(ckpt_dir, pattern)}")
        return None

    print("\n" + "=" * 90)
    print(" [+] BATCH EVALUATING V17 SEARCH CHECKPOINTS")
    print("=" * 90)
    print(f"{'Checkpoint':<35} | {'Reward':<9} | {'Destruct Rate %':<15} | {'Drones Lost/Ep':<15} | {'Completeness %':<14} | {'Surv Detect %':<13}")
    print("-" * 90)

    results = []
    for ckpt in ckpt_files:
        res = evaluate_v17_checkpoint(ckpt, num_episodes=15)
        results.append(res)
        bname = os.path.basename(ckpt)
        print(f"{bname:<35} | {res['mean_reward']:<9.1f} | {res['destruction_rate']:<15.1f} | {res['drones_lost_per_ep']:<15.2f} | {res['search_completeness']:<14.1f} | {res['survivor_detection_rate']:<13.1f}")

    # Best selection priority:
    # 1. Lowest destruction rate
    # 2. Highest search completeness
    # 3. Highest survivor detection rate
    results.sort(key=lambda x: (x['destruction_rate'], -x['search_completeness'], -x['survivor_detection_rate']))
    best = results[0]

    target_best_path = os.path.join(ckpt_dir, "ppo_FINAL_BEST_v17.zip")
    shutil.copy(best['model_path'], target_best_path)

    print("-" * 90)
    print(f"[*] BEST CHECKPOINT SELECTED: {os.path.basename(best['model_path'])}")
    print(f"    - Destruction Rate: {best['destruction_rate']:.1f}%")
    print(f"    - Drones Lost / Ep: {best['drones_lost_per_ep']:.2f}")
    print(f"    - Search Completeness: {best['search_completeness']:.1f}%")
    print(f"    - Survivor Detection Rate: {best['survivor_detection_rate']:.1f}%")
    print(f"    - Saved to: {target_best_path}")
    print("=" * 90 + "\n")

    return best


if __name__ == "__main__":
    batch_evaluate_v17_checkpoints()
