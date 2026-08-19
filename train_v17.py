import os
import torch
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env.search_rescue_env_v17 import parallel_env_v17, SearchAndRescueEnvV17
from eval_v17_checkpoints import batch_evaluate_v17_checkpoints, evaluate_v17_checkpoint

TOTAL_TIMESTEPS       = 1200000
CHECKPOINT_FREQ_STEPS = 100000
STATUS_UPDATE_FREQ    = 200000
LEARNING_RATE         = 3e-4
ENT_COEF              = 0.01
PPO_GAMMA             = 0.99


class V17StatusCallback(BaseCallback):
    """
    Callback that logs/prints a clear status update every 200,000 timesteps.
    Metrics tracked:
      - Current average episode reward
      - Current destruction rate (%)
      - Current search completeness (%)
      - Average drones lost per episode
    """
    def __init__(self, raw_env, status_freq=STATUS_UPDATE_FREQ, verbose=1):
        super().__init__(verbose)
        self.raw_env = raw_env
        self.status_freq = status_freq
        self.next_status_step = status_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_status_step:
            print("\n" + "=" * 80)
            print(f" [STATUS UPDATE] TIMESTEP {self.num_timesteps:,} / {TOTAL_TIMESTEPS:,}")
            print("-" * 80)

            # Quick evaluation rollout to get current metrics
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                temp_env = SearchAndRescueEnvV17()
                ep_rewards, ep_destructions, ep_drones_lost, ep_completeness = [], [], [], []

                for ep in range(10):
                    obs_dict, _ = temp_env.reset(seed=2000 + ep)
                    step_count = 0
                    total_r = 0.0

                    while step_count < temp_env.max_steps:
                        step_count += 1
                        actions = {}
                        for a in temp_env.possible_agents:
                            if a in temp_env.failed_agents:
                                actions[a] = 0
                            else:
                                o = obs_dict[a]
                                act, _ = self.model.predict(o, deterministic=True)
                                actions[a] = int(act)

                        obs_dict, rewards, terms, truncs, infos = temp_env.step(actions)
                        total_r += sum(rewards.values())
                        if all(terms.values()) or all(truncs.values()):
                            break

                    visited_count = int(np.sum(temp_env.visited_grid))
                    total_walkable = len(temp_env.total_walkable_cells)
                    comp = min(100.0, (visited_count / max(1.0, float(total_walkable))) * 100.0)

                    ep_rewards.append(total_r)
                    ep_destructions.append(1.0 if temp_env.drones_destroyed_count > 0 else 0.0)
                    ep_drones_lost.append(temp_env.drones_destroyed_count)
                    ep_completeness.append(comp)

                avg_reward = np.mean(ep_rewards)
                destruct_rate = np.mean(ep_destructions) * 100.0
                avg_drones_lost = np.mean(ep_drones_lost)
                avg_comp = np.mean(ep_completeness)

                print(f"  • Avg Episode Reward    : {avg_reward:.2f}")
                print(f"  • Destruction Rate      : {destruct_rate:.1f}%")
                print(f"  • Search Completeness   : {avg_comp:.1f}%")
                print(f"  • Drones Lost / Episode : {avg_drones_lost:.2f}")
            except Exception as e:
                print(f"  [!] Error calculating status metrics: {e}")

            print("=" * 80 + "\n")
            self.next_status_step += self.status_freq

        return True


def check_device():
    if torch.cuda.is_available():
        print(f"[+] GPU Active: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[i] CPU mode active.")
    return "cpu"


def train_v17_search():
    device = check_device()
    print("=" * 80)
    print(" [+] V17 TRAINING: EXHAUSTIVE SWARM SEARCH (1,200,000 TIMESTEPS)")
    print("     - Unknown Survivor Count Framing (Search Completeness)")
    print("     - One-Life Permanent Collision Destruction")
    print("     - Delayed Reassignment Priority Order")
    print("     - Checkpoints every 100,000 steps")
    print("     - Status updates every 200,000 steps")
    print("=" * 80)

    raw_env = parallel_env_v17()
    env = ss.black_death_v3(raw_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    save_freq = max(1, CHECKPOINT_FREQ_STEPS // vec_env.num_envs)
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path="./models/",
        name_prefix="v17_search_ckpt",
        verbose=1
    )
    status_cb = V17StatusCallback(raw_env=raw_env, status_freq=STATUS_UPDATE_FREQ)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=PPO_GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log="./logs/v17_search/",
        device=device
    )

    print("\n[+] Launching PPO model training for 1,200,000 timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[ckpt_cb, status_cb])

    # Save final model checkpoint
    final_path = "./models/v17_search_ckpt_1200000_steps.zip"
    model.save(final_path)
    print(f"\n[+] Training complete! Final checkpoint saved to: {final_path}")

    # Auto batch evaluate all saved checkpoints and select best
    print("\n[+] Starting post-training batch evaluation and best model auto-selection...")
    batch_evaluate_v17_checkpoints()


if __name__ == "__main__":
    train_v17_search()
