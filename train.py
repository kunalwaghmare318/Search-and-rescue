import os
import torch
import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env.search_rescue_env import (
    parallel_env,
    REWARD_NEW_CELL_COVERED,
    REWARD_SURVIVOR_FOUND,
    PENALTY_PER_STEP,
    PENALTY_REVISIT_CELL,
    PENALTY_COLLISION,
    PROXIMITY_REPULSION_WEIGHT,
    SHAPING_FRONTIER_WEIGHT,
    SHAPING_SURVIVOR_APPROACH_WEIGHT,
    PENALTY_ANTI_STALL,
)


class CurriculumCallback(BaseCallback):
    """
    Updates the environment's curriculum stage based on training progress.
    """

    def __init__(self, total_timesteps: int, check_freq: int = 25000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)

        # Access underlying raw environments in vectorized SuperSuit env
        try:
            vec_env = self.training_env
            for env in vec_env.envs:
                curr_env = env
                while hasattr(curr_env, "env") or hasattr(curr_env, "par_env"):
                    if hasattr(curr_env, "par_env"):
                        curr_env = curr_env.par_env
                    elif hasattr(curr_env, "env"):
                        curr_env = curr_env.env
                if hasattr(curr_env, "set_curriculum_progress"):
                    curr_env.set_curriculum_progress(progress)
        except Exception:
            pass

        if self.num_timesteps - self.last_logged_step >= self.check_freq:
            self.last_logged_step = self.num_timesteps
            grid_sz = int(round(6 + (10 - 6) * progress))
            print(
                f"[Curriculum Progress] Step {self.num_timesteps}/{self.total_timesteps} "
                f"({progress*100:.1f}%) -> Active Grid Size: {grid_sz}x{grid_sz}"
            )

        return True


def check_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[+] GPU Detected & Active: {device_name} (CUDA {torch.version.cuda})")
        return "cuda"
    else:
        print("[i] CUDA/GPU not detected. Running training on CPU.")
        return "cpu"


def sanity_check_env(raw_env):
    print("\n--- Running Random-Action Rollout Sanity Check ---")
    obs, infos = raw_env.reset(seed=42)
    step_count = 0
    total_reward = {agent: 0.0 for agent in raw_env.agents}

    for _ in range(10):
        actions = {agent: raw_env.action_space(agent).sample() for agent in raw_env.agents}
        obs, rewards, terminations, truncations, infos = raw_env.step(actions)
        step_count += 1
        for agent, r in rewards.items():
            total_reward[agent] += r
        if not raw_env.agents:
            break

    print(f"[+] Sanity check completed successfully ({step_count} steps executed). Total rewards: {total_reward}\n")


def run_variant1_v3_training(total_timesteps=500000):
    device = check_device()

    print("==================================================")
    print(" [+] REWARD & CURRICULUM CONFIGURATION: Variant 1 v3 (500k Run)")
    print(f"     - REWARD_NEW_CELL_COVERED       = {REWARD_NEW_CELL_COVERED}")
    print(f"     - REWARD_SURVIVOR_FOUND        = {REWARD_SURVIVOR_FOUND}")
    print(f"     - PENALTY_PER_STEP             = {PENALTY_PER_STEP}")
    print(f"     - PENALTY_REVISIT_CELL         = {PENALTY_REVISIT_CELL}")
    print(f"     - PENALTY_COLLISION            = {PENALTY_COLLISION}")
    print(f"     - PROXIMITY_REPULSION_WEIGHT   = {PROXIMITY_REPULSION_WEIGHT}")
    print(f"     - SHAPING_FRONTIER_WEIGHT      = {SHAPING_FRONTIER_WEIGHT}")
    print(f"     - SHAPING_SURVIVOR_APPROACH    = {SHAPING_SURVIVOR_APPROACH_WEIGHT}")
    print(f"     - PENALTY_ANTI_STALL           = {PENALTY_ANTI_STALL}")
    print(f"     - PPO GAMMA                    = 0.995")
    print(f"     - PPO ENTROPY COEF (ent_coef)  = 0.03")
    print(f"     - PPO NETWORK ARCHITECTURE     = [256, 256]")
    print(f" [+] Hardware acceleration: {device.upper()}")
    print("==================================================")

    # 1. Instantiate raw environment
    raw_env = parallel_env()

    # 2. Perform sanity check rollout
    sanity_check_env(raw_env)

    # 3. SuperSuit wrapping for SB3 compatibility
    env = parallel_env()
    env = ss.black_death_v3(env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

    # 4. Create Callbacks (Checkpoint every 50,000 steps + Curriculum callback)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    save_freq_steps = max(1, 50000 // vec_env.num_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path="./models/",
        name_prefix="ppo_variant1_v3",
        verbose=1,
    )
    curriculum_callback = CurriculumCallback(total_timesteps=total_timesteps, check_freq=25000)

    callbacks = [checkpoint_callback, curriculum_callback]

    print(f"\n--- Starting PPO Variant 1 v3 Training ({total_timesteps} timesteps) on device: {device} ---")

    # 5. Initialize PPO Model
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.995,
        ent_coef=0.03,
        verbose=1,
        device=device,
        tensorboard_log="./logs/",
    )

    # 6. Train model for 500,000 timesteps
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # 7. Save final policy checkpoint
    final_checkpoint_path = os.path.join("models", "ppo_variant1_v3_final.zip")
    model.save(final_checkpoint_path)
    print(f"\n[+] Training complete! Final checkpoint saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    run_variant1_v3_training(total_timesteps=500000)
