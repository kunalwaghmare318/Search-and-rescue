"""
Generate 3D Area Mapping V2 Replay Log from the 2,000,000-timestep trained PPO Model.
"""
import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from env.search_rescue_area_mapping_v2 import SearchAndRescueAreaMappingEnvV2, GRID_SIZE, NUM_AGENTS

MODEL_PATH = "models/ppo_area_mapping_v2_2000000_steps.zip" if os.path.exists("models/ppo_area_mapping_v2_2000000_steps.zip") else "models/ppo_area_mapping_v2_BEST.zip"
OUTPUT_PUBLIC = "public/area_mapping_v2_replay.json"
OUTPUT_FRONTEND = "frontend/area_mapping_v2_replay.json"

def generate_replay():
    print(f"Loading trained AreaMapping_v2 model from: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(MODEL_PATH, device=device)

    env = SearchAndRescueAreaMappingEnvV2(curriculum_stage=1.0)
    obs, _ = env.reset(seed=42)

    # 3D Environment Structures
    structures = []
    for pos, h in env.obstacle_heights.items():
        structures.append({
            "r": int(pos[0]),
            "c": int(pos[1]),
            "height": int(h),
            "type": "building" if pos in env.buildings else "obstacle"
        })

    steps_data = []

    # Step 0
    initial_positions = {a: list(env.agent_positions[a]) for a in env.possible_agents}
    initial_points = [list(p) for p in env.scanned_voxels]

    steps_data.append({
        "step": 0,
        "agent_positions": initial_positions,
        "new_point_cloud": initial_points,
        "scanned_pct": 0.0,
        "events": []
    })

    done = False
    step_idx = 0

    while not done and step_idx < 300:
        step_idx += 1
        actions = {}
        for agent in env.agents:
            act, _ = model.predict(obs[agent], deterministic=True)
            actions[agent] = int(act)

        prev_scanned = set(env.scanned_voxels)
        obs, rewards, terms, truncs, infos = env.step(actions)
        done = any(terms.values()) or any(truncs.values())

        new_voxels = env.scanned_voxels - prev_scanned
        new_points = [list(p) for p in new_voxels]

        scanned_walkable = len(env.scanned_voxels.intersection(env.total_walkable_voxels))
        cov_pct = round(min(100.0, (scanned_walkable / max(1, len(env.total_walkable_voxels))) * 100.0), 1)

        steps_data.append({
            "step": step_idx,
            "agent_positions": {a: list(env.agent_positions[a]) for a in env.possible_agents},
            "new_point_cloud": new_points,
            "scanned_pct": cov_pct,
            "events": []
        })

    total_scanned_voxels = [list(p) for p in env.scanned_voxels]

    replay_payload = {
        "metadata": {
            "version": "AreaMapping_v2_2M_PPO",
            "grid_size": GRID_SIZE,
            "num_agents": NUM_AGENTS,
            "total_steps": len(steps_data) - 1,
            "final_coverage_pct": steps_data[-1]["scanned_pct"],
            "structures": structures
        },
        "initial_state": {
            "agent_positions": initial_positions
        },
        "steps": steps_data,
        "final_point_cloud": total_scanned_voxels
    }

    # Save to public and frontend
    for path in [OUTPUT_PUBLIC, OUTPUT_FRONTEND]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(replay_payload, f)
        print(f"[+] Saved AreaMapping_v2 replay to {path} (Coverage: {steps_data[-1]['scanned_pct']}%)")

if __name__ == "__main__":
    generate_replay()