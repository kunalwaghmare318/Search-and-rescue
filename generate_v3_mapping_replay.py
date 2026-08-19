"""
Generate 3D Area Mapping V3 Replay Log from trained PPO model.
"""
import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from env.search_rescue_area_mapping_v3 import AreaMappingEnvV3, GRID_SIZE, NUM_AGENTS, COMPLETION_THRESHOLD_PCT

MODEL_PATH = "models/ppo_area_mapping_v3_BEST.zip" if os.path.exists("models/ppo_area_mapping_v3_BEST.zip") else "models/ppo_area_mapping_v3_final.zip"
OUTPUT_PUBLIC = "public/area_mapping_v3_replay.json"
OUTPUT_FRONTEND = "frontend/area_mapping_v3_replay.json"


def generate():
    print(f"Loading v3 model: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(MODEL_PATH, device=device)

    env = AreaMappingEnvV3(curriculum_stage=1.0)
    obs, _ = env.reset(seed=42)

    structures = []
    for pos, h in env.obstacle_heights.items():
        structures.append({
            "r": int(pos[0]), "c": int(pos[1]), "height": int(h),
            "type": "building" if pos in env.buildings else "obstacle"
        })

    steps_data = []
    initial_pts = [list(p) for p in env.scanned_voxels]
    steps_data.append({
        "step": 0,
        "agent_positions": {a: list(env.agent_positions[a]) for a in env.possible_agents},
        "new_point_cloud": initial_pts,
        "scanned_pct": 0.0, "events": []
    })

    done = False
    step_idx = 0
    while not done and step_idx < 500:
        step_idx += 1
        actions = {}
        for a in env.agents:
            act, _ = model.predict(obs[a], deterministic=True)
            actions[a] = int(act)

        prev = set(env.scanned_voxels)
        obs, _, terms, truncs, infos = env.step(actions)
        done = any(terms.values()) or any(truncs.values())

        new_pts = [list(p) for p in env.scanned_voxels - prev]
        info = infos[env.possible_agents[0]]
        cov = info.get("coverage_pct", 0)

        steps_data.append({
            "step": step_idx,
            "agent_positions": {a: list(env.agent_positions[a]) for a in env.possible_agents},
            "new_point_cloud": new_pts,
            "scanned_pct": cov, "events": []
        })

    final_info = infos[env.possible_agents[0]]
    replay = {
        "metadata": {
            "version": "AreaMapping_v3_2M_PPO",
            "grid_size": GRID_SIZE, "num_agents": NUM_AGENTS,
            "total_steps": len(steps_data) - 1,
            "final_coverage_pct": final_info.get("coverage_pct", 0),
            "completion_threshold": COMPLETION_THRESHOLD_PCT,
            "collisions": final_info.get("collisions", 0),
            "zone_completion_rate": final_info.get("zone_completion_rate", 0),
            "structures": structures
        },
        "initial_state": {"agent_positions": steps_data[0]["agent_positions"]},
        "steps": steps_data,
        "final_point_cloud": [list(p) for p in env.scanned_voxels]
    }

    for path in [OUTPUT_PUBLIC, OUTPUT_FRONTEND]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(replay, f)
        print(f"[+] Saved {path} (Coverage: {final_info.get('coverage_pct', 0)}%)")


if __name__ == "__main__":
    generate()
