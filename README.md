# Multi-Agent Search-and-Rescue (MARL) Pipeline

3-agent swarm search-and-rescue grid-world simulation using PettingZoo, SuperSuit, and Stable-Baselines3 (PPO).

## Features
- **Environment**: Custom PettingZoo `ParallelEnv` representing a 2D grid world with obstacles and survivors.
- **Reward Shaping**: Explicitly defined reward constants for coverage, survivor discovery, time penalty, and collision avoidance.
- **MARL Wrapper**: SuperSuit vectorization enabling standard SB3 algorithms (PPO) to train single-policy multi-agent control.
- **Hardware Acceleration**: Automatic GPU/CUDA detection with CPU fallback.

## Quick Start

1. **Activate Virtual Environment** (Windows PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run Smoke Test Training**:
   ```powershell
   python train.py
   ```

3. **Checkpoints**:
   Trained policy models are saved in the `models/` directory.
