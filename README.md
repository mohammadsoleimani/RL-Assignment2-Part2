# RL-Assignment2-Part2
Part 2 of Assignment 2 implements Monte Carlo methods (exploring starts, ε-soft, off-policy) on a 5x5 gridworld with special states (B, G, R, Y) and terminals. Using γ=0.95, it learns an optimal policy navigating to green (+1). Outputs include value function heatmaps and policy plots. Code runs in ~15–20s, seed=42.
# RL-Assignment2-Part2-GridWorld

This repository contains the solution for Part 2 of Assignment 2 in the Reinforcement Learning course. It implements Monte Carlo methods (exploring starts, ε-soft, off-policy with importance sampling) on a 5x5 gridworld with special states (blue (0,1), green (0,4, +1, terminal), red (4,2), yellow (4,4)) and terminals ((2,0), (2,4), (4,0)). The environment has deterministic dynamics (normal moves: -0.2, off-grid: -0.5, blue off-grid: -0.2) and a discount factor of γ = 0.95.

## Features
- **Environment**: 5x5 gridworld with deterministic transitions and terminal states.
- **Methods**:
  - Question 1.1: Monte Carlo with exploring starts (random initial states).
  - Question 1.2: Monte Carlo ε-soft (ε = 0.1, starts at blue).
  - Question 2: Off-policy Monte Carlo with equiprobable behavior policy and importance sampling.
- **Outputs**: 7 plots (3 value function heatmaps, 3 policy plots, 1 comparison plot) saved as PNGs, plus console output with value functions, policy consistency check, and analysis.
- **Reproducibility**: Random seed set to 42; 10,000 episodes (exploring starts, ε-soft), 20,000 (off-policy).

## Requirements
- Python 3
- `numpy`, `matplotlib`, `seaborn` (`pip install numpy matplotlib seaborn`)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RL-Assignment2-Part2-GridWorld.git
