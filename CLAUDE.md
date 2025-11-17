# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a hands-on learning repository for **reinforcement learning from scratch**. It contains custom RL algorithm implementations (REINFORCE, Actor-Critic, PPO), custom environments (drone landing game), and training notebooks documenting real challenges and solutions. The focus is on **understanding through implementation**, not production-ready code.

## Project Structure

```
reinforcement-learning-101/
├── Policy_Gradients_Baseline.ipynb    # REINFORCE with baseline (complete)
├── Test_Policy_Gradients.ipynb        # Testing/experimentation
├── Actor_Critic_Basic.ipynb           # Basic Actor-Critic (complete)
├── Actor_Critic_GAE.ipynb             # Generalized Advantage Estimation (planned)
├── Actor_Critic_PPO.ipynb             # Proximal Policy Optimization (planned)
├── rl_helpers/                        # Shared RL utilities
│   └── scalers.py                     # Reward shaping functions
├── models/                            # Saved model weights
│   ├── policy-gradients/              # REINFORCE models
│   └── actor-critic-basic/            # Actor-Critic models
└── delivery_drone/                    # Custom drone landing environment
    ├── game/                          # Core game engine
    ├── examples/                      # Example agents
    ├── socket_server.py              # Parallel training server
    └── manual_play.py                # Human play mode
```

## Running Code

### Training RL Agents

Training uses a **two-process architecture**: socket server for game instances + training script/notebook.

```bash
# Terminal 1: Start game server with parallel instances
cd delivery_drone
python socket_server.py --num-games 6 --render none --randomize-drone

# Terminal 2: Run training notebook
jupyter notebook
# Open Policy_Gradients_Baseline.ipynb or Actor_Critic_Basic.ipynb
```

### Server Options

```bash
# Visual mode (single game, human-viewable)
python socket_server.py --render human

# Headless mode (fast training)
python socket_server.py --render none

# Parallel training (6 games)
python socket_server.py --num-games 6 --render none

# Spawn randomization (robust policies)
python socket_server.py --randomize-drone --randomize-platform

# Fixed spawns (consistent training)
python socket_server.py --fixed-spawn
```

### Testing the Environment

```bash
cd delivery_drone

# Play manually to understand the task
python manual_play.py

# Run example agents
PYTHONPATH=. python examples/random_agent.py
PYTHONPATH=. python examples/simple_agent.py

# Test game and socket
python test_game.py
python test_socket.py
```

### Dependencies

```bash
# Core dependencies
pip install torch numpy pygame jupyter tqdm matplotlib

# Or use environment-specific requirements
cd delivery_drone
pip install -r requirements.txt
```

## Architecture Patterns

### 1. Policy Network Architecture

Both REINFORCE and Actor-Critic use a consistent 3-layer MLP with LayerNorm:

```python
DroneGamerBoi (Actor/Policy):
  Linear(15, 128) → LayerNorm → ReLU
  Linear(128, 128) → LayerNorm → ReLU
  Linear(128, 64) → LayerNorm → ReLU
  Linear(64, 3) → Sigmoid  # Binary action probabilities [main, left, right thrust]
```

Actor-Critic adds a separate value network:

```python
DroneTeacherBoi (Critic/Value):
  Linear(15, 128) → LayerNorm → ReLU
  Linear(128, 128) → LayerNorm → ReLU
  Linear(128, 64) → LayerNorm → ReLU
  Linear(64, 1)  # State value V(s)
```

### 2. Modular Reward Engineering

The codebase separates reward calculation from the environment:
- Environment provides basic terminal rewards (+100 landing, -100 crash)
- **Custom `calc_reward()` functions** in notebooks override default behavior
- All reward components tracked separately for debugging

Example reward components (Policy Gradients):
```python
{
    'time_penalty': -0.3 to -1.0,
    'distance': velocity_alignment-based movement reward,
    'velocity_alignment': bonus for moving toward platform,
    'angle': penalty for excess tilt,
    'speed': penalty for excessive speed,
    'vertical_position': penalty for being below platform,
    'terminal': +500 landing, -200/-300 crashes,
    'total': sum of all components
}
```

Use scalers from `rl_helpers/scalers.py`:
- `gaussian_scaler()`: Gaussian bell curve rewards
- `exponential_decay()`: Exponential decay based on distance
- `inverse_quadratic()`: 1/(1 + decay*value²) - for time penalties
- `scaled_shifted_negative_sigmoid()`: Shifted sigmoid for distance rewards

### 3. Parallel Environment Training

Both implementations use parallel game instances for sample efficiency:
- Socket server runs N game instances (typically 6)
- Batch inference across all games simultaneously (GPU-efficient)
- Single forward pass for all states in parallel

```python
# Example: Collect from 6 parallel games
for game_id in range(client.num_games):
    state = client.reset(game_id)
    # ... collect episode
```

### 4. State-to-Tensor Conversion

Environment handles all normalization. States are ready for neural networks:

```python
def state_to_array(state, device='cpu'):
    """Converts DroneState dataclass → torch.Tensor"""
    # All 15 dimensions pre-normalized by environment
    return torch.tensor([
        state.drone_x, state.drone_y,
        state.drone_vx, state.drone_vy,
        # ... (15 total)
    ], dtype=torch.float32, device=device)
```

## Environment Details

### State Space (15 dimensions, all normalized)

```python
{
    'drone_x': [0, 1],               # Position
    'drone_y': [0, 1],
    'drone_vx': [-2.5, 2.5],         # Velocity
    'drone_vy': [-2.5, 3.0],
    'drone_angle': [-1, 1],          # Normalized angle
    'drone_angular_vel': [-1.5, 1.5],
    'drone_fuel': [0, 1],
    'platform_x': [0, 1],            # Target position
    'platform_y': [0, 1],
    'distance_to_platform': [0, 1.41],
    'dx_to_platform': [-1.125, 1.125],  # Relative position
    'dy_to_platform': [-1.083, 1.083],
    'speed': [0, 3.9],
    'landed': bool,
    'crashed': bool
}
```

### Action Space (3 binary actions)

```python
{
    'main_thrust': 0 or 1,   # Upward thrust
    'left_thrust': 0 or 1,   # Counter-clockwise rotation
    'right_thrust': 0 or 1   # Clockwise rotation
}
```

### Landing Criteria

- Speed ≤ 3.0 pixels/frame (0.3 normalized)
- Angle ≤ 20° from upright (0.111 normalized)
- Bottom center of drone must be on platform (100px wide)
- Must have fuel remaining

## Algorithm Implementations

### REINFORCE with Baseline (Complete)

**File**: [Policy_Gradients_Baseline.ipynb](Policy_Gradients_Baseline.ipynb)

**Key Functions**:
- `collect_episodes()`: Parallel episode collection from multiple games
- `compute_returns()`: Monte Carlo return computation using Bellman equation
- `calc_reward()`: Custom reward shaping (7 components)
- `evaluate_policy_simple()`: Evaluation with live plotting

**Algorithm Flow**:
```
1. Collect full episodes: {states, actions, rewards}
2. Compute returns: G_t = Σ γ^k * r_{t+k}
3. Baseline: b = mean(returns)
4. Advantages: A_t = (G_t - b) / std(G_t - b)
5. Loss: -log π(a|s) * A_t
```

**Real Challenges Documented**:
- **Hovering exploit**: Policy learned to hover near platform → Time penalties
- **Spinning exploit**: Rotation for rewards → Angular velocity penalties
- **Horizontal misalignment**: Missing platform → Explicit alignment rewards
- **Training instability**: Gradient clipping (max_norm=0.5) + normalization

### Actor-Critic Basic (Complete)

**File**: [Actor_Critic_Basic.ipynb](Actor_Critic_Basic.ipynb)

**Key Functions**:
- `run_single_step()`: Single-step execution across parallel games
- `compute_td_error()`: Temporal Difference error calculation
- `calc_reward()`: Same reward shaping as Policy Gradients

**Algorithm Flow**:
```
1. Single step from each parallel game
2. TD target: y = r + γ*V(s')
3. TD error: δ = y - V(s)
4. Critic loss: δ²
5. Actor loss: -log π(a|s) * δ (detached)
```

**Key Differences from REINFORCE**:
- Online updates (per-step) vs episodic
- Lower variance, slight bias
- Faster learning, more stable
- Separate optimizers for actor/critic

## Common Hyperparameters

```python
learning_rate = 1e-3          # AdamW optimizer
bellman_gamma = 0.90-0.99     # Discount factor
gradient_clip = 0.5           # max_norm for grad clipping
num_parallel_games = 6        # Parallel environments
max_steps = 300-500           # Episode length (curriculum)
eval_interval = 10            # Evaluate every N iterations
temperature = 0.3             # Stochastic sampling (0=greedy)
```

## Model Saving/Loading

Models are saved in `models/{algorithm-name}/`:

```python
# Save both state_dict and full model
torch.save(agent.state_dict(), f'models/policy-gradients/agent_v1.2.pth')
torch.save(agent, f'models/policy-gradients/agent_v1.2_full.pt')

# Load
agent.load_state_dict(torch.load('models/policy-gradients/agent_v1.2.pth'))
```

## Debugging Reward Shaping

**Most debugging happens in reward functions**. Every component is tracked separately:

```python
reward_dict = calc_reward(state)
# Returns: {
#     'time_penalty': -0.5,
#     'distance': +2.3,
#     'velocity_alignment': +1.2,
#     'angle': -0.1,
#     'speed': -0.3,
#     'vertical_position': -0.5,
#     'terminal': 0.0,
#     'total': 2.1
# }
```

**Common exploit patterns to watch for**:
- **Hovering**: Drone stays near platform without landing → Add hovering penalties
- **Spinning**: Rotating in place for rewards → Add angular velocity penalties
- **Oscillating**: Moving back/forth for distance rewards → Use velocity-aligned rewards
- **Fuel waste**: Unnecessary thrust → Fuel penalties or remove fuel rewards

## Evaluation Strategy

Use `temperature` parameter for stochastic vs deterministic evaluation:

```python
# Greedy (deterministic)
evaluate_policy_simple(agent, client, temperature=0.0)

# Slightly stochastic (for training eval)
evaluate_policy_simple(agent, client, temperature=0.3)

# Fully stochastic (exploration)
evaluate_policy_simple(agent, client, temperature=1.0)
```

## GPU Usage

Code auto-detects CUDA and moves tensors to GPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = DroneGamerBoi().to(device)

# State conversion uses device parameter
state_tensor = state_to_array(state, device=device)
```

## Key Learning Insights

1. **Reward engineering is 90% of the work** - Most time is spent debugging reward functions, not the algorithm
2. **Velocity-based rewards > proximity rewards** - Moving toward target is more informative than distance alone
3. **Gradient clipping is essential** - Large updates cause policy collapse
4. **Advantage normalization reduces variance** - Normalizing advantages stabilizes training
5. **Parallel environments improve sample efficiency** - 6 games = 6x more data per iteration
6. **Curriculum learning helps** - Gradually increase episode length (75→250 steps)

## Documentation

- [README.md](README.md) - Project overview and learning path
- [delivery_drone/README.md](delivery_drone/README.md) - Environment API documentation
- [delivery_drone/SOCKET_API.md](delivery_drone/SOCKET_API.md) - Socket protocol details
- [delivery_drone/docs/RL_METHODS_SUMMARY.md](delivery_drone/docs/RL_METHODS_SUMMARY.md) - Theory notes

## Important Notes

- This is a **learning repository** - code is documented but not optimized for production
- Training is **stochastic** - results vary between runs
- The environment is **deterministic** given fixed random seed
- Socket server must be running before training notebooks
- All state normalization is handled by the environment
- Model versioning uses simple incrementing (v1.2, v1.3, etc.)
