# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a hands-on learning repository for **reinforcement learning from scratch**. It contains custom RL algorithm implementations (REINFORCE, Actor-Critic with TD Error), custom environments (drone landing, visual driving), and training notebooks documenting real challenges and solutions. The focus is on **understanding through implementation**, not production-ready code.

## Project Structure

```
reinforcement-learning-101/
├── Policy_Gradients_Baseline.ipynb    # REINFORCE with baseline (COMPLETE)
├── Test_Policy_Gradients.ipynb        # Testing/experimentation (2 MB, extensive)
├── Actor_Critic_Basic.ipynb           # Basic Actor-Critic with TD Error (COMPLETE)
├── Actor_Critic_GAE.ipynb             # GAE implementation (22 cells, work in progress)
├── Actor_Critic_PPO.ipynb             # PPO implementation (28 cells, functional)
├── ACTOR_CRITIC_LESSONS_LEARNED.md    # Critical bugs & fixes (26 KB, IMPORTANT)
├── README.md                          # Project overview and learning path
├── rl_helpers/                        # Shared RL utilities
│   └── scalers.py                     # Reward shaping functions
├── models/                            # Saved model weights
│   ├── policy-gradients/              # REINFORCE models (v1.2-v1.4)
│   └── actor-critic-basic/            # Actor-Critic models (v1-v3)
├── blogs/                             # Technical blog posts
│   ├── RL-1/                          # Policy Gradient blog (40 KB)
│   └── RL-Actor-critic/               # Actor-Critic blog (35 KB)
├── delivery_drone/                    # Custom drone landing environment
│   ├── game/                          # Core game engine
│   ├── examples/                      # Example agents
│   ├── socket_server.py              # Parallel training server
│   └── manual_play.py                # Human play mode
└── visual_driving_2d/                 # CarRacing RL environment (Gymnasium-based)
    ├── game/                          # Game engine (Box2D physics)
    ├── socket_server_gymnasium.py    # Gymnasium CarRacing socket server
    ├── GYMNASIUM_SETUP.md            # WSL2 pygame rendering setup
    └── manual_play_pygame.py         # Manual play with pygame
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

**Reward Components** (Actor-Critic uses improved velocity-magnitude-weighted system):
```python
{
    'time_penalty': -0.5,                    # Constant per step
    'distance': +4.5 or 0.0,                 # Velocity-magnitude-weighted approach reward
    'hovering': -1.0 or 0.0,                 # Penalty for speed < 0.05
    'angle': -0.1,                           # Distance-dependent angle tolerance
    'speed': -2.0 or 0.0,                    # Penalty for excessive speed when close
    'vertical_position': -0.5 or 0.0,        # Penalty for being below platform
    'terminal': +800, -200, or -300,         # Landing (+ fuel bonus), crash far, crash close
    'total': sum of all components
}
```

**Key Improvement in Actor-Critic**: Velocity-magnitude-weighted rewards require `prev_state` parameter to calculate `distance_delta`, preventing hovering/zooming exploits. See "Debugging Reward Shaping" section for details.

**Reward Shaping Utilities** from `rl_helpers/scalers.py`:
- `gausian_scaler(value, sigma, scaler)`: Gaussian bell curve rewards
- `exponential_decay(value, decay, scaler)`: Exponential decay based on distance
- `inverse_quadratic(value, decay, scaler)`: 1/(1 + decay*value²) - for time penalties
- `inverse_linear(value, decay, scaler)`: 1/(1 + decay*|value|)
- `scaled_shifted_negative_sigmoid(value, sigma, scaler)`: Shifted sigmoid for distance rewards
- `linear_scaler(value, scaler)`: Simple linear scaling

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

### Actor-Critic Basic with TD Error (Complete)

**File**: [Actor_Critic_Basic.ipynb](Actor_Critic_Basic.ipynb)

**Status**: ✅ COMPLETE - Full implementation with extensive debugging

**Key Functions**:
- `run_single_step()`: Single-step execution across parallel games with batched inference
- `compute_td_error()`: Temporal Difference error calculation
- `calc_reward()`: Velocity-magnitude-weighted reward system (requires prev_state)
- `evaluate_policy_simple()`: Evaluation with live plotting and auto-detected reward components

**Algorithm Flow**:
```
1. Single step from each parallel game (batched forward pass)
2. TD target: y = r + γ*V(s') (with torch.no_grad() to detach!)
3. TD error: δ = y - V(s)
4. Critic loss: δ² + value regularization
5. Actor loss: -log π(a|s) * δ (detached)
6. Update both networks with gradient clipping (max_norm=0.5)
```

**Key Differences from REINFORCE**:
- **Online updates** (per-step) vs episodic
- **Lower variance**, slight bias (bootstrapping)
- **Faster learning**, more stable convergence
- **Separate optimizers** for actor/critic
- **Two networks**: DroneGamerBoi (actor) + DroneTeacherBoi (critic)

**Critical Bugs Fixed** (see [ACTOR_CRITIC_LESSONS_LEARNED.md](ACTOR_CRITIC_LESSONS_LEARNED.md)):
1. **Moving target problem**: `next_values` must be detached with `torch.no_grad()` or critic never converges
2. **Gamma too low**: Increased from 0.90 → 0.99 for 100-300 step episodes
3. **Reward hacking**: Fixed hovering/zooming exploits with velocity-magnitude-weighted rewards
4. **Missing velocity alignment bonus**: Re-added +0.5 reward for directional progress

**Real Challenges Documented**:
- **"Zoom past" exploit**: Policy learned to accelerate toward platform then crash far away
  - Solution: Velocity-magnitude-weighted rewards with MIN_MEANINGFUL_SPEED threshold
- **Hovering exploit**: Drone stayed near platform making tiny movements for infinite rewards
  - Solution: Hovering penalty (-1.0) when speed < 0.05, no rewards when speed < 0.15
- **Critic oscillation**: Values never converged due to gradients flowing through both V(s) and V(s')
  - Solution: Detach TD target with `torch.no_grad()` around `next_values = critic(next_states)`
- **Invisible terminal rewards**: Landing reward (+500) was worth ~0 after 100 steps with gamma=0.90
  - Solution: Increased gamma to 0.99 (effective horizon ~100 steps)

### Actor-Critic GAE (In Progress)

**File**: [Actor_Critic_GAE.ipynb](Actor_Critic_GAE.ipynb)

**Status**: 🚧 Work in progress (22 cells, debugging phase)

Note: GAE debugging proved challenging; project pivoted to PPO for better results.

Generalized Advantage Estimation will reduce variance further using exponentially-weighted advantages.

### Actor-Critic PPO (Complete)

**File**: [Actor_Critic_PPO.ipynb](Actor_Critic_PPO.ipynb)

**Status**: ✅ COMPLETE - Functional PPO implementation with metrics tracking

Successfully trained with best results yet. Includes clipped objective, multi-epoch training, and comprehensive metrics logging.

Proximal Policy Optimization adds clipped policy updates for more stable training.

## Common Hyperparameters

```python
learning_rate = 1e-3          # AdamW optimizer (both actor/critic)
bellman_gamma = 0.99          # Discount factor (CRITICAL: use 0.99 for 100-300 step tasks!)
gradient_clip = 0.5           # max_norm for grad clipping (essential for stability)
num_parallel_games = 6        # Parallel environments
max_steps = 300-500           # Episode length (can use curriculum)
eval_interval = 10            # Evaluate every N iterations
temperature = 0.3             # Stochastic sampling (0=greedy, 1=fully stochastic)
```

**Important Notes on Hyperparameters**:
- **Gamma = 0.99 is critical**: For 100-300 step episodes, gamma must be ≥0.99 or terminal rewards become invisible
  - Rule: `gamma^T ≥ 0.1` where T is episode length
  - Effective horizon ≈ 1/(1-gamma): 0.99 → 100 steps, 0.90 → 10 steps
- **Gradient clipping = 0.5**: Conservative but stable, prevents policy collapse
- **Learning rate = 1e-3**: Good default, consider 5e-4 for actor if updates too aggressive

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
- **Hovering**: Drone stays near platform without landing → Add hovering penalties, require MIN_MEANINGFUL_SPEED
- **Zooming**: Accelerating past target at high speed → Add speed penalties when close, velocity-magnitude weighting
- **Spinning**: Rotating in place for rewards → Add angular velocity penalties
- **Oscillating**: Moving back/forth for distance rewards → Use velocity-aligned rewards based on distance_delta
- **Fuel waste**: Unnecessary thrust → Fuel penalties or remove fuel rewards

**Velocity-Magnitude-Weighted Rewards** (Actor-Critic improvement):
```python
if prev_state is not None:
    distance_delta = prev_state.distance_to_platform - state.distance_to_platform
    speed = state.speed
    MIN_MEANINGFUL_SPEED = 0.15  # Prevents hovering exploit

    if speed >= MIN_MEANINGFUL_SPEED and velocity_toward_platform > 0.1:
        # Reward: (distance closed) × (speed multiplier)
        speed_multiplier = 1.0 + speed * 2.0
        rewards['approach'] = distance_delta * 15.0 * speed_multiplier
    elif speed < 0.05:
        rewards['hovering'] = -1.0  # Harsh penalty for hovering
```

This requires tracking `prev_state` in training loop and passing it to `calc_reward()`.

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

## Environments

### Delivery Drone Landing (Complete)

**Directory**: [delivery_drone/](delivery_drone/)

**Status**: ✅ Complete and well-tested

The primary training environment - a physics-based drone landing game with clean gym-style API.

**Key Features**:
- 15-dimensional state space (all normalized)
- 3 binary actions (main thrust, left thrust, right thrust)
- Socket-based parallel training (6+ games simultaneously)
- Spawn randomization for robust policies
- Manual play mode for understanding the task

**Training Architecture**:
- Two-process system: socket server + training client
- Blocking game loop (waits for action before stepping)
- Deterministic physics given fixed seed

See [delivery_drone/README.md](delivery_drone/README.md) for full documentation.

### Visual Driving 2D (In Development)

**Directory**: [visual_driving_2d/](visual_driving_2d/)

**Status**: 🚧 Environment functional, training not yet demonstrated

A CarRacing environment based on Gymnasium's CarRacing-v3 with custom socket server for parallel training.

**Key Features**:
- Based on official Gymnasium CarRacing-v3 (pygame rendering)
- Box2D physics for realistic car dynamics
- 96×96×3 RGB pixel observations + telemetry
- 3 continuous actions (steer, gas, brake)
- Free-running game loop at 50 FPS (async action submission)
- Gymnasium integration for WSL2 compatibility (pygame works, pyglet doesn't)

**Major Architecture Differences from Drone**:

| Feature | Delivery Drone | Visual Driving 2D |
|---------|---------------|-------------------|
| **Observation** | 15D state vector | 96×96×3 RGB + telemetry dict |
| **Game Loop** | Blocking (sync) | Free-running at 50 FPS (async) |
| **Actions** | 3 binary | 3 continuous |
| **Physics** | Custom (gravity, thrust) | Box2D (realistic car) |
| **Rendering** | Pygame | Pygame (via Gymnasium) |
| **Task** | Land on platform | Complete racing track |

**WSL2 Rendering Solution**:
Original implementation used pyglet (broken on WSL2 due to OpenGL context issues). Solution: Use Gymnasium's CarRacing-v3 with pygame backend, wrapped in custom socket server. See [visual_driving_2d/GYMNASIUM_SETUP.md](visual_driving_2d/GYMNASIUM_SETUP.md).

**Socket Server**:
```bash
# Start server with Gymnasium CarRacing-v3
python socket_server_gymnasium.py --num-games 6 --render none

# Test manually
python manual_play_pygame.py
```

## Key Learning Insights

1. **Reward engineering is 90% of the work** - Most time is spent debugging reward functions, not the algorithm
2. **Always detach TD targets** - Critical for Actor-Critic: `with torch.no_grad(): next_values = critic(next_states)`
3. **Match gamma to task horizon** - Use gamma ≥ 0.99 for tasks with 100+ step episodes
4. **Watch for reward hacking** - Hovering, zooming, spinning are common exploits
5. **Velocity-based rewards > proximity rewards** - Reward distance_delta with speed weighting, not just proximity
6. **Gradient clipping is essential** - Large updates cause policy collapse
7. **Advantage normalization reduces variance** - Normalizing advantages stabilizes training
8. **Parallel environments improve sample efficiency** - 6 games = 6x more data per iteration
9. **Require meaningful progress** - Use MIN_MEANINGFUL_SPEED thresholds to prevent micro-movement exploitation

## Documentation

### Core Documentation
- [README.md](README.md) - Project overview and learning path
- [ACTOR_CRITIC_LESSONS_LEARNED.md](ACTOR_CRITIC_LESSONS_LEARNED.md) - **Critical bugs & fixes (26 KB)** - Must read for Actor-Critic implementation!

### Delivery Drone Environment
- [delivery_drone/README.md](delivery_drone/README.md) - Environment API documentation
- [delivery_drone/SOCKET_API.md](delivery_drone/SOCKET_API.md) - Socket protocol details
- [delivery_drone/docs/RL_METHODS_SUMMARY.md](delivery_drone/docs/RL_METHODS_SUMMARY.md) - Theory notes

### Visual Driving 2D Environment
- [visual_driving_2d/README.md](visual_driving_2d/README.md) - CarRacing environment overview
- [visual_driving_2d/GYMNASIUM_SETUP.md](visual_driving_2d/GYMNASIUM_SETUP.md) - WSL2 pygame rendering setup
- [visual_driving_2d/IMPLEMENTATION_NOTES.md](visual_driving_2d/IMPLEMENTATION_NOTES.md) - Design decisions
- [visual_driving_2d/SOCKET_API.md](visual_driving_2d/SOCKET_API.md) - Socket protocol for async game loop

### Blog Posts (Technical Writing)
- [blogs/RL-1/policy_gradient.md](blogs/RL-1/policy_gradient.md) - Policy Gradients blog post (40 KB)
- [blogs/RL-Actor-critic/TD_error.md](blogs/RL-Actor-critic/TD_error.md) - Actor-Critic blog post (35 KB)

## Training Timeline & Expectations

### Actor-Critic Training (with all fixes applied)

Based on 6 parallel games, 300 steps/episode, ~1800 updates/iteration:

| Iterations | Total Steps | Expected Behavior | Success Rate |
|-----------|-------------|-------------------|--------------|
| **0-100** | ~180K | Random exploration, learning basic values | 0% |
| **100-300** | ~540K | Approaching platform, values stabilizing | 0-5% |
| **300-500** | ~900K | Gets close, sometimes overshoots | 5-15% |
| **500-1000** | ~1.8M | Controlled approaches, some landings | 10-30% |
| **1000-2000** | ~3.6M | Consistent landings | 50-70% |
| **2000-3000** | ~5.4M | Expert performance | 70-90% |

**Minimum Useful Training**: 1000 iterations (~20-30% landing rate)
**Target Training**: 2000-3000 iterations (60-80% landing rate)
**Diminishing Returns**: After 5000 iterations

### Signs of Good Training
- ✓ Critic loss decreasing
- ✓ TD errors getting smaller
- ✓ Drone moves toward platform (not just falling)
- ✓ Occasional landings by iteration 500
- ✓ Landing rate increasing over time

### Red Flags
- ❌ Critic loss not decreasing after 500 iterations → Check if `next_values` is detached
- ❌ Drone behavior not improving at iteration 300 → Check gamma value
- ❌ Values exploding (>1000) → Add value regularization
- ❌ Policy collapse (always same action) → Add entropy bonus
- ❌ Reward not increasing → Check for reward hacking exploits

## Important Notes

- This is a **learning repository** - code is documented but not optimized for production
- Training is **stochastic** - results vary between runs
- The **delivery_drone environment is deterministic** given fixed random seed
- Socket server must be running before training notebooks
- All state normalization is handled by the environment
- Model versioning uses simple incrementing (v1, v2, v3, etc.)
- **Read [ACTOR_CRITIC_LESSONS_LEARNED.md](ACTOR_CRITIC_LESSONS_LEARNED.md)** before implementing Actor-Critic - it documents 4 critical bugs!
