# 2D Visual Driving RL

**Learning autonomous driving from raw pixel observations using progressive complexity training.**

This project implements a 2D top-down driving environment where an RL agent learns to navigate, avoid obstacles, and handle traffic by processing visual input through convolutional neural networks - no clean state vectors, only pixels.

---

## 🎯 Project Philosophy

**Visual Feature Learning:** Unlike traditional RL environments that provide clean state vectors (positions, velocities, etc.), this environment only provides RGB pixel observations. The agent must learn to extract relevant features (lane positions, obstacle locations, traffic patterns) directly from images using CNNs.

**Progressive Complexity:** Start simple, add complexity incrementally. Each phase builds on skills learned in previous phases, creating a curriculum that mirrors real-world driving learning.

**Extensibility:** This is an open-ended project. There's always another challenge to add, another behavior to learn, another edge case to handle.

---

## 🚗 Progressive Complexity Roadmap

### Phase 1: Basic Driving Mechanics
**Goal:** Learn fundamental vehicle control from visual input

**Environment:**
- Simple roads with lane markings (straight and curved sections)
- Single vehicle, no obstacles or traffic
- Top-down camera view showing local area around car

**Agent Must Learn:**
- Stay within lane boundaries
- Maintain appropriate speed
- Handle curves smoothly
- Basic steering control

**Observation Space:**
- RGB image: 84x84x3 (or similar)
- Shows road, lane markings, and vehicle
- May use frame stacking (4 consecutive frames)

**Action Space:**
- Continuous: `[steering_angle, acceleration]`
  - Steering: [-1.0, 1.0] (left to right)
  - Acceleration: [-1.0, 1.0] (brake to accelerate)

**Reward Components:**
- `+1.0` per timestep on road
- `-0.1` per timestep off road or crossed lane line
- `-0.01 * |steering_angle|` (penalty for jerky steering)
- `-1.0` for collision with road boundary

**Success Criteria:**
- Complete circuit without leaving lane
- Smooth steering behavior (low variance)

---

### Phase 2: Goal-Directed Navigation
**Goal:** Navigate a road network to reach target destinations

**New Elements:**
- Road network with intersections (T-junctions, 4-way intersections)
- Spawn point and goal location visualization
- Multiple possible routes to destination

**Agent Must Learn:**
- Make correct turn decisions at intersections
- Follow routes efficiently
- Remember navigation context (requires memory/attention)
- Balance exploration vs. exploitation in route finding

**Observation Space:**
- Same visual input as Phase 1
- Optional: Small goal direction indicator in observation

**Action Space:**
- Same as Phase 1

**Reward Components:**
- All Phase 1 rewards
- `+10.0` for reaching goal
- `+0.1 * progress_toward_goal` (reward shaping)
- `-0.01` per timestep (efficiency incentive)
- `-5.0` for wrong turn at intersection

**Success Criteria:**
- Reach goal from random spawn in < N timesteps
- Success rate > 80% over 100 episodes
- Learn shortest path over time

---

### Phase 3: Static Obstacles
**Goal:** Collision avoidance with stationary objects

**New Elements:**
- Parked cars on road sides
- Road construction barriers
- Debris, cones, barricades
- Narrow passages requiring precise control

**Agent Must Learn:**
- Detect obstacles from visual input
- Plan collision-free paths
- Adjust trajectory mid-drive
- Safe but efficient obstacle avoidance

**Observation Space:**
- Same visual format
- Now includes rendered obstacles

**Action Space:**
- Same as Phase 1 & 2

**Reward Components:**
- All Phase 2 rewards
- `-10.0` for collision with obstacle
- `-0.5` for getting too close to obstacles (danger zone)
- `+0.1` bonus for maintaining safe distance while progressing

**Success Criteria:**
- Navigate to goal without collisions
- Handle cluttered environments (many obstacles)
- Maintain reasonable speed (not over-cautious)

---

### Phase 4: Dynamic Traffic (Simple AI)
**Goal:** Drive safely in traffic with other vehicles

**New Elements:**
- Other vehicles following scripted behaviors:
  - A* or Dijkstra pathfinding to their goals
  - Simple rule-based lane keeping
  - Constant speed control
- Multiple lanes on major roads
- Traffic density variations

**Agent Must Learn:**
- Detect moving vehicles from visual input (motion cues)
- Predict other vehicle trajectories
- Safe following distance
- Lane changing when beneficial
- Yielding and merging behavior
- Overtaking slower vehicles

**Observation Space:**
- Same visual input (now with moving objects)
- Frame stacking becomes critical for motion perception

**Action Space:**
- Same as previous phases

**Reward Components:**
- All Phase 3 rewards
- `-20.0` for collision with other vehicles
- `-1.0` for dangerous lane changes (cutting off others)
- `+0.5` for successful overtaking maneuvers
- `-0.1` per timestep following too closely

**Success Criteria:**
- Zero-collision driving in moderate traffic
- Efficient navigation (not stuck behind slow vehicles)
- Smooth, predictable behavior (other cars can plan around you)

---

### Phase 5: Traffic Control Signals
**Goal:** Follow traffic rules and signals

**New Elements:**
- Traffic lights at intersections (red, yellow, green)
- Stop signs
- Yield signs
- Right-of-way rules

**Agent Must Learn:**
- Recognize traffic light states from visual input
- Stop at red lights, proceed on green
- Handle yellow light dilemmas
- Stop sign complete stops
- Right-of-way at uncontrolled intersections

**Reward Components:**
- All Phase 4 rewards
- `-15.0` for running red light or stop sign
- `+1.0` for correct traffic signal behavior
- `-5.0` for causing accidents at intersections

---

### Phase 6: Pedestrians and Crosswalks
**Goal:** Share road safely with pedestrians

**New Elements:**
- Pedestrians crossing at crosswalks
- Unpredictable pedestrian behavior
- School zones with children

**Agent Must Learn:**
- Detect pedestrians from visual input
- Yield at crosswalks
- Slow down in pedestrian-heavy areas
- Emergency braking when needed

---

### Phase 7: Environmental Variations
**Goal:** Robust driving under varying conditions

**New Elements:**
- Time of day (day, dusk, night)
- Weather effects (rain reducing visibility)
- Different road textures and appearances
- Lighting changes

**Agent Must Learn:**
- Generalize lane detection across conditions
- Adjust speed for conditions (slower in rain)
- Robust feature extraction (not overfit to one visual style)

---

### Phase 8: Highway Driving
**Goal:** High-speed multi-lane navigation

**New Elements:**
- High-speed roads with 3-4 lanes
- Merge ramps and exits
- Heavy traffic at high speeds
- Lane-specific rules (HOV lanes, exit-only lanes)

**Agent Must Learn:**
- High-speed control precision
- Multi-lane planning
- Safe merging at speed
- Exit planning and execution

---

### Phase 9: Parking Scenarios
**Goal:** Precision control in tight spaces

**New Elements:**
- Parking lots with spaces
- Parallel parking on streets
- Backing up required

**Agent Must Learn:**
- Reverse driving
- Precision positioning
- Multi-point turn maneuvers

---

### Phase 10+: Advanced Challenges
- **Roundabouts** - Complex right-of-way and yielding
- **Construction zones** - Dynamic road closures
- **Accidents** - Emergency obstacle avoidance
- **Rush hour** - Dense traffic patterns
- **Multi-agent RL** - All cars learning simultaneously
- **Vehicle-to-vehicle communication** - Cooperative driving
- **Adversarial scenarios** - Handling aggressive drivers

---

## 🧠 Technical Architecture

### Visual Observation Processing

**Input Pipeline:**
```
Raw Game Frame (800x600 RGB)
    ↓
Crop to relevant area (center on vehicle)
    ↓
Resize to 84x84x3
    ↓
Grayscale conversion (optional, → 84x84x1)
    ↓
Frame stacking (stack 4 frames → 84x84x4)
    ↓
Normalize to [0, 1]
    ↓
Feed to CNN
```

**CNN Architecture (Example):**
```python
Conv2D(32, kernel=8, stride=4) + ReLU
    ↓
Conv2D(64, kernel=4, stride=2) + ReLU
    ↓
Conv2D(64, kernel=3, stride=1) + ReLU
    ↓
Flatten → Dense(512) + ReLU
    ↓
Policy Head: Dense(action_dim)  # Mean of continuous actions
Value Head: Dense(1)             # State value (for actor-critic)
```

### Policy Network

**Policy Gradient (REINFORCE/PPO):**
- CNN feature extractor (shared trunk)
- Policy head outputs mean and log_std for continuous actions
- Gaussian policy: `action ~ N(μ(observation), σ²)`

**Actor-Critic:**
- Shared CNN feature extractor
- Separate heads for actor (policy) and critic (value function)
- Reduces variance compared to pure policy gradient

### Reward Engineering

**Key Principles:**
- Dense rewards for learning signal (every timestep)
- Large penalties for safety violations (collisions, traffic violations)
- Efficiency incentives (time penalties, progress rewards)
- Smooth behavior rewards (penalize jerky steering)

**Reward Debugging:**
- Track individual reward components separately
- Plot reward composition over training
- Watch for reward exploitation (e.g., spinning in place)

---

## 🔧 Implementation Plan

### Socket-Based Distributed Training Architecture

**One of the most important features**: This project uses a **TCP socket interface** to separate the game process from the agent training process. This enables:

✅ **Distributed Training** - Run game on one machine, RL agent on another (e.g., GPU server)
✅ **Parallel Environments** - Easily scale to multiple game instances for faster data collection
✅ **Language Agnostic** - Any language with socket support can control the game
✅ **Clean Separation** - Game rendering and physics separate from ML training code
✅ **Development Flexibility** - Update agent without restarting game, or vice versa

**Architecture:**
```
┌──────────────────────┐         ┌──────────────────────┐
│   Socket Server      │◄───────►│   Socket Client      │
│   (Game Process)     │   TCP   │   (Agent Process)    │
│                      │  5555   │                      │
│  • Runs game loop    │         │  • Sends actions     │
│  • Renders visuals   │         │  • Gets observations │
│  • Physics sim       │         │  • Trains CNN policy │
│  • Multiple games    │         │  • Collects episodes │
└──────────────────────┘         └──────────────────────┘
```

**Protocol:**
- JSON messages over TCP, newline-delimited
- Commands: `RESET`, `STEP`, `GET_STATE`, `CLOSE`
- Responses: State (RGB frame + metadata), reward, done, info
- Low latency: ~1-3ms per step on localhost

**Example Usage:**
```python
# Terminal 1: Start game server with 6 parallel games
python socket_server.py --num-games 6 --render none

# Terminal 2: Train agent that connects to all 6 games
from game.socket_client import DrivingGameClient

clients = [DrivingGameClient(port=5555) for _ in range(6)]

# Each client can control a different game instance
for i, client in enumerate(clients):
    obs = client.reset(game_id=i)  # Reset game i

# Collect parallel rollouts
for step in range(max_steps):
    for i, client in enumerate(clients):
        action = policy(observations[i])
        obs, reward, done, info = client.step(action, game_id=i)
```

### Environment Structure

**Core Components:**
```
visual_driving_2d/
├── game/
│   ├── game_engine.py        # Core driving game logic
│   ├── physics.py            # Vehicle physics simulation
│   ├── road_network.py       # Road graph and pathfinding
│   ├── socket_server.py      # TCP server for remote training
│   ├── socket_client.py      # Client library for agents
│   └── config.py             # Game configuration
├── environments/
│   ├── base_env.py           # Abstract base driving environment
│   ├── phase1_basic.py       # Phase 1: Basic driving
│   ├── phase2_navigation.py  # Phase 2: Goal-directed
│   ├── phase3_obstacles.py   # Phase 3: Static obstacles
│   └── phase4_traffic.py     # Phase 4: Dynamic traffic
├── rendering/
│   ├── renderer.py           # Pygame rendering engine
│   ├── assets/               # Sprites, textures
│   └── camera.py             # Top-down camera logic
├── agents/
│   ├── traffic_ai.py         # Simple rule-based cars (A*)
│   └── pedestrian_ai.py      # Pedestrian behavior
├── networks/
│   ├── cnn_policy.py         # CNN-based policy network
│   ├── actor_critic.py       # Actor-critic architecture
│   └── feature_extractors.py # Reusable CNN architectures
├── training/
│   ├── train_pg.py           # Policy gradient training
│   ├── train_ppo.py          # PPO training
│   └── train_a2c.py          # Actor-critic training
├── utils/
│   ├── road_generator.py     # Procedural road network generation
│   ├── replay_buffer.py      # For off-policy methods
│   └── visualization.py      # Training plots, attention maps
├── examples/
│   ├── remote_agent.py       # Example remote agent
│   └── benchmark_latency.py  # Socket performance testing
├── notebooks/
│   ├── phase1_training.ipynb
│   ├── phase2_training.ipynb
│   └── visual_analysis.ipynb # Analyze what the CNN learned
├── socket_server.py          # Main server script
├── manual_play.py            # Play manually with keyboard
├── SOCKET_API.md             # Complete socket protocol docs
└── README.md
```

### Gym-Style API

```python
import numpy as np
from visual_driving_2d import Phase1Environment

env = Phase1Environment(render_mode='rgb_array')
observation, info = env.reset()

# observation: np.ndarray, shape (84, 84, 3), dtype=uint8
# info: dict with episode metadata

done = False
total_reward = 0

while not done:
    action = policy(observation)  # shape (2,) for [steering, accel]
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print(f"Episode reward: {total_reward}")
```

---

## 🎨 Visual Feature Engineering

One of the key learning opportunities in this project is understanding what visual features matter for driving:

### Feature Visualization Techniques

1. **Saliency Maps** - What pixels does the policy look at?
2. **Grad-CAM** - Which regions of the image activate the CNN?
3. **Filter Visualization** - What patterns do convolutional filters detect?
4. **Activation Maximization** - What input maximizes specific neurons?

### Ablation Studies

Compare performance with different visual inputs:
- Raw RGB vs. Grayscale
- Different frame stack sizes (1, 2, 4, 8 frames)
- Different image resolutions (64x64, 84x84, 128x128)
- With/without data augmentation
- Semantic segmentation vs. raw pixels

### Feature Engineering Baselines

Before end-to-end learning, try explicit feature extraction:
- Edge detection (Canny, Sobel)
- Lane line detection (Hough transform)
- Optical flow for motion
- Color-based segmentation

Compare hand-crafted features vs. learned features.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy pygame opencv-python matplotlib tqdm jupyter
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- Pygame 2.5+
- NumPy, OpenCV for image processing

### Phase 1 Quick Start

**Option 1: Integrated (Local Development)**
```bash
cd visual_driving_2d

# Manual play to understand the task
python manual_play.py --phase 1

# Train directly (game + agent in same process)
python training/train_pg.py --phase 1 --episodes 1000 --integrated

# Visualize learned policy
python visualize_policy.py --checkpoint checkpoints/phase1_best.pt
```

**Option 2: Distributed (Production Training)**
```bash
# Terminal 1: Start game server with 6 parallel games
python socket_server.py --num-games 6 --render none --phase 1

# Terminal 2: Train agent remotely
python training/train_pg.py --phase 1 --episodes 1000 --distributed --num-games 6

# Benefits:
# - Faster training (6x parallel data collection)
# - Separate processes (game crash won't kill training)
# - Can run on different machines (game on workstation, agent on GPU server)
# - Headless rendering for speed
```

**Option 3: Remote Training (Different Machines)**
```bash
# Machine 1 (Game Server): Run multiple game instances
python socket_server.py --num-games 10 --render none --host 0.0.0.0 --port 5555

# Machine 2 (GPU Server): Train agent
python training/train_pg.py --host game-server-ip --port 5555 --num-games 10

# Advantages:
# - Offload rendering to dedicated machine
# - Use GPU resources efficiently
# - Scale horizontally (multiple game servers)
```

---

## 🔌 Socket-Based Training Workflow

### Why Use the Socket Architecture?

**Performance Benefits:**
- **Parallel Data Collection**: Run 6-10 game instances simultaneously
- **Sample Efficiency**: Collect diverse experiences faster
- **Fault Tolerance**: Game crash doesn't lose training progress
- **Resource Optimization**: Separate CPU (game) and GPU (training) workloads

**Development Benefits:**
- **Hot Reload**: Update agent code without restarting game
- **Debug Friendly**: Inspect game state independently
- **Monitoring**: Track game FPS separately from training throughput
- **Language Flexibility**: Write agent in any language (Python, Julia, C++, etc.)

### Server Configuration

```bash
# Headless training (fastest, no visualization)
python socket_server.py --num-games 6 --render none

# Visualize one game while training on 6
python socket_server.py --num-games 6 --render human

# Custom FPS for faster/slower simulation
python socket_server.py --num-games 6 --render none --fps 120

# Randomize spawn for curriculum learning
python socket_server.py --num-games 6 --render none --randomize-spawn
```

### Client Training Pattern

```python
from game.socket_client import DrivingGameClient
import torch

# Connect to server
client = DrivingGameClient(host='localhost', port=5555)
num_games = client.num_games  # Discovered during handshake

# Training loop with parallel games
for iteration in range(num_iterations):
    # Reset all games
    observations = [client.reset(game_id=i) for i in range(num_games)]

    # Collect rollouts in parallel
    for step in range(max_steps_per_episode):
        # Batch inference for efficiency
        actions = policy.get_actions(observations)  # Batched

        # Step all games
        transitions = []
        for i in range(num_games):
            obs, reward, done, info = client.step(actions[i], game_id=i)
            transitions.append((observations[i], actions[i], reward, done))
            observations[i] = obs

            if done:
                observations[i] = client.reset(game_id=i)

        # Update policy using transitions from all games
        policy.update(transitions)
```

### Performance Benchmarks

**Expected Throughput (localhost):**
- Single game: ~60 FPS (limited by rendering)
- Headless single game: ~300-500 FPS
- 6 parallel headless games: ~1800-3000 samples/sec
- 10 parallel headless games: ~3000-5000 samples/sec

**Latency:**
- Step command: 1-3ms (localhost)
- Reset command: 2-5ms (localhost)
- Remote (same network): +5-10ms
- Remote (internet): +50-200ms (still usable!)

---

## 📊 Evaluation Metrics

### Performance Metrics
- **Success Rate**: % episodes reaching goal without collisions
- **Average Return**: Mean cumulative reward per episode
- **Time to Goal**: Average timesteps to reach goal
- **Collision Rate**: Collisions per 1000 timesteps

### Driving Quality Metrics
- **Lane Keeping**: % time spent in lane
- **Steering Smoothness**: Standard deviation of steering actions
- **Speed Profile**: Average speed, speed variance
- **Safety Distance**: % time maintaining safe following distance

### Learning Efficiency
- **Sample Efficiency**: Success rate vs. timesteps trained
- **Wall-Clock Time**: Training time to reach success threshold
- **Stability**: Variance in performance across seeds

---

## 🔬 Research Questions

This project explores several interesting questions:

1. **Visual Representations**: What features do CNNs learn for driving? Do they detect lane lines, vehicles, motion?

2. **Sample Efficiency**: How many samples needed to learn each phase? How does visual input affect sample complexity vs. state-based input?

3. **Transfer Learning**: Can skills learned in Phase 1 transfer to Phase 2? Can we freeze early CNN layers?

4. **Curriculum Learning**: Is progressive complexity better than training on full complexity from the start?

5. **Generalization**: How well do policies generalize to new road layouts, traffic patterns, visual appearances?

6. **Human Comparison**: How does RL agent learning curve compare to human learning to drive?

---

## 📚 References & Inspiration

### Visual RL Papers
- **Playing Atari with Deep RL** (Mnih et al., 2013) - DQN, the foundational visual RL work
- **Human-level control through deep RL** (Mnih et al., 2015) - Nature DQN paper
- **Asynchronous Methods for Deep RL** (Mnih et al., 2016) - A3C
- **End to End Learning for Self-Driving Cars** (NVIDIA, 2016) - CNN for steering prediction

### Autonomous Driving RL
- **Deep Reinforcement Learning for Autonomous Driving** (Kendall et al., 2019)
- **Learning to Drive in a Day** (Kendall et al., 2019) - Sample efficiency
- **ChauffeurNet** (Bansal et al., 2019) - Imitation learning + RL

### Simulators & Environments
- **CARLA** - Realistic 3D autonomous driving simulator
- **TORCS** - Open-source racing simulator
- **Highway-Env** - Minimalist highway driving environments
- **Atari 2600** - Classic visual RL benchmark

### Implementation References
- **Stable-Baselines3** - Production RL algorithms
- **CleanRL** - Single-file RL implementations
- **Dopamine** - Research framework for RL

---

## 🔮 Future Extensions

This project is intentionally open-ended. Ideas for extension:

### Multi-Agent Scenarios
- All cars learning simultaneously (multi-agent RL)
- Emergent traffic patterns
- Communication protocols between vehicles

### Sim-to-Real Transfer
- Domain randomization for visual robustness
- Reality gap analysis
- Transfer to real RC car with camera

### Hierarchical RL
- High-level route planning + low-level control
- Option learning for maneuvers (lane change, turn, park)

### Inverse RL / Imitation Learning
- Learn from human driving demonstrations
- Behavioral cloning baseline
- GAIL or similar

### Safety & Robustness
- Constrained RL (hard safety constraints)
- Adversarial robustness (perturbations to images)
- Out-of-distribution detection

### Advanced Perception
- 3D perception from 2D images
- Depth estimation
- Object detection and tracking
- Semantic segmentation

---

## 🎯 Learning Objectives

By completing this project, you will learn:

✅ **Visual RL Fundamentals**
- CNN-based policy networks
- Frame stacking and preprocessing
- Visual feature learning

✅ **Curriculum Design**
- Progressive complexity training
- Reward shaping for complex behaviors
- Transfer learning between tasks

✅ **Policy Gradient Methods**
- REINFORCE, PPO, A2C/A3C for continuous control
- Handling high-dimensional observation spaces
- Variance reduction techniques

✅ **Autonomous Driving Challenges**
- Perception from raw sensors
- Multi-objective optimization (safety + efficiency)
- Handling dynamic environments

✅ **Deep RL Engineering**
- Debugging visual RL agents
- Hyperparameter tuning for CNNs
- Efficient training infrastructure

---

## 🤝 Contributing

This is a personal learning project, but discussions and ideas are welcome!

**Areas for Contribution:**
- New phase ideas and challenges
- Improved reward functions
- CNN architecture experiments
- Training efficiency improvements
- Visualization tools

---

## 📝 License

MIT License - Free to use for learning, research, and experimentation.

---

## 🙏 Acknowledgments

Inspired by:
- **DeepMind's Atari DQN** - Showed visual RL at scale
- **OpenAI Gym** - Standard RL environment API
- **CARLA Simulator** - Realistic driving simulation
- **Spinning Up in Deep RL** - RL education resource

---

**Status:** 🚧 In Development

This README will be updated as the project progresses through each phase. Follow along for the real journey of building visual RL from scratch!
