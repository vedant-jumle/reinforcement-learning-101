# Reinforcement Learning 101

My hands-on journey learning reinforcement learning from scratch. This repository contains:
- **From-scratch RL algorithm implementations** (policy gradients, actor-critic, etc.)
- **Custom environments** for experimentation (drone landing)
- **Training notebooks** documenting real challenges and solutions
- **Reward engineering** lessons learned through trial and error

*A learning repository featuring real implementations with all their messy details - not just wrappers around libraries.*

---

## RL Algorithm Implementations

### Policy Gradient Methods

#### [Policy_Gradients.ipynb](Policy_Gradients.ipynb) - REINFORCE with Baseline
**Status:** Complete

A complete from-scratch implementation of REINFORCE with baseline for the drone landing task.

**What's Implemented:**
- Policy network (3-layer MLP with LayerNorm)
- Parallel episode collection from multiple game instances
- Batched policy inference for GPU efficiency
- Monte Carlo returns computation (Bellman equation)
- Advantage estimation with baseline
- Gradient clipping and advantage normalization
- Custom reward shaping (velocity alignment, horizontal positioning)

**Real Challenges Documented:**
- **Hovering exploit** - Policy learned to hover near platform instead of landing
  - Solution: Time penalties + hovering-specific penalties
- **Spinning exploit** - Policy learned to rotate in place for positive rewards
  - Solution: Angular velocity penalties + removed continuous distance rewards
- **Horizontal misalignment** - Drone approaching but missing the 100px-wide platform
  - Solution: Explicit horizontal alignment rewards with tight tolerances
- **Training instability** - Large gradient updates causing policy collapse
  - Solution: Gradient clipping (max_norm=0.5) + advantage normalization

**Key Techniques:**
```python
# Advantage calculation with baseline
returns_tensor = torch.tensor(batch_returns)
baseline = returns_tensor.mean()
advantages = (returns_tensor - baseline)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Policy gradient loss
loss = -(log_probs_tensor * advantages).mean()
```

**Training Results:**
- Trained with 6 parallel game instances
- Curriculum learning: 75 → 250 max steps over 1000 iterations
- Achieved occasional landings after ~500 iterations
- Baseline improved from -324 → +16 (positive returns!)

---

### Actor-Critic Methods

#### [Actor_Critic_Basic.ipynb](Actor_Critic_Basic.ipynb) - Basic Actor-Critic with TD Error
**Status:** Complete

Implementation of basic Actor-Critic using Temporal Difference learning for real-time, per-step updates.

**What's Implemented:**
- Two-network architecture (Actor policy + Critic value function)
- Online learning with TD error as advantage signal
- Batched parallel training across 6 game instances
- Velocity-magnitude-weighted reward system
- Critical bug fixes: moving target problem, gamma tuning, reward hacking

**Key Improvements over REINFORCE:**
- Per-step updates (no waiting for episode completion)
- Lower variance through value function baseline
- Faster convergence (68% success in half the training time)

**Critical Bugs Fixed** (see [docs/ACTOR_CRITIC_LESSONS_LEARNED.md](docs/ACTOR_CRITIC_LESSONS_LEARNED.md)):
1. **Moving target problem** - TD target must be detached with `torch.no_grad()`
2. **Gamma too low** - Increased from 0.90 to 0.99 for 100-300 step episodes
3. **Reward hacking** - Fixed hovering/zooming exploits with velocity-based rewards

---

#### [Actor_Critic_GAE.ipynb](Actor_Critic_GAE.ipynb) - Generalized Advantage Estimation
**Status:** Attempted but failed

Experimental implementation of GAE for variance reduction. Training completed but failed to solve the task effectively. Documented as a learning experience in [blogs/RL-2-Actor-critic/part2_gae_to_ppo.md](blogs/RL-2-Actor-critic/part2_gae_to_ppo.md).

**Outcome:** GAE debugging proved challenging; pivoted to PPO for better results.

---

#### [Actor_Critic_PPO.ipynb](Actor_Critic_PPO.ipynb) - Proximal Policy Optimization
**Status:** Complete

Best-performing implementation using PPO's clipped objective for stable policy updates.

**What's Implemented:**
- Clipped surrogate objective for policy updates
- Multi-epoch training on collected trajectories
- Both policy and value network training
- Comprehensive metrics tracking

**Training Results:**
- Achieved 76% landing success rate
- Most stable training of all implementations
- Saved model weights: policy v1 + critic v1

---

### Inference and Evaluation

#### [Policy_Gradients_inference.ipynb](Policy_Gradients_inference.ipynb)
Load and evaluate trained REINFORCE models. Includes visualization and performance analysis.

#### [Actor_Critic_inference.ipynb](Actor_Critic_inference.ipynb)
Load and evaluate trained Actor-Critic models (Basic, GAE, PPO). Supports all three variants with model loading utilities.

---

### Future Implementations (Planned)
- **Deep Q-Networks (DQN)** - Value-based methods for discrete actions
- **Soft Actor-Critic (SAC)** - Off-policy with entropy regularization
- **Group Relative Policy Optimization (GRPO)** - Methods used to train Deepseek-r1

---

## Environments

### [Delivery Drone Landing](delivery_drone/)

A physics-based drone landing game designed as a clean RL environment for experimentation.

**Features:**
- Clean gym-style API for RL training
- Socket-based distributed training support
- Parallel environment support (multi-game instances)
- Spawn randomization for robust policy learning
- Physics-based simulation (gravity, thrust, drag)
- Manual play mode for understanding the task

**Key RL Challenges:**
- Continuous control (thrust and rotation)
- Sparse rewards with terminal success
- Multi-objective optimization (speed, angle, position, fuel)
- Reward shaping to avoid exploitation (hovering, spinning)
- Horizontal alignment precision (100px platform width)

**Landing Criteria:**
- Speed ≤ 3.0 pixels/frame (0.3 normalized)
- Angle ≤ 20° from upright (0.111 normalized)
- Bottom center of drone must be on platform
- Must have fuel remaining

**Status:** Complete - Successfully trained with multiple RL algorithms

[View Full Documentation](delivery_drone/README.md)

---

## Quick Start

### Run Algorithm Implementations

```bash
# Install dependencies
pip install torch numpy pygame jupyter tqdm

# Start Jupyter
jupyter notebook

# Open and run:
# - Policy_Gradients.ipynb (REINFORCE with baseline)
# - Actor_Critic_Basic.ipynb (Actor-Critic with TD error)
# - Actor_Critic_PPO.ipynb (PPO implementation)
```

### Train in the Drone Environment

```bash
# Navigate to environment
cd delivery_drone

# Install dependencies
pip install pygame numpy torch

# Play manually to understand the task
python manual_play.py

# Start training server with 6 parallel games
python socket_server.py --num-games 6 --render none --randomize-drone

# Run your training script
# See Policy_Gradients.ipynb or Actor_Critic_Basic.ipynb for training code
```

---

## Learning Path

This repository documents learning RL through **implementation, not just theory**.

### Phase 1: Understanding Policy Gradients (Complete)
- Implemented REINFORCE with baseline from scratch
- Built custom drone landing environment with socket server
- Learned reward shaping the hard way (debugging hovering/spinning exploits)
- Implemented parallel training infrastructure
- Discovered importance of velocity-based rewards over proximity rewards
- Learned gradient clipping and advantage normalization

**Key Insight:** *Reward engineering is 90% of the work in RL. The algorithm is the easy part.*

### Phase 2: Actor-Critic Methods (Complete)
- Implemented basic Actor-Critic with TD error (68% success rate)
- Attempted Generalized Advantage Estimation (failed, documented lessons)
- Implemented Proximal Policy Optimization (76% success rate, best results)
- Learned critical importance of detaching TD targets
- Discovered gamma must match task horizon (0.99 for 100-300 step tasks)
- Fixed reward hacking exploits (hovering, zooming) with velocity-based rewards

**Key Insight:** *Three bugs cost me days: moving target problem, invisible rewards from low gamma, and reward function exploits. Always detach your TD targets.*

### Phase 3: Value-Based Methods (Planned)
- Deep Q-Networks (DQN) for discrete action spaces
- Experience replay and target networks
- Comparing sample efficiency with policy gradient methods

---

## Technologies

- **Python 3.12+** - Primary language
- **PyTorch** - Neural networks and automatic differentiation
- **Pygame** - Game rendering and physics simulation
- **NumPy** - Numerical computations
- **Jupyter** - Interactive development and documentation
- **Socket Programming** - Distributed training infrastructure

---

## Documentation

Each implementation and environment contains detailed documentation:

- **Training Notebooks** - Complete implementations with explanations
- **Environment READMEs** - API documentation, setup instructions
- **Implementation Notes** - Design decisions and lessons learned
- **[docs/ACTOR_CRITIC_LESSONS_LEARNED.md](docs/ACTOR_CRITIC_LESSONS_LEARNED.md)** - Critical Actor-Critic bugs and fixes
- **[delivery_drone/docs/RL_METHODS_SUMMARY.md](delivery_drone/docs/RL_METHODS_SUMMARY.md)** - Theory notes

### Blog Posts
- **[blogs/RL-1-Policy-Gradients/policy_gradient.md](blogs/RL-1-Policy-Gradients/policy_gradient.md)** - REINFORCE implementation journey
- **[blogs/RL-2-Actor-critic/actor_critic.md](blogs/RL-2-Actor-critic/actor_critic.md)** - Comprehensive Actor-Critic blog (91 KB)
- **[blogs/RL-2-Actor-critic/part1_actor_critic_bugs.md](blogs/RL-2-Actor-critic/part1_actor_critic_bugs.md)** - Debugging Actor-Critic (Part 1)
- **[blogs/RL-2-Actor-critic/part2_gae_to_ppo.md](blogs/RL-2-Actor-critic/part2_gae_to_ppo.md)** - From GAE to PPO (Part 2)

---

## Resources

### RL Theory
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI's comprehensive RL guide
- [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html) - The RL textbook
- [CS285 Deep RL](http://rail.eecs.berkeley.edu/deeprlcourse/) - Berkeley's Deep RL course
- [My RL Methods Summary](delivery_drone/docs/RL_METHODS_SUMMARY.md) - Personal notes on RL algorithms

### Implementation References
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - Production RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file RL implementations
- [Spinning Up Code](https://github.com/openai/spinningup) - OpenAI's RL implementations

---

## Contributing

This is a personal learning repository, but discussions are welcome! Feel free to:
- Open issues for questions about implementations
- Share your own RL experiments or training tips
- Suggest improvements to reward functions or architectures
- Report bugs in environments or training code

**Note:** This repo prioritizes **learning and experimentation** over production-ready code. Code is documented but not optimized.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

Free to use for learning, research, and experimentation.

---

## Acknowledgments

This learning journey is inspired by:
- **OpenAI Spinning Up** - For the excellent RL curriculum
- **CS285** (UC Berkeley) - For deep RL theoretical foundations
- **Sutton & Barto** - For the RL bible
- **The RL research community** - For publishing implementations and lessons learned

**Note:** This repository is actively being developed as I learn. Expect regular updates, new implementations, and improved training techniques. Follow along for the messy reality of learning RL!
