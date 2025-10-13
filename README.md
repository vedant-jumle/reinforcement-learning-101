# Reinforcement Learning 101

My hands-on journey learning reinforcement learning from scratch. This repository contains:
- ✅ **From-scratch RL algorithm implementations** (policy gradients, actor-critic, etc.)
- ✅ **Custom environments** for experimentation (drone landing, racing games)
- ✅ **Training notebooks** documenting real challenges and solutions
- ✅ **Reward engineering** lessons learned through trial and error

*A learning repository featuring real implementations with all their messy details - not just wrappers around libraries.*

---

## 📝 RL Algorithm Implementations

### Policy Gradient Methods

#### [Policy_Gradients_Baseline.ipynb](Policy_Gradients_Baseline.ipynb) - REINFORCE with Baseline
**Status:** ✅ Complete

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
- 🐛 **Hovering exploit** - Policy learned to hover near platform instead of landing
  - Solution: Time penalties + hovering-specific penalties
- 🐛 **Spinning exploit** - Policy learned to rotate in place for positive rewards
  - Solution: Angular velocity penalties + removed continuous distance rewards
- 🐛 **Horizontal misalignment** - Drone approaching but missing the 100px-wide platform
  - Solution: Explicit horizontal alignment rewards with tight tolerances
- 🐛 **Training instability** - Large gradient updates causing policy collapse
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

### Future Implementations *(Planned)*
- **Actor-Critic (A2C)** - Reduce variance with value function
- **Proximal Policy Optimization (PPO)** - Stable policy updates with clipping
- **Deep Q-Networks (DQN)** - Value-based methods for discrete actions
- **Soft Actor-Critic (SAC)** - Off-policy with entropy regularization
- **Group Relative Policy Optimization (GRPO)** - Stuff used to train Deepseek-r1

---

## 🎯 Environments

### 🚁 [Delivery Drone Landing](delivery_drone/)

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

**Status:** ✅ Complete - Trainable with policy gradient methods

[→ View Full Documentation](delivery_drone/README.md)

---

### 🏎️ Forza RL *(Planned)*

Reinforcement learning for racing game control and optimization.

**Status:** 🚧 In Development

---

## 🚀 Quick Start

### Run Algorithm Implementations

```bash
# Install dependencies
pip install torch numpy pygame jupyter tqdm

# Start Jupyter
jupyter notebook

# Open and run:
# - Policy_Gradients_Baseline.ipynb (REINFORCE with baseline implementation)
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
# See Policy_Gradients_Baseline.ipynb for complete training code
```

---

## 📚 Learning Path

This repository documents learning RL through **implementation, not just theory**.

### Phase 1: Understanding Policy Gradients (✅ Complete)
- ✅ Implemented REINFORCE with baseline from scratch
- ✅ Built custom drone landing environment with socket server
- ✅ Learned reward shaping the hard way (debugging hovering/spinning exploits)
- ✅ Implemented parallel training infrastructure
- ✅ Discovered importance of velocity-based rewards over proximity rewards
- ✅ Learned gradient clipping and advantage normalization

**Key Insight:** *Reward engineering is 90% of the work in RL. The algorithm is the easy part.*

### Phase 2: Value-Based Methods (🚧 In Progress)
- Implementing DQN for discrete action spaces
- Exploring experience replay and target networks
- Comparing sample efficiency with policy gradients

### Phase 3: Advanced Policy Gradients (📋 Planned)
- Actor-Critic methods (A2C, A3C)
- Proximal Policy Optimization (PPO)
- Trust region methods

---

## 🛠️ Technologies

- **Python 3.12+** - Primary language
- **PyTorch** - Neural networks and automatic differentiation
- **Pygame** - Game rendering and physics simulation
- **NumPy** - Numerical computations
- **Jupyter** - Interactive development and documentation
- **Socket Programming** - Distributed training infrastructure

---

## 📖 Documentation

Each implementation and environment contains detailed documentation:

- **Training Notebooks** - Complete implementations with explanations
- **Environment READMEs** - API documentation, setup instructions
- **Implementation Notes** - Design decisions and lessons learned
- **[RL Methods Summary](delivery_drone/docs/RL_METHODS_SUMMARY.md)** - Theory notes

---

## 📊 Resources

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

## 🤝 Contributing

This is a personal learning repository, but discussions are welcome! Feel free to:
- Open issues for questions about implementations
- Share your own RL experiments or training tips
- Suggest improvements to reward functions or architectures
- Report bugs in environments or training code

**Note:** This repo prioritizes **learning and experimentation** over production-ready code. Code is documented but not optimized.

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

Free to use for learning, research, and experimentation.

---

## 🙏 Acknowledgments

This learning journey is inspired by:
- **OpenAI Spinning Up** - For the excellent RL curriculum
- **CS285** (UC Berkeley) - For deep RL theoretical foundations
- **Sutton & Barto** - For the RL bible
- **The RL research community** - For publishing implementations and lessons learned

**Note:** This repository is actively being developed as I learn. Expect regular updates, new implementations, and improved training techniques. Follow along for the messy reality of learning RL! 🚀
