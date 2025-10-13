# RL Optimization Methods Summary

This document provides a comprehensive overview of reinforcement learning optimization methods discussed for training the Delivery Drone agent.

---

## 1. **Vanilla Policy Gradient (REINFORCE)**

**Core idea**: Increase probability of actions that led to high returns

```python
# Collect episode
for step in episode:
    action = policy(state)
    next_state, reward, done = env.step(action)

# Compute returns
G_t = Σ γ^k * reward_{t+k}

# Loss
loss = -log_prob(action) * G_t
```

**Pros**: Simple, works
**Cons**: High variance, slow learning

---

## 2. **Policy Gradient with Baseline**

**Core idea**: Subtract average return to reduce variance

```python
# Collect multiple episodes
all_returns = [G_1, G_2, ..., G_n]

# Baseline = mean
baseline = mean(all_returns)

# Loss with advantage
advantage = G_t - baseline
loss = -log_prob(action) * advantage
```

**Pros**: Lower variance than vanilla
**Cons**: Still relatively high variance

---

## 3. **Actor-Critic (A2C)**

**Core idea**: Use a learned value function V(s) as baseline

```python
# Two networks
policy_net = Actor()   # Outputs action probabilities
value_net = Critic()   # Outputs V(s)

# Collect experience
action = policy_net(state)
next_state, reward, done = env.step(action)

# Compute advantage using critic
V_s = value_net(state)
V_s_next = value_net(next_state)
advantage = reward + gamma * V_s_next - V_s  # TD advantage

# Update actor
actor_loss = -log_prob(action) * advantage.detach()

# Update critic
critic_loss = (V_s - (reward + gamma * V_s_next)) ** 2
```

**Pros**: Lower variance, faster learning
**Cons**: Two networks to train, can be unstable

---

## 4. **Proximal Policy Optimization (PPO)**

**Core idea**: Limit how much policy can change per update (clipping)

```python
# Collect trajectories with OLD policy
old_log_probs = collect_episodes(old_policy)

# Compute advantages (using value network)
advantages = compute_advantages(value_net)

# Update with clipping
for epoch in range(K_epochs):
    new_log_probs = policy_net(states)

    # Ratio of new/old probabilities
    ratio = exp(new_log_probs - old_log_probs)

    # Clipped objective
    clip_ratio = clip(ratio, 1-epsilon, 1+epsilon)
    loss = -min(ratio * advantage, clip_ratio * advantage)
```

**Pros**: Stable, sample efficient, SOTA for many tasks
**Cons**: More complex implementation

---

## 5. **Q-Learning / Deep Q-Network (DQN)**

**Core idea**: Learn Q(s,a) directly, then pick best action

```python
# Q-network outputs Q-values for all actions
q_values = q_net(state)  # [Q(s,a0), Q(s,a1), ...]

# Pick best action
action = argmax(q_values)

# TD learning
q_predicted = q_values[action]
q_target = reward + gamma * max(q_net(next_state))

loss = (q_predicted - q_target) ** 2
```

**Pros**: Off-policy (can reuse old data), simple concept
**Cons**: Only for discrete actions, can be unstable

---

## 6. **GRPO (Group Relative Policy Optimization)**

**Core idea**: Use group of rollouts from same state as baseline

```python
# Sample K trajectories from SAME state
for k in range(K):
    state = env.reset(seed=fixed)  # Same start
    G_k = run_episode()

# Group baseline
baseline = mean([G_1, G_2, ..., G_K])

# Advantages relative to group
for k in range(K):
    advantage_k = G_k - baseline
    loss += -log_prob_k * advantage_k
```

**Pros**: No value network needed
**Cons**: Need multiple samples per state, harder with random resets

---

## Comparison Table

| Method | Networks Needed | Variance | Sample Efficiency | Complexity | Best For |
|--------|----------------|----------|-------------------|------------|----------|
| **REINFORCE** | 1 (policy) | High | Low | Simple | Learning/prototyping |
| **Baseline** | 1 (policy) | Medium | Medium | Simple | Small problems |
| **A2C** | 2 (actor+critic) | Low | Medium | Medium | Continuous control |
| **PPO** | 2 (actor+critic) | Low | High | High | Most tasks (SOTA) |
| **DQN** | 1 (Q-network) | Medium | High | Medium | Discrete actions |
| **GRPO** | 1 (policy) | Medium | Low | Medium | LLM fine-tuning |

---

## Additional Concepts Discussed

### **Advantage Estimation Methods**

1. **Monte Carlo**: `A = G_t - baseline`
2. **TD advantage**: `A = r + γ*V(s') - V(s)`
3. **GAE (Generalized Advantage Estimation)**: Blend of both

### **Reward Shaping**

- Distance-based rewards: `(500 - distance) / 5000`
- Gaussian closeness: `exp(-distance²/σ²)`
- Exponential decay: `exp(-k * distance)`

### **Batching Strategies**

1. Single step updates (bad - high variance)
2. Full episode batching (okay - better variance)
3. Multiple episode batching (good - much better variance)
4. Experience replay buffer (great for DQN - sample efficiency)

---

## Key RL Concepts

### **Value Function (V)**
Expected total return from a state:
```
V(s) = Expected return from state s (following policy π)
```

### **Q-Function (Action-Value)**
Expected total return from a state-action pair:
```
Q(s, a) = Expected return from taking action a in state s, then following policy π
```

### **Advantage Function (A)**
How much better is an action compared to the average:
```
A(s, a) = Q(s, a) - V(s)
```

### **Relationship**
```
Q(s, a) = R(s, a) + γ * V(s')
V(s) = E[Q(s, a)]  (expectation over actions)
A(s, a) = Q(s, a) - V(s)
```

---

## Recommendation for Delivery Drone

### **Start with**: Policy Gradient + Mean Baseline (Method 2)
- Simple to implement
- Works well for problem size
- Good learning experience

### **Graduate to**: PPO (Method 4)
- Industry standard
- Best performance
- Worth the complexity once you understand basics

### **Avoid for now**:
- DQN (would need to discretize 3 actions into 8 combinations)
- GRPO (overkill, hard with random environment resets)

---

## Implementation Priority

```
1. ✅ Vanilla Policy Gradient (understand basics)
   ↓
2. ✅ Add mean baseline (reduce variance)
   ↓
3. ✅ Add value network (Actor-Critic)
   ↓
4. ✅ Add PPO clipping (stable SOTA performance)
```

---

## Example: Policy Gradient with Baseline

Here's a complete implementation example for the drone:

```python
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

# Policy network (3 independent binary actions)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [main, left, right]
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.network(state)

# Training parameters
EPISODES_PER_UPDATE = 10
GAMMA = 0.99
LR = 3e-4

policy_net = PolicyNetwork()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

# Training loop
for iteration in range(1000):
    batch_log_probs = []
    batch_returns = []

    # Collect multiple episodes
    for episode in range(EPISODES_PER_UPDATE):
        state = env.reset()
        episode_log_probs = []
        episode_rewards = []
        done = False

        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state_to_array(state), dtype=torch.float32)

            # Get action probabilities
            action_probs = policy_net(state_tensor)

            # Sample actions
            dist = Bernoulli(probs=action_probs)
            actions = dist.sample()

            # Compute log probability
            log_prob = dist.log_prob(actions).sum()

            # Execute action
            action_dict = {
                'main_thrust': int(actions[0]),
                'left_thrust': int(actions[1]),
                'right_thrust': int(actions[2])
            }
            next_state, reward, done, info = env.step(action_dict)

            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            state = next_state

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        batch_log_probs.extend(episode_log_probs)
        batch_returns.extend(returns)

    # Convert to tensors
    log_probs_tensor = torch.stack(batch_log_probs)
    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32)

    # Compute advantages (baseline = mean)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy gradient loss
    loss = -(log_probs_tensor * advantages).mean()

    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Iteration {iteration}, Loss: {loss.item():.4f}, Avg Return: {returns_tensor.mean().item():.2f}")
```

---

## References

- Sutton & Barto - Reinforcement Learning: An Introduction
- Schulman et al. - Proximal Policy Optimization Algorithms (PPO)
- Mnih et al. - Playing Atari with Deep Reinforcement Learning (DQN)
- Williams - Simple Statistical Gradient-Following Algorithms (REINFORCE)
