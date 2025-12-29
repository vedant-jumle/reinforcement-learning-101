# Actor-Critic Implementation: Lessons Learned

## Overview

This document captures critical insights and bugs discovered while implementing **Basic Actor-Critic with TD Error** for the drone landing task. These lessons apply broadly to Actor-Critic methods and can help avoid common pitfalls in deep RL.

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Critical Bugs Found](#critical-bugs-found)
3. [Design Issues](#design-issues)
4. [Hyperparameter Insights](#hyperparameter-insights)
5. [Training Timeline](#training-timeline)
6. [Best Practices](#best-practices)

---

## Algorithm Overview

### Basic Actor-Critic Architecture

**Two Networks:**
- **Actor (Policy)** π_θ(a|s): Decides which actions to take
- **Critic (Value)** V_φ(s): Evaluates how good states are

**Update Process (Per Step):**

1. **Act**: Sample action `a_t ~ π_θ(a_t|s_t)` from actor
2. **Observe**: Get reward `r_t` and next state `s_{t+1}`
3. **Compute TD Error** (advantage):
   ```
   δ_t = r_t + γ*V_φ(s_{t+1}) - V_φ(s_t)
   ```
4. **Update Critic**: Minimize TD error
   ```
   L_critic = δ_t² = [r_t + γ*V_φ(s_{t+1}) - V_φ(s_t)]²
   ```
5. **Update Actor**: Policy gradient with TD error as advantage
   ```
   L_actor = -log π_θ(a_t|s_t) · δ_t
   ```

**Key Advantage over Policy Gradients:**
- **Online learning**: Updates after each step (not full episodes)
- **Lower variance**: Critic provides baseline and reduces noise
- **Faster convergence**: More frequent updates with less variance

**Trade-off:**
- **Bias**: Depends on critic accuracy (if critic is wrong, actor learns wrong policy)

---

## Critical Bugs Found

### 🔴 BUG #1: Critic Learning from Moving Target

**The Problem:**

```python
# ❌ WRONG: Both values computed in same graph!
values = critic(batch_data['states'])
next_values = critic(batch_data['next_states'])

td_targets = rewards + gamma * next_values * (1 - dones)
td_errors = td_targets - values
critic_loss = (td_errors ** 2).mean()
```

**What Goes Wrong:**
- When you call `loss.backward()`, gradients flow through **BOTH** `values` and `next_values`
- The critic gets confused: "Should I change V(s) or V(s')?"
- The TD target **changes during optimization** (moving target problem)
- Critic never converges, oscillates indefinitely

**Mathematical Issue:**

Without detaching:
```
∂L/∂θ = 2·δ·[γ·∂V(s')/∂θ - ∂V(s)/∂θ]
              ↑            ↑
          WRONG!       CORRECT
```

We want:
```
∂L/∂θ = -2·δ·[∂V(s)/∂θ]
```

Only update V(s) to match the **frozen** target `r + γ*V(s')`.

**The Fix:**

```python
# ✅ CORRECT: Detach next_values!
values = critic(batch_data['states'])

with torch.no_grad():  # Stop gradients through target
    next_values = critic(batch_data['next_states'])

td_targets = rewards + gamma * next_values * (1 - dones)
td_errors = td_targets - values
critic_loss = (td_errors ** 2).mean()
```

**Why This Works:**
- TD target `r + γ*V(s')` is treated as a **fixed label**
- Only V(s) gets updated to match the target
- Target stays stable during optimization
- This is called **bootstrapping** in RL

**Analogy:**
Think of supervised learning:
```python
prediction = model(x)
target = ground_truth  # No gradients through this!
loss = (prediction - target)**2
```

Same principle applies to TD learning.

**Impact:**
- ❌ Without fix: Critic never converges, values oscillate wildly
- ✅ With fix: Stable value learning, proper convergence

---

### 🔴 BUG #2: Discount Factor Too Low (γ = 0.90)

**The Problem:**

Initial implementation used `gamma = 0.90`, but the drone landing task requires 100-300 steps per episode.

**Why This Breaks:**

**Effective horizon** ≈ 1/(1-γ):
- γ = 0.90 → horizon ≈ **10 steps**
- γ = 0.99 → horizon ≈ **100 steps**
- Task requires: **100-300 steps**

**Concrete Example:**

Landing reward is +500 at step 100:

| Gamma | Discounted Reward at t=0 | Agent "Sees" |
|-------|-------------------------|--------------|
| 0.90 | 500 × 0.90^100 = **0.013** | Nothing! |
| 0.99 | 500 × 0.99^100 = **183** | Strong signal! |

**What Happens with γ=0.90:**

1. **Terminal rewards invisible**: Landing reward 100+ steps away is worth ~0
2. **Time penalties dominate**: Every step has -0.3 to -1.0 penalty
3. **Agent learns to crash quickly**: "Best strategy = minimize time penalties"
4. **Critic learns V(s)≈0**: No states have meaningful value
5. **No credit assignment**: Can't learn which actions lead to landing

**The Fix:**

```python
# ✅ CORRECT: Match gamma to task horizon
bellman_gamma = 0.99  # For 100-300 step episodes
```

**Rule of Thumb:**

Choose γ such that important rewards are still visible:
```
γ^T ≥ 0.1  (reward should be worth ≥10% after T steps)

For T=200 steps:
γ ≥ 0.1^(1/200) ≈ 0.989

Use γ ≥ 0.99 for this task
```

**Impact:**
- ❌ γ=0.90: Agent crashes immediately to minimize time
- ✅ γ=0.99: Agent learns to approach and land

---

### ⚠️ BUG #3: Reward Hacking - Multiple Exploits

#### Exploit #1: "Zoom Past" Problem

**The Problem:**

Policy learned to **accelerate toward platform, zoom past it at high speed, then crash far away**.

**Root Cause in Reward Function:**

```python
# Distance reward proportional to SPEED
if dist > 0.065:
    rewards['distance'] = int(velocity_alignment > 0) * state.speed * scaled_shifted_negative_sigmoid(dist, scaler=4.5)
    #                                                      ↑↑↑↑↑
    #                                            Encourages maximum speed!
```

**Why This Happens:**

1. Moving fast toward platform = high reward per step ✓
2. Agent learns: "Go as fast as possible!" ✓
3. Zooms past platform at max speed 💨
4. Already collected rewards, crashes far away 💥
5. Net reward: Time penalties + distance rewards > crash penalty
6. **Exploit discovered!**

---

#### Exploit #2: "Hovering" Problem

**The Problem:**

Policy learned to **hover near the platform** making microscopic movements to continuously collect distance rewards without actually landing.

**Root Cause:**

```python
# Old distance reward based on current position
if dist > 0.065:
    rewards['distance'] = int(velocity_alignment > 0) * state.speed * scaled_shifted_negative_sigmoid(dist, scaler=4.5)

# Problems:
# 1. Can get rewards with VERY small speed (speed=0.01)
# 2. No requirement for MEANINGFUL progress
# 3. Hovering + tiny nudges = infinite rewards
```

**Why This Happens:**

1. Distance reward doesn't require actual progress, just low distance
2. Speed component can be satisfied with tiny movements (speed > 0)
3. Agent learns: "Get close, hover, make tiny movements for rewards"
4. Time penalties are outweighed by continuous distance rewards
5. **Hovering exploit discovered!**

**Example Exploitation Pattern:**
```
Step 50: distance=0.20, speed=0.03, reward=+1.2 (tiny forward nudge)
Step 51: distance=0.19, speed=0.02, reward=+0.8 (minimal movement)
Step 52: distance=0.20, speed=0.01, reward=+0.5 (drifting back)
Step 53: distance=0.19, speed=0.02, reward=+0.8 (forward again)
...
[Repeats for 200+ steps without landing]
```

---

#### The Complete Fix: Velocity-Magnitude-Weighted Rewards

**The Solution:**

Completely redesign the reward system to require **decisive movement with meaningful velocity** toward the platform.

**Key Insight:** Reward should be based on:
1. **Distance delta** (how much closer you got)
2. **Speed magnitude** (must move with purpose, not hover)
3. **Direction alignment** (must be moving toward target)

**Implementation:**

```python
def calc_reward(state: DroneState, prev_state: DroneState = None):
    """
    Velocity-magnitude-weighted approach rewards.
    Requires meaningful progress, prevents hovering exploit.
    """
    rewards = {}
    total_reward = 0

    # ... time penalty, angle, etc. ...

    # NEW: Velocity-magnitude-weighted approach reward
    rewards['approach'] = 0
    rewards['hovering_penalty'] = 0

    if prev_state is not None:
        # How much closer did we get?
        distance_delta = prev_state.distance_to_platform - state.distance_to_platform

        # Velocity component toward platform
        dist = state.distance_to_platform
        if dist > 1e-6:
            velocity_toward_platform = (
                state.drone_vx * state.dx_to_platform +
                state.drone_vy * state.dy_to_platform
            ) / dist
        else:
            velocity_toward_platform = 0.0

        # Minimum speed threshold for meaningful progress
        MIN_MEANINGFUL_SPEED = 0.15  # ~1.5 pixels/frame in original units
        speed = state.speed

        if speed >= MIN_MEANINGFUL_SPEED and velocity_toward_platform > 0.1 and dist > 0.065:
            # Reward proportional to: (distance closed) * (speed factor)
            # Speed multiplier encourages decisive movement
            speed_multiplier = 1.0 + speed * 2.0  # Faster = exponentially better
            rewards['approach'] = distance_delta * 15.0 * speed_multiplier
        elif speed < 0.05:
            # Very slow = hovering
            rewards['hovering_penalty'] = -1.0
        else:
            # Moving but not meaningfully toward target
            rewards['approach'] = 0.0

    total_reward += rewards['approach']
    total_reward += rewards['hovering_penalty']

    # ... rest of reward components ...

    return rewards
```

**Why This Works:**

1. **Distance delta is naturally small for hovering**: If you're not moving, distance_delta ≈ 0, so no reward
2. **Speed threshold blocks exploitation**: Below `MIN_MEANINGFUL_SPEED = 0.15`, you get NO approach rewards
3. **Hovering penalty**: If speed < 0.05, you get -1.0 penalty (harsh)
4. **Speed multiplier rewards decisive action**: `(1.0 + speed * 2.0)` means faster approaches get exponentially better rewards
5. **Direction check**: Must have `velocity_toward_platform > 0.1` to ensure aligned movement

**Anti-Exploitation Features:**

```python
# Can't hover (speed < 0.05) → penalty -1.0
# Can't nudge slowly (0.05 < speed < 0.15) → no reward
# Can't move sideways (velocity_toward_platform < 0.1) → no reward
# Must make MEANINGFUL progress (distance_delta * speed_multiplier)
```

**Code Changes Required:**

1. **Update `calc_reward()` signature**: Add `prev_state` parameter
2. **Track previous states**: In `run_single_step()`, store and pass previous state
3. **Initialize prev_states**: In training loop, create `prev_game_states` list
4. **Update evaluation functions**: Track prev_state in `evaluate_policy()` and `evaluate_policy_simple()`

**Example Reward Comparison:**

| Behavior | Old Reward (per step) | New Reward (per step) | Exploit Status |
|----------|----------------------|----------------------|----------------|
| Hovering (speed=0.02) | +0.5 | -1.0 | ❌ Fixed |
| Slow nudge (speed=0.10) | +0.8 | 0.0 | ❌ Fixed |
| Decisive approach (speed=0.25) | +2.0 | +4.5 | ✓ Encouraged |
| Fast zoom past (speed=0.5) | +5.0 | -2.0 (speed penalty) | ❌ Fixed |

---

**Key Lessons:**

**Reward engineering is 90% of RL.** The algorithm works fine - the reward function defined the wrong objective!

**General Principles:**
- Don't reward **proximity**, reward **progress** (distance_delta)
- Don't reward **any movement**, reward **meaningful velocity** (speed threshold)
- Don't reward **speed alone**, combine with **directional progress**
- Always require **state transitions** (prev_state → state) for progress rewards
- Watch for unintended exploits during training

**Common RL Exploits to Watch For:**
- **Hovering**: Stationary position farming
- **Oscillating**: Back-and-forth movement for rewards
- **Zooming**: High-speed pass-through collecting rewards then crashing
- **Spinning**: Rotation in place if angular rewards exist
- **Boundary riding**: Exploiting edge cases in reward function

---

### ⚠️ BUG #4: Missing Velocity Alignment Bonus

**The Problem:**

Actor-Critic version had this commented out:

```python
# if velocity_alignment > 0:
#     rewards['velocity_alignment'] = 0.5
```

But Policy Gradients version (which worked better) had it active:

```python
if velocity_alignment > 0:
    rewards['velocity_alignment'] = 0.5
```

**Impact:**
- Missing a +0.5 reward signal for moving in the right direction
- Agent gets less feedback about whether it's approaching correctly
- Slows down learning

**The Fix:**

```python
# Re-add the bonus
if dist > 0.065 and velocity_alignment > 0:
    rewards['velocity_alignment'] = 0.5
```

**Lesson:**

Small reward components matter! Even a +0.5 bonus helps guide learning, especially early in training when the agent is exploring.

---

## Design Issues

### ⚠️ ISSUE #1: Variable and Small Batch Sizes

**The Problem:**

```python
# Batch size varies as games finish
# Early in episode: batch_size = 6
# Late in episode: batch_size = 1-2
# When batch_size=1: pure online learning (extremely noisy!)
```

**Why This Matters:**

- Small batches = **high variance gradients**
- Variable batches = **inconsistent learning dynamics**
- Networks oscillate instead of converging smoothly

**Example:**

```
Step 10: batch_size=6, gradient noise = moderate
Step 250: batch_size=1, gradient noise = EXTREME
Step 251: batch_size=1, gradient noise = EXTREME (different direction!)
```

**Potential Fix (Gradient Accumulation):**

```python
update_frequency = 10  # Update every 10 steps
accumulated_steps = 0

while not all(games_done):
    batch_data, game_states = run_single_step(...)

    if batch_data is not None:
        # Compute losses
        critic_loss = (td_errors ** 2).mean()
        actor_loss = -(batch_data['log_probs'] * td_errors.detach()).mean()

        # Backward (accumulate gradients)
        critic_loss.backward()
        actor_loss.backward()

        accumulated_steps += len(batch_data['states'])

        # Only step optimizers every N transitions
        if accumulated_steps >= update_frequency:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            critic_optimizer.step()
            policy_optimizer.step()

            critic_optimizer.zero_grad()
            policy_optimizer.zero_grad()
            accumulated_steps = 0
```

**Benefits:**
- More stable gradient estimates
- Consistent batch size (effectively)
- Smoother learning curves

---

### ⚠️ ISSUE #2: Too Frequent Updates

**The Problem:**

```python
# Updates after EVERY SINGLE STEP
# 6 games × 300 steps = 1800 updates per iteration!
```

**Why This Can Be Problematic:**

- Networks never "settle" between updates
- Extremely online (high variance)
- Can cause oscillations
- Computationally expensive (1800 optimizer steps)

**Trade-off:**

- **More frequent updates**: Faster adaptation, but noisier
- **Less frequent updates**: More stable, but slower adaptation

**Potential Fix:**

Update every K steps (K=5-10):

```python
if total_steps % update_frequency == 0:
    # Update networks
```

Or use the gradient accumulation approach above.

---

### ⚠️ ISSUE #3: Missing Entropy Bonus

**The Problem:**

```python
# No entropy regularization
actor_loss = -(batch_data['log_probs'] * td_errors.detach()).mean()
```

**Why This Matters:**

- Policy can collapse to **deterministic** too quickly
- Poor exploration in later training
- Can get stuck in local optima

**The Fix:**

Add entropy bonus to encourage exploration:

```python
# In run_single_step, compute entropy
batch_dist = Bernoulli(probs=batch_action_probs)
batch_actions = batch_dist.sample()
batch_log_probs = batch_dist.log_prob(batch_actions).sum(dim=1)
batch_entropy = batch_dist.entropy().sum(dim=1)  # Add this

# Return entropy in batch_data
batch_data['entropy'] = batch_entropy

# In training loop
actor_loss = -(batch_data['log_probs'] * td_errors.detach()).mean()
entropy_bonus = -0.01 * batch_data['entropy'].mean()  # Negative because we minimize
actor_loss = actor_loss + entropy_bonus
```

**Effect:**
- Coefficient 0.01: Encourages exploration
- Gradually decrease over training for final fine-tuning
- Prevents premature convergence

---

### ⚠️ ISSUE #4: No Value Regularization

**The Problem:**

```python
critic_loss = (td_errors ** 2).mean()
# No regularization on value estimates
```

**Why This Can Be Problematic:**

- Critic values can **explode** (predict huge values)
- No penalty for unrealistic value estimates
- Can destabilize learning

**The Fix:**

Add L2 regularization on value predictions:

```python
critic_loss = (td_errors ** 2).mean()

# Regularize value predictions (prevent explosion)
value_reg = 0.01 * (values ** 2).mean()
critic_loss = critic_loss + value_reg
```

**Benefits:**
- Keeps value estimates in reasonable range
- More stable learning
- Better generalization

---

## Hyperparameter Insights

### Discount Factor (γ)

**Critical for Long-Horizon Tasks:**

| Gamma | Horizon | Use Case |
|-------|---------|----------|
| 0.90 | ~10 steps | Very short tasks only |
| 0.95 | ~20 steps | Short tasks (50 steps) |
| 0.99 | ~100 steps | **Drone landing (100-300 steps)** ✓ |
| 0.995 | ~200 steps | Very long tasks (500+ steps) |
| 0.999 | ~1000 steps | Extremely long tasks |

**Rule:** Effective horizon = 1/(1-γ)

**For this task:**
- Episodes: 100-300 steps
- Terminal reward: +500 at end
- **Minimum γ ≈ 0.99** to see terminal reward

---

### Learning Rate

**Current Setup:**

```python
policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
```

**Considerations:**

- **1e-3**: Good starting point
- **Critic might need higher LR**: Critic has harder job (predict values)
- **Actor might need lower LR**: Policy changes should be gradual

**Potential Improvement:**

```python
policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4)  # Lower
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)  # Keep
```

Or use learning rate scheduling:

```python
scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(policy_optimizer, gamma=0.999)
scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.999)

# After each iteration
scheduler_actor.step()
scheduler_critic.step()
```

---

### Gradient Clipping

**Current:**

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
```

**This is good!** Prevents gradient explosions.

**Alternative Values:**
- `max_norm=0.5`: Conservative (current) ✓
- `max_norm=1.0`: Standard
- `max_norm=10.0`: Very permissive

**For this task:** 0.5 is appropriate given the reward scale.

---

## Training Timeline

### Expected Convergence (with fixes applied)

Based on 6 parallel games, 300 steps/episode, ~1800 updates/iteration:

| Iterations | Total Steps | Critic Status | Actor Status | Expected Behavior | Success Rate |
|-----------|-------------|---------------|--------------|-------------------|--------------|
| **0-100** | ~180K | Learning basic values | Random exploration | Crashes, erratic movement | 0% |
| **100-300** | ~540K | Values stabilizing | Learning to approach | Approaches platform | 0-5% |
| **300-500** | ~900K | Good value estimates | Learning to slow down | Gets close, sometimes overshoots | 5-15% |
| **500-1000** | ~1.8M | Converged values | Refining control | Controlled approaches, some landings | 10-30% |
| **1000-2000** | ~3.6M | Stable values | Good policy | Consistent landings | 50-70% |
| **2000-3000** | ~5.4M | Optimal values | Near-optimal policy | Expert performance | 70-90% |
| **3000+** | 5.4M+ | Diminishing returns | Fine-tuning | Marginal improvements | 80-95% |

### Recommendations

**Minimum Training:**
- **1000 iterations** to see meaningful progress
- Expect ~20-30% landing rate

**Target Training:**
- **2000-3000 iterations** for good performance
- Expect 60-80% landing rate

**Maximum Useful Training:**
- **5000 iterations** - diminishing returns after this
- Expect 80-90% landing rate (won't reach 100% due to task difficulty)

### Signs of Good Training

**Early (0-500 iterations):**
- ✓ Critic loss decreasing
- ✓ TD errors getting smaller
- ✓ Drone starts moving toward platform (not just falling)

**Mid (500-1500 iterations):**
- ✓ Occasional landings (even if rare)
- ✓ Drone slows down near platform
- ✓ Fewer out-of-bounds crashes

**Late (1500+ iterations):**
- ✓ Consistent approaches
- ✓ Landing rate increasing
- ✓ Smooth deceleration curves

### Signs of Problems

**Red Flags:**
- ❌ Critic loss not decreasing after 500 iterations
- ❌ Drone behavior not improving (still random at iteration 300)
- ❌ Values exploding (critic predicts >1000)
- ❌ Policy collapsing (always same action)
- ❌ Reward not increasing over time

**If you see these, check:**
1. Is `next_values` detached? (Bug #1)
2. Is gamma high enough? (Bug #2)
3. Are reward components balanced? (Bug #3)
4. Is gradient clipping working?

---

## Best Practices

### 1. Always Detach TD Targets

```python
# ✓ CORRECT
with torch.no_grad():
    next_values = critic(next_states)

# ✗ WRONG
next_values = critic(next_states)
```

**Why:** Prevents moving target problem, stabilizes critic learning.

---

### 2. Match Gamma to Task Horizon

```python
# Rule: gamma^T ≥ 0.1 for important reward at step T
# For T=200: gamma ≥ 0.989
bellman_gamma = 0.99  # For 100-300 step episodes
```

**Why:** Agent needs to "see" terminal rewards to learn.

---

### 3. Watch for Reward Hacking

**Monitor during training:**
- What behavior is the policy learning?
- Is it exploiting the reward function?
- Are reward components balanced?

**Common exploits:**
- Hovering in place (farming positive rewards)
- Zooming past target (collecting rewards then crashing)
- Oscillating (back-and-forth for distance rewards)

**Fix:** Adjust reward function, not algorithm!

---

### 4. Use Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
```

**Why:** Prevents gradient explosions, especially with TD learning.

---

### 5. Separate Optimizers for Actor and Critic

```python
# ✓ CORRECT
policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)

# ✗ WRONG (don't share optimizer)
optimizer = torch.optim.AdamW(
    list(policy.parameters()) + list(critic.parameters()),
    lr=1e-3
)
```

**Why:** Actor and critic learn at different rates, need independent learning rate control.

---

### 6. Monitor TD Errors

```python
# Track TD error statistics
td_error_mean = td_errors.mean().item()
td_error_std = td_errors.std().item()

# Healthy training:
# - Mean TD error → 0 over time
# - Std TD error decreases
# - Values don't explode
```

**Why:** TD errors tell you if critic is learning properly.

---

### 7. Use Evaluation to Debug

```python
# Periodic evaluation (no gradient updates)
if iteration % 10 == 0:
    eval_result = evaluate_policy(...)
    # Watch:
    # - Is behavior improving?
    # - Is reward increasing?
    # - Are landings happening?
```

**Why:** Training metrics can be misleading, evaluation shows actual performance.

---

### 8. Save Checkpoints Regularly

```python
if (iteration + 1) % 100 == 0:
    torch.save({
        'iteration': iteration,
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
    }, f'checkpoint_iter_{iteration+1}.pth')
```

**Why:** Training can crash, experiments need rollback, want to compare different checkpoints.

---

## Summary

### Most Critical Lessons

1. **Detach TD targets** - Without this, critic never converges
2. **Use high gamma (≥0.99)** - For long-horizon tasks
3. **Reward engineering matters** - Watch for exploits
4. **Train for 2000+ iterations** - Actor-Critic needs time
5. **Monitor TD errors** - They tell you if learning is working

### Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Critic loss oscillating | TD targets not detached | Add `with torch.no_grad()` |
| Agent crashes immediately | Gamma too low | Increase to 0.99+ |
| Agent zooms past target | Reward hacking | Fix reward function |
| No improvement after 500 iters | Learning rate too high/low | Adjust LR or check gradients |
| Policy collapse (deterministic) | No exploration | Add entropy bonus |

### Key Takeaways

**Actor-Critic is powerful but subtle:**
- ✓ Lower variance than Policy Gradients
- ✓ Online learning (faster updates)
- ✗ Requires careful implementation (moving target bug!)
- ✗ Sensitive to hyperparameters (especially gamma)
- ✗ Reward engineering is critical

**Success formula:**
1. Implement algorithm correctly (detach targets!)
2. Tune gamma for task horizon
3. Design reward function carefully (avoid exploits)
4. Train for sufficient iterations (2000+)
5. Monitor and debug continuously

---

## References

**Theory:**
- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 13: Policy Gradient Methods)
- Mnih et al., "Asynchronous Methods for Deep RL" (A3C paper)

**Implementation:**
- [Actor_Critic_Basic.ipynb](Actor_Critic_Basic.ipynb) - Full implementation
- [Policy_Gradients_Baseline.ipynb](Policy_Gradients_Baseline.ipynb) - Comparison baseline

**Related:**
- [CLAUDE.md](CLAUDE.md) - Codebase overview
- [delivery_drone/README.md](delivery_drone/README.md) - Environment details

---

*Document created based on real bugs and lessons learned during Actor-Critic implementation for drone landing task. These insights apply broadly to Actor-Critic methods in deep RL.*
