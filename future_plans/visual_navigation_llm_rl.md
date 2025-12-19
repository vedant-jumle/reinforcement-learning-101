# Visual Navigation with LLM + RL

## Problem Statement

**Task**: Agent navigates to target location using only visual observations (Street View images or similar).

**Core Challenge**: Combine vision-language model reasoning with reinforcement learning to solve vision-based navigation in real-world imagery.

---

## Environment Setup

### Street View Navigation

**State Space**:
- RGB image (640x640 or similar) from Street View
- Target description (e.g., "Eiffel Tower" or address)
- Optional: Navigation history (past actions/observations)

**Action Space**:
- `move_forward` - Move 50-100m ahead in current direction
- `turn_left` - Rotate view 90° left
- `turn_right` - Rotate view 90° right
- `move_backward` - Move back along path

**Episode**:
- Start: Random Street View location in city
- Goal: Navigate to target landmark/address
- Terminal: Reach target (within 100m) OR max steps exceeded (50 steps)

**Reward Structure**:
```python
reward = -0.1  # Time penalty per step
if distance_decreased:
    reward += distance_delta * scale  # Progress reward
if reached_target:
    reward += 100.0  # Success bonus
```

---

## Research Questions

### 1. VLM vs RL Comparison
**Question**: How do vision-language models compare to learned RL policies for visual navigation?

**Approaches to compare**:
- VLM policy (GPT-4V, LLaVA) with in-context learning
- RL policy (PPO) with CNN encoder
- RL policy with CLIP encoder (vision-language pretrained)

**Metrics**:
- Success rate
- Average steps to target
- Generalization to new cities
- Sample efficiency (for RL)

### 2. Hybrid LLM + RL
**Question**: Can LLMs improve RL training for vision-based navigation?

**Hybrid architectures**:
- **VLM as reward model**: VLM judges action quality → auxiliary reward signal
- **VLM for exploration**: VLM suggests promising directions → guides RL exploration
- **Hierarchical**: VLM plans subgoals → RL executes low-level navigation
- **Behavior cloning initialization**: Train on VLM demonstrations → fine-tune with RL

### 3. Representation Learning
**Question**: Do vision-language representations help RL sample efficiency?

**Comparison**:
- Random initialized CNN encoder (learned from scratch)
- Frozen CLIP encoder (pretrained vision-language)
- Fine-tuned CLIP encoder (adapt to navigation)

**Hypothesis**: CLIP's scene understanding should help navigation vs random features.

### 4. Sparse Reward Challenge
**Question**: How do different exploration strategies handle sparse rewards?

**Strategies**:
- Random exploration (baseline)
- Curiosity-driven (intrinsic motivation)
- VLM-guided exploration (semantic understanding)
- Curriculum learning (start close, increase distance)

---

## Technical Challenges

### 1. Sparse Rewards
- Target might be kilometers away
- Random exploration unlikely to succeed
- Need reward shaping or curriculum

### 2. Large State Space
- Millions of possible Street View locations
- Every image is unique
- Generalization is critical

### 3. Long Horizon
- 20-50 steps to reach target
- Credit assignment problem
- Need memory/recurrence?

### 4. API Costs
- Google Street View API: ~$7/1000 requests
- Training could require thousands of episodes
- May need cached dataset or free alternative (Mapillary)

---

## Progressive Difficulty Levels

### Level 1: Landmark Finding (Easy)
- Start: Random location 500m from landmark
- Target: Famous landmark (visually distinctive)
- Success criteria: Within 100m
- Expected RL convergence: 500-1000 episodes

### Level 2: Address Navigation (Medium)
- Start: Random location 1km away
- Target: Street address (need to read signs)
- Success criteria: Within 50m
- Expected RL convergence: 2000-5000 episodes

### Level 3: Cross-City Generalization (Hard)
- Train: Paris + London
- Test: Tokyo (never seen)
- Measures: Transfer learning capability

---

## Alternative: Grid-World with Real Images

**Simpler version for prototyping**:

**Setup**:
- 10x10 grid, each cell has real Street View image
- Agent moves grid cell to cell (discrete)
- One cell is goal (landmark)
- Deterministic transitions

**Advantages**:
- Finite state space (100 cells vs infinite locations)
- Faster training (no API calls after dataset collection)
- Easier to debug and visualize
- Can still use vision (real images as observations)

**Progression**:
1. Solve 10x10 grid world with RL
2. Scale to larger grids (20x20)
3. Move to continuous Street View navigation

---

## Implementation Options

### Option A: Pure VLM (No Training)
- Use GPT-4V or LLaVA API
- In-context learning only
- Fastest to prototype
- Establishes baseline performance

### Option B: RL from Scratch
- PPO with CNN/CLIP encoder
- Learn entirely from rewards
- Tests pure RL capability
- Slowest to converge (sparse rewards)

### Option C: VLM → RL Distillation
1. VLM generates expert trajectories
2. Train RL policy via behavior cloning
3. Fine-tune with RL (PPO)
4. Best of both: VLM reasoning + RL efficiency

### Option D: Hybrid Architecture
- VLM high-level planner (subgoals)
- RL low-level executor (navigation)
- Continuous collaboration during execution

---

## Datasets & APIs

### Google Street View API
- Coverage: Global
- Cost: $7/1000 image requests
- Quality: High, 360° panoramas
- Limitation: Paid

### Mapillary
- Coverage: Good (community-sourced)
- Cost: Free
- Quality: Variable
- Limitation: Less dense coverage

### StreetLearn (DeepMind)
- Pre-downloaded Street View dataset
- Coverage: NYC, Pittsburgh, Paris
- Cost: Free
- Limitation: Static dataset (no new locations)

---

## Success Criteria

**Minimum viable result**:
- RL agent learns to navigate to landmarks in single city
- Success rate >50% on test landmarks
- Comparison shows VLM vs RL trade-offs

**Strong result**:
- Hybrid VLM+RL outperforms either alone
- Generalizes across cities (train Paris, test Tokyo)
- Clear insights on when to use LLM vs RL

**Exceptional result**:
- Novel hybrid architecture
- Published benchmark for vision-language navigation
- Open-source dataset and evaluation framework

---

## Connection to Existing Work

**Builds on your RL experience**:
- Similar to drone landing (vision → actions)
- Same PPO implementation (reusable)
- Reward shaping skills directly applicable

**New elements to learn**:
- Vision-language models (CLIP, GPT-4V, LLaVA)
- Large-scale image datasets
- Transfer learning and generalization
- LLM + RL integration patterns

**Potential extensions**:
- Apply learnings to visual driving environment
- Try on other vision-based decision tasks
- Explore multimodal RL more broadly

---

## Open Questions

1. **Memory**: Should policy be Markov (current image only) or recurrent (remember history)?
2. **Action space**: Discrete (4 directions) vs continuous (arbitrary heading/distance)?
3. **Multi-modal**: Include GPS coordinates or purely vision-based?
4. **Objective**: Minimize steps vs maximize exploration coverage?
5. **Real-time**: Could this work on live robot with camera?

---

## Next Steps (When Ready)

1. Choose environment (Street View API vs grid-world)
2. Implement VLM baseline (in-context navigation)
3. Collect small dataset (10-20 routes)
4. Train RL baseline (PPO with CLIP)
5. Compare and analyze failure modes
6. Iterate on most interesting findings
