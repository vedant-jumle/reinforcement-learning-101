# Self-Play RL: Progressive Game Complexity

## Project Overview

A progressive learning journey through self-play reinforcement learning across three games of increasing complexity. Each game builds on lessons from the previous, culminating in emergent combat strategies in a 2D fighting game.

**Core Concept**: Train agents to master games by competing against past versions of themselves, documenting the evolution of strategies and emergence of complex behaviors.

---

## Why Self-Play?

Self-play RL is how AlphaGo, AlphaZero, and OpenAI Five achieved superhuman performance. Key advantages:

1. **No human data needed** - Agent bootstraps from random play
2. **Continuous curriculum** - Past versions provide increasingly difficult opponents
3. **Emergent strategies** - Discovers novel tactics through exploration
4. **Scalable** - One agent, playing against itself
5. **Compelling narrative** - Watch AI evolve from beginner to expert

**Blog Appeal**: Self-play produces visually compelling progression stories with clear metrics (Elo ratings, win rates) and emergent behaviors that surprise even the developer.

---

## Game Progression

### Game 1: Pong (2-3 weeks)
**Complexity**: Low
**Type**: Real-time, continuous control
**Information**: Perfect (full state observable)

### Game 2: Connect Four (1-2 weeks)
**Complexity**: Medium
**Type**: Turn-based, discrete actions
**Information**: Perfect (full board visible)

### Game 3: Simple 2D Fighting Game (3-4 weeks)
**Complexity**: High
**Type**: Real-time, discrete/continuous hybrid
**Information**: Perfect (both fighters visible)

---

## Game 1: Self-Play Pong

### Environment Setup

**Use Gymnasium Built-in**:
```python
import gymnasium as gym
env = gym.make('PongNoFrameskip-v4')
```

**Observation**: 210×160×3 RGB frames (preprocess to 84×84 grayscale)
**Action Space**: 6 discrete actions (NOOP, FIRE, UP, DOWN, etc.)
**Episode Length**: Until 21 points scored

### Self-Play Architecture

**Approach 1: Fixed Opponents** (Simple, start here)
```python
# Training loop
for iteration in range(num_iterations):
    # Agent plays against frozen past version
    opponent = agent.copy()  # Freeze current agent
    opponent.eval()  # No gradient updates

    # Collect episodes: agent vs opponent
    episodes = collect_self_play_episodes(agent, opponent, num_episodes=100)

    # Train only the agent (not opponent)
    agent.train()
    update_policy(agent, episodes)

    # Save checkpoint every N iterations
    if iteration % save_interval == 0:
        save_checkpoint(agent, f'agent_iter_{iteration}.pth')
```

**Approach 2: Population-Based** (More robust, try later)
- Maintain pool of past agents (e.g., last 10 checkpoints)
- Sample opponents from pool during training
- Prevents overfitting to single opponent strategy

### Atari Preprocessing

```python
import cv2
import numpy as np

class AtariPreprocessing:
    """Standard Atari preprocessing for RL."""

    def __init__(self, frame_skip=4, frame_stack=4):
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = []

    def preprocess_frame(self, frame):
        """Convert 210×160×3 RGB → 84×84 grayscale"""
        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84×84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        normalized = resized / 255.0
        return normalized

    def stack_frames(self, frame):
        """Stack last 4 frames for temporal information"""
        self.frames.append(frame)
        if len(self.frames) > self.frame_stack:
            self.frames.pop(0)

        # Pad if not enough frames yet
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)

        # Stack along channel dimension: (4, 84, 84)
        return np.stack(self.frames, axis=0)
```

### Network Architecture

**CNN Policy Network** (similar to DQN):
```python
import torch
import torch.nn as nn

class PongPolicyNetwork(nn.Module):
    """
    CNN policy for Atari Pong.
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch, num_actions) - action probabilities
    """

    def __init__(self, num_actions=6):
        super().__init__()

        # Convolutional layers (like DQN/A3C)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (64, 7, 7)
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flat)
        return logits  # Raw logits for policy
```

**Value Network** (for Actor-Critic/PPO):
```python
class PongValueNetwork(nn.Module):
    """Critic network for value estimation."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single value output
        )

    def forward(self, x):
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        value = self.fc(flat)
        return value
```

### Training Details

**Algorithm**: PPO (reuse your existing implementation!)
**Hyperparameters** (starting point):
```python
learning_rate = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01  # Encourage exploration
value_loss_coef = 0.5
max_grad_norm = 0.5
num_steps = 128  # Steps per rollout
num_epochs = 4
batch_size = 256
```

**Opponent Update Schedule**:
- Option A: Fixed intervals (every 50 iterations)
- Option B: Performance-based (when win rate > 60% against current opponent)
- Option C: Gradual blend (linearly interpolate between old and new weights)

### Evaluation Metrics

Track these metrics over training:

1. **Elo Rating** - Standard chess-style rating system
2. **Win Rate vs Historical Opponents** - Test against checkpoints from iterations [0, 25, 50, 75, 100]
3. **Average Episode Reward** - Raw score (max = 21)
4. **Episode Length** - How long until 21 points scored
5. **Action Distribution** - Are actions diverse or converging to dominant strategy?

### Expected Progression

**Iterations 0-100**: Random flailing, occasional hits
**Iterations 100-300**: Learns to track ball, moves paddle toward it
**Iterations 300-600**: Develops basic blocking, returns some shots
**Iterations 600-1000**: Competent player, 50%+ win rate vs built-in AI
**Iterations 1000+**: Strategic play, exploiting opponent weaknesses

### Challenges & Solutions

**Challenge 1: Reward Sparsity**
- Problem: Only get +1/-1 when point scored
- Solution: Frame-skip (4 frames per action) + reward shaping (optional: small reward for ball contact)

**Challenge 2: Overfitting to Opponent**
- Problem: Agent exploits specific opponent strategy, fails against others
- Solution: Population-based opponents (sample from pool of past checkpoints)

**Challenge 3: Training Instability**
- Problem: Performance oscillates, catastrophic forgetting
- Solution: Conservative updates (small learning rate, clip_epsilon=0.2), trust region methods (PPO)

**Challenge 4: Stagnation**
- Problem: Agent and opponent reach local equilibrium, no improvement
- Solution: Inject diversity (increase entropy bonus temporarily, add noise to observations)

### Blog Post Structure

**Title**: "Teaching AI to Play Pong Against Itself: A Self-Play Journey"

**Sections**:
1. **Introduction** - What is self-play RL? (AlphaGo example)
2. **Environment Setup** - Gymnasium, preprocessing, why Pong?
3. **Algorithm** - PPO + self-play loop (with code snippets)
4. **Training Progression** - GIFs/videos showing iterations [0, 100, 300, 600, 1000]
5. **Emergent Strategies** - Did agent discover interesting tactics?
6. **Evaluation** - Elo ratings, win rate curves, action distributions
7. **Lessons Learned** - What worked, what didn't, hyperparameter sensitivity
8. **Next Steps** - Preview Connect Four (turn-based self-play)

---

## Game 2: Connect Four

### Why Connect Four?

**Advantages**:
- Simple rules, easy to implement from scratch
- Turn-based (no real-time complexity)
- Discrete action space (7 columns)
- Small state space (6×7 board = 42 positions)
- Fast training (games complete in < 42 moves)
- Perfect information (no hidden state)

**Progression from Pong**:
- Pong: Real-time continuous → Connect Four: Turn-based discrete
- Tests if self-play strategies transfer to different game types

### Environment Implementation

**State Representation**:
```python
# Option 1: Raw board (simple)
state = board  # (6, 7) array, values in {0, 1, 2} for {empty, player1, player2}

# Option 2: Bit boards (efficient)
# Two (6, 7) binary arrays: one for each player

# Option 3: CNN-friendly (if using conv layers)
state = np.stack([board == 1, board == 2], axis=0)  # (2, 6, 7)
```

**Action Space**: 7 discrete actions (drop piece in column 0-6)

**Reward Structure**:
```python
if win:
    reward = +1.0
elif draw:
    reward = 0.0
elif invalid_move:
    reward = -1.0  # Tried to drop in full column
else:
    reward = 0.0  # Continue playing
```

### Network Architecture

**Option 1: MLP** (Start here - simpler)
```python
class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 42 values (6×7 flattened board)
        self.network = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 7 columns
        )

    def forward(self, x):
        # x: (batch, 6, 7)
        flat = x.view(x.size(0), -1)  # (batch, 42)
        logits = self.network(flat)
        return logits
```

**Option 2: CNN** (More sophisticated)
```python
class ConnectFourCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (2, 6, 7) - two channels for two players
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flat)
        return logits
```

### Self-Play Training Loop

```python
class ConnectFourSelfPlay:
    """Self-play trainer for Connect Four."""

    def __init__(self, policy_network, value_network):
        self.policy = policy_network
        self.value = value_network
        self.opponent = None  # Frozen copy

    def play_game(self):
        """
        Play single game: agent vs opponent.
        Returns trajectory for training.
        """
        board = np.zeros((6, 7))
        trajectory = []  # [(state, action, reward, next_state, done)]

        current_player = 1  # Agent is player 1
        done = False

        while not done:
            state = board.copy()

            # Select action (agent or opponent)
            if current_player == 1:
                action = self.select_action(state, network=self.policy)
            else:
                action = self.select_action(state, network=self.opponent)

            # Execute action
            next_state, reward, done = self.step(board, action, current_player)

            # Store transition (only for agent, player 1)
            if current_player == 1:
                trajectory.append((state, action, reward, next_state, done))

            # Switch players
            current_player = 3 - current_player  # Toggle between 1 and 2
            board = next_state

        return trajectory

    def select_action(self, state, network):
        """Select action using policy network."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = network(state_tensor)

            # Mask invalid actions (full columns)
            valid_mask = self.get_valid_actions(state)
            logits[~valid_mask] = -1e9

            # Sample from softmax distribution
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

        return action

    def step(self, board, action, player):
        """Execute action on board."""
        # Find lowest empty row in column
        row = self.get_lowest_empty_row(board, action)

        if row is None:
            # Invalid move (column full)
            return board, -1.0, True

        # Place piece
        new_board = board.copy()
        new_board[row, action] = player

        # Check win condition
        if self.check_win(new_board, player):
            reward = +1.0 if player == 1 else -1.0
            done = True
        elif self.is_board_full(new_board):
            reward = 0.0  # Draw
            done = True
        else:
            reward = 0.0
            done = False

        return new_board, reward, done

    def update_opponent(self):
        """Freeze current agent as new opponent."""
        self.opponent = copy.deepcopy(self.policy)
        self.opponent.eval()  # Set to evaluation mode
```

### Training Strategy

**Curriculum**:
1. **Phase 1 (Iterations 0-100)**: Agent vs random opponent
2. **Phase 2 (Iterations 100-300)**: Agent vs self (frozen every 20 iterations)
3. **Phase 3 (Iterations 300+)**: Agent vs population (sample from pool of past 10 checkpoints)

**When to Update Opponent**:
- Fixed schedule: Every 20-50 iterations
- Performance-based: When win rate > 70% against current opponent
- Adaptive: More frequent updates early, less frequent later

### Advanced: Monte Carlo Tree Search (MCTS)

For stronger play (optional, AlphaGo-style):

```python
class MCTSNode:
    """Node in MCTS game tree."""

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action → child node
        self.visits = 0
        self.value = 0.0

    def uct_score(self, c=1.41):
        """Upper confidence bound for trees."""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = c * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        """Select child with highest UCT score."""
        return max(self.children.values(), key=lambda node: node.uct_score())

def mcts_search(root_state, policy_network, value_network, num_simulations=100):
    """
    Run MCTS to select best action.
    Combines neural network guidance with tree search.
    """
    root = MCTSNode(root_state)

    for _ in range(num_simulations):
        node = root

        # 1. Selection: Traverse tree using UCT
        while node.children and not is_terminal(node.state):
            node = node.select_child()

        # 2. Expansion: Add children for unexplored actions
        if not is_terminal(node.state) and node.visits > 0:
            for action in get_valid_actions(node.state):
                child_state = simulate_action(node.state, action)
                node.children[action] = MCTSNode(child_state, parent=node)

        # 3. Evaluation: Use value network
        value = value_network(node.state)

        # 4. Backpropagation: Update path to root
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
            value = -value  # Flip sign for opponent

    # Select action with most visits
    best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
    return best_action
```

**Note**: MCTS is powerful but complex. Start with pure neural network policy, add MCTS later if you want AlphaGo-style performance.

### Expected Results

**Iterations 0-50**: Random play, no strategy
**Iterations 50-150**: Learns to complete obvious 3-in-a-row
**Iterations 150-300**: Basic blocking, prevents opponent wins
**Iterations 300-500**: Strategic positioning, sets up multiple threats
**Iterations 500+**: Near-optimal play, rarely loses

### Blog Post Structure

**Title**: "Connect Four Self-Play: From Random Drops to Strategic Mastery"

**Sections**:
1. **From Pong to Connect Four** - Why turn-based games?
2. **Environment Design** - Board representation, action masking
3. **Self-Play Evolution** - Progression GIFs/diagrams
4. **Strategy Emergence** - What tactics did agent discover? (center control, double threats)
5. **MCTS Extension** (optional) - Combining neural nets with tree search
6. **Comparison to Pong** - Self-play learnings across game types

---

## Game 3: Simple 2D Fighting Game

### Why a Fighting Game?

**Advantages**:
- Real-time + discrete actions (combines Pong and Connect Four lessons)
- Rich strategy space (spacing, combos, mixups)
- Emergent behaviors are visually compelling
- Tests generalization (more complex than previous games)

**Challenges**:
- Requires game implementation (but can be minimal!)
- Longer training time
- More hyperparameter tuning

### Game Design (Keep It Minimal!)

**Goal**: Create simplest possible fighting game that still has strategic depth.

**Core Mechanics**:
- 2D side-view (like Street Fighter)
- Two characters face each other
- Each has HP (100), stamina (100)
- Actions: Move left/right, jump, block, 3 attacks (light/medium/heavy)
- Win condition: Reduce opponent HP to 0

**State Space** (19 dimensions):
```python
{
    # Player 1 (self)
    'p1_x': float,          # Position (0-800)
    'p1_y': float,          # Height (0 = ground)
    'p1_vx': float,         # Velocity X
    'p1_vy': float,         # Velocity Y
    'p1_hp': float,         # Health (0-100)
    'p1_stamina': float,    # Stamina (0-100)
    'p1_state': int,        # State: idle/attacking/blocking/hitstun
    'p1_facing': int,       # Direction: -1 (left) or +1 (right)

    # Player 2 (opponent)
    'p2_x': float,
    'p2_y': float,
    'p2_vx': float,
    'p2_vy': float,
    'p2_hp': float,
    'p2_stamina': float,
    'p2_state': int,
    'p2_facing': int,

    # Relative info
    'distance': float,      # Distance between fighters
    'hp_diff': float,       # p1_hp - p2_hp
    'stamina_diff': float   # p1_stamina - p2_stamina
}
```

**Action Space** (8 discrete actions):
```python
{
    0: 'idle',          # Do nothing
    1: 'move_left',     # Move backward
    2: 'move_right',    # Move forward
    3: 'jump',          # Jump
    4: 'block',         # Defensive stance
    5: 'light_attack',  # Fast, low damage (5 HP, 10 stamina)
    6: 'medium_attack', # Medium speed/damage (12 HP, 20 stamina)
    7: 'heavy_attack'   # Slow, high damage (25 HP, 35 stamina)
}
```

**Combat Mechanics** (Rock-Paper-Scissors):
- **Attacks beat Idle/Movement** (deal damage)
- **Blocks beat Attacks** (reduce damage by 80%, no stamina cost to blocker)
- **Movement/Grabs beat Blocks** (can't block while moving)

**Frame Data** (simplified):
```python
ATTACK_FRAMES = {
    'light': {
        'startup': 3,      # Frames before hitbox active
        'active': 2,       # Frames hitbox is active
        'recovery': 5,     # Frames before can act again
        'damage': 5,
        'stamina': 10
    },
    'medium': {
        'startup': 6,
        'active': 3,
        'recovery': 8,
        'damage': 12,
        'stamina': 20
    },
    'heavy': {
        'startup': 12,
        'active': 4,
        'recovery': 15,
        'damage': 25,
        'stamina': 35
    }
}
```

### Environment Implementation

**Don't build from scratch!** Use existing physics engines:

**Option 1: Pygame + Simple Physics** (Recommended)
```python
import pygame
import numpy as np

class FightingGameEnv:
    """Minimal 2D fighting game for RL."""

    def __init__(self, render_mode='rgb_array'):
        self.screen_width = 800
        self.screen_height = 600
        self.ground_y = 500
        self.render_mode = render_mode

        # Initialize fighters
        self.reset()

    def reset(self):
        """Reset to starting positions."""
        self.p1 = Fighter(x=200, y=self.ground_y, facing=1)
        self.p2 = Fighter(x=600, y=self.ground_y, facing=-1)
        self.frame = 0
        return self.get_state()

    def step(self, action_p1, action_p2):
        """
        Execute one frame of gameplay.

        Args:
            action_p1: Action for player 1 (0-7)
            action_p2: Action for player 2 (0-7)

        Returns:
            state, reward_p1, reward_p2, done
        """
        # Update both fighters
        self.p1.update(action_p1)
        self.p2.update(action_p2)

        # Apply physics (gravity, movement)
        self.p1.apply_physics(self.ground_y)
        self.p2.apply_physics(self.ground_y)

        # Check collisions (attacks hitting)
        reward_p1, reward_p2 = self.resolve_combat()

        # Check win condition
        done = self.p1.hp <= 0 or self.p2.hp <= 0
        if done:
            if self.p1.hp > self.p2.hp:
                reward_p1 += 100  # Win bonus
                reward_p2 -= 100
            else:
                reward_p1 -= 100
                reward_p2 += 100

        self.frame += 1

        # Timeout after 1800 frames (30 seconds at 60 FPS)
        if self.frame >= 1800:
            done = True
            # Reward based on HP remaining
            reward_p1 += (self.p1.hp - self.p2.hp) * 0.1
            reward_p2 += (self.p2.hp - self.p1.hp) * 0.1

        return self.get_state(), reward_p1, reward_p2, done

    def resolve_combat(self):
        """Check if attacks hit, apply damage."""
        reward_p1 = 0.0
        reward_p2 = 0.0

        # Check if fighters are in range
        distance = abs(self.p1.x - self.p2.x)
        in_range = distance < 100

        if not in_range:
            return reward_p1, reward_p2

        # P1 attack hits P2
        if self.p1.is_attacking() and not self.p2.is_blocking():
            damage = self.p1.get_attack_damage()
            self.p2.take_damage(damage)
            self.p2.enter_hitstun()
            reward_p1 += damage * 0.5  # Reward for landing hit
            reward_p2 -= damage * 0.5

        # P2 attack hits P1
        if self.p2.is_attacking() and not self.p1.is_blocking():
            damage = self.p2.get_attack_damage()
            self.p1.take_damage(damage)
            self.p1.enter_hitstun()
            reward_p2 += damage * 0.5
            reward_p1 -= damage * 0.5

        # Successful blocks
        if self.p1.is_attacking() and self.p2.is_blocking():
            reward_p2 += 1.0  # Small reward for successful block
        if self.p2.is_attacking() and self.p1.is_blocking():
            reward_p1 += 1.0

        return reward_p1, reward_p2

    def get_state(self):
        """Return observation for both players."""
        state = np.array([
            # Player 1
            self.p1.x / self.screen_width,
            self.p1.y / self.screen_height,
            self.p1.vx / 10.0,
            self.p1.vy / 10.0,
            self.p1.hp / 100.0,
            self.p1.stamina / 100.0,
            self.p1.state,
            self.p1.facing,

            # Player 2
            self.p2.x / self.screen_width,
            self.p2.y / self.screen_height,
            self.p2.vx / 10.0,
            self.p2.vy / 10.0,
            self.p2.hp / 100.0,
            self.p2.stamina / 100.0,
            self.p2.state,
            self.p2.facing,

            # Relative
            distance / self.screen_width,
            (self.p1.hp - self.p2.hp) / 100.0,
            (self.p1.stamina - self.p2.stamina) / 100.0
        ])
        return state

class Fighter:
    """Fighter character with state machine."""

    STATES = {
        'idle': 0,
        'moving': 1,
        'jumping': 2,
        'attacking': 3,
        'blocking': 4,
        'hitstun': 5
    }

    def __init__(self, x, y, facing):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.facing = facing
        self.hp = 100
        self.stamina = 100
        self.state = 0  # idle
        self.state_timer = 0
        self.current_attack = None

    def update(self, action):
        """Update fighter based on action."""
        # Regenerate stamina
        self.stamina = min(100, self.stamina + 0.2)

        # Can't act during hitstun or attack recovery
        if self.state in [3, 5]:  # attacking or hitstun
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.state = 0  # Return to idle
            return

        # Execute action
        if action == 0:  # idle
            self.state = 0
            self.vx = 0

        elif action == 1:  # move_left
            self.state = 1
            self.vx = -5 * self.facing

        elif action == 2:  # move_right
            self.state = 1
            self.vx = 5 * self.facing

        elif action == 3:  # jump
            if self.y == 500:  # On ground
                self.state = 2
                self.vy = -15

        elif action == 4:  # block
            self.state = 4
            self.vx = 0

        elif action in [5, 6, 7]:  # attacks
            attack_type = ['light', 'medium', 'heavy'][action - 5]
            attack_data = ATTACK_FRAMES[attack_type]

            # Check stamina
            if self.stamina >= attack_data['stamina']:
                self.state = 3  # attacking
                self.current_attack = attack_type
                self.state_timer = attack_data['startup'] + attack_data['active'] + attack_data['recovery']
                self.stamina -= attack_data['stamina']
                self.vx = 0

    def apply_physics(self, ground_y):
        """Apply gravity and movement."""
        # Gravity
        if self.y < ground_y:
            self.vy += 0.8  # Gravity

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Ground collision
        if self.y >= ground_y:
            self.y = ground_y
            self.vy = 0
            if self.state == 2:  # Landing from jump
                self.state = 0

        # Screen bounds
        self.x = max(50, min(750, self.x))

    def is_attacking(self):
        return self.state == 3 and self.state_timer > 0

    def is_blocking(self):
        return self.state == 4

    def get_attack_damage(self):
        if self.current_attack:
            return ATTACK_FRAMES[self.current_attack]['damage']
        return 0

    def take_damage(self, damage):
        self.hp = max(0, self.hp - damage)

    def enter_hitstun(self):
        """Enter hitstun state (can't act)."""
        self.state = 5
        self.state_timer = 10  # 10 frames of hitstun
```

**Option 2: Use Existing Engine** (If you want to skip implementation)
- **PettingZoo** - Multi-agent RL environments (has some fighting games)
- **Slime Volleyball** - Simple 2D physics fighting game
- **Custom Smash Bros Clone** - Several open-source implementations available

### Network Architecture

**Policy Network**:
```python
class FighterPolicy(nn.Module):
    """Policy network for fighting game."""

    def __init__(self, state_dim=19, action_dim=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        logits = self.network(state)
        return logits
```

**Value Network** (similar structure, output = 1)

### Self-Play Training

**Key Differences from Pong/Connect Four**:
1. **Simultaneous actions** - Both players act every frame (not turn-based)
2. **Longer episodes** - 1800 frames (30 seconds) vs Pong's shorter episodes
3. **Richer reward shaping** - Damage dealt, successful blocks, spacing, etc.

**Training Loop**:
```python
for iteration in range(num_iterations):
    # Sample opponent from population
    opponent = sample_from_population(past_checkpoints)

    # Collect episodes (agent vs opponent)
    episodes = []
    for _ in range(num_episodes_per_iteration):
        states, actions, rewards = play_episode(agent, opponent)
        episodes.append((states, actions, rewards))

    # Train with PPO
    update_policy(agent, episodes)

    # Update population every N iterations
    if iteration % 20 == 0:
        past_checkpoints.append(agent.copy())
        if len(past_checkpoints) > 10:
            past_checkpoints.pop(0)  # Keep last 10
```

### Expected Emergent Strategies

If training succeeds, watch for these behaviors:

1. **Spacing** - Agent learns optimal distance (close enough to hit, far enough to react)
2. **Whiff punishing** - Waits for opponent to miss, then attacks during recovery
3. **Frame traps** - Uses attacks that leave opponent in disadvantage
4. **Mixups** - Alternates between high/low attacks to keep opponent guessing
5. **Stamina management** - Doesn't spam heavy attacks when low stamina
6. **Defensive patterns** - Blocks when low HP, aggressive when ahead

**Warning**: These strategies may NOT emerge if:
- Training time insufficient (may need 50K+ iterations)
- Reward shaping too sparse/dense
- Hyperparameters not tuned
- Environment too complex for agent to learn

### Reward Shaping (Critical!)

**Sparse vs Dense Rewards**:

**Sparse** (start here):
```python
reward = 0.0
if hit_landed:
    reward += damage * 0.5
if took_damage:
    reward -= damage * 0.5
if won:
    reward += 100.0
if lost:
    reward -= 100.0
```

**Dense** (if sparse doesn't work):
```python
reward = 0.0

# Combat rewards
if hit_landed:
    reward += damage * 0.5
if successful_block:
    reward += 2.0
if whiff_punish:
    reward += 5.0  # Bonus for punishing opponent's miss

# Spacing rewards (encourage optimal distance)
optimal_distance = 80  # Pixels
distance_error = abs(distance - optimal_distance)
reward -= distance_error * 0.01

# Stamina management
if low_stamina and heavy_attack:
    reward -= 1.0  # Penalty for poor stamina management

# HP advantage
reward += (agent_hp - opponent_hp) * 0.01
```

**Experimentation Required**: You'll need to tune these weights through trial and error!

### Training Time Estimate

**Baseline** (with well-tuned hyperparameters):
- **Iterations 0-1000**: Random flailing, occasional hits
- **Iterations 1000-5000**: Learns to land attacks consistently
- **Iterations 5000-10000**: Basic spacing, blocking
- **Iterations 10000-20000**: Strategic play emerges
- **Iterations 20000+**: Complex combos, mixups (if environment supports)

**GPU Requirements**:
- Training: RTX 3070 or better (PPO + parallel environments)
- Expect 2-4 weeks of training time with good GPU

### Evaluation Metrics

1. **Elo Ratings** - Track relative strength over time
2. **Win Rate vs Historical Opponents**
3. **Average Damage Dealt per Episode**
4. **Hit Accuracy** (attacks landed / attacks thrown)
5. **Block Success Rate** (successful blocks / opponent attacks)
6. **Action Distribution** - Are all actions used or dominated by one?
7. **Strategy Diversity** - Do different agents discover different playstyles?

### Visualization

**Critical for Blog Appeal**:
- Record GIFs of gameplay at iterations [0, 1000, 5000, 10000, 20000]
- Heatmaps of positioning (where do agents spend time?)
- Action distribution pie charts
- Elo progression curves
- Highlight reels of "sick combos" or unexpected strategies

### Blog Post Structure

**Title**: "Emergent Combat: Teaching AI to Fight Through Self-Play"

**Sections**:
1. **Journey So Far** - Recap Pong + Connect Four learnings
2. **Why Fighting Games?** - Complexity, real-time decisions, visual appeal
3. **Game Design** - Minimal viable fighting game, mechanics
4. **Self-Play at Scale** - Population-based training, opponent sampling
5. **Emergent Strategies** - What behaviors emerged? (spacing, whiff punishing, mixups)
6. **Failures & Debugging** - What didn't work? Reward shaping iterations
7. **Comparison Across Three Games** - Self-play lessons learned
8. **Future Directions** - More complex games, multi-agent cooperation

---

## Implementation Timeline

### Overall Project: 8-12 weeks

**Weeks 1-3: Pong Self-Play**
- Week 1: Environment setup, PPO implementation, baseline vs built-in AI
- Week 2: Self-play loop, opponent population, training
- Week 3: Evaluation, visualization, blog post writing

**Weeks 4-5: Connect Four Self-Play**
- Week 4: Environment implementation, MLP policy, self-play training
- Week 5: MCTS experiments (optional), blog post writing

**Weeks 6-12: Fighting Game Self-Play**
- Week 6-7: Game implementation (or adapt existing)
- Week 8-9: Self-play training (slow, iterative)
- Week 10-11: Reward shaping tuning, extended training
- Week 12: Evaluation, visualization, final blog post

---

## Technical Stack

**Core Libraries**:
```python
# RL
torch                 # Neural networks, PPO
gymnasium             # Atari environments (Pong)
numpy                 # Numerical computing

# Game Development
pygame                # 2D game rendering (Connect Four, Fighting Game)

# Visualization
matplotlib            # Plots, charts
imageio               # GIF creation
opencv-python         # Video processing

# Utilities
tqdm                  # Progress bars
wandb                 # Experiment tracking (optional but recommended)
```

**Hardware Requirements**:
- **GPU**: RTX 3060 or better (12GB+ VRAM)
- **CPU**: 8+ cores for parallel environment sampling
- **RAM**: 16GB+
- **Storage**: 50GB for checkpoints, videos, logs

---

## Success Criteria

### Minimum Viable Results (Must Achieve)

**Pong**:
- [ ] Agent learns to track ball and move paddle
- [ ] Win rate > 50% vs built-in AI after training
- [ ] Clear Elo progression over training iterations

**Connect Four**:
- [ ] Agent learns to complete 3-in-a-row
- [ ] Agent learns basic blocking (prevents opponent wins)
- [ ] Win rate > 70% vs random opponent

**Fighting Game**:
- [ ] Agent learns to land attacks consistently (hit accuracy > 40%)
- [ ] Agent uses blocking defensively
- [ ] Beats random opponent > 80% of the time

### Target Results (Strong Success)

**Pong**:
- [ ] Discovers strategic positioning (not just reactive)
- [ ] Generalizes to unseen opponent strategies
- [ ] Elo rating increases monotonically

**Connect Four**:
- [ ] Discovers center control strategy
- [ ] Sets up multiple threats (forcing wins)
- [ ] Near-optimal play (hard for humans to beat)

**Fighting Game**:
- [ ] Emergent spacing behavior (maintains optimal distance)
- [ ] Whiff punishing (exploits opponent recovery frames)
- [ ] Diverse action usage (not dominated by single strategy)

### Stretch Goals (Exceptional)

**Pong**:
- [ ] Add MCTS for superhuman performance
- [ ] Train on multiple Atari games, transfer learn

**Connect Four**:
- [ ] AlphaZero-style MCTS + neural nets
- [ ] Solve game theoretically (provably optimal)

**Fighting Game**:
- [ ] Complex combos emerge (multi-hit sequences)
- [ ] Mixups and mind games (unpredictable patterns)
- [ ] Transfer to more complex fighting game (e.g., PettingZoo environments)

---

## Common Pitfalls & Solutions

### Pitfall 1: Self-Play Collapse
**Problem**: Agent and opponent both converge to trivial strategy (e.g., both always block)

**Solutions**:
- Increase entropy bonus (encourage exploration)
- Population-based training (diverse opponents)
- Inject random opponents periodically
- Reward shaping to encourage aggressive play

### Pitfall 2: Overfitting to Opponent
**Problem**: Agent exploits specific opponent weakness, fails against others

**Solutions**:
- Opponent population (sample from past 10 checkpoints)
- Evaluation against diverse opponents
- Regularization (weight decay, dropout)

### Pitfall 3: Training Instability
**Problem**: Performance oscillates wildly, catastrophic forgetting

**Solutions**:
- Conservative PPO (small clip_epsilon = 0.1-0.2)
- Gradient clipping (max_norm = 0.5)
- Smaller learning rate (1e-4 to 5e-4)
- Trust region methods (TRPO if PPO too unstable)

### Pitfall 4: Reward Hacking
**Problem**: Agent exploits reward function (e.g., runs away to avoid damage)

**Solutions**:
- Careful reward shaping (test thoroughly)
- Time limits (force engagement)
- Distance penalties (can't run too far)
- Iterative refinement based on observed behavior

### Pitfall 5: No Emergent Strategies
**Problem**: Agent learns basics but no advanced tactics

**Solutions**:
- Longer training (may need 50K+ iterations)
- Curriculum learning (gradually increase difficulty)
- Reward shaping for desired behaviors
- Simplify environment (reduce complexity until learning works)

---

## Learning Outcomes

By completing this project, you will:

1. **Master self-play RL** - Core technique behind AlphaGo, OpenAI Five, AlphaStar
2. **Understand opponent modeling** - How agents adapt to adversaries
3. **Practice reward engineering** - Critical RL skill across multiple domains
4. **Gain intuition for emergence** - When/why complex behaviors arise
5. **Build portfolio project** - Visually compelling, technically impressive
6. **Create blog series** - Three posts documenting journey, strategy evolution

---

## Blog Series Outline

### Post 1: "Teaching AI to Play Pong Against Itself"
- Introduction to self-play RL
- Atari preprocessing, CNN policies
- Training progression (GIFs)
- Strategy evolution
- Elo ratings and evaluation

### Post 2: "Connect Four Self-Play: Turn-Based Strategy Emergence"
- From real-time to turn-based
- MLP vs CNN architectures
- Win condition checking, action masking
- Strategic behaviors (center control, double threats)
- Optional: MCTS extension

### Post 3: "Emergent Combat: AI Learning to Fight Through Competition"
- Game design (minimal fighting game)
- Simultaneous actions, frame data
- Reward shaping challenges
- Emergent strategies (spacing, mixups, combos)
- Lessons learned across three games

### Post 4 (Meta): "Self-Play RL: Lessons from Pong to Fighting Games"
- Comparison across all three games
- When self-play works (and when it doesn't)
- Practical tips for practitioners
- Future directions (more complex games, multi-agent cooperation)

---

## Future Extensions (Beyond Core Project)

### Extension 1: Multi-Agent Cooperation
- 2v2 fighting game (team battles)
- Requires coordination, not just competition
- Communication protocols between agents

### Extension 2: Imitation Learning + Self-Play
- Collect human demonstrations (play manually)
- Bootstrap agent with behavior cloning
- Fine-tune with self-play
- Compare sample efficiency

### Extension 3: Opponent Modeling
- Agent explicitly models opponent strategy
- Predicts opponent's next action
- Exploits opponent weaknesses
- More advanced than pure self-play

### Extension 4: Transfer Learning
- Train on simple fighting game
- Transfer to complex game (e.g., real Street Fighter)
- Test generalization across game mechanics

### Extension 5: Curriculum Learning
- Gradually increase game complexity
- Start: Single attack type → Multiple attacks → Blocking → Full game
- Measure impact on training speed

---

## Resources & References

### Self-Play RL Papers
1. **AlphaGo** (Silver et al., 2016) - Original self-play + MCTS
2. **AlphaZero** (Silver et al., 2017) - Pure self-play, no human data
3. **OpenAI Five** (OpenAI, 2019) - Dota 2 self-play at scale
4. **AlphaStar** (Vinyals et al., 2019) - StarCraft II self-play

### Fighting Game AI
1. **DareFightingICE** - Annual fighting game AI competition
2. **Fighting Game AI Competition** - Research community, datasets
3. **FightingICE** - Java-based fighting game for AI research

### Code References
1. **Stable-Baselines3** - PPO implementation
2. **CleanRL** - Single-file RL implementations
3. **PettingZoo** - Multi-agent RL environments
4. **OpenSpiel** - Board game RL environments (includes Connect Four)

### Tutorials
1. **Spinning Up in Deep RL** (OpenAI) - RL fundamentals
2. **Deep RL Course** (HuggingFace) - Practical RL implementations
3. **Atari Preprocessing Guide** - Standard techniques for Atari games

---

## Repository Structure (Proposed)

```
reinforcement-learning-101/
├── self_play_progression/
│   ├── 1_pong/
│   │   ├── train_pong.py
│   │   ├── pong_policy.py
│   │   ├── atari_preprocessing.py
│   │   ├── self_play_trainer.py
│   │   ├── evaluate.py
│   │   └── checkpoints/
│   │       ├── agent_iter_0.pth
│   │       ├── agent_iter_100.pth
│   │       └── ...
│   │
│   ├── 2_connect_four/
│   │   ├── connect_four_env.py
│   │   ├── train_connect_four.py
│   │   ├── connect_four_policy.py
│   │   ├── mcts.py (optional)
│   │   ├── evaluate.py
│   │   └── checkpoints/
│   │
│   ├── 3_fighting_game/
│   │   ├── fighting_game_env.py
│   │   ├── fighter.py
│   │   ├── train_fighting.py
│   │   ├── fighter_policy.py
│   │   ├── evaluate.py
│   │   ├── manual_play.py
│   │   └── checkpoints/
│   │
│   ├── shared/
│   │   ├── ppo.py                  # PPO implementation (reused!)
│   │   ├── self_play_utils.py     # Common self-play utilities
│   │   ├── elo_rating.py          # Elo rating system
│   │   ├── visualization.py       # Plotting, GIF creation
│   │   └── evaluation.py          # Standard evaluation metrics
│   │
│   └── notebooks/
│       ├── pong_analysis.ipynb
│       ├── connect_four_analysis.ipynb
│       └── fighting_game_analysis.ipynb
│
├── blogs/
│   └── self_play_series/
│       ├── part1_pong.md
│       ├── part2_connect_four.md
│       ├── part3_fighting_game.md
│       ├── part4_lessons_learned.md
│       └── media/
│           ├── pong_progression.gif
│           ├── connect_four_heatmap.png
│           ├── fighting_game_combo.gif
│           └── elo_curves.png
│
└── future_plans/
    └── SELF_PLAY_PROGRESSION.md (this file)
```

---

## Next Steps (When Ready to Start)

### Immediate (Week 1)
1. **Set up Pong environment**
   ```bash
   pip install gymnasium[atari]
   pip install ale-py
   gymnasium.make('PongNoFrameskip-v4')
   ```

2. **Implement Atari preprocessing**
   - Frame stacking (4 frames)
   - Grayscale conversion
   - Resize to 84×84

3. **Test PPO on Pong vs built-in AI**
   - Establish baseline performance
   - Validate training pipeline

### Week 2
4. **Implement self-play loop**
   - Opponent freezing/copying
   - Episode collection with two agents
   - Training only the main agent

5. **Add Elo rating system**
   - Track relative strength over time
   - Evaluate against historical checkpoints

### Week 3
6. **Train for 1000+ iterations**
   - Monitor for convergence
   - Watch for emergent strategies
   - Save checkpoints every 50 iterations

7. **Visualize results**
   - Create GIFs of progression
   - Plot Elo curves
   - Analyze action distributions

8. **Write blog post**
   - Document journey, challenges, results
   - Include visuals, code snippets

---

## Open Questions to Resolve

1. **Opponent Update Frequency**: Fixed schedule vs performance-based vs adaptive?
2. **Population Size**: How many past checkpoints to maintain? (5? 10? 20?)
3. **MCTS Priority**: Worth implementing for Connect Four or skip to focus on fighting game?
4. **Fighting Game Complexity**: Minimal viable (8 actions) vs richer (combos, specials)?
5. **Reward Shaping**: Start sparse or dense? How much hand-tuning is acceptable?
6. **Training Budget**: How many GPU-hours are you willing to invest? (Affects scope)
7. **Blog Frequency**: One post after all three games vs continuous posting?

---

## Conclusion

This project provides a **structured path from simple (Pong) to complex (Fighting Game) self-play RL**, building skills progressively while creating compelling blog content at each stage.

**Key Success Factors**:
1. ✅ **Reuse PPO implementation** - Core algorithm stays the same
2. ✅ **Incremental complexity** - Each game builds on previous lessons
3. ✅ **Visual appeal** - All three games produce great GIFs/videos
4. ✅ **Clear metrics** - Elo ratings, win rates, strategy emergence
5. ✅ **Blog-worthy** - Emergent behavior stories resonate with readers

**Estimated Total Time**: 8-12 weeks (3 weeks per game + blog writing)

**Expected Outcome**: A comprehensive blog series documenting self-play RL across three game types, with trained models and evaluation results. Perfect portfolio piece demonstrating mastery of adversarial RL and emergent strategy development.

Ready to start with Pong when you are! 🎮🤖
