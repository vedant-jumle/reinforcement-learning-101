#!/usr/bin/env python3
"""
Connect Four Self-Play with PPO

True multi-agent self-play where agents compete against themselves!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


class ConnectFourEnv:
    """Connect Four environment for two-player self-play."""

    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = None
        self.current_player = 1

    def reset(self):
        """Reset to empty board."""
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        """
        Return state from current player's perspective.
        Returns: (2, 6, 7) array - [my pieces, opponent pieces]
        """
        state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        opponent = 3 - self.current_player
        state[1] = (self.board == opponent).astype(np.float32)
        return state

    def get_valid_moves(self):
        """Get valid columns. Returns: Boolean array (7,)"""
        return self.board[0, :] == 0

    def step(self, action):
        """Execute action. Returns: (state, reward, done, info)"""
        # Invalid move check
        if not self.get_valid_moves()[action]:
            return self.get_state(), -10.0, True, {'invalid_move': True}

        # Find lowest row
        row = self._get_lowest_row(action)

        # Place piece
        self.board[row, action] = self.current_player

        # Check win
        if self._check_win(row, action):
            return self.get_state(), 1.0, True, {'winner': self.current_player}

        # Check draw
        if not self.get_valid_moves().any():
            return self.get_state(), 0.0, True, {'draw': True}

        # Switch players
        self.current_player = 3 - self.current_player
        return self.get_state(), 0.0, False, {}

    def _get_lowest_row(self, col):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        return -1

    def _check_win(self, row, col):
        player = self.board[row, col]
        # Check all 4 directions
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            if self._check_direction(row, col, dr, dc, player):
                return True
        return False

    def _check_direction(self, row, col, dr, dc, player):
        count = 1
        # Positive direction
        r, c = row + dr, col + dc
        while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
            count += 1
            r, c = r + dr, c + dc
        # Negative direction
        r, c = row - dr, col - dc
        while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
            count += 1
            r, c = r - dr, c - dc
        return count >= 4

    def render(self):
        """Print board."""
        symbols = ['.', 'X', 'O']
        print("\n  0 1 2 3 4 5 6")
        for row in range(self.rows):
            print(f"{row} ", end="")
            for col in range(self.cols):
                print(symbols[self.board[row, col]], end=" ")
            print()
        print()


class ConnectFourPolicy(nn.Module):
    """Policy network - outputs action logits."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.network(x)


class ConnectFourValue(nn.Module):
    """Value network - outputs state value."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def select_action(policy, state, valid_moves, temperature=1.0):
    """Select action using policy with action masking."""
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits = policy(state_tensor).squeeze(0)

        # Mask invalid actions
        logits = logits.cpu().numpy()
        logits[~valid_moves] = -1e9

        # Apply temperature
        logits = logits / max(temperature, 1e-8)

        # Sample
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-10)

    return action, log_prob


def play_self_play_game(policy1, policy2, env, temperature=1.0):
    """
    Play single game: policy1 vs policy2.
    Returns trajectories for both players.
    """
    # Player 1 trajectory
    states1, actions1, rewards1, log_probs1 = [], [], [], []
    # Player 2 trajectory
    states2, actions2, rewards2, log_probs2 = [], [], [], []

    state = env.reset()
    done = False

    while not done:
        valid_moves = env.get_valid_moves()

        # Current player selects action
        if env.current_player == 1:
            action, log_prob = select_action(policy1, state, valid_moves, temperature)
            states1.append(state)
            actions1.append(action)
            log_probs1.append(log_prob)
        else:
            action, log_prob = select_action(policy2, state, valid_moves, temperature)
            states2.append(state)
            actions2.append(action)
            log_probs2.append(log_prob)

        # Execute action
        next_state, reward, done, info = env.step(action)

        # Store reward for current player
        if env.current_player == 2:  # Player 1 just moved (before switch)
            rewards1.append(reward)
        else:  # Player 2 just moved
            rewards2.append(reward)

        state = next_state

    # Final reward (from winner's perspective)
    if 'winner' in info:
        winner = info['winner']
        if winner == 1:
            if len(rewards1) > 0:
                rewards1[-1] = 1.0
            if len(states2) > 0:  # Player 2 made moves
                rewards2.append(-1.0)
        else:
            if len(rewards2) > 0:
                rewards2[-1] = 1.0
            if len(states1) > 0:  # Player 1 made moves
                rewards1.append(-1.0)
    elif 'draw' in info:
        if len(rewards1) > 0:
            rewards1[-1] = 0.0
        if len(rewards2) > 0:
            rewards2[-1] = 0.0

    return (states1, actions1, rewards1, log_probs1), (states2, actions2, rewards2, log_probs2)


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def ppo_update(policy, value_net, policy_optimizer, value_optimizer,
               states, actions, old_log_probs, returns,
               clip_epsilon=0.2, num_epochs=4, batch_size=32):
    """PPO update."""
    if len(states) == 0:
        return 0, 0, 0

    # Convert to tensors
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

    # Normalize returns
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    num_updates = 0

    for epoch in range(num_epochs):
        indices = np.random.permutation(len(states))

        for start_idx in range(0, len(states), batch_size):
            end_idx = min(start_idx + batch_size, len(states))
            batch_indices = indices[start_idx:end_idx]

            batch_states = states_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]

            # Policy update
            logits = policy(batch_states)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()

            ratio = torch.exp(action_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_returns
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_returns
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            policy_optimizer.zero_grad()
            (policy_loss - 0.01 * entropy).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_optimizer.step()

            # Value update
            values = value_net(batch_states).squeeze()
            value_loss = F.mse_loss(values, batch_returns)

            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    return total_policy_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates


def main():
    print("=" * 60)
    print("CONNECT FOUR SELF-PLAY TRAINING")
    print("=" * 60)

    # Hyperparameters
    NUM_ITERATIONS = 200
    GAMES_PER_ITERATION = 10
    GAMMA = 0.99
    LEARNING_RATE = 3e-4
    OPPONENT_UPDATE_INTERVAL = 20  # Update opponent every N iterations

    # Initialize
    policy = ConnectFourPolicy().to(device)
    value_net = ConnectFourValue().to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    value_optimizer = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

    # Start with self-play against itself
    opponent = copy.deepcopy(policy)
    opponent.eval()

    env = ConnectFourEnv()

    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Games per iteration: {GAMES_PER_ITERATION}")
    print("=" * 60)
    print()

    history = []

    for iteration in tqdm(range(NUM_ITERATIONS), desc="Training"):
        all_states, all_actions, all_rewards, all_log_probs = [], [], [], []
        wins_as_p1 = 0
        wins_as_p2 = 0
        draws = 0

        # Collect games
        for game_idx in range(GAMES_PER_ITERATION):
            # Alternate who goes first
            if game_idx % 2 == 0:
                traj1, traj2 = play_self_play_game(policy, opponent, env, temperature=1.0)
            else:
                traj2, traj1 = play_self_play_game(opponent, policy, env, temperature=1.0)

            # Only train on policy's trajectory
            states, actions, rewards, log_probs = traj1 if game_idx % 2 == 0 else traj2

            if len(rewards) > 0 and rewards[-1] == 1.0:
                wins_as_p1 += 1 if game_idx % 2 == 0 else 0
                wins_as_p2 += 1 if game_idx % 2 == 1 else 0
            elif len(rewards) > 0 and rewards[-1] == 0.0:
                draws += 1

            # Compute returns
            returns = compute_returns(rewards, gamma=GAMMA)

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)

        # PPO update
        returns_all = compute_returns(all_rewards, gamma=GAMMA)
        policy_loss, value_loss, entropy = ppo_update(
            policy, value_net, policy_optimizer, value_optimizer,
            all_states, all_actions, all_log_probs, returns_all
        )

        # Update opponent periodically
        if (iteration + 1) % OPPONENT_UPDATE_INTERVAL == 0:
            opponent = copy.deepcopy(policy)
            opponent.eval()
            print(f"\n[Iteration {iteration + 1}] Opponent updated!")

        # Log
        win_rate = (wins_as_p1 + wins_as_p2) / GAMES_PER_ITERATION
        history.append({
            'iteration': iteration,
            'win_rate': win_rate,
            'wins_p1': wins_as_p1,
            'wins_p2': wins_as_p2,
            'draws': draws,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        })

        if (iteration + 1) % 10 == 0:
            print(f"\n[Iteration {iteration + 1}]")
            print(f"  Win rate: {win_rate:.2%} (P1: {wins_as_p1}, P2: {wins_as_p2}, Draws: {draws})")
            print(f"  Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}, Entropy: {entropy:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    # Save model
    torch.save(policy.state_dict(), 'connect_four_policy.pth')
    print("Model saved to connect_four_policy.pth")

    # Demo game
    print("\nDEMO: Agent vs Agent")
    env.reset()
    traj1, traj2 = play_self_play_game(policy, policy, env, temperature=0.1)
    env.render()

    return history


if __name__ == "__main__":
    history = main()
