#!/usr/bin/env python3
"""Random agent - demonstrates the game API with random actions"""

import sys
import random
sys.path.insert(0, '..')

from game.game_engine import DroneGame


def main():
    """Run random agent for multiple episodes"""
    # Create game
    game = DroneGame(render_mode='human')

    num_episodes = 5
    print(f"Running {num_episodes} episodes with random actions...\n")

    for episode in range(num_episodes):
        # Reset game
        state = game.reset()

        episode_reward = 0
        step = 0

        while True:
            # Random action (simple strategy: sometimes thrust)
            action = {
                'main_thrust': random.choice([0, 1]),
                'left_thrust': random.choice([0, 0, 0, 1]),  # Less frequent
                'right_thrust': random.choice([0, 0, 0, 1])  # Less frequent
            }

            # Step
            state, reward, done, info = game.step(action)
            episode_reward += reward
            step += 1

            # Render
            game.render()

            if done:
                status = "SUCCESS" if game.drone.landed else "CRASHED"
                print(f"Episode {episode + 1}: {status} | Steps: {step} | Reward: {episode_reward:.2f}")
                break

    game.close()
    print("\nRandom agent demo complete!")


if __name__ == "__main__":
    main()
