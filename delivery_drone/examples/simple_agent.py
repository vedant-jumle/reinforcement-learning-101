#!/usr/bin/env python3
"""Simple rule-based agent - demonstrates using state information for decisions"""

import sys
sys.path.insert(0, '..')

from game.game_engine import DroneGame


def simple_policy(state):
    """Simple rule-based policy using state information

    Args:
        state: State dictionary from game

    Returns:
        action dictionary
    """
    action = {
        'main_thrust': 0,
        'left_thrust': 0,
        'right_thrust': 0
    }

    # Extract useful state info
    drone_vy = state['drone_vy']  # Vertical velocity (positive = down)
    drone_angle = state['drone_angle']  # Angle in normalized form
    dx = state['dx_to_platform']  # Horizontal distance to platform
    dy = state['dy_to_platform']  # Vertical distance to platform
    speed = state['speed']

    # Strategy:
    # 1. Stabilize angle first
    # 2. Counteract falling
    # 3. Navigate toward platform

    # Stabilize angle
    if drone_angle > 0.1:  # Tilted right, thrust left
        action['left_thrust'] = 1
    elif drone_angle < -0.1:  # Tilted left, thrust right
        action['right_thrust'] = 1

    # Counteract falling if moving down too fast
    if drone_vy > 0.3:  # Falling too fast
        action['main_thrust'] = 1
    # Or if above platform and falling slowly, give some thrust
    elif dy < 0 and drone_vy > 0.1:
        action['main_thrust'] = 1

    # If we're close and moving slow, reduce thrust
    if dy > -0.1 and speed < 0.3:  # Close to platform
        action['main_thrust'] = 0

    return action


def main():
    """Run simple rule-based agent"""
    game = DroneGame(render_mode='human')

    num_episodes = 10
    successes = 0

    print(f"Running {num_episodes} episodes with simple rule-based agent...\n")

    for episode in range(num_episodes):
        state = game.reset()
        episode_reward = 0
        step = 0

        while True:
            # Get action from simple policy
            action = simple_policy(state)

            # Step
            state, reward, done, info = game.step(action)
            episode_reward += reward
            step += 1

            # Render
            game.render()

            if done:
                if game.drone.landed:
                    status = "SUCCESS ✓"
                    successes += 1
                else:
                    status = "CRASHED ✗"

                print(f"Episode {episode + 1}: {status} | Steps: {step} | Reward: {episode_reward:.2f}")
                break

    print(f"\nResults: {successes}/{num_episodes} successful landings ({100*successes/num_episodes:.1f}%)")
    game.close()


if __name__ == "__main__":
    main()
