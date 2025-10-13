#!/usr/bin/env python3
"""Remote agent - demonstrates controlling the game over socket

This agent connects to a running socket server and controls the drone remotely.
Start the server first: python socket_server.py
"""

import sys
sys.path.insert(0, '..')

from game.socket_client import DroneGameClient
import time


def simple_remote_policy(state):
    """Simple rule-based policy for remote control

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
    dy = state['dy_to_platform']  # Vertical distance to platform

    # Stabilize angle
    if drone_angle > 0.1:  # Tilted right, thrust left
        action['left_thrust'] = 1
    elif drone_angle < -0.1:  # Tilted left, thrust right
        action['right_thrust'] = 1

    # Counteract falling
    if drone_vy > 0.3:  # Falling too fast
        action['main_thrust'] = 1
    elif dy < 0 and drone_vy > 0.1:  # Near platform and falling
        action['main_thrust'] = 1

    return action


def main():
    """Run remote agent"""
    print("=" * 60)
    print("REMOTE AGENT - Socket-Based Control")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  python socket_server.py --render human")
    print("\nConnecting to server...")
    print("=" * 60)

    # Create client
    client = DroneGameClient(host='localhost', port=5555)

    try:
        # Connect and reset
        state = client.reset()
        print("\n✓ Connected to game server!")
        print(f"✓ Initial state received: {len(state)} keys")

        # Run episodes
        num_episodes = 5
        successes = 0

        print(f"\nRunning {num_episodes} episodes with remote control...\n")

        for episode in range(num_episodes):
            state = client.reset()
            episode_reward = 0
            step = 0
            episode_start = time.time()

            while True:
                # Get action from policy
                action = simple_remote_policy(state)

                # Send action, receive next state
                state, reward, done, info = client.step(action)
                episode_reward += reward
                step += 1

                if done:
                    episode_time = time.time() - episode_start

                    if state['landed']:
                        status = "SUCCESS ✓"
                        successes += 1
                    else:
                        status = "CRASHED ✗"

                    print(f"Episode {episode + 1}: {status} | "
                          f"Steps: {step} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Time: {episode_time:.2f}s")
                    break

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS: {successes}/{num_episodes} successful landings "
              f"({100*successes/num_episodes:.1f}%)")
        print(f"{'='*60}")

    except ConnectionError as e:
        print(f"\n✗ Connection error: {e}")
        print("\nMake sure the server is running:")
        print("  python socket_server.py --render human")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Disconnect
        client.close()
        print("\nDisconnected from server")


if __name__ == "__main__":
    main()
