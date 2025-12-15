#!/usr/bin/env python
"""Test the gymnasium CarRacing socket system"""

import time
from game.socket_client import RacingGameClient


def test_socket_system():
    print("=" * 60)
    print("Testing Gymnasium CarRacing Socket System")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  python socket_server_gymnasium.py --num-games 1 --render human")
    print("\nStarting test in 2 seconds...\n")
    time.sleep(2)

    try:
        # Connect to server
        with RacingGameClient(host='localhost', port=5555) as client:
            print(f"✓ Connected to server with {client.num_games} game(s)")

            # Reset
            print("\nResetting game...")
            telemetry, frame, reward, done, info = client.reset(game_id=0)
            print(f"✓ Reset successful")
            if frame is not None:
                print(f"  Frame shape: {frame.shape}")
            print(f"  Telemetry: {telemetry}")

            # Run for 200 steps
            print("\nRunning 200 steps with simple agent...")
            total_reward = 0.0

            for step in range(200):
                # Simple agent: gas + slight steering
                action = {
                    'steer': 0.0,
                    'gas': 1.0 if step < 50 else 0.5,  # Full gas first, then moderate
                    'brake': 0.0
                }

                # Send action
                client.set_action(action, game_id=0)

                # Get state
                telemetry, frame, reward, done, info = client.get_state(game_id=0)
                total_reward += reward

                if step % 40 == 0:
                    print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.1f}, done={done}")

                if done:
                    print(f"\n  Episode ended at step {step}")
                    telemetry, frame, reward, done, info = client.reset(game_id=0)
                    total_reward = 0.0

            print(f"\n✓ Test completed successfully!")
            print(f"  Final total reward: {total_reward:.1f}")

    except ConnectionRefusedError:
        print("\n✗ ERROR: Could not connect to server")
        print("Please start the server first:")
        print("  python socket_server_gymnasium.py --render human\n")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_socket_system()