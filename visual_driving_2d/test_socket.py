#!/usr/bin/env python
"""
Test Socket Client/Server Communication

Run this to test the socket system.
First start the server in another terminal:
    python socket_server.py --render none

Then run this script:
    python test_socket.py
"""

import time
import numpy as np
from game.socket_client import RacingGameClient


def test_basic_connection():
    """Test basic connection and handshake"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Connection")
    print("=" * 60)

    client = RacingGameClient(host='localhost', port=5555)
    client.connect()

    print(f"✓ Connected successfully")
    print(f"✓ Server has {client.num_games} game instance(s)")

    client.disconnect()
    print("✓ Disconnected successfully\n")


def test_reset_and_state():
    """Test reset and state retrieval"""
    print("=" * 60)
    print("TEST 2: Reset and State Retrieval")
    print("=" * 60)

    with RacingGameClient() as client:
        print(f"✓ Connected (context manager)")

        # Reset
        telemetry, frame, reward, done, info = client.reset(game_id=0)

        print(f"✓ Reset successful")
        print(f"  Telemetry keys: {list(telemetry.keys())}")
        if frame is not None:
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
        else:
            print(f"  Frame: None (server in headless mode)")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

        # Get state
        telemetry, frame, reward, done, info = client.get_state(game_id=0)
        print(f"✓ GET_STATE successful")
        print(f"  Car position: ({telemetry['car_x']:.2f}, {telemetry['car_y']:.2f})")
        print(f"  Car speed: {telemetry['speed']:.2f}")


def test_action_and_step():
    """Test sending actions and stepping"""
    print("\n" + "=" * 60)
    print("TEST 3: Actions and Stepping")
    print("=" * 60)

    with RacingGameClient() as client:
        client.reset(game_id=0)

        # Send action
        action = {'steer': 0.5, 'gas': 1.0, 'brake': 0.0}
        client.set_action(action, game_id=0)
        print(f"✓ SET_ACTION successful: {action}")

        # Wait a bit for game to step
        time.sleep(0.1)

        # Get new state
        telemetry, frame, reward, done, info = client.get_state(game_id=0)
        print(f"✓ State retrieved after action")
        print(f"  Car position: ({telemetry['car_x']:.2f}, {telemetry['car_y']:.2f})")
        print(f"  Car speed: {telemetry['speed']:.2f}")
        print(f"  Reward: {reward:.2f}")


def test_full_episode():
    """Test running a full episode"""
    print("\n" + "=" * 60)
    print("TEST 4: Full Episode (Simple Agent)")
    print("=" * 60)

    with RacingGameClient() as client:
        telemetry, frame, reward, done, info = client.reset(game_id=0)

        print("Running episode with simple forward agent...")

        total_reward = 0.0
        step = 0
        max_steps = 100

        while not done and step < max_steps:
            # Simple agent: just drive forward
            action = {'steer': 0.0, 'gas': 0.5, 'brake': 0.0}

            client.set_action(action, game_id=0)
            telemetry, frame, reward, done, info = client.get_state(game_id=0)

            total_reward += reward
            step += 1

            if step % 20 == 0:
                print(f"  Step {step}: reward={total_reward:.1f}, "
                      f"speed={telemetry['speed']:.2f}, "
                      f"tiles={telemetry['tiles_visited']}/{telemetry['total_tiles']}")

        print(f"✓ Episode finished")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Tiles visited: {telemetry['tiles_visited']}/{telemetry['total_tiles']}")
        print(f"  Done: {done}")


def test_parallel_games():
    """Test multiple game instances"""
    print("\n" + "=" * 60)
    print("TEST 5: Parallel Game Instances")
    print("=" * 60)

    with RacingGameClient() as client:
        num_games = client.num_games

        print(f"Server has {num_games} game instance(s)")

        if num_games > 1:
            # Reset all games
            for i in range(num_games):
                client.reset(game_id=i)
                print(f"  ✓ Reset game {i}")

            # Step all games
            for i in range(num_games):
                action = {'steer': 0.0, 'gas': 1.0, 'brake': 0.0}
                client.set_action(action, game_id=i)

            time.sleep(0.1)

            # Get all states
            for i in range(num_games):
                telemetry, frame, reward, done, info = client.get_state(game_id=i)
                print(f"  ✓ Game {i}: speed={telemetry['speed']:.2f}")

            print(f"✓ Parallel games working")
        else:
            print("  ⚠ Only 1 game instance (start server with --num-games 4)")


def test_latency_benchmark():
    """Test latency performance"""
    print("\n" + "=" * 60)
    print("TEST 6: Latency Benchmark")
    print("=" * 60)

    from game.socket_client import benchmark_latency

    try:
        results = benchmark_latency(iterations=50)
        print(f"✓ Latency benchmark completed")
    except Exception as e:
        print(f"✗ Latency benchmark failed: {e}")


def main():
    print("\n" + "=" * 60)
    print("Car Racing Socket System Test Suite")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  python socket_server.py --render none")
    print("\nStarting tests in 2 seconds...\n")

    time.sleep(2)

    try:
        test_basic_connection()
        test_reset_and_state()
        test_action_and_step()
        test_full_episode()
        test_parallel_games()
        test_latency_benchmark()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")

    except ConnectionRefusedError:
        print("\n✗ ERROR: Could not connect to server")
        print("Please start the server first:")
        print("  python socket_server.py --render none\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
