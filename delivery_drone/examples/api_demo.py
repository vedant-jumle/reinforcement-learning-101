#!/usr/bin/env python3
"""API Demo - Shows how to use the game interface programmatically"""

import sys
sys.path.insert(0, '..')

from game.game_engine import DroneGame
import time


def demo_basic_api():
    """Demonstrate basic API usage"""
    print("="*60)
    print("DELIVERY DRONE - API DEMO")
    print("="*60)

    # Create game (headless mode for this demo)
    print("\n1. Creating game instance...")
    game = DroneGame(render_mode=None)
    print("   ✓ Game created")

    # Reset game
    print("\n2. Resetting game...")
    state = game.reset()
    print("   ✓ Game reset")
    print(f"   Initial state keys: {list(state.keys())}")

    # Show initial state
    print("\n3. Initial state values:")
    for key, value in state.items():
        if isinstance(value, float):
            print(f"   {key:25s}: {value:.4f}")
        else:
            print(f"   {key:25s}: {value}")

    # Execute actions
    print("\n4. Executing actions...")
    actions_to_test = [
        ("No thrust (falling)", {'main_thrust': 0, 'left_thrust': 0, 'right_thrust': 0}),
        ("Main thrust only", {'main_thrust': 1, 'left_thrust': 0, 'right_thrust': 0}),
        ("Left thrust only", {'main_thrust': 0, 'left_thrust': 1, 'right_thrust': 0}),
        ("Right thrust only", {'main_thrust': 0, 'left_thrust': 0, 'right_thrust': 1}),
    ]

    for description, action in actions_to_test:
        print(f"\n   Testing: {description}")
        print(f"   Action: {action}")

        state, reward, done, info = game.step(action)

        print(f"   → Reward: {reward:.3f}")
        print(f"   → Speed: {state['speed']:.3f}")
        print(f"   → Angle: {state['drone_angle']:.3f}")
        print(f"   → Fuel: {state['drone_fuel']:.3f}")
        print(f"   → Done: {done}")

    # Run a full episode
    print("\n5. Running a full episode with simple controller...")
    game.reset()

    step = 0
    total_reward = 0

    while step < 500:  # Max 500 steps
        # Simple controller: just thrust up
        action = {'main_thrust': 1, 'left_thrust': 0, 'right_thrust': 0}

        state, reward, done, info = game.step(action)
        total_reward += reward
        step += 1

        if done:
            break

    print(f"   Episode finished after {step} steps")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Result: {'LANDED ✓' if state['landed'] else 'CRASHED ✗'}")
    print(f"   Fuel remaining: {state['drone_fuel']*100:.1f}%")

    # Show info dictionary
    print("\n6. Info dictionary contents:")
    info = game._get_info()
    for key, value in info.items():
        print(f"   {key:20s}: {value}")

    print("\n" + "="*60)
    print("API Demo Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  • game.reset() → returns initial state")
    print("  • game.step(action) → returns (state, reward, done, info)")
    print("  • game.get_state() → get current state without stepping")
    print("  • State is a dictionary with normalized values")
    print("  • Actions are dictionaries with binary values (0 or 1)")
    print("\nYou can now build RL agents on top of this interface!")


if __name__ == "__main__":
    demo_basic_api()
