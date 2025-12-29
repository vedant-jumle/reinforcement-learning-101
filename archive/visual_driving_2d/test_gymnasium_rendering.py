#!/usr/bin/env python
"""Test if gymnasium's CarRacing works on WSL2"""

import gymnasium as gym

print("Testing gymnasium CarRacing environment...")
print("Creating environment...")

try:
    env = gym.make('CarRacing-v3', render_mode='human')
    print("✓ Environment created successfully!")

    print("Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful, observation shape: {obs.shape}")

    print("Running 100 steps with visualization...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 20 == 0:
            print(f"  Step {i}: reward={reward:.2f}")

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

    env.close()
    print("\n✓ SUCCESS! Gymnasium rendering works on WSL2!")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
