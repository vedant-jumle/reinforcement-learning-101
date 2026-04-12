#!/usr/bin/env python3
"""
Test script to verify gymnasium Atari Pong works on WSL2.
Tests environment creation, rendering, and basic interaction.
"""

import sys
import numpy as np

def test_gymnasium_import():
    """Test if gymnasium is installed."""
    print("=" * 60)
    print("Test 1: Importing gymnasium...")
    print("=" * 60)
    try:
        import gymnasium as gym
        print(f"✓ gymnasium imported successfully (version: {gym.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import gymnasium: {e}")
        print("\nInstall with: pip install gymnasium[atari]")
        return False


def test_ale_import():
    """Test if ALE (Atari Learning Environment) is installed."""
    print("\n" + "=" * 60)
    print("Test 2: Importing ale_py (Atari Learning Environment)...")
    print("=" * 60)
    try:
        import ale_py
        print(f"✓ ale_py imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ale_py: {e}")
        print("\nInstall with: pip install ale-py")
        return False


def test_pong_environment():
    """Test if PongNoFrameskip-v4 environment can be created."""
    print("\n" + "=" * 60)
    print("Test 3: Creating PongNoFrameskip-v4 environment...")
    print("=" * 60)
    try:
        import gymnasium as gym
        env = gym.make('PongNoFrameskip-v4')
        print(f"✓ Environment created successfully")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Action meanings: {env.unwrapped.get_action_meanings()}")
        env.close()
        return True, env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False, None


def test_environment_reset():
    """Test if environment can be reset and returns valid observations."""
    print("\n" + "=" * 60)
    print("Test 4: Testing environment reset...")
    print("=" * 60)
    try:
        import gymnasium as gym
        env = gym.make('PongNoFrameskip-v4')
        obs, info = env.reset(seed=42)

        print(f"✓ Environment reset successfully")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation dtype: {obs.dtype}")
        print(f"  - Observation range: [{obs.min()}, {obs.max()}]")
        print(f"  - Info keys: {list(info.keys())}")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_step():
    """Test if environment can execute actions."""
    print("\n" + "=" * 60)
    print("Test 5: Testing environment step (random actions)...")
    print("=" * 60)
    try:
        import gymnasium as gym
        env = gym.make('PongNoFrameskip-v4')
        obs, info = env.reset(seed=42)

        # Take 10 random steps
        print("  Taking 10 random steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if i == 0:
                print(f"    Step {i+1}: action={action}, reward={reward}, done={done}")
                print(f"            obs.shape={obs.shape}, obs range=[{obs.min()}, {obs.max()}]")

        print(f"✓ Environment step works correctly")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rendering():
    """Test if rendering works (rgb_array mode for WSL2)."""
    print("\n" + "=" * 60)
    print("Test 6: Testing rendering (rgb_array mode)...")
    print("=" * 60)
    try:
        import gymnasium as gym

        # Use rgb_array for WSL2 compatibility (no window display)
        env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')
        obs, info = env.reset(seed=42)

        # Take a few steps and render
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Get rendered frame
            frame = env.render()

            if i == 0:
                print(f"✓ Rendering works!")
                print(f"  - Rendered frame shape: {frame.shape}")
                print(f"  - Rendered frame dtype: {frame.dtype}")
                print(f"  - Frame range: [{frame.min()}, {frame.max()}]")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to render: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test Atari preprocessing pipeline (grayscale, resize)."""
    print("\n" + "=" * 60)
    print("Test 7: Testing Atari preprocessing pipeline...")
    print("=" * 60)
    try:
        import gymnasium as gym
        import cv2

        env = gym.make('PongNoFrameskip-v4')
        obs, info = env.reset(seed=42)

        print(f"  Original frame: {obs.shape} (210x160x3 RGB)")

        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        print(f"  After grayscale: {gray.shape}")

        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        print(f"  After resize: {resized.shape}")

        # Normalize to [0, 1]
        normalized = resized / 255.0
        print(f"  After normalize: range=[{normalized.min():.3f}, {normalized.max():.3f}]")

        print(f"✓ Preprocessing pipeline works!")
        print(f"  Final preprocessed frame: {normalized.shape}, dtype={normalized.dtype}")

        env.close()
        return True
    except ImportError as e:
        print(f"✗ opencv-python not installed: {e}")
        print("\nInstall with: pip install opencv-python")
        return False
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_stacking():
    """Test frame stacking (4 frames for temporal information)."""
    print("\n" + "=" * 60)
    print("Test 8: Testing frame stacking (4 frames)...")
    print("=" * 60)
    try:
        import gymnasium as gym
        import cv2

        env = gym.make('PongNoFrameskip-v4')
        obs, info = env.reset(seed=42)

        # Collect 4 frames
        frames = []
        for i in range(4):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Preprocess
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0

            frames.append(normalized)

        # Stack frames
        stacked = np.stack(frames, axis=0)
        print(f"✓ Frame stacking works!")
        print(f"  Stacked frames shape: {stacked.shape} (4, 84, 84)")
        print(f"  This is ready for CNN input: (batch, channels, height, width)")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Frame stacking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "GYMNASIUM ATARI PONG TEST SUITE" + " " * 16 + "║")
    print("║" + " " * 15 + "WSL2 Compatibility Check" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")

    results = []

    # Test 1: gymnasium import
    results.append(("gymnasium import", test_gymnasium_import()))
    if not results[-1][1]:
        print("\n⚠ Cannot proceed without gymnasium. Install and retry.")
        return

    # Test 2: ale_py import
    results.append(("ale_py import", test_ale_import()))
    if not results[-1][1]:
        print("\n⚠ Cannot proceed without ale_py. Install and retry.")
        return

    # Test 3-8: Environment tests
    results.append(("Pong environment creation", test_pong_environment()[0]))
    results.append(("Environment reset", test_environment_reset()))
    results.append(("Environment step", test_environment_step()))
    results.append(("Rendering (rgb_array)", test_rendering()))
    results.append(("Atari preprocessing", test_preprocessing()))
    results.append(("Frame stacking", test_frame_stacking()))

    # Summary
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed! You're ready to start Pong self-play training!")
        print("\nNext steps:")
        print("  1. Implement CNN policy network")
        print("  2. Adapt your PPO code for visual inputs")
        print("  3. Create self-play training loop")
        print("  4. Train and evaluate!")
    else:
        print("\n⚠ Some tests failed. Fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  - pip install gymnasium[atari]")
        print("  - pip install ale-py")
        print("  - pip install opencv-python")

    print("\n")


if __name__ == "__main__":
    main()
