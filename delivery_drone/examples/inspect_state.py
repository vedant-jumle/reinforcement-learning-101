#!/usr/bin/env python3
"""Inspect game state - useful for debugging and understanding the state space"""

import sys
sys.path.insert(0, '..')

from game.game_engine import DroneGame
import time


def main():
    """Print state information in real-time"""
    game = DroneGame(render_mode='human')
    state = game.reset()

    print("=" * 60)
    print("STATE INSPECTOR")
    print("=" * 60)
    print("\nThis demo prints the game state every 30 frames.")
    print("Watch how the state changes as the drone falls!")
    print("\nPress ESC to quit\n")
    print("=" * 60 + "\n")

    import pygame
    running = True
    frame = 0

    while running:
        # No action (just let it fall)
        action = {
            'main_thrust': 0,
            'left_thrust': 0,
            'right_thrust': 0
        }

        # Allow some manual control for testing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = game.reset()
                    frame = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action['main_thrust'] = 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action['left_thrust'] = 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action['right_thrust'] = 1

        # Step
        state, reward, done, info = game.step(action)
        game.render()

        # Print state every 30 frames
        if frame % 30 == 0:
            print(f"\n--- Frame {frame} ---")
            print(f"Position: ({state['drone_x']:.3f}, {state['drone_y']:.3f})")
            print(f"Velocity: ({state['drone_vx']:.3f}, {state['drone_vy']:.3f})")
            print(f"Angle: {state['drone_angle']:.3f} | Speed: {state['speed']:.3f}")
            print(f"Fuel: {state['drone_fuel']:.3f}")
            print(f"Distance to platform: {state['distance_to_platform']:.3f}")
            print(f"Reward: {reward:.3f} | Done: {done}")

        if done:
            print(f"\n{'='*60}")
            print(f"EPISODE FINISHED")
            print(f"{'='*60}")
            print(f"Result: {'LANDED' if game.drone.landed else 'CRASHED'}")
            print(f"Steps: {info['steps']}")
            print(f"Total Reward: {info['total_reward']:.2f}")
            print(f"Fuel Remaining: {info['fuel_remaining']:.1f}")
            print(f"{'='*60}\n")

            # Auto-reset
            time.sleep(2)
            state = game.reset()
            frame = 0

        frame += 1

    game.close()


if __name__ == "__main__":
    main()
