#!/usr/bin/env python3
"""Manual play mode - control the drone with keyboard"""

import pygame
import sys
from game.game_engine import DroneGame


def main():
    """Main game loop for manual play"""
    # Create game
    game = DroneGame(render_mode='human')

    # Reset to start
    game.reset()

    print("=" * 50)
    print("DELIVERY DRONE - Manual Play Mode")
    print("=" * 50)
    print("\nControls:")
    print("  W or ↑  : Main thrust (upward)")
    print("  A or ←  : Left thruster (rotate left)")
    print("  D or →  : Right thruster (rotate right)")
    print("  R       : Reset/Restart")
    print("  ESC     : Quit")
    print("\nObjective:")
    print("  Land on the green platform safely!")
    print("  - Keep speed low")
    print("  - Stay upright")
    print("  - Don't run out of fuel")
    print("\n" + "=" * 50 + "\n")

    running = True
    episode_printed = False  # Track if we've printed the episode results

    while running:
        # Handle input
        action = {
            'main_thrust': 0,
            'left_thrust': 0,
            'right_thrust': 0
        }

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset game
                    game.reset()
                    episode_printed = False  # Reset flag for new episode
                    print(f"\n[Episode {game.episode}] Game reset!")

        # Get continuous key presses
        keys = pygame.key.get_pressed()

        # Main thrust
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action['main_thrust'] = 1

        # Left thrust
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action['left_thrust'] = 1

        # Right thrust
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action['right_thrust'] = 1

        # Step game
        state, reward, done, info = game.step(action)

        # Render
        game.render()

        # Print episode results when done (only once)
        if done and not episode_printed:
            if game.drone.landed:
                print(f"\n[Episode {game.episode}] ✓ SUCCESS! Landed safely!")
            else:
                print(f"\n[Episode {game.episode}] ✗ CRASHED!")

            print(f"  Steps: {info['steps']}")
            print(f"  Total Reward: {info['total_reward']:.2f}")
            print(f"  Fuel Remaining: {info['fuel_remaining']:.1f}")
            print(f"  Press R to restart\n")
            episode_printed = True  # Mark as printed

    # Cleanup
    game.close()
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
