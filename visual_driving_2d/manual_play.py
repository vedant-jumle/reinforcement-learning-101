#!/usr/bin/env python3
"""Manual play mode with keyboard control"""

import pygame
import argparse
from game.game_engine import DrivingGame
from game import config


def main():
    parser = argparse.ArgumentParser(description='Manual Play Mode')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                       help='Game phase')
    args = parser.parse_args()

    # Create game
    game = DrivingGame(phase=args.phase, render_mode='human')
    game.reset()

    print("Manual Play Mode")
    print("Controls:")
    print("  Arrow Keys: Steer and Accelerate/Brake")
    print("  UP: Accelerate")
    print("  DOWN: Brake")
    print("  LEFT/RIGHT: Steer")
    print("  R: Reset")
    print("  ESC: Quit")
    print("="*50)

    running = True
    clock = pygame.time.Clock()

    # Start physics loop
    game.start_physics_loop()

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    print(f"Reset - Episode {game.episode}")

        # Get keyboard input
        keys = pygame.key.get_pressed()

        # Build action
        action = {
            'steering': 0.0,
            'acceleration': 0.0
        }

        # Steering
        if keys[pygame.K_LEFT]:
            action['steering'] = -1.0
        elif keys[pygame.K_RIGHT]:
            action['steering'] = 1.0

        # Acceleration/Braking
        if keys[pygame.K_UP]:
            action['acceleration'] = 1.0
        elif keys[pygame.K_DOWN]:
            action['acceleration'] = -1.0

        # Set action
        game.set_action(action)

        # Render
        game.render()

        # Check if done
        if game.done:
            print(f"Episode ended! Total reward: {game.total_reward:.2f}")
            print(f"Steps: {game.steps}, Distance: {game.vehicle.distance_traveled:.0f}px")
            print("Press R to reset")

        # Maintain FPS
        clock.tick(60)

    # Cleanup
    game.close()
    pygame.quit()


if __name__ == '__main__':
    main()
