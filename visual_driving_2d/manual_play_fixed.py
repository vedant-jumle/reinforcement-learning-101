#!/usr/bin/env python
"""
Manual Play Mode for Car Racing Game - WSL2 Compatible

Uses pygame for key handling to avoid pyglet GL context issues.
Play the game with keyboard controls to understand the task.

Controls:
    Arrow Keys / WASD: Steer, Gas, Brake
    R: Reset
    ESC/Q: Quit
"""

import sys
import pygame
from game.game_engine import RacingGame
from game import config


class ManualController:
    """Keyboard controller for manual play using pygame"""

    def __init__(self):
        self.steer = 0.0
        self.gas = 0.0
        self.brake = 0.0
        self.reset_pressed = False
        self.quit_pressed = False

        # Initialize pygame for key handling only
        pygame.init()

    def handle_pygame_events(self):
        """Process pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_pressed = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.quit_pressed = True
                elif event.key == pygame.K_r:
                    self.reset_pressed = True

    def update_from_pygame_keys(self):
        """Update action from currently pressed keys"""
        keys = pygame.key.get_pressed()

        # Steering
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.steer = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.steer = +1.0
        else:
            self.steer = 0.0

        # Gas
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.gas = 1.0
        else:
            self.gas = 0.0

        # Brake
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.brake = 0.8
        else:
            self.brake = 0.0

    def get_action(self):
        """Get current action dict"""
        return {
            'steer': self.steer,
            'gas': self.gas,
            'brake': self.brake
        }


def main():
    print("=" * 60)
    print("Car Racing - Manual Play Mode (WSL2 Compatible)")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow Keys / WASD: Steer, Gas, Brake")
    print("    LEFT/A:  Steer left")
    print("    RIGHT/D: Steer right")
    print("    UP/W:    Gas")
    print("    DOWN/S:  Brake")
    print("  R:   Reset")
    print("  ESC/Q: Quit")
    print("\nTip: This is a powerful rear-wheel drive car!")
    print("     Don't accelerate and turn at the same time.\n")
    print("=" * 60)

    # Create game with gymnasium rendering (works on WSL2)
    game = RacingGame(render_mode='human', verbose=1)
    game.reset()

    # Create controller
    controller = ManualController()

    # Game loop
    running = True
    episode = 0
    episode_reward = 0.0
    episode_steps = 0

    print(f"\n[Episode {episode}] Starting...")

    try:
        while running:
            # Handle pygame events (for key handling)
            controller.handle_pygame_events()

            # Check quit
            if controller.quit_pressed:
                break

            # Check reset
            if controller.reset_pressed:
                print(f"\n[Episode {episode}] Manual reset")
                print(f"  Steps: {episode_steps}")
                print(f"  Reward: {episode_reward:.1f}")
                print(f"  Tiles: {game.tile_visited_count}/{len(game.track)}")

                game.reset()
                controller.reset_pressed = False
                episode += 1
                episode_reward = 0.0
                episode_steps = 0
                print(f"\n[Episode {episode}] Starting...")
                continue

            # Update action from keyboard
            controller.update_from_pygame_keys()
            action = controller.get_action()

            # Step game
            state, reward, done, info = game.step(action)
            episode_reward += reward
            episode_steps += 1

            # Render (uses gymnasium rendering)
            game.render('human')

            # Check if episode done
            if done:
                print(f"\n[Episode {episode}] FINISHED!")
                print(f"  Steps: {episode_steps}")
                print(f"  Reward: {episode_reward:.1f}")
                print(f"  Tiles: {game.tile_visited_count}/{len(game.track)}")

                if game.tile_visited_count >= len(game.track):
                    print("  STATUS: ✓ Track completed!")
                elif abs(game.car.hull.position[0]) > config.PLAYFIELD or \
                     abs(game.car.hull.position[1]) > config.PLAYFIELD:
                    print("  STATUS: ✗ Out of bounds")
                else:
                    print("  STATUS: ✗ Timeout")

                # Reset for next episode
                game.reset()
                episode += 1
                episode_reward = 0.0
                episode_steps = 0
                print(f"\n[Episode {episode}] Starting...")

            # Print periodic updates
            if episode_steps % 100 == 0 and episode_steps > 0:
                print(f"  Step {episode_steps}, Reward: {episode_reward:.1f}, "
                      f"Tiles: {game.tile_visited_count}/{len(game.track)}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\nClosing game...")
        game.close()
        pygame.quit()
        print("Goodbye!")


if __name__ == '__main__':
    main()
