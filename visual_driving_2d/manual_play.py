#!/usr/bin/env python
"""
Manual Play Mode for Car Racing Game

Play the game with keyboard controls to understand the task.
Controls:
    Arrow Keys / WASD: Steer, Gas, Brake
    R: Reset
    ESC: Quit
"""

import sys
import pyglet
from pyglet.window import key

from game.game_engine import RacingGame
from game import config


class ManualController:
    """Keyboard controller for manual play"""

    def __init__(self):
        self.steer = 0.0
        self.gas = 0.0
        self.brake = 0.0
        self.reset_pressed = False
        self.quit_pressed = False

    def on_key_press(self, symbol, modifiers):
        """Handle key press"""
        # Quit
        if symbol == key.ESCAPE:
            self.quit_pressed = True

        # Reset
        if symbol == key.R:
            self.reset_pressed = True

    def on_key_release(self, symbol, modifiers):
        """Handle key release"""
        pass

    def update_from_keys(self, keys):
        """Update action from currently pressed keys"""
        # Steering
        if keys[key.LEFT] or keys[key.A]:
            self.steer = -1.0
        elif keys[key.RIGHT] or keys[key.D]:
            self.steer = +1.0
        else:
            self.steer = 0.0

        # Gas
        if keys[key.UP] or keys[key.W]:
            self.gas = 1.0
        else:
            self.gas = 0.0

        # Brake
        if keys[key.DOWN] or keys[key.S]:
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
    print("Car Racing - Manual Play Mode")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow Keys / WASD: Steer, Gas, Brake")
    print("    LEFT/A:  Steer left")
    print("    RIGHT/D: Steer right")
    print("    UP/W:    Gas")
    print("    DOWN/S:  Brake")
    print("  R:   Reset")
    print("  ESC: Quit")
    print("\nTip: This is a powerful rear-wheel drive car!")
    print("     Don't accelerate and turn at the same time.\n")
    print("=" * 60)

    # Create game
    game = RacingGame(render_mode='human', verbose=1)
    game.reset()

    # Create controller
    controller = ManualController()

    # Setup pyglet window event handlers
    if game.viewer:
        @game.viewer.event
        def on_key_press(symbol, modifiers):
            controller.on_key_press(symbol, modifiers)

        @game.viewer.event
        def on_key_release(symbol, modifiers):
            controller.on_key_release(symbol, modifiers)

    # Game loop
    running = True
    episode = 0
    episode_reward = 0.0
    episode_steps = 0

    print(f"\n[Episode {episode}] Starting...")

    try:
        while running:
            # Handle window events
            if game.viewer:
                game.viewer.dispatch_events()

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
            keys = key.KeyStateHandler()
            if game.viewer:
                # Pyglet key state
                keys = game.viewer._keyboard
            controller.update_from_keys(keys)

            action = controller.get_action()

            # Step game
            state, reward, done, info = game.step(action)
            episode_reward += reward
            episode_steps += 1

            # Render
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
            if episode_steps % 100 == 0:
                print(f"  Step {episode_steps}, Reward: {episode_reward:.1f}, "
                      f"Tiles: {game.tile_visited_count}/{len(game.track)}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\nClosing game...")
        game.close()
        print("Goodbye!")


if __name__ == '__main__':
    main()
