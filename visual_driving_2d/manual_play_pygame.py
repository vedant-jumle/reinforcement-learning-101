#!/usr/bin/env python
"""
Manual Play Mode for Car Racing Game - PYGAME VERSION
This version uses pygame for rendering and works on WSL2.

Play the game with keyboard controls to understand the task.
Controls:
    Arrow Keys / WASD: Steer, Gas, Brake
    R: Reset
    ESC: Quit
"""

import sys
import numpy as np
import pygame
import Box2D
from Box2D.b2 import (fixtureDef, polygonShape)

from game import config
from game.game_engine import RacingGame
from game.car_dynamics import Car


class PygameRacingGame(RacingGame):
    """Racing game with pygame rendering instead of pyglet"""

    def __init__(self, **kwargs):
        # Don't call super().__init__() with render_mode
        # Initialize without rendering first
        self.render_mode = 'pygame'  # Custom render mode
        self.verbose = kwargs.get('verbose', 1)
        self.total_episode_steps = kwargs.get('total_episode_steps', config.DEFAULT_EPISODE_STEPS)

        # Random number generator
        seed = kwargs.get('seed', None)
        self.np_random = np.random.RandomState(seed)

        # Box2D physics world
        from game.game_engine import FrictionDetector
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        # Rendering - pygame instead of pyglet
        self.screen = None
        self.clock = None
        self.font = None

        # Game state
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.current_steps = 0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.track = None

        # Track tile fixture definition
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
        pygame.display.set_caption("Car Racing - Manual Play")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)

    def render(self, mode='human'):
        """Render using pygame"""
        if self.screen is None:
            return None

        # Clear screen
        self.screen.fill((int(config.GRASS_COLOR[0] * 255),
                         int(config.GRASS_COLOR[1] * 255),
                         int(config.GRASS_COLOR[2] * 255)))

        if not hasattr(self, 't'):
            return None

        # Calculate camera transform
        zoom = config.ZOOM * config.SCALE
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]

        # Draw track
        for poly, color in self.road_poly:
            points = []
            for p in poly:
                # Transform world coordinates to screen coordinates
                screen_x = config.WINDOW_W / 2 + (p[0] - scroll_x) * zoom
                screen_y = config.WINDOW_H / 4 + (p[1] - scroll_y) * zoom
                points.append((screen_x, config.WINDOW_H - screen_y))

            if len(points) >= 3:
                pygame_color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                pygame.draw.polygon(self.screen, pygame_color, points)

        # Draw car
        car_x = config.WINDOW_W / 2
        car_y = config.WINDOW_H / 4
        angle_deg = np.degrees(self.car.hull.angle)

        # Simple car representation (red rectangle)
        car_length = 30
        car_width = 15
        car_rect = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        car_rect.fill((255, 0, 0, 255))
        rotated_car = pygame.transform.rotate(car_rect, angle_deg)
        car_rect_pos = rotated_car.get_rect(center=(car_x, config.WINDOW_H - car_y))
        self.screen.blit(rotated_car, car_rect_pos)

        # Draw score
        score_text = self.font.render(f"{int(self.reward)}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))

        # Draw telemetry
        small_font = pygame.font.Font(None, 24)
        speed_text = small_font.render(f"Speed: {self.car.hull.linearVelocity.length:.1f}", True, (255, 255, 255))
        tiles_text = small_font.render(f"Tiles: {self.tile_visited_count}/{len(self.track) if self.track else 0}", True, (255, 255, 255))
        self.screen.blit(speed_text, (20, 70))
        self.screen.blit(tiles_text, (20, 95))

        pygame.display.flip()
        self.clock.tick(config.FPS)

        return None

    def close(self):
        """Clean up pygame"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class ManualController:
    """Keyboard controller for manual play"""

    def __init__(self):
        self.steer = 0.0
        self.gas = 0.0
        self.brake = 0.0
        self.reset_pressed = False
        self.quit_pressed = False

    def update_from_pygame_keys(self):
        """Update action from pygame key state"""
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
    print("Car Racing - Manual Play Mode (PYGAME)")
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

    # Create game with pygame rendering
    game = PygameRacingGame(verbose=1)
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
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        controller.reset_pressed = True

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
            if episode_steps % 100 == 0 and episode_steps > 0:
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
