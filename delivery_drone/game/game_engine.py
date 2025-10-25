"""Main game engine with game loop and API interface"""

import pygame
import numpy as np
from . import config
from .drone import Drone
from .platform import Platform
from . import physics as phys


class DroneGame:
    """Main game controller with clean API for external control"""

    def __init__(self, render_mode='human', randomize_drone=False, randomize_platform=True):
        """Initialize game

        Args:
            render_mode: 'human' for display, 'rgb_array' for numpy array, None for headless
            randomize_drone: If True, randomize drone spawn position on reset
            randomize_platform: If True, randomize platform position on reset
        """
        self.render_mode = render_mode
        self.randomize_drone = randomize_drone
        self.randomize_platform = randomize_platform

        # Initialize pygame
        if render_mode != None:
            pygame.init()
            if render_mode == 'human':
                self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
                pygame.display.set_caption("Delivery Drone")
            else:
                # Offscreen rendering for rgb_array mode
                self.screen = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))

            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.large_font = pygame.font.Font(None, 48)

        # Create game objects
        self.drone = Drone(config.DRONE_START_X, config.DRONE_START_Y)
        self.platform = Platform(
            config.WINDOW_WIDTH // 2,
            config.PLATFORM_Y,
            config.PLATFORM_WIDTH,
            config.PLATFORM_HEIGHT
        )

        # Game state
        self.steps = 0
        self.total_reward = 0
        self.episode = 0
        self.done = False

        # Wind (optional)
        self.wind_x = 0.0
        self.wind_y = 0.0

    def reset(self):
        """Reset game to initial state

        Returns:
            Initial state dictionary
        """
        # Reset drone - randomize if enabled
        if self.randomize_drone:
            drone_x = np.random.randint(config.DRONE_START_X_MIN, config.DRONE_START_X_MAX + 1)
            drone_y = np.random.randint(config.DRONE_START_Y_MIN, config.DRONE_START_Y_MAX + 1)
            self.drone.reset(drone_x, drone_y)
        else:
            self.drone.reset(config.DRONE_START_X, config.DRONE_START_Y)

        # Reset platform - randomize if enabled
        if self.randomize_platform:
            platform_x = np.random.randint(
                config.PLATFORM_WIDTH // 2 + 50,  # Margin from left edge
                config.WINDOW_WIDTH - config.PLATFORM_WIDTH // 2 - 50  # Margin from right edge
            )
            platform_y = np.random.randint(
                config.PLATFORM_Y_MIN,  # Higher position (easier)
                config.PLATFORM_Y_MAX   # Lower position (harder)
            )
            self.platform.reset(platform_x, platform_y)
        else:
            self.platform.reset(config.WINDOW_WIDTH // 2, config.PLATFORM_Y)

        # Reset game state
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.episode += 1

        return self.get_state()

    def step(self, action):
        """Execute one game step

        Args:
            action: Dictionary with keys:
                - 'main_thrust': 0 or 1
                - 'left_thrust': 0 or 1
                - 'right_thrust': 0 or 1

        Returns:
            (state, reward, done, info) tuple
        """
        if self.done:
            # Episode already finished, return current info
            info = self._get_info()
            info['needs_reset'] = True
            return self.get_state(), 0, True, info

        # Apply thrust
        self.drone.apply_thrust(
            main=bool(action.get('main_thrust', 0)),
            left=bool(action.get('left_thrust', 0)),
            right=bool(action.get('right_thrust', 0))
        )

        # Apply wind
        if config.WIND_ENABLED:
            self.drone.vx += self.wind_x
            self.drone.vy += self.wind_y

        # Update physics
        self.drone.update()
        self.platform.update()

        # Check collisions and conditions
        reward = self._calculate_reward()
        self.total_reward += reward
        self.steps += 1

        # Get state
        state = self.get_state()
        info = self._get_info()

        return state, reward, self.done, info

    def get_state(self):
        """Get current observable state

        Returns:
            Dictionary with state information
        """
        # Calculate distance to platform
        dx = self.platform.x - self.drone.x
        dy = self.platform.y - self.drone.y
        distance = phys.distance(self.drone.x, self.drone.y, self.platform.x, self.platform.y)

        state = {
            # Drone state
            'drone_x': self.drone.x / config.WINDOW_WIDTH,  # Normalized [0, 1]
            'drone_y': self.drone.y / config.WINDOW_HEIGHT,
            'drone_vx': self.drone.vx / 10.0,  # Normalized velocity
            'drone_vy': self.drone.vy / 10.0,
            'drone_angle': self.drone.angle / 180.0,  # Normalized [-1, 1]
            'drone_angular_vel': self.drone.angular_velocity / 10.0,
            'drone_fuel': self.drone.fuel / config.MAX_FUEL,  # Normalized [0, 1]

            # Platform state
            'platform_x': self.platform.x / config.WINDOW_WIDTH,
            'platform_y': self.platform.y / config.WINDOW_HEIGHT,

            # Relative state
            'distance_to_platform': distance / config.WINDOW_WIDTH,
            'dx_to_platform': dx / config.WINDOW_WIDTH,
            'dy_to_platform': dy / config.WINDOW_HEIGHT,

            # Status
            'speed': self.drone.get_speed() / 10.0,
            'landed': self.drone.landed,
            'crashed': self.drone.crashed,
            'steps': self.steps,
        }

        return state

    def _calculate_reward(self):
        """Calculate reward for current step

        Returns:
            Reward value
        """
        reward = config.REWARD_STEP  # Small step penalty

        # Check for terminal conditions
        if self._check_landing():
            self.drone.landed = True
            self.done = True
            reward += config.REWARD_LANDING
            return reward

        if self._check_crash():
            self.drone.crashed = True
            self.done = True
            reward += config.REWARD_CRASH
            return reward

        if self.drone.fuel <= 0:
            self.drone.crashed = True
            self.done = True
            reward += config.REWARD_OUT_OF_FUEL
            return reward

        if self._check_out_of_bounds():
            self.drone.crashed = True
            self.done = True
            reward += config.REWARD_OUT_OF_BOUNDS
            return reward

        # Reward for getting closer to platform (shaping)
        distance = phys.distance(self.drone.x, self.drone.y, self.platform.x, self.platform.y)
        reward += (500 - distance) / 5000  # Small reward for proximity

        return reward

    def _check_landing(self):
        """Check if drone has landed successfully

        Returns:
            True if successful landing
        """
        if self.drone.crashed or self.drone.landed:
            return False

        # Get drone bottom center
        bottom_x, bottom_y = self.drone.get_bottom_center()

        # Check if on platform
        if not self.platform.is_point_on_platform(bottom_x, bottom_y):
            return False

        # Check velocity
        if self.drone.get_speed() > config.MAX_LANDING_VELOCITY:
            return False

        # Check angle
        if not self.drone.is_upright():
            return False

        return True

    def _check_crash(self):
        """Check if drone has crashed

        Returns:
            True if crashed
        """
        if self.drone.crashed:
            return True

        # Check ground collision (below platform level)
        ground_level = config.WINDOW_HEIGHT - 50
        if self.drone.y > ground_level:
            # Check if NOT on platform
            bottom_x, bottom_y = self.drone.get_bottom_center()
            if not self.platform.is_point_on_platform(bottom_x, bottom_y):
                return True

            # On platform but too fast or wrong angle
            if self.drone.get_speed() > config.MAX_LANDING_VELOCITY:
                return True
            if not self.drone.is_upright():
                return True

        return False

    def _check_out_of_bounds(self):
        """Check if drone is out of bounds

        Returns:
            True if out of bounds
        """
        margin = config.OUT_OF_BOUNDS_MARGIN
        return (self.drone.x < -margin or
                self.drone.x > config.WINDOW_WIDTH + margin or
                self.drone.y < -margin or
                self.drone.y > config.WINDOW_HEIGHT + margin)

    def _get_info(self):
        """Get additional information dictionary

        Returns:
            Dictionary with extra information
        """
        return {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'episode': self.episode,
            'fuel_remaining': self.drone.fuel,
            'distance_to_platform': phys.distance(
                self.drone.x, self.drone.y,
                self.platform.x, self.platform.y
            ),
            'speed': self.drone.get_speed(),
            'angle': self.drone.angle,
        }

    def render(self):
        """Render the game

        Returns:
            None for 'human' mode, numpy array for 'rgb_array' mode
        """
        if self.render_mode is None:
            return None

        # Clear screen
        self.screen.fill(config.COLOR_SKY)

        # Draw ground
        ground_y = config.WINDOW_HEIGHT - 50
        pygame.draw.rect(self.screen, config.COLOR_GROUND,
                        (0, ground_y, config.WINDOW_WIDTH, 50))

        # Draw platform
        self.platform.render(self.screen)

        # Draw drone
        self.drone.render(self.screen)

        # Draw HUD
        self._render_hud()

        # Draw game over messages
        if self.done:
            self._render_game_over()

        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(config.FPS)
        elif self.render_mode == 'rgb_array':
            # Return as numpy array
            return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

        return None

    def _render_hud(self):
        """Render heads-up display"""
        hud_margin = 10
        line_height = 25

        # Fuel bar
        fuel_bar_width = 200
        fuel_bar_height = 20
        fuel_x = hud_margin
        fuel_y = hud_margin

        # Background
        pygame.draw.rect(self.screen, (50, 50, 50),
                        (fuel_x, fuel_y, fuel_bar_width, fuel_bar_height))

        # Fuel level
        fuel_percent = self.drone.fuel / config.MAX_FUEL
        fuel_color = (0, 255, 0) if fuel_percent > 0.3 else (255, 255, 0) if fuel_percent > 0.1 else (255, 0, 0)
        pygame.draw.rect(self.screen, fuel_color,
                        (fuel_x, fuel_y, fuel_bar_width * fuel_percent, fuel_bar_height))

        # Fuel text
        fuel_text = self.font.render(f"Fuel: {int(self.drone.fuel)}", True, config.COLOR_TEXT)
        self.screen.blit(fuel_text, (fuel_x + 5, fuel_y + 2))

        # Speed
        y_offset = fuel_y + fuel_bar_height + 10
        speed_text = self.font.render(f"Speed: {self.drone.get_speed():.1f}", True, config.COLOR_TEXT)
        self.screen.blit(speed_text, (hud_margin, y_offset))

        # Angle
        y_offset += line_height
        angle_text = self.font.render(f"Angle: {self.drone.angle:.1f}°", True, config.COLOR_TEXT)
        self.screen.blit(angle_text, (hud_margin, y_offset))

        # Distance to platform
        y_offset += line_height
        distance = phys.distance(self.drone.x, self.drone.y, self.platform.x, self.platform.y)
        dist_text = self.font.render(f"Distance: {distance:.0f}", True, config.COLOR_TEXT)
        self.screen.blit(dist_text, (hud_margin, y_offset))

        # Episode/Steps (top right)
        episode_text = self.font.render(f"Episode: {self.episode}", True, config.COLOR_TEXT)
        self.screen.blit(episode_text, (config.WINDOW_WIDTH - 150, hud_margin))

        steps_text = self.font.render(f"Steps: {self.steps}", True, config.COLOR_TEXT)
        self.screen.blit(steps_text, (config.WINDOW_WIDTH - 150, hud_margin + line_height))

    def _render_game_over(self):
        """Render game over message"""
        # Semi-transparent overlay
        overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Message
        if self.drone.landed:
            message = "SUCCESSFUL LANDING!"
            color = (0, 255, 0)
        else:
            message = "CRASHED!"
            color = (255, 0, 0)

        text = self.large_font.render(message, True, color)
        text_rect = text.get_rect(center=(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 - 30))
        self.screen.blit(text, text_rect)

        # Score
        score_text = self.font.render(f"Total Reward: {self.total_reward:.1f}", True, config.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + 20))
        self.screen.blit(score_text, score_rect)

        # Restart prompt
        restart_text = self.font.render("Press R to restart", True, config.COLOR_TEXT)
        restart_rect = restart_text.get_rect(center=(config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(restart_text, restart_rect)

    def close(self):
        """Clean up resources"""
        if self.render_mode is not None:
            pygame.quit()
