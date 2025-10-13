"""Drone class with physics simulation"""

import numpy as np
import pygame
from . import config
from . import physics as phys


class Drone:
    """Drone with physics-based movement and rendering"""

    def __init__(self, x, y):
        """Initialize drone at position (x, y)

        Args:
            x, y: Starting position
        """
        # Position
        self.x = x
        self.y = y

        # Velocity
        self.vx = 0.0
        self.vy = 0.0

        # Rotation (0 = upright, positive = clockwise)
        self.angle = 0.0  # degrees
        self.angular_velocity = 0.0

        # State
        self.fuel = config.MAX_FUEL
        self.crashed = False
        self.landed = False

        # Dimensions
        self.width = config.DRONE_WIDTH
        self.height = config.DRONE_HEIGHT

        # Thrust indicators (for rendering)
        self.main_thrust_active = False
        self.left_thrust_active = False
        self.right_thrust_active = False

    def apply_thrust(self, main=False, left=False, right=False):
        """Apply thrust forces to the drone

        Args:
            main: Main upward thrust
            left: Left rotational thrust
            right: Right rotational thrust
        """
        # Track for rendering
        self.main_thrust_active = main
        self.left_thrust_active = left
        self.right_thrust_active = right

        # Main thrust (in direction drone is facing)
        if main and self.fuel > 0:
            # Calculate thrust vector based on drone angle
            # 0 degrees = upright, thrust goes up
            thrust_x, thrust_y = phys.rotate_point(0, -config.MAIN_THRUST_POWER, self.angle)
            self.vx += thrust_x
            self.vy += thrust_y
            self.fuel -= config.FUEL_CONSUMPTION_MAIN

        # Side thrusters (rotation)
        if left and self.fuel > 0:
            self.angular_velocity -= config.SIDE_THRUST_POWER
            self.fuel -= config.FUEL_CONSUMPTION_SIDE

        if right and self.fuel > 0:
            self.angular_velocity += config.SIDE_THRUST_POWER
            self.fuel -= config.FUEL_CONSUMPTION_SIDE

        # Ensure fuel doesn't go negative
        self.fuel = max(0, self.fuel)

    def update(self, dt=1.0):
        """Update drone physics

        Args:
            dt: Time delta (default 1.0 for fixed timestep)
        """
        if self.crashed or self.landed:
            return

        # Apply gravity
        self.vy += config.GRAVITY * dt

        # Apply drag
        self.vx *= config.DRAG
        self.vy *= config.DRAG

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update rotation
        self.angle += self.angular_velocity * dt
        self.angular_velocity *= config.ANGULAR_DRAG

        # Normalize angle
        self.angle = phys.normalize_angle(self.angle)

    def get_corners(self):
        """Get drone corner positions (for collision detection)

        Returns:
            List of (x, y) tuples for each corner
        """
        # Define corners relative to center
        half_w = self.width / 2
        half_h = self.height / 2

        corners = [
            (-half_w, -half_h),  # Top-left
            (half_w, -half_h),   # Top-right
            (half_w, half_h),    # Bottom-right
            (-half_w, half_h),   # Bottom-left
        ]

        # Rotate and translate corners
        rotated_corners = []
        for cx, cy in corners:
            rx, ry = phys.rotate_point(cx, cy, self.angle)
            rotated_corners.append((self.x + rx, self.y + ry))

        return rotated_corners

    def get_bottom_center(self):
        """Get position of drone's bottom center point (for landing detection)

        Returns:
            (x, y) tuple
        """
        bottom_offset_x, bottom_offset_y = phys.rotate_point(0, self.height / 2, self.angle)
        return self.x + bottom_offset_x, self.y + bottom_offset_y

    def get_speed(self):
        """Get total speed magnitude

        Returns:
            Speed in pixels per frame
        """
        return np.sqrt(self.vx**2 + self.vy**2)

    def is_upright(self):
        """Check if drone is upright enough to land

        Returns:
            True if angle is within landing tolerance
        """
        return abs(self.angle) <= config.MAX_LANDING_ANGLE

    def render(self, screen):
        """Render drone on pygame screen

        Args:
            screen: Pygame surface
        """
        # Create drone surface
        drone_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Draw drone body (simple rectangle)
        pygame.draw.rect(drone_surface, config.COLOR_DRONE,
                        (0, 0, self.width, self.height))

        # Draw rotors (small circles on sides)
        rotor_radius = 5
        pygame.draw.circle(drone_surface, (100, 100, 100),
                          (rotor_radius, self.height // 2), rotor_radius)
        pygame.draw.circle(drone_surface, (100, 100, 100),
                          (self.width - rotor_radius, self.height // 2), rotor_radius)

        # Draw center marker
        pygame.draw.circle(drone_surface, (50, 50, 50),
                          (self.width // 2, self.height // 2), 3)

        # Rotate drone surface
        rotated_surface = pygame.transform.rotate(drone_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x, self.y))

        # Draw to screen
        screen.blit(rotated_surface, rotated_rect.topleft)

        # Draw thrust flames
        self._render_thrust(screen)

    def _render_thrust(self, screen):
        """Render thrust flames

        Args:
            screen: Pygame surface
        """
        flame_length = 15

        # Main thrust flame (bottom of drone)
        if self.main_thrust_active and self.fuel > 0:
            # Calculate flame position (bottom center of rotated drone)
            flame_x, flame_y = phys.rotate_point(0, self.height / 2 + flame_length / 2, self.angle)
            flame_pos = (int(self.x + flame_x), int(self.y + flame_y))

            # Draw flame as elongated circle
            pygame.draw.ellipse(screen, config.COLOR_THRUST,
                              (flame_pos[0] - 8, flame_pos[1] - flame_length // 2,
                               16, flame_length))

        # Side thrust indicators (smaller)
        side_flame_length = 10

        if self.left_thrust_active and self.fuel > 0:
            flame_x, flame_y = phys.rotate_point(-self.width / 2 - side_flame_length / 2, 0, self.angle)
            flame_pos = (int(self.x + flame_x), int(self.y + flame_y))
            pygame.draw.circle(screen, config.COLOR_THRUST, flame_pos, 5)

        if self.right_thrust_active and self.fuel > 0:
            flame_x, flame_y = phys.rotate_point(self.width / 2 + side_flame_length / 2, 0, self.angle)
            flame_pos = (int(self.x + flame_x), int(self.y + flame_y))
            pygame.draw.circle(screen, config.COLOR_THRUST, flame_pos, 5)

    def reset(self, x, y):
        """Reset drone to initial state

        Args:
            x, y: Starting position
        """
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.fuel = config.MAX_FUEL
        self.crashed = False
        self.landed = False
        self.main_thrust_active = False
        self.left_thrust_active = False
        self.right_thrust_active = False
