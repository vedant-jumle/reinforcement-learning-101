"""Vehicle physics using simplified bicycle model"""

import math
import numpy as np
from . import config


class Vehicle:
    """Vehicle with bicycle model physics"""

    def __init__(self, x, y, heading):
        """Initialize vehicle

        Args:
            x, y: Position (pixels)
            heading: Angle in degrees (0 = right, 90 = up)
        """
        # Position & orientation
        self.x = x
        self.y = y
        self.heading = heading  # degrees

        # Velocity
        self.velocity = 0.0  # pixels/sec (forward speed)

        # Steering
        self.steering_angle = 0.0  # degrees

        # Dimensions
        self.width = config.VEHICLE_WIDTH
        self.length = config.VEHICLE_LENGTH
        self.wheelbase = config.VEHICLE_WHEELBASE

        # State tracking
        self.distance_traveled = 0.0
        self.assigned_lane = None  # Set by game on spawn

    def update(self, dt, action):
        """Update vehicle physics for one timestep

        Args:
            dt: Delta time (seconds)
            action: {'steering': [-1, 1], 'acceleration': [-1, 1]}
        """
        # Parse action
        steering_input = np.clip(action.get('steering', 0.0), -1.0, 1.0)
        accel_input = np.clip(action.get('acceleration', 0.0), -1.0, 1.0)

        # Update steering angle (with rate limit)
        target_steering = steering_input * config.MAX_STEERING_ANGLE
        steering_diff = target_steering - self.steering_angle
        max_change = config.STEERING_RATE * dt
        steering_diff = np.clip(steering_diff, -max_change, max_change)
        self.steering_angle += steering_diff

        # Update velocity (acceleration/braking)
        if accel_input > 0:
            # Accelerate
            accel = accel_input * config.ACCELERATION_RATE
            self.velocity += accel * dt
            self.velocity = min(self.velocity, config.MAX_SPEED_FORWARD)
        elif accel_input < 0:
            # Brake
            brake = abs(accel_input) * config.BRAKE_RATE
            self.velocity -= brake * dt
            self.velocity = max(self.velocity, -config.MAX_SPEED_REVERSE)
        else:
            # Friction
            self.velocity *= config.FRICTION_COEFFICIENT
            if abs(self.velocity) < 0.1:
                self.velocity = 0.0

        # Bicycle model: update heading based on steering
        # heading_rate = velocity * tan(steering_angle) / wheelbase
        # In Y-UP system: positive steering should DECREASE heading (turn right/clockwise)
        if abs(self.velocity) > 0.1 and abs(self.steering_angle) > 0.1:
            steering_rad = math.radians(self.steering_angle)
            heading_rate = self.velocity * math.tan(steering_rad) / self.wheelbase
            self.heading -= math.degrees(heading_rate * dt)  # Negative to match Y-UP system
            self.heading = self.heading % 360  # Normalize to [0, 360)

        # Update position
        heading_rad = math.radians(self.heading)
        dx = self.velocity * math.cos(heading_rad) * dt
        dy = -self.velocity * math.sin(heading_rad) * dt  # Negative because y increases downward

        self.x += dx
        self.y += dy
        self.distance_traveled += abs(self.velocity * dt)

    def get_corners(self):
        """Get vehicle corner positions for collision detection

        Returns:
            List of 4 (x, y) tuples representing corners
        """
        # Calculate corners relative to center
        half_length = self.length / 2
        half_width = self.width / 2

        # Corners in vehicle frame (before rotation)
        corners_local = [
            (half_length, half_width),   # Front right
            (half_length, -half_width),  # Front left
            (-half_length, -half_width), # Rear left
            (-half_length, half_width),  # Rear right
        ]

        # Rotate and translate to world frame
        # Coordinate system: X-right, Y-up (like movement uses -sin for y)
        # Local vehicle frame: X=forward (heading direction), Y=left
        heading_rad = math.radians(self.heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        corners_world = []
        for lx, ly in corners_local:
            # Transform: world = center + lx*heading_vec + ly*left_vec
            # heading_vec = (cos, -sin), left_vec = (-sin, -cos)
            wx = self.x + lx * cos_h - ly * sin_h
            wy = self.y - lx * sin_h - ly * cos_h
            corners_world.append((wx, wy))

        return corners_world

    def get_front_center(self):
        """Get position of front center of vehicle

        Front is at local position (length/2, 0) in vehicle frame
        """
        heading_rad = math.radians(self.heading)
        front_x = self.x + (self.length / 2) * math.cos(heading_rad)
        front_y = self.y - (self.length / 2) * math.sin(heading_rad)
        return (front_x, front_y)

    def reset(self, x, y, heading):
        """Reset vehicle to new position"""
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.distance_traveled = 0.0
