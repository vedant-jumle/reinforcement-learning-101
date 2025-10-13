"""Main game engine with continuous physics loop"""

import time
import threading
import numpy as np
import pygame
import cv2
from . import config
from .vehicle import Vehicle
from .road import RoadNetwork
from .camera import Camera
from .renderer import Renderer
from .collision import CollisionDetector
from .rewards import RewardCalculator


class DrivingGame:
    """Main game with continuous physics loop"""

    def __init__(self, phase=1, render_mode='human', **kwargs):
        """Initialize game

        Args:
            phase: Game phase (1, 2, or 3)
            render_mode: 'human', 'rgb_array', or None
            **kwargs: Override config flags
        """
        self.phase = phase
        self.render_mode = render_mode

        # Override config from kwargs
        for key, value in kwargs.items():
            key_upper = key.upper()
            if hasattr(config, key_upper):
                setattr(config, key_upper, value)

        # Initialize pygame if rendering
        if render_mode is not None:
            pygame.init()
            if render_mode == 'human':
                self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
                pygame.display.set_caption(f"Visual Driving 2D - Phase {phase}")
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
                self.clock = None
        else:
            self.screen = None
            self.clock = None

        # Game objects
        self.vehicle = None
        self.road_network = None
        self.camera = Camera(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.renderer = Renderer(self.screen)
        self.collision_detector = CollisionDetector(phase)
        self.reward_calculator = RewardCalculator(phase)

        # Game state
        self.steps = 0
        self.episode = 0
        self.done = False
        self.time = 0.0
        self.total_reward = 0.0

        # Action buffering (thread-safe)
        self.current_action = {'steering': 0.0, 'acceleration': 0.0}
        self.action_lock = threading.Lock()
        self.last_action_time = time.time()

        # Physics loop control
        self.running = False
        self.physics_thread = None

        # Observations for client
        self.latest_observation = None
        self.observation_lock = threading.Lock()

        # Reward tracking
        self.last_reward = 0.0
        self.prev_vehicle_state = None

    def start_physics_loop(self):
        """Start continuous physics updates at fixed FPS"""
        if self.physics_thread is not None and self.physics_thread.is_alive():
            return  # Already running

        self.running = True
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()
        print(f"[Game {id(self) % 10000}] Physics loop started")

    def stop_physics_loop(self):
        """Stop physics loop"""
        self.running = False
        if self.physics_thread:
            self.physics_thread.join(timeout=2.0)
        print(f"[Game {id(self) % 10000}] Physics loop stopped")

    def _physics_loop(self):
        """Run physics at fixed rate (60 FPS)"""
        dt = 1.0 / config.PHYSICS_FPS

        while self.running:
            loop_start = time.time()

            if not self.done and self.vehicle is not None:
                # Get latest action (thread-safe)
                with self.action_lock:
                    action = self.current_action.copy()
                    action_age = time.time() - self.last_action_time

                # Use zero action if too old
                if action_age > config.MAX_ACTION_AGE:
                    action = {'steering': 0.0, 'acceleration': 0.0}

                # Update vehicle physics
                self.vehicle.update(dt, action)

                # Check collisions and compute rewards
                self._step_logic()

                # Generate observation
                obs = self._generate_observation()
                with self.observation_lock:
                    self.latest_observation = obs

                self.steps += 1
                self.time += dt

            # Maintain fixed timestep
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step_logic(self):
        """Run collision detection and reward calculation"""
        # Save previous state for reward calculation
        if self.prev_vehicle_state is None:
            self.prev_vehicle_state = {
                'x': self.vehicle.x,
                'y': self.vehicle.y,
                'distance_traveled': self.vehicle.distance_traveled
            }

        # Grace period: skip collision checks for first few steps after spawn
        # This prevents false positives during initialization
        GRACE_PERIOD_STEPS = 5

        if self.steps < GRACE_PERIOD_STEPS:
            # During grace period, no crashes detected
            # Return same structure as collision detector for consistency
            crashes = {
                'terminal': False,
                'reason': None,
                'off_road': False,
                'wrong_lane': False,
                'out_of_bounds': False
            }
        else:
            # Check collisions normally
            crashes = self.collision_detector.check_crashes(
                self.vehicle,
                self.road_network,
                phase=self.phase
            )

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self.vehicle,
            self.road_network,
            crashes,
            self.prev_vehicle_state,
            phase=self.phase
        )

        self.last_reward = reward
        self.total_reward += reward

        # Check terminal conditions
        if crashes.get('terminal', False):
            self.done = True
            print(f"[Game {id(self) % 10000}] Episode ended: {crashes.get('reason', 'Unknown')}")

        # Check step limit
        max_steps_attr = f'MAX_STEPS_PHASE{self.phase}'
        max_steps = getattr(config, max_steps_attr, 1000)
        if self.steps >= max_steps:
            self.done = True
            print(f"[Game {id(self) % 10000}] Episode ended: Step limit reached ({max_steps})")

        # Update prev state
        self.prev_vehicle_state = {
            'x': self.vehicle.x,
            'y': self.vehicle.y,
            'distance_traveled': self.vehicle.distance_traveled
        }

    def set_action(self, action):
        """Set action from socket client (thread-safe)

        Args:
            action: {'steering': [-1, 1], 'acceleration': [-1, 1]}
        """
        with self.action_lock:
            self.current_action = action.copy()
            self.last_action_time = time.time()

    def get_observation(self):
        """Get latest visual observation (thread-safe)

        Returns:
            numpy array (84, 84, 3) or latest available
        """
        with self.observation_lock:
            if self.latest_observation is not None:
                return self.latest_observation.copy()
            else:
                # Generate one if none available
                return self._generate_observation()

    def _generate_observation(self):
        """Generate visual observation from current game state

        Returns:
            numpy array (84, 84, 3) - RGB image
        """
        if self.screen is None or self.vehicle is None:
            # Return blank observation
            return np.zeros((config.OBSERVATION_HEIGHT, config.OBSERVATION_WIDTH, 3), dtype=np.uint8)

        # Create temporary surface for rendering
        temp_surface = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))

        # Render game state
        self.renderer.render(temp_surface, self.vehicle, self.road_network, self.camera)

        # Convert to numpy array
        frame = pygame.surfarray.array3d(temp_surface)

        # Transpose (pygame uses width x height x channels)
        frame = np.transpose(frame, (1, 0, 2))  # -> height x width x channels

        # Crop around vehicle
        cropped = self.camera.crop_around_vehicle(frame, self.vehicle)

        # Resize to observation size
        resized = cv2.resize(cropped, (config.OBSERVATION_WIDTH, config.OBSERVATION_HEIGHT))

        return resized.astype(np.uint8)

    def reset(self):
        """Reset game to initial state

        Returns:
            Initial observation
        """
        # Generate road network
        self.road_network = RoadNetwork(phase=self.phase)

        # Spawn vehicle
        if config.RANDOMIZE_SPAWN_POSITION:
            x, y, heading, lane = self.road_network.get_random_spawn_position()
        else:
            x, y, heading, lane = 400, 300, 0, 0

        self.vehicle = Vehicle(x, y, heading)
        self.vehicle.assigned_lane = lane

        # Reset state
        self.steps = 0
        self.done = False
        self.time = 0.0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.episode += 1
        self.prev_vehicle_state = None

        # Reset action
        with self.action_lock:
            self.current_action = {'steering': 0.0, 'acceleration': 0.0}
            self.last_action_time = time.time()

        # Generate initial observation
        obs = self._generate_observation()
        with self.observation_lock:
            self.latest_observation = obs

        print(f"[Game {id(self) % 10000}] Reset - Episode {self.episode}")

        return obs

    def get_state(self):
        """Get state dictionary (for debugging/logging)

        Returns:
            Dict with vehicle position, velocity, etc.
        """
        if self.vehicle is None:
            return {}

        segment, lane = self.road_network.get_segment_and_lane(self.vehicle.x, self.vehicle.y)

        return {
            'x': float(self.vehicle.x),
            'y': float(self.vehicle.y),
            'heading': float(self.vehicle.heading),
            'velocity': float(self.vehicle.velocity),
            'steering_angle': float(self.vehicle.steering_angle),
            'on_road': segment is not None,
            'lane': int(lane) if lane is not None else -1,
            'distance_traveled': float(self.vehicle.distance_traveled),
            'steps': int(self.steps),
            'time': float(self.time),
            'done': bool(self.done)
        }

    def render(self):
        """Render game (only in human mode)"""
        if self.render_mode == 'human' and self.screen is not None and self.vehicle is not None:
            self.renderer.render(self.screen, self.vehicle, self.road_network, self.camera)
            pygame.display.flip()
            if self.clock:
                self.clock.tick(config.RENDER_FPS)

    def close(self):
        """Cleanup resources"""
        self.stop_physics_loop()
        if self.screen is not None:
            pygame.quit()
