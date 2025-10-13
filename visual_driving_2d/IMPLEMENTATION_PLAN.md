# Implementation Plan: Visual Driving 2D (Phase 1-3)

**Target**: Complete implementation of Phases 1-3 with full feature toggles, comprehensive randomization, and continuous game loop architecture.

---

## 🎯 Core Architecture Decisions

### 1. Continuous Game Loop (Critical Difference from Delivery Drone)

**Problem**: In delivery_drone, game only updates when client sends action. This works for single-agent scenarios but **breaks down with multiple autonomous vehicles** that need to move independently.

**Solution**:
```
┌─────────────────────────────────────────────────────────┐
│  Game Process (Separate Thread)                        │
│                                                         │
│  ┌──────────────────────────────────────┐             │
│  │  Physics Loop (60 FPS, continuous)   │             │
│  │  • Update RL vehicle with latest action            │
│  │  • Update ALL NPC vehicles (A* pathfinding)        │
│  │  • Update physics for all entities   │             │
│  │  • Check collisions                  │             │
│  │  • Render frame                      │             │
│  └──────────────────────────────────────┘             │
│            ▲                                            │
│            │ Latest Action (buffered)                   │
│            │                                            │
│  ┌──────────────────────────────────────┐             │
│  │  Socket Server (Network Thread)      │             │
│  │  • Receives actions from client      │             │
│  │  • Updates action buffer             │             │
│  │  • Sends back latest observation     │             │
│  └──────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
                       ▲ TCP
                       │
           ┌───────────┴───────────┐
           │  RL Agent (Client)    │
           │  • Sends actions      │
           │  • Gets observations  │
           │  • Trains policy      │
           └───────────────────────┘
```

**Key Differences:**
- Game runs at **fixed 60 FPS** regardless of agent speed
- Agent actions are **buffered** and applied on next physics tick
- Socket client gets **latest observation** immediately (no waiting for next step)
- NPCs move continuously even when agent is "thinking"

### 2. Feature Toggle System

Every feature can be enabled/disabled via:
1. **Config file** (`config.py`)
2. **CLI arguments** (`--enable-curves`, `--disable-intersections`)
3. **Runtime API** (for experiments)

### 3. Comprehensive Randomization

**Philosophy**: Randomize everything possible for robust, generalizable policies.

### 4. Progressive Crash Detection

Each phase adds new failure modes with specific detection logic and penalties.

---

## 📦 Phase 1: Basic Driving Mechanics

### Goal
Learn fundamental vehicle control from visual input on simple roads.

### File Structure
```
visual_driving_2d/
├── game/
│   ├── __init__.py
│   ├── config.py              # Configuration + feature flags
│   ├── vehicle.py             # Vehicle physics (bicycle model)
│   ├── road.py                # Road representation + generation
│   ├── camera.py              # Top-down camera (follow vehicle)
│   ├── renderer.py            # Pygame rendering
│   ├── collision.py           # Collision detection utilities
│   ├── rewards.py             # Reward calculation per phase
│   ├── game_engine.py         # Main game with continuous loop
│   ├── socket_server.py       # TCP server with threading
│   └── socket_client.py       # Client library
├── socket_server.py           # Main server script
├── manual_play.py             # Keyboard control for testing
├── test_physics.py            # Unit tests for vehicle physics
└── IMPLEMENTATION_PLAN.md     # This file
```

---

## 🔧 Configuration System (`game/config.py`)

### Phase Control
```python
# ============================================================
# PHASE SELECTION
# ============================================================
PHASE = 1  # Current phase (1, 2, or 3)

# ============================================================
# FEATURE FLAGS - PHASE 1: BASIC DRIVING
# ============================================================
ENABLE_LANE_BOUNDARIES = True      # Road has edges
ENABLE_CURVES = True               # Roads can curve
ENABLE_LANE_MARKINGS = True        # Visual lane lines
ENABLE_MULTIPLE_LANES = True       # 2+ lanes per road

# ============================================================
# FEATURE FLAGS - PHASE 2: NAVIGATION
# ============================================================
ENABLE_INTERSECTIONS = False       # T-junctions, 4-way intersections
ENABLE_GOAL_NAVIGATION = False     # Goal marker on map
ENABLE_MULTIPLE_ROUTES = False     # Multiple paths to goal
ENABLE_ROUTE_PLANNING = False      # Show optimal route hint

# ============================================================
# FEATURE FLAGS - PHASE 3: OBSTACLES
# ============================================================
ENABLE_STATIC_OBSTACLES = False    # Any obstacles at all
ENABLE_PARKED_CARS = False         # Parked vehicles on roadside
ENABLE_BARRIERS = False            # Construction barriers
ENABLE_CONES = False               # Traffic cones
ENABLE_DEBRIS = False              # Random debris

# ============================================================
# RANDOMIZATION FLAGS - SPAWNING
# ============================================================
RANDOMIZE_SPAWN_POSITION = True    # Random (x, y) spawn
RANDOMIZE_SPAWN_HEADING = True     # Random initial angle
RANDOMIZE_SPAWN_LANE = True        # Random lane within road

# Spawn ranges (pixels)
SPAWN_X_RANGE = (100, 700)
SPAWN_Y_RANGE = (200, 400)
SPAWN_HEADING_RANGE = (-30, 30)    # Degrees from road direction

# ============================================================
# RANDOMIZATION FLAGS - ROAD NETWORK
# ============================================================
RANDOMIZE_ROAD_LAYOUT = True       # Different road configurations
RANDOMIZE_ROAD_CURVATURE = True    # Straight vs. curved sections
RANDOMIZE_ROAD_LENGTH = True       # Length of road segments
RANDOMIZE_LANE_WIDTH = True        # Width of lanes
RANDOMIZE_NUM_LANES = True         # 1-4 lanes

# Road generation ranges
ROAD_CURVATURE_RANGE = (0.0, 0.5)  # 0=straight, 1=tight curve
ROAD_LENGTH_RANGE = (500, 2000)     # Pixels per segment
LANE_WIDTH_RANGE = (60, 100)        # Pixels
NUM_LANES_RANGE = (1, 3)            # Number of lanes

# ============================================================
# RANDOMIZATION FLAGS - PHASE 2 (GOALS/INTERSECTIONS)
# ============================================================
RANDOMIZE_GOAL_POSITION = False     # Random goal location
RANDOMIZE_INTERSECTION_LAYOUT = False  # Random intersection positions
RANDOMIZE_ROUTE_COMPLEXITY = False  # Number of turns required

GOAL_DISTANCE_RANGE = (500, 1500)   # Distance from spawn
INTERSECTION_SPACING_RANGE = (300, 600)  # Distance between intersections

# ============================================================
# RANDOMIZATION FLAGS - PHASE 3 (OBSTACLES)
# ============================================================
RANDOMIZE_OBSTACLE_COUNT = False    # Number of obstacles
RANDOMIZE_OBSTACLE_POSITIONS = False  # Where obstacles spawn
RANDOMIZE_OBSTACLE_SIZES = False    # Size variation within type
RANDOMIZE_OBSTACLE_TYPES = False    # Mix of different obstacle types
RANDOMIZE_OBSTACLE_DENSITY = False  # Clustered vs. spread out

# Obstacle ranges
MIN_OBSTACLES = 3
MAX_OBSTACLES = 15
OBSTACLE_SIZE_VARIANCE = 0.3        # ±30% size variation
MIN_OBSTACLE_SPACING = 150          # Pixels between obstacles
MIN_SPAWN_CLEARANCE = 200           # Don't spawn near vehicle/goal

# ============================================================
# VEHICLE PHYSICS
# ============================================================
VEHICLE_WIDTH = 30                  # Pixels
VEHICLE_LENGTH = 50                 # Pixels
VEHICLE_WHEELBASE = 40              # For steering calculations

# Speed (pixels/second)
MAX_SPEED_FORWARD = 200
MAX_SPEED_REVERSE = 100
ACCELERATION_RATE = 150             # pixels/sec²
BRAKE_RATE = 200                    # pixels/sec²
FRICTION_COEFFICIENT = 0.98         # Velocity decay per frame

# Steering
MAX_STEERING_ANGLE = 35             # Degrees
STEERING_RATE = 120                 # Degrees/sec (how fast steering responds)

# ============================================================
# RENDERING & OBSERVATION
# ============================================================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
RENDER_FPS = 60
PHYSICS_FPS = 60                    # Should match for simplicity

# Camera
CAMERA_FOLLOW_VEHICLE = True
CAMERA_OFFSET_Y = -100              # Camera centered slightly ahead

# Visual observation (what CNN sees)
OBSERVATION_WIDTH = 84
OBSERVATION_HEIGHT = 84
OBSERVATION_CHANNELS = 3            # RGB
FRAME_STACK_SIZE = 4                # Stack 4 frames for motion

# What to render in observation
RENDER_LANE_LINES = True
RENDER_ROAD_EDGES = True
RENDER_OTHER_VEHICLES = True        # For Phase 4+
RENDER_GOAL_INDICATOR = False       # Small arrow pointing to goal

# ============================================================
# GAME LOOP & NETWORKING
# ============================================================
SOCKET_PORT = 5555
SOCKET_HOST = '0.0.0.0'
SOCKET_BUFFER_SIZE = 4096
SOCKET_TIMEOUT = 30.0

# Action buffering
MAX_ACTION_AGE = 0.1                # Seconds (use last action if no new one)

# Episode limits
MAX_STEPS_PHASE1 = 1000             # Time limit for Phase 1
MAX_STEPS_PHASE2 = 2000             # More time for navigation
MAX_STEPS_PHASE3 = 2500             # Even more with obstacles

# ============================================================
# REWARDS - PHASE 1
# ============================================================
# Positive rewards
REWARD_ON_ROAD = 0.01               # Per timestep on road
REWARD_FORWARD_PROGRESS = 0.1       # Per pixel forward along road
REWARD_LANE_CENTER = 0.05           # Bonus for staying centered

# Negative rewards
PENALTY_OFF_ROAD = -1.0             # Per timestep off road
PENALTY_CROSSED_LANE = -0.1         # Per timestep in wrong lane
PENALTY_JERKY_STEERING = -0.01      # Per degree of steering * action
PENALTY_SPEED_ERROR = -0.05         # Penalty for too fast/slow

# Terminal rewards (end episode)
REWARD_COMPLETE_CIRCUIT = 50.0      # Completed one lap
CRASH_PENALTY_OFF_ROAD = -50.0      # Drove off road
CRASH_PENALTY_OUT_OF_BOUNDS = -50.0 # Too far from play area

# ============================================================
# REWARDS - PHASE 2
# ============================================================
REWARD_GOAL_REACHED = 100.0         # Reached goal marker
REWARD_PROGRESS_TO_GOAL = 0.2       # Per pixel closer to goal
PENALTY_WRONG_TURN = -5.0           # Turned wrong way at intersection
PENALTY_TIME_STEP = -0.01           # Encourage efficiency
CRASH_PENALTY_WRONG_TURN = -50.0    # Terminal wrong turn

# ============================================================
# REWARDS - PHASE 3
# ============================================================
REWARD_SAFE_OBSTACLE_PASS = 0.5     # Passed obstacle with safe distance
PENALTY_TOO_CLOSE_OBSTACLE = -0.5   # Per timestep near obstacle
CRASH_PENALTY_OBSTACLE = -100.0     # Hit obstacle

# Collision detection distances
SAFE_DISTANCE_THRESHOLD = 60        # Pixels (warning zone)
COLLISION_DISTANCE_THRESHOLD = 25   # Pixels (actual collision)

# ============================================================
# COLORS (RGB)
# ============================================================
COLOR_ROAD = (60, 60, 60)           # Dark gray
COLOR_GRASS = (34, 139, 34)         # Green
COLOR_LANE_LINE = (255, 255, 255)   # White
COLOR_ROAD_EDGE = (255, 255, 0)     # Yellow
COLOR_VEHICLE_PLAYER = (0, 100, 255)  # Blue
COLOR_VEHICLE_NPC = (200, 200, 200) # Gray
COLOR_GOAL = (0, 255, 0)            # Green
COLOR_OBSTACLE = (139, 69, 19)      # Brown
```

---

## 🚗 Vehicle Physics (`game/vehicle.py`)

### Bicycle Model Implementation
```python
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
        if abs(self.velocity) > 0.1 and abs(self.steering_angle) > 0.1:
            steering_rad = math.radians(self.steering_angle)
            heading_rate = self.velocity * math.tan(steering_rad) / self.wheelbase
            self.heading += math.degrees(heading_rate * dt)
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
        heading_rad = math.radians(self.heading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        corners_world = []
        for lx, ly in corners_local:
            # Rotation matrix
            wx = lx * cos_h - ly * sin_h + self.x
            wy = lx * sin_h + ly * cos_h + self.y
            corners_world.append((wx, wy))

        return corners_world

    def get_front_center(self):
        """Get position of front center of vehicle"""
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
```

---

## 🛣️ Road System (`game/road.py`)

### Phase 1: Simple Circuit
```python
"""Road network generation and queries"""

import math
import numpy as np
from . import config

class RoadSegment:
    """A single road segment (straight or curved)"""

    def __init__(self, start_x, start_y, length, heading, curvature, num_lanes, lane_width):
        """Initialize road segment

        Args:
            start_x, start_y: Starting position
            length: Length of segment (pixels)
            heading: Initial heading (degrees)
            curvature: Curvature factor (0 = straight, >0 = curved)
            num_lanes: Number of parallel lanes
            lane_width: Width of each lane (pixels)
        """
        self.start_x = start_x
        self.start_y = start_y
        self.length = length
        self.heading = heading
        self.curvature = curvature
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.total_width = num_lanes * lane_width

        # Pre-compute road path
        self.path_points = self._generate_path()
        self.left_edge = self._generate_edge(-self.total_width / 2)
        self.right_edge = self._generate_edge(self.total_width / 2)
        self.lane_centers = [self._generate_edge((i - (num_lanes - 1) / 2) * lane_width)
                            for i in range(num_lanes)]

    def _generate_path(self, num_points=100):
        """Generate points along road centerline"""
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            distance = t * self.length

            # Curved or straight
            if self.curvature > 0:
                # Circular arc
                radius = self.length / (2 * math.pi * self.curvature)
                angle = (distance / radius)
                x = self.start_x + radius * math.sin(angle)
                y = self.start_y - radius * (1 - math.cos(angle))
            else:
                # Straight line
                heading_rad = math.radians(self.heading)
                x = self.start_x + distance * math.cos(heading_rad)
                y = self.start_y - distance * math.sin(heading_rad)

            points.append((x, y))

        return points

    def _generate_edge(self, offset):
        """Generate edge line with perpendicular offset from centerline"""
        edge = []
        for i, (cx, cy) in enumerate(self.path_points):
            # Get tangent direction at this point
            if i < len(self.path_points) - 1:
                next_x, next_y = self.path_points[i + 1]
                tangent_angle = math.atan2(-(next_y - cy), next_x - cx)
            else:
                # Use previous tangent for last point
                prev_x, prev_y = self.path_points[i - 1]
                tangent_angle = math.atan2(-(cy - prev_y), cx - prev_x)

            # Perpendicular offset
            perp_angle = tangent_angle + math.pi / 2
            ex = cx + offset * math.cos(perp_angle)
            ey = cy + offset * math.sin(perp_angle)
            edge.append((ex, ey))

        return edge

    def is_point_on_road(self, x, y):
        """Check if point is within road boundaries"""
        # Find nearest point on centerline
        min_dist = float('inf')
        nearest_idx = 0

        for i, (px, py) in enumerate(self.path_points):
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Check if distance from centerline is within road width
        cx, cy = self.path_points[nearest_idx]

        # Get perpendicular distance to centerline
        # (Simplified: use distance to nearest point)
        return min_dist <= self.total_width / 2

    def get_lane_at_position(self, x, y):
        """Get which lane (0 to num_lanes-1) the position is in"""
        if not self.is_point_on_road(x, y):
            return None

        # Find nearest centerline point
        min_dist = float('inf')
        nearest_idx = 0
        for i, (px, py) in enumerate(self.path_points):
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Calculate lateral offset from centerline
        cx, cy = self.path_points[nearest_idx]

        # Get tangent direction
        if nearest_idx < len(self.path_points) - 1:
            next_x, next_y = self.path_points[nearest_idx + 1]
            tangent_angle = math.atan2(-(next_y - cy), next_x - cx)
        else:
            prev_x, prev_y = self.path_points[nearest_idx - 1]
            tangent_angle = math.atan2(-(cy - prev_y), cx - prev_x)

        # Vector from centerline to point
        dx = x - cx
        dy = y - cy

        # Perpendicular offset (signed distance from centerline)
        perp_angle = tangent_angle + math.pi / 2
        lateral_offset = dx * math.cos(perp_angle) + dy * math.sin(perp_angle)

        # Convert to lane index
        lane_idx = int((lateral_offset + self.total_width / 2) / self.lane_width)
        lane_idx = np.clip(lane_idx, 0, self.num_lanes - 1)

        return lane_idx

class RoadNetwork:
    """Collection of road segments forming a network"""

    def __init__(self, phase=1):
        """Initialize road network

        Args:
            phase: Game phase (1, 2, or 3)
        """
        self.phase = phase
        self.segments = []
        self.intersections = []  # For Phase 2+
        self.obstacles = []      # For Phase 3+

        # Generate network
        self._generate_network()

    def _generate_network(self):
        """Generate road network based on phase"""
        if self.phase == 1:
            self._generate_simple_circuit()
        elif self.phase == 2:
            self._generate_intersection_network()
        elif self.phase == 3:
            self._generate_intersection_network()  # Same as phase 2, obstacles added separately

    def _generate_simple_circuit(self):
        """Generate simple oval/circuit for Phase 1"""
        self.segments = []

        # Number of segments in circuit
        num_segments = 4

        for i in range(num_segments):
            # Randomize segment properties if enabled
            if config.RANDOMIZE_ROAD_LENGTH:
                length = np.random.uniform(*config.ROAD_LENGTH_RANGE)
            else:
                length = 600

            if config.RANDOMIZE_ROAD_CURVATURE:
                curvature = np.random.uniform(*config.ROAD_CURVATURE_RANGE)
            else:
                curvature = 0.25 if i % 2 == 1 else 0.0  # Alternate straight/curved

            if config.RANDOMIZE_NUM_LANES:
                num_lanes = np.random.randint(*config.NUM_LANES_RANGE)
            else:
                num_lanes = 2

            if config.RANDOMIZE_LANE_WIDTH:
                lane_width = np.random.uniform(*config.LANE_WIDTH_RANGE)
            else:
                lane_width = 80

            # Position and heading (connect segments end-to-end)
            if i == 0:
                start_x, start_y = 200, 300
                heading = 0  # Right
            else:
                # Connect to previous segment
                prev_segment = self.segments[-1]
                last_point = prev_segment.path_points[-1]
                start_x, start_y = last_point

                # Heading based on curvature
                heading = (i * 90) % 360

            segment = RoadSegment(
                start_x, start_y,
                length, heading, curvature,
                num_lanes, lane_width
            )
            self.segments.append(segment)

    def is_point_on_road(self, x, y):
        """Check if point is on any road segment"""
        for segment in self.segments:
            if segment.is_point_on_road(x, y):
                return True
        return False

    def get_segment_and_lane(self, x, y):
        """Get which segment and lane the point is in

        Returns:
            (segment, lane_idx) or (None, None) if not on road
        """
        for segment in self.segments:
            if segment.is_point_on_road(x, y):
                lane_idx = segment.get_lane_at_position(x, y)
                return segment, lane_idx
        return None, None

    def get_random_spawn_position(self):
        """Get random valid spawn position on road"""
        # Pick random segment
        segment = np.random.choice(self.segments)

        # Pick random point along segment
        point_idx = np.random.randint(len(segment.path_points))
        cx, cy = segment.path_points[point_idx]

        # Pick random lane if enabled
        if config.RANDOMIZE_SPAWN_LANE:
            lane_idx = np.random.randint(segment.num_lanes)
        else:
            lane_idx = 0  # Default to first lane

        # Get lane center position
        lx, ly = segment.lane_centers[lane_idx][point_idx]

        # Get heading at this point
        if point_idx < len(segment.path_points) - 1:
            next_x, next_y = segment.path_points[point_idx + 1]
            heading = math.degrees(math.atan2(-(next_y - cy), next_x - cx))
        else:
            prev_x, prev_y = segment.path_points[point_idx - 1]
            heading = math.degrees(math.atan2(-(cy - prev_y), cx - prev_x))

        # Add random heading variation if enabled
        if config.RANDOMIZE_SPAWN_HEADING:
            heading += np.random.uniform(*config.SPAWN_HEADING_RANGE)

        return lx, ly, heading, lane_idx
```

---

## 🎮 Game Engine with Continuous Loop (`game/game_engine.py`)

```python
"""Main game engine with continuous physics loop"""

import time
import threading
import numpy as np
import pygame
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
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)

        # Initialize pygame if rendering
        if render_mode is not None:
            pygame.init()
            if render_mode == 'human':
                self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
                pygame.display.set_caption(f"Visual Driving 2D - Phase {phase}")
            else:
                self.screen = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        else:
            self.screen = None

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
        if self.physics_thread is not None:
            return  # Already running

        self.running = True
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()
        print(f"[Game {id(self)}] Physics loop started")

    def stop_physics_loop(self):
        """Stop physics loop"""
        self.running = False
        if self.physics_thread:
            self.physics_thread.join(timeout=2.0)
        print(f"[Game {id(self)}] Physics loop stopped")

    def _physics_loop(self):
        """Run physics at fixed rate (60 FPS)"""
        dt = 1.0 / config.PHYSICS_FPS

        while self.running:
            loop_start = time.time()

            if not self.done:
                # Get latest action (thread-safe)
                with self.action_lock:
                    action = self.current_action.copy()
                    action_age = time.time() - self.last_action_time

                # Use zero action if too old
                if action_age > config.MAX_ACTION_AGE:
                    action = {'steering': 0.0, 'acceleration': 0.0}

                # Update vehicle physics
                if self.vehicle:
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

        # Check collisions
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
            print(f"Episode ended: {crashes.get('reason', 'Unknown')}")

        # Check step limit
        max_steps = getattr(config, f'MAX_STEPS_PHASE{self.phase}', 1000)
        if self.steps >= max_steps:
            self.done = True
            print(f"Episode ended: Step limit reached ({max_steps})")

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
        if self.screen is None:
            # Headless mode: create temporary surface
            temp_surface = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
            self.renderer.render(temp_surface, self.vehicle, self.road_network, self.camera)
            frame = pygame.surfarray.array3d(temp_surface)
        else:
            # Render to screen
            self.renderer.render(self.screen, self.vehicle, self.road_network, self.camera)
            frame = pygame.surfarray.array3d(self.screen)

        # Transpose (pygame uses width x height x channels)
        frame = np.transpose(frame, (1, 0, 2))  # -> height x width x channels

        # Crop around vehicle
        cropped = self.camera.crop_around_vehicle(frame, self.vehicle)

        # Resize to observation size
        import cv2
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

        print(f"[Game {id(self)}] Reset - Episode {self.episode}")

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
            'x': self.vehicle.x,
            'y': self.vehicle.y,
            'heading': self.vehicle.heading,
            'velocity': self.vehicle.velocity,
            'steering_angle': self.vehicle.steering_angle,
            'on_road': segment is not None,
            'lane': lane,
            'distance_traveled': self.vehicle.distance_traveled,
            'steps': self.steps,
            'time': self.time,
            'done': self.done
        }

    def render(self):
        """Render game (only in human mode)"""
        if self.render_mode == 'human' and self.screen is not None:
            self.renderer.render(self.screen, self.vehicle, self.road_network, self.camera)
            pygame.display.flip()

    def close(self):
        """Cleanup resources"""
        self.stop_physics_loop()
        if self.screen is not None:
            pygame.quit()
```

---

## 💥 Collision Detection (`game/collision.py`)

```python
"""Collision detection for all phases"""

import math
from . import config

class CollisionDetector:
    """Detects crashes and dangerous situations"""

    def __init__(self, phase):
        self.phase = phase

    def check_crashes(self, vehicle, road_network, **kwargs):
        """Check all crash conditions for current phase

        Returns:
            Dict with crash info: {'terminal': bool, 'reason': str, ...}
        """
        if self.phase == 1:
            return self._check_phase1(vehicle, road_network)
        elif self.phase == 2:
            return self._check_phase2(vehicle, road_network, kwargs.get('goal'))
        elif self.phase == 3:
            return self._check_phase3(vehicle, road_network, kwargs.get('goal'), kwargs.get('obstacles'))

    def _check_phase1(self, vehicle, road_network):
        """Phase 1: Basic driving crashes"""
        crashes = {
            'terminal': False,
            'reason': None,
            'off_road': False,
            'wrong_lane': False,
            'out_of_bounds': False
        }

        # 1. Check if on road
        segment, lane = road_network.get_segment_and_lane(vehicle.x, vehicle.y)

        if segment is None:
            crashes['off_road'] = True
            crashes['terminal'] = True
            crashes['reason'] = 'Vehicle drove off road'
            return crashes

        # 2. Check if in correct lane
        if lane != vehicle.assigned_lane:
            crashes['wrong_lane'] = True
            # Not terminal, just penalty

        # 3. Check out of bounds
        margin = 200
        if (vehicle.x < -margin or vehicle.x > config.WINDOW_WIDTH + margin or
            vehicle.y < -margin or vehicle.y > config.WINDOW_HEIGHT + margin):
            crashes['out_of_bounds'] = True
            crashes['terminal'] = True
            crashes['reason'] = 'Vehicle out of bounds'
            return crashes

        return crashes

    def _check_phase2(self, vehicle, road_network, goal):
        """Phase 2: Navigation crashes (includes Phase 1)"""
        crashes = self._check_phase1(vehicle, road_network)
        if crashes['terminal']:
            return crashes

        # TODO: Implement Phase 2 specific crashes
        # - Wrong turn at intersection
        # - Driving opposite direction

        return crashes

    def _check_phase3(self, vehicle, road_network, goal, obstacles):
        """Phase 3: Obstacle crashes (includes Phase 1 & 2)"""
        crashes = self._check_phase2(vehicle, road_network, goal)
        if crashes['terminal']:
            return crashes

        # Check obstacle collisions
        if obstacles:
            for obstacle in obstacles:
                if self._vehicle_obstacle_collision(vehicle, obstacle):
                    crashes['obstacle_collision'] = True
                    crashes['terminal'] = True
                    crashes['reason'] = f'Collision with {obstacle.type}'
                    crashes['obstacle'] = obstacle
                    return crashes

                # Check proximity
                dist = math.hypot(vehicle.x - obstacle.x, vehicle.y - obstacle.y)
                if dist < config.SAFE_DISTANCE_THRESHOLD:
                    crashes['too_close_to_obstacle'] = True

        return crashes

    def _vehicle_obstacle_collision(self, vehicle, obstacle):
        """Check if vehicle collides with obstacle using rectangle intersection"""
        # Get vehicle corners
        vehicle_corners = vehicle.get_corners()

        # Get obstacle corners
        obs_corners = [
            (obstacle.x - obstacle.width/2, obstacle.y - obstacle.height/2),
            (obstacle.x + obstacle.width/2, obstacle.y - obstacle.height/2),
            (obstacle.x + obstacle.width/2, obstacle.y + obstacle.height/2),
            (obstacle.x - obstacle.width/2, obstacle.y + obstacle.height/2),
        ]

        # Simplified: Check if any vehicle corner is inside obstacle bounding box
        for vx, vy in vehicle_corners:
            if (obstacle.x - obstacle.width/2 <= vx <= obstacle.x + obstacle.width/2 and
                obstacle.y - obstacle.height/2 <= vy <= obstacle.y + obstacle.height/2):
                return True

        # Check if any obstacle corner is inside vehicle
        # (Use separating axis theorem for precise collision)

        return False
```

---

## 🎁 Reward Calculation (`game/rewards.py`)

```python
"""Reward calculation for each phase"""

import math
from . import config

class RewardCalculator:
    """Calculate rewards based on phase"""

    def __init__(self, phase):
        self.phase = phase

    def calculate_reward(self, vehicle, road_network, crashes, prev_state, **kwargs):
        """Calculate reward for current timestep

        Returns:
            Float reward value
        """
        if self.phase == 1:
            return self._calculate_phase1(vehicle, road_network, crashes, prev_state)
        elif self.phase == 2:
            return self._calculate_phase2(vehicle, road_network, crashes, prev_state, kwargs.get('goal'))
        elif self.phase == 3:
            return self._calculate_phase3(vehicle, road_network, crashes, prev_state,
                                         kwargs.get('goal'), kwargs.get('obstacles'))

    def _calculate_phase1(self, vehicle, road_network, crashes, prev_state):
        """Phase 1: Basic driving rewards"""
        reward = 0.0

        # Terminal penalties
        if crashes['terminal']:
            if crashes['off_road']:
                return config.CRASH_PENALTY_OFF_ROAD
            elif crashes['out_of_bounds']:
                return config.CRASH_PENALTY_OUT_OF_BOUNDS

        # Positive: On road
        segment, lane = road_network.get_segment_and_lane(vehicle.x, vehicle.y)
        if segment is not None:
            reward += config.REWARD_ON_ROAD
        else:
            reward += config.PENALTY_OFF_ROAD

        # Penalty: Wrong lane
        if crashes['wrong_lane']:
            reward += config.PENALTY_CROSSED_LANE

        # Reward: Forward progress
        if prev_state:
            distance_progress = vehicle.distance_traveled - prev_state['distance_traveled']
            reward += config.REWARD_FORWARD_PROGRESS * distance_progress

        # Penalty: Jerky steering
        reward += config.PENALTY_JERKY_STEERING * abs(vehicle.steering_angle)

        # Penalty: Speed error (encourage optimal speed)
        optimal_speed = 100  # pixels/sec
        speed_error = abs(vehicle.velocity - optimal_speed) / optimal_speed
        reward += config.PENALTY_SPEED_ERROR * speed_error

        return reward

    def _calculate_phase2(self, vehicle, road_network, crashes, prev_state, goal):
        """Phase 2: Navigation rewards (includes Phase 1)"""
        reward = self._calculate_phase1(vehicle, road_network, crashes, prev_state)

        # Goal reached
        if goal and goal.is_reached(vehicle):
            reward += config.REWARD_GOAL_REACHED

        # Progress toward goal
        if goal and prev_state:
            curr_dist = math.hypot(vehicle.x - goal.x, vehicle.y - goal.y)
            prev_dist = math.hypot(prev_state['x'] - goal.x, prev_state['y'] - goal.y)
            progress = prev_dist - curr_dist
            reward += config.REWARD_PROGRESS_TO_GOAL * progress

        # Time penalty
        reward += config.PENALTY_TIME_STEP

        return reward

    def _calculate_phase3(self, vehicle, road_network, crashes, prev_state, goal, obstacles):
        """Phase 3: Obstacle rewards (includes Phase 1 & 2)"""
        reward = self._calculate_phase2(vehicle, road_network, crashes, prev_state, goal)

        # Obstacle collision
        if crashes.get('obstacle_collision'):
            return config.CRASH_PENALTY_OBSTACLE

        # Too close to obstacle
        if crashes.get('too_close_to_obstacle'):
            reward += config.PENALTY_TOO_CLOSE_OBSTACLE

        return reward
```

---

## 🔌 Socket Server with Continuous Loop (`game/socket_server.py`)

Similar structure to delivery_drone but with key differences:

```python
"""Socket server for remote game control with continuous physics"""

import socket
import json
import threading
import queue
import time
from typing import List

class DrivingGameSocketServer:
    """Socket server that runs game at fixed FPS"""

    def __init__(self, games, host='0.0.0.0', port=5555):
        """Initialize socket server

        Args:
            games: List of DrivingGame instances
        """
        self.games = games if isinstance(games, list) else [games]
        self.num_games = len(self.games)
        self.host = host
        self.port = port

        # Socket
        self.server_socket = None
        self.client_socket = None
        self.connected = False

        # Network thread
        self.running = False
        self.network_thread = None

        # Message buffer
        self.send_queue = queue.Queue()

    def start(self):
        """Start server and game physics loops"""
        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"Server listening on {self.host}:{self.port}")

        # Accept connection
        self.client_socket, addr = self.server_socket.accept()
        self.connected = True
        print(f"Client connected from {addr}")

        # Send handshake
        self._send_message({'type': 'HANDSHAKE', 'num_games': self.num_games})

        # Start network thread
        self.running = True
        self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.network_thread.start()

        # Start physics loops for all games
        for game in self.games:
            game.start_physics_loop()

        print(f"Started physics loops for {self.num_games} game(s)")

    def _network_loop(self):
        """Handle incoming messages and send responses"""
        buffer = ""

        while self.running:
            try:
                # Receive data
                data = self.client_socket.recv(4096).decode('utf-8')
                if not data:
                    print("Client disconnected")
                    self.connected = False
                    break

                buffer += data

                # Process complete messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    message = json.loads(line)

                    # Handle message
                    response = self._handle_message(message)
                    if response:
                        self._send_message(response)

            except Exception as e:
                print(f"Network error: {e}")
                self.connected = False
                break

    def _handle_message(self, message):
        """Handle client message

        Returns:
            Response dict or None
        """
        msg_type = message.get('type')
        game_id = message.get('game_id', 0)

        if game_id >= self.num_games:
            return {'type': 'ERROR', 'message': f'Invalid game_id {game_id}'}

        game = self.games[game_id]

        if msg_type == 'RESET':
            obs = game.reset()
            state = game.get_state()
            return {
                'type': 'STATE',
                'observation': obs.tolist(),
                'state': state,
                'reward': 0.0,
                'done': False,
                'info': {'episode': game.episode}
            }

        elif msg_type == 'STEP':
            # Update action buffer
            action = message.get('action', {})
            game.set_action(action)

            # Get latest observation (game is running in background)
            obs = game.get_observation()
            state = game.get_state()
            reward = game.last_reward
            done = game.done

            return {
                'type': 'STATE',
                'observation': obs.tolist(),
                'state': state,
                'reward': reward,
                'done': done,
                'info': {'steps': game.steps, 'total_reward': game.total_reward}
            }

        elif msg_type == 'GET_STATE':
            obs = game.get_observation()
            state = game.get_state()
            return {
                'type': 'STATE',
                'observation': obs.tolist(),
                'state': state,
                'reward': game.last_reward,
                'done': game.done,
                'info': {}
            }

        elif msg_type == 'CLOSE':
            self.stop()
            return None

        else:
            return {'type': 'ERROR', 'message': f'Unknown message type: {msg_type}'}

    def _send_message(self, message):
        """Send JSON message to client"""
        try:
            data = json.dumps(message) + '\n'
            self.client_socket.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False

    def stop(self):
        """Stop server and game loops"""
        self.running = False

        # Stop all game physics loops
        for game in self.games:
            game.stop_physics_loop()

        if self.network_thread:
            self.network_thread.join(timeout=2.0)

        if self.client_socket:
            self.client_socket.close()

        if self.server_socket:
            self.server_socket.close()

        print("Server stopped")
```

---

## 🎮 Main Server Script (`socket_server.py`)

```python
#!/usr/bin/env python3
"""Socket server for Visual Driving 2D

Usage:
    # Phase 1, single game, visual
    python socket_server.py --phase 1 --render human

    # Phase 3, 6 parallel games, headless
    python socket_server.py --phase 3 --num-games 6 --render none \\
        --randomize-spawn --randomize-obstacles --min-obstacles 5 --max-obstacles 10
"""

import argparse
import signal
import sys
from game.game_engine import DrivingGame
from game.socket_server import DrivingGameSocketServer
from game import config

def main():
    parser = argparse.ArgumentParser(description='Visual Driving 2D Socket Server')

    # Phase selection
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                       help='Game phase (1=basic, 2=navigation, 3=obstacles)')

    # Server config
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port to listen on')
    parser.add_argument('--num-games', type=int, default=1,
                       help='Number of parallel game instances')

    # Rendering
    parser.add_argument('--render', type=str, default='human',
                       choices=['human', 'rgb_array', 'none'],
                       help='Render mode')
    parser.add_argument('--fps', type=int, default=60,
                       help='Physics FPS')

    # Feature toggles - Phase 1
    parser.add_argument('--enable-curves', action='store_true', default=True,
                       help='Enable curved road segments')
    parser.add_argument('--enable-multiple-lanes', action='store_true', default=True,
                       help='Enable multiple lanes')

    # Feature toggles - Phase 2
    parser.add_argument('--enable-intersections', action='store_true',
                       help='Enable intersections (auto-enabled for phase 2+)')
    parser.add_argument('--enable-goal', action='store_true',
                       help='Enable goal navigation (auto-enabled for phase 2+)')

    # Feature toggles - Phase 3
    parser.add_argument('--enable-obstacles', action='store_true',
                       help='Enable obstacles (auto-enabled for phase 3)')
    parser.add_argument('--enable-parked-cars', action='store_true', default=True,
                       help='Enable parked car obstacles')
    parser.add_argument('--enable-barriers', action='store_true', default=True,
                       help='Enable barrier obstacles')
    parser.add_argument('--enable-cones', action='store_true', default=True,
                       help='Enable cone obstacles')

    # Randomization - Spawn
    parser.add_argument('--randomize-spawn', action='store_true', default=True,
                       help='Randomize vehicle spawn position')
    parser.add_argument('--randomize-heading', action='store_true', default=True,
                       help='Randomize vehicle spawn heading')

    # Randomization - Roads
    parser.add_argument('--randomize-road-layout', action='store_true', default=True,
                       help='Randomize road network layout')
    parser.add_argument('--randomize-curvature', action='store_true', default=True,
                       help='Randomize road curvature')
    parser.add_argument('--randomize-lane-width', action='store_true', default=True,
                       help='Randomize lane widths')

    # Randomization - Phase 2
    parser.add_argument('--randomize-goal', action='store_true', default=True,
                       help='Randomize goal position')

    # Randomization - Phase 3
    parser.add_argument('--randomize-obstacles', action='store_true', default=True,
                       help='Randomize obstacle count and positions')
    parser.add_argument('--min-obstacles', type=int, default=3,
                       help='Minimum number of obstacles')
    parser.add_argument('--max-obstacles', type=int, default=10,
                       help='Maximum number of obstacles')

    args = parser.parse_args()

    # Apply config overrides
    config.PHASE = args.phase
    config.PHYSICS_FPS = args.fps
    config.ENABLE_CURVES = args.enable_curves
    config.ENABLE_MULTIPLE_LANES = args.enable_multiple_lanes

    # Auto-enable features based on phase
    if args.phase >= 2:
        config.ENABLE_INTERSECTIONS = True
        config.ENABLE_GOAL_NAVIGATION = True

    if args.phase >= 3:
        config.ENABLE_STATIC_OBSTACLES = True
        config.ENABLE_PARKED_CARS = args.enable_parked_cars
        config.ENABLE_BARRIERS = args.enable_barriers
        config.ENABLE_CONES = args.enable_cones

    # Randomization
    config.RANDOMIZE_SPAWN_POSITION = args.randomize_spawn
    config.RANDOMIZE_SPAWN_HEADING = args.randomize_heading
    config.RANDOMIZE_ROAD_LAYOUT = args.randomize_road_layout
    config.RANDOMIZE_ROAD_CURVATURE = args.randomize_curvature
    config.RANDOMIZE_LANE_WIDTH = args.randomize_lane_width
    config.RANDOMIZE_GOAL_POSITION = args.randomize_goal
    config.RANDOMIZE_OBSTACLE_COUNT = args.randomize_obstacles
    config.MIN_OBSTACLES = args.min_obstacles
    config.MAX_OBSTACLES = args.max_obstacles

    # Create games
    render_mode = None if args.render == 'none' else args.render

    print(f"Creating {args.num_games} game instance(s) - Phase {args.phase}")
    games = []
    for i in range(args.num_games):
        # Only render first game in human mode
        game_render = render_mode if (i == 0 and render_mode == 'human') else None
        game = DrivingGame(phase=args.phase, render_mode=game_render)
        games.append(game)

    # Create server
    server = DrivingGameSocketServer(games, host=args.host, port=args.port)

    # Signal handling
    def signal_handler(sig, frame):
        print("\nShutting down...")
        server.stop()
        for game in games:
            game.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start server
    server.start()

    print("Server running. Press Ctrl+C to stop.")

    # Keep main thread alive
    try:
        while server.connected:
            if render_mode == 'human' and games[0]:
                games[0].render()
            time.sleep(0.016)  # ~60 FPS render
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        for game in games:
            game.close()

if __name__ == '__main__':
    main()
```

---

## 📋 Crash Summary by Phase

| Phase | Crash Type | Trigger | Terminal? | Penalty | Detection |
|-------|-----------|---------|-----------|---------|-----------|
| **Phase 1** |
| 1 | Off Road | Vehicle leaves road surface | Yes | -50 | Point-in-polygon test |
| 1 | Wrong Lane | Crossed into adjacent lane | No | -0.1/step | Lane index comparison |
| 1 | Out of Bounds | Vehicle too far from world | Yes | -50 | Position bounds check |
| **Phase 2** |
| 2 | Wrong Turn | Turned away from goal at intersection | Yes | -50 | Intersection navigation logic |
| 2 | Timeout | Didn't reach goal in time limit | Yes | -20 | Step counter |
| 2 | Opposite Direction | Driving wrong way on one-way road | Yes | -50 | Heading vs road direction |
| **Phase 3** |
| 3 | Obstacle Collision | Hit parked car/barrier/cone/debris | Yes | -100 | Rectangle intersection |
| 3 | Too Close Warning | Dangerously close to obstacle | No | -0.5/step | Distance threshold |

---

## 🎲 Randomization Summary

### What Gets Randomized Per Phase

**Phase 1:**
- ✅ Spawn position (x, y) within road
- ✅ Spawn heading (±30° from road direction)
- ✅ Spawn lane (random lane assignment)
- ✅ Road curvature (straight to curved sections)
- ✅ Road segment lengths (500-2000 pixels)
- ✅ Lane widths (60-100 pixels)
- ✅ Number of lanes per road (1-3)
- ✅ Road circuit layout (different configurations)

**Phase 2 (adds):**
- ✅ Goal position (on valid road, far from spawn)
- ✅ Intersection positions (grid with variation)
- ✅ Intersection types (T, 4-way, varied)
- ✅ Road connections (which roads meet at intersections)
- ✅ Route complexity (number of turns to goal)
- ✅ Intersection spacing (300-600 pixels)

**Phase 3 (adds):**
- ✅ Number of obstacles (3-15)
- ✅ Obstacle positions (on road, avoiding spawn/goal)
- ✅ Obstacle types (parked_car, barrier, cone, debris)
- ✅ Obstacle sizes (±30% variance within type)
- ✅ Obstacle orientations (random angles)
- ✅ Obstacle density (clustered vs spread)
- ✅ Obstacle spacing (minimum 150 pixels apart)

---

## 🧪 Testing Strategy

### Unit Tests
```
tests/
├── test_vehicle_physics.py   # Bicycle model correctness
├── test_road_network.py       # Point-in-polygon, lane detection
├── test_collision.py          # All crash conditions
├── test_rewards.py            # Reward calculation logic
└── test_socket.py             # Client/server communication
```

### Integration Tests
1. **Manual Play Mode** (`manual_play.py`) - Human keyboard control
2. **Random Agent** - Collects crash statistics
3. **Socket Performance** - Verify 3000+ samples/sec with 6 parallel games
4. **Visualization Tools** - Plot trajectories, crashes, rewards

---

## 📅 Implementation Order

### Week 1: Phase 1 Core
- [ ] Day 1-2: `config.py`, `vehicle.py`, `road.py` (simple circuit)
- [ ] Day 3: `game_engine.py` (continuous loop)
- [ ] Day 4: `renderer.py`, `camera.py`, `collision.py`
- [ ] Day 5: `socket_server.py`, `socket_client.py`
- [ ] Day 6-7: Testing, manual play mode, bug fixes

### Week 2: Phase 1 Randomization + Phase 2 Core
- [ ] Day 1: Spawn randomization, road randomization
- [ ] Day 2: Test random agent statistics
- [ ] Day 3-4: Intersection system, goal system
- [ ] Day 5: Phase 2 collision detection, rewards
- [ ] Day 6-7: Testing Phase 2

### Week 3: Phase 2 Randomization + Phase 3 Core
- [ ] Day 1: Goal/intersection randomization
- [ ] Day 2: Test Phase 2 with random agent
- [ ] Day 3-4: Obstacle system, spawning logic
- [ ] Day 5: Phase 3 collision detection
- [ ] Day 6-7: Testing Phase 3

### Week 4: Polish & Integration
- [ ] Day 1-2: Train actual CNN policy on Phase 1
- [ ] Day 3: Visual debugging tools
- [ ] Day 4: Socket performance optimization
- [ ] Day 5: Documentation
- [ ] Day 6-7: Final testing, video demos

---

## ✅ Success Criteria

### Phase 1
- [ ] Vehicle physics feels responsive (manual play test)
- [ ] Random agent crashes <50% of episodes
- [ ] Socket server handles 6 parallel games at 60 FPS each
- [ ] CNN policy can learn to stay on road after 100k samples

### Phase 2
- [ ] Random agent reaches goal >10% of episodes
- [ ] Navigation logic works for all intersection types
- [ ] CNN policy can learn to reach goal after 500k samples

### Phase 3
- [ ] Random agent survives obstacles >5% of episodes
- [ ] Collision detection has no false positives
- [ ] CNN policy can navigate with obstacles after 1M samples

---

**This plan is complete and ready for implementation approval!**
