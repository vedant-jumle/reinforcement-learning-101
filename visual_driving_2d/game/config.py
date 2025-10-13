"""Game configuration and feature flags"""

# ============================================================
# PHASE SELECTION
# ============================================================
PHASE = 1  # Current phase (1, 2, or 3)

# ============================================================
# FEATURE FLAGS - PHASE 1: BASIC DRIVING
# ============================================================
ENABLE_LANE_BOUNDARIES = True      # Road has edges
ENABLE_CURVES = False              # Roads can curve (DISABLED - curves are broken)
ENABLE_LANE_MARKINGS = False        # Visual lane lines
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
RANDOMIZE_ROAD_CURVATURE = False   # DISABLED - curves not supported
RANDOMIZE_ROAD_LENGTH = True       # Length of road segments
RANDOMIZE_LANE_WIDTH = True        # Width of lanes
RANDOMIZE_NUM_LANES = True         # 1-4 lanes

# Road generation ranges
ROAD_CURVATURE_RANGE = (0.0, 0.0)  # ALWAYS 0 - no curves supported
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
MAX_SPEED_FORWARD = 500
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
COLOR_HUD_TEXT = (255, 255, 255)    # White
COLOR_HUD_BG = (0, 0, 0, 180)       # Semi-transparent black
