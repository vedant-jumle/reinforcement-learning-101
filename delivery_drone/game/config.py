"""Game configuration and constants"""

# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors (RGB)
COLOR_SKY = (135, 206, 235)
COLOR_GROUND = (101, 67, 33)
COLOR_PLATFORM = (50, 205, 50)
COLOR_DRONE = (200, 200, 200)
COLOR_THRUST = (255, 100, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_HUD_BG = (0, 0, 0, 128)

# Physics constants
GRAVITY = 0.3  # Pixels per frame^2
DRAG = 0.99  # Velocity multiplier per frame
ANGULAR_DRAG = 0.95

# Drone settings
DRONE_WIDTH = 40
DRONE_HEIGHT = 20
MAIN_THRUST_POWER = 0.6  # Upward thrust strength
SIDE_THRUST_POWER = 0.3  # Rotational thrust strength
MAX_FUEL = 1000.0
FUEL_CONSUMPTION_MAIN = 2.0  # Fuel per frame when main thrust active
FUEL_CONSUMPTION_SIDE = 1.0  # Fuel per frame when side thrust active

# Platform settings
PLATFORM_WIDTH = 100
PLATFORM_HEIGHT = 20
PLATFORM_Y = WINDOW_HEIGHT - 100  # Default height from top (for backward compatibility)
PLATFORM_Y_MIN = 350  # Minimum Y position (higher up, easier)
PLATFORM_Y_MAX = 550  # Maximum Y position (lower down, harder)

# Landing conditions
MAX_LANDING_VELOCITY = 3.0  # Max speed to land safely
MAX_LANDING_ANGLE = 20.0  # Max tilt angle (degrees) to land safely

# Game bounds
WORLD_WIDTH = WINDOW_WIDTH
WORLD_HEIGHT = WINDOW_HEIGHT
OUT_OF_BOUNDS_MARGIN = 50  # Pixels outside screen before failure

# Difficulty settings
WIND_ENABLED = False
WIND_STRENGTH = 0.1
PLATFORM_MOVING = False
PLATFORM_SPEED = 1.0

# Reward settings (for RL interface)
REWARD_LANDING = 100.0
REWARD_CRASH = -100.0
REWARD_OUT_OF_FUEL = -50.0
REWARD_OUT_OF_BOUNDS = -50.0
REWARD_STEP = -0.1  # Small penalty per step (encourages efficiency)

# Starting position
DRONE_START_X = WINDOW_WIDTH // 2
DRONE_START_Y = 100

# Drone spawn randomization ranges
DRONE_START_X_MIN = 100  # Minimum X spawn position
DRONE_START_X_MAX = 700  # Maximum X spawn position
DRONE_START_Y_MIN = 50   # Minimum Y spawn position (higher up)
DRONE_START_Y_MAX = 250  # Maximum Y spawn position (lower down)

# Socket server settings
SOCKET_HOST = '0.0.0.0'  # Listen on all interfaces (use 'localhost' for local only)
SOCKET_PORT = 5555
SOCKET_BUFFER_SIZE = 4096
SOCKET_TIMEOUT = 30.0  # Seconds
