"""
Configuration for Visual Driving 2D Car Racing Game
"""

# Window and rendering settings
STATE_W = 96        # State observation width (less than Atari 160x192)
STATE_H = 96        # State observation height
VIDEO_W = 600       # Video recording width
VIDEO_H = 400       # Video recording height
WINDOW_W = 1000     # Display window width
WINDOW_H = 800      # Display window height

# Game physics and scale
SCALE = 6.0         # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50            # Frames per second
ZOOM = 2.7          # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

# Track generation parameters
TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

# Colors
ROAD_COLOR = [0.4, 0.4, 0.4]
GRASS_COLOR = [0.4, 0.8, 0.4]
GRASS_LIGHT_COLOR = [0.4, 0.9, 0.4]

# Reward settings
REWARD_PER_TILE = 1000.0  # Total reward for visiting all tiles
REWARD_PER_FRAME = -0.1   # Time penalty per frame
REWARD_OUT_OF_BOUNDS = -100  # Penalty for going off-track
REWARD_COMPLETE_TRACK = 0  # Bonus for completing (already in tile rewards)

# Episode settings
DEFAULT_EPISODE_STEPS = 1000  # Maximum steps per episode
TRACK_CHECKPOINTS = 12  # Number of checkpoints for track generation

# Socket server settings
SOCKET_HOST = '0.0.0.0'
SOCKET_PORT = 5555
SOCKET_BUFFER_SIZE = 8192
SOCKET_TIMEOUT = 30.0

# Car physics (from Box2D car_dynamics)
# These will be used by the Car class
ENGINE_POWER = 100000000 * SCALE * SCALE
WHEEL_MOMENT_OF_INERTIA = 4000 * SCALE * SCALE
FRICTION_LIMIT = 1000000 * SCALE * SCALE
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [(+25, +20), (+50, -10), (+50, -40), (+20, -90), (-20, -90), (-50, -40), (-50, -10), (-25, +20)]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]

# Car colors
CAR_COLORS = [0.8, 0.0, 0.0]  # Red car
WHEEL_COLOR = [0.0, 0.0, 0.0]  # Black wheels
WHEEL_WHITE = [0.3, 0.3, 0.3]  # Wheel highlights
MUD_COLOR = [0.4, 0.4, 0.0, 0.0]  # Mud particles
