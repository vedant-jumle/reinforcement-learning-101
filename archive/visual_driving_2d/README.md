# Visual Driving 2D - Car Racing Game

A top-down car racing game designed for reinforcement learning research. Features realistic Box2D physics, procedural track generation, and socket-based distributed training.

**Key Features:**
- ✅ Realistic car physics (Box2D engine)
- ✅ Procedural random track generation
- ✅ Visual observations (96x96 RGB) + telemetry
- ✅ Socket interface for distributed RL training
- ✅ Free-running game loop (50 FPS, non-blocking)
- ✅ Multiple parallel game instances
- ✅ Manual play mode for understanding the task

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `gymnasium>=0.29.0` - RL environment interface
- `Box2D>=2.3.10` - Physics engine
- `pyglet>=1.5.0` - OpenGL rendering
- `numpy>=1.24.0` - Numerical computation
- `pillow>=9.0.0` - Image processing

## Quick Start

### 1. Play Manually

Understand the task by playing with keyboard controls:

```bash
python manual_play.py
```

**Controls:**
- **Arrow Keys / WASD**: Steer, Gas, Brake
  - LEFT/A: Steer left
  - RIGHT/D: Steer right
  - UP/W: Gas
  - DOWN/S: Brake
- **R**: Reset episode
- **ESC**: Quit

**Tip:** This is a powerful rear-wheel drive car! Don't accelerate and turn at the same time.

### 2. Start Training Server

Run game server with socket interface:

```bash
# Single game instance (visual)
python socket_server.py

# Multiple parallel games (headless, for training)
python socket_server.py --num-games 6 --render none

# Custom port
python socket_server.py --port 5556

# Visual mode with multiple games (render only first)
python socket_server.py --num-games 4 --render human
```

**Server Options:**
- `--host HOST`: Server bind address (default: 0.0.0.0)
- `--port PORT`: Server port (default: 5555)
- `--num-games N`: Number of parallel game instances (default: 1)
- `--render MODE`: Render mode - `human`, `rgb_array`, `none` (default: human)
- `--render-all`: Render all game instances (only with --num-games > 1)
- `--max-steps N`: Maximum steps per episode (default: 1000)
- `--seed N`: Random seed for track generation
- `--verbose {0,1}`: Print debug info (default: 1)

### 3. Connect Client for Training

```python
from game.socket_client import RacingGameClient

# Connect to server
with RacingGameClient(host='localhost', port=5555) as client:
    print(f"Connected to {client.num_games} game(s)")

    # Reset
    telemetry, frame, reward, done, info = client.reset(game_id=0)

    # Training loop
    for step in range(1000):
        # Your RL agent decides action
        action = {
            'steer': 0.5,   # [-1, +1]
            'gas': 1.0,     # [0, +1]
            'brake': 0.0    # [0, +1]
        }

        # Send action (non-blocking)
        client.set_action(action, game_id=0)

        # Get state (on-demand)
        telemetry, frame, reward, done, info = client.get_state(game_id=0)

        # Use frame (96x96x3 numpy array) for CNN
        # Use telemetry (dict) for MLP

        if done:
            client.reset(game_id=0)
```

See [SOCKET_API.md](SOCKET_API.md) for complete protocol documentation.

## Game Mechanics

### Objective

Drive a car around a procedurally generated racing track, visiting all track tiles as quickly as possible.

### Observations

**Visual:**
- RGB image: 96×96×3 pixels (uint8)
- Top-down view of car and track
- Indicators at bottom: speed, wheel sensors, steering, gyroscope

**Telemetry:**
```python
{
    'car_x': float,              # Position (world coordinates)
    'car_y': float,
    'car_vx': float,             # Velocity
    'car_vy': float,
    'car_angle': float,          # Heading (radians)
    'car_angular_vel': float,    # Angular velocity
    'speed': float,              # Total speed magnitude
    'wheel_speeds': [w1, w2, w3, w4],  # Wheel angular velocities
    'tiles_visited': int,        # Progress
    'total_tiles': int,
    'steps': int
}
```

### Actions

Continuous control (3 dimensions):

```python
action = {
    'steer': float,   # [-1.0, +1.0] - left/right steering
    'gas': float,     # [0.0, +1.0] - throttle
    'brake': float    # [0.0, +1.0] - brake pressure
}
```

### Rewards

- **+1000/N** for each new track tile visited (N = total tiles)
- **-0.1** per frame (time penalty)
- **-100** for going out of bounds
- **Episode completes** when all tiles visited or max steps reached

**Solved:** Consistently score 900+ points

**Example:** Finish in 732 frames = 1000 - 0.1×732 = 926.8 points

### Physics

- **Box2D engine**: Realistic car dynamics with friction, tire slip
- **4-wheel model**: Independent wheel physics with ABS sensors
- **Rear-wheel drive**: Powerful but requires skill to control
- **Track generation**: Random procedural tracks (12 checkpoints, varying curvature)

## Architecture

### Free-Running Game Loop

Unlike the delivery drone game, the car racing server runs continuously at 50 FPS:

```
Server Game Loop (50 FPS):
  1. Process network commands (non-blocking)
  2. Apply latest action to each game
  3. Step physics (Box2D)
  4. Send state if requested (on-demand)
  5. Auto-reset if episode done
  6. Render (if enabled)
  7. Sleep to maintain 50 FPS
```

**Benefits:**
- Realistic timing (50 FPS consistent)
- Non-blocking training
- Multiple games in parallel
- Client can request state whenever needed

### Directory Structure

```
visual_driving_2d/
├── game/
│   ├── __init__.py
│   ├── config.py           # Game constants
│   ├── car_dynamics.py     # Car physics (Box2D)
│   ├── game_engine.py      # Main game class
│   ├── socket_server.py    # Async socket server
│   └── socket_client.py    # Client with frame decoding
│
├── reference/
│   └── car_racing.py       # Original OpenAI Gym reference
│
├── socket_server.py        # Main server script
├── manual_play.py          # Keyboard control
├── requirements.txt        # Dependencies
├── SOCKET_API.md          # Protocol documentation
└── README.md              # This file
```

## RL Training Tips

### 1. Start with Telemetry

Visual RL is hard! Start with telemetry-only:
- Use `telemetry` dict as input to MLP
- Ignore `frame` initially
- Get basic driving working first

### 2. Then Add Visual

Once telemetry works, add CNN:
- Stack 4 consecutive frames (temporal info)
- Use CNN to extract features
- Combine CNN features with telemetry

### 3. Reward Shaping

Default rewards work, but you can customize:
- Lane keeping bonus
- Smoothness penalty (jerky steering)
- Speed targets
- Collision avoidance

### 4. Curriculum Learning

Start easy, gradually increase difficulty:
1. Simple tracks (low curvature)
2. Longer episodes (more steps)
3. Complex tracks (high curvature)
4. Multiple laps

### 5. Parallel Training

Use multiple game instances for sample efficiency:

```python
# Server: 6 parallel games
python socket_server.py --num-games 6 --render none

# Client: Collect from all games
for game_id in range(6):
    client.set_action(actions[game_id], game_id)
    states[game_id] = client.get_state(game_id)
```

## Performance

**Latency** (localhost):
- Mean: 2-5 ms per state request
- Action update: <1 ms (non-blocking)
- Frame encoding: ~1-2 ms

**Throughput:**
- Single game: ~200-300 FPS training
- 6 parallel games: ~50-60 FPS per game

**Optimization:**
- Use headless mode (`--render none`)
- Run on same machine (localhost)
- Batch state requests for multiple games

## Differences from Delivery Drone

| Feature | Drone Game | Car Racing |
|---------|------------|------------|
| **Physics** | Custom (gravity, thrust) | Box2D (realistic car) |
| **Observation** | 15D state vector | 96×96×3 RGB + telemetry |
| **Game Loop** | Blocking (waits for action) | Free-running (50 FPS) |
| **Actions** | 3 binary thrusters | 3 continuous controls |
| **Task** | Land on platform | Complete racing track |
| **Difficulty** | Moderate | Hard |
| **Network** | MLP sufficient | CNN required (visual) |

## Troubleshooting

**ImportError: No module named 'Box2D'**
```bash
pip install Box2D
```

**ImportError: No module named 'pyglet'**
```bash
pip install pyglet
```

**Connection Refused**
- Ensure server is running first
- Check firewall settings
- Verify correct port (default: 5555)

**Game Runs Slow**
- Use headless mode: `--render none`
- Reduce number of games: `--num-games 1`
- Check CPU usage (Box2D is CPU-intensive)

**Frame Decode Errors**
- Verify server has `render_mode != none`
- Check frame_shape matches: [96, 96, 3]
- Update numpy if issues persist

## References

- Original CarRacing environment: [OpenAI Gym](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)
- Box2D physics: [pybox2d.readthedocs.io](https://pybox2d.readthedocs.io/)
- RL algorithms: See `../Policy_Gradients_Baseline.ipynb`, `../Actor_Critic_Basic.ipynb`

## License

MIT License - See [../LICENSE](../LICENSE) for details

## Credits

Adapted from OpenAI Gym's CarRacing environment.
Created by Oleg Klimov. Licensed on the same terms as OpenAI Gym.
Modified for socket-based distributed RL training.

---

**Ready to train?** Start the server and see [SOCKET_API.md](SOCKET_API.md) for training examples!
