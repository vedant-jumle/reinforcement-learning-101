# Delivery Drone Game

A physics-based drone delivery game where you control a drone to land on platforms. Built as a clean, simple environment for learning reinforcement learning concepts.

## Features

✅ **Playable Game** - Fun physics-based gameplay with keyboard controls
✅ **Clean API** - Simple interface for building AI/RL agents
✅ **Socket Interface** - Remote control over TCP sockets for distributed training
✅ **State Observation** - Full access to game state for decision making
✅ **Example Agents** - Random, rule-based, and remote control examples
✅ **No Dependencies** - Just pygame and numpy, no RL frameworks required

## Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pygame numpy
```

## Quick Start

### 1. Play Manually (Keyboard)

```bash
python manual_play.py
```

### 2. Run Example Agents

```bash
# Random agent
PYTHONPATH=. python examples/random_agent.py

# Simple rule-based agent
PYTHONPATH=. python examples/simple_agent.py

# API demo (no rendering)
PYTHONPATH=. python examples/api_demo.py

# State inspector
PYTHONPATH=. python examples/inspect_state.py

# Record your gameplay for imitation learning
PYTHONPATH=. python examples/record_gameplay.py
```

### 3. Remote Control (Socket Interface)

Control the game from a separate process over TCP sockets:

```bash
# Terminal 1: Start the server
python socket_server.py --render human

# Terminal 2: Run remote agent
PYTHONPATH=. python examples/remote_agent.py

# Benchmark latency
PYTHONPATH=. python examples/benchmark_latency.py
```

See [SOCKET_API.md](SOCKET_API.md) for complete socket documentation.

### 4. Test the Game

```bash
python test_game.py
python test_socket.py  # Test socket interface
```

## Controls (Manual Play)

- **W** or **↑**: Main thrust (upward)
- **A** or **←**: Left thruster (rotate/move left)
- **D** or **→**: Right thruster (rotate/move right)
- **R**: Reset level
- **ESC**: Quit

## Objective

Land your drone safely on the green platform (marked with "H"):
- ✅ Keep your speed low (< 3.0 pixels/frame)
- ✅ Keep the drone upright (< 20° tilt)
- ✅ Land on the platform
- ✅ Don't run out of fuel!

## Building AI Agents

The game provides a clean API for external control - perfect for RL experiments!

### Basic Usage

```python
from game.game_engine import DroneGame

# Create game (render_mode: 'human', 'rgb_array', or None)
game = DroneGame(render_mode='human')

# Reset to start
state = game.reset()

# Game loop
while not done:
    # Your agent decides action
    action = {
        'main_thrust': 1,   # 0 or 1
        'left_thrust': 0,   # 0 or 1
        'right_thrust': 0   # 0 or 1
    }

    # Step the game
    state, reward, done, info = game.step(action)

    # Render (if needed)
    game.render()

# Cleanup
game.close()
```

### State Space

The `state` dictionary contains 15 normalized values:

```python
{
    'drone_x': 0.5,              # Normalized position [0, 1]
    'drone_y': 0.167,
    'drone_vx': 0.0,             # Normalized velocity
    'drone_vy': 0.03,
    'drone_angle': 0.0,          # Normalized angle [-1, 1]
    'drone_angular_vel': 0.0,
    'drone_fuel': 1.0,           # Fuel remaining [0, 1]
    'platform_x': 0.5,
    'platform_y': 0.833,
    'distance_to_platform': 0.5,
    'dx_to_platform': 0.0,
    'dy_to_platform': 0.667,
    'speed': 0.03,               # Total speed magnitude
    'landed': False,             # Success flag
    'crashed': False             # Failure flag
}
```

### Action Space

Actions are binary (0 or 1):

```python
action = {
    'main_thrust': 0,    # Upward thrust (in direction drone is facing)
    'left_thrust': 0,    # Rotate counter-clockwise
    'right_thrust': 0    # Rotate clockwise
}
```

### Rewards

- **+100**: Successful landing
- **-100**: Crash
- **-50**: Out of fuel
- **-50**: Out of bounds
- **-0.1**: Per step (encourages efficiency)
- **+0.0-0.1**: Bonus for getting closer to platform

## Socket Interface (Remote Control)

Control the game remotely over TCP sockets for distributed training!

### Server

```bash
# Start server (visual mode)
python socket_server.py --render human

# Headless mode (faster training)
python socket_server.py --render none

# Custom port
python socket_server.py --port 5556

# Parallel training - multiple game instances
python socket_server.py --num-games 6 --render none

# Spawn randomization
python socket_server.py --randomize-drone --randomize-platform

# Fixed spawns (no randomization)
python socket_server.py --fixed-spawn

# Render all games in grid layout
python socket_server.py --num-games 4 --render human --render-all
```

### Client

```python
from game.socket_client import DroneGameClient

# Connect to server
client = DroneGameClient(host='localhost', port=5555)

# Check number of available game instances
print(f"Server has {client.num_games} game instance(s)")

# Reset and step with game_id parameter
state = client.reset(game_id=0)
state, reward, done, info = client.step(action, game_id=0)

# Parallel episode collection
for game_id in range(client.num_games):
    state = client.reset(game_id)
    # ... collect episode from this game

client.close()
```

### Benefits

- **Distributed training**: Run game on one machine, agent on another
- **Language agnostic**: Any language can implement the client
- **Parallel environments**: Multiple game instances in single server process
- **Low latency**: ~1-3ms per step on localhost
- **Spawn randomization**: Train robust policies with randomized initial conditions

See **[SOCKET_API.md](SOCKET_API.md)** for complete documentation.

### Server Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--host HOST` | Server bind address | `0.0.0.0` |
| `--port PORT` | Server port | `5555` |
| `--render MODE` | Render mode: `human`, `none` | `human` |
| `--num-games N` | Number of parallel game instances | `1` |
| `--render-all` | Render all games in grid (only with `--num-games` > 1) | Only game 0 |
| `--randomize-drone` | Randomize drone spawn position on reset | `False` |
| `--randomize-platform` | Randomize platform position on reset | `True` |
| `--fixed-spawn` | Disable all spawn randomization | `False` |

## Project Structure

```
delivery-drone/
├── game/                      # Core game engine
│   ├── config.py             # Game configuration
│   ├── physics.py            # Physics utilities
│   ├── drone.py              # Drone class with physics
│   ├── platform.py           # Landing platform
│   ├── game_engine.py        # Main game controller + API
│   ├── socket_server.py      # Socket server class
│   └── socket_client.py      # Socket client library
│
├── examples/                  # Example scripts
│   ├── random_agent.py       # Random action agent
│   ├── simple_agent.py       # Rule-based agent
│   ├── remote_agent.py       # Remote control agent
│   ├── benchmark_latency.py  # Socket latency benchmark
│   ├── api_demo.py           # API usage demo
│   ├── inspect_state.py      # State visualization
│   └── record_gameplay.py    # Record human gameplay
│
├── socket_server.py          # Socket server main script
├── manual_play.py            # Play with keyboard
├── test_game.py              # Test suite
├── test_socket.py            # Socket tests
├── requirements.txt          # Dependencies
├── SOCKET_API.md             # Socket documentation
└── README.md                 # This file
```

## Configuration

Edit `game/config.py` to customize:

- Physics parameters (gravity, thrust power, drag)
- Game difficulty (fuel amount, landing tolerances)
- Visual settings (colors, window size, FPS)
- Reward values
- Enable/disable wind, moving platforms, etc.

## Tips for Manual Play

1. **Start gentle** - Small bursts of thrust, not constant
2. **Watch your angle** - Stay upright! Use side thrusters to correct
3. **Kill your velocity** - Slow down before reaching the platform
4. **Fuel management** - You have 1000 fuel, use it wisely
5. **Don't panic** - Smooth corrections are better than jerky movements

## Tips for Building RL Agents

1. **Start simple** - Try a rule-based agent first to understand the challenge
2. **State representation** - The normalized state is ready for neural networks
3. **Reward shaping** - The default rewards work, but you can customize in `config.py`
4. **Curriculum learning** - Start with lots of fuel and forgiving landing conditions
5. **Record human play** - Use `record_gameplay.py` for imitation learning bootstrapping

## Next Steps for RL

Once you've played the game and understand the interface, you can:

1. **Wrap it in Gymnasium** - Create a gym.Env wrapper (easy!)
2. **Use Stable-Baselines3** - Train PPO, SAC, or DQN agents
3. **Imitation Learning** - Bootstrap from human demonstrations
4. **Curriculum Learning** - Gradually increase difficulty
5. **Transfer to your Forza project** - Apply the same RL concepts!

## License

Feel free to use this for learning and research!

## Credits

Built as a learning project for reinforcement learning research.
