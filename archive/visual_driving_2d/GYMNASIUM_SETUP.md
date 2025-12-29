# Using Gymnasium CarRacing-v3 with Socket System

## Problem Solved

Pyglet rendering doesn't work on WSL2 due to OpenGL context creation issues. **Solution**: Use gymnasium's built-in CarRacing-v3 environment which uses **pygame rendering** (works perfectly on WSL2) with our custom socket server for parallel training.

## Setup

### 1. Install Dependencies

Already installed:
- `gymnasium` (1.2.2)
- `Box2D` (2.3.10)
- `pygame` (2.6.1)

### 2. Start Socket Server

```bash
# Single game with visualization (human play)
python socket_server_gymnasium.py --num-games 1 --render human

# Parallel training (6 games, headless)
python socket_server_gymnasium.py --num-games 6 --render none

# Custom settings
python socket_server_gymnasium.py --num-games 4 --render human --port 5556 --max-steps 1000
```

**Server Options:**
- `--host HOST`: Server address (default: 0.0.0.0)
- `--port PORT`: Server port (default: 5555)
- `--num-games N`: Number of parallel environments (default: 1)
- `--render MODE`: Render mode - `human`, `rgb_array`, `none` (default: human)
- `--max-steps N`: Max steps per episode (default: 1000)
- `--seed N`: Random seed for track generation
- `--verbose {0,1}`: Print debug info (default: 1)

### 3. Connect Client for Training

The existing `RacingGameClient` works with the gymnasium server!

```python
from game.socket_client import RacingGameClient

# Connect to server
with RacingGameClient(host='localhost', port=5555) as client:
    print(f"Connected to {client.num_games} game(s)")

    # Reset
    telemetry, frame, reward, done, info = client.reset(game_id=0)
    # frame is now 96x96x3 RGB numpy array (not base64)

    # Training loop
    for step in range(1000):
        # Your RL agent
        action = {
            'steer': 0.0,   # [-1, +1] left/right
            'gas': 1.0,     # [0, +1] throttle
            'brake': 0.0    # [0, +1] brake
        }

        # Send action (non-blocking)
        client.set_action(action, game_id=0)

        # Get state
        telemetry, frame, reward, done, info = client.get_state(game_id=0)

        if done:
            client.reset(game_id=0)
```

## Key Differences from Custom Implementation

| Feature | Custom (pyglet) | Gymnasium (pygame) |
|---------|----------------|-------------------|
| **Rendering** | Pyglet + OpenGL (broken on WSL2) | Pygame (works on WSL2) ✓ |
| **Physics** | Box2D | Box2D (identical) |
| **Observation** | Telemetry dict + 96×96 RGB | 96×96 RGB pixels |
| **Actions** | 3 continuous | 3 continuous (identical) |
| **Socket Protocol** | Custom | Same protocol ✓ |
| **Free-running loop** | 50 FPS | 50 FPS (identical) |
| **Parallel games** | Yes | Yes ✓ |

## Advantages of Gymnasium Approach

✅ **Works on WSL2** - Pygame rendering has no GL context issues
✅ **Maintained** - Official gymnasium environment, regularly updated
✅ **Tested** - Used by thousands of researchers
✅ **Features** - Domain randomization, lap tracking, etc.
✅ **Compatible** - Same socket protocol as before

## Test the System

```bash
# Terminal 1: Start server with visualization
python socket_server_gymnasium.py --render human

# Terminal 2: Run test
python test_gymnasium_socket.py
```

Expected output:
```
✓ Connected to server with 1 game(s)
✓ Reset successful
  Frame shape: (96, 96, 3)
Running 200 steps...
✓ Test completed successfully!
```

## For Training

Use the exact same training code as before, just:

1. **Start server**: `python socket_server_gymnasium.py --num-games 6 --render none`
2. **Connect**: Use existing `RacingGameClient`
3. **Train**: Same API as delivery_drone

```python
# Parallel training example
from game.socket_client import RacingGameClient

with RacingGameClient() as client:
    num_games = client.num_games  # 6 parallel games

    # Reset all
    for i in range(num_games):
        client.reset(game_id=i)

    # Training loop
    for iteration in range(1000):
        # Collect from all games
        for game_id in range(num_games):
            # Your policy
            action = policy(state)

            # Step
            client.set_action(action, game_id)
            state, reward, done, info = client.get_state(game_id)

            # Store for training
            # ...

            if done:
                client.reset(game_id)
```

## What Changed

**Before**: Custom game engine with pyglet → doesn't work on WSL2
**Now**: Gymnasium CarRacing-v3 with pygame → works perfectly on WSL2

**Compatibility**: Socket protocol unchanged, training code unchanged!

## Files

- **[socket_server_gymnasium.py](socket_server_gymnasium.py)** - New socket server wrapping gymnasium
- **[test_gymnasium_socket.py](test_gymnasium_socket.py)** - Test script
- **[game/socket_client.py](game/socket_client.py)** - Same client (no changes needed)

---

**Result**: Working visualization on WSL2 + parallel training via sockets + official gymnasium environment!
