# Socket API Documentation

## Overview

The Car Racing game uses a **JSON-over-TCP** socket protocol for distributed RL training. Unlike the delivery drone game, the car racing server runs a **free-running game loop** at 50 FPS regardless of client input.

**Key Features:**
- Multiple parallel game instances
- Asynchronous action updates (non-blocking)
- On-demand state retrieval with telemetry + RGB frames
- Base64-encoded frames for easy transmission
- Auto-reset on episode completion

## Architecture

```
┌─────────────────┐                    ┌──────────────────┐
│  Training Code  │ ◄──TCP Socket────► │  Socket Server   │
│  (Client)       │    JSON + \n       │  (Game Loop)     │
└─────────────────┘                    └──────────────────┘
                                              │
                                              ├─► Game 0 (50 FPS)
                                              ├─► Game 1 (50 FPS)
                                              └─► Game N (50 FPS)
```

## Connection Flow

1. **Client connects** to server (TCP)
2. **Server sends HANDSHAKE** with `num_games`
3. **Client sends commands** (SET_ACTION, GET_STATE, RESET)
4. **Server responds** with STATE messages (on-demand)
5. **Games run continuously** at 50 FPS with latest actions

## Message Format

All messages are **JSON objects** terminated by `\n` (newline).

```
{"type": "...", "param": "..."}\n
```

## Protocol Messages

### Client → Server

#### HANDSHAKE (Automatic)
Sent by server immediately after connection.

```json
{
  "type": "HANDSHAKE",
  "num_games": 4
}
```

#### SET_ACTION
Update action for a game instance (non-blocking, fire-and-forget).

```json
{
  "type": "SET_ACTION",
  "game_id": 0,
  "action": {
    "steer": 0.5,    // [-1.0, +1.0] left/right
    "gas": 1.0,      // [0.0, +1.0] throttle
    "brake": 0.0     // [0.0, +1.0] brake
  }
}
```

**Notes:**
- Game continues with last action if no new action received
- Default action: `{steer: 0, gas: 0, brake: 0}` (coast)

#### GET_STATE
Request current state (telemetry + frame).

```json
{
  "type": "GET_STATE",
  "game_id": 0
}
```

**Response:** STATE message (see below)

#### RESET
Reset a game instance to initial state.

```json
{
  "type": "RESET",
  "game_id": 0
}
```

**Response:** STATE message with initial state

#### CLOSE
Disconnect from server.

```json
{
  "type": "CLOSE"
}
```

### Server → Client

#### HANDSHAKE
First message after connection.

```json
{
  "type": "HANDSHAKE",
  "num_games": 4
}
```

#### STATE
Game state with telemetry and RGB frame.

```json
{
  "type": "STATE",
  "game_id": 0,
  "telemetry": {
    "car_x": 150.5,
    "car_y": 0.3,
    "car_vx": 2.5,
    "car_vy": 0.1,
    "car_angle": 1.57,
    "car_angular_vel": 0.05,
    "speed": 2.51,
    "wheel_speeds": [12.5, 12.5, 15.2, 15.2],
    "tiles_visited": 45,
    "total_tiles": 120,
    "steps": 234
  },
  "frame": "base64_encoded_string_here...",
  "frame_shape": [96, 96, 3],
  "reward": 42.5,
  "done": false,
  "info": {}
}
```

**Telemetry Fields:**
- `car_x`, `car_y`: Car position in world coordinates
- `car_vx`, `car_vy`: Car velocity (x, y components)
- `car_angle`: Car heading angle (radians)
- `car_angular_vel`: Angular velocity (radians/sec)
- `speed`: Total speed magnitude
- `wheel_speeds`: Angular velocity of 4 wheels
- `tiles_visited`: Number of track tiles visited
- `total_tiles`: Total tiles in track
- `steps`: Current step count in episode

**Frame Encoding:**
- RGB image as base64-encoded bytes
- Shape: `[96, 96, 3]` (height, width, channels)
- Dtype: `uint8`
- Decode with: `np.frombuffer(base64.b64decode(frame), dtype=np.uint8).reshape(96, 96, 3)`

**Note:** `frame` is `null` if server render_mode is `None`

#### ERROR
Error message from server.

```json
{
  "type": "ERROR",
  "message": "Invalid game_id: 5"
}
```

## Usage Examples

### Python Client (Basic)

```python
from game.socket_client import RacingGameClient

# Connect
client = RacingGameClient(host='localhost', port=5555)
client.connect()

print(f"Connected to server with {client.num_games} games")

# Reset game
telemetry, frame, reward, done, info = client.reset(game_id=0)

# Training loop
for step in range(1000):
    # Decide action (your RL agent)
    action = {
        'steer': 0.5,
        'gas': 1.0,
        'brake': 0.0
    }

    # Send action (non-blocking)
    client.set_action(action, game_id=0)

    # Get state (blocking, on-demand)
    telemetry, frame, reward, done, info = client.get_state(game_id=0)

    # Use frame for CNN input (96x96x3 numpy array)
    # Use telemetry for MLP input (dict of floats)

    if done:
        telemetry, frame, reward, done, info = client.reset(game_id=0)

client.disconnect()
```

### Python Client (Context Manager)

```python
from game.socket_client import RacingGameClient

with RacingGameClient(host='localhost', port=5555) as client:
    # Auto-connect and disconnect

    for episode in range(10):
        client.reset(game_id=0)
        done = False

        while not done:
            action = {'steer': 0, 'gas': 1.0, 'brake': 0}
            client.set_action(action, game_id=0)
            telemetry, frame, reward, done, info = client.get_state(game_id=0)
```

### Parallel Training (Multiple Games)

```python
from game.socket_client import RacingGameClient

with RacingGameClient() as client:
    num_games = client.num_games

    # Reset all games
    states = []
    for i in range(num_games):
        telemetry, frame, _, _, _ = client.reset(game_id=i)
        states.append((telemetry, frame))

    # Training loop
    for step in range(1000):
        # Batch inference (your model)
        actions = compute_actions_batch(states)

        # Send all actions (fire-and-forget)
        for i, action in enumerate(actions):
            client.set_action(action, game_id=i)

        # Collect all states
        states = []
        for i in range(num_games):
            telemetry, frame, reward, done, info = client.get_state(game_id=i)
            states.append((telemetry, frame))

            if done:
                client.reset(game_id=i)
```

## Performance

**Latency Benchmarks** (localhost):
- Mean: 2-5 ms per GET_STATE request
- SET_ACTION: <1 ms (non-blocking)
- Throughput: ~200-300 state requests/sec
- Frame encoding overhead: ~1-2 ms (96x96 RGB)

**Tips for Low Latency:**
1. Run server and client on same machine (localhost)
2. Use headless mode (`--render none`) for training
3. Batch GET_STATE requests when using multiple games
4. Consider disabling frame retrieval if using telemetry only

## Differences from Drone Game

| Feature | Drone Game | Car Racing |
|---------|------------|------------|
| **Game Loop** | Blocks waiting for action | Free-running 50 FPS |
| **Action Timing** | Synchronous (step-by-step) | Asynchronous (latest action) |
| **State Request** | Automatic after step | On-demand (GET_STATE) |
| **Frame Format** | N/A (state vector only) | Base64 RGB (96x96x3) |
| **Auto-Reset** | Manual | Automatic on episode end |

## Troubleshooting

**Connection Refused:**
- Ensure server is running: `python socket_server.py`
- Check firewall settings
- Verify correct port (default: 5555)

**Timeout Errors:**
- Increase client timeout: `RacingGameClient(timeout=60)`
- Check network latency
- Ensure server game loop is running

**Frame Decode Errors:**
- Verify `frame_shape` matches decode shape
- Check base64 encoding/decoding
- Ensure server has `render_mode != None`

**Game Not Responding:**
- Check server console for errors
- Verify `game_id` is valid (0 to num_games-1)
- Ensure game loop is running (not crashed)

## Advanced: Custom Protocol

You can implement custom clients in any language:

**Message Format:** JSON + `\n`
**Encoding:** UTF-8
**Frame Decoding:** Base64 → bytes → reshape(96, 96, 3)

Example in JavaScript, C++, etc. follows the same protocol!
