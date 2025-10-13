## Socket API Documentation

Complete reference for the Delivery Drone socket interface.

## Overview

The socket interface allows you to control the Delivery Drone game remotely over TCP sockets. This enables:

- **Distributed training**: Run game on one machine, RL agent on another
- **Language-agnostic clients**: Any language with socket support can control the game
- **Multiple environments**: Easy to scale to parallel environments (future)
- **Same API**: Client provides the same interface as `DroneGame` class

## Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Socket Server   │◄───────►│  Socket Client   │
│  (Game Process)  │   TCP   │  (Agent Process) │
│                  │  5555   │                  │
│  • Runs game     │         │  • Sends actions │
│  • Renders       │         │  • Gets obs      │
│  • Physics       │         │  • Trains model  │
└──────────────────┘         └──────────────────┘
```

## Quick Start

### 1. Start the Server

```bash
# Visual mode (watch the agent play)
python socket_server.py --render human

# Headless mode (faster, no visualization)
python socket_server.py --render none

# Custom port
python socket_server.py --port 5556

# Allow remote connections
python socket_server.py --host 0.0.0.0
```

### 2. Connect a Client

```python
from game.socket_client import DroneGameClient

# Connect
client = DroneGameClient(host='localhost', port=5555)

# Reset environment
state = client.reset()

# Run game loop
while not done:
    action = {'main_thrust': 1, 'left_thrust': 0, 'right_thrust': 0}
    state, reward, done, info = client.step(action)

# Disconnect
client.close()
```

## Protocol Specification

### Message Format

Messages are JSON objects, newline-delimited (`\n`).

**Client → Server Messages:**

```json
{"type": "RESET"}
{"type": "STEP", "action": {"main_thrust": 1, "left_thrust": 0, "right_thrust": 0}}
{"type": "GET_STATE"}
{"type": "CLOSE"}
```

**Server → Client Messages:**

```json
{
  "type": "STATE",
  "state": {
    "drone_x": 0.5,
    "drone_y": 0.167,
    ...
  },
  "reward": -0.1,
  "done": false,
  "info": {
    "steps": 42,
    "total_reward": -4.2,
    ...
  }
}

{
  "type": "ERROR",
  "message": "Error description"
}
```

### Message Types

#### RESET

Reset the environment to initial state.

**Client Request:**
```json
{"type": "RESET"}
```

**Server Response:**
```json
{
  "type": "STATE",
  "state": {...},
  "reward": 0.0,
  "done": false,
  "info": {...}
}
```

#### STEP

Execute one step with the given action.

**Client Request:**
```json
{
  "type": "STEP",
  "action": {
    "main_thrust": 1,   // 0 or 1
    "left_thrust": 0,   // 0 or 1
    "right_thrust": 0   // 0 or 1
  }
}
```

**Server Response:**
```json
{
  "type": "STATE",
  "state": {...},
  "reward": -0.1,
  "done": false,
  "info": {...}
}
```

#### GET_STATE

Get current state without stepping.

**Client Request:**
```json
{"type": "GET_STATE"}
```

**Server Response:**
```json
{
  "type": "STATE",
  "state": {...},
  "reward": 0.0,
  "done": false,
  "info": {...}
}
```

#### CLOSE

Request server shutdown.

**Client Request:**
```json
{"type": "CLOSE"}
```

**Server Response:**
Server closes connection.

#### ERROR

Server error response.

**Server Response:**
```json
{
  "type": "ERROR",
  "message": "Description of error"
}
```

## State Space

The state dictionary contains 15 normalized values:

```python
{
    # Position (normalized [0, 1])
    'drone_x': 0.5,              # Horizontal position
    'drone_y': 0.167,            # Vertical position

    # Velocity (normalized)
    'drone_vx': 0.0,             # Horizontal velocity
    'drone_vy': 0.03,            # Vertical velocity (positive = down)
    'speed': 0.03,               # Total speed magnitude

    # Orientation (normalized [-1, 1])
    'drone_angle': 0.0,          # Tilt angle (-1=left, 0=upright, 1=right)
    'drone_angular_vel': 0.0,    # Rotational velocity

    # Resources (normalized [0, 1])
    'drone_fuel': 1.0,           # Fuel remaining

    # Target (normalized [0, 1])
    'platform_x': 0.5,
    'platform_y': 0.833,

    # Distances (normalized)
    'distance_to_platform': 0.5,
    'dx_to_platform': 0.0,       # Horizontal distance
    'dy_to_platform': 0.667,     # Vertical distance

    # Status (boolean)
    'landed': False,
    'crashed': False
}
```

## Action Space

Actions are binary (0 or 1):

```python
{
    'main_thrust': 0,    # Thrust in direction drone is facing
    'left_thrust': 0,    # Rotate counter-clockwise
    'right_thrust': 0    # Rotate clockwise
}
```

All 8 combinations are valid (including all off or all on).

## Info Dictionary

Contains additional episode information:

```python
{
    'steps': 42,                    # Steps in current episode
    'total_reward': -4.2,           # Cumulative reward
    'episode': 1,                   # Episode number
    'fuel_remaining': 916.0,        # Fuel units left
    'distance_to_platform': 342.5,  # Pixels to platform
    'speed': 2.3,                   # Current speed
    'angle': 5.2                    # Current angle (degrees)
}
```

## Client API

### DroneGameClient Class

```python
class DroneGameClient:
    """Client for remote game control"""

    def __init__(self, host='localhost', port=5555, timeout=30.0):
        """Initialize client"""

    def connect(self):
        """Connect to server"""

    def disconnect(self):
        """Disconnect from server"""

    def reset(self) -> Dict[str, Any]:
        """Reset environment, return initial state"""

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute action, return (state, reward, done, info)"""

    def get_state(self) -> Dict[str, Any]:
        """Get current state without stepping"""

    def close(self):
        """Close connection (alias for disconnect)"""
```

### Context Manager Support

```python
with DroneGameClient() as client:
    state = client.reset()
    # ... use client ...
# Automatically disconnects
```

### Example Usage

```python
from game.socket_client import DroneGameClient

# Create and connect
client = DroneGameClient(host='localhost', port=5555)
state = client.reset()

# Run episode
episode_reward = 0
done = False

while not done:
    # Your policy decides action
    action = my_policy(state)

    # Step environment
    state, reward, done, info = client.step(action)
    episode_reward += reward

print(f"Episode reward: {episode_reward}")

# Cleanup
client.close()
```

## Server Configuration

### Command-Line Arguments

```bash
python socket_server.py [OPTIONS]

Options:
  --host HOST          Host to bind to (default: 0.0.0.0)
  --port PORT          Port to listen on (default: 5555)
  --render MODE        Render mode: human, rgb_array, none (default: human)
  --fps FPS            Target FPS (default: 60)
```

### Examples

```bash
# Visual mode with default settings
python socket_server.py

# Headless mode on custom port
python socket_server.py --render none --port 5556

# Allow remote connections
python socket_server.py --host 0.0.0.0 --port 5555

# High FPS training
python socket_server.py --render none --fps 120
```

## Performance

### Latency Benchmarks

Typical latencies for localhost connections:

| Operation | Mean | Min | Max |
|-----------|------|-----|-----|
| Reset | ~2-5ms | ~1ms | ~10ms |
| Step | ~1-3ms | ~0.5ms | ~10ms |
| Get State | ~1-2ms | ~0.5ms | ~5ms |

**Estimated throughput:** 300-500 FPS (steps per second)

### Benchmark Tool

```bash
# Run latency benchmark
PYTHONPATH=. python examples/benchmark_latency.py --steps 1000

# Custom server
PYTHONPATH=. python examples/benchmark_latency.py --host localhost --port 5555
```

## Error Handling

### Connection Errors

```python
try:
    client = DroneGameClient()
    client.connect()
except ConnectionError as e:
    print(f"Connection failed: {e}")
    print("Make sure the server is running!")
```

### Timeout Errors

```python
client = DroneGameClient(timeout=5.0)  # 5 second timeout

try:
    state = client.reset()
except socket.timeout:
    print("Server took too long to respond")
```

### Server Errors

```python
try:
    state, reward, done, info = client.step(action)
except RuntimeError as e:
    print(f"Server error: {e}")
```

## Best Practices

### 1. Use Context Managers

```python
with DroneGameClient() as client:
    state = client.reset()
    # ... training loop ...
# Automatically cleaned up
```

### 2. Handle Disconnects

```python
while training:
    try:
        state, reward, done, info = client.step(action)
    except ConnectionError:
        print("Disconnected, reconnecting...")
        client.connect()
        state = client.reset()
```

### 3. Headless Mode for Training

```bash
# Faster training without visualization
python socket_server.py --render none
```

### 4. Monitor Latency

```python
import time

times = []
for _ in range(100):
    start = time.time()
    state, reward, done, info = client.step(action)
    times.append((time.time() - start) * 1000)

print(f"Average latency: {sum(times)/len(times):.2f}ms")
```

## Advanced Usage

### Multiple Clients (Future)

Currently, the server supports one client at a time. For multiple parallel environments, run multiple servers:

```bash
# Terminal 1
python socket_server.py --port 5555 --render none

# Terminal 2
python socket_server.py --port 5556 --render none

# Terminal 3
python socket_server.py --port 5557 --render none
```

Then connect multiple clients:

```python
clients = [
    DroneGameClient(port=5555),
    DroneGameClient(port=5556),
    DroneGameClient(port=5557),
]

for client in clients:
    client.reset()
```

### Custom Protocols

To implement your own client in another language:

1. **Connect**: TCP socket to `host:port`
2. **Send**: JSON message + `\n`
3. **Receive**: Read until `\n`, parse JSON
4. **Close**: Send `{"type": "CLOSE"}\n`

Example in Python (low-level):

```python
import socket
import json

# Connect
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5555))

# Reset
msg = json.dumps({"type": "RESET"}) + "\n"
sock.sendall(msg.encode())

# Receive
data = b""
while b"\n" not in data:
    data += sock.recv(4096)
response = json.loads(data.decode().split("\n")[0])

print(response['state'])

# Close
sock.close()
```

## Troubleshooting

### Server Won't Start

**Problem:** "Address already in use"

**Solution:** Port is already in use
```bash
# Find process using port 5555
lsof -i :5555

# Kill it or use different port
python socket_server.py --port 5556
```

### Client Can't Connect

**Problem:** "Connection refused"

**Solution:**
1. Make sure server is running
2. Check host/port match server settings
3. Check firewall settings (for remote connections)

### High Latency

**Problem:** Steps taking too long

**Solutions:**
1. Use headless mode: `--render none`
2. Run on same machine (localhost)
3. Reduce network latency (faster connection)
4. Lower server FPS: `--fps 30`

### Server Freezes

**Problem:** Server stops responding

**Solution:**
1. Check if client is sending malformed messages
2. Restart server
3. Check server logs for errors

## Security Considerations

### For Production Use

1. **Authentication**: Add authentication layer
2. **Encryption**: Use TLS/SSL for remote connections
3. **Rate Limiting**: Limit connections per IP
4. **Input Validation**: Validate all client messages
5. **Firewall**: Block external access if not needed

**Current implementation is for development/research only.**

## Example Applications

### 1. Distributed RL Training

Train on a GPU machine, run game on another:

```python
# On GPU machine
from game.socket_client import DroneGameClient

client = DroneGameClient(host='game-server.local', port=5555)
agent = PPO(...)  # Your RL agent

for episode in range(1000):
    state = client.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, info = client.step(action)
        agent.train_step(state, action, reward)
```

### 2. Multi-Language Clients

Implement client in any language:

```javascript
// Node.js client example
const net = require('net');

const client = net.connect({host: 'localhost', port: 5555}, () => {
    // Reset
    client.write(JSON.stringify({type: 'RESET'}) + '\n');
});

client.on('data', (data) => {
    const response = JSON.parse(data.toString());
    console.log('State:', response.state);
});
```

### 3. Web-Based Control

Create a web interface to control the game:

```python
# Flask server that proxies to game server
from flask import Flask, jsonify
from game.socket_client import DroneGameClient

app = Flask(__name__)
client = DroneGameClient()

@app.route('/reset')
def reset():
    state = client.reset()
    return jsonify(state)

@app.route('/step/<action>')
def step(action):
    # Parse action from URL
    state, reward, done, info = client.step(parse_action(action))
    return jsonify({
        'state': state,
        'reward': reward,
        'done': done
    })
```

## Related Tools

- **Manual Play**: `python manual_play.py` - Play with keyboard
- **Remote Agent**: `PYTHONPATH=. python examples/remote_agent.py` - Example remote agent
- **Latency Benchmark**: `PYTHONPATH=. python examples/benchmark_latency.py` - Measure performance
- **Socket Tests**: `python test_socket.py` - Test socket interface

## Support

For issues, questions, or contributions related to the socket interface, please check:
- **README.md** - General documentation
- **GETTING_STARTED.md** - Beginner guide
- **examples/** - Example scripts

## Future Features

Planned enhancements:
- [ ] Multiple concurrent clients
- [ ] Binary protocol (msgpack/protobuf)
- [ ] WebSocket support
- [ ] Authentication system
- [ ] Compression for observations
- [ ] Recording/replay over network
- [ ] Load balancing across servers

---

**Happy distributed training! 🚁**
