"""Socket client for remote game control"""

import socket
import json
import time
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DroneState:
    """Dataclass representing the drone game state"""
    drone_x: float
    drone_y: float
    drone_vx: float
    drone_vy: float
    drone_angle: float
    drone_angular_vel: float
    drone_fuel: float
    platform_x: float
    platform_y: float
    distance_to_platform: float
    dx_to_platform: float
    dy_to_platform: float
    speed: float
    landed: bool
    crashed: bool
    steps: int


class DroneGameClient:
    """Client for connecting to DroneGame socket server

    Provides the same API as DroneGame but over network.
    """

    def __init__(self, host: str = 'localhost', port: int = 5555, timeout: float = 30.0):
        """Initialize client

        Args:
            host: Server hostname/IP
            port: Server port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout

        self.socket = None
        self.connected = False
        self.buffer = ""
        self.num_games = 1  # Will be set during handshake

    def connect(self):
        """Connect to server"""
        if self.connected:
            return

        print(f"Connecting to {self.host}:{self.port}...")

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to server at {self.host}:{self.port}")

            # Receive handshake
            handshake = self._receive_message()
            if handshake['type'] == 'HANDSHAKE':
                self.num_games = handshake['num_games']
                print(f"Server has {self.num_games} game instance(s)")
            else:
                print(f"Warning: Expected HANDSHAKE, got {handshake['type']}")

        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            try:
                self._send_message({'type': 'CLOSE'})
                self.socket.close()
            except:
                pass

        self.connected = False
        self.socket = None
        print("Disconnected from server")

    def reset(self, game_id: int = 0) -> DroneState:
        """Reset the game environment

        Args:
            game_id: Game instance ID (0 to num_games-1)

        Returns:
            Initial state as DroneState dataclass
        """
        if not self.connected:
            self.connect()

        # Validate game_id
        if game_id < 0 or game_id >= self.num_games:
            raise ValueError(f"Invalid game_id: {game_id}. Must be in range [0, {self.num_games})")

        # Send reset command
        self._send_message({'type': 'RESET', 'game_id': game_id})

        # Receive response
        response = self._receive_message()

        if response['type'] == 'ERROR':
            raise RuntimeError(f"Server error: {response['message']}")

        return DroneState(**response['state'])

    def step(self, action: Dict[str, int], game_id: int = 0) -> Tuple[DroneState, float, bool, Dict]:
        """Execute one step in the environment

        Args:
            action: Action dictionary with keys:
                - 'main_thrust': 0 or 1
                - 'left_thrust': 0 or 1
                - 'right_thrust': 0 or 1
            game_id: Game instance ID (0 to num_games-1)

        Returns:
            (state, reward, done, info) tuple where state is DroneState dataclass
        """
        if not self.connected:
            raise RuntimeError("Not connected to server. Call connect() or reset() first.")

        # Validate game_id
        if game_id < 0 or game_id >= self.num_games:
            raise ValueError(f"Invalid game_id: {game_id}. Must be in range [0, {self.num_games})")

        # Send step command
        self._send_message({
            'type': 'STEP',
            'action': action,
            'game_id': game_id
        })

        # Receive response
        response = self._receive_message()

        if response['type'] == 'ERROR':
            raise RuntimeError(f"Server error: {response['message']}")

        return (
            DroneState(**response['state']),
            response['reward'],
            response['done'],
            response['info']
        )

    def get_state(self, game_id: int = 0) -> DroneState:
        """Get current state without stepping

        Args:
            game_id: Game instance ID (0 to num_games-1)

        Returns:
            Current state as DroneState dataclass
        """
        if not self.connected:
            raise RuntimeError("Not connected to server")

        # Validate game_id
        if game_id < 0 or game_id >= self.num_games:
            raise ValueError(f"Invalid game_id: {game_id}. Must be in range [0, {self.num_games})")

        # Send get state command
        self._send_message({'type': 'GET_STATE', 'game_id': game_id})

        # Receive response
        response = self._receive_message()

        if response['type'] == 'ERROR':
            raise RuntimeError(f"Server error: {response['message']}")

        return DroneState(**response['state'])

    def _send_message(self, message: Dict):
        """Send JSON message to server

        Args:
            message: Message dictionary
        """
        msg_str = json.dumps(message) + '\n'
        self.socket.sendall(msg_str.encode('utf-8'))

    def _receive_message(self) -> Dict:
        """Receive JSON message from server

        Returns:
            Message dictionary
        """
        # Read until we have a complete message (newline-delimited)
        while '\n' not in self.buffer:
            data = self.socket.recv(4096)
            if not data:
                raise ConnectionError("Server closed connection")
            self.buffer += data.decode('utf-8')

        # Extract one message
        line, self.buffer = self.buffer.split('\n', 1)
        return json.loads(line)

    def close(self):
        """Close connection (alias for disconnect)"""
        self.disconnect()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def benchmark_latency(host: str = 'localhost', port: int = 5555, num_steps: int = 100) -> Dict[str, float]:
    """Benchmark client-server latency

    Args:
        host: Server host
        port: Server port
        num_steps: Number of steps to benchmark

    Returns:
        Dictionary with latency statistics (ms)
    """
    print(f"\nBenchmarking latency over {num_steps} steps...")

    client = DroneGameClient(host, port)
    client.connect()

    # Reset
    reset_start = time.time()
    client.reset()
    reset_time = (time.time() - reset_start) * 1000

    # Measure step latency
    step_times = []
    action = {'main_thrust': 1, 'left_thrust': 0, 'right_thrust': 0}

    for i in range(num_steps):
        step_start = time.time()
        state, reward, done, info = client.step(action)
        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)

        if done:
            client.reset()

    client.close()

    # Calculate statistics
    import numpy as np
    stats = {
        'reset_time_ms': reset_time,
        'step_mean_ms': np.mean(step_times),
        'step_std_ms': np.std(step_times),
        'step_min_ms': np.min(step_times),
        'step_max_ms': np.max(step_times),
        'step_median_ms': np.median(step_times)
    }

    print(f"\nLatency Statistics:")
    print(f"  Reset time:    {stats['reset_time_ms']:.2f} ms")
    print(f"  Step mean:     {stats['step_mean_ms']:.2f} ms")
    print(f"  Step std:      {stats['step_std_ms']:.2f} ms")
    print(f"  Step min:      {stats['step_min_ms']:.2f} ms")
    print(f"  Step max:      {stats['step_max_ms']:.2f} ms")
    print(f"  Step median:   {stats['step_median_ms']:.2f} ms")

    return stats
