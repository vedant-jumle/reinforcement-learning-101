"""
Socket Client for Car Racing Game
Handles state parsing and base64 frame decoding
"""

import socket
import json
import base64
import numpy as np

from . import config


class RacingGameClient:
    """
    Client for connecting to RacingGame socket server

    Provides clean interface for RL training:
    - set_action(action, game_id): Send action to game
    - get_state(game_id): Get telemetry + RGB frame
    - reset(game_id): Reset game instance
    """

    def __init__(self, host='localhost', port=None, timeout=None):
        """
        Initialize client

        Args:
            host: Server hostname/IP
            port: Server port (default: from config)
            timeout: Socket timeout in seconds (default: from config)
        """
        self.host = host
        self.port = port or config.SOCKET_PORT
        self.timeout = timeout or config.SOCKET_TIMEOUT

        self.socket = None
        self.connected = False
        self.num_games = 0
        self.buffer = ""

    def connect(self):
        """Connect to server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)

        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[Client] Connected to {self.host}:{self.port}")

            # Receive handshake
            handshake = self._receive_message()
            if handshake['type'] == 'HANDSHAKE':
                self.num_games = handshake['num_games']
                print(f"[Client] Server has {self.num_games} game instance(s)")
            else:
                raise ConnectionError("Expected HANDSHAKE message")

        except Exception as e:
            print(f"[Client] Connection failed: {e}")
            self.connected = False
            raise

    def disconnect(self):
        """Disconnect from server"""
        if self.connected:
            try:
                self._send_message({'type': 'CLOSE'})
            except:
                pass

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        self.connected = False
        print("[Client] Disconnected")

    def set_action(self, action, game_id=0):
        """
        Send action to game (non-blocking)

        Args:
            action: dict with keys 'steer', 'gas', 'brake'
                   steer: [-1, +1]
                   gas: [0, +1]
                   brake: [0, +1]
            game_id: Which game instance (0 to num_games-1)
        """
        self._validate_game_id(game_id)

        message = {
            'type': 'SET_ACTION',
            'action': {
                'steer': float(action.get('steer', 0.0)),
                'gas': float(action.get('gas', 0.0)),
                'brake': float(action.get('brake', 0.0))
            },
            'game_id': game_id
        }

        self._send_message(message)

    def get_state(self, game_id=0):
        """
        Get current state (on-demand)

        Args:
            game_id: Which game instance

        Returns:
            telemetry: dict with car state (position, velocity, etc.)
            frame: numpy array (H, W, 3) RGB image or None
            reward: float
            done: bool
            info: dict
        """
        self._validate_game_id(game_id)

        # Request state
        self._send_message({
            'type': 'GET_STATE',
            'game_id': game_id
        })

        # Receive state
        response = self._receive_message()

        if response['type'] != 'STATE':
            raise RuntimeError(f"Expected STATE, got {response['type']}")

        # Parse telemetry
        telemetry = response.get('telemetry', {})

        # Decode frame if present
        frame = None
        if response.get('frame') is not None:
            frame = self._decode_frame(
                response['frame'],
                response.get('frame_shape', [config.STATE_H, config.STATE_W, 3])
            )

        reward = response.get('reward', 0.0)
        done = response.get('done', False)
        info = response.get('info', {})

        return telemetry, frame, reward, done, info

    def reset(self, game_id=0):
        """
        Reset game instance

        Args:
            game_id: Which game instance

        Returns:
            Same as get_state()
        """
        self._validate_game_id(game_id)

        # Send reset command
        self._send_message({
            'type': 'RESET',
            'game_id': game_id
        })

        # Receive initial state
        response = self._receive_message()

        if response['type'] != 'STATE':
            raise RuntimeError(f"Expected STATE after RESET, got {response['type']}")

        # Parse same as get_state
        telemetry = response.get('telemetry', {})

        frame = None
        if response.get('frame') is not None:
            frame = self._decode_frame(
                response['frame'],
                response.get('frame_shape', [config.STATE_H, config.STATE_W, 3])
            )

        reward = response.get('reward', 0.0)
        done = response.get('done', False)
        info = response.get('info', {})

        return telemetry, frame, reward, done, info

    def _send_message(self, message):
        """Send JSON message to server"""
        if not self.connected:
            raise ConnectionError("Not connected to server")

        try:
            data = json.dumps(message) + '\n'
            self.socket.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"[Client] Send error: {e}")
            self.connected = False
            raise

    def _receive_message(self):
        """Receive JSON message from server"""
        if not self.connected:
            raise ConnectionError("Not connected to server")

        # Read until we have a complete message (newline-delimited)
        while '\n' not in self.buffer:
            try:
                data = self.socket.recv(config.SOCKET_BUFFER_SIZE)
                if not data:
                    raise ConnectionError("Server closed connection")
                self.buffer += data.decode('utf-8')
            except socket.timeout:
                raise TimeoutError("Server response timeout")
            except Exception as e:
                print(f"[Client] Receive error: {e}")
                self.connected = False
                raise

        # Extract one message
        line, self.buffer = self.buffer.split('\n', 1)

        try:
            message = json.loads(line)
            return message
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from server: {e}")

    def _decode_frame(self, frame_b64, shape):
        """
        Decode base64 frame to numpy array

        Args:
            frame_b64: base64-encoded string
            shape: [H, W, C]

        Returns:
            numpy array (H, W, C) uint8
        """
        try:
            frame_bytes = base64.b64decode(frame_b64)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame.reshape(shape)
            return frame
        except Exception as e:
            print(f"[Client] Frame decode error: {e}")
            return None

    def _validate_game_id(self, game_id):
        """Validate game_id is in valid range"""
        if game_id < 0 or game_id >= self.num_games:
            raise ValueError(f"game_id {game_id} out of range [0, {self.num_games})")

    def __enter__(self):
        """Context manager support"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.disconnect()


def benchmark_latency(host='localhost', port=None, iterations=100):
    """
    Benchmark client-server latency

    Args:
        host: Server hostname
        port: Server port
        iterations: Number of test iterations

    Returns:
        dict with latency statistics
    """
    import time

    client = RacingGameClient(host=host, port=port)
    client.connect()

    print(f"[Benchmark] Running {iterations} iterations...")

    latencies = []

    try:
        # Warm-up
        for _ in range(10):
            client.get_state(0)

        # Measure
        for i in range(iterations):
            start = time.time()
            client.get_state(0)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{iterations} completed...")

    finally:
        client.disconnect()

    latencies = np.array(latencies)

    results = {
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'std_ms': np.std(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }

    print("\n[Benchmark] Results:")
    print(f"  Mean latency: {results['mean_ms']:.2f} ms")
    print(f"  Median latency: {results['median_ms']:.2f} ms")
    print(f"  Min latency: {results['min_ms']:.2f} ms")
    print(f"  Max latency: {results['max_ms']:.2f} ms")
    print(f"  Std dev: {results['std_ms']:.2f} ms")
    print(f"  95th percentile: {results['p95_ms']:.2f} ms")
    print(f"  99th percentile: {results['p99_ms']:.2f} ms")

    return results
