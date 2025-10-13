"""Socket client for remote game control"""

import socket
import json
import numpy as np
from typing import Dict, Any, Tuple


class DrivingGameClient:
    """Client for connecting to DrivingGame socket server"""

    def __init__(self, host='localhost', port=5555, timeout=30.0):
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

    def reset(self, game_id=0):
        """Reset the game environment

        Args:
            game_id: Game instance ID (0 to num_games-1)

        Returns:
            Initial observation (numpy array)
        """
        if not self.connected:
            self.connect()

        self._send_message({'type': 'RESET', 'game_id': game_id})
        response = self._receive_message()

        if response['type'] == 'ERROR':
            raise RuntimeError(f"Server error: {response['message']}")

        obs = np.array(response['observation'], dtype=np.uint8)
        return obs

    def step(self, action, game_id=0):
        """Execute action in game

        Args:
            action: Action dict {'steering': [-1, 1], 'acceleration': [-1, 1]}
            game_id: Game instance ID

        Returns:
            (observation, reward, done, info) tuple
        """
        if not self.connected:
            self.connect()

        self._send_message({'type': 'STEP', 'action': action, 'game_id': game_id})
        response = self._receive_message()

        if response['type'] == 'ERROR':
            raise RuntimeError(f"Server error: {response['message']}")

        obs = np.array(response['observation'], dtype=np.uint8)
        reward = response['reward']
        done = response['done']
        info = response['info']

        return obs, reward, done, info

    def get_state(self, game_id=0):
        """Get current state without stepping

        Args:
            game_id: Game instance ID

        Returns:
            State dictionary
        """
        if not self.connected:
            self.connect()

        self._send_message({'type': 'GET_STATE', 'game_id': game_id})
        response = self._receive_message()

        if response['type'] == 'ERROR':
            raise RuntimeError(f"Server error: {response['message']}")

        return response['state']

    def _send_message(self, message):
        """Send JSON message to server"""
        data = json.dumps(message) + '\n'
        self.socket.sendall(data.encode('utf-8'))

    def _receive_message(self):
        """Receive JSON message from server"""
        while '\n' not in self.buffer:
            data = self.socket.recv(4096).decode('utf-8')
            if not data:
                raise ConnectionError("Server closed connection")
            self.buffer += data

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
