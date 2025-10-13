"""Socket server for remote game control with continuous physics"""

import socket
import json
import threading
import queue
import time
from typing import List


class DrivingGameSocketServer:
    """Socket server that runs game at fixed FPS"""

    def __init__(self, games, host='0.0.0.0', port=5555):
        """Initialize socket server

        Args:
            games: List of DrivingGame instances or single instance
        """
        self.games = games if isinstance(games, list) else [games]
        self.num_games = len(self.games)
        self.host = host
        self.port = port

        # Socket
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.connected = False

        # Network thread
        self.running = False
        self.network_thread = None

    def start(self):
        """Start server and game physics loops"""
        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"Server listening on {self.host}:{self.port}")
        print("Waiting for client connection...")

        # Accept connection
        self.client_socket, self.client_address = self.server_socket.accept()
        self.connected = True
        print(f"Client connected from {self.client_address}")

        # Send handshake
        self._send_message({'type': 'HANDSHAKE', 'num_games': self.num_games})

        # Start network thread
        self.running = True
        self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.network_thread.start()

        # Start physics loops for all games
        for game in self.games:
            game.start_physics_loop()

        print(f"Started physics loops for {self.num_games} game(s)")

    def stop(self):
        """Stop server and game loops"""
        print("\nShutting down server...")
        self.running = False
        self.connected = False

        # Stop all game physics loops
        for game in self.games:
            game.stop_physics_loop()

        if self.network_thread:
            self.network_thread.join(timeout=2.0)

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print("Server stopped")

    def _network_loop(self):
        """Handle incoming messages and send responses"""
        buffer = ""

        while self.running and self.connected:
            try:
                # Receive data
                data = self.client_socket.recv(4096).decode('utf-8')
                if not data:
                    print("Client disconnected")
                    self.connected = False
                    break

                buffer += data

                # Process complete messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            # Handle message
                            response = self._handle_message(message)
                            if response:
                                self._send_message(response)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            self._send_message({'type': 'ERROR', 'message': 'Invalid JSON'})

            except Exception as e:
                print(f"Network error: {e}")
                self.connected = False
                break

    def _handle_message(self, message):
        """Handle client message

        Returns:
            Response dict or None
        """
        msg_type = message.get('type')
        game_id = message.get('game_id', 0)

        if game_id >= self.num_games:
            return {'type': 'ERROR', 'message': f'Invalid game_id {game_id}'}

        game = self.games[game_id]

        if msg_type == 'RESET':
            obs = game.reset()
            state = game.get_state()
            return {
                'type': 'STATE',
                'observation': obs.tolist(),
                'state': state,
                'reward': 0.0,
                'done': False,
                'info': {'episode': game.episode}
            }

        elif msg_type == 'STEP':
            # Update action buffer
            action = message.get('action', {})
            game.set_action(action)

            # Get latest observation (game is running in background)
            obs = game.get_observation()
            state = game.get_state()
            reward = game.last_reward
            done = game.done

            return {
                'type': 'STATE',
                'observation': obs.tolist(),
                'state': state,
                'reward': float(reward),
                'done': bool(done),
                'info': {'steps': game.steps, 'total_reward': game.total_reward}
            }

        elif msg_type == 'GET_STATE':
            obs = game.get_observation()
            state = game.get_state()
            return {
                'type': 'STATE',
                'observation': obs.tolist(),
                'state': state,
                'reward': float(game.last_reward),
                'done': bool(game.done),
                'info': {}
            }

        elif msg_type == 'CLOSE':
            self.stop()
            return None

        else:
            return {'type': 'ERROR', 'message': f'Unknown message type: {msg_type}'}

    def _send_message(self, message):
        """Send JSON message to client"""
        try:
            data = json.dumps(message) + '\n'
            self.client_socket.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False

    def is_connected(self):
        """Check if client is connected"""
        return self.connected

    def process_commands(self):
        """Process pending commands (for compatibility with delivery_drone)"""
        # Not needed since we have separate network thread
        pass
