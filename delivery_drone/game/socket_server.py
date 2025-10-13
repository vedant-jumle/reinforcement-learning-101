"""Socket server for remote game control"""

import socket
import json
import threading
import queue
import time
from typing import Optional, Dict, Any
from . import config
from .game_engine import DroneGame


class GameSocketServer:
    """Socket server that exposes DroneGame over TCP"""

    def __init__(self, games, host: str = '0.0.0.0', port: int = 5555):
        """Initialize socket server

        Args:
            games: Single DroneGame instance or list of DroneGame instances
            host: Host to bind to (0.0.0.0 = all interfaces)
            port: Port to listen on
        """
        # Support both single game and list of games for backward compatibility
        if isinstance(games, list):
            self.games = games
        else:
            self.games = [games]

        self.num_games = len(self.games)
        self.host = host
        self.port = port

        # Socket
        self.server_socket = None
        self.client_socket = None
        self.client_address = None

        # Threading
        self.running = False
        self.network_thread = None

        # Communication queues
        self.action_queue = queue.Queue(maxsize=1)  # Actions from client
        self.response_queue = queue.Queue(maxsize=1)  # Responses to client

        # State
        self.connected = False

    def start(self):
        """Start the server"""
        # Create socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"Socket server listening on {self.host}:{self.port}")
        print("Waiting for client connection...")

        # Accept connection (blocking)
        self.client_socket, self.client_address = self.server_socket.accept()
        self.connected = True
        print(f"Client connected from {self.client_address}")

        # Send handshake with number of games
        self._send_handshake()

        # Start network thread
        self.running = True
        self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.network_thread.start()

    def stop(self):
        """Stop the server"""
        print("\nShutting down socket server...")
        self.running = False

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

        self.connected = False
        print("Socket server stopped")

    def _network_loop(self):
        """Network thread - handles receiving commands and sending responses"""
        buffer = ""

        while self.running:
            try:
                # Receive data
                data = self.client_socket.recv(4096)
                if not data:
                    print("Client disconnected")
                    self.connected = False
                    break

                # Decode
                buffer += data.decode('utf-8')

                # Process complete messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self._handle_message(line.strip())

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Network error: {e}")
                self.connected = False
                break

    def _handle_message(self, message_str: str):
        """Handle incoming message from client

        Args:
            message_str: JSON message string
        """
        try:
            message = json.loads(message_str)
            msg_type = message.get('type')
            game_id = message.get('game_id', 0)  # Default to game 0 for backward compatibility

            # Validate game_id
            if game_id < 0 or game_id >= self.num_games:
                self._send_error(f"Invalid game_id: {game_id}. Must be in range [0, {self.num_games})")
                return

            if msg_type == 'RESET':
                # Put reset command in queue
                try:
                    self.action_queue.put_nowait({'type': 'RESET', 'game_id': game_id})
                except queue.Full:
                    # Drop old action
                    try:
                        self.action_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.action_queue.put_nowait({'type': 'RESET', 'game_id': game_id})

            elif msg_type == 'STEP':
                # Put step command in queue
                action = message.get('action', {})
                try:
                    self.action_queue.put_nowait({'type': 'STEP', 'action': action, 'game_id': game_id})
                except queue.Full:
                    # Drop old action
                    try:
                        self.action_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.action_queue.put_nowait({'type': 'STEP', 'action': action, 'game_id': game_id})

            elif msg_type == 'GET_STATE':
                # Put get state command in queue
                try:
                    self.action_queue.put_nowait({'type': 'GET_STATE', 'game_id': game_id})
                except queue.Full:
                    try:
                        self.action_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.action_queue.put_nowait({'type': 'GET_STATE', 'game_id': game_id})

            elif msg_type == 'CLOSE':
                # Shutdown
                self.running = False

            else:
                # Unknown message type
                self._send_error(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            self._send_error(f"Invalid JSON: {e}")
        except Exception as e:
            self._send_error(f"Error handling message: {e}")

    def process_commands(self):
        """Process pending commands from client (call from main game loop)

        Returns:
            True if processed a command, False otherwise
        """
        if not self.connected:
            return False

        try:
            # Check for command (non-blocking)
            command = self.action_queue.get_nowait()
            game_id = command.get('game_id', 0)
            game = self.games[game_id]

            if command['type'] == 'RESET':
                state = game.reset()
                self._send_state(state, 0.0, False, {}, game_id)

            elif command['type'] == 'STEP':
                action = command['action']
                state, reward, done, info = game.step(action)
                self._send_state(state, reward, done, info, game_id)

            elif command['type'] == 'GET_STATE':
                state = game.get_state()
                info = game._get_info()
                self._send_state(state, 0.0, game.done, info, game_id)

            return True

        except queue.Empty:
            return False

    def _send_handshake(self):
        """Send handshake message to client with server info"""
        message = {
            'type': 'HANDSHAKE',
            'num_games': self.num_games
        }
        self._send_message(message)

    def _send_state(self, state: Dict, reward: float, done: bool, info: Dict, game_id: int = 0):
        """Send state to client

        Args:
            state: State dictionary
            reward: Reward value
            done: Done flag
            info: Info dictionary
            game_id: Game instance ID
        """
        message = {
            'type': 'STATE',
            'game_id': game_id,
            'state': state,
            'reward': float(reward),
            'done': bool(done),
            'info': info
        }
        self._send_message(message)

    def _send_error(self, error_msg: str):
        """Send error message to client

        Args:
            error_msg: Error message
        """
        message = {
            'type': 'ERROR',
            'message': error_msg
        }
        self._send_message(message)

    def _send_message(self, message: Dict):
        """Send JSON message to client

        Args:
            message: Message dictionary
        """
        if not self.connected or not self.client_socket:
            return

        try:
            msg_str = json.dumps(message) + '\n'
            self.client_socket.sendall(msg_str.encode('utf-8'))
        except Exception as e:
            print(f"Error sending message: {e}")
            self.connected = False

    def is_connected(self):
        """Check if client is connected

        Returns:
            True if connected
        """
        return self.connected
