"""
Async Socket Server for Car Racing Game
Free-running game loop (unlike drone game's blocking wait)
"""

import socket
import json
import threading
import queue
import time
import base64
import numpy as np

from . import config


class GameSocketServer:
    """
    Socket server for distributed RL training

    Key difference from drone game: Game loop runs continuously at 50 FPS
    regardless of client input. Client can request state on-demand.
    """

    def __init__(self, games, host=None, port=None):
        """
        Initialize socket server

        Args:
            games: Single RacingGame or list of RacingGame instances
            host: Server bind address (default: from config)
            port: Server port (default: from config)
        """
        # Handle single game or list
        if not isinstance(games, list):
            games = [games]
        self.games = games
        self.num_games = len(games)

        # Network settings
        self.host = host or config.SOCKET_HOST
        self.port = port or config.SOCKET_PORT

        # Socket
        self.server_socket = None
        self.client_socket = None
        self.connected = False

        # Threading
        self.network_thread = None
        self.running = False

        # Command queue (from network thread to main thread)
        self.command_queue = queue.Queue()

        # Action buffers (latest action for each game)
        self.action_buffers = [{
            'steer': 0.0,
            'gas': 0.0,
            'brake': 0.0
        } for _ in range(self.num_games)]

        # State request flags (which games need state sent)
        self.pending_state_requests = [False] * self.num_games

    def start(self):
        """Start server and wait for client connection"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"[Server] Listening on {self.host}:{self.port}")
        print(f"[Server] Managing {self.num_games} game instance(s)")

        # Accept connection
        self.client_socket, client_address = self.server_socket.accept()
        self.connected = True
        print(f"[Server] Client connected from {client_address}")

        # Send handshake
        self._send_message({
            'type': 'HANDSHAKE',
            'num_games': self.num_games
        })

        # Start network thread
        self.running = True
        self.network_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.network_thread.start()

        print("[Server] Ready for commands")

    def stop(self):
        """Stop server"""
        self.running = False
        self.connected = False

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

        if self.network_thread:
            self.network_thread.join(timeout=1.0)

        print("[Server] Stopped")

    def is_connected(self):
        """Check if client is connected"""
        return self.connected

    def process_commands(self):
        """
        Process pending commands from network thread

        Called from main game loop (non-blocking)
        Handles SET_ACTION, GET_STATE, RESET commands
        """
        try:
            # Process all queued commands (non-blocking)
            while True:
                try:
                    command = self.command_queue.get_nowait()
                    self._handle_command(command)
                except queue.Empty:
                    break
        except Exception as e:
            print(f"[Server] Error processing commands: {e}")
            self.connected = False

    def send_state_if_requested(self, game_id):
        """
        Send state to client if GET_STATE was requested

        Args:
            game_id: Which game instance
        """
        if self.pending_state_requests[game_id]:
            self._send_state(game_id)
            self.pending_state_requests[game_id] = False

    def get_action(self, game_id):
        """
        Get latest action for a game instance

        Args:
            game_id: Which game instance

        Returns:
            Action dict with keys 'steer', 'gas', 'brake'
        """
        if 0 <= game_id < self.num_games:
            return self.action_buffers[game_id].copy()
        return {'steer': 0.0, 'gas': 0.0, 'brake': 0.0}

    def _network_loop(self):
        """Network thread: receive commands from client"""
        buffer = ""

        while self.running and self.connected:
            try:
                # Receive data
                data = self.client_socket.recv(config.SOCKET_BUFFER_SIZE)
                if not data:
                    print("[Server] Client disconnected")
                    self.connected = False
                    break

                buffer += data.decode('utf-8')

                # Process complete messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            command = json.loads(line)
                            self.command_queue.put(command)
                        except json.JSONDecodeError as e:
                            print(f"[Server] Invalid JSON: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Server] Network error: {e}")
                self.connected = False
                break

    def _handle_command(self, command):
        """Handle a command from the client"""
        cmd_type = command.get('type')
        game_id = command.get('game_id', 0)

        if game_id < 0 or game_id >= self.num_games:
            self._send_error(f"Invalid game_id: {game_id}")
            return

        if cmd_type == 'SET_ACTION':
            # Update action buffer
            action = command.get('action', {})
            self.action_buffers[game_id] = {
                'steer': float(action.get('steer', 0.0)),
                'gas': float(action.get('gas', 0.0)),
                'brake': float(action.get('brake', 0.0))
            }

        elif cmd_type == 'GET_STATE':
            # Mark that this game needs state sent
            self.pending_state_requests[game_id] = True

        elif cmd_type == 'RESET':
            # Reset game
            self.games[game_id].reset()
            # Send initial state
            self._send_state(game_id)

        elif cmd_type == 'CLOSE':
            print("[Server] Client requested close")
            self.connected = False

        else:
            self._send_error(f"Unknown command type: {cmd_type}")

    def _send_state(self, game_id):
        """Send current state to client"""
        game = self.games[game_id]

        # Get telemetry
        telemetry = game.get_telemetry()

        # Get frame (render if needed)
        frame = None
        if game.render_mode is not None:
            frame_rgb = game.render('state_pixels')
            if frame_rgb is not None:
                # Encode frame as base64
                frame_bytes = frame_rgb.tobytes()
                frame = base64.b64encode(frame_bytes).decode('utf-8')

        # Get reward and done
        reward = game.reward
        done = (game.current_steps >= game.total_episode_steps or
                game.tile_visited_count >= len(game.track))

        # Send state message
        message = {
            'type': 'STATE',
            'game_id': game_id,
            'telemetry': telemetry,
            'frame': frame,
            'frame_shape': [config.STATE_H, config.STATE_W, 3] if frame else None,
            'reward': float(reward),
            'done': done,
            'info': {}
        }

        self._send_message(message)

    def _send_message(self, message):
        """Send JSON message to client"""
        if not self.connected or not self.client_socket:
            return

        try:
            data = json.dumps(message) + '\n'
            self.client_socket.sendall(data.encode('utf-8'))
        except Exception as e:
            print(f"[Server] Send error: {e}")
            self.connected = False

    def _send_error(self, error_message):
        """Send error message to client"""
        self._send_message({
            'type': 'ERROR',
            'message': error_message
        })
