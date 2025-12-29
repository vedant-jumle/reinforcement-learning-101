#!/usr/bin/env python
"""
Socket Server for Gymnasium CarRacing-v3 Environment
Wraps gymnasium's CarRacing to enable parallel training via sockets
"""

import argparse
import json
import socket
import threading
import time
import base64
import numpy as np
import gymnasium as gym


class CarRacingSocketServer:
    """Socket server for parallel CarRacing environments"""

    def __init__(self, host='0.0.0.0', port=5555, num_games=1, render_mode='human',
                 max_steps=1000, seed=None, verbose=1):
        self.host = host
        self.port = port
        self.num_games = num_games
        self.render_mode = render_mode if num_games == 1 else None  # Only render for single game
        self.max_steps = max_steps
        self.seed = seed
        self.verbose = verbose

        # Create gymnasium environments
        self.envs = []
        for i in range(num_games):
            env_seed = (seed + i) if seed is not None else None
            env = gym.make(
                'CarRacing-v3',
                render_mode=self.render_mode if i == 0 else None,  # Only render first env
                domain_randomize=False,
                continuous=True
            )
            if env_seed is not None:
                env.reset(seed=env_seed)
            self.envs.append(env)

        # Game states
        self.current_states = [None] * num_games
        self.current_actions = [{'steer': 0.0, 'gas': 0.0, 'brake': 0.0}] * num_games
        self.episode_steps = [0] * num_games
        self.total_rewards = [0.0] * num_games

        # Socket server
        self.server_socket = None
        self.client_socket = None
        self.running = False

        # Free-running game loop
        self.game_thread = None
        self.game_running = False

    def start(self):
        """Start the socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        if self.verbose:
            print(f"Socket server listening on {self.host}:{self.port}")
            print(f"Number of games: {self.num_games}")
            print(f"Render mode: {self.render_mode}")
            print(f"Waiting for client connection...")

        self.running = True

        # Accept client connection
        self.client_socket, client_address = self.server_socket.accept()
        if self.verbose:
            print(f"Client connected from {client_address}")

        # Send handshake
        self._send_message({'type': 'HANDSHAKE', 'num_games': self.num_games})

        # Start free-running game loop
        self.game_running = True
        self.game_thread = threading.Thread(target=self._game_loop, daemon=True)
        self.game_thread.start()

        # Handle client commands
        self._handle_client()

    def _game_loop(self):
        """Free-running game loop at 50 FPS"""
        target_fps = 50
        frame_time = 1.0 / target_fps

        while self.game_running:
            start_time = time.time()

            # Step all environments
            for game_id in range(self.num_games):
                if self.current_states[game_id] is None:
                    continue

                # Get current action
                action_dict = self.current_actions[game_id]
                action = np.array([action_dict['steer'], action_dict['gas'], action_dict['brake']])

                # Step environment
                obs, reward, terminated, truncated, info = self.envs[game_id].step(action)
                done = terminated or truncated

                # Update state
                self.current_states[game_id] = (obs, reward, done, info)
                self.total_rewards[game_id] += reward
                self.episode_steps[game_id] += 1

                # Auto-reset if done or max steps reached
                if done or self.episode_steps[game_id] >= self.max_steps:
                    if self.verbose and game_id == 0:
                        print(f"[Game {game_id}] Episode done: steps={self.episode_steps[game_id]}, "
                              f"reward={self.total_rewards[game_id]:.1f}")
                    obs, info = self.envs[game_id].reset()
                    self.current_states[game_id] = (obs, 0.0, False, info)
                    self.episode_steps[game_id] = 0
                    self.total_rewards[game_id] = 0.0

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)

    def _handle_client(self):
        """Handle client commands"""
        buffer = b''

        while self.running:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break

                buffer += data

                # Process complete messages
                while b'\n' in buffer:
                    message_bytes, buffer = buffer.split(b'\n', 1)
                    try:
                        message = json.loads(message_bytes.decode('utf-8'))
                        self._process_command(message)
                    except json.JSONDecodeError:
                        if self.verbose:
                            print(f"Invalid JSON received: {message_bytes}")

            except Exception as e:
                if self.verbose:
                    print(f"Error handling client: {e}")
                break

        self.stop()

    def _process_command(self, message):
        """Process client command"""
        msg_type = message.get('type')
        game_id = message.get('game_id', 0)

        if msg_type == 'RESET':
            self._handle_reset(game_id)

        elif msg_type == 'SET_ACTION':
            action = message.get('action', {})
            self.current_actions[game_id] = {
                'steer': action.get('steer', 0.0),
                'gas': action.get('gas', 0.0),
                'brake': action.get('brake', 0.0)
            }

        elif msg_type == 'GET_STATE':
            self._handle_get_state(game_id)

        elif msg_type == 'CLOSE':
            self.running = False

    def _handle_reset(self, game_id):
        """Reset a game environment"""
        obs, info = self.envs[game_id].reset()
        self.current_states[game_id] = (obs, 0.0, False, info)
        self.episode_steps[game_id] = 0
        self.total_rewards[game_id] = 0.0

        # Send initial state
        self._send_state(game_id, obs, 0.0, False, info)

    def _handle_get_state(self, game_id):
        """Send current state to client"""
        if self.current_states[game_id] is None:
            # Not initialized yet, reset first
            self._handle_reset(game_id)
        else:
            obs, reward, done, info = self.current_states[game_id]
            self._send_state(game_id, obs, reward, done, info)

    def _send_state(self, game_id, obs, reward, done, info):
        """Send state message to client"""
        # Encode frame as base64
        frame_base64 = None
        if obs is not None:
            frame_bytes = obs.tobytes()
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

        # Extract telemetry from CarRacing state
        # Note: CarRacing-v3 returns pixel observations, not telemetry
        # We'll provide basic info from the environment
        telemetry = {
            'steps': self.episode_steps[game_id],
            'total_reward': self.total_rewards[game_id]
        }

        message = {
            'type': 'STATE',
            'game_id': game_id,
            'frame': frame_base64,
            'frame_shape': list(obs.shape) if obs is not None else None,
            'telemetry': telemetry,
            'reward': float(reward),
            'done': bool(done),
            'info': info
        }

        self._send_message(message)

    def _send_message(self, message):
        """Send JSON message to client"""
        try:
            message_str = json.dumps(message) + '\n'
            self.client_socket.sendall(message_str.encode('utf-8'))
        except Exception as e:
            if self.verbose:
                print(f"Error sending message: {e}")

    def stop(self):
        """Stop the server"""
        if self.verbose:
            print("Stopping server...")

        self.running = False
        self.game_running = False

        if self.game_thread:
            self.game_thread.join(timeout=2)

        for env in self.envs:
            env.close()

        if self.client_socket:
            self.client_socket.close()

        if self.server_socket:
            self.server_socket.close()

        if self.verbose:
            print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description='CarRacing Socket Server (Gymnasium)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--num-games', type=int, default=1, help='Number of parallel games')
    parser.add_argument('--render', type=str, default='human',
                       choices=['human', 'rgb_array', 'none'], help='Render mode')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1], help='Verbosity')

    args = parser.parse_args()

    render_mode = None if args.render == 'none' else args.render

    server = CarRacingSocketServer(
        host=args.host,
        port=args.port,
        num_games=args.num_games,
        render_mode=render_mode,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose=args.verbose
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()