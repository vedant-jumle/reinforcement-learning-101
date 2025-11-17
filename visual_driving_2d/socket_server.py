#!/usr/bin/env python
"""
Main Socket Server Script for Car Racing Game

Runs game instances with free-running loop at 50 FPS
Accepts commands from remote clients over TCP sockets
"""

import argparse
import time
import sys

from game.game_engine import RacingGame
from game.socket_server import GameSocketServer
from game import config


def main():
    parser = argparse.ArgumentParser(description='Car Racing Socket Server')

    parser.add_argument('--host', type=str, default=config.SOCKET_HOST,
                        help=f'Server bind address (default: {config.SOCKET_HOST})')
    parser.add_argument('--port', type=int, default=config.SOCKET_PORT,
                        help=f'Server port (default: {config.SOCKET_PORT})')
    parser.add_argument('--num-games', type=int, default=1,
                        help='Number of parallel game instances (default: 1)')
    parser.add_argument('--render', type=str, default='human',
                        choices=['human', 'rgb_array', 'none'],
                        help='Render mode (default: human)')
    parser.add_argument('--render-all', action='store_true',
                        help='Render all game instances (only with num-games > 1)')
    parser.add_argument('--max-steps', type=int, default=config.DEFAULT_EPISODE_STEPS,
                        help=f'Maximum steps per episode (default: {config.DEFAULT_EPISODE_STEPS})')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for track generation')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1],
                        help='Print debug info (default: 1)')

    args = parser.parse_args()

    # Convert render mode
    render_mode = None if args.render == 'none' else args.render

    # Create game instances
    print(f"[Main] Creating {args.num_games} game instance(s)...")
    games = []

    for i in range(args.num_games):
        # Only render first game unless --render-all
        if args.render_all or i == 0:
            mode = render_mode
        else:
            mode = None

        game = RacingGame(
            render_mode=mode,
            total_episode_steps=args.max_steps,
            verbose=args.verbose if i == 0 else 0,  # Only first game prints
            seed=args.seed + i if args.seed is not None else None
        )

        # Initialize game
        game.reset()
        games.append(game)

        print(f"  Game {i}: render_mode={mode}")

    # Create socket server
    server = GameSocketServer(games, host=args.host, port=args.port)

    try:
        # Start server
        server.start()

        # Main game loop (free-running at 50 FPS)
        print(f"[Main] Starting game loop at {config.FPS} FPS...")
        print("[Main] Press Ctrl+C to stop")

        frame_time = 1.0 / config.FPS
        frame_count = 0

        while server.is_connected():
            frame_start = time.time()

            # Process network commands (non-blocking)
            server.process_commands()

            # Step all games
            for game_id, game in enumerate(games):
                # Get latest action from client
                action = server.get_action(game_id)

                # Step game
                state, reward, done, info = game.step(action)

                # Auto-reset if done
                if done:
                    if game.verbose:
                        print(f"\n[Game {game_id}] Episode finished!")
                        print(f"  Steps: {game.current_steps}")
                        print(f"  Tiles visited: {game.tile_visited_count}/{len(game.track)}")
                        print(f"  Total reward: {game.reward:.1f}")
                    game.reset()

                # Send state if requested
                server.send_state_if_requested(game_id)

                # Render if needed
                if game.render_mode == 'human':
                    game.render('human')

            frame_count += 1

            # FPS limiting
            elapsed = time.time() - frame_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Print stats every 100 frames
            if frame_count % 100 == 0:
                actual_fps = 1.0 / max(elapsed, 0.001)
                print(f"[Main] Frame {frame_count}, FPS: {actual_fps:.1f}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")

    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("[Main] Shutting down...")
        server.stop()

        for game in games:
            game.close()

        print("[Main] Goodbye!")


if __name__ == '__main__':
    main()
