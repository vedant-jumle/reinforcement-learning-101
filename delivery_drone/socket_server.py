#!/usr/bin/env python3
"""Socket server for remote Delivery Drone control

This script starts a game server that clients can connect to over TCP sockets.
Clients can send actions and receive observations remotely.

Usage:
    # Visual mode (watch the game)
    python socket_server.py --render human

    # Headless mode (faster, no visualization)
    python socket_server.py --render none

    # Custom port
    python socket_server.py --port 5556

    # Custom host (for remote connections)
    python socket_server.py --host 0.0.0.0
"""

import argparse
import sys
import signal
from game.game_engine import DroneGame
from game.socket_server import GameSocketServer


def main():
    """Main server entry point"""
    parser = argparse.ArgumentParser(
        description='Delivery Drone Socket Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0 for all interfaces)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5555,
        help='Port to listen on (default: 5555)'
    )

    parser.add_argument(
        '--render',
        type=str,
        choices=['human', 'rgb_array', 'none'],
        default='human',
        help='Render mode: human (visual), rgb_array (offscreen), none (headless)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='Target FPS for game loop (default: 60)'
    )

    parser.add_argument(
        '--num-games',
        type=int,
        default=1,
        help='Number of parallel game instances (default: 1)'
    )

    parser.add_argument(
        '--render-all',
        action='store_true',
        help='Render all game instances in separate windows (default: only render game 0)'
    )

    parser.add_argument(
        '--randomize-drone',
        action='store_true',
        help='Randomize drone spawn position on reset'
    )

    parser.add_argument(
        '--randomize-platform',
        action='store_true',
        default=True,
        help='Randomize platform position on reset (default: True)'
    )

    parser.add_argument(
        '--fixed-spawn',
        action='store_true',
        help='Disable all spawn randomization (drone and platform)'
    )

    args = parser.parse_args()

    # Handle spawn randomization flags
    if args.fixed_spawn:
        randomize_drone = False
        randomize_platform = False
    else:
        randomize_drone = args.randomize_drone
        randomize_platform = args.randomize_platform

    # Create games
    render_mode = args.render if args.render != 'none' else None

    print(f"Starting {args.num_games} Delivery Drone game(s) (render={args.render})...")
    print(f"Spawn settings: drone_random={randomize_drone}, platform_random={randomize_platform}")

    if args.render_all and args.num_games > 1:
        # Render all games using rgb_array mode (we'll composite them)
        games = [DroneGame(render_mode='rgb_array' if render_mode else None,
                          randomize_drone=randomize_drone,
                          randomize_platform=randomize_platform) for i in range(args.num_games)]
        if render_mode is not None:
            print(f"Note: Rendering all {args.num_games} games in a grid layout")
    else:
        # Render only game 0
        games = [DroneGame(render_mode=render_mode if i == 0 else None,
                          randomize_drone=randomize_drone,
                          randomize_platform=randomize_platform) for i in range(args.num_games)]
        if args.num_games > 1 and render_mode is not None:
            print(f"Note: Only game 0 will render (use --render-all to render all games)")

    # Create socket server
    print(f"Starting socket server on {args.host}:{args.port}...")
    server = GameSocketServer(games, host=args.host, port=args.port)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal")
        server.stop()
        for game in games:
            game.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start server (blocks until client connects)
        server.start()

        # Reset all games
        for game in games:
            game.reset()

        print("\nGame server running!")
        print(f"Managing {len(games)} parallel game(s)")
        print("Press Ctrl+C to shutdown")
        print("="*50)

        # Setup composite rendering if needed
        composite_screen = None
        if args.render_all and args.num_games > 1 and render_mode is not None:
            import pygame
            import numpy as np
            import math

            pygame.init()

            # Calculate grid dimensions
            cols = math.ceil(math.sqrt(args.num_games))
            rows = math.ceil(args.num_games / cols)

            # Get game dimensions from first game
            game_width = 800  # config.WINDOW_WIDTH
            game_height = 600  # config.WINDOW_HEIGHT

            # Create composite window
            composite_width = game_width * cols
            composite_height = game_height * rows
            composite_screen = pygame.display.set_mode((composite_width, composite_height))
            pygame.display.set_caption(f"Delivery Drone - {args.num_games} Games")

            print(f"Composite window: {cols}x{rows} grid ({composite_width}x{composite_height})")

        # Main game loop
        import time
        frame_time = 1.0 / args.fps
        last_frame = time.time()

        while server.is_connected():
            loop_start = time.time()

            # Process any pending commands from client
            server.process_commands()

            # Render game(s)
            if render_mode is not None:
                if args.render_all and args.num_games > 1 and composite_screen:
                    # Render all games in grid
                    import numpy as np
                    import math

                    cols = math.ceil(math.sqrt(args.num_games))

                    for idx, game in enumerate(games):
                        # Get game frame as RGB array
                        game.render()
                        frame = pygame.surfarray.array3d(game.screen)
                        frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height, channels)

                        # Calculate position in grid
                        col = idx % cols
                        row = idx // cols
                        x = col * game_width
                        y = row * game_height

                        # Blit to composite screen
                        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                        composite_screen.blit(surf, (x, y))

                        # Draw game ID label
                        font = pygame.font.Font(None, 36)
                        label = font.render(f"Game {idx}", True, (255, 255, 0))
                        composite_screen.blit(label, (x + 10, y + 10))

                    pygame.display.flip()
                else:
                    # Render only game 0
                    if len(games) > 0:
                        games[0].render()

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Track actual FPS
            current_time = time.time()
            actual_fps = 1.0 / max(0.001, current_time - last_frame)
            last_frame = current_time

            # Print FPS occasionally (every 60 frames) - show game 0 stats
            if len(games) > 0 and games[0].steps % 60 == 0 and games[0].steps > 0:
                print(f"\rGame[0] Episode {games[0].episode} | Step {games[0].steps} | FPS: {actual_fps:.1f} | Connected: {server.is_connected()}", end='')

        print("\n\nClient disconnected")

    except KeyboardInterrupt:
        print("\nShutdown requested")

    finally:
        # Cleanup
        server.stop()
        for game in games:
            game.close()
        print("Server shutdown complete")


if __name__ == "__main__":
    main()
