#!/usr/bin/env python3
"""Socket server for Visual Driving 2D

Usage:
    # Phase 1, single game, visual
    python socket_server.py --phase 1 --render human

    # Phase 3, 6 parallel games, headless
    python socket_server.py --phase 3 --num-games 6 --render none \\
        --randomize-spawn --randomize-obstacles --min-obstacles 5 --max-obstacles 10
"""

import argparse
import signal
import sys
import time
from game.game_engine import DrivingGame
from game.socket_server import DrivingGameSocketServer
from game import config


def main():
    parser = argparse.ArgumentParser(description='Visual Driving 2D Socket Server')

    # Phase selection
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                       help='Game phase (1=basic, 2=navigation, 3=obstacles)')

    # Server config
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5555,
                       help='Port to listen on')
    parser.add_argument('--num-games', type=int, default=1,
                       help='Number of parallel game instances')

    # Rendering
    parser.add_argument('--render', type=str, default='human',
                       choices=['human', 'rgb_array', 'none'],
                       help='Render mode')
    parser.add_argument('--fps', type=int, default=60,
                       help='Physics FPS')

    # Feature toggles - Phase 1
    parser.add_argument('--enable-curves', action='store_true', default=True,
                       help='Enable curved road segments')
    parser.add_argument('--enable-multiple-lanes', action='store_true', default=True,
                       help='Enable multiple lanes')

    # Feature toggles - Phase 2
    parser.add_argument('--enable-intersections', action='store_true',
                       help='Enable intersections (auto-enabled for phase 2+)')
    parser.add_argument('--enable-goal', action='store_true',
                       help='Enable goal navigation (auto-enabled for phase 2+)')

    # Feature toggles - Phase 3
    parser.add_argument('--enable-obstacles', action='store_true',
                       help='Enable obstacles (auto-enabled for phase 3)')
    parser.add_argument('--enable-parked-cars', action='store_true', default=True,
                       help='Enable parked car obstacles')
    parser.add_argument('--enable-barriers', action='store_true', default=True,
                       help='Enable barrier obstacles')
    parser.add_argument('--enable-cones', action='store_true', default=True,
                       help='Enable cone obstacles')

    # Randomization - Spawn
    parser.add_argument('--randomize-spawn', action='store_true', default=True,
                       help='Randomize vehicle spawn position')
    parser.add_argument('--randomize-heading', action='store_true', default=True,
                       help='Randomize vehicle spawn heading')

    # Randomization - Roads
    parser.add_argument('--randomize-road-layout', action='store_true', default=True,
                       help='Randomize road network layout')
    parser.add_argument('--randomize-curvature', action='store_true', default=True,
                       help='Randomize road curvature')
    parser.add_argument('--randomize-lane-width', action='store_true', default=True,
                       help='Randomize lane widths')

    # Randomization - Phase 2
    parser.add_argument('--randomize-goal', action='store_true', default=True,
                       help='Randomize goal position')

    # Randomization - Phase 3
    parser.add_argument('--randomize-obstacles', action='store_true', default=True,
                       help='Randomize obstacle count and positions')
    parser.add_argument('--min-obstacles', type=int, default=3,
                       help='Minimum number of obstacles')
    parser.add_argument('--max-obstacles', type=int, default=10,
                       help='Maximum number of obstacles')

    args = parser.parse_args()

    # Apply config overrides
    config.PHASE = args.phase
    config.PHYSICS_FPS = args.fps
    config.ENABLE_CURVES = args.enable_curves
    config.ENABLE_MULTIPLE_LANES = args.enable_multiple_lanes

    # Auto-enable features based on phase
    if args.phase >= 2:
        config.ENABLE_INTERSECTIONS = True
        config.ENABLE_GOAL_NAVIGATION = True

    if args.phase >= 3:
        config.ENABLE_STATIC_OBSTACLES = True
        config.ENABLE_PARKED_CARS = args.enable_parked_cars
        config.ENABLE_BARRIERS = args.enable_barriers
        config.ENABLE_CONES = args.enable_cones

    # Randomization
    config.RANDOMIZE_SPAWN_POSITION = args.randomize_spawn
    config.RANDOMIZE_SPAWN_HEADING = args.randomize_heading
    config.RANDOMIZE_ROAD_LAYOUT = args.randomize_road_layout
    config.RANDOMIZE_ROAD_CURVATURE = args.randomize_curvature
    config.RANDOMIZE_LANE_WIDTH = args.randomize_lane_width
    config.RANDOMIZE_GOAL_POSITION = args.randomize_goal
    config.RANDOMIZE_OBSTACLE_COUNT = args.randomize_obstacles
    config.MIN_OBSTACLES = args.min_obstacles
    config.MAX_OBSTACLES = args.max_obstacles

    # Create games
    render_mode = None if args.render == 'none' else args.render

    print(f"Creating {args.num_games} game instance(s) - Phase {args.phase}")
    games = []
    for i in range(args.num_games):
        # Only render first game in human mode
        game_render = render_mode if (i == 0 and render_mode == 'human') else None
        game = DrivingGame(phase=args.phase, render_mode=game_render)
        games.append(game)

    # Create server
    server = DrivingGameSocketServer(games, host=args.host, port=args.port)

    # Signal handling
    def signal_handler(sig, frame):
        print("\nShutting down...")
        server.stop()
        for game in games:
            game.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start server
    try:
        server.start()

        print("Server running. Press Ctrl+C to stop.")
        print("="*50)

        # Keep main thread alive
        while server.is_connected():
            if render_mode == 'human' and len(games) > 0:
                games[0].render()
            time.sleep(0.016)  # ~60 FPS render

    except KeyboardInterrupt:
        print("\nKeyboard interrupt")
    finally:
        server.stop()
        for game in games:
            game.close()

if __name__ == '__main__':
    main()
