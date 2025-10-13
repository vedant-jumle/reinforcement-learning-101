#!/usr/bin/env python3
"""Benchmark socket latency

Measures the round-trip latency of socket communication.
Start the server first: python socket_server.py --render none
"""

import sys
sys.path.insert(0, '..')

from game.socket_client import benchmark_latency
import argparse


def main():
    """Run latency benchmark"""
    parser = argparse.ArgumentParser(description='Benchmark socket latency')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to benchmark')

    args = parser.parse_args()

    print("=" * 60)
    print("SOCKET LATENCY BENCHMARK")
    print("=" * 60)
    print(f"\nServer: {args.host}:{args.port}")
    print(f"Steps:  {args.steps}")
    print("\nMake sure the server is running:")
    print("  python socket_server.py --render none")
    print("\n" + "=" * 60)

    try:
        stats = benchmark_latency(args.host, args.port, args.steps)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)

        # Estimate max FPS
        max_fps = 1000.0 / stats['step_mean_ms']
        print(f"\nEstimated max throughput: {max_fps:.1f} FPS")
        print(f"(Based on mean latency of {stats['step_mean_ms']:.2f}ms)")

    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        print("\nMake sure the server is running:")
        print("  python socket_server.py --render none")


if __name__ == "__main__":
    main()
