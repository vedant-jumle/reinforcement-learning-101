#!/bin/bash
# Quick launcher for Delivery Drone game

echo "╔════════════════════════════════════════════╗"
echo "║      DELIVERY DRONE GAME LAUNCHER          ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "What would you like to do?"
echo ""
echo "1) Play manually (keyboard controls)"
echo "2) Watch random agent"
echo "3) Watch simple rule-based agent"
echo "4) Run API demo (no visuals)"
echo "5) Inspect state in real-time"
echo "6) Record gameplay for imitation learning"
echo "7) Run tests"
echo ""
read -p "Enter choice (1-7): " choice

case $choice in
    1)
        echo "Starting manual play mode..."
        python manual_play.py
        ;;
    2)
        echo "Starting random agent..."
        PYTHONPATH=. python examples/random_agent.py
        ;;
    3)
        echo "Starting simple rule-based agent..."
        PYTHONPATH=. python examples/simple_agent.py
        ;;
    4)
        echo "Running API demo..."
        PYTHONPATH=. python examples/api_demo.py
        ;;
    5)
        echo "Starting state inspector..."
        PYTHONPATH=. python examples/inspect_state.py
        ;;
    6)
        echo "Starting gameplay recorder..."
        PYTHONPATH=. python examples/record_gameplay.py
        ;;
    7)
        echo "Running tests..."
        python test_game.py
        ;;
    *)
        echo "Invalid choice. Run: python manual_play.py"
        ;;
esac
