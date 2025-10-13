# reinforcement-learning-101
Repository for all of my reinforcement learning escapades

A physics-based drone landing game designed for reinforcement learning research and experimentation. Control a drone to safely land on platforms while managing fuel, speed, and angle.

![Game Preview](https://via.placeholder.com/800x400/87CEEB/000000?text=Drone+Landing+Game)

## Features

- **Clean RL API** - Simple gym-style interface for training agents
- **Socket Interface** - Train remotely over TCP for distributed RL
- **Parallel Training** - Run multiple game instances simultaneously
- **Randomization** - Randomize drone/platform spawns for robust policies
- **Physics-based** - Realistic gravity, thrust, drag, and rotation
- **Manual Play** - Test the challenge yourself with keyboard controls

## Quick Start

```bash
# Install dependencies
pip install pygame numpy torch

# Play manually to understand the task
cd delivery_drone
python manual_play.py

# Run parallel training server (6 games)
python socket_server.py --num-games 6 --render none

# Train an RL agent (in another terminal)
python train_agent.py
```

## Project Structure

```
delivery_drone/
├── delivery_drone/        # Main package
│   ├── game/             # Game engine
│   ├── examples/         # Example agents
│   ├── models/           # Saved models
│   └── docs/             # Documentation
├── RL_1.ipynb            # Training notebook (example)
└── README.md             # This file
```

## Documentation

See [delivery_drone/README.md](delivery_drone/README.md) for:
- Complete API documentation
- State space and action space details
- Reward structure
- Socket API reference
- Example agents and training tips

## Landing Criteria

Successfully land by meeting ALL conditions:
- **Speed** ≤ 3.0 pixels/frame (0.3 normalized)
- **Angle** ≤ 20° from upright (0.111 normalized)
- **Position**: Bottom center of drone on platform
- **Fuel**: Must have fuel remaining

## RL Training Tips

1. **Reward shaping** - Avoid continuous proximity rewards (causes hovering exploits)
2. **Horizontal alignment** - Platform is only 100px wide, alignment is critical
3. **Parallel environments** - Use `--num-games 6-12` for faster training
4. **Randomization** - Use `--randomize-drone` and `--randomize-platform` for robust policies
5. **Velocity alignment** - Reward moving *toward* platform, not just being near it

## Training Features

### Parallel Game Instances
```bash
# Run 6 parallel games for faster data collection
python socket_server.py --num-games 6 --render none
```

### Spawn Randomization
```bash
# Randomize both drone and platform positions
python socket_server.py --randomize-drone --randomize-platform

# Fixed spawns for debugging
python socket_server.py --fixed-spawn

# Platform random, drone fixed (default)
python socket_server.py
```

## Example Training Code

```python
from delivery_drone.game.socket_client import DroneGameClient

client = DroneGameClient()
client.connect()

# Collect episodes from all parallel games
for episode in range(num_episodes):
    for game_id in range(client.num_games):
        state = client.reset(game_id)
        done = False

        while not done:
            action = policy(state)
            state, reward, done, info = client.step(action, game_id)
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

This is a learning project. Feel free to fork, experiment, and share your trained policies!

## Acknowledgments

Built as a reinforcement learning research environment for exploring policy gradient methods, reward shaping, and curriculum learning.
