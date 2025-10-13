# Getting Started with Delivery Drone

Welcome! This guide will help you get started with the Delivery Drone game.

## What is This?

Delivery Drone is a **physics-based game** where you control a drone trying to land safely on a platform. It's designed to be:

1. **Fun to play** - Challenge yourself to perfect landings
2. **Easy to understand** - Simple controls and clear objectives
3. **Perfect for learning RL** - Clean API for building AI agents

## Installation (30 seconds)

```bash
# From the delivery-drone directory
pip install pygame numpy
```

That's it! No complex dependencies.

## Quick Start Guide

### Step 1: Play the Game (5 minutes)

Learn what the challenge is:

```bash
python manual_play.py
```

**Controls:**
- W/↑ = Main thrust (upward)
- A/← = Rotate left
- D/→ = Rotate right
- R = Reset
- ESC = Quit

**Goal:** Land on the green platform (marked "H") while:
- Keeping speed low (< 3.0)
- Staying upright (< 20° tilt)
- Not running out of fuel

Try to land successfully a few times to understand the physics!

### Step 2: Watch the API Demo (2 minutes)

See how the game interface works:

```bash
PYTHONPATH=. python examples/api_demo.py
```

This shows you:
- How to create a game instance
- What the state looks like
- How to send actions
- What information you get back

### Step 3: Run Example Agents (5 minutes)

See different agent strategies:

```bash
# Random actions (will crash a lot!)
PYTHONPATH=. python examples/random_agent.py

# Simple rule-based agent (better, but still imperfect)
PYTHONPATH=. python examples/simple_agent.py
```

Watch how they perform and think about how you'd improve them!

### Step 4: Inspect the State (5 minutes)

Understand what information is available:

```bash
PYTHONPATH=. python examples/inspect_state.py
```

This prints the game state in real-time. You can control the drone with W/A/D and see how the state changes.

## Understanding the Game

### The Challenge

Landing a drone is **hard** because:
1. Gravity constantly pulls you down
2. Your thrusters affect both position AND rotation
3. You need to be gentle near the platform
4. You have limited fuel

This is why it's perfect for RL - it requires:
- **Continuous control** (like Forza steering/throttle)
- **Balance** (multiple objectives: speed, angle, position, fuel)
- **Planning ahead** (can't just thrust at the last second)

### The State Space

The game gives you 15 pieces of information:

```python
state = {
    # Where you are
    'drone_x': 0.5,              # 0=left edge, 1=right edge
    'drone_y': 0.167,            # 0=top, 1=bottom

    # How you're moving
    'drone_vx': 0.0,             # Horizontal velocity
    'drone_vy': 0.03,            # Vertical velocity (positive = falling)
    'speed': 0.03,               # Total speed magnitude

    # Your orientation
    'drone_angle': 0.0,          # -1=left, 0=upright, 1=right
    'drone_angular_vel': 0.0,    # How fast you're rotating

    # Resources
    'drone_fuel': 1.0,           # 1.0=full, 0.0=empty

    # Target location
    'platform_x': 0.5,
    'platform_y': 0.833,
    'distance_to_platform': 0.5,
    'dx_to_platform': 0.0,       # Horizontal distance
    'dy_to_platform': 0.667,     # Vertical distance

    # Status
    'landed': False,
    'crashed': False
}
```

All values are normalized (0 to 1 or -1 to 1) so they're ready for neural networks!

### The Action Space

You have 3 binary controls:

```python
action = {
    'main_thrust': 0,    # 0=off, 1=on (thrust in direction you're facing)
    'left_thrust': 0,    # 0=off, 1=on (rotate counter-clockwise)
    'right_thrust': 0    # 0=off, 1=on (rotate clockwise)
}
```

Simple, but powerful! All combinations are valid.

### The Rewards

The game gives you feedback:
- **+100** = Successful landing! 🎉
- **-100** = Crashed 💥
- **-50** = Ran out of fuel or went out of bounds
- **-0.1** = Each step (encourages efficiency)
- **+0.0-0.1** = Small bonus for getting closer to platform

## Building Your First Agent

### Option 1: Rule-Based Agent (Start Here!)

Open `examples/simple_agent.py` and modify the `simple_policy()` function:

```python
def simple_policy(state):
    action = {'main_thrust': 0, 'left_thrust': 0, 'right_thrust': 0}

    # Your strategy here!
    # Example: If falling too fast, thrust
    if state['drone_vy'] > 0.3:
        action['main_thrust'] = 1

    return action
```

Try to beat the existing simple agent's success rate!

### Option 2: Reinforcement Learning

Ready for RL? Here's the minimal code:

```python
from game.game_engine import DroneGame

# Create environment
env = DroneGame(render_mode='human')

# Training loop
for episode in range(100):
    state = env.reset()
    done = False

    while not done:
        # Your RL agent decides action
        action = your_policy(state)  # TODO: implement this

        # Step environment
        next_state, reward, done, info = env.step(action)

        # Train your agent
        your_agent.learn(state, action, reward, next_state, done)

        state = next_state
        env.render()
```

Later, you can wrap this in Gymnasium and use Stable-Baselines3!

### Option 3: Imitation Learning

Learn from human demonstrations:

```bash
# Record yourself playing
PYTHONPATH=. python examples/record_gameplay.py

# This saves your gameplay to .pkl files
# You can then train an agent to imitate you!
```

## Tips for Success

### For Manual Play:
1. **Start high** - Give yourself time to correct
2. **Pulse thrust** - Tap W, don't hold it
3. **Stay upright** - Fix your angle before worrying about position
4. **Slow down early** - Kill velocity before reaching platform
5. **Watch the HUD** - Speed and angle are your critical metrics

### For Building Agents:
1. **Play the game first** - You need to understand the challenge
2. **Start with rules** - A simple policy is better than random
3. **Use the state** - All the info you need is in the state dict
4. **Watch it train** - Visual feedback helps debug
5. **Iterate quickly** - The game runs fast, test ideas rapidly

## Next Steps

1. ✅ **Play manually** - Get a few successful landings
2. ✅ **Run examples** - See what's possible
3. ✅ **Read the code** - Look at `game/drone.py` and `game/game_engine.py`
4. ✅ **Modify config** - Try changing physics in `game/config.py`
5. ✅ **Build an agent** - Start with rules, move to RL

## Common Questions

**Q: Why isn't the simple agent perfect?**
A: Because it's... simple! It doesn't plan ahead or optimize. That's where ML comes in.

**Q: Can I change the physics?**
A: Yes! Edit `game/config.py` - change gravity, thrust power, fuel, etc.

**Q: Why is this good for learning RL?**
A: It's:
- Fast (hundreds of episodes per minute)
- Visual (you can see what's happening)
- Challenging but not impossible
- Similar to real control problems (like driving!)

**Q: What's the connection to your Forza project?**
A: Same concepts:
- Continuous control (steering/throttle = thrust/rotation)
- Physics-based (momentum, friction)
- Multi-objective (speed + accuracy)
- State → Action → Reward → Learn

## Getting Help

- Read the code - it's well-commented!
- Check `README.md` for API reference
- Modify and experiment - nothing will break!
- The game is deterministic - same actions = same results

## Have Fun!

This is a **learning project**. The goal is to:
1. Have fun playing
2. Understand RL concepts
3. Build intuition for agent design
4. Apply to harder problems (like Forza!)

Don't worry about perfect performance. Focus on learning!

Good luck! 🚁
