```py
def calc_reward(state: DroneState):
    rewards = {}
    total_reward = 0
    
    time_step = state.steps
    
    # Time penalty
    minimum_time_penalty = 0.3
    maximum_time_penalty = 1
    rewards['time_penalty'] = -inverse_quadratic(
        state.distance_to_platform, 
        decay=50, 
        scaler=maximum_time_penalty-minimum_time_penalty) - minimum_time_penalty
    
    # Distance-based time penalty
    # Penalty gets smaller as drone gets closer to platform
    # Uses inverse quadratic function: higher penalty when far, reduces as distance decreases
    # Minimum penalty of 0.5, maximum of 2.0 per timestep
    total_reward += rewards['time_penalty']
    
    velocity_alignment = calc_velocity_alignment(state)
    dist = state.distance_to_platform
    
    rewards['distance'] = 0
    rewards['velocity_alignment'] = 0

    if dist > 0.065 and state.dy_to_platform > 0:  # ADD: only if drone ABOVE platform
        rewards['distance'] = int(velocity_alignment > 0) * state.speed * scaled_shifted_negative_sigmoid(dist, scaler=4.5)
        
        if velocity_alignment > 0:
            rewards['velocity_alignment'] = 0.5

    total_reward += rewards['distance']
    total_reward += rewards['velocity_alignment']
    
    # Angle penalty (define a distance based max threshold)
    abs_angle = abs(state.drone_angle)
    max_angle = 0.20
    max_permissible_angle = ((max_angle-0.111)*dist) + 0.111
    excess = abs_angle - max_permissible_angle # excess angle
    rewards['angle'] = -max(excess, 0) # maximum reward is 0 (we dont want it to reward hack for stability)
    
    total_reward += rewards['angle']
    
    # Speed - penalize excessive speed
    rewards['speed'] = 0
    speed = state.speed
    max_speed = 0.4
    if dist < 1:
        rewards['speed'] = -2 * max(speed-0.1, 0)
    else:
        rewards['speed'] = -1 * max(speed-max_speed, 0)
    total_reward += rewards['speed']
    
    # Penalize being below platform
    rewards['vertical_position'] = 0
    if state.dy_to_platform > 0:  # Platform is below drone (drone is above - GOOD)
        rewards['vertical_position'] = 0
    else:  # Drone is below platform (BAD!)
        rewards['vertical_position'] = state.dy_to_platform * 4.0  # Negative penalty
    total_reward += rewards['vertical_position']
    
    # Terminal
    rewards['terminal'] = 0
    if state.landed:
        rewards['terminal'] = 500.0 + state.drone_fuel * 100.0
    elif state.crashed:
        rewards['terminal'] = -200.0
        # Extra penalty for crashing far from target
        if state.distance_to_platform > 0.3:
            rewards['terminal'] -= 100.0
    total_reward += rewards['terminal']
    
    rewards['total'] = total_reward
    return rewards
```
