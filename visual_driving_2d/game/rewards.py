"""Reward calculation for each phase"""

import math
from . import config


class RewardCalculator:
    """Calculate rewards based on phase"""

    def __init__(self, phase):
        self.phase = phase

    def calculate_reward(self, vehicle, road_network, crashes, prev_state, **kwargs):
        """Calculate reward for current timestep

        Returns:
            Float reward value
        """
        if self.phase == 1:
            return self._calculate_phase1(vehicle, road_network, crashes, prev_state)
        elif self.phase == 2:
            return self._calculate_phase2(vehicle, road_network, crashes, prev_state, kwargs.get('goal'))
        elif self.phase == 3:
            return self._calculate_phase3(vehicle, road_network, crashes, prev_state,
                                         kwargs.get('goal'), kwargs.get('obstacles'))

    def _calculate_phase1(self, vehicle, road_network, crashes, prev_state):
        """Phase 1: Basic driving rewards"""
        reward = 0.0

        # Terminal penalties
        if crashes['terminal']:
            if crashes['off_road']:
                return config.CRASH_PENALTY_OFF_ROAD
            elif crashes['out_of_bounds']:
                return config.CRASH_PENALTY_OUT_OF_BOUNDS

        # Positive: On road
        segment, lane = road_network.get_segment_and_lane(vehicle.x, vehicle.y)
        if segment is not None:
            reward += config.REWARD_ON_ROAD
        else:
            reward += config.PENALTY_OFF_ROAD

        # Penalty: Wrong lane
        if crashes['wrong_lane']:
            reward += config.PENALTY_CROSSED_LANE

        # Reward: Forward progress
        if prev_state:
            distance_progress = vehicle.distance_traveled - prev_state['distance_traveled']
            reward += config.REWARD_FORWARD_PROGRESS * distance_progress

        # Penalty: Jerky steering
        reward += config.PENALTY_JERKY_STEERING * abs(vehicle.steering_angle)

        # Penalty: Speed error (encourage optimal speed)
        optimal_speed = 100  # pixels/sec
        if abs(vehicle.velocity) > 0.1:
            speed_error = abs(vehicle.velocity - optimal_speed) / optimal_speed
            reward += config.PENALTY_SPEED_ERROR * speed_error

        return reward

    def _calculate_phase2(self, vehicle, road_network, crashes, prev_state, goal):
        """Phase 2: Navigation rewards (includes Phase 1)"""
        reward = self._calculate_phase1(vehicle, road_network, crashes, prev_state)

        if goal is None:
            return reward

        # Goal reached
        if goal.is_reached(vehicle):
            reward += config.REWARD_GOAL_REACHED

        # Progress toward goal
        if prev_state:
            curr_dist = math.hypot(vehicle.x - goal.x, vehicle.y - goal.y)
            prev_dist = math.hypot(prev_state['x'] - goal.x, prev_state['y'] - goal.y)
            progress = prev_dist - curr_dist
            reward += config.REWARD_PROGRESS_TO_GOAL * progress

        # Time penalty
        reward += config.PENALTY_TIME_STEP

        return reward

    def _calculate_phase3(self, vehicle, road_network, crashes, prev_state, goal, obstacles):
        """Phase 3: Obstacle rewards (includes Phase 1 & 2)"""
        reward = self._calculate_phase2(vehicle, road_network, crashes, prev_state, goal)

        # Obstacle collision
        if crashes.get('obstacle_collision'):
            return config.CRASH_PENALTY_OBSTACLE

        # Too close to obstacle
        if crashes.get('too_close_to_obstacle'):
            reward += config.PENALTY_TOO_CLOSE_OBSTACLE

        return reward
