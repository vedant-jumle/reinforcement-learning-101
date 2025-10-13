"""Collision detection for all phases"""

import math
from . import config


class CollisionDetector:
    """Detects crashes and dangerous situations"""

    def __init__(self, phase):
        self.phase = phase

    def check_crashes(self, vehicle, road_network, **kwargs):
        """Check all crash conditions for current phase

        Returns:
            Dict with crash info: {'terminal': bool, 'reason': str, ...}
        """
        if self.phase == 1:
            return self._check_phase1(vehicle, road_network)
        elif self.phase == 2:
            return self._check_phase2(vehicle, road_network, kwargs.get('goal'))
        elif self.phase == 3:
            return self._check_phase3(vehicle, road_network, kwargs.get('goal'), kwargs.get('obstacles'))

    def _check_phase1(self, vehicle, road_network):
        """Phase 1: Basic driving crashes"""
        crashes = {
            'terminal': False,
            'reason': None,
            'off_road': False,
            'wrong_lane': False,
            'out_of_bounds': False
        }

        # 1. Check if on road
        segment, lane = road_network.get_segment_and_lane(vehicle.x, vehicle.y)

        if segment is None:
            crashes['off_road'] = True
            crashes['terminal'] = True
            crashes['reason'] = 'Vehicle drove off road'
            return crashes

        # 2. Check if in correct lane
        if vehicle.assigned_lane is not None and lane != vehicle.assigned_lane:
            crashes['wrong_lane'] = True
            # Not terminal, just penalty

        # 3. Check out of bounds (large margin from road network)
        # The road network can extend beyond window bounds (camera follows vehicle)
        # Only fail if VERY far from road (e.g., 1000px away)
        margin = 1000

        # Find nearest road point to check distance
        min_dist_to_road = float('inf')
        for seg in road_network.segments:
            for px, py in seg.path_points:
                dist = math.hypot(vehicle.x - px, vehicle.y - py)
                min_dist_to_road = min(min_dist_to_road, dist)

        if min_dist_to_road > margin:
            crashes['out_of_bounds'] = True
            crashes['terminal'] = True
            crashes['reason'] = f'Vehicle too far from road ({min_dist_to_road:.0f}px)'
            return crashes

        return crashes

    def _check_phase2(self, vehicle, road_network, goal):
        """Phase 2: Navigation crashes (includes Phase 1)"""
        crashes = self._check_phase1(vehicle, road_network)
        if crashes['terminal']:
            return crashes

        # TODO: Implement Phase 2 specific crashes
        # - Wrong turn at intersection
        # - Driving opposite direction

        return crashes

    def _check_phase3(self, vehicle, road_network, goal, obstacles):
        """Phase 3: Obstacle crashes (includes Phase 1 & 2)"""
        crashes = self._check_phase2(vehicle, road_network, goal)
        if crashes['terminal']:
            return crashes

        # Check obstacle collisions
        if obstacles:
            for obstacle in obstacles:
                if self._vehicle_obstacle_collision(vehicle, obstacle):
                    crashes['obstacle_collision'] = True
                    crashes['terminal'] = True
                    crashes['reason'] = f'Collision with {obstacle.type}'
                    crashes['obstacle'] = obstacle
                    return crashes

                # Check proximity
                dist = math.hypot(vehicle.x - obstacle.x, vehicle.y - obstacle.y)
                if dist < config.SAFE_DISTANCE_THRESHOLD:
                    crashes['too_close_to_obstacle'] = True

        return crashes

    def _vehicle_obstacle_collision(self, vehicle, obstacle):
        """Check if vehicle collides with obstacle using rectangle intersection"""
        # Get vehicle corners
        vehicle_corners = vehicle.get_corners()

        # Get obstacle bounding box
        obs_left = obstacle.x - obstacle.width / 2
        obs_right = obstacle.x + obstacle.width / 2
        obs_top = obstacle.y - obstacle.height / 2
        obs_bottom = obstacle.y + obstacle.height / 2

        # Check if any vehicle corner is inside obstacle bounding box
        for vx, vy in vehicle_corners:
            if (obs_left <= vx <= obs_right and obs_top <= vy <= obs_bottom):
                return True

        # Check if obstacle center is inside vehicle (for small obstacles)
        # Simple point-in-polygon test
        return False
