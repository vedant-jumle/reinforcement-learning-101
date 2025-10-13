"""Road network generation and queries"""

import math
import numpy as np
from . import config


class RoadSegment:
    """A single road segment (straight or curved)"""

    def __init__(self, start_x, start_y, length, heading, curvature, num_lanes, lane_width):
        """Initialize road segment

        Args:
            start_x, start_y: Starting position
            length: Length of segment (pixels)
            heading: Initial heading (degrees)
            curvature: Curvature factor (0 = straight, >0 = curved)
            num_lanes: Number of parallel lanes
            lane_width: Width of each lane (pixels)
        """
        self.start_x = start_x
        self.start_y = start_y
        self.length = length
        self.heading = heading
        self.curvature = curvature
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.total_width = num_lanes * lane_width

        # Pre-compute road path with analytical tangents
        self.path_points, self.tangents = self._generate_path()
        self.left_edge = self._generate_edge(-self.total_width / 2)
        self.right_edge = self._generate_edge(self.total_width / 2)
        self.lane_centers = [self._generate_edge((i - (num_lanes - 1) / 2) * lane_width)
                            for i in range(num_lanes)]

    def _generate_path(self, num_points=200):
        """Generate points along road centerline (STRAIGHT SEGMENTS ONLY)

        Args:
            num_points: Number of points to generate

        Returns:
            (points, tangents) where:
                points: list of (x, y) tuples
                tangents: list of tangent angles in radians at each point

        Note: Curves are NOT supported. All segments are straight lines.
        """
        points = []
        tangents = []

        heading_rad = math.radians(self.heading)

        for i in range(num_points + 1):
            t = i / num_points
            distance = t * self.length

            # Straight line only
            x = self.start_x + distance * math.cos(heading_rad)
            y = self.start_y - distance * math.sin(heading_rad)

            # Tangent is constant for straight road
            tangent_angle = heading_rad

            points.append((x, y))
            tangents.append(tangent_angle)

        return points, tangents

    def _generate_edge(self, offset):
        """Generate edge line with perpendicular offset from centerline

        Args:
            offset: Lateral offset (positive = left, negative = right in vehicle frame)

        Coordinate system: pygame/screen coordinates where Y increases DOWN
        - Heading 0° = right, 90° = down, 180° = left, 270° = up
        - Tangent at angle θ has direction (cos(θ), -sin(θ))
        - Perpendicular (90° CCW) has direction (sin(θ), cos(θ))

        Uses stored analytical tangent angles for exact perpendicular calculation.
        """
        edge = []
        for i, (cx, cy) in enumerate(self.path_points):
            # Use pre-computed analytical tangent angle
            tangent_angle = self.tangents[i]

            # Apply perpendicular offset
            # For tangent at angle θ, perpendicular direction is (sin(θ), cos(θ))
            ex = cx + offset * math.sin(tangent_angle)
            ey = cy + offset * math.cos(tangent_angle)
            edge.append((ex, ey))

        return edge

    def is_point_on_road(self, x, y):
        """Check if point is within road boundaries"""
        # Find nearest point on centerline
        min_dist = float('inf')
        nearest_idx = 0

        for i, (px, py) in enumerate(self.path_points):
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Check if distance from centerline is within road width
        return min_dist <= self.total_width / 2

    def get_lane_at_position(self, x, y):
        """Get which lane (0 to num_lanes-1) the position is in"""
        if not self.is_point_on_road(x, y):
            return None

        # Find nearest centerline point
        min_dist = float('inf')
        nearest_idx = 0
        for i, (px, py) in enumerate(self.path_points):
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Calculate lateral offset from centerline
        cx, cy = self.path_points[nearest_idx]

        # Use pre-computed analytical tangent angle
        tangent_angle = self.tangents[nearest_idx]

        # Vector from centerline to point
        dx = x - cx
        dy = y - cy

        # Perpendicular offset (signed distance from centerline)
        # Project the vector onto the perpendicular direction (sin(θ), cos(θ))
        lateral_offset = dx * math.sin(tangent_angle) + dy * math.cos(tangent_angle)

        # Convert to lane index
        lane_idx = int((lateral_offset + self.total_width / 2) / self.lane_width)
        lane_idx = np.clip(lane_idx, 0, self.num_lanes - 1)

        return lane_idx


class RoadNetwork:
    """Collection of road segments forming a network"""

    def __init__(self, phase=1):
        """Initialize road network

        Args:
            phase: Game phase (1, 2, or 3)
        """
        self.phase = phase
        self.segments = []
        self.intersections = []  # For Phase 2+
        self.obstacles = []      # For Phase 3+

        # Generate network
        self._generate_network()

        # Generate continuous network-level edges (fixes junction discontinuities)
        self._generate_network_edges()

    def _generate_network(self):
        """Generate road network based on phase"""
        if self.phase == 1:
            self._generate_simple_circuit()
        elif self.phase == 2:
            self._generate_intersection_network()
        elif self.phase == 3:
            self._generate_intersection_network()  # Same as phase 2, obstacles added separately

    def _generate_network_edges(self):
        """Generate continuous edges for entire road network

        This creates network-level left/right edges by combining all segment paths,
        ensuring smooth edge connections at segment junctions (no gaps or overlaps).
        """
        if len(self.segments) == 0:
            self.network_left_edge = []
            self.network_right_edge = []
            self.network_lane_centers = []
            return

        # Combine all segment path points and tangents into one continuous path
        combined_points = []
        combined_tangents = []

        for segment in self.segments:
            # Add all points from this segment (skip first point after first segment to avoid duplicates)
            if len(combined_points) == 0:
                combined_points.extend(segment.path_points)
                combined_tangents.extend(segment.tangents)
            else:
                combined_points.extend(segment.path_points[1:])  # Skip duplicate start point
                combined_tangents.extend(segment.tangents[1:])

        # Get lane width from first segment (all segments should have same width)
        if len(self.segments) > 0:
            total_width = self.segments[0].total_width
            num_lanes = self.segments[0].num_lanes
            lane_width = self.segments[0].lane_width
        else:
            return

        # Generate continuous edges from combined path
        self.network_left_edge = []
        self.network_right_edge = []

        for i, (cx, cy) in enumerate(combined_points):
            tangent = combined_tangents[i]

            # Left edge (negative offset)
            lx = cx + (-total_width / 2) * math.sin(tangent)
            ly = cy + (-total_width / 2) * math.cos(tangent)
            self.network_left_edge.append((lx, ly))

            # Right edge (positive offset)
            rx = cx + (total_width / 2) * math.sin(tangent)
            ry = cy + (total_width / 2) * math.cos(tangent)
            self.network_right_edge.append((rx, ry))

        # Generate continuous lane centers
        self.network_lane_centers = []
        for lane_idx in range(num_lanes):
            lane_offset = (lane_idx - (num_lanes - 1) / 2) * lane_width
            lane_center = []

            for i, (cx, cy) in enumerate(combined_points):
                tangent = combined_tangents[i]
                lx = cx + lane_offset * math.sin(tangent)
                ly = cy + lane_offset * math.cos(tangent)
                lane_center.append((lx, ly))

            self.network_lane_centers.append(lane_center)

    def _generate_simple_circuit(self):
        """Generate simple rectangular circuit for Phase 1 (straight segments only)"""
        self.segments = []

        # Rectangular circuit with 90° corners (NO CURVES)
        # All segments are straight (curvature = 0.0)
        segment_configs = [
            {'length': 400, 'heading': 0},    # Straight right
            {'length': 50,  'heading': 90},   # Corner: turn down
            {'length': 400, 'heading': 90},   # Straight down
            {'length': 50,  'heading': 180},  # Corner: turn left
            {'length': 400, 'heading': 180},  # Straight left
            {'length': 50,  'heading': 270},  # Corner: turn up
            {'length': 400, 'heading': 270},  # Straight up
            {'length': 50,  'heading': 0},    # Corner: turn right
        ]

        # Starting position
        current_x = 150
        current_y = 300
        current_heading = 0

        for i, seg_config in enumerate(segment_configs):
            # Apply randomization
            if config.RANDOMIZE_ROAD_LENGTH:
                length = np.random.uniform(*config.ROAD_LENGTH_RANGE)
            else:
                length = seg_config['length']

            # NO CURVES - always use curvature = 0.0
            curvature = 0.0

            if config.RANDOMIZE_NUM_LANES and i == 0:  # Only randomize once for circuit
                num_lanes = np.random.randint(*config.NUM_LANES_RANGE)
            elif i == 0:
                num_lanes = 2
            else:
                num_lanes = self.segments[0].num_lanes  # Keep same for entire circuit

            if config.RANDOMIZE_LANE_WIDTH and i == 0:
                lane_width = np.random.uniform(*config.LANE_WIDTH_RANGE)
            elif i == 0:
                lane_width = 80
            else:
                lane_width = self.segments[0].lane_width

            # Use heading from config (for straight segments at 90° angles)
            heading = seg_config['heading']

            segment = RoadSegment(
                current_x, current_y,
                length, heading, curvature,
                num_lanes, lane_width
            )
            self.segments.append(segment)

            # Update position and heading for next segment (connect end-to-end)
            if len(segment.path_points) > 1:
                # Get endpoint
                current_x, current_y = segment.path_points[-1]

                # Use analytical tangent at end of this segment (exact, not approximation)
                current_heading = math.degrees(segment.tangents[-1])
                current_heading = current_heading % 360

    def _generate_intersection_network(self):
        """Generate road network with intersections for Phase 2+"""
        # TODO: Phase 2 implementation
        # For now, use simple circuit
        self._generate_simple_circuit()

    def is_point_on_road(self, x, y):
        """Check if point is on any road segment"""
        for segment in self.segments:
            if segment.is_point_on_road(x, y):
                return True
        return False

    def get_segment_and_lane(self, x, y):
        """Get which segment and lane the point is in

        Returns:
            (segment, lane_idx) or (None, None) if not on road
        """
        for segment in self.segments:
            if segment.is_point_on_road(x, y):
                lane_idx = segment.get_lane_at_position(x, y)
                return segment, lane_idx
        return None, None

    def get_random_spawn_position(self):
        """Get random valid spawn position on road"""
        if len(self.segments) == 0:
            # Fallback
            return 400, 300, 0, 0

        # Pick random segment
        segment = np.random.choice(self.segments)

        # Pick random point along segment (avoid very start/end)
        point_idx = np.random.randint(10, len(segment.path_points) - 10)
        cx, cy = segment.path_points[point_idx]

        # Pick random lane if enabled
        if config.RANDOMIZE_SPAWN_LANE:
            lane_idx = np.random.randint(segment.num_lanes)
        else:
            lane_idx = 0  # Default to first lane

        # Get lane center position
        if lane_idx < len(segment.lane_centers):
            lx, ly = segment.lane_centers[lane_idx][point_idx]
        else:
            lx, ly = cx, cy

        # Get heading at this point using stored analytical tangent
        heading = math.degrees(segment.tangents[point_idx])

        # Add random heading variation if enabled
        if config.RANDOMIZE_SPAWN_HEADING:
            heading += np.random.uniform(*config.SPAWN_HEADING_RANGE)

        return lx, ly, heading, lane_idx
