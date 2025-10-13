"""Debug spawn position generation"""

from game.road import RoadNetwork
from game.vehicle import Vehicle
from game import config

# Disable randomization for testing
config.RANDOMIZE_SPAWN_POSITION = False
config.RANDOMIZE_SPAWN_HEADING = False
config.RANDOMIZE_SPAWN_LANE = False
config.RANDOMIZE_ROAD_LAYOUT = False
config.RANDOMIZE_ROAD_CURVATURE = False
config.RANDOMIZE_ROAD_LENGTH = False
config.RANDOMIZE_LANE_WIDTH = False
config.RANDOMIZE_NUM_LANES = False

print("Creating road network...")
road = RoadNetwork(phase=1)
print(f"Generated {len(road.segments)} segments")

print("\nTesting spawn positions...")
for i in range(10):
    x, y, heading, lane = road.get_random_spawn_position()
    print(f"Spawn {i}: x={x:.1f}, y={y:.1f}, heading={heading:.1f}, lane={lane}")

    # Check if on road
    segment, detected_lane = road.get_segment_and_lane(x, y)
    on_road = segment is not None
    print(f"  On road: {on_road}, Detected lane: {detected_lane}")

    if not on_road:
        print(f"  ERROR: Spawn position off road!")
        # Check distance to nearest path point
        min_dist = float('inf')
        for seg in road.segments:
            for px, py in seg.path_points:
                dist = ((x - px)**2 + (y - py)**2)**0.5
                min_dist = min(min_dist, dist)
        print(f"  Distance to nearest path point: {min_dist:.1f}")
