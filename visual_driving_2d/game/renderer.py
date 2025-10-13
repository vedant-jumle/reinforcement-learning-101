"""Pygame rendering engine"""

import pygame
import math
from . import config


class Renderer:
    """Renders game state to pygame surface"""

    def __init__(self, screen):
        """Initialize renderer

        Args:
            screen: Pygame surface to render to (can be None for headless)
        """
        self.screen = screen
        if screen is not None:
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

    def render(self, surface, vehicle, road_network, camera):
        """Render complete game state

        Args:
            surface: Pygame surface to draw on
            vehicle: Vehicle object
            road_network: RoadNetwork object
            camera: Camera object
        """
        if surface is None:
            return

        # Update camera
        camera.follow_vehicle(vehicle)

        # Clear screen (grass)
        surface.fill(config.COLOR_GRASS)

        # Draw road network
        self._draw_roads(surface, road_network, camera)

        # Draw vehicle
        self._draw_vehicle(surface, vehicle, camera)

        # Draw HUD
        self._draw_hud(surface, vehicle)

    def _draw_roads(self, surface, road_network, camera):
        """Draw road network using continuous network-level edges"""
        # Draw road surface as single continuous polygon (fixes junction gaps)
        self._draw_network_road_surface(surface, road_network, camera)

        # Draw lane markings using network-level lane centers
        if config.ENABLE_LANE_MARKINGS:
            self._draw_network_lane_markings(surface, road_network, camera)

        # Draw road edges using network-level edges
        if config.ENABLE_LANE_BOUNDARIES:
            self._draw_network_road_edges(surface, road_network, camera)

    def _draw_network_road_surface(self, surface, road_network, camera):
        """Draw continuous road surface using network-level edges"""
        if not hasattr(road_network, 'network_left_edge') or not hasattr(road_network, 'network_right_edge'):
            # Fallback to per-segment rendering if network edges not available
            for segment in road_network.segments:
                self._draw_road_segment(surface, segment, camera)
            return

        left_edge = road_network.network_left_edge
        right_edge = road_network.network_right_edge

        if len(left_edge) > 0 and len(right_edge) > 0:
            # Combine edges to form single continuous polygon
            points = []

            # Left edge
            for px, py in left_edge:
                sx, sy = camera.world_to_screen(px, py)
                points.append((sx, sy))

            # Right edge (reversed)
            for px, py in reversed(right_edge):
                sx, sy = camera.world_to_screen(px, py)
                points.append((sx, sy))

            if len(points) >= 3:
                try:
                    pygame.draw.polygon(surface, config.COLOR_ROAD, points)
                except:
                    pass  # Skip if points are off-screen

    def _draw_road_segment(self, surface, segment, camera):
        """Draw road surface (fallback method for compatibility)"""
        # Draw filled polygon for road
        if len(segment.left_edge) > 0 and len(segment.right_edge) > 0:
            # Combine edges to form polygon
            points = []

            # Left edge
            for px, py in segment.left_edge:
                sx, sy = camera.world_to_screen(px, py)
                points.append((sx, sy))

            # Right edge (reversed)
            for px, py in reversed(segment.right_edge):
                sx, sy = camera.world_to_screen(px, py)
                points.append((sx, sy))

            if len(points) >= 3:
                try:
                    pygame.draw.polygon(surface, config.COLOR_ROAD, points)
                except:
                    pass  # Skip if points are off-screen

    def _draw_network_lane_markings(self, surface, road_network, camera):
        """Draw continuous lane markings using network-level lane centers"""
        if not hasattr(road_network, 'network_lane_centers'):
            # Fallback to per-segment rendering
            for segment in road_network.segments:
                self._draw_lane_markings(surface, segment, camera)
            return

        if len(road_network.network_lane_centers) < 2:
            return

        # Draw lines between lanes (network-level continuous)
        for i in range(1, len(road_network.network_lane_centers)):
            lane_line = road_network.network_lane_centers[i]

            # Draw dashed line
            for j in range(0, len(lane_line) - 1, 3):  # Every 3rd point for dashes
                if j + 1 < len(lane_line):
                    px1, py1 = lane_line[j]
                    px2, py2 = lane_line[j + 1]

                    sx1, sy1 = camera.world_to_screen(px1, py1)
                    sx2, sy2 = camera.world_to_screen(px2, py2)

                    try:
                        pygame.draw.line(surface, config.COLOR_LANE_LINE,
                                       (sx1, sy1), (sx2, sy2), 2)
                    except:
                        pass

    def _draw_network_road_edges(self, surface, road_network, camera):
        """Draw continuous road edge lines using network-level edges"""
        if not hasattr(road_network, 'network_left_edge') or not hasattr(road_network, 'network_right_edge'):
            # Fallback to per-segment rendering
            for segment in road_network.segments:
                self._draw_road_edges(surface, segment, camera)
            return

        left_edge = road_network.network_left_edge
        right_edge = road_network.network_right_edge

        # Left edge (continuous)
        for i in range(len(left_edge) - 1):
            px1, py1 = left_edge[i]
            px2, py2 = left_edge[i + 1]

            sx1, sy1 = camera.world_to_screen(px1, py1)
            sx2, sy2 = camera.world_to_screen(px2, py2)

            try:
                pygame.draw.line(surface, config.COLOR_ROAD_EDGE,
                               (sx1, sy1), (sx2, sy2), 3)
            except:
                pass

        # Right edge (continuous)
        for i in range(len(right_edge) - 1):
            px1, py1 = right_edge[i]
            px2, py2 = right_edge[i + 1]

            sx1, sy1 = camera.world_to_screen(px1, py1)
            sx2, sy2 = camera.world_to_screen(px2, py2)

            try:
                pygame.draw.line(surface, config.COLOR_ROAD_EDGE,
                               (sx1, sy1), (sx2, sy2), 3)
            except:
                pass

    def _draw_lane_markings(self, surface, segment, camera):
        """Draw dashed lane lines (fallback method for compatibility)"""
        if segment.num_lanes < 2:
            return

        # Draw lines between lanes
        for i in range(1, segment.num_lanes):
            lane_line = segment.lane_centers[i]

            # Draw dashed line
            for j in range(0, len(lane_line) - 1, 3):  # Every 3rd point for dashes
                if j + 1 < len(lane_line):
                    px1, py1 = lane_line[j]
                    px2, py2 = lane_line[j + 1]

                    sx1, sy1 = camera.world_to_screen(px1, py1)
                    sx2, sy2 = camera.world_to_screen(px2, py2)

                    try:
                        pygame.draw.line(surface, config.COLOR_LANE_LINE,
                                       (sx1, sy1), (sx2, sy2), 2)
                    except:
                        pass

    def _draw_road_edges(self, surface, segment, camera):
        """Draw road edge lines (fallback method for compatibility)"""
        # Left edge
        for i in range(len(segment.left_edge) - 1):
            px1, py1 = segment.left_edge[i]
            px2, py2 = segment.left_edge[i + 1]

            sx1, sy1 = camera.world_to_screen(px1, py1)
            sx2, sy2 = camera.world_to_screen(px2, py2)

            try:
                pygame.draw.line(surface, config.COLOR_ROAD_EDGE,
                               (sx1, sy1), (sx2, sy2), 3)
            except:
                pass

        # Right edge
        for i in range(len(segment.right_edge) - 1):
            px1, py1 = segment.right_edge[i]
            px2, py2 = segment.right_edge[i + 1]

            sx1, sy1 = camera.world_to_screen(px1, py1)
            sx2, sy2 = camera.world_to_screen(px2, py2)

            try:
                pygame.draw.line(surface, config.COLOR_ROAD_EDGE,
                               (sx1, sy1), (sx2, sy2), 3)
            except:
                pass

    def _draw_vehicle(self, surface, vehicle, camera):
        """Draw vehicle as rotated rectangle"""
        # Get vehicle corners in world coordinates
        corners = vehicle.get_corners()

        # Convert to screen coordinates
        screen_corners = []
        for wx, wy in corners:
            sx, sy = camera.world_to_screen(wx, wy)
            screen_corners.append((sx, sy))

        # Draw vehicle
        try:
            pygame.draw.polygon(surface, config.COLOR_VEHICLE_PLAYER, screen_corners)
            pygame.draw.polygon(surface, (0, 0, 0), screen_corners, 2)  # Outline

            # Draw heading indicator (small line)
            front_x, front_y = vehicle.get_front_center()
            fsx, fsy = camera.world_to_screen(front_x, front_y)
            vsx, vsy = camera.world_to_screen(vehicle.x, vehicle.y)
            pygame.draw.line(surface, (255, 0, 0), (vsx, vsy), (fsx, fsy), 3)
        except:
            pass

    def _draw_hud(self, surface, vehicle):
        """Draw heads-up display with vehicle stats"""
        if self.font is None:
            return

        # HUD background
        hud_rect = pygame.Rect(10, 10, 250, 120)
        hud_surface = pygame.Surface((hud_rect.width, hud_rect.height), pygame.SRCALPHA)
        hud_surface.fill(config.COLOR_HUD_BG)
        surface.blit(hud_surface, hud_rect)

        # Text
        y_offset = 15
        texts = [
            f"Speed: {abs(vehicle.velocity):.1f} px/s",
            f"Heading: {vehicle.heading:.1f}°",
            f"Steering: {vehicle.steering_angle:.1f}°",
            f"Distance: {vehicle.distance_traveled:.0f} px",
            f"Lane: {vehicle.assigned_lane if vehicle.assigned_lane is not None else 'N/A'}",
        ]

        for text in texts:
            text_surface = self.small_font.render(text, True, config.COLOR_HUD_TEXT)
            surface.blit(text_surface, (15, y_offset))
            y_offset += 20
