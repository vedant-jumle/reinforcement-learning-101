"""Top-down camera system"""

import numpy as np
from . import config


class Camera:
    """Top-down camera that can follow the vehicle"""

    def __init__(self, width, height):
        """Initialize camera

        Args:
            width, height: Camera viewport size (pixels)
        """
        self.width = width
        self.height = height
        self.x = 0  # Camera center position
        self.y = 0

    def follow_vehicle(self, vehicle):
        """Update camera to follow vehicle"""
        if config.CAMERA_FOLLOW_VEHICLE:
            self.x = vehicle.x
            self.y = vehicle.y + config.CAMERA_OFFSET_Y
        else:
            # Fixed camera
            self.x = config.WINDOW_WIDTH / 2
            self.y = config.WINDOW_HEIGHT / 2

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates

        Args:
            world_x, world_y: World coordinates

        Returns:
            (screen_x, screen_y) tuple
        """
        screen_x = world_x - self.x + self.width / 2
        screen_y = world_y - self.y + self.height / 2
        return int(screen_x), int(screen_y)

    def crop_around_vehicle(self, frame, vehicle, crop_size=(200, 200)):
        """Crop frame around vehicle position

        Args:
            frame: Full frame (H x W x 3)
            vehicle: Vehicle object
            crop_size: (width, height) of crop

        Returns:
            Cropped frame centered on vehicle
        """
        h, w = frame.shape[:2]
        crop_w, crop_h = crop_size

        # Get vehicle screen position
        vx, vy = self.world_to_screen(vehicle.x, vehicle.y)

        # Crop bounds
        x1 = max(0, vx - crop_w // 2)
        y1 = max(0, vy - crop_h // 2)
        x2 = min(w, vx + crop_w // 2)
        y2 = min(h, vy + crop_h // 2)

        cropped = frame[y1:y2, x1:x2]

        # Pad if near edge
        if cropped.shape[0] < crop_h or cropped.shape[1] < crop_w:
            padded = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)
            py = (crop_h - cropped.shape[0]) // 2
            px = (crop_w - cropped.shape[1]) // 2
            padded[py:py+cropped.shape[0], px:px+cropped.shape[1]] = cropped
            return padded

        return cropped
