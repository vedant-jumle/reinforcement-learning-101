"""Physics utilities and helpers"""

import numpy as np


def rotate_point(x, y, angle_deg):
    """Rotate a point around origin by angle in degrees

    Args:
        x, y: Point coordinates
        angle_deg: Rotation angle in degrees

    Returns:
        (rotated_x, rotated_y)
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a

    return new_x, new_y


def normalize_angle(angle):
    """Normalize angle to [-180, 180] range

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))
