"""Landing platform class"""

import pygame
import numpy as np
from . import config


class Platform:
    """Landing platform for the drone"""

    def __init__(self, x, y, width=None, height=None):
        """Initialize platform

        Args:
            x, y: Center position
            width: Platform width (default from config)
            height: Platform height (default from config)
        """
        self.x = x
        self.y = y
        self.width = width or config.PLATFORM_WIDTH
        self.height = height or config.PLATFORM_HEIGHT

        # Movement (for moving platform variant)
        self.moving = config.PLATFORM_MOVING
        self.speed = config.PLATFORM_SPEED
        self.direction = 1  # 1 = right, -1 = left
        self.min_x = self.width // 2
        self.max_x = config.WINDOW_WIDTH - self.width // 2

    def update(self, dt=1.0):
        """Update platform (for moving platforms)

        Args:
            dt: Time delta
        """
        if not self.moving:
            return

        # Move platform
        self.x += self.speed * self.direction * dt

        # Bounce at boundaries
        if self.x <= self.min_x:
            self.x = self.min_x
            self.direction = 1
        elif self.x >= self.max_x:
            self.x = self.max_x
            self.direction = -1

    def get_bounds(self):
        """Get platform bounding box

        Returns:
            (left, right, top, bottom) tuple
        """
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        top = self.y - self.height / 2
        bottom = self.y + self.height / 2

        return left, right, top, bottom

    def is_point_on_platform(self, x, y):
        """Check if a point is on the platform surface

        Args:
            x, y: Point coordinates

        Returns:
            True if point is on platform
        """
        left, right, top, bottom = self.get_bounds()
        return (left <= x <= right) and (top <= y <= bottom)

    def render(self, screen):
        """Render platform on pygame screen

        Args:
            screen: Pygame surface
        """
        left, right, top, bottom = self.get_bounds()

        # Draw platform
        pygame.draw.rect(screen, config.COLOR_PLATFORM,
                        (left, top, self.width, self.height))

        # Draw platform outline
        pygame.draw.rect(screen, (0, 150, 0),
                        (left, top, self.width, self.height), 2)

        # Draw landing target indicator (center line)
        center_x = int(self.x)
        pygame.draw.line(screen, (255, 255, 255),
                        (center_x, int(top)),
                        (center_x, int(bottom)), 2)

        # Draw "H" landing symbol
        font = pygame.font.Font(None, 30)
        text = font.render("H", True, (255, 255, 255))
        text_rect = text.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(text, text_rect)

    def reset(self, x=None, y=None):
        """Reset platform position

        Args:
            x, y: New position (optional)
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        self.direction = 1
