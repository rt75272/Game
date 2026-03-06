"""Enemy alien sprite with two-frame animation."""

from __future__ import annotations

import pygame

from game.constants import (
    GREEN,
    DARK_GREEN,
    YELLOW,
    ORANGE,
    PURPLE,
    WHITE,
    ENEMY_ANIM_DELAY,
)


def _draw_alien_a(color: tuple, accent: tuple, size: int = 32) -> pygame.Surface:
    """Frame A of the alien – legs spread wide."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2

    # Body
    pygame.draw.ellipse(s, color, (cx - 10, cy - 8, 20, 16))

    # Eyes
    pygame.draw.circle(s, WHITE, (cx - 4, cy - 4), 4)
    pygame.draw.circle(s, WHITE, (cx + 4, cy - 4), 4)
    pygame.draw.circle(s, accent, (cx - 4, cy - 4), 2)
    pygame.draw.circle(s, accent, (cx + 4, cy - 4), 2)

    # Antennae (frame A – spread)
    pygame.draw.line(s, color, (cx - 6, cy - 8), (cx - 12, cy - 14), 2)
    pygame.draw.line(s, color, (cx + 6, cy - 8), (cx + 12, cy - 14), 2)

    # Legs (frame A – wide)
    for dx in (-10, -5, 5, 10):
        pygame.draw.line(s, color, (cx + dx, cy + 8), (cx + dx - 4, cy + 16), 2)
    return s


def _draw_alien_b(color: tuple, accent: tuple, size: int = 32) -> pygame.Surface:
    """Frame B of the alien – legs together."""
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2

    # Body
    pygame.draw.ellipse(s, color, (cx - 10, cy - 8, 20, 16))

    # Eyes
    pygame.draw.circle(s, WHITE, (cx - 4, cy - 4), 4)
    pygame.draw.circle(s, WHITE, (cx + 4, cy - 4), 4)
    pygame.draw.circle(s, accent, (cx - 4, cy - 4), 2)
    pygame.draw.circle(s, accent, (cx + 4, cy - 4), 2)

    # Antennae (frame B – upright)
    pygame.draw.line(s, color, (cx - 6, cy - 8), (cx - 8, cy - 15), 2)
    pygame.draw.line(s, color, (cx + 6, cy - 8), (cx + 8, cy - 15), 2)

    # Legs (frame B – narrow)
    for dx in (-10, -5, 5, 10):
        pygame.draw.line(s, color, (cx + dx, cy + 8), (cx + dx + 3, cy + 16), 2)
    return s


# One palette per row (color, eye-accent)
_ROW_PALETTES = [
    (PURPLE, WHITE),
    (GREEN, DARK_GREEN),
    (ORANGE, YELLOW),
]

# Pre-built animation frames keyed by row index
_ALIEN_FRAMES: dict[int, list[pygame.Surface]] = {}


def _get_frames(row: int) -> list[pygame.Surface]:
    if row not in _ALIEN_FRAMES:
        color, accent = _ROW_PALETTES[row % len(_ROW_PALETTES)]
        _ALIEN_FRAMES[row] = [
            _draw_alien_a(color, accent),
            _draw_alien_b(color, accent),
        ]
    return _ALIEN_FRAMES[row]


class Enemy(pygame.sprite.Sprite):
    """An animated alien enemy.

    Args:
        col: Column index (0-based) within the enemy grid.
        row: Row index (0-based) within the enemy grid.
        x: Initial pixel x-coordinate.
        y: Initial pixel y-coordinate.
    """

    point_values = [30, 20, 10]   # points awarded per row (top → bottom)

    def __init__(self, col: int, row: int, x: int, y: int) -> None:
        super().__init__()
        self.col = col
        self.row = row
        self.points: int = self.point_values[row % len(self.point_values)]

        self._frames = _get_frames(row)
        self._frame_idx = 0
        self.image = self._frames[self._frame_idx]
        self.rect = self.image.get_rect(center=(x, y))

        self._last_anim = pygame.time.get_ticks()

    def animate(self) -> None:
        """Advance to the next animation frame if the delay has elapsed."""
        now = pygame.time.get_ticks()
        if now - self._last_anim >= ENEMY_ANIM_DELAY:
            self._last_anim = now
            self._frame_idx = (self._frame_idx + 1) % len(self._frames)
            center = self.rect.center
            self.image = self._frames[self._frame_idx]
            self.rect = self.image.get_rect(center=center)

    def update(self) -> None:  # type: ignore[override]
        self.animate()
