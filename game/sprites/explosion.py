"""Animated explosion sprite."""

from __future__ import annotations

import math
import pygame

from game.constants import (
    YELLOW,
    ORANGE,
    BRIGHT_RED,
    WHITE,
    EXPLOSION_FRAMES,
    EXPLOSION_DURATION,
)

# Colour sequence cycling through the explosion life-time
_COLOURS = [WHITE, YELLOW, YELLOW, ORANGE, ORANGE, BRIGHT_RED, BRIGHT_RED, (80, 80, 80)]


def _build_explosion_frames(max_radius: int = 24) -> list[pygame.Surface]:
    """Pre-render all explosion frames."""
    frames: list[pygame.Surface] = []
    size = max_radius * 2 + 4
    for i in range(EXPLOSION_FRAMES):
        progress = (i + 1) / EXPLOSION_FRAMES
        radius = int(max_radius * progress)
        colour = _COLOURS[i % len(_COLOURS)]

        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx = cy = size // 2

        # Outer ring fades out
        alpha = int(255 * (1.0 - progress * 0.6))
        ring_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(ring_surf, (*colour, alpha), (cx, cy), radius, 3)
        surf.blit(ring_surf, (0, 0))

        # Inner glow
        inner_r = max(2, int(radius * 0.5))
        inner_alpha = int(255 * (1.0 - progress))
        inner_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(inner_surf, (*WHITE, inner_alpha), (cx, cy), inner_r)
        surf.blit(inner_surf, (0, 0))

        # Sparks radiating outward
        num_sparks = 6
        for j in range(num_sparks):
            angle = math.radians(j * 360 / num_sparks + i * 15)
            sx = int(cx + radius * math.cos(angle))
            sy = int(cy + radius * math.sin(angle))
            spark_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(spark_surf, (*YELLOW, max(0, alpha - 50)), (sx, sy), 2)
            surf.blit(spark_surf, (0, 0))

        frames.append(surf)
    return frames


# Lazily built so pygame is already initialised when first used
_FRAMES: list[pygame.Surface] | None = None


def _get_frames() -> list[pygame.Surface]:
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = _build_explosion_frames()
    return _FRAMES


class Explosion(pygame.sprite.Sprite):
    """A frame-based explosion animation centred on a given point.

    Args:
        center: ``(x, y)`` pixel position for the explosion centre.
    """

    def __init__(self, center: tuple[int, int]) -> None:
        super().__init__()
        self._frames = _get_frames()
        self._frame_idx: int = 0
        self._frame_duration: int = EXPLOSION_DURATION // EXPLOSION_FRAMES
        self._last_update: int = pygame.time.get_ticks()

        self.image = self._frames[0]
        self.rect = self.image.get_rect(center=center)

    def update(self) -> None:  # type: ignore[override]
        now = pygame.time.get_ticks()
        if now - self._last_update >= self._frame_duration:
            self._last_update = now
            self._frame_idx += 1
            if self._frame_idx >= len(self._frames):
                self.kill()
            else:
                center = self.rect.center
                self.image = self._frames[self._frame_idx]
                self.rect = self.image.get_rect(center=center)
