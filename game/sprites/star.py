"""Scrolling background star sprite."""

from __future__ import annotations

import random

import pygame

from game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, GRAY, LIGHT_CYAN


class Star(pygame.sprite.Sprite):
    """A tiny star that scrolls downward to give a sense of forward motion.

    Stars are randomly placed and travel at varying speeds to create a
    simple parallax depth effect.
    """

    _COLOURS = [WHITE, WHITE, GRAY, LIGHT_CYAN]

    def __init__(self) -> None:
        super().__init__()
        self._reset(initial=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset(self, *, initial: bool = False) -> None:
        """Reposition the star at a random location (top of screen on wrap)."""
        self.speed = random.randint(1, 4)
        size = 1 if self.speed <= 2 else 2
        colour = random.choice(self._COLOURS)

        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        self.image.fill(colour)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, SCREEN_WIDTH - 1)
        if initial:
            self.rect.y = random.randint(0, SCREEN_HEIGHT - 1)
        else:
            self.rect.y = 0

    # ------------------------------------------------------------------
    # Sprite interface
    # ------------------------------------------------------------------

    def update(self) -> None:  # type: ignore[override]
        self.rect.y += self.speed
        if self.rect.top > SCREEN_HEIGHT:
            self._reset()
