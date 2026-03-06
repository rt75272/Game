"""Bullet sprite – used for both player and enemy projectiles."""

from __future__ import annotations

import pygame

from game.constants import (
    SCREEN_HEIGHT,
    YELLOW,
    BRIGHT_RED,
    PLAYER_BULLET_SPEED,
    ENEMY_BULLET_SPEED,
)


def _build_player_bullet() -> pygame.Surface:
    s = pygame.Surface((4, 14), pygame.SRCALPHA)
    # Bright core
    pygame.draw.rect(s, YELLOW, (1, 0, 2, 14), border_radius=1)
    # Tip glow
    pygame.draw.circle(s, (255, 255, 200), (2, 1), 2)
    return s


def _build_enemy_bullet() -> pygame.Surface:
    s = pygame.Surface((5, 12), pygame.SRCALPHA)
    pygame.draw.rect(s, BRIGHT_RED, (1, 0, 3, 12), border_radius=1)
    pygame.draw.circle(s, (255, 180, 180), (2, 11), 2)
    return s


class Bullet(pygame.sprite.Sprite):
    """A single bullet travelling vertically.

    Args:
        x: Horizontal spawn position (centre of bullet).
        y: Vertical spawn position (top of bullet for player, bottom for enemy).
        is_player: ``True`` if fired by the player, ``False`` for enemies.
    """

    def __init__(self, x: int, y: int, *, is_player: bool) -> None:
        super().__init__()
        self.is_player = is_player
        if is_player:
            self.image = _build_player_bullet()
            self.speed = PLAYER_BULLET_SPEED
            self.rect = self.image.get_rect(midbottom=(x, y))
        else:
            self.image = _build_enemy_bullet()
            self.speed = ENEMY_BULLET_SPEED
            self.rect = self.image.get_rect(midtop=(x, y))

    def update(self) -> None:  # type: ignore[override]
        self.rect.y += self.speed
        # Remove bullet when it leaves the screen
        if self.rect.bottom < 0 or self.rect.top > SCREEN_HEIGHT:
            self.kill()
