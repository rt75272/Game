"""Player sprite – the spaceship controlled by the user."""

from __future__ import annotations

import pygame

from game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    CYAN,
    BLUE,
    WHITE,
    YELLOW,
    PLAYER_SPEED,
    PLAYER_SHOOT_DELAY,
    PLAYER_LIVES,
)


def _build_player_image() -> pygame.Surface:
    """Return a programmatically-drawn player spaceship surface."""
    surface = pygame.Surface((44, 36), pygame.SRCALPHA)

    # Main hull – elongated teardrop pointing upward
    hull_pts = [
        (22, 0),   # nose
        (44, 36),  # right rear
        (34, 30),
        (22, 34),
        (10, 30),
        (0, 36),   # left rear
    ]
    pygame.draw.polygon(surface, CYAN, hull_pts)

    # Cockpit glass
    pygame.draw.ellipse(surface, WHITE, (16, 6, 12, 14))
    pygame.draw.ellipse(surface, BLUE, (18, 8, 8, 10))

    # Engine nozzles (two side rectangles)
    pygame.draw.rect(surface, YELLOW, (6, 28, 8, 8), border_radius=2)
    pygame.draw.rect(surface, YELLOW, (30, 28, 8, 8), border_radius=2)

    # Wing accent lines
    pygame.draw.line(surface, WHITE, (22, 4), (8, 32), 1)
    pygame.draw.line(surface, WHITE, (22, 4), (36, 32), 1)

    return surface


class Player(pygame.sprite.Sprite):
    """The player-controlled spaceship.

    Attributes:
        score: Accumulated score for this session.
        lives: Remaining lives.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image = _build_player_image()
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 12

        self.speed: int = PLAYER_SPEED
        self.shoot_delay: int = PLAYER_SHOOT_DELAY
        self._last_shot: int = pygame.time.get_ticks()

        self.score: int = 0
        self.lives: int = PLAYER_LIVES

        self._hidden: bool = False
        self._hide_start: int = 0
        self._hide_duration: int = 1_500   # ms before respawning

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def hide(self) -> None:
        """Temporarily hide the player after being hit."""
        self._hidden = True
        self._hide_start = pygame.time.get_ticks()
        # Move off-screen so no further collisions are detected
        self.rect.center = (-200, -200)

    def shoot(self) -> "Bullet | None":  # noqa: F821  (forward ref)
        """Return a new Bullet if the shoot cooldown has elapsed, else None."""
        from game.sprites.bullet import Bullet

        now = pygame.time.get_ticks()
        if now - self._last_shot >= self.shoot_delay:
            self._last_shot = now
            return Bullet(self.rect.centerx, self.rect.top, is_player=True)
        return None

    @property
    def is_hidden(self) -> bool:
        return self._hidden

    def can_shoot(self) -> bool:
        return pygame.time.get_ticks() - self._last_shot >= self.shoot_delay

    # ------------------------------------------------------------------
    # Sprite interface
    # ------------------------------------------------------------------

    def update(self) -> None:  # type: ignore[override]
        # Respawn check
        if self._hidden:
            if pygame.time.get_ticks() - self._hide_start >= self._hide_duration:
                self._hidden = False
                self.rect.centerx = SCREEN_WIDTH // 2
                self.rect.bottom = SCREEN_HEIGHT - 12
            return

        keys = pygame.key.get_pressed()
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and self.rect.left > 0:
            self.rect.x -= self.speed
        if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed
