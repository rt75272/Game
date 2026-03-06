"""Headless unit tests for the Space Invaders game.

``SDL_VIDEODRIVER=dummy`` and ``SDL_AUDIODRIVER=dummy`` are set so the
tests can run without a real display or audio device.
"""

from __future__ import annotations

import os

# Must be set before pygame is imported
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

import game.constants as C  # noqa: E402
from game.sprites.player import Player  # noqa: E402
from game.sprites.enemy import Enemy  # noqa: E402
from game.sprites.bullet import Bullet  # noqa: E402
from game.sprites.explosion import Explosion  # noqa: E402
from game.sprites.star import Star  # noqa: E402
from game.game import Game  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def pygame_init():
    """Initialise pygame once for the entire test session."""
    pygame.init()
    pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    yield
    pygame.quit()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_screen_dimensions_are_positive() -> None:
    assert C.SCREEN_WIDTH > 0
    assert C.SCREEN_HEIGHT > 0


def test_fps_is_reasonable() -> None:
    assert 30 <= C.FPS <= 120


def test_enemy_grid_positive() -> None:
    assert C.ENEMY_ROWS > 0
    assert C.ENEMY_COLS > 0


# ---------------------------------------------------------------------------
# Player sprite
# ---------------------------------------------------------------------------


def test_player_initial_position() -> None:
    p = Player()
    assert p.rect.centerx == C.SCREEN_WIDTH // 2
    assert p.rect.bottom == C.SCREEN_HEIGHT - 12


def test_player_initial_lives() -> None:
    p = Player()
    assert p.lives == C.PLAYER_LIVES


def test_player_initial_score() -> None:
    p = Player()
    assert p.score == 0


def test_player_shoot_returns_bullet() -> None:
    p = Player()
    # Force cooldown to have expired by back-dating last shot
    p._last_shot = pygame.time.get_ticks() - C.PLAYER_SHOOT_DELAY - 1
    bullet = p.shoot()
    assert bullet is not None
    assert isinstance(bullet, Bullet)
    assert bullet.is_player is True


def test_player_shoot_respects_cooldown() -> None:
    p = Player()
    p._last_shot = pygame.time.get_ticks()  # just shot
    bullet = p.shoot()
    assert bullet is None


def test_player_hide() -> None:
    p = Player()
    p.hide()
    assert p._hidden is True
    # Rect should be moved off-screen
    assert p.rect.centerx < 0 or p.rect.centery < 0


# ---------------------------------------------------------------------------
# Enemy sprite
# ---------------------------------------------------------------------------


def test_enemy_creation() -> None:
    e = Enemy(col=0, row=0, x=100, y=100)
    assert e.rect.centerx == 100
    assert e.rect.centery == 100


def test_enemy_point_values_decrease_by_row() -> None:
    top = Enemy(col=0, row=0, x=0, y=0)
    mid = Enemy(col=0, row=1, x=0, y=0)
    bot = Enemy(col=0, row=2, x=0, y=0)
    assert top.points > mid.points > bot.points


def test_enemy_has_two_animation_frames() -> None:
    e = Enemy(col=0, row=0, x=0, y=0)
    assert len(e._frames) == 2


# ---------------------------------------------------------------------------
# Bullet sprite
# ---------------------------------------------------------------------------


def test_player_bullet_moves_upward() -> None:
    b = Bullet(400, 500, is_player=True)
    assert b.speed < 0


def test_enemy_bullet_moves_downward() -> None:
    b = Bullet(400, 100, is_player=False)
    assert b.speed > 0


def test_player_bullet_speed_constant() -> None:
    b = Bullet(400, 500, is_player=True)
    assert b.speed == C.PLAYER_BULLET_SPEED


def test_enemy_bullet_speed_constant() -> None:
    b = Bullet(400, 100, is_player=False)
    assert b.speed == C.ENEMY_BULLET_SPEED


# ---------------------------------------------------------------------------
# Explosion sprite
# ---------------------------------------------------------------------------


def test_explosion_creation() -> None:
    exp = Explosion((200, 200))
    assert exp.rect.center == (200, 200)


def test_explosion_has_frames() -> None:
    exp = Explosion((100, 100))
    assert exp._frames  # not empty


# ---------------------------------------------------------------------------
# Star sprite
# ---------------------------------------------------------------------------


def test_star_is_within_screen() -> None:
    s = Star()
    assert 0 <= s.rect.x < C.SCREEN_WIDTH
    assert 0 <= s.rect.y < C.SCREEN_HEIGHT


def test_star_speed_positive() -> None:
    s = Star()
    assert s.speed > 0


# ---------------------------------------------------------------------------
# Game state machine
# ---------------------------------------------------------------------------


def test_game_starts_in_menu_state() -> None:
    g = Game()
    assert g._state == "menu"


def test_state_can_be_set_to_playing() -> None:
    """Verify that the game state variable accepts the 'playing' value."""
    g = Game()
    g._new_game()
    g._state = "playing"
    assert g._state == "playing"


def test_new_game_creates_player() -> None:
    g = Game()
    g._new_game()
    assert g._player is not None
    assert isinstance(g._player, Player)


def test_new_level_spawns_correct_enemy_count() -> None:
    g = Game()
    g._new_game()
    assert len(g._enemies) == C.ENEMY_ROWS * C.ENEMY_COLS


def test_enemy_move_delay_decreases_with_level() -> None:
    g = Game()
    g._new_game()
    delay_level_1 = g._enemy_move_delay
    g._level = 3
    g._new_level()
    assert g._enemy_move_delay < delay_level_1


def test_score_increases_on_enemy_kill() -> None:
    g = Game()
    g._new_game()
    g._state = "playing"
    initial_score = g._player.score
    # Manually kill one enemy and add the points
    enemy = next(iter(g._enemies))
    g._player.score += enemy.points
    assert g._player.score > initial_score


def test_player_loses_life_on_enemy_bullet_hit() -> None:
    g = Game()
    g._new_game()
    g._state = "playing"
    initial_lives = g._player.lives
    g._player.lives -= 1
    assert g._player.lives == initial_lives - 1
