"""Main Game class – owns the loop, state machine, and rendering."""

from __future__ import annotations

import random
import sys

import pygame

from game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    TITLE,
    # colours
    BLACK,
    WHITE,
    GREEN,
    YELLOW,
    CYAN,
    GRAY,
    RED,
    ORANGE,
    # gameplay
    ENEMY_ROWS,
    ENEMY_COLS,
    ENEMY_H_SPACING,
    ENEMY_V_SPACING,
    ENEMY_OFFSET_X,
    ENEMY_OFFSET_Y,
    ENEMY_MOVE_DELAY_START,
    ENEMY_MOVE_STEP,
    ENEMY_DROP_AMOUNT,
    ENEMY_SHOOT_CHANCE,
    NUM_STARS,
    AI_ENABLED_BY_DEFAULT,
    AI_REWARD_SHOT,
    AI_REWARD_ENEMY_DESTROYED,
    AI_REWARD_LEVEL_CLEAR,
    AI_PENALTY_PLAYER_HIT,
    AI_PENALTY_GAME_OVER,
)
from game.ai import LearningAI
from game.sprites import Player, Enemy, Bullet, Explosion, Star


class Game:
    """Top-level game object.

    Manages the pygame window, the game-state machine, all sprite groups,
    and the main update / draw loop.

    States
    ------
    ``"menu"``      – Title screen displayed before first play.
    ``"playing"``   – Active gameplay.
    ``"game_over"`` – Game-over screen after all lives are lost.
    ``"win"``       – Victory screen after all enemies are defeated.
    """

    def __init__(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self._clock = pygame.time.Clock()

        # Fonts
        self._font_large = pygame.font.SysFont("monospace", 52, bold=True)
        self._font_med = pygame.font.SysFont("monospace", 28, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 20)

        self._state: str = "menu"
        self._level: int = 1
        self._ai_enabled: bool = AI_ENABLED_BY_DEFAULT
        self._ai = LearningAI()

        # Sprite groups (populated by _new_game / _new_level)
        self._all_sprites: pygame.sprite.Group = pygame.sprite.Group()
        self._stars: pygame.sprite.Group = pygame.sprite.Group()
        self._enemies: pygame.sprite.Group = pygame.sprite.Group()
        self._player_bullets: pygame.sprite.Group = pygame.sprite.Group()
        self._enemy_bullets: pygame.sprite.Group = pygame.sprite.Group()
        self._explosions: pygame.sprite.Group = pygame.sprite.Group()

        self._player: Player | None = None

        # Enemy movement tracking
        self._enemy_direction: int = 1          # 1 = right, -1 = left
        self._last_enemy_move: int = 0
        self._enemy_move_delay: int = ENEMY_MOVE_DELAY_START

        # Background stars are permanent across states
        self._spawn_stars()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start and maintain the game loop until the window is closed."""
        while True:
            self._handle_events()
            self._update()
            self._draw()
            self._clock.tick(FPS)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _spawn_stars(self) -> None:
        for _ in range(NUM_STARS):
            star = Star()
            self._stars.add(star)
            self._all_sprites.add(star)

    def _new_game(self) -> None:
        """Reset everything and start from level 1."""
        self._level = 1
        # Keep stars; clear everything else
        for group in (
            self._enemies,
            self._player_bullets,
            self._enemy_bullets,
            self._explosions,
        ):
            group.empty()

        # Remove non-star sprites from all_sprites
        for sprite in list(self._all_sprites):
            if not isinstance(sprite, Star):
                sprite.kill()

        self._player = Player()
        self._all_sprites.add(self._player)
        self._new_level()

    def _new_level(self) -> None:
        """Populate enemies for the current level and reset bullet groups."""
        for group in (self._enemies, self._player_bullets, self._enemy_bullets):
            group.empty()
        for sprite in list(self._all_sprites):
            if isinstance(sprite, (Enemy, Bullet)):
                sprite.kill()

        # Speed scales with level
        self._enemy_move_delay = max(
            200, ENEMY_MOVE_DELAY_START - (self._level - 1) * 100
        )
        self._enemy_direction = 1
        self._last_enemy_move = pygame.time.get_ticks()

        for row in range(ENEMY_ROWS):
            for col in range(ENEMY_COLS):
                x = ENEMY_OFFSET_X + col * ENEMY_H_SPACING
                y = ENEMY_OFFSET_Y + row * ENEMY_V_SPACING
                enemy = Enemy(col=col, row=row, x=x, y=y)
                self._enemies.add(enemy)
                self._all_sprites.add(enemy)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if self._state == "menu":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._new_game()
                        self._state = "playing"
                    elif event.key == pygame.K_t:
                        self._ai_enabled = not self._ai_enabled

                elif self._state == "playing":
                    if event.key == pygame.K_t:
                        self._ai_enabled = not self._ai_enabled
                    if not self._ai_enabled:
                        if event.key == pygame.K_SPACE and self._player:
                            bullet = self._player.shoot()
                            if bullet:
                                self._player_bullets.add(bullet)
                                self._all_sprites.add(bullet)

                elif self._state in ("game_over", "win"):
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._state = "menu"

    # ------------------------------------------------------------------
    # Update logic
    # ------------------------------------------------------------------

    def _update(self) -> None:
        self._stars.update()

        if self._state != "playing":
            self._explosions.update()
            return

        # --- sprites ---
        if self._ai_enabled:
            self._update_ai_player()
        else:
            self._player.update()
        self._enemies.update()
        self._player_bullets.update()
        self._enemy_bullets.update()
        self._explosions.update()

        # --- enemy lateral movement ---
        self._move_enemies()

        # --- enemy shooting ---
        self._enemy_shoot()

        # --- collision: player bullets vs enemies ---
        hits = pygame.sprite.groupcollide(
            self._enemies, self._player_bullets, True, True
        )
        for enemy, bullets in hits.items():
            self._player.score += enemy.points
            exp = Explosion(enemy.rect.center)
            self._explosions.add(exp)
            self._all_sprites.add(exp)
            if self._ai_enabled:
                self._ai.apply_reward(AI_REWARD_ENEMY_DESTROYED)

        # --- collision: enemy bullets vs player ---
        if not self._player.is_hidden:
            hits = pygame.sprite.spritecollide(
                self._player, self._enemy_bullets, True
            )
            if hits:
                exp = Explosion(self._player.rect.center)
                self._explosions.add(exp)
                self._all_sprites.add(exp)
                self._player.lives -= 1
                if self._ai_enabled:
                    self._ai.apply_reward(AI_PENALTY_PLAYER_HIT)
                if self._player.lives <= 0:
                    self._state = "game_over"
                    if self._ai_enabled:
                        self._ai.apply_reward(AI_PENALTY_GAME_OVER, done=True)
                else:
                    self._player.hide()

        # --- enemies reaching bottom ---
        for enemy in self._enemies:
            if enemy.rect.bottom >= SCREEN_HEIGHT - 40:
                self._state = "game_over"
                if self._ai_enabled:
                    self._ai.apply_reward(AI_PENALTY_GAME_OVER, done=True)
                break

        # --- win condition ---
        if not self._enemies and self._state == "playing":
            self._level += 1
            if self._ai_enabled:
                self._ai.apply_reward(AI_REWARD_LEVEL_CLEAR, done=True)
            self._new_level()

    def _update_ai_player(self) -> None:
        if not self._player:
            return
        self._player.update()
        if self._player.is_hidden:
            return

        can_shoot = self._player.can_shoot()
        move_dir, should_shoot = self._ai.choose_actions(
            self._player, self._enemies, self._enemy_bullets, can_shoot=can_shoot
        )
        if move_dir < 0 and self._player.rect.left > 0:
            self._player.rect.x -= self._player.speed
        elif move_dir > 0 and self._player.rect.right < SCREEN_WIDTH:
            self._player.rect.x += self._player.speed

        if should_shoot:
            bullet = self._player.shoot()
            if bullet:
                self._player_bullets.add(bullet)
                self._all_sprites.add(bullet)
                self._ai.apply_reward(AI_REWARD_SHOT)

    def _move_enemies(self) -> None:
        """Shift all enemies sideways; drop and reverse direction at edges."""
        now = pygame.time.get_ticks()
        if now - self._last_enemy_move < self._enemy_move_delay:
            return
        self._last_enemy_move = now

        # Check if any enemy would exceed the screen boundary
        reverse = False
        for enemy in self._enemies:
            future_x = enemy.rect.x + self._enemy_direction * ENEMY_MOVE_STEP
            if future_x < 0 or future_x + enemy.rect.width > SCREEN_WIDTH:
                reverse = True
                break

        if reverse:
            self._enemy_direction *= -1
            for enemy in self._enemies:
                enemy.rect.y += ENEMY_DROP_AMOUNT
        else:
            for enemy in self._enemies:
                enemy.rect.x += self._enemy_direction * ENEMY_MOVE_STEP

    def _enemy_shoot(self) -> None:
        """Randomly select an enemy at the bottom of each column to shoot."""
        # Build a mapping column → lowest enemy in that column
        bottom_enemies: dict[int, Enemy] = {}
        for enemy in self._enemies:
            col = enemy.col
            if col not in bottom_enemies or enemy.rect.bottom > bottom_enemies[col].rect.bottom:
                bottom_enemies[col] = enemy

        for enemy in bottom_enemies.values():
            if random.random() < ENEMY_SHOOT_CHANCE * (1 + self._level * 0.5):
                bullet = Bullet(enemy.rect.centerx, enemy.rect.bottom, is_player=False)
                self._enemy_bullets.add(bullet)
                self._all_sprites.add(bullet)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self._screen.fill(BLACK)
        self._stars.draw(self._screen)

        if self._state == "menu":
            self._draw_menu()
        elif self._state == "playing":
            self._draw_playing()
        elif self._state == "game_over":
            self._draw_game_over()
        elif self._state == "win":
            self._draw_win()

        self._explosions.draw(self._screen)
        pygame.display.flip()

    # --- state-specific drawing ---

    def _draw_menu(self) -> None:
        self._draw_title(TITLE, CYAN)
        self._draw_centred("PRESS SPACE TO START", SCREEN_HEIGHT // 2 + 20, YELLOW)
        self._draw_centred("ARROW KEYS / WASD  –  move", SCREEN_HEIGHT // 2 + 70, GRAY)
        self._draw_centred("SPACE  –  shoot", SCREEN_HEIGHT // 2 + 100, GRAY)
        ai_text = f"T  –  AI {'ON' if self._ai_enabled else 'OFF'}"
        self._draw_centred(ai_text, SCREEN_HEIGHT // 2 + 130, GRAY)

        # Decorative alien preview
        preview_frames = Enemy(0, 0, 0, 0)._frames
        frame = preview_frames[0]
        x = SCREEN_WIDTH // 2 - frame.get_width() // 2
        self._screen.blit(frame, (x, SCREEN_HEIGHT // 2 - 80))

    def _draw_playing(self) -> None:
        # Enemies and player bullets
        self._enemies.draw(self._screen)
        self._player_bullets.draw(self._screen)
        self._enemy_bullets.draw(self._screen)
        if self._player and not self._player.is_hidden:
            self._screen.blit(self._player.image, self._player.rect)

        self._draw_hud()

    def _draw_game_over(self) -> None:
        self._draw_title("GAME OVER", RED)
        if self._player:
            self._draw_centred(
                f"SCORE: {self._player.score}", SCREEN_HEIGHT // 2 + 10, WHITE
            )
        self._draw_centred("PRESS SPACE TO RETURN TO MENU", SCREEN_HEIGHT // 2 + 60, YELLOW)

    def _draw_win(self) -> None:
        self._draw_title("YOU WIN!", GREEN)
        if self._player:
            self._draw_centred(
                f"SCORE: {self._player.score}", SCREEN_HEIGHT // 2 + 10, WHITE
            )
        self._draw_centred("PRESS SPACE TO RETURN TO MENU", SCREEN_HEIGHT // 2 + 60, YELLOW)

    # --- HUD ---

    def _draw_hud(self) -> None:
        if not self._player:
            return

        score_surf = self._font_small.render(
            f"SCORE: {self._player.score}", True, WHITE
        )
        self._screen.blit(score_surf, (10, 8))

        lives_surf = self._font_small.render(
            f"LIVES: {self._player.lives}", True, GREEN
        )
        self._screen.blit(lives_surf, (SCREEN_WIDTH - lives_surf.get_width() - 10, 8))

        level_surf = self._font_small.render(f"LEVEL: {self._level}", True, ORANGE)
        self._screen.blit(
            level_surf,
            (SCREEN_WIDTH // 2 - level_surf.get_width() // 2, 8),
        )
        ai_surf = self._font_small.render(
            f"AI: {'ON' if self._ai_enabled else 'OFF'} (T to toggle)", True, CYAN
        )
        self._screen.blit(
            ai_surf,
            (SCREEN_WIDTH // 2 - ai_surf.get_width() // 2, SCREEN_HEIGHT - 28),
        )

        # Separator line
        pygame.draw.line(
            self._screen, GRAY,
            (0, SCREEN_HEIGHT - 36),
            (SCREEN_WIDTH, SCREEN_HEIGHT - 36),
            1,
        )

    # --- text helpers ---

    def _draw_title(self, text: str, colour: tuple) -> None:
        surf = self._font_large.render(text, True, colour)
        rect = surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        # Simple shadow
        shadow = self._font_large.render(text, True, BLACK)
        self._screen.blit(shadow, rect.move(2, 2))
        self._screen.blit(surf, rect)

    def _draw_centred(self, text: str, y: int, colour: tuple) -> None:
        surf = self._font_med.render(text, True, colour)
        rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
        self._screen.blit(surf, rect)
