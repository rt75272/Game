"""Focused tests for learning AI behaviour."""

from __future__ import annotations

import pygame

from game.ai import LearningAI


class _Obj:
    def __init__(self, centerx: int, centery: int) -> None:
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.rect.center = (centerx, centery)


def _create_sample_game_state():
    player = _Obj(300, 560)
    enemies = [_Obj(360, 120), _Obj(260, 170)]
    enemy_bullets = [_Obj(315, 520)]
    return player, enemies, enemy_bullets


def test_ai_persists_learned_preferences(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    player, enemies, enemy_bullets = _create_sample_game_state()

    ai = LearningAI(memory_path=memory_path, epsilon=0.0)
    move_action, should_shoot = ai.choose_actions(
        player, enemies, enemy_bullets, can_shoot=True
    )
    ai.apply_reward(4.0, done=True)
    ai.save()

    restored = LearningAI(memory_path=memory_path, epsilon=0.0)
    new_move_action, new_should_shoot = restored.choose_actions(
        player, enemies, enemy_bullets, can_shoot=True
    )
    assert new_move_action == move_action
    assert new_should_shoot == should_shoot


def test_ai_respects_shoot_cooldown_gate(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    player, enemies, enemy_bullets = _create_sample_game_state()
    ai = LearningAI(memory_path=memory_path, epsilon=0.0)
    _move_action, should_shoot = ai.choose_actions(
        player, enemies, enemy_bullets, can_shoot=False
    )
    assert should_shoot is False
