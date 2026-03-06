"""Focused tests for learning AI behaviour."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest
import torch

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


@pytest.fixture(scope="module", autouse=True)
def pygame_init() -> None:
    pygame.init()
    yield
    pygame.quit()


def test_ai_persists_manual_preferences(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    player, enemies, enemy_bullets = _create_sample_game_state()

    ai = LearningAI(memory_path=memory_path, epsilon=0.0)
    ai.observe_player_action(
        player, enemies, enemy_bullets, can_shoot=True
        , move_dir=1, should_shoot=False
    )
    ai.save()

    restored = LearningAI(memory_path=memory_path, epsilon=0.0)
    new_move_action, new_should_shoot = restored.choose_actions(
        player, enemies, enemy_bullets, can_shoot=True
    )
    assert new_move_action == 1
    assert new_should_shoot is False


def test_ai_respects_shoot_cooldown_gate(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    player, enemies, enemy_bullets = _create_sample_game_state()
    ai = LearningAI(memory_path=memory_path, epsilon=0.0)
    _move_action, should_shoot = ai.choose_actions(
        player, enemies, enemy_bullets, can_shoot=False
    )
    assert should_shoot is False


def test_ai_repeats_rewarded_action_after_learning(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    player, enemies, enemy_bullets = _create_sample_game_state()

    ai = LearningAI(memory_path=memory_path, epsilon=0.0, imitation_weight=0.0)
    move_action, should_shoot = ai.choose_actions(
        player, enemies, enemy_bullets, can_shoot=True
    )
    ai.apply_reward(5.0)

    learned_move_action, learned_should_shoot = ai.choose_actions(
        player, enemies, enemy_bullets, can_shoot=True
    )

    assert learned_move_action == move_action
    assert learned_should_shoot == should_shoot


def test_ai_exposes_training_metrics(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    player, enemies, enemy_bullets = _create_sample_game_state()

    ai = LearningAI(memory_path=memory_path, epsilon=0.0)
    ai.observe_player_action(
        player,
        enemies,
        enemy_bullets,
        move_dir=0,
        should_shoot=True,
        can_shoot=True,
    )

    lines = ai.training_overlay_lines()

    assert any("backend: torch" in line for line in lines)
    assert any("device:" in line for line in lines)
    assert any("demos: 1" in line for line in lines)


def test_ai_ignores_incompatible_saved_model(tmp_path) -> None:
    memory_path = str(tmp_path / "ai_memory.json")
    model_path = str(tmp_path / "ai_memory.pt")

    torch.save(
        {
            "policy_state": {
                "0.weight": torch.zeros((64, 30), dtype=torch.float32),
                "0.bias": torch.zeros((64,), dtype=torch.float32),
            }
        },
        model_path,
    )

    ai = LearningAI(memory_path=memory_path, epsilon=0.0)

    assert ai.device_name() in {"cpu", "cuda"}
