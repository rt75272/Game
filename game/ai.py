"""Simple learning AI controller for player movement and shooting."""

from __future__ import annotations

import json
import os
import random

from game.constants import (
    AI_EPSILON_MIN,
    AI_EPSILON_START,
    AI_LEARNING_RATE,
    AI_MEMORY_FILE,
)


class LearningAI:
    """Tiny contextual-bandit AI that learns from reward feedback over time."""

    def __init__(
        self,
        *,
        memory_path: str | None = None,
        epsilon: float = AI_EPSILON_START,
        epsilon_min: float = AI_EPSILON_MIN,
        learning_rate: float = AI_LEARNING_RATE,
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        default_path = os.path.join(base_dir, AI_MEMORY_FILE)
        self._memory_path = memory_path or default_path
        self._epsilon_start = epsilon
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._learning_rate = learning_rate
        self._episodes: int = 0

        self._move_values: dict[str, dict[str, float]] = {}
        self._shoot_values: dict[str, dict[str, float]] = {}
        self._last_move_state: str | None = None
        self._last_move_action: int | None = None
        self._last_shoot_state: str | None = None
        self._last_shoot_action: int | None = None
        self._load()

    def choose_actions(
        self, player, enemies, enemy_bullets, *, can_shoot: bool
    ) -> tuple[int, bool]:
        """Return movement direction (-1, 0, 1) and whether to shoot."""
        move_state, shoot_state = self._state_keys(player, enemies, enemy_bullets)
        move_action = self._choose_action(self._move_values, move_state, [-1, 0, 1])
        shoot_action = self._choose_action(self._shoot_values, shoot_state, [0, 1])
        if not can_shoot:
            shoot_action = 0

        self._last_move_state = move_state
        self._last_move_action = move_action
        self._last_shoot_state = shoot_state
        self._last_shoot_action = shoot_action
        return move_action, shoot_action == 1

    def apply_reward(self, reward: float, *, done: bool = False) -> None:
        """Apply reward to the most recent movement and shoot decisions."""
        self._update_action_value(
            self._move_values, self._last_move_state, self._last_move_action, reward
        )
        self._update_action_value(
            self._shoot_values, self._last_shoot_state, self._last_shoot_action, reward
        )
        if done:
            self._episodes += 1
            self._epsilon = max(self._epsilon_min, self._epsilon * 0.995)
            self.save()

    def save(self) -> None:
        """Persist learned values to disk."""
        payload = {
            "epsilon": self._epsilon,
            "episodes": self._episodes,
            "move_values": self._move_values,
            "shoot_values": self._shoot_values,
        }
        with open(self._memory_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _load(self) -> None:
        if not os.path.exists(self._memory_path):
            return
        try:
            with open(self._memory_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._epsilon = max(self._epsilon_min, float(payload.get("epsilon", self._epsilon)))
        self._episodes = int(payload.get("episodes", 0))
        self._move_values = payload.get("move_values", {})
        self._shoot_values = payload.get("shoot_values", {})

    def _choose_action(
        self,
        table: dict[str, dict[str, float]],
        state: str,
        actions: list[int],
    ) -> int:
        state_values = self._ensure_state_values(table, state, actions)
        if random.random() < self._epsilon:
            return random.choice(actions)
        best_action = actions[0]
        best_value = state_values[str(best_action)]
        for action in actions[1:]:
            value = state_values[str(action)]
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def _update_action_value(
        self,
        table: dict[str, dict[str, float]],
        state: str | None,
        action: int | None,
        reward: float,
    ) -> None:
        if state is None or action is None:
            return
        actions = self._ensure_state_values(table, state, [action])
        key = str(action)
        old_value = float(actions.get(key, 0.0))
        actions[key] = old_value + self._learning_rate * (reward - old_value)

    @staticmethod
    def _ensure_state_values(
        table: dict[str, dict[str, float]], state: str, actions: list[int]
    ) -> dict[str, float]:
        state_values = table.setdefault(state, {})
        for action in actions:
            state_values.setdefault(str(action), 0.0)
        return state_values

    def _state_keys(self, player, enemies, enemy_bullets) -> tuple[str, str]:
        player_x = player.rect.centerx

        nearest_enemy_dx = 0
        nearest_enemy_dy = 9999
        for enemy in enemies:
            dy = enemy.rect.centery - player.rect.centery
            if dy < nearest_enemy_dy:
                nearest_enemy_dy = dy
                nearest_enemy_dx = enemy.rect.centerx - player_x

        nearest_bullet_dx = 0
        nearest_bullet_dy = 9999
        for bullet in enemy_bullets:
            dy = player.rect.centery - bullet.rect.centery
            if 0 <= dy < nearest_bullet_dy:
                nearest_bullet_dy = dy
                nearest_bullet_dx = bullet.rect.centerx - player_x

        enemy_bucket = self._bucket_dx(nearest_enemy_dx)
        bullet_bucket = self._bucket_dx(nearest_bullet_dx)
        danger_bucket = "danger" if nearest_bullet_dy < 120 else "safe"

        move_state = f"move:{enemy_bucket}:{bullet_bucket}:{danger_bucket}"
        shoot_state = f"shoot:{enemy_bucket}:{danger_bucket}"
        return move_state, shoot_state

    @staticmethod
    def _bucket_dx(dx: int) -> str:
        if dx < -80:
            return "far_left"
        if dx < -25:
            return "left"
        if dx > 80:
            return "far_right"
        if dx > 25:
            return "right"
        return "center"
