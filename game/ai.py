"""Learning AI controller that mimics the player and improves with reward."""

from __future__ import annotations

import json
import os
import random

from game.constants import (
    AI_DISCOUNT_FACTOR,
    AI_EPSILON_DECAY,
    AI_EPSILON_MIN,
    AI_EPSILON_START,
    AI_IMITATION_WEIGHT,
    AI_LEARNING_RATE,
    AI_MEMORY_FILE,
    SCREEN_WIDTH,
)


Action = tuple[int, bool]


class LearningAI:
    """Tabular Q-learning agent with imitation bias from manual play."""

    _ACTIONS: tuple[Action, ...] = (
        (-1, False),
        (0, False),
        (1, False),
        (-1, True),
        (0, True),
        (1, True),
    )

    def __init__(
        self,
        *,
        memory_path: str | None = None,
        epsilon: float = AI_EPSILON_START,
        epsilon_min: float = AI_EPSILON_MIN,
        epsilon_decay: float = AI_EPSILON_DECAY,
        learning_rate: float = AI_LEARNING_RATE,
        discount_factor: float = AI_DISCOUNT_FACTOR,
        imitation_weight: float = AI_IMITATION_WEIGHT,
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        default_path = os.path.join(base_dir, AI_MEMORY_FILE)
        self._memory_path = memory_path or default_path
        self._epsilon_start = epsilon
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._imitation_weight = imitation_weight
        self._episodes: int = 0
        self._manual_samples: int = 0

        self._q_values: dict[str, dict[str, float]] = {}
        self._policy_counts: dict[str, dict[str, int]] = {}
        self._last_state: str | None = None
        self._last_action_key: str | None = None
        self._pending_reward: float = 0.0
        self._dirty_updates: int = 0
        self._load()

    def choose_actions(
        self, player, enemies, enemy_bullets, *, can_shoot: bool
    ) -> tuple[int, bool]:
        """Return movement direction (-1, 0, 1) and whether to shoot."""
        state = self._state_key(player, enemies, enemy_bullets, can_shoot=can_shoot)
        self._update_from_transition(next_state=state)

        valid_actions = self._valid_action_keys(can_shoot)
        action_key = self._choose_action_key(state, valid_actions)
        self._last_state = state
        self._last_action_key = action_key
        return self._decode_action(action_key)

    def observe_player_action(
        self,
        player,
        enemies,
        enemy_bullets,
        move_dir: int,
        should_shoot: bool,
        *,
        can_shoot: bool,
    ) -> None:
        """Capture manual play so the AI can mimic the user's tendencies."""
        state = self._state_key(player, enemies, enemy_bullets, can_shoot=can_shoot)
        action_key = self._encode_action(move_dir, should_shoot and can_shoot)
        action_counts = self._ensure_action_store(self._policy_counts, state, int)
        action_counts[action_key] += 1
        self._manual_samples += 1
        self._dirty_updates += 1
        if self._dirty_updates >= 25:
            self.save()

    def apply_reward(self, reward: float, *, done: bool = False) -> None:
        """Apply reward to the in-flight transition."""
        self._pending_reward += reward
        if done:
            self._update_from_transition(next_state=None, done=True)
            self._episodes += 1
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
            self.save()

    def reset_episode(self) -> None:
        """Drop any in-flight transition when control mode or episode changes."""
        self._last_state = None
        self._last_action_key = None
        self._pending_reward = 0.0

    def status_text(self) -> str:
        """Return a short status string for the HUD."""
        return (
            f"eps {self._epsilon:.2f} | episodes {self._episodes}"
            f" | demos {self._manual_samples}"
        )

    def save(self) -> None:
        """Persist learned values to disk."""
        payload = {
            "epsilon": self._epsilon,
            "episodes": self._episodes,
            "manual_samples": self._manual_samples,
            "q_values": self._q_values,
            "policy_counts": self._policy_counts,
        }
        with open(self._memory_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        self._dirty_updates = 0

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
        self._manual_samples = int(payload.get("manual_samples", 0))
        self._q_values = payload.get("q_values", {})
        self._policy_counts = payload.get("policy_counts", {})

    def _choose_action_key(self, state: str, valid_actions: list[str]) -> str:
        self._ensure_action_store(self._q_values, state, float)
        if random.random() < self._epsilon:
            return random.choice(valid_actions)

        best_action = valid_actions[0]
        best_value = self._action_score(state, best_action)
        for action in valid_actions[1:]:
            value = self._action_score(state, action)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def _action_score(self, state: str, action_key: str) -> float:
        q_values = self._ensure_action_store(self._q_values, state, float)
        manual_counts = self._ensure_action_store(self._policy_counts, state, int)
        total_manual = sum(manual_counts.values())
        manual_bias = 0.0
        if total_manual > 0:
            manual_bias = manual_counts[action_key] / total_manual
        return float(q_values[action_key]) + self._imitation_weight * manual_bias

    def _update_from_transition(
        self, next_state: str | None, done: bool = False
    ) -> None:
        if self._last_state is None or self._last_action_key is None:
            if done:
                self.reset_episode()
            return

        action_values = self._ensure_action_store(self._q_values, self._last_state, float)
        old_value = float(action_values[self._last_action_key])
        future_reward = 0.0
        if not done and next_state is not None:
            next_values = self._ensure_action_store(self._q_values, next_state, float)
            future_reward = max(float(value) for value in next_values.values())

        target = self._pending_reward + self._discount_factor * future_reward
        action_values[self._last_action_key] = old_value + self._learning_rate * (target - old_value)
        self._pending_reward = 0.0
        self._dirty_updates += 1
        if done:
            self.reset_episode()

    @staticmethod
    def _ensure_action_store(table: dict[str, dict[str, float]], state: str, value_type):
        state_values = table.setdefault(state, {})
        default_value = value_type()
        for action in LearningAI._ACTIONS:
            state_values.setdefault(LearningAI._encode_action(*action), default_value)
        return state_values

    def _state_key(self, player, enemies, enemy_bullets, *, can_shoot: bool) -> str:
        player_x = player.rect.centerx
        player_bucket = self._bucket_position(player_x)

        nearest_enemy_dx = 0
        nearest_enemy_dy = 9999
        for enemy in enemies:
            dy = enemy.rect.centery - player.rect.centery
            if abs(dy) < abs(nearest_enemy_dy):
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
        enemy_distance = self._bucket_distance(nearest_enemy_dy)
        bullet_bucket = self._bucket_dx(nearest_bullet_dx)
        bullet_distance = self._bucket_distance(nearest_bullet_dy)
        danger_bucket = "danger" if nearest_bullet_dy < 120 and abs(nearest_bullet_dx) < 80 else "safe"
        enemy_count_bucket = self._bucket_enemy_count(len(enemies))
        cooldown_bucket = "ready" if can_shoot else "wait"

        return ":".join(
            (
                player_bucket,
                enemy_bucket,
                enemy_distance,
                bullet_bucket,
                bullet_distance,
                danger_bucket,
                enemy_count_bucket,
                cooldown_bucket,
            )
        )

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

    @staticmethod
    def _bucket_distance(distance: int) -> str:
        if distance == 9999:
            return "none"
        if distance < 90:
            return "near"
        if distance < 220:
            return "mid"
        return "far"

    @staticmethod
    def _bucket_enemy_count(enemy_count: int) -> str:
        if enemy_count <= 4:
            return "few"
        if enemy_count <= 12:
            return "some"
        return "many"

    @staticmethod
    def _bucket_position(player_x: int) -> str:
        lane_width = SCREEN_WIDTH // 5
        lane = max(0, min(4, player_x // lane_width))
        return f"lane_{lane}"

    @classmethod
    def _valid_action_keys(cls, can_shoot: bool) -> list[str]:
        valid_actions: list[str] = []
        for move_dir, should_shoot in cls._ACTIONS:
            if should_shoot and not can_shoot:
                continue
            valid_actions.append(cls._encode_action(move_dir, should_shoot))
        return valid_actions

    @staticmethod
    def _encode_action(move_dir: int, should_shoot: bool) -> str:
        return f"{move_dir}:{1 if should_shoot else 0}"

    @staticmethod
    def _decode_action(action_key: str) -> tuple[int, bool]:
        move_text, shoot_text = action_key.split(":", maxsplit=1)
        return int(move_text), shoot_text == "1"
