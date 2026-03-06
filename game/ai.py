"""Learning AI controller backed by a small PyTorch Q-network."""

from __future__ import annotations

import json
import os
import random
from collections import deque

import torch
from torch import nn

from game.constants import (
    AI_BATCH_SIZE,
    AI_DISCOUNT_FACTOR,
    AI_EPSILON_DECAY,
    AI_EPSILON_MIN,
    AI_EPSILON_START,
    AI_HIDDEN_DIM,
    AI_IMITATION_WEIGHT,
    AI_LEARNING_RATE,
    AI_MEMORY_FILE,
    AI_BACKGROUND_UPDATES_PER_FRAME,
    AI_MIN_REPLAY_TO_TRAIN,
    AI_MODEL_FILE,
    AI_REPLAY_BUFFER_SIZE,
    AI_TARGET_SYNC_INTERVAL,
    AI_UPDATES_PER_STEP,
    SCREEN_WIDTH,
)


Action = tuple[int, bool]
Transition = tuple[list[float], int, float, list[float] | None, bool, bool]


class LearningAI:
    """PyTorch-powered Q-network with imitation updates from manual play."""

    _ACTIONS: tuple[Action, ...] = (
        (-1, False),
        (0, False),
        (1, False),
        (-1, True),
        (0, True),
        (1, True),
    )
    _PLAYER_BUCKETS = ("lane_0", "lane_1", "lane_2", "lane_3", "lane_4")
    _DX_BUCKETS = ("far_left", "left", "center", "right", "far_right")
    _DISTANCE_BUCKETS = ("none", "near", "mid", "far")
    _DANGER_BUCKETS = ("safe", "danger")
    _ENEMY_COUNT_BUCKETS = ("few", "some", "many")
    _COOLDOWN_BUCKETS = ("ready", "wait")
    _TOKEN_GROUPS = (
        _PLAYER_BUCKETS,
        _DX_BUCKETS,
        _DISTANCE_BUCKETS,
        _DX_BUCKETS,
        _DISTANCE_BUCKETS,
        _DANGER_BUCKETS,
        _ENEMY_COUNT_BUCKETS,
        _COOLDOWN_BUCKETS,
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
        self._model_path = self._derive_model_path(self._memory_path)
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._imitation_weight = imitation_weight
        self._episodes: int = 0
        self._manual_samples: int = 0
        self._train_steps: int = 0
        self._rl_updates: int = 0
        self._imitation_updates: int = 0
        self._policy_counts: dict[str, dict[str, int]] = {}
        self._recent_losses: deque[float] = deque(maxlen=120)
        self._recent_rewards: deque[float] = deque(maxlen=120)
        self._replay_buffer: deque[Transition] = deque(maxlen=AI_REPLAY_BUFFER_SIZE)
        self._last_state_key: str | None = None
        self._last_state_vector: list[float] | None = None
        self._last_action_index: int | None = None
        self._last_can_shoot: bool = False
        self._pending_reward: float = 0.0
        self._dirty_updates: int = 0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = sum(len(group) for group in self._TOKEN_GROUPS)
        output_dim = len(self._ACTIONS)
        self._policy_net = self._build_network(input_dim, output_dim)
        self._target_net = self._build_network(input_dim, output_dim)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self._learning_rate)
        self._loss_fn = nn.SmoothL1Loss()

        self._load()

    def choose_actions(
        self, player, enemies, enemy_bullets, *, can_shoot: bool
    ) -> tuple[int, bool]:
        """Return movement direction (-1, 0, 1) and whether to shoot."""
        state_key, state_vector = self._encode_state(
            player, enemies, enemy_bullets, can_shoot=can_shoot
        )
        self._update_from_transition(
            next_state_vector=state_vector,
            next_can_shoot=can_shoot,
        )

        action_index = self._choose_action_index(state_key, state_vector, can_shoot)
        self._last_state_key = state_key
        self._last_state_vector = state_vector
        self._last_action_index = action_index
        self._last_can_shoot = can_shoot
        return self._ACTIONS[action_index]

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
        """Capture manual play so the model learns from user demonstrations."""
        state_key, state_vector = self._encode_state(
            player, enemies, enemy_bullets, can_shoot=can_shoot
        )
        action = (move_dir, should_shoot and can_shoot)
        action_key = self._encode_action(*action)
        action_counts = self._ensure_action_store(self._policy_counts, state_key)
        action_counts[action_key] += 1
        self._manual_samples += 1
        self._train_imitation_step(state_vector, self._action_to_index(action))
        self._dirty_updates += 1
        if self._dirty_updates >= 25:
            self.save()

    def apply_reward(self, reward: float, *, done: bool = False) -> None:
        """Apply reward to the in-flight transition."""
        self._pending_reward += reward
        self._recent_rewards.append(reward)
        if done:
            self._update_from_transition(next_state_vector=None, next_can_shoot=False, done=True)
            self._episodes += 1
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
            self.save()

    def reset_episode(self) -> None:
        """Drop any in-flight transition when control mode or episode changes."""
        self._last_state_key = None
        self._last_state_vector = None
        self._last_action_index = None
        self._last_can_shoot = False
        self._pending_reward = 0.0

    def status_text(self) -> str:
        """Return a short status string for the HUD."""
        loss_value = self._average_loss()
        loss_text = f"{loss_value:.4f}" if loss_value is not None else "--"
        return (
            f"torch/{self.device_name()} | eps {self._epsilon:.2f} | ep {self._episodes}"
            f" | loss {loss_text}"
        )

    def training_overlay_lines(self) -> list[str]:
        """Return detailed metrics that can be rendered in the HUD."""
        avg_loss = self._average_loss()
        avg_reward = self._average_reward()
        return [
            f"backend: torch  device: {self.device_name()}",
            f"episodes: {self._episodes}  epsilon: {self._epsilon:.3f}  demos: {self._manual_samples}",
            f"updates: {self._train_steps}  rl: {self._rl_updates}  imitation: {self._imitation_updates}",
            f"loss: {avg_loss:.4f}" if avg_loss is not None else "loss: --",
            f"avg reward: {avg_reward:.3f}  replay: {len(self._replay_buffer)}",
        ]

    def device_name(self) -> str:
        return "cuda" if self._device.type == "cuda" else "cpu"

    def train_background(self) -> None:
        """Run extra replay-only updates to keep the GPU busier during play."""
        self._train_from_replay(update_count=AI_BACKGROUND_UPDATES_PER_FRAME)

    def save(self) -> None:
        """Persist model weights, optimizer state, and training metadata to disk."""
        payload = {
            "backend": "torch",
            "device": self.device_name(),
            "epsilon": self._epsilon,
            "episodes": self._episodes,
            "manual_samples": self._manual_samples,
            "train_steps": self._train_steps,
            "rl_updates": self._rl_updates,
            "imitation_updates": self._imitation_updates,
            "policy_counts": self._policy_counts,
        }
        with open(self._memory_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        torch.save(
            {
                "policy_state": self._policy_net.state_dict(),
                "target_state": self._target_net.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
            },
            self._model_path,
        )
        self._dirty_updates = 0

    def _load(self) -> None:
        payload = None
        if os.path.exists(self._memory_path):
            try:
                with open(self._memory_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload:
            self._epsilon = max(self._epsilon_min, float(payload.get("epsilon", self._epsilon)))
            self._episodes = int(payload.get("episodes", 0))
            self._manual_samples = int(payload.get("manual_samples", 0))
            self._train_steps = int(payload.get("train_steps", 0))
            self._rl_updates = int(payload.get("rl_updates", 0))
            self._imitation_updates = int(payload.get("imitation_updates", 0))
            self._policy_counts = payload.get("policy_counts", {})

        if os.path.exists(self._model_path):
            try:
                checkpoint = torch.load(self._model_path, map_location=self._device)
            except (OSError, RuntimeError):
                checkpoint = None
            if checkpoint:
                policy_state = checkpoint.get("policy_state", {})
                target_state = checkpoint.get("target_state", policy_state)
                policy_loaded = self._load_matching_state_dict(self._policy_net, policy_state)
                target_loaded = self._load_matching_state_dict(self._target_net, target_state)
                optimizer_state = checkpoint.get("optimizer_state")
                if optimizer_state and policy_loaded and target_loaded:
                    self._optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def _load_matching_state_dict(model: nn.Module, state_dict: object) -> bool:
        if not isinstance(state_dict, dict):
            return False

        current_state = model.state_dict()
        if set(state_dict.keys()) != set(current_state.keys()):
            return False

        for key, current_value in current_state.items():
            if getattr(state_dict[key], "shape", None) != current_value.shape:
                return False

        model.load_state_dict(state_dict, strict=True)
        return True

    def _build_network(self, input_dim: int, output_dim: int) -> nn.Module:
        network = nn.Sequential(
            nn.Linear(input_dim, AI_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(AI_HIDDEN_DIM, AI_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(AI_HIDDEN_DIM, output_dim),
        ).to(self._device)
        final_layer = network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        return network

    def _choose_action_index(
        self,
        state_key: str,
        state_vector: list[float],
        can_shoot: bool,
    ) -> int:
        valid_indices = self._valid_action_indices(can_shoot)
        if random.random() < self._epsilon:
            return random.choice(valid_indices)

        state_tensor = self._vector_tensor(state_vector).unsqueeze(0)
        with torch.no_grad():
            q_values = self._policy_net(state_tensor).squeeze(0)
            manual_bias = self._manual_bias_tensor(state_key)
            scores = q_values + self._imitation_weight * manual_bias

        best_index = valid_indices[0]
        best_score = float(scores[best_index].item())
        for action_index in valid_indices[1:]:
            score = float(scores[action_index].item())
            if score > best_score:
                best_index = action_index
                best_score = score
        return best_index

    def _manual_bias_tensor(self, state_key: str) -> torch.Tensor:
        manual_counts = self._ensure_action_store(self._policy_counts, state_key)
        total_manual = sum(manual_counts.values())
        values = []
        for action in self._ACTIONS:
            action_key = self._encode_action(*action)
            if total_manual == 0:
                values.append(0.0)
            else:
                values.append(manual_counts[action_key] / total_manual)
        return torch.tensor(values, dtype=torch.float32, device=self._device)

    def _update_from_transition(
        self,
        *,
        next_state_vector: list[float] | None,
        next_can_shoot: bool,
        done: bool = False,
    ) -> None:
        if self._last_state_vector is None or self._last_action_index is None:
            if done:
                self.reset_episode()
            return

        transition: Transition = (
            list(self._last_state_vector),
            self._last_action_index,
            self._pending_reward,
            list(next_state_vector) if next_state_vector is not None else None,
            next_can_shoot,
            done,
        )
        self._replay_buffer.append(transition)
        self._train_from_replay()
        self._pending_reward = 0.0
        self._dirty_updates += 1
        if done:
            self.reset_episode()

    def _train_imitation_step(self, state_vector: list[float], action_index: int) -> None:
        state_tensor = self._vector_tensor(state_vector).unsqueeze(0)
        action_tensor = torch.tensor([action_index], dtype=torch.long, device=self._device)

        self._policy_net.train()
        logits = self._policy_net(state_tensor)
        loss = nn.functional.cross_entropy(logits, action_tensor)
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        self._train_steps += 1
        self._imitation_updates += 1
        self._recent_losses.append(float(loss.item()))
        self._sync_target_if_needed()

    def _train_from_replay(self, update_count: int = AI_UPDATES_PER_STEP) -> None:
        if len(self._replay_buffer) < AI_MIN_REPLAY_TO_TRAIN:
            return

        for _ in range(update_count):
            batch_size = min(AI_BATCH_SIZE, len(self._replay_buffer))
            batch = random.sample(self._replay_buffer, batch_size)
            states = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=self._device)
            actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self._device)
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self._device)
            non_terminal_mask = torch.tensor(
                [item[3] is not None and not item[5] for item in batch],
                dtype=torch.bool,
                device=self._device,
            )

            next_state_values = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
            if bool(non_terminal_mask.any().item()):
                next_states = torch.tensor(
                    [item[3] for item in batch if item[3] is not None and not item[5]],
                    dtype=torch.float32,
                    device=self._device,
                )
                next_can_shoot_values = [item[4] for item in batch if item[3] is not None and not item[5]]
                with torch.no_grad():
                    target_q = self._target_net(next_states)
                    best_future = []
                    for row_index, can_shoot in enumerate(next_can_shoot_values):
                        valid_indices = self._valid_action_indices(can_shoot)
                        best_future.append(torch.max(target_q[row_index, valid_indices]))
                    next_state_values[non_terminal_mask] = torch.stack(best_future)

            targets = rewards + self._discount_factor * next_state_values
            current_q = self._policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = self._loss_fn(current_q, targets)

            self._policy_net.train()
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()

            self._train_steps += 1
            self._rl_updates += 1
            self._recent_losses.append(float(loss.item()))
            self._sync_target_if_needed()

    def _sync_target_if_needed(self) -> None:
        if self._train_steps % AI_TARGET_SYNC_INTERVAL == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

    def _encode_state(
        self,
        player,
        enemies,
        enemy_bullets,
        *,
        can_shoot: bool,
    ) -> tuple[str, list[float]]:
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

        tokens = (
            player_bucket,
            self._bucket_dx(nearest_enemy_dx),
            self._bucket_distance(nearest_enemy_dy),
            self._bucket_dx(nearest_bullet_dx),
            self._bucket_distance(nearest_bullet_dy),
            "danger" if nearest_bullet_dy < 120 and abs(nearest_bullet_dx) < 80 else "safe",
            self._bucket_enemy_count(len(enemies)),
            "ready" if can_shoot else "wait",
        )
        return ":".join(tokens), self._state_vector(tokens)

    def _state_vector(self, tokens: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for token, token_group in zip(tokens, self._TOKEN_GROUPS, strict=True):
            for candidate in token_group:
                values.append(1.0 if token == candidate else 0.0)
        return values

    def _vector_tensor(self, state_vector: list[float]) -> torch.Tensor:
        return torch.tensor(state_vector, dtype=torch.float32, device=self._device)

    def _average_loss(self) -> float | None:
        if not self._recent_losses:
            return None
        return sum(self._recent_losses) / len(self._recent_losses)

    def _average_reward(self) -> float:
        if not self._recent_rewards:
            return 0.0
        return sum(self._recent_rewards) / len(self._recent_rewards)

    @staticmethod
    def _derive_model_path(memory_path: str) -> str:
        if memory_path.endswith(".json"):
            return memory_path[:-5] + ".pt"
        directory = os.path.dirname(memory_path)
        return os.path.join(directory, AI_MODEL_FILE)

    @staticmethod
    def _ensure_action_store(table: dict[str, dict[str, int]], state: str) -> dict[str, int]:
        state_values = table.setdefault(state, {})
        for action in LearningAI._ACTIONS:
            state_values.setdefault(LearningAI._encode_action(*action), 0)
        return state_values

    @staticmethod
    def _action_to_index(action: Action) -> int:
        return LearningAI._ACTIONS.index(action)

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
    def _valid_action_indices(cls, can_shoot: bool) -> list[int]:
        valid_actions: list[int] = []
        for index, (_, should_shoot) in enumerate(cls._ACTIONS):
            if should_shoot and not can_shoot:
                continue
            valid_actions.append(index)
        return valid_actions

    @staticmethod
    def _encode_action(move_dir: int, should_shoot: bool) -> str:
        return f"{move_dir}:{1 if should_shoot else 0}"
