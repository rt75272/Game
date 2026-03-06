"""Game-wide constants and configuration values."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Screen settings
# ---------------------------------------------------------------------------
SCREEN_WIDTH: int = 800
SCREEN_HEIGHT: int = 600
FPS: int = 60
TITLE: str = "Space Invaders"

# ---------------------------------------------------------------------------
# Colors (R, G, B)
# ---------------------------------------------------------------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 220, 50)
DARK_GREEN = (0, 150, 0)
RED = (220, 60, 60)
BRIGHT_RED = (255, 50, 50)
BLUE = (50, 100, 255)
CYAN = (0, 220, 255)
YELLOW = (255, 220, 50)
ORANGE = (255, 140, 0)
PURPLE = (180, 50, 220)
GRAY = (150, 150, 150)
DARK_GRAY = (40, 40, 40)
LIGHT_CYAN = (180, 255, 255)

# ---------------------------------------------------------------------------
# Player settings
# ---------------------------------------------------------------------------
PLAYER_SPEED: int = 5
PLAYER_SHOOT_DELAY: int = 500        # milliseconds between shots
PLAYER_LIVES: int = 3

# ---------------------------------------------------------------------------
# Enemy settings
# ---------------------------------------------------------------------------
ENEMY_ROWS: int = 3
ENEMY_COLS: int = 8
ENEMY_H_SPACING: int = 70           # horizontal gap between enemy centres
ENEMY_V_SPACING: int = 60           # vertical gap between enemy centres
ENEMY_OFFSET_X: int = 100           # left margin for the first enemy
ENEMY_OFFSET_Y: int = 80            # top margin for the first row
ENEMY_MOVE_DELAY_START: int = 800   # ms between lateral moves (decreases)
ENEMY_MOVE_STEP: int = 20           # pixels per lateral move
ENEMY_DROP_AMOUNT: int = 20         # pixels to drop when hitting screen edge
ENEMY_ANIM_DELAY: int = 600         # ms between sprite animation frames

# ---------------------------------------------------------------------------
# Bullet settings
# ---------------------------------------------------------------------------
PLAYER_BULLET_SPEED: int = -7      # negative = up
ENEMY_BULLET_SPEED: int = 7
ENEMY_SHOOT_CHANCE: float = 0.001   # per-enemy probability each frame

# ---------------------------------------------------------------------------
# Explosion settings
# ---------------------------------------------------------------------------
EXPLOSION_FRAMES: int = 8           # number of animation frames
EXPLOSION_DURATION: int = 600       # total ms for explosion animation

# ---------------------------------------------------------------------------
# Star settings
# ---------------------------------------------------------------------------
NUM_STARS: int = 80

# ---------------------------------------------------------------------------
# AI settings
# ---------------------------------------------------------------------------
AI_ENABLED_BY_DEFAULT: bool = True
AI_EPSILON_START: float = 0.18
AI_EPSILON_MIN: float = 0.03
AI_EPSILON_DECAY: float = 0.995
AI_LEARNING_RATE: float = 0.25
AI_DISCOUNT_FACTOR: float = 0.92
AI_IMITATION_WEIGHT: float = 0.6
AI_MEMORY_FILE: str = "ai_memory.json"
AI_MODEL_FILE: str = "ai_model.pt"
AI_HIDDEN_DIM: int = 128
AI_BATCH_SIZE: int = 128
AI_REPLAY_BUFFER_SIZE: int = 8192
AI_TARGET_SYNC_INTERVAL: int = 64
AI_MIN_REPLAY_TO_TRAIN: int = 32
AI_UPDATES_PER_STEP: int = 2
AI_BACKGROUND_UPDATES_PER_FRAME: int = 2
AI_BACKGROUND_TRAIN_INTERVAL_FRAMES: int = 6
AI_REWARD_SURVIVAL: float = 0.02
AI_REWARD_ALIGNMENT: float = 0.03
AI_REWARD_ENEMY_DESTROYED: float = 2.5
AI_REWARD_LEVEL_CLEAR: float = 5.0
AI_PENALTY_SHOT: float = -0.005
AI_PENALTY_PLAYER_HIT: float = -4.0
AI_PENALTY_GAME_OVER: float = -8.0
