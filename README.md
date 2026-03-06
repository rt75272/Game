# Space Invaders 🎮

A graphical Space Invaders game built with Python and [pygame](https://www.pygame.org/).

## Features

* **Animated alien sprites** – two-frame walking animation per enemy row, each row with its own colour palette
* **Scrolling starfield** – parallax star background that scrolls continuously
* **Explosion effects** – multi-frame animated explosions with radiating sparks
* **Progressive difficulty** – enemy movement speed increases with every cleared level
* **HUD** – live score, lives remaining, and current level displayed on screen
* **Game-state machine** – Menu → Playing → Game Over / Level Clear, all with smooth transitions
* **Learning AI mode** – built-in PyTorch AI learns from your manual play, improves from rewards, persists knowledge across runs, and shows live training metrics in the HUD

## Controls

| Key | Action |
|-----|--------|
| `←` / `A` | Move left |
| `→` / `D` | Move right |
| `Space` | Shoot |
| `T` | Toggle AI on/off |
| `Space` / `Enter` | Confirm menu selection |

## Setup

```bash
uv sync
uv run space-invaders
```

If CUDA is available, the AI training loop automatically runs on the GPU through PyTorch. When no compatible GPU is present, it falls back to the CPU and the HUD shows the active device.

## Project Structure

```
Game/
├── main.py              # Entry point
├── pyproject.toml
├── game/
│   ├── constants.py     # Screen size, colours, gameplay tuning values
│   ├── game.py          # Main Game class – loop, state machine, rendering
│   └── sprites/
│       ├── player.py    # Player spaceship
│       ├── enemy.py     # Alien enemies (animated)
│       ├── bullet.py    # Player & enemy projectiles
│       ├── explosion.py # Frame-based explosion animation
│       └── star.py      # Scrolling background stars
└── tests/
    └── test_game.py     # Headless unit tests (pytest)
```

## Running Tests

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run pytest tests/ -v
```
