# Space Invaders 🎮

A graphical Space Invaders game built with Python and [pygame](https://www.pygame.org/).

## Features

* **Animated alien sprites** – two-frame walking animation per enemy row, each row with its own colour palette
* **Scrolling starfield** – parallax star background that scrolls continuously
* **Explosion effects** – multi-frame animated explosions with radiating sparks
* **Progressive difficulty** – enemy movement speed increases with every cleared level
* **HUD** – live score, lives remaining, and current level displayed on screen
* **Game-state machine** – Menu → Playing → Game Over / Level Clear, all with smooth transitions
* **Learning AI mode** – built-in AI can take control, learn from rewards, and persist knowledge across runs

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
pip install -r requirements.txt
python main.py
```

## Project Structure

```
Game/
├── main.py              # Entry point
├── requirements.txt
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
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy pytest tests/ -v
```
