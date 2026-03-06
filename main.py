"""Entry point for the Space Invaders game.

Run with::

    uv run space-invaders
"""

from game.game import Game


def main() -> None:
    """Initialise and run the game."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
