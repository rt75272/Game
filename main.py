"""Entry point for the Space Invaders game.

Run with::

    python main.py
"""

from game.game import Game


def main() -> None:
    """Initialise and run the game."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
