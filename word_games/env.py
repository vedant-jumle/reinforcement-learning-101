"""Unified word game environment wrapper."""

from .utils import load_word_list, DEFAULT_WORD_LIST
from .wordle import WordleGame
from .quordle import QuordleGame
from .absurdle import AbsurdleGame

GAME_TYPES = ('wordle', 'quordle', 'absurdle')


class WordGameEnv:
    """Unified wrapper for Wordle, Quordle, and Absurdle.

    Provides a single reset()/step() API for all three games.
    action_space is the full valid word list (strings, not indices).
    """

    def __init__(self, game_type, word_list_path=None):
        """Args:
            game_type: One of 'wordle', 'quordle', 'absurdle'.
            word_list_path: Path to word list file.
        """
        if game_type not in GAME_TYPES:
            raise ValueError(f"game_type must be one of {GAME_TYPES}, got {game_type!r}")

        self.game_type = game_type
        path = word_list_path or DEFAULT_WORD_LIST
        self.action_space = load_word_list(path)

        if game_type == 'wordle':
            self._game = WordleGame(path)
        elif game_type == 'quordle':
            self._game = QuordleGame(path)
        elif game_type == 'absurdle':
            self._game = AbsurdleGame(path)

    def reset(self, **kwargs):
        """Reset the underlying game. kwargs passed to game.reset()."""
        return self._game.reset(**kwargs)

    def step(self, word):
        """Execute one guess. Returns (state, reward, done, info)."""
        return self._game.step(word)

    def encode_state(self, state):
        """Return state as-is. LLM handles formatting."""
        return state

    @property
    def game(self):
        return self._game
