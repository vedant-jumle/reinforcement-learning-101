"""Wordle environment — standard 6-guess single-word game."""

import random
from .utils import load_word_list, score_guess, DEFAULT_WORD_LIST

MAX_GUESSES = 6


class WordleGame:
    """Guess a 5-letter word in up to 6 tries.

    reset() / step(word) → (state, reward, done, info) API.
    Reward: +1.0 on win, -1.0 on loss (6 wrong guesses), 0.0 intermediate.
    """

    def __init__(self, word_list_path=None):
        self.word_list = load_word_list(word_list_path or DEFAULT_WORD_LIST)
        self.word_set = set(self.word_list)
        self.target = None
        self.guesses = []
        self.feedbacks = []
        self.done = False
        self.won = False

    def reset(self, target=None):
        """Reset game. target: fixed word or None for random."""
        self.target = target.lower() if target else random.choice(self.word_list)
        self.guesses = []
        self.feedbacks = []
        self.done = False
        self.won = False
        return self.get_state()

    def step(self, word):
        """Guess a word. Returns (state, reward, done, info)."""
        if self.done:
            return self.get_state(), 0.0, True, {'needs_reset': True}

        word = word.lower()
        if word not in self.word_set:
            return self.get_state(), 0.0, False, {'error': f'"{word}" not in valid word list'}

        feedback = score_guess(word, self.target)
        self.guesses.append(word)
        self.feedbacks.append(feedback)

        if word == self.target:
            self.won = True
            self.done = True
            return self.get_state(), 1.0, True, {}

        if len(self.guesses) >= MAX_GUESSES:
            self.done = True
            return self.get_state(), -1.0, True, {'target': self.target}

        return self.get_state(), 0.0, False, {}

    def get_state(self):
        return {
            'guesses': list(self.guesses),
            'feedbacks': [list(fb) for fb in self.feedbacks],
            'guess_count': len(self.guesses),
            'done': self.done,
            'won': self.won,
        }
