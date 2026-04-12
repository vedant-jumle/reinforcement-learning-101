"""Absurdle environment — adversarial Wordle with no fixed secret word."""

from collections import defaultdict
from .utils import load_word_list, score_guess, DEFAULT_WORD_LIST


class AbsurdleGame:
    """Adversarial Wordle: the adversary maintains a candidate set and picks
    the feedback that maximizes remaining candidates after each guess.

    No fixed secret word at start. Win occurs when adversary is forced to
    give all-green feedback (candidates collapsed to 1 and agent guesses it).
    No loss state — reward is -1 * guess_count on win.
    """

    def __init__(self, word_list_path=None):
        self.word_list = load_word_list(word_list_path or DEFAULT_WORD_LIST)
        self.word_set = set(self.word_list)
        self.candidates = None
        self.guesses = []
        self.feedbacks = []
        self.done = False
        self.won = False

    def reset(self):
        """Reset game with full candidate set."""
        self.candidates = set(self.word_list)
        self.guesses = []
        self.feedbacks = []
        self.done = False
        self.won = False
        return self.get_state()

    def step(self, word):
        """Guess a word. Adversary picks worst-case feedback.

        Returns (state, reward, done, info).
        """
        if self.done:
            return self.get_state(), 0.0, True, {'needs_reset': True}

        word = word.lower()
        if word not in self.word_set:
            return self.get_state(), 0.0, False, {'error': f'"{word}" not in valid word list'}

        # Partition candidates by feedback; pick largest partition
        partitions = defaultdict(list)
        for candidate in self.candidates:
            fb = tuple(score_guess(word, candidate))
            partitions[fb].append(candidate)

        best_fb_tuple = max(partitions, key=lambda k: len(partitions[k]))
        best_fb = list(best_fb_tuple)

        self.candidates = set(partitions[best_fb_tuple])
        self.guesses.append(word)
        self.feedbacks.append(best_fb)

        if best_fb == ['G', 'G', 'G', 'G', 'G']:
            self.won = True
            self.done = True
            return self.get_state(), -1.0 * len(self.guesses), True, {}

        return self.get_state(), 0.0, False, {}

    def get_state(self):
        return {
            'guesses': list(self.guesses),
            'feedbacks': [list(fb) for fb in self.feedbacks],
            'candidates_remaining': len(self.candidates),
            'guess_count': len(self.guesses),
            'done': self.done,
            'won': self.won,
        }
