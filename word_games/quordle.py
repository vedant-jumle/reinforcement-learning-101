"""Quordle environment — 4 simultaneous Wordle boards, 9 guesses."""

import random
from .utils import load_word_list, score_guess, DEFAULT_WORD_LIST

NUM_BOARDS = 4
MAX_GUESSES = 9


class QuordleGame:
    """Solve 4 Wordle boards simultaneously in up to 9 guesses.

    Each guess applies to all 4 boards. Win requires all 4 solved.
    Reward: +0.25 per new board solved, -1.0 on loss.
    """

    def __init__(self, word_list_path=None):
        self.word_list = load_word_list(word_list_path or DEFAULT_WORD_LIST)
        self.word_set = set(self.word_list)
        self.targets = []
        self.guesses = []
        self.feedbacks = []
        self.boards_solved = [False] * NUM_BOARDS
        self.done = False
        self.won = False

    def reset(self, targets=None):
        """Reset game. targets: list of 4 words or None for random."""
        if targets is not None:
            assert len(targets) == NUM_BOARDS, f"Need exactly {NUM_BOARDS} targets"
            self.targets = [t.lower() for t in targets]
        else:
            self.targets = random.sample(self.word_list, NUM_BOARDS)
        self.guesses = []
        self.feedbacks = []
        self.boards_solved = [False] * NUM_BOARDS
        self.done = False
        self.won = False
        return self.get_state()

    def step(self, word):
        """Guess a word across all 4 boards. Returns (state, reward, done, info)."""
        if self.done:
            return self.get_state(), 0.0, True, {'needs_reset': True}

        word = word.lower()
        if word not in self.word_set:
            return self.get_state(), 0.0, False, {'error': f'"{word}" not in valid word list'}

        self.guesses.append(word)
        guess_feedbacks = []
        newly_solved = 0

        for i, target in enumerate(self.targets):
            fb = score_guess(word, target)
            guess_feedbacks.append(fb)
            if not self.boards_solved[i] and word == target:
                self.boards_solved[i] = True
                newly_solved += 1

        self.feedbacks.append(guess_feedbacks)
        reward = newly_solved * 0.25

        if all(self.boards_solved):
            self.won = True
            self.done = True
            return self.get_state(), reward, True, {}

        if len(self.guesses) >= MAX_GUESSES:
            self.done = True
            unsolved = [self.targets[i] for i in range(NUM_BOARDS) if not self.boards_solved[i]]
            return self.get_state(), -1.0, True, {'unsolved': unsolved}

        return self.get_state(), reward, False, {}

    def get_state(self):
        return {
            'guesses': list(self.guesses),
            'feedbacks': [[list(fb) for fb in guess_row] for guess_row in self.feedbacks],
            'boards_solved': list(self.boards_solved),
            'guess_count': len(self.guesses),
            'done': self.done,
            'won': self.won,
        }
