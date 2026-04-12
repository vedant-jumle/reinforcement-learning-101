"""Shared utilities for word game environments."""

import os

DEFAULT_WORD_LIST = os.path.join(os.path.dirname(__file__), '..', 'data', 'wordle_list.txt')


def load_word_list(path=None):
    """Load word list from file.

    Args:
        path: Path to word list. Defaults to data/wordle_list.txt relative to package.

    Returns:
        List of lowercase 5-letter word strings.
    """
    if path is None:
        path = DEFAULT_WORD_LIST
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]


def score_guess(guess, target):
    """Score a guess against a target using standard Wordle rules.

    Greens are resolved first, then yellows are assigned left-to-right
    without reusing positions already consumed by a green or yellow.

    Args:
        guess: 5-letter guess string.
        target: 5-letter target string.

    Returns:
        List of 5 strings: 'G' (correct position), 'Y' (wrong position), 'X' (absent).
    """
    result = ['X'] * 5
    target_consumed = [False] * 5

    # First pass: greens
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 'G'
            target_consumed[i] = True

    # Second pass: yellows
    for i in range(5):
        if result[i] == 'G':
            continue
        for j in range(5):
            if not target_consumed[j] and guess[i] == target[j]:
                result[i] = 'Y'
                target_consumed[j] = True
                break

    return result


def filter_candidates(candidates, guess, feedback):
    """Filter word list to words consistent with a guess/feedback pair.

    A word is consistent if scoring the guess against it produces exactly
    the given feedback. Uses score_guess for correctness on duplicate letters.

    Args:
        candidates: List of candidate word strings.
        guess: The guess that was made.
        feedback: List of 5 feedback chars ('G', 'Y', 'X') for that guess.

    Returns:
        Filtered list of candidate strings.
    """
    return [w for w in candidates if score_guess(guess, w) == feedback]
