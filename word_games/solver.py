"""Entropy-maximizing Wordle solver.

Uses Shannon entropy over remaining candidate partitions to pick the
guess that reveals the most information on average.

Usage:
    from word_games.solver import find_best_guess, get_remaining_candidates
    from word_games.utils import load_word_list, filter_candidates

    word_list = load_word_list()
    candidates = word_list[:]

    # First guess (all candidates remain)
    guess = find_best_guess(candidates, word_list)

    # After feedback, narrow candidates and pick next
    candidates = filter_candidates(candidates, guess, feedback)
    guess = find_best_guess(candidates, word_list)
"""

import math
from .utils import score_guess, filter_candidates


def find_best_guess(possible_words: list[str], allowed_guesses: list[str]) -> str:
    """Return the guess that maximises Shannon entropy over remaining answers.

    Args:
        possible_words: Words still consistent with all feedback so far.
        allowed_guesses: Full valid guess pool (superset of possible_words).

    Returns:
        Best guess string (lowercase).
    """
    if not possible_words:
        return allowed_guesses[0] if allowed_guesses else ""

    if len(possible_words) <= 2:
        return possible_words[0]

    # When solution space is large, restrict search to intersection for speed.
    # When small, search full allowed list (a non-answer may be best probe).
    if len(possible_words) > 100:
        guess_pool = list(set(allowed_guesses) & set(possible_words))
        if not guess_pool:
            guess_pool = allowed_guesses
    else:
        guess_pool = allowed_guesses

    best_guess = possible_words[0]
    best_entropy = -1.0
    n = len(possible_words)

    for guess in guess_pool:
        # Partition remaining answers by the feedback this guess would produce.
        partition: dict[tuple, int] = {}
        for answer in possible_words:
            fb = tuple(score_guess(guess, answer))
            partition[fb] = partition.get(fb, 0) + 1

        # Shannon entropy: -Σ p * log2(p)
        entropy = 0.0
        for count in partition.values():
            p = count / n
            entropy -= p * math.log2(p)

        if entropy > best_entropy:
            best_entropy = entropy
            best_guess = guess

    return best_guess


def get_remaining_candidates(
    word_list: list[str],
    guesses: list[str],
    feedbacks: list[list[str]],
) -> list[str]:
    """Filter word_list to words consistent with all (guess, feedback) pairs.

    Args:
        word_list: Full word list.
        guesses: Guesses made so far.
        feedbacks: Feedback for each guess (list of ['G','Y','X',...]).

    Returns:
        Filtered candidate list.
    """
    candidates = list(word_list)
    for guess, feedback in zip(guesses, feedbacks):
        candidates = filter_candidates(candidates, guess, feedback)
    return candidates


def play_solver_game(
    target: str,
    word_list: list[str],
    max_guesses: int = 6,
) -> tuple[list[str], list[list[str]], bool]:
    """Play a complete game using the entropy solver.

    Args:
        target: The secret word.
        word_list: Full valid word list.
        max_guesses: Maximum allowed guesses.

    Returns:
        (guesses, feedbacks, solved) tuple.
    """
    guesses: list[str] = []
    feedbacks: list[list[str]] = []
    candidates = list(word_list)

    for _ in range(max_guesses):
        guess = find_best_guess(candidates, word_list)
        feedback = score_guess(guess, target)

        guesses.append(guess)
        feedbacks.append(feedback)

        if feedback == ['G', 'G', 'G', 'G', 'G']:
            return guesses, feedbacks, True

        candidates = filter_candidates(candidates, guess, feedback)
        if not candidates:
            break

    return guesses, feedbacks, False
