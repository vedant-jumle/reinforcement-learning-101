"""Reward functions for Wordle GRPO training.

Reward hierarchy (single-guess-per-example setup):
  1. Format penalty   (-8.0) if no valid <guess> tag parsed
  2. Invalid word     (-4.0) if parsed word not in word list
  3. Repeat penalty   (-10.0) if guess already played this game
  4. Step reward      = (letter_score + candidate_bonus) * 0.5
  5. Think bonus      = +0.5 if guess respects all known constraints, else 0.0
  6. Terminal         = WIN_REWARD[guess_index] on win, LOSS_REWARD on loss (last guess)

Design principles:
  - Terminal signal dominates: win reward (5-20) >> total step rewards (~9 max)
  - Format penalties are all worse than the worst possible game outcome
  - Think bonus rewards constraint-consistent guesses, not text length
  - Step rewards are guidance only, small relative to terminal
"""

import math
import re

from .utils import score_guess, filter_candidates
from .prompts.wordle import parse_guess

# Average letter score of a uniformly random 5-letter guess (~3.5 empirically).
AVG_RANDOM_SCORE = 3.5

# Format/validity penalties — all worse than worst possible game outcome
NO_TAG_PENALTY = -8.0
INVALID_WORD_PENALTY = -4.0
REPEAT_PENALTY = -10.0

# Terminal rewards — dominate all step rewards
WIN_REWARD = {0: 20, 1: 16, 2: 13, 3: 10, 4: 7, 5: 5}   # keyed by 0-based guess_index
LOSS_REWARD = -5.0

# Think bonus
THINK_BONUS_CONSISTENT = 0.5    # guess respects all known constraints
THINK_BONUS_VALID = 0.2         # valid guess but no/empty think block
THINK_PENALTY_VIOLATION = -0.5  # guess violates known constraints


def compute_letter_score(feedback):
    """Score feedback: Green=1.0, Yellow=0.5, X=0. Range [0, 5]."""
    return sum(1.0 if f == 'G' else 0.5 if f == 'Y' else 0.0 for f in feedback)


def compute_candidate_bonus(word_list, guesses_so_far, feedbacks_so_far, new_guess, new_feedback):
    """log2(candidates_before / candidates_after) for the new guess."""
    candidates = list(word_list)
    for g, fb in zip(guesses_so_far, feedbacks_so_far):
        candidates = filter_candidates(candidates, g, fb)
    candidates_before = len(candidates)

    candidates_after = len(filter_candidates(candidates, new_guess, new_feedback))

    if candidates_before <= 0:
        return 0.0
    return math.log2(candidates_before / max(candidates_after, 1))


def check_constraint_violations(guess, guesses_so_far, feedbacks_so_far):
    """Count how many known constraints the guess violates.

    Constraints from prior feedback:
      - Grey letter (X): letter must not appear in guess
      - Green letter (G): letter must appear at that exact position
      - Yellow letter (Y): letter must appear somewhere, but NOT at that position

    Returns:
        int: number of violations (0 = fully consistent)
    """
    if not guesses_so_far:
        return 0

    absent = set()       # letters confirmed absent (grey, not also green/yellow)
    green = {}           # position → letter (confirmed positions)
    yellow = {}          # position → set of letters that can't be there
    present = set()      # letters confirmed present somewhere

    for prior_guess, feedback in zip(guesses_so_far, feedbacks_so_far):
        for i, (letter, fb) in enumerate(zip(prior_guess, feedback)):
            if fb == 'G':
                green[i] = letter
                present.add(letter)
            elif fb == 'Y':
                yellow.setdefault(i, set()).add(letter)
                present.add(letter)
            else:  # X
                absent.add(letter)

    # Letters in present are not truly absent even if they got X elsewhere
    absent -= present

    violations = 0
    for i, letter in enumerate(guess):
        if letter in absent:
            violations += 1
        if i in green and green[i] != letter:
            violations += 1
        if i in yellow and letter in yellow[i]:
            violations += 1

    for letter in present:
        if letter not in guess:
            violations += 1

    return violations


def compute_think_bonus(completion, guess, guesses_so_far, feedbacks_so_far):
    """Reward think block quality based on constraint consistency.

    +0.5 if the guess respects all known constraints (and think block exists)
    +0.2 if valid guess but think block is empty/missing
    -0.5 if the guess violates known constraints
    """
    violations = check_constraint_violations(guess, guesses_so_far, feedbacks_so_far)

    if violations > 0:
        return THINK_PENALTY_VIOLATION

    has_think = bool(re.search(r'<think>\s*.+?\s*</think>', completion, re.DOTALL))
    return THINK_BONUS_CONSISTENT if has_think else THINK_BONUS_VALID


def compute_reward(
    completion,
    target,
    guess_index,
    prev_score,
    word_list,
    guesses_so_far,
    feedbacks_so_far,
    discount_base=0.9,
    candidate_bonus_weight=0.2,
):
    """Compute scalar reward for one model completion (single guess).

    Args:
        completion: Raw model response string (may contain <think> + <guess> tags).
        target: The target word for this game.
        guess_index: 0-based index of this guess in the game (0 = first guess).
        prev_score: Letter score of the previous guess (AVG_RANDOM_SCORE if first).
        word_list: Full word list (list of str).
        guesses_so_far: List of guesses made before this one.
        feedbacks_so_far: Corresponding feedbacks (list of list[str]).
        discount_base: Discount factor applied to positive step rewards.
        candidate_bonus_weight: Weight for candidate reduction bonus.

    Returns:
        Float reward.
    """
    word_set = set(word_list)

    # 1. Parse guess — format failure is the worst outcome
    guess = parse_guess(completion)
    if guess is None:
        return NO_TAG_PENALTY

    # 2. Validate word
    if guess not in word_set:
        return INVALID_WORD_PENALTY

    # 3. Repeat guess — strategic failure, harshest penalty
    if guess in guesses_so_far:
        return REPEAT_PENALTY

    # 4. Step reward (guidance signal, small relative to terminal)
    feedback = score_guess(guess, target)
    letter_score = compute_letter_score(feedback)
    candidate_bonus = compute_candidate_bonus(
        word_list, guesses_so_far, feedbacks_so_far, guess, feedback
    )
    step_score = (letter_score + candidate_bonus_weight * candidate_bonus) * 0.5

    # 5. Think bonus — rewards constraint-consistent reasoning
    think_bonus = compute_think_bonus(completion, guess, guesses_so_far, feedbacks_so_far)

    # 6. Terminal signal — dominates everything
    is_win = feedback == ['G', 'G', 'G', 'G', 'G']
    is_last_guess = guess_index >= 5

    if is_win:
        terminal = float(WIN_REWARD.get(guess_index, 5))
    elif is_last_guess:
        terminal = LOSS_REWARD
    else:
        terminal = 0.0

    return step_score + think_bonus + terminal


def build_reward_fn(word_list, discount_base=0.9, candidate_bonus_weight=0.2):
    """Return a GRPOTrainer-compatible reward function."""
    import json

    def reward_func(prompts, completions, target, guess_index, prev_score,
                    guesses_so_far, feedbacks_so_far, **kwargs):
        rewards = []
        for completion, tgt, gi, ps, gsf, fdsf in zip(
            completions, target, guess_index, prev_score,
            guesses_so_far, feedbacks_so_far
        ):
            gsf_parsed = json.loads(gsf) if isinstance(gsf, str) else gsf
            fdsf_parsed = json.loads(fdsf) if isinstance(fdsf, str) else fdsf

            r = compute_reward(
                completion=completion,
                target=tgt,
                guess_index=int(gi),
                prev_score=float(ps),
                word_list=word_list,
                guesses_so_far=gsf_parsed,
                feedbacks_so_far=fdsf_parsed,
                discount_base=discount_base,
                candidate_bonus_weight=candidate_bonus_weight,
            )
            rewards.append(r)
        return rewards

    return reward_func
