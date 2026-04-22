"""Reward functions for Wordle GRPO training.

Reward hierarchy (single-guess-per-example setup):
  1. Format penalty   (-2.0) if no valid <guess> tag parsed
  2. Invalid word     (-1.0) if parsed word not in word list
  3. Repeat penalty   (-5.0) if guess already played this game
  4. Think bonus      bell-curve over <think> block char length (max +1.0)
  5. Step reward      = improvement * discount^guess_index  (positive)
                      = improvement                         (negative, no discount)
                      = -0.5 if improvement == 0 (small nudge against stagnation)
  6. Terminal bonus   = (7 - (guess_index+1)) if all-green feedback (win)

Step score = letter_score + candidate_bonus_weight * log2(candidates_before / candidates_after)
Improvement = step_score - prev_score
Baseline for guess 0: AVG_RANDOM_SCORE (~3.5) instead of 0, so model must beat random.
"""

import math
import re

from .utils import score_guess, filter_candidates
from .prompts.wordle import parse_guess

# Average letter score of a uniformly random 5-letter guess (~3.5 empirically).
# Used as the baseline prev_score for the first guess so the model must beat random.
AVG_RANDOM_SCORE = 3.5

# Think block reward parameters (bell curve)
_THINK_PEAK_CHARS = 120     # chars at which think reward peaks
_THINK_SIGMA = 80           # width of the bell curve
_THINK_MAX_REWARD = 1.0     # ceiling


def compute_think_reward(response):
    """Bell-curve reward over <think> block character length.

    Peaks at _THINK_PEAK_CHARS chars, decays symmetrically on both sides.
    Empty/no think → 0. Too long → reward decays back toward 0.
    The raw bell curve is normalised so the peak = _THINK_MAX_REWARD and
    the value at 0 chars is subtracted out so empty think gives exactly 0.
    """
    match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    think_len = len(match.group(1).strip()) if match else 0

    def _bell(x):
        return math.exp(-((x - _THINK_PEAK_CHARS) ** 2) / (2 * _THINK_SIGMA ** 2))

    raw = _bell(think_len) - _bell(0)
    # Normalise so peak (at _THINK_PEAK_CHARS) maps to _THINK_MAX_REWARD
    normaliser = _bell(_THINK_PEAK_CHARS) - _bell(0)
    return _THINK_MAX_REWARD * max(raw / normaliser, 0.0)


def compute_letter_score(feedback):
    """Score feedback: Green=2, Yellow=1, X=0. Range [0, 10]."""
    return sum(2 if f == 'G' else 1 if f == 'Y' else 0 for f in feedback)


def compute_candidate_bonus(word_list, guesses_so_far, feedbacks_so_far, new_guess, new_feedback):
    """log2(candidates_before / candidates_after) for the new guess.

    Candidates_before: words consistent with all prior guesses.
    Candidates_after:  words consistent with all prior guesses + new guess.

    Args:
        word_list: Full word list.
        guesses_so_far: Guesses before the new one (may be empty).
        feedbacks_so_far: Feedbacks for those guesses.
        new_guess: The new guess being scored.
        new_feedback: Feedback for the new guess.

    Returns:
        Float >= 0. Returns 0 if candidates_before == 0 (degenerate).
    """
    candidates = list(word_list)
    for g, fb in zip(guesses_so_far, feedbacks_so_far):
        candidates = filter_candidates(candidates, g, fb)
    candidates_before = len(candidates)

    candidates_after_list = filter_candidates(candidates, new_guess, new_feedback)
    candidates_after = len(candidates_after_list)

    if candidates_before <= 0:
        return 0.0
    return math.log2(candidates_before / max(candidates_after, 1))


def compute_reward(
    completion,
    target,
    guess_index,
    prev_score,
    word_list,
    guesses_so_far,
    feedbacks_so_far,
    discount_base=0.9,
    candidate_bonus_weight=0.3,
):
    """Compute scalar reward for one model completion (single guess).

    Args:
        completion: Raw model response string (may contain <think> + <guess> tags).
        target: The target word for this game.
        guess_index: 0-based index of this guess in the game (0 = first guess).
        prev_score: Letter score of the previous guess (0.0 if first guess).
        word_list: Full word list (list of str).
        guesses_so_far: List of guesses made before this one.
        feedbacks_so_far: Corresponding feedbacks (list of list[str]).
        discount_base: Discount factor applied to positive step rewards.
        candidate_bonus_weight: Weight for candidate reduction bonus.

    Returns:
        Float reward.
    """
    word_set = set(word_list)

    # 1. Parse guess
    guess = parse_guess(completion)
    if guess is None:
        return -2.0

    # 2. Validate word
    if guess not in word_set:
        return -1.0

    # 3. Hard penalty for repeating a guess already played this game
    if guess in guesses_so_far:
        return -5.0

    # 4. Think block bonus (bell curve, max +1.0)
    think_bonus = compute_think_reward(completion)

    # 5. Score the guess
    feedback = score_guess(guess, target)
    letter_score = compute_letter_score(feedback)

    candidate_bonus = compute_candidate_bonus(
        word_list, guesses_so_far, feedbacks_so_far, guess, feedback
    )
    step_score = letter_score + candidate_bonus_weight * candidate_bonus

    # Baseline: first guess is compared against average random score, not zero,
    # so the model must beat random to get a positive reward on guess 0.
    baseline = prev_score if guess_index > 0 else AVG_RANDOM_SCORE
    improvement = step_score - baseline

    if improvement > 0:
        step_reward = improvement * (discount_base ** guess_index)
    elif improvement < 0:
        step_reward = improvement   # full penalty, no discount
    else:
        step_reward = -0.5          # small nudge against stagnation

    # 6. Terminal bonus: win detected when feedback is all-green
    terminal = 0.0
    if feedback == ['G', 'G', 'G', 'G', 'G']:
        terminal = float(7 - (guess_index + 1))   # win in 1 → 6, win in 6 → 1

    return think_bonus + step_reward + terminal


def build_reward_fn(word_list, discount_base=0.9, candidate_bonus_weight=0.3):
    """Return a GRPOTrainer-compatible reward function.

    GRPOTrainer calls reward_func(prompts, completions, **dataset_columns).
    The dataset must include columns: target, guess_index, prev_score,
    guesses_so_far (JSON string), feedbacks_so_far (JSON string).

    Args:
        word_list: Full word list.
        discount_base: Discount factor for positive step rewards.
        candidate_bonus_weight: Weight for candidate reduction bonus.

    Returns:
        Callable matching GRPOTrainer's reward_funcs signature.
    """
    import json

    def reward_func(prompts, completions, target, guess_index, prev_score,
                    guesses_so_far, feedbacks_so_far, **kwargs):
        rewards = []
        for completion, tgt, gi, ps, gsf, fdsf in zip(
            completions, target, guess_index, prev_score,
            guesses_so_far, feedbacks_so_far
        ):
            # Dataset stores these as JSON strings for HF Dataset compatibility
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
