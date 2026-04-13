"""Reward functions for Wordle GRPO training.

Reward hierarchy (single-guess-per-example setup):
  1. Format penalty   (-2.0) if no valid <guess> tag parsed
  2. Invalid word     (-1.0) if parsed word not in word list
  3. Step reward      = improvement * discount^guess_index  (positive)
                      = improvement                         (negative, no discount)
  4. Terminal bonus   = (7 - (guess_index+1)) if all-green feedback (win)

Step score = letter_score + candidate_bonus_weight * log2(candidates_before / candidates_after)
Improvement = step_score - prev_score
"""

import math

from .utils import score_guess, filter_candidates
from .prompts.wordle import parse_guess


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

    # Penalise repeating a word already guessed in this game
    if guess in guesses_so_far:
        return -1.5

    # 3. Score the guess
    feedback = score_guess(guess, target)
    letter_score = compute_letter_score(feedback)

    candidate_bonus = compute_candidate_bonus(
        word_list, guesses_so_far, feedbacks_so_far, guess, feedback
    )
    step_score = letter_score + candidate_bonus_weight * candidate_bonus

    # 4. Improvement over previous step
    improvement = step_score - prev_score

    if improvement > 0:
        step_reward = improvement * (discount_base ** guess_index)
    elif improvement < 0:
        step_reward = improvement   # full penalty, no discount
    else:
        step_reward = 0.0

    # 5. Terminal bonus: win detected when feedback is all-green
    terminal = 0.0
    if feedback == ['G', 'G', 'G', 'G', 'G']:
        terminal = float(7 - (guess_index + 1))   # win in 1 → 6, win in 6 → 1

    return step_reward + terminal


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
