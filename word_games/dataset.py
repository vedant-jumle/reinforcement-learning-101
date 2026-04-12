"""Dataset generation for Wordle GRPO training.

Generates a HuggingFace Dataset of single-guess training examples by playing
random games offline. Each row represents one (game_state → next_guess) decision
point, giving a diverse mix of board states from empty to near-complete.

Each row contains:
  prompt          str   — chat-template-formatted prompt (ready for tokenizer)
  target          str   — target word (passed to reward fn)
  guess_index     int   — 0-based index of the guess being trained on
  prev_score      float — letter score of the previous guess (0 if first)
  guesses_so_far  str   — JSON list of prior guesses
  feedbacks_so_far str  — JSON list of prior feedbacks (list[list[str]])

Usage:
    from word_games.dataset import generate_wordle_dataset
    from word_games.utils import load_word_list

    word_list = load_word_list()
    train_words = word_list[:-500]   # hold out last 500 for eval
    ds = generate_wordle_dataset(train_words, n_games=5000)
"""

import json
import random

from .utils import load_word_list, score_guess, DEFAULT_WORD_LIST
from .reward import compute_letter_score
from .prompts.wordle import build_conversation


def _build_prompt_string(state, tokenizer, max_guesses=6):
    """Apply chat template to a game state and return the prompt string.

    If tokenizer is None, returns a plain text prompt (useful for testing
    dataset generation without loading a model).

    Args:
        state: WordleGame state dict.
        tokenizer: HuggingFace tokenizer, or None for plain text.
        max_guesses: Max guesses for the game (used in system prompt).

    Returns:
        String prompt ready to be tokenized.
    """
    conversation = build_conversation(state, max_guesses=max_guesses)
    if tokenizer is None:
        # Fallback: concatenate role+content for testing
        parts = []
        for msg in conversation:
            parts.append(f"[{msg['role'].upper()}]\n{msg['content']}")
        return '\n\n'.join(parts)
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_wordle_dataset(
    word_list,
    n_games,
    tokenizer=None,
    max_guesses=6,
    seed=42,
):
    """Generate a dataset of single-guess Wordle training examples.

    For each game:
      1. Pick a random target word from word_list.
      2. Play 0–(max_guesses-1) random guesses to advance the board.
      3. Record the board state at each step as a training example.

    This ensures the dataset contains examples at all difficulty levels:
    empty boards (first guess) through near-complete boards (5th guess).

    Args:
        word_list: List of valid 5-letter words to use as targets and guesses.
        n_games: Number of random games to simulate.
        tokenizer: HuggingFace tokenizer for apply_chat_template, or None.
        max_guesses: Maximum guesses per game (default 6).
        seed: Random seed for reproducibility.

    Returns:
        datasets.Dataset with columns:
            prompt, target, guess_index, prev_score,
            guesses_so_far, feedbacks_so_far
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets")

    from tqdm.auto import tqdm

    rng = random.Random(seed)
    word_set = set(word_list)

    rows = []

    for _ in tqdm(range(n_games), desc="Generating games"):
        target = rng.choice(word_list)

        guesses = []
        feedbacks = []
        prev_score = 0.0

        # Number of random guesses to play before recording a training state.
        # Uniform over [0, max_guesses-1] so all board depths are represented.
        n_random = rng.randint(0, max_guesses - 1)

        for step in range(n_random):
            # Build current state and record it as a training example
            state = {
                'guesses': list(guesses),
                'feedbacks': [list(fb) for fb in feedbacks],
                'guess_count': len(guesses),
                'done': False,
                'won': False,
            }

            prompt = _build_prompt_string(state, tokenizer, max_guesses)

            rows.append({
                'prompt': prompt,
                'target': target,
                'guess_index': step,
                'prev_score': prev_score,
                'guesses_so_far': json.dumps(list(guesses)),
                'feedbacks_so_far': json.dumps([list(fb) for fb in feedbacks]),
            })

            # Advance game with a random guess (not model — offline generation)
            random_guess = rng.choice(word_list)
            feedback = score_guess(random_guess, target)
            prev_score = float(compute_letter_score(feedback))
            guesses.append(random_guess)
            feedbacks.append(feedback)

            # Stop early if we accidentally won
            if feedback == ['G', 'G', 'G', 'G', 'G']:
                break

        # Always add the state at position n_random (the actual training point)
        if len(guesses) < max_guesses:
            state = {
                'guesses': list(guesses),
                'feedbacks': [list(fb) for fb in feedbacks],
                'guess_count': len(guesses),
                'done': False,
                'won': False,
            }
            prompt = _build_prompt_string(state, tokenizer, max_guesses)
            rows.append({
                'prompt': prompt,
                'target': target,
                'guess_index': len(guesses),
                'prev_score': prev_score,
                'guesses_so_far': json.dumps(list(guesses)),
                'feedbacks_so_far': json.dumps([list(fb) for fb in feedbacks]),
            })

    rng.shuffle(rows)
    return Dataset.from_list(rows)


def get_train_eval_split(word_list_path=None):
    """Return (train_words, eval_words) split from the word list.

    Eval split = last 500 words (matches word_games/eval.py convention).

    Args:
        word_list_path: Path to word list file, or None for default.

    Returns:
        (train_words, eval_words) tuple of lists.
    """
    words = load_word_list(word_list_path or DEFAULT_WORD_LIST)
    return words[:-500], words[-500:]
