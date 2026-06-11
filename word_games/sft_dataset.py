"""SFT dataset generation using the entropy solver.

Plays complete games with the entropy-maximizing solver and records each
decision point as a (prompt, completion) training example. The completion
includes a structured <think> block that mirrors the system prompt template,
teaching the model the constraint-tracking format before GRPO fine-tuning.

Usage:
    from word_games.sft_dataset import generate_sft_dataset
    from word_games.utils import load_word_list

    word_list = load_word_list()
    ds = generate_sft_dataset(word_list, n_games=3000, tokenizer=tokenizer)
    # Columns: prompt (str), completion (str)
"""

import random
from .utils import load_word_list, DEFAULT_WORD_LIST
from .solver import play_solver_game
from .prompts.wordle import build_conversation, build_system_prompt


def _build_think_block(guesses: list[str], feedbacks: list[list[str]], next_guess: str) -> str:
    """Build a structured <think> block matching the system prompt template.

    Derives constraint state from prior feedback and fills in the 5-position
    template with confirmed letters or 'unknown', then justifies next_guess.

    Args:
        guesses: Prior guesses (not including next_guess).
        feedbacks: Feedback for each prior guess.
        next_guess: The solver's chosen next guess.

    Returns:
        Full <think>...</think> string.
    """
    # Derive constraint state from feedback history
    green = {}        # position → letter
    yellow_pos = {}   # letter → set of positions it cannot be in
    yellow = set()    # letters confirmed present
    absent = set()    # letters confirmed absent

    for guess, feedback in zip(guesses, feedbacks):
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == 'G':
                green[i] = letter
                yellow.discard(letter)  # green supersedes yellow
            elif fb == 'Y':
                yellow.add(letter)
                yellow_pos.setdefault(letter, set()).add(i)
            else:  # X
                absent.add(letter)

    # Letters confirmed present (green or yellow) cannot be truly absent
    confirmed = set(green.values()) | yellow
    absent -= confirmed

    # Build position lines
    position_lines = []
    for i in range(5):
        if i in green:
            position_lines.append(f"Position {i+1}: {green[i].upper()} (confirmed green)")
        else:
            excluded = {letter for letter, positions in yellow_pos.items() if i in positions}
            if excluded:
                position_lines.append(
                    f"Position {i+1}: unknown (not {', '.join(sorted(excluded)).upper()})"
                )
            else:
                position_lines.append(f"Position {i+1}: unknown")

    absent_str = ', '.join(sorted(absent)).upper() if absent else 'none'
    present_str = ', '.join(sorted(yellow)).upper() if yellow else 'none'

    think_lines = position_lines + [
        f"Absent letters: {absent_str}",
        f"Present but unplaced: {present_str}",
        f"Best guess: {next_guess.upper()} — fits all known constraints",
    ]

    return "<think>\n" + "\n".join(think_lines) + "\n</think>"


def _build_completion(
    guesses: list[str],
    feedbacks: list[list[str]],
    next_guess: str,
) -> str:
    """Build the full assistant completion for one solver step."""
    think = _build_think_block(guesses, feedbacks, next_guess)
    return f"{think}\n<guess>{next_guess}</guess>"


def generate_sft_dataset(
    word_list: list[str],
    n_games: int,
    tokenizer=None,
    max_guesses: int = 6,
    seed: int = 42,
):
    """Generate SFT training examples from entropy-solver game traces.

    For each game, plays all turns with the solver and records every
    decision point as a separate training example. The prompt is the
    chat-template-formatted game state; the completion is a structured
    <think> block plus <guess> tag.

    Args:
        word_list: Valid 5-letter words (targets and guesses).
        n_games: Number of games to simulate.
        tokenizer: HuggingFace tokenizer for apply_chat_template. If None,
                   stores raw conversation dicts instead.
        max_guesses: Max guesses per game.
        seed: Random seed.

    Returns:
        datasets.Dataset with columns: prompt (str), completion (str).
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("pip install datasets")

    from tqdm.auto import tqdm

    rng = random.Random(seed)
    rows = []

    for _ in tqdm(range(n_games), desc="Generating SFT games"):
        target = rng.choice(word_list)
        guesses, feedbacks, solved = play_solver_game(target, word_list, max_guesses)

        # Record each step as a training example
        for step in range(len(guesses)):
            prior_guesses = guesses[:step]
            prior_feedbacks = feedbacks[:step]
            next_guess = guesses[step]

            state = {
                'guesses': prior_guesses,
                'feedbacks': [list(fb) for fb in prior_feedbacks],
                'guess_count': step,
                'done': False,
                'won': False,
            }

            conversation = build_conversation(state, max_guesses=max_guesses)

            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in conversation]
                prompt = '\n\n'.join(parts)

            completion = _build_completion(prior_guesses, prior_feedbacks, next_guess)

            rows.append({'prompt': prompt, 'completion': completion})

    rng.shuffle(rows)
    return Dataset.from_list(rows)
