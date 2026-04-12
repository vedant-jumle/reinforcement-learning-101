"""Standalone eval script for word game environments.

Loads a model via Unsloth FastLanguageModel and plays full multi-turn games,
reporting win rate, efficiency, and invalid guess metrics.

Usage:
    python -m word_games.eval --model unsloth/Qwen2.5-1.5B-Instruct --game wordle --n-games 200
    python -m word_games.eval --model ./my_lora_checkpoint --game quordle --n-games 100 --output results.json
"""

import argparse
import json
import math
import random
import sys

from .wordle import WordleGame
from .quordle import QuordleGame
from .absurdle import AbsurdleGame
from .utils import load_word_list, DEFAULT_WORD_LIST
from .prompts import get_prompt_builder


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name, max_seq_length=2048, load_in_4bit=True):
    """Load model + tokenizer via Unsloth FastLanguageModel.

    Calls FastLanguageModel.for_inference() to enable 2x faster generation
    and disable training hooks. LoRA adapters are merged automatically if
    present in the checkpoint.

    Args:
        model_name: HuggingFace repo name or local path.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Use 4-bit quantisation (recommended for 8GB GPU).

    Returns:
        (model, tokenizer) tuple ready for model.generate().
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # auto-detect: bfloat16 on Ampere+, float16 otherwise
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)   # 2x faster; disables gradient tracking
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def generate_guess(model, tokenizer, conversation, max_new_tokens, prompt_builder):
    """Run one generation step and parse the <guess> tag.

    Args:
        model: Loaded Unsloth model (for_inference already called).
        tokenizer: Corresponding tokenizer.
        conversation: List of {"role": ..., "content": ...} dicts.
        max_new_tokens: Max tokens to generate.
        prompt_builder: Prompt module with parse_guess().

    Returns:
        (guess, response_text) where guess is None if parsing failed.
    """
    inputs = tokenizer.apply_chat_template(
        conversation,
        return_tensors='pt',
        add_generation_prompt=True,
    ).to(model.device)

    output_ids = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,     # Unsloth-optimised KV cache
    )
    response = tokenizer.decode(
        output_ids[0][inputs.shape[1]:],
        skip_special_tokens=True,
    )
    return prompt_builder.parse_guess(response), response


# ── Per-game eval loops ───────────────────────────────────────────────────────

def eval_wordle(model, tokenizer, word_list, n_games, max_new_tokens, max_guesses=6):
    """Play n_games of Wordle, sampling targets from word_list."""
    pb = get_prompt_builder('wordle')
    game = WordleGame()
    results = []

    for i in range(n_games):
        target = random.choice(word_list)
        state = game.reset(target=target)
        invalid_count = 0
        info_gains = []

        while not state['done']:
            conversation = pb.build_conversation(state, max_guesses=max_guesses)
            guess, _ = generate_guess(model, tokenizer, conversation, max_new_tokens, pb)

            if guess is None:
                invalid_count += 1
                guess = random.choice(game.word_list)   # fallback: advance game

            from .utils import filter_candidates, load_word_list as _load
            # Information gain: log2(candidates_before / candidates_after)
            if state['guesses']:
                candidates_before = len(filter_candidates(
                    game.word_list, state['guesses'][-1], state['feedbacks'][-1]
                ))
            else:
                candidates_before = len(game.word_list)

            state, reward, done, info = game.step(guess)

            if 'error' in info:
                invalid_count += 1
            elif state['guesses']:
                candidates_after = len(filter_candidates(
                    game.word_list, state['guesses'][-1], state['feedbacks'][-1]
                ))
                if candidates_before > 0 and candidates_after > 0:
                    info_gains.append(math.log2(candidates_before / max(candidates_after, 1)))

        results.append({
            'won': state['won'],
            'guess_count': state['guess_count'],
            'invalid_count': invalid_count,
            'avg_info_gain': sum(info_gains) / len(info_gains) if info_gains else 0.0,
            'target': target,
        })

        if (i + 1) % 10 == 0:
            wins = sum(r['won'] for r in results)
            print(f"  [{i+1}/{n_games}] win_rate={wins/(i+1):.2%}")

    return results


def eval_quordle(model, tokenizer, word_list, n_games, max_new_tokens, max_guesses=9):
    """Play n_games of Quordle, sampling targets from word_list."""
    pb = get_prompt_builder('quordle')
    game = QuordleGame()
    results = []

    for i in range(n_games):
        targets = random.sample(word_list, 4)
        state = game.reset(targets=targets)
        invalid_count = 0

        while not state['done']:
            conversation = pb.build_conversation(state, max_guesses=max_guesses)
            guess, _ = generate_guess(model, tokenizer, conversation, max_new_tokens, pb)

            if guess is None:
                invalid_count += 1
                guess = random.choice(game.word_list)

            state, reward, done, info = game.step(guess)
            if 'error' in info:
                invalid_count += 1

        results.append({
            'won': state['won'],
            'guess_count': state['guess_count'],
            'boards_solved': sum(state['boards_solved']),
            'invalid_count': invalid_count,
            'targets': targets,
        })

        if (i + 1) % 10 == 0:
            wins = sum(r['won'] for r in results)
            print(f"  [{i+1}/{n_games}] win_rate={wins/(i+1):.2%}")

    return results


def eval_absurdle(model, tokenizer, word_list, n_games, max_new_tokens, max_steps=50):
    """Play n_games of Absurdle. Games are capped at max_steps to prevent infinite loops."""
    pb = get_prompt_builder('absurdle')
    game = AbsurdleGame()
    results = []

    for i in range(n_games):
        state = game.reset()
        invalid_count = 0
        step = 0

        while not state['done'] and step < max_steps:
            conversation = pb.build_conversation(state)
            guess, _ = generate_guess(model, tokenizer, conversation, max_new_tokens, pb)

            if guess is None:
                invalid_count += 1
                guess = random.choice(game.word_list)

            state, reward, done, info = game.step(guess)
            if 'error' in info:
                invalid_count += 1
            step += 1

        results.append({
            'won': state['won'],
            'guess_count': state['guess_count'],
            'candidates_remaining': state['candidates_remaining'],
            'invalid_count': invalid_count,
            'hit_step_limit': step >= max_steps and not state['done'],
        })

        if (i + 1) % 10 == 0:
            wins = sum(r['won'] for r in results)
            print(f"  [{i+1}/{n_games}] win_rate={wins/(i+1):.2%}")

    return results


# ── Metrics aggregation ───────────────────────────────────────────────────────

def summarise(results, game_type):
    """Compute aggregate metrics from per-game result dicts."""
    n = len(results)
    wins = [r for r in results if r['won']]
    total_guesses = sum(r['guess_count'] for r in results)
    total_invalids = sum(r['invalid_count'] for r in results)

    summary = {
        'game_type': game_type,
        'n_games': n,
        'win_rate': len(wins) / n if n else 0.0,
        'avg_guesses': total_guesses / n if n else 0.0,
        'avg_guesses_to_win': (
            sum(r['guess_count'] for r in wins) / len(wins) if wins else None
        ),
        'avg_invalid_rate': total_invalids / total_guesses if total_guesses else 0.0,
    }

    if game_type == 'wordle':
        gains = [r['avg_info_gain'] for r in results if r['avg_info_gain'] > 0]
        summary['avg_info_gain_per_guess'] = sum(gains) / len(gains) if gains else 0.0

    if game_type == 'quordle':
        summary['avg_boards_solved'] = sum(r['boards_solved'] for r in results) / n

    if game_type == 'absurdle':
        summary['pct_hit_step_limit'] = sum(r['hit_step_limit'] for r in results) / n

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM on word game environments.")
    parser.add_argument('--model', required=True, help='HuggingFace repo or local path')
    parser.add_argument('--game', choices=['wordle', 'quordle', 'absurdle'], default='wordle')
    parser.add_argument('--n-games', type=int, default=200)
    parser.add_argument('--max-new-tokens', type=int, default=300)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantisation')
    parser.add_argument('--word-list', default=None, help='Path to word list file')
    parser.add_argument('--eval-split', type=int, default=500,
                        help='Use last N words as held-out eval set (0 = use full list)')
    parser.add_argument('--output', default=None, help='Save JSON results to this path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load word list and optionally take a held-out eval split
    word_list = load_word_list(args.word_list or DEFAULT_WORD_LIST)
    if args.eval_split > 0:
        eval_words = word_list[-args.eval_split:]
        print(f"Using last {len(eval_words)} words as eval split")
    else:
        eval_words = word_list
        print(f"Using full word list ({len(eval_words)} words) for eval")

    # Load model
    model, tokenizer = load_model(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )

    # Run eval
    print(f"\nRunning {args.n_games} games of {args.game}...")
    if args.game == 'wordle':
        results = eval_wordle(model, tokenizer, eval_words, args.n_games, args.max_new_tokens)
    elif args.game == 'quordle':
        results = eval_quordle(model, tokenizer, eval_words, args.n_games, args.max_new_tokens)
    elif args.game == 'absurdle':
        results = eval_absurdle(model, tokenizer, eval_words, args.n_games, args.max_new_tokens)

    # Summarise
    summary = summarise(results, args.game)
    print("\n── Eval Results ──────────────────────────────")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    if args.output:
        out = {'summary': summary, 'per_game': results}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
