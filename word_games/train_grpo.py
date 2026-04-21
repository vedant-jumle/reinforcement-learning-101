"""GRPO fine-tuning for Wordle using TRL GRPOTrainer + Unsloth FastLanguageModel.

Usage:
    # 8GB GPU (default)
    python -m word_games.train_grpo

    # V100 32GB — disable 4-bit, increase batch size
    python -m word_games.train_grpo --no-4bit --num-generations 16 --per-device-batch 2

    # Custom model / output dir
    python -m word_games.train_grpo --model unsloth/Qwen2.5-1.5B-Instruct --output-dir ./my-run

Design:
    - Single-guess-per-example: each training row is one (game_state → guess) decision.
    - Dataset generated offline by playing random games (see word_games/dataset.py).
    - Reward = improvement-tracked letter score + candidate bonus + terminal win bonus.
    - GRPOTrainer samples num_generations completions per prompt, ranks by reward.
"""

import argparse
import sys
from dataclasses import dataclass, field


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # True for 8GB GPU; False for V100 32GB
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0          # Unsloth recommends 0 for speed
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO
    num_generations: int = 8           # K completions per prompt; reduce to 4 for 8GB if OOM
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # must satisfy: per_device_batch * grad_accum % num_generations == 0
    max_completion_length: int = 200   # max tokens for one guess + think block
    num_train_epochs: int = 1
    learning_rate: float = 5e-6
    warmup_steps: int = 20
    beta: float = 0.04                 # KL penalty — keeps model from collapsing away from reference
    temperature: float = 1.0           # generation temperature — keep at 1.0 for diversity
    output_dir: str = "./checkpoints/wordle-grpo"
    logging_steps: int = 10
    save_steps: int = 100

    # Dataset
    n_games: int = 5000                # random games to simulate for training data
    word_list_path: str = None         # None = use default data/wordle_list.txt

    # Reward
    discount_base: float = 0.9
    candidate_bonus_weight: float = 0.3


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_for_training(config):
    """Load model + tokenizer via Unsloth and attach LoRA adapters.

    Returns:
        (model, tokenizer) tuple with LoRA applied, ready for GRPOTrainer.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    dtype = None   # auto-detect: bfloat16 on Ampere+, float16 on V100
    if not config.load_in_4bit:
        import torch
        dtype = torch.float16   # V100 doesn't support bfloat16

    print(f"Loading model: {config.model_name} (4bit={config.load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=dtype,
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves memory
        random_state=42,
    )
    print(f"LoRA applied (r={config.lora_r}, alpha={config.lora_alpha})")
    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def train(config):
    """Main training loop."""
    from trl import GRPOTrainer, GRPOConfig

    from .utils import load_word_list
    from .dataset import generate_wordle_dataset, get_train_eval_split
    from .reward import build_reward_fn

    # 1. Load model
    model, tokenizer = load_model_for_training(config)

    # Clear max_length from the baked-in generation_config so it doesn't
    # conflict with max_completion_length set by GRPOConfig.
    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = None

    # 2. Build dataset
    train_words, _ = get_train_eval_split(config.word_list_path)
    print(f"Generating dataset from {config.n_games} random games "
          f"({len(train_words)} training words)...")
    train_dataset = generate_wordle_dataset(
        word_list=train_words,
        n_games=config.n_games,
        tokenizer=tokenizer,
    )
    print(f"Dataset size: {len(train_dataset)} examples")

    # 3. Build reward function (closure captures word_list + hyperparams)
    full_word_list = load_word_list(config.word_list_path)
    reward_fn = build_reward_fn(
        word_list=full_word_list,
        discount_base=config.discount_base,
        candidate_bonus_weight=config.candidate_bonus_weight,
    )

    # 4. GRPO training config
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_completion_length=config.max_completion_length,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        beta=config.beta,
        temperature=config.temperature,
        bf16=False,
        fp16=False,  # let unsloth handle mixed precision internally
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to="none",
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # 6. Train
    print(f"\nStarting GRPO training...")
    trainer.train()

    # 7. Save
    print(f"\nSaving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning for Wordle with Unsloth + TRL."
    )
    # Model
    parser.add_argument('--model', default=TrainConfig.model_name)
    parser.add_argument('--max-seq-length', type=int, default=TrainConfig.max_seq_length)
    parser.add_argument('--no-4bit', action='store_true',
                        help='Disable 4-bit quantisation (use for V100 32GB)')
    parser.add_argument('--lora-r', type=int, default=TrainConfig.lora_r)
    parser.add_argument('--lora-alpha', type=int, default=TrainConfig.lora_alpha)
    # GRPO
    parser.add_argument('--num-generations', type=int, default=TrainConfig.num_generations)
    parser.add_argument('--per-device-batch', type=int,
                        default=TrainConfig.per_device_train_batch_size)
    parser.add_argument('--grad-accum', type=int,
                        default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument('--max-completion-length', type=int, default=TrainConfig.max_completion_length)
    parser.add_argument('--epochs', type=int, default=TrainConfig.num_train_epochs)
    parser.add_argument('--lr', type=float, default=TrainConfig.learning_rate)
    parser.add_argument('--output-dir', default=TrainConfig.output_dir)
    # Dataset
    parser.add_argument('--n-games', type=int, default=TrainConfig.n_games)
    parser.add_argument('--word-list', default=None)
    # Reward
    parser.add_argument('--discount-base', type=float, default=TrainConfig.discount_base)
    parser.add_argument('--candidate-bonus-weight', type=float,
                        default=TrainConfig.candidate_bonus_weight)

    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        n_games=args.n_games,
        word_list_path=args.word_list,
        discount_base=args.discount_base,
        candidate_bonus_weight=args.candidate_bonus_weight,
    )
    train(config)


if __name__ == '__main__':
    main()
