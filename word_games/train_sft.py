"""SFT fine-tuning for Wordle using TRL SFTTrainer + Unsloth FastLanguageModel.

Trains the model on entropy-solver game traces to give it a strong warm start
before GRPO. After SFT the model reliably produces valid <think>...</think>
<guess>word</guess> completions that respect constraints — so GRPO only has to
learn strategy, not format.

Usage:
    # V100 32GB
    python -m word_games.train_sft --no-4bit --output-dir ./checkpoints/wordle-sft

    # 8GB GPU
    python -m word_games.train_sft --output-dir ./checkpoints/wordle-sft
"""

import argparse
import sys
from dataclasses import dataclass


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SFTConfig:
    # Model
    model_name: str = "unsloth/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # SFT training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 2
    learning_rate: float = 2e-4        # higher than GRPO — SFT can be more aggressive
    warmup_steps: int = 20
    output_dir: str = "./checkpoints/wordle-sft"
    logging_steps: int = 10
    save_steps: int = 200
    max_completion_length: int = 300   # SFT completions can be longer (full think block)

    # Dataset
    n_games: int = 3000
    word_list_path: str = None


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: SFTConfig):
    from trl import SFTTrainer, SFTConfig as TRLSFTConfig
    from .train_grpo import load_model_for_training
    from .utils import load_word_list
    from .sft_dataset import generate_sft_dataset

    # 1. Load model (reuse the same loader as GRPO)
    model, tokenizer = load_model_for_training(config)

    # 2. Build SFT dataset from solver traces
    word_list = load_word_list(config.word_list_path)
    print(f"Generating SFT dataset from {config.n_games} solver games...")
    dataset = generate_sft_dataset(
        word_list=word_list,
        n_games=config.n_games,
        tokenizer=tokenizer,
    )
    print(f"SFT dataset: {len(dataset)} examples")

    # 3. SFTTrainer expects a 'text' column OR prompt+completion columns.
    # We combine prompt + completion into a single 'text' field.
    def combine(example):
        return {'text': example['prompt'] + example['completion']}

    dataset = dataset.map(combine, remove_columns=['prompt', 'completion'])

    # 4. Training args
    training_args = TRLSFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        bf16=False,
        fp16=False,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to="none",
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 6. Train
    print("Starting SFT training...")
    trainer.train()

    # 7. Save
    print(f"Saving SFT model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("SFT done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SFT warm-start for Wordle GRPO.")
    parser.add_argument('--model', default=SFTConfig.model_name)
    parser.add_argument('--max-seq-length', type=int, default=SFTConfig.max_seq_length)
    parser.add_argument('--no-4bit', action='store_true')
    parser.add_argument('--lora-r', type=int, default=SFTConfig.lora_r)
    parser.add_argument('--lora-alpha', type=int, default=SFTConfig.lora_alpha)
    parser.add_argument('--per-device-batch', type=int, default=SFTConfig.per_device_train_batch_size)
    parser.add_argument('--grad-accum', type=int, default=SFTConfig.gradient_accumulation_steps)
    parser.add_argument('--epochs', type=int, default=SFTConfig.num_train_epochs)
    parser.add_argument('--lr', type=float, default=SFTConfig.learning_rate)
    parser.add_argument('--n-games', type=int, default=SFTConfig.n_games)
    parser.add_argument('--word-list', default=None)
    parser.add_argument('--output-dir', default=SFTConfig.output_dir)
    return parser.parse_args()


def main():
    args = parse_args()
    config = SFTConfig(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        n_games=args.n_games,
        word_list_path=args.word_list,
        output_dir=args.output_dir,
    )
    train(config)


if __name__ == '__main__':
    main()
