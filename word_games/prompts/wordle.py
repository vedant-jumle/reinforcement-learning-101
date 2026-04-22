"""Prompt builder for Wordle."""

import re

SYSTEM_PROMPT = """You are playing Wordle. The target is a secret 5-letter word.

Rules:
- Each guess must be a valid 5-letter word
- After each guess you get feedback:
  G = correct letter, correct position
  Y = correct letter, wrong position
  X = letter not in the word
- You have {max_guesses} guesses to find the word
- NEVER repeat a word you have already guessed

YOU MUST RESPOND IN EXACTLY THIS FORMAT — NO EXCEPTIONS:
<think>
[reason about which letters are confirmed, excluded, or misplaced, then pick a word that uses this information]
</think>
<guess>yourword</guess>

Example: if you guessed "crane" and got "X Y X G X", you know:
- C is not in the word
- R is in the word but not position 2
- A is not in the word
- N is in position 4
- E is not in the word
So your think block should reason about these constraints and pick a new word that fits.

The <guess> tag is REQUIRED. It must contain exactly one 5-letter word."""


def build_system_prompt(max_guesses=6):
    return SYSTEM_PROMPT.format(max_guesses=max_guesses)


def build_user_message(state):
    """Build the user message from current game state."""
    lines = []
    for i, (guess, feedback) in enumerate(zip(state['guesses'], state['feedbacks'])):
        fb_str = ' '.join(feedback)
        lines.append(f"Guess {i + 1}: {guess} → {fb_str}")
    lines.append("\nWhat is your next guess? Respond with <think>...</think><guess>word</guess>")
    return '\n'.join(lines)


def build_conversation(state, max_guesses=6):
    """Build full conversation list for model input."""
    return [
        {"role": "system", "content": build_system_prompt(max_guesses)},
        {"role": "user", "content": build_user_message(state)},
    ]


def parse_guess(text):
    """Extract guess word from <guess>word</guess> tag. Returns None if not found."""
    match = re.search(r'<guess>\s*([a-zA-Z]{5})\s*</guess>', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None
