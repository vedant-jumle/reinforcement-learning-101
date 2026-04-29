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
- Your guess MUST NOT contain any letter marked X (absent)
- Your guess MUST place any letter marked G in its exact position
- Your guess MUST include any letter marked Y somewhere (but not in the same position)

YOU MUST RESPOND IN EXACTLY THIS FORMAT:
<think>
Position 1: [confirmed letter, or options/unknown]
Position 2: [confirmed letter, or options/unknown]
Position 3: [confirmed letter, or options/unknown]
Position 4: [confirmed letter, or options/unknown]
Position 5: [confirmed letter, or options/unknown]
Absent letters: [letters confirmed not in word]
Present but unplaced: [letters that must appear somewhere]
Best guess: [word and why it uses the above constraints]
</think>
<guess>word</guess>

The <guess> tag is REQUIRED. It must contain exactly one 5-letter word in lowercase."""


def build_system_prompt(max_guesses=6):
    return SYSTEM_PROMPT.format(max_guesses=max_guesses)


def build_user_message(state):
    """Build the user message from current game state."""
    lines = []
    for i, (guess, feedback) in enumerate(zip(state['guesses'], state['feedbacks'])):
        fb_str = ' '.join(feedback)
        lines.append(f"Guess {i + 1}: {guess} → {fb_str}")
    if state['guesses']:
        lines.append("\nFill in the template and guess a word consistent with all constraints above.")
    else:
        lines.append("\nNo guesses yet. Fill in the template and make your first guess.")
    lines.append("Respond with <think>...</think><guess>word</guess>")
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
