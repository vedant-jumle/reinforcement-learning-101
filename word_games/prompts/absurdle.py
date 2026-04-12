"""Prompt builder for Absurdle."""

import re

SYSTEM_PROMPT = """You are playing Absurdle. This is an adversarial version of Wordle.
Rules:
- Each guess must be a valid 5-letter word from the allowed list
- After each guess you get feedback:
  G = correct letter, correct position
  Y = correct letter, wrong position
  X = letter not in the word
- The adversary picks the feedback that keeps the MOST words still possible
- You win when the adversary is forced to give all-green feedback (G G G G G)
- There is no guess limit, but fewer guesses is better

OUTPUT FORMAT:
<think>
[reason about which letters are confirmed/eliminated, how to shrink the candidate set fastest]
</think>
<guess>yourword</guess>"""


def build_system_prompt():
    return SYSTEM_PROMPT


def build_user_message(state):
    """Build the user message from current game state."""
    lines = []
    for i, (guess, feedback) in enumerate(zip(state['guesses'], state['feedbacks'])):
        fb_str = ' '.join(feedback)
        lines.append(f"Guess {i + 1}: {guess} → {fb_str}")

    candidates = state.get('candidates_remaining', '?')
    lines.append(f"\nCandidates remaining: {candidates}")
    lines.append("\nWhat is your next guess?")
    return '\n'.join(lines)


def build_conversation(state):
    """Build full conversation list for model input."""
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_message(state)},
    ]


def parse_guess(text):
    """Extract guess word from <guess>word</guess> tag. Returns None if not found."""
    match = re.search(r'<guess>\s*([a-zA-Z]{5})\s*</guess>', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None
