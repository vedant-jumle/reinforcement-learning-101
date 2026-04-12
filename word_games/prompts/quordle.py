"""Prompt builder for Quordle."""

import re

SYSTEM_PROMPT = """You are playing Quordle. You must solve 4 Wordle boards simultaneously.
Rules:
- Each guess applies to all 4 boards at once
- After each guess you get feedback for each board:
  G = correct letter, correct position
  Y = correct letter, wrong position
  X = letter not in the word
- You have {max_guesses} guesses to solve all 4 boards
- Each guess must be a valid 5-letter word from the allowed list

OUTPUT FORMAT:
<think>
[reason about which letters are confirmed/eliminated across all boards, what words remain possible]
</think>
<guess>yourword</guess>"""


def build_system_prompt(max_guesses=9):
    return SYSTEM_PROMPT.format(max_guesses=max_guesses)


def build_user_message(state):
    """Build the user message from current game state.

    feedbacks is nested: feedbacks[guess_idx][board_idx] = list of 5 chars.
    boards_solved marks which boards are already solved.
    """
    lines = []
    boards_solved = state.get('boards_solved', [False, False, False, False])

    for i, (guess, board_feedbacks) in enumerate(zip(state['guesses'], state['feedbacks'])):
        lines.append(f"Guess {i + 1}: {guess}")
        for b, fb in enumerate(board_feedbacks):
            solved_marker = " (solved)" if boards_solved[b] and all(c == 'G' for c in fb) else ""
            fb_str = ' '.join(fb)
            lines.append(f"  Board {b + 1}: {fb_str}{solved_marker}")

    # Show which boards are still unsolved
    unsolved = [b + 1 for b, solved in enumerate(boards_solved) if not solved]
    if unsolved:
        lines.append(f"\nBoards still unsolved: {unsolved}")

    lines.append("\nWhat is your next guess?")
    return '\n'.join(lines)


def build_conversation(state, max_guesses=9):
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
