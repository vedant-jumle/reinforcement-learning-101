"""Prompt builders for word game environments.

Each game module exposes:
    build_system_prompt(**kwargs) -> str
    build_user_message(state) -> str
    build_conversation(state, **kwargs) -> list[dict]
    parse_guess(text) -> str | None
"""

from . import wordle, quordle, absurdle

_BUILDERS = {
    'wordle': wordle,
    'quordle': quordle,
    'absurdle': absurdle,
}


def get_prompt_builder(game_type):
    """Return the prompt module for the given game type.

    Args:
        game_type: One of 'wordle', 'quordle', 'absurdle'.

    Returns:
        Module with build_conversation(state) and parse_guess(text).
    """
    if game_type not in _BUILDERS:
        raise ValueError(f"game_type must be one of {list(_BUILDERS)}, got {game_type!r}")
    return _BUILDERS[game_type]


def parse_guess(text):
    """Shared guess parser — same <guess>word</guess> format for all games."""
    return wordle.parse_guess(text)


__all__ = ['get_prompt_builder', 'parse_guess', 'wordle', 'quordle', 'absurdle']
