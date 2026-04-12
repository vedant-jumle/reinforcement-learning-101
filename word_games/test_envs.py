#!/usr/bin/env python3
"""Test script for word game environments.
Run from repo root: python -m word_games.test_envs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from word_games.utils import load_word_list, score_guess, filter_candidates
from word_games.wordle import WordleGame
from word_games.quordle import QuordleGame
from word_games.absurdle import AbsurdleGame
from word_games.env import WordGameEnv

WORD_LIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'wordle_list.txt')
PASS = "  OK"
FAIL = "  FAIL"


def test_utils():
    print("=" * 60)
    print("Test: utils.py")
    print("=" * 60)

    words = load_word_list(WORD_LIST_PATH)
    assert len(words) == 5757, f"Expected 5757 words, got {len(words)}"
    assert all(len(w) == 5 for w in words), "All words must be 5 letters"
    assert 'which' in words
    print(f"{PASS} load_word_list: {len(words)} words loaded")

    fb = score_guess("crane", "crane")
    assert fb == ['G', 'G', 'G', 'G', 'G'], f"Expected all G, got {fb}"
    print(f"{PASS} score_guess: exact match = {fb}")

    # crane vs snake: c->X, r->X, a->G(pos2), n->Y(n is in snake at pos1), e->G(pos4)
    fb = score_guess("crane", "snake")
    assert fb == ['X', 'X', 'G', 'Y', 'G'], f"Expected ['X','X','G','Y','G'], got {fb}"
    print(f"{PASS} score_guess: crane vs snake = {fb}")

    # speed vs steep: s->G, p->Y(p at pos4 in steep), e->G, e->G, d->X
    fb = score_guess("speed", "steep")
    assert fb == ['G', 'Y', 'G', 'G', 'X'], f"Expected ['G','Y','G','G','X'], got {fb}"
    print(f"{PASS} score_guess: duplicate letters (speed vs steep) = {fb}")

    cands = filter_candidates(words, "crane", ['X', 'X', 'G', 'Y', 'G'])
    for w in cands:
        assert w[2] == 'a', f"Candidate {w} missing a at pos2"
        assert w[4] == 'e', f"Candidate {w} missing e at pos4"
        assert 'n' in w, f"Candidate {w} missing n"
        assert w[3] != 'n', f"Candidate {w} has n at wrong pos"
        assert 'c' not in w, f"Candidate {w} contains excluded c"
        assert 'r' not in w, f"Candidate {w} contains excluded r"
    print(f"{PASS} filter_candidates: {len(cands)} candidates remain after crane/snake feedback")


def test_wordle():
    print("\n" + "=" * 60)
    print("Test: WordleGame")
    print("=" * 60)

    game = WordleGame(WORD_LIST_PATH)

    state = game.reset(target='crane')
    assert state['guess_count'] == 0 and not state['done'] and not state['won']
    assert state['guesses'] == []
    print(f"{PASS} reset with fixed target")

    state, reward, done, info = game.step('zzzzz')
    assert 'error' in info
    assert state['guess_count'] == 0
    print(f"{PASS} invalid word rejected, guess_count unchanged")

    state, reward, done, info = game.step('crane')
    assert state['won'] and done and reward == 1.0 and state['guess_count'] == 1
    assert state['feedbacks'][0] == ['G', 'G', 'G', 'G', 'G']
    print(f"{PASS} correct guess: won=True, reward=1.0, feedback=all-G")

    state = game.reset(target='crane')
    wrong_words = ['which', 'there', 'their', 'about', 'would', 'these']
    for w in wrong_words:
        state, reward, done, info = game.step(w)
    assert done and not state['won'] and reward == -1.0
    print(f"{PASS} loss after 6 wrong guesses: reward=-1.0")

    state = game.reset(target='crane')
    state, reward, done, info = game.step('which')
    assert reward == 0.0 and not done
    print(f"{PASS} intermediate step: reward=0.0, done=False")

    for _ in range(10):
        state = game.reset()
        state, reward, done, info = game.step(game.target)
        assert done and state['won']
    print(f"{PASS} random target is always in valid word list")


def test_quordle():
    print("\n" + "=" * 60)
    print("Test: QuordleGame")
    print("=" * 60)

    game = QuordleGame(WORD_LIST_PATH)
    targets = ['crane', 'which', 'there', 'their']

    state = game.reset(targets=targets)
    assert state['guess_count'] == 0
    assert len(state['feedbacks']) == 0
    assert state['boards_solved'] == [False, False, False, False]
    print(f"{PASS} reset state structure correct")

    state, reward, done, info = game.step('crane')
    assert state['boards_solved'][0] == True
    assert abs(reward - 0.25) < 1e-9
    assert not done
    assert len(state['feedbacks']) == 1 and len(state['feedbacks'][0]) == 4
    print(f"{PASS} solving board 0: boards_solved[0]=True, reward=0.25")

    state = game.reset(targets=targets)
    for word in targets:
        state, reward, done, info = game.step(word)
    assert state['won'] and done
    print(f"{PASS} solving all 4 boards: won=True")

    state = game.reset(targets=targets)
    nine_words = ['which', 'there', 'their', 'about', 'would', 'these', 'other', 'words', 'could']
    for w in nine_words:
        state, reward, done, info = game.step(w)
    assert done
    print(f"{PASS} 9-guess limit enforced: done=True after 9 guesses")


def test_absurdle():
    print("\n" + "=" * 60)
    print("Test: AbsurdleGame")
    print("=" * 60)

    game = AbsurdleGame(WORD_LIST_PATH)
    words = load_word_list(WORD_LIST_PATH)

    state = game.reset()
    assert state['candidates_remaining'] == len(words)
    assert state['guess_count'] == 0
    print(f"{PASS} reset: candidates_remaining={state['candidates_remaining']}")

    state, reward, done, info = game.step('crane')
    assert state['candidates_remaining'] > 10, \
        f"Adversary gave up too easily: {state['candidates_remaining']} candidates left"
    assert not done
    print(f"{PASS} adversarial: {state['candidates_remaining']} candidates remain after first guess")

    # Win: when candidates=1 and agent guesses that word, adversary is forced to give all-G
    game2 = AbsurdleGame(WORD_LIST_PATH)
    game2.reset()
    won = False
    for _ in range(25):
        st = game2.get_state()
        if st['candidates_remaining'] == 1:
            target = list(game2.candidates)[0]
            state, reward, done, info = game2.step(target)
            assert done and state['won'] and reward < 0
            print(f"{PASS} win with 1 candidate: reward={reward}, guess_count={state['guess_count']}")
            won = True
            break
        game2.step(game2.word_list[st['guess_count'] % len(game2.word_list)])
    if not won:
        print("  SKIP win test (couldn't reach 1 candidate in 25 guesses)")


def test_env_wrapper():
    print("\n" + "=" * 60)
    print("Test: WordGameEnv wrapper")
    print("=" * 60)

    for game_type in ['wordle', 'quordle', 'absurdle']:
        env = WordGameEnv(game_type, WORD_LIST_PATH)
        state = env.reset()
        assert isinstance(state, dict)
        assert isinstance(env.action_space, list) and len(env.action_space) == 5757
        print(f"{PASS} WordGameEnv({game_type!r}): reset OK, action_space={len(env.action_space)} words")
        assert env.encode_state(state) == state
        print(f"{PASS} WordGameEnv({game_type!r}): encode_state is identity")

    try:
        WordGameEnv('invalid', WORD_LIST_PATH)
        print(f"  FAIL invalid game_type should raise ValueError")
    except ValueError:
        print(f"{PASS} invalid game_type raises ValueError")


def main():
    print("\n" + "=" * 60)
    print("WORD GAME ENVIRONMENTS — TEST SUITE")
    print("=" * 60)

    tests = [
        ("utils", test_utils),
        ("WordleGame", test_wordle),
        ("QuordleGame", test_quordle),
        ("AbsurdleGame", test_absurdle),
        ("WordGameEnv wrapper", test_env_wrapper),
    ]

    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, True))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((name, False))
            print(f"  FAIL {name}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"{'  OK  ' if ok else '  FAIL'} - {name}")
    print(f"Results: {passed}/{len(results)} test groups passed")
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
