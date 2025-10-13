# Bug Fix: KeyError in manual_play.py

## Issue

When playing the game manually, after an episode ended (crashed or landed), the game would print episode statistics but then crash with:

```
KeyError: 'steps'
```

## Root Cause

The bug had two parts:

### Part 1: Game Engine Issue

In `game/game_engine.py`, when `step()` was called after an episode had already ended (`self.done == True`), it returned early with a minimal info dict:

```python
if self.done:
    return self.get_state(), 0, True, {'needs_reset': True}
```

This minimal dict only contained `{'needs_reset': True}`, missing all the expected keys like `'steps'`, `'total_reward'`, and `'fuel_remaining'`.

### Part 2: Manual Play Issue

In `manual_play.py`, the game loop continued running even after `done=True`. On subsequent frames, the user could still be holding keys, causing more `step()` calls. When the episode results were printed, it would try to access:

```python
print(f"  Steps: {info['steps']}")  # KeyError!
```

But the info dict from the subsequent step after done didn't have these keys.

Additionally, the episode results were printed multiple times because there was no flag to prevent re-printing on every frame after done.

## Solution

### Fix 1: Game Engine (game/game_engine.py)

Changed the early return to include the full info dictionary:

```python
if self.done:
    # Episode already finished, return current info
    info = self._get_info()
    info['needs_reset'] = True
    return self.get_state(), 0, True, info
```

Now even after the episode ends, subsequent `step()` calls return complete info with all required keys.

### Fix 2: Manual Play (manual_play.py)

Added a flag to ensure episode results are only printed once:

```python
running = True
episode_printed = False  # Track if we've printed the episode results

while running:
    # ... game loop ...

    # Print episode results when done (only once)
    if done and not episode_printed:
        # Print results
        episode_printed = True  # Mark as printed
```

When the user presses 'R' to reset, the flag is cleared:

```python
elif event.key == pygame.K_r:
    game.reset()
    episode_printed = False  # Reset flag for new episode
```

## Testing

Created `test_fix.py` to verify:

1. ✅ Info dict has all required keys when episode ends
2. ✅ Info dict still has required keys on subsequent steps after done
3. ✅ Can safely access `info['steps']` even after multiple steps post-done
4. ✅ No KeyError occurs

## Files Modified

- `game/game_engine.py` - Lines 87-91 (step method)
- `manual_play.py` - Lines 33-89 (main loop)

## Verification

Run the tests:

```bash
# Run full test suite
python test_game.py

# Run specific KeyError fix test
python test_fix.py
```

Both should pass without errors.

## Impact

- ✅ No more KeyError crashes
- ✅ Episode statistics print correctly
- ✅ Episode results print only once (cleaner console output)
- ✅ Game continues to work normally after episode ends
- ✅ All existing functionality preserved
- ✅ All example scripts still work correctly

## Status

**FIXED** ✓

The game now handles episode completion gracefully without crashing.
