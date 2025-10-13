#!/usr/bin/env python3
"""Record human gameplay for imitation learning"""

import sys
import json
import pickle
from datetime import datetime
sys.path.insert(0, '..')

import pygame
from game.game_engine import DroneGame


class GameplayRecorder:
    """Records state-action pairs during gameplay"""

    def __init__(self):
        self.episodes = []
        self.current_episode = []

    def record_step(self, state, action, reward, done, info):
        """Record a single step"""
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })

    def finish_episode(self):
        """Mark current episode as complete"""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def save(self, filename=None):
        """Save recorded gameplay to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gameplay_recording_{timestamp}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self.episodes, f)

        print(f"\nSaved {len(self.episodes)} episodes to {filename}")

        # Also save summary as JSON for easy inspection
        summary_file = filename.replace('.pkl', '_summary.json')
        summary = {
            'num_episodes': len(self.episodes),
            'episodes': [
                {
                    'steps': len(ep),
                    'total_reward': sum(step['reward'] for step in ep),
                    'success': ep[-1]['info'].get('landed', False) if ep else False
                }
                for ep in self.episodes
            ]
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to {summary_file}")


def main():
    """Record human gameplay"""
    game = DroneGame(render_mode='human')
    recorder = GameplayRecorder()

    print("=" * 60)
    print("GAMEPLAY RECORDER")
    print("=" * 60)
    print("\nControls:")
    print("  W or ↑  : Main thrust")
    print("  A or ←  : Left thruster")
    print("  D or →  : Right thruster")
    print("  R       : Reset (finishes current episode)")
    print("  ESC     : Quit and save")
    print("\nPlay as many episodes as you want. Your gameplay will be saved.")
    print("=" * 60 + "\n")

    state = game.reset()
    running = True

    while running:
        action = {
            'main_thrust': 0,
            'left_thrust': 0,
            'right_thrust': 0
        }

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Finish and reset
                    recorder.finish_episode()
                    state = game.reset()
                    print(f"Episode {len(recorder.episodes)} recorded. Starting new episode...")

        # Get key presses
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action['main_thrust'] = 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action['left_thrust'] = 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action['right_thrust'] = 1

        # Step
        next_state, reward, done, info = game.step(action)

        # Record
        recorder.record_step(state, action, reward, done, info)

        # Render
        game.render()

        # Update state
        state = next_state

        # Auto-reset on done
        if done:
            if game.drone.landed:
                print(f"Episode {len(recorder.episodes) + 1}: SUCCESS!")
            else:
                print(f"Episode {len(recorder.episodes) + 1}: CRASHED")

            recorder.finish_episode()
            state = game.reset()

    # Save before exit
    if recorder.current_episode:
        recorder.finish_episode()

    if recorder.episodes:
        recorder.save()
        print(f"\nTotal episodes recorded: {len(recorder.episodes)}")
    else:
        print("\nNo episodes recorded.")

    game.close()


if __name__ == "__main__":
    main()
