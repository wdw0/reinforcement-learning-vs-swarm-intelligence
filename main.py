"""
main.py — Q-Learning Training Entry Point
==========================================
Run:  python main.py

Trains a QLearningAgent on the SurvivalGame and saves weights to
'q_weights.npy'.  After training, runs 30 test episodes and prints
performance statistics.

The Bat Algorithm and Genetic Algorithm are no longer used.
"""

import numpy as np
import matplotlib.pyplot as plt

from game.core   import SurvivalGame
from game.config import GameConfig

# ── Q-learning imports ───────────────────────────────────────────────────────
from heuristics.q_learning import QLearningAgent, build_features, train_q_learning

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_EPISODES  = 3000    # training episodes (increase for better performance)
ALPHA       = 0.001   # learning rate
GAMMA       = 0.99    # discount factor
EPS_START   = 1.0     # initial ε (full exploration)
EPS_DECAY   = 0.9995  # ε decay per episode
EPS_END     = 0.05    # minimum ε
SAVE_PATH   = "q_weights.npy"


def main():
    # ── Determine feature dimensionality ────────────────────────────────────
    # Run one dummy step to find out the feature vector size
    _cfg   = GameConfig(num_players=1, fps=0)
    _game  = SurvivalGame(config=_cfg, render=False)
    _state = _game.get_state(0, include_internals=True)
    n_features = len(build_features(_state))
    print(f"[Setup] State dim: {len(_state)}  |  Feature dim: {n_features}")

    # ── Build agent ──────────────────────────────────────────────────────────
    agent = QLearningAgent(
        n_features = n_features,
        n_actions  = 3,
        alpha      = ALPHA,
        gamma      = GAMMA,
        epsilon    = EPS_START,
        eps_decay  = EPS_DECAY,
        eps_end    = EPS_END,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n=== Training Q-Learning Agent for {N_EPISODES} episodes ===\n")
    score_history = train_q_learning(
        agent        = agent,
        GameClass    = SurvivalGame,
        GameConfigClass = GameConfig,
        n_episodes   = N_EPISODES,
        render       = False,
        save_path    = SAVE_PATH,
        print_every  = 100,
    )

    # ── Plot training curve ──────────────────────────────────────────────────
    window = 50
    smoothed = np.convolve(score_history, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(score_history, alpha=0.3, color='steelblue', label='Episode score')
    plt.plot(range(window - 1, len(score_history)), smoothed,
             color='steelblue', linewidth=2, label=f'Moving avg ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Q-Learning Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q_learning_training.png', dpi=120)
    plt.show()
    print("Training curve saved to 'q_learning_training.png'.")

    # ── Test ─────────────────────────────────────────────────────────────────
    print("\n=== Testing trained agent (30 episodes, no render) ===\n")
    agent.load(SAVE_PATH)
    agent.epsilon = 0.0          # pure exploitation during test

    test_scores = []
    cfg  = GameConfig(num_players=1, fps=0)
    for i in range(30):
        game  = SurvivalGame(config=cfg, render=False)
        state = game.get_state(0, include_internals=True)
        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            state = game.get_state(0, include_internals=True)
        test_scores.append(game.players[0].score)

    test_scores = np.round(test_scores, 2)
    print(test_scores)
    print(f"\nResults over 30 tests:")
    print(f"  Max score  : {np.max(test_scores):.2f}")
    print(f"  Mean score : {np.mean(test_scores):.2f}")
    print(f"  Std dev    : {np.std(test_scores):.2f}")


if __name__ == "__main__":
    main()
