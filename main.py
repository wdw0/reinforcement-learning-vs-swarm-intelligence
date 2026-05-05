"""
main.py — Q-Learning Training Entry Point
==========================================
Run:  python main.py

Trains a QLearningAgent on the SurvivalGame and saves weights to
'q_weights.npy'.  After training, runs 30 test episodes and prints
performance statistics. Results are saved to 'training_results.txt'.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from game.core   import SurvivalGame
from game.config import GameConfig
from heuristics.q_learning import QLearningAgent, build_features, train_q_learning

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_EPISODES  = 3000
ALPHA       = 0.001
GAMMA       = 0.99
EPS_START   = 1.0
EPS_DECAY   = 0.9995
EPS_END     = 0.05
SAVE_PATH   = "q_weights.npy"
RESULTS_PATH = "training_results.txt"


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def main():
    # ── Determine feature dimensionality ────────────────────────────────────
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
    train_start = time.time()

    score_history = train_q_learning(
        agent           = agent,
        GameClass       = SurvivalGame,
        GameConfigClass = GameConfig,
        n_episodes      = N_EPISODES,
        render          = False,
        save_path       = SAVE_PATH,
        print_every     = 100,
    )

    train_end = time.time()
    training_duration = train_end - train_start

    # ── Training summary ─────────────────────────────────────────────────────
    scores_arr = np.array(score_history)
    train_summary = (
        f"\n{'='*50}\n"
        f"  TRAINING COMPLETE\n"
        f"{'='*50}\n"
        f"  Episodes            : {N_EPISODES}\n"
        f"  Time elapsed        : {format_duration(training_duration)}\n"
        f"  Best score          : {np.max(scores_arr):.2f}\n"
        f"  Final avg (last 100): {np.mean(scores_arr[-100:]):.2f}\n"
        f"  Final epsilon       : {agent.epsilon:.4f}\n"
        f"{'='*50}\n"
    )
    print(train_summary)

    # ── Plot training curve ──────────────────────────────────────────────────
    window = 50
    smoothed = np.convolve(scores_arr, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(scores_arr, alpha=0.3, color='steelblue', label='Episode score')
    plt.plot(range(window - 1, len(scores_arr)), smoothed,
             color='steelblue', linewidth=2, label=f'Moving avg ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Q-Learning Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q_learning_training.png', dpi=120)
    plt.show()

    # ── Test ─────────────────────────────────────────────────────────────────
    print("\n=== Testing trained agent (30 episodes) ===\n")
    agent.load(SAVE_PATH)
    agent.epsilon = 0.0

    test_scores = []
    cfg = GameConfig(num_players=1, fps=0)
    test_start = time.time()

    for i in range(30):
        game  = SurvivalGame(config=cfg, render=False)
        state = game.get_state(0, include_internals=True)
        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            state = game.get_state(0, include_internals=True)
        test_scores.append(game.players[0].score)

    test_duration = time.time() - test_start
    test_scores = np.round(test_scores, 2)

    test_summary = (
        f"\n{'='*50}\n"
        f"  TEST RESULTS (30 episodes)\n"
        f"{'='*50}\n"
        f"  Scores        : {test_scores.tolist()}\n"
        f"  Max score     : {np.max(test_scores):.2f}\n"
        f"  Mean score    : {np.mean(test_scores):.2f}\n"
        f"  Std deviation : {np.std(test_scores):.2f}\n"
        f"  Test time     : {format_duration(test_duration)}\n"
        f"{'='*50}\n"
    )
    print(test_summary)

    # ── Save results to file ─────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        f.write("Q-LEARNING TRAINING RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write(f"  Episodes  : {N_EPISODES}\n")
        f.write(f"  Alpha     : {ALPHA}\n")
        f.write(f"  Gamma     : {GAMMA}\n")
        f.write(f"  Eps start : {EPS_START}\n")
        f.write(f"  Eps decay : {EPS_DECAY}\n")
        f.write(f"  Eps end   : {EPS_END}\n\n")

        f.write(train_summary)
        f.write(test_summary)

        f.write("\nFULL SCORE HISTORY (one per episode)\n")
        f.write(", ".join(f"{s:.2f}" for s in score_history) + "\n")

    print(f"Results saved to '{RESULTS_PATH}'.")


if __name__ == "__main__":
    main()
