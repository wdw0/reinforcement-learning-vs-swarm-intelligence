"""
main.py — Unified Training Entry Point
=======================================
Usage:
    python main.py --model linear       # Q-Learning, engineered features (Phase 2)
    python main.py --model tile         # Q-Learning, tile coding (Phase 3)
    python main.py --model linear --episodes 8000   # override episode count
    python main.py --model tile --render             # with Pygame window
    python main.py --help                            # show all options
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from game.core   import SurvivalGame
from game.config import GameConfig


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def run_test_episodes(agent, n: int = 30) -> np.ndarray:
    """Run n test episodes with epsilon=0, return array of scores."""
    agent.epsilon = 0.0
    cfg    = GameConfig(num_players=1, fps=0)
    scores = []
    for _ in range(n):
        game  = SurvivalGame(config=cfg, render=False)
        state = game.get_state(0, include_internals=True)
        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            state  = game.get_state(0, include_internals=True)
        scores.append(game.players[0].score)
    return np.round(scores, 2)


# ─── Model configs ────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "linear": {
        "label"        : "Q-Learning — Linear Function Approximation (Phase 2)",
        "module"       : "heuristics.q_learning",
        "agent_class"  : "QLearningAgent",
        "train_fn"     : "train_q_learning",
        "save_path"    : "q_weights.npy",
        "results_path" : "training_results_linear.txt",
        "plot_path"    : "q_learning_linear_training.png",
        "plot_color"   : "steelblue",
        "n_episodes"   : 5000,
    },
    "tile": {
        "label"        : "Q-Learning — Tile Coding (Phase 3)",
        "module"       : "heuristics.q_learning_tile",
        "agent_class"  : "QLearningTileAgent",
        "train_fn"     : "train_q_learning_tile",
        "save_path"    : "q_weights_tile.npy",
        "results_path" : "training_results_tile.txt",
        "plot_path"    : "q_learning_tile_training.png",
        "plot_color"   : "darkorange",
        "n_episodes"   : 5000,
    },
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a Q-Learning agent on the Survival Game.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["linear", "tile"],
        required=True,
        help=(
            "Which Q-learning variant to train:\n"
            "  linear  →  engineered features, Phase 2 (q_weights.npy)\n"
            "  tile    →  tile coding,          Phase 3 (q_weights_tile.npy)"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the default number of training episodes.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Pygame window during training (slow).",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and only run 30 test episodes on saved weights.",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    # Dynamically import the selected module
    import importlib
    mod         = importlib.import_module(cfg["module"])
    AgentClass  = getattr(mod, cfg["agent_class"])
    train_fn    = getattr(mod, cfg["train_fn"])

    n_episodes  = args.episodes if args.episodes is not None else cfg["n_episodes"]

    # ── Print header ─────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  {cfg['label']}")
    print("=" * 65)
    print(f"  Episodes     : {n_episodes}")
    print(f"  Save path    : {cfg['save_path']}")
    print(f"  Results file : {cfg['results_path']}")
    print("=" * 65)

    # ── Build agent ──────────────────────────────────────────────────────────
    agent = AgentClass()
    print(f"\n  Feature dim  : {agent.n_features}")
    print(f"  Weight matrix: {agent.n_actions} × {agent.n_features} "
          f"= {agent.n_actions * agent.n_features:,} parameters\n")

    # ── Test-only mode ────────────────────────────────────────────────────────
    if args.test_only:
        print(f"[test-only] Loading weights from '{cfg['save_path']}'...")
        agent.load(cfg["save_path"])
        print("Running 30 test episodes...\n")
        test_scores = run_test_episodes(agent, n=30)
        print(test_scores)
        print(f"\n  Max   : {np.max(test_scores):.2f}")
        print(f"  Mean  : {np.mean(test_scores):.2f}")
        print(f"  Std   : {np.std(test_scores):.2f}")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training for {n_episodes} episodes...\n")
    train_start   = time.time()

    score_history = train_fn(
        agent           = agent,
        GameClass       = SurvivalGame,
        GameConfigClass = GameConfig,
        n_episodes      = n_episodes,
        render          = args.render,
        save_path       = cfg["save_path"],
        print_every     = 100,
    )

    training_duration = time.time() - train_start
    scores_arr        = np.array(score_history)

    train_summary = (
        f"\n{'='*65}\n"
        f"  TRAINING COMPLETE\n"
        f"{'='*65}\n"
        f"  Model                : {args.model}\n"
        f"  Episodes             : {n_episodes}\n"
        f"  Time elapsed         : {format_duration(training_duration)}\n"
        f"  Best score (training): {np.max(scores_arr):.2f}\n"
        f"  Avg last 100 episodes: {np.mean(scores_arr[-100:]):.2f}\n"
        f"  Final epsilon        : {agent.epsilon:.4f}\n"
        f"  Total replay updates : {agent.total_steps}\n"
        f"{'='*65}\n"
    )
    print(train_summary)

    # ── Plot ──────────────────────────────────────────────────────────────────
    window   = 100
    smoothed = np.convolve(scores_arr, np.ones(window) / window, mode='valid')
    color    = cfg["plot_color"]

    plt.figure(figsize=(12, 5))
    plt.plot(scores_arr, alpha=0.2, color=color, label='Episode score')
    plt.plot(
        range(window - 1, len(scores_arr)), smoothed,
        color=color, linewidth=2, label=f'Moving avg ({window} ep)'
    )
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(cfg["label"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg["plot_path"], dpi=120)
    plt.show()
    print(f"Training curve saved → '{cfg['plot_path']}'")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\nLoading best weights for testing...")
    agent.load(cfg["save_path"])

    print("Running 30 test episodes...\n")
    test_start    = time.time()
    test_scores   = run_test_episodes(agent, n=30)
    test_duration = time.time() - test_start

    test_summary = (
        f"\n{'='*65}\n"
        f"  TEST RESULTS (30 episodes)\n"
        f"{'='*65}\n"
        f"  Scores       : {test_scores.tolist()}\n"
        f"  Max score    : {np.max(test_scores):.2f}\n"
        f"  Mean score   : {np.mean(test_scores):.2f}\n"
        f"  Std deviation: {np.std(test_scores):.2f}\n"
        f"  Test time    : {format_duration(test_duration)}\n"
        f"{'='*65}\n"
    )
    print(test_summary)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(cfg["results_path"], "w") as f:
        f.write(f"{cfg['label'].upper()}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("HYPERPARAMETERS\n")
        f.write(f"  Model          : {args.model}\n")
        f.write(f"  Feature dim    : {agent.n_features}\n")
        f.write(f"  Parameters     : {agent.n_actions * agent.n_features:,}\n")
        f.write(f"  Episodes       : {n_episodes}\n\n")

        f.write(train_summary)
        f.write(test_summary)

        f.write("\nFULL SCORE HISTORY (one per episode)\n")
        f.write(", ".join(f"{s:.2f}" for s in score_history) + "\n")

    print(f"Results saved → '{cfg['results_path']}'")


if __name__ == "__main__":
    main()
