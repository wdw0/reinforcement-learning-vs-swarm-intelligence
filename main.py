"""
main.py — Unified Training Entry Point
=======================================
Usage:
    python main.py --model linear       # Q-Learning, linear features  (Phase 2)
    python main.py --model tile         # Q-Learning, tile coding       (Phase 3)
    python main.py --model dqn          # Deep Q-Network                (Phase 4)
    python main.py --model abc          # Artificial Bee Colony + NN    (Phase 5)
    python main.py --model abc --basic  # Basic ABC (no iABC improvement)

    python main.py --model dqn --episodes 2000    # override episode/iteration count
    python main.py --model abc --iterations 500   # override ABC iteration count
    python main.py --model abc --render            # with Pygame window
    python main.py --model dqn --test-only         # evaluate saved weights only
    python main.py --help                          # show all options

INSTALL REQUIREMENTS
--------------------
    pip install numpy matplotlib scipy      # all models
    pip install torch                       # DQN only
    pip install torch --index-url https://download.pytorch.org/whl/cpu
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


def run_test_episodes_rl(agent, n: int = 30) -> np.ndarray:
    """Test a Q-learning or DQN agent over n episodes (epsilon=0)."""
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


def run_test_episodes_nn(weights: np.ndarray, layer_sizes: list,
                         n: int = 30) -> np.ndarray:
    """Test a neural network agent (from ABC weights) over n episodes."""
    from classifier.neural_network import NeuralNetwork
    from game.agents import NeuralNetworkAgent

    nn = NeuralNetwork(layer_sizes)
    nn.set_weights(weights)
    agent  = NeuralNetworkAgent(nn)
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


# ─── Model configs (RL models) ────────────────────────────────────────────────

RL_CONFIGS = {
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
        "n_episodes"   : 8000,
    },
    "dqn": {
        "label"        : "Deep Q-Network (Phase 4)",
        "module"       : "heuristics.dqn",
        "agent_class"  : "DQNAgent",
        "train_fn"     : "train_dqn",
        "save_path"    : "dqn_weights.pth",
        "results_path" : "training_results_dqn.txt",
        "plot_path"    : "dqn_training.png",
        "plot_color"   : "mediumseagreen",
        "n_episodes"   : 5000,
    },
}


# ─── ABC training runner ──────────────────────────────────────────────────────

def run_abc(args):
    """Handle ABC / iABC training and testing."""
    from heuristics.abc import ArtificialBeeColony, LAYER_SIZES, N_BEES, N_ITER, LIMIT, N_RUNS

    use_iabc  = not args.basic
    algo_name = "iABC" if use_iabc else "ABC"
    label     = f"Artificial Bee Colony ({'iABC' if use_iabc else 'basic ABC'}) + Neural Network (Phase 5)"
    save_path    = "abc_weights.npy"
    results_path = f"training_results_{'iabc' if use_iabc else 'abc'}.txt"
    plot_path    = f"{'iabc' if use_iabc else 'abc'}_training.png"
    n_iter       = args.iterations if args.iterations else N_ITER

    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print(f"  Bees (population) : {N_BEES}")
    print(f"  Iterations        : {n_iter}")
    print(f"  Limit (abandon)   : {LIMIT}")
    print(f"  Runs per eval     : {N_RUNS}")
    print(f"  Network           : {LAYER_SIZES}")
    print(f"  iABC improvement  : {use_iabc}")
    print("=" * 65)

    abc = ArtificialBeeColony(
        layer_sizes = LAYER_SIZES,
        n_bees      = N_BEES,
        n_iter      = n_iter,
        limit       = LIMIT,
        n_runs      = N_RUNS,
        use_iabc    = use_iabc,
    )

    # ── Test-only mode ────────────────────────────────────────────────────────
    if args.test_only:
        print(f"[test-only] Loading weights from '{save_path}'...")
        weights = np.load(save_path)
        print("Running 30 test episodes...\n")
        test_scores = run_test_episodes_nn(weights, LAYER_SIZES, n=30)
        print(test_scores)
        print(f"\n  Max   : {np.max(test_scores):.2f}")
        print(f"  Mean  : {np.mean(test_scores):.2f}")
        print(f"  Std   : {np.std(test_scores):.2f}")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nOptimising for {n_iter} iterations...\n")
    train_start = time.time()

    best_weights, best_score, history = abc.optimise(
        GameClass       = SurvivalGame,
        GameConfigClass = GameConfig,
        render          = args.render,
        save_path       = save_path,
        print_every     = 10,
    )

    training_duration = time.time() - train_start
    scores_arr        = np.array(history)

    train_summary = (
        f"\n{'='*65}\n"
        f"  TRAINING COMPLETE\n"
        f"{'='*65}\n"
        f"  Algorithm            : {algo_name}\n"
        f"  Iterations           : {n_iter}\n"
        f"  Time elapsed         : {format_duration(training_duration)}\n"
        f"  Best score           : {best_score:.2f}\n"
        f"  Final avg (pop)      : {np.mean([s for s in history[-10:]]):.2f}\n"
        f"{'='*65}\n"
    )
    print(train_summary)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(scores_arr)), scores_arr,
             color='mediumpurple', linewidth=2, label='Best score per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.title(label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.show()
    print(f"Training curve saved → '{plot_path}'")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\nRunning 30 test episodes with best weights...\n")
    test_start    = time.time()
    test_scores   = run_test_episodes_nn(best_weights, LAYER_SIZES, n=30)
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
    with open(results_path, "w") as f:
        f.write(f"{label.upper()}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("HYPERPARAMETERS\n")
        f.write(f"  Algorithm    : {algo_name}\n")
        f.write(f"  Network      : {LAYER_SIZES}\n")
        f.write(f"  Population   : {N_BEES}\n")
        f.write(f"  Iterations   : {n_iter}\n")
        f.write(f"  Limit        : {LIMIT}\n")
        f.write(f"  Runs/eval    : {N_RUNS}\n")
        f.write(f"  iABC         : {use_iabc}\n\n")
        f.write(train_summary)
        f.write(test_summary)
        f.write("\nBEST SCORE PER ITERATION\n")
        f.write(", ".join(f"{s:.2f}" for s in history) + "\n")

    print(f"Results saved → '{results_path}'")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train an agent on the Survival Game.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["linear", "tile", "dqn", "abc"],
        required=True,
        help=(
            "Which model to train:\n"
            "  linear → Linear Q-learning, engineered features  (Phase 2)\n"
            "  tile   → Linear Q-learning, tile coding           (Phase 3)\n"
            "  dqn    → Deep Q-Network (requires PyTorch)         (Phase 4)\n"
            "  abc    → Artificial Bee Colony + Neural Network    (Phase 5)"
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override default training episodes (RL models).",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override default iteration count (ABC model).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render Pygame window during training (slow).",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip training — load saved weights and run 30 test episodes.",
    )
    parser.add_argument(
        "--basic", action="store_true",
        help="[ABC only] Use basic ABC instead of iABC improvement.",
    )
    args = parser.parse_args()

    # ── ABC: separate runner (different training paradigm) ────────────────────
    if args.model == "abc":
        run_abc(args)
        return

    # ── RL models (linear, tile, dqn) ─────────────────────────────────────────
    cfg = RL_CONFIGS[args.model]

    import importlib
    mod        = importlib.import_module(cfg["module"])
    AgentClass = getattr(mod, cfg["agent_class"])
    train_fn   = getattr(mod, cfg["train_fn"])

    n_episodes = args.episodes if args.episodes is not None else cfg["n_episodes"]

    print("=" * 65)
    print(f"  {cfg['label']}")
    print("=" * 65)
    print(f"  Episodes     : {n_episodes}")
    print(f"  Save path    : {cfg['save_path']}")
    print(f"  Results file : {cfg['results_path']}")
    print("=" * 65)

    agent = AgentClass()

    if hasattr(agent, 'n_features'):
        print(f"\n  Feature dim  : {agent.n_features}")
        print(f"  Weight matrix: {agent.n_actions} × {agent.n_features} "
              f"= {agent.n_actions * agent.n_features:,} parameters\n")
    elif hasattr(agent, 'online_net'):
        n_params = sum(p.numel() for p in agent.online_net.parameters())
        print(f"\n  Network      : {cfg.get('label','')}")
        print(f"  Parameters   : {n_params:,}\n")

    # ── Test-only ─────────────────────────────────────────────────────────────
    if args.test_only:
        print(f"[test-only] Loading weights from '{cfg['save_path']}'...")
        agent.load(cfg["save_path"])
        print("Running 30 test episodes...\n")
        test_scores = run_test_episodes_rl(agent, n=30)
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
        print_every     = 50 if args.model == "dqn" else 100,
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
        f"  Avg last 50 episodes : {np.mean(scores_arr[-50:]):.2f}\n"
        f"  Final epsilon        : {agent.epsilon:.4f}\n"
        f"  Total learn steps    : {agent.total_steps}\n"
        f"{'='*65}\n"
    )
    print(train_summary)

    # ── Plot ──────────────────────────────────────────────────────────────────
    window   = 50
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
    test_scores   = run_test_episodes_rl(agent, n=30)
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
        if hasattr(agent, 'n_features'):
            f.write(f"  Feature dim    : {agent.n_features}\n")
            f.write(f"  Parameters     : {agent.n_actions * agent.n_features:,}\n")
        elif hasattr(agent, 'online_net'):
            n_params = sum(p.numel() for p in agent.online_net.parameters())
            f.write(f"  Parameters     : {n_params:,}\n")
        f.write(f"  Episodes       : {n_episodes}\n\n")
        f.write(train_summary)
        f.write(test_summary)
        f.write("\nFULL SCORE HISTORY (one per episode)\n")
        f.write(", ".join(f"{s:.2f}" for s in score_history) + "\n")

    print(f"Results saved → '{cfg['results_path']}'")


if __name__ == "__main__":
    main()