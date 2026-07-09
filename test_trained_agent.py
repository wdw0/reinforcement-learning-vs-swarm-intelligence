"""
test_trained_agent.py — Evaluate any trained agent
====================================================
Usage:
    python test_trained_agent.py --model linear
    python test_trained_agent.py --model tile
    python test_trained_agent.py --model dqn
    python test_trained_agent.py --model abc
    python test_trained_agent.py --model bat
    python test_trained_agent.py --model gwo
    python test_trained_agent.py --model firefly
    python test_trained_agent.py --model fss

    python test_trained_agent.py --model dqn --n 50
    python test_trained_agent.py --model abc --render
    python test_trained_agent.py --model linear --weights my_weights.npy
    python test_trained_agent.py --help
"""

import argparse
import numpy as np
from game.core   import SurvivalGame
from game.config import GameConfig


# ─── Default weight paths per model ──────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "linear"  : "q_weights.npy",
    "tile"    : "q_weights_tile.npy",
    "dqn"     : "dqn_weights.pth",
    "abc"     : "abc_weights.npy",
    "bat"     : "bat_weights.npy",
    "gwo"     : "gwo_weights.npy",
    "firefly" : "firefly_weights.npy",
    "fss"     : "fss_weights.npy",
}

# Metaheuristic models that use the shared NeuralNetwork + weights file
META_MODULES = {
    "abc"     : "heuristics.abc",
    "bat"     : "heuristics.bat",
    "gwo"     : "heuristics.gwo",
    "firefly" : "heuristics.firefly",
    "fss"     : "heuristics.fss",
}


# ─── Test runners ─────────────────────────────────────────────────────────────

def test_linear(weights_path: str, num_tests: int, render: bool) -> np.ndarray:
    from heuristics.q_learning import QLearningAgent
    agent = QLearningAgent()
    agent.load(weights_path)
    agent.epsilon = 0.0
    return _run_episodes_rl(agent, num_tests, render)


def test_tile(weights_path: str, num_tests: int, render: bool) -> np.ndarray:
    from heuristics.q_learning_tile import QLearningTileAgent
    agent = QLearningTileAgent()
    agent.load(weights_path)
    agent.epsilon = 0.0
    return _run_episodes_rl(agent, num_tests, render)


def test_dqn(weights_path: str, num_tests: int, render: bool) -> np.ndarray:
    from heuristics.dqn import DQNAgent
    agent = DQNAgent()
    agent.load(weights_path)
    agent.epsilon = 0.0
    return _run_episodes_rl(agent, num_tests, render)


def test_meta(model_key: str, weights_path: str, num_tests: int, render: bool) -> np.ndarray:
    """Shared test runner for all metaheuristic NN agents."""
    import importlib
    from classifier.neural_network import NeuralNetwork
    from game.agents import NeuralNetworkAgent

    mod         = importlib.import_module(META_MODULES[model_key])
    layer_sizes = mod.LAYER_SIZES

    weights = np.load(weights_path)
    nn      = NeuralNetwork(layer_sizes)
    nn.set_weights(weights)
    agent   = NeuralNetworkAgent(nn)

    cfg    = GameConfig(num_players=1, fps=60 if render else 0,
                        render_grid=render)
    scores = []
    for _ in range(num_tests):
        game  = SurvivalGame(config=cfg, render=render)
        state = game.get_state(0, include_internals=True)
        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
            state = game.get_state(0, include_internals=True)
        scores.append(round(game.players[0].score, 2))
    return np.array(scores)


def _run_episodes_rl(agent, num_tests: int, render: bool) -> np.ndarray:
    """Generic episode runner for RL agents (linear, tile, dqn)."""
    cfg    = GameConfig(num_players=1, fps=60 if render else 0,
                        render_grid=render)
    scores = []
    for _ in range(num_tests):
        game  = SurvivalGame(config=cfg, render=render)
        state = game.get_state(0, include_internals=True)
        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
            state = game.get_state(0, include_internals=True)
        scores.append(round(game.players[0].score, 2))
    return np.array(scores)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_models = list(DEFAULT_WEIGHTS.keys())

    parser = argparse.ArgumentParser(
        description="Test a trained agent on the Survival Game.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=all_models,
        required=True,
        help=(
            "Which agent to test:\n"
            "  linear   → Q-Learning linear        (q_weights.npy)\n"
            "  tile     → Q-Learning tile coding    (q_weights_tile.npy)\n"
            "  dqn      → Deep Q-Network            (dqn_weights.pth)\n"
            "  abc      → Bee Colony + NN           (abc_weights.npy)\n"
            "  bat      → Bat Algorithm + NN        (bat_weights.npy)\n"
            "  gwo      → Grey Wolf Optimizer + NN  (gwo_weights.npy)\n"
            "  firefly  → Firefly Algorithm + NN    (firefly_weights.npy)\n"
            "  fss      → Fish School Search + NN   (fss_weights.npy)"
        ),
    )
    parser.add_argument(
        "--weights", default=None,
        help="Override default weights file path.",
    )
    parser.add_argument(
        "--n", type=int, default=30,
        help="Number of test episodes (default: 30).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render the Pygame window during testing.",
    )
    args = parser.parse_args()

    weights_path = args.weights if args.weights else DEFAULT_WEIGHTS[args.model]

    print(f"\n--- Testing {args.model.upper()} agent ---")
    print(f"    Weights : {weights_path}")
    print(f"    Episodes: {args.n}\n")

    if args.model in META_MODULES:
        scores = test_meta(args.model, weights_path, args.n, args.render)
    elif args.model == "linear":
        scores = test_linear(weights_path, args.n, args.render)
    elif args.model == "tile":
        scores = test_tile(weights_path, args.n, args.render)
    elif args.model == "dqn":
        scores = test_dqn(weights_path, args.n, args.render)

    print(scores.tolist())
    print(f"\nResults after {args.n} episodes:")
    print(f"  Max score  : {np.max(scores):.2f}")
    print(f"  Mean score : {np.mean(scores):.2f}")
    print(f"  Std dev    : {np.std(scores):.2f}")


if __name__ == "__main__":
    main()
