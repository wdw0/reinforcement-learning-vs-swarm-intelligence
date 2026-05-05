"""
test_trained_agent.py
=====================
Load a trained QLearningAgent from 'q_weights.npy' and evaluate it
over multiple episodes.

Usage:
    python test_trained_agent.py                   # 30 tests, no render
    python test_trained_agent.py --render           # with Pygame window
    python test_trained_agent.py --n 50            # 50 tests
"""

import argparse
import numpy as np

from game.core   import SurvivalGame
from game.config import GameConfig
from heuristics.q_learning import QLearningAgent, build_features


def test_q_agent(
    weights_path: str  = "q_weights.npy",
    num_tests:    int  = 30,
    render:       bool = False,
) -> np.ndarray:
    """
    Test a pre-trained QLearningAgent.

    Parameters
    ----------
    weights_path : path to saved q_weights.npy
    num_tests    : number of test episodes
    render       : whether to render Pygame frames

    Returns
    -------
    np.ndarray of scores (one per episode)
    """
    print(f"\n--- Testing Q-Learning Agent ({num_tests} episodes) ---")

    # Determine feature size from a dummy state
    _cfg   = GameConfig(num_players=1, fps=0)
    _game  = SurvivalGame(config=_cfg, render=False)
    _state = _game.get_state(0, include_internals=True)
    n_features = len(build_features(_state))

    # Build agent and load weights
    agent = QLearningAgent(n_features=n_features, n_actions=3)
    agent.load(weights_path)
    agent.epsilon = 0.0   # pure exploitation

    cfg = GameConfig(num_players=1, fps=60 if render else 0, render_grid=render)
    scores = []

    for i in range(num_tests):
        game  = SurvivalGame(config=cfg, render=render)
        state = game.get_state(0, include_internals=True)

        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
            state = game.get_state(0, include_internals=True)

        score = game.players[0].score
        scores.append(round(score, 2))

    scores = np.array(scores)
    print(scores)
    print(f"\nResults after {num_tests} tests:")
    print(f"  Max score  : {np.max(scores):.2f}")
    print(f"  Mean score : {np.mean(scores):.2f}")
    print(f"  Std dev    : {np.std(scores):.2f}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="q_weights.npy")
    parser.add_argument("--n",      type=int,  default=30)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    test_q_agent(
        weights_path = args.weights,
        num_tests    = args.n,
        render       = args.render,
    )
