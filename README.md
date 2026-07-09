# Reinforcement Learning vs Swarm Intelligence

This repository contains the code for my bachelor's thesis (TCC), which compares Reinforcement Learning algorithms and Swarm Intelligence metaheuristics for training agents to play a simple 2D survival/obstacle-avoidance game, all under the same computational budget.

The task is a side-scrolling game where the agent controls a player that can move up, down, or stay still to dodge incoming obstacles. The score increases over time, so the goal is simply to survive as long as possible.

Eight models are implemented and compared:

- **Q-Learning (linear features)** — classic Q-Learning with handcrafted linear features
- **Q-Learning (tile coding)** — Q-Learning with tile coding for state generalization
- **DQN** — Deep Q-Network (PyTorch)
- **ABC** — Artificial Bee Colony optimizing a neural network's weights
- **BAT** — Bat Algorithm optimizing a neural network's weights
- **GWO** — Grey Wolf Optimizer optimizing a neural network's weights
- **Firefly** — Firefly Algorithm optimizing a neural network's weights
- **FSS** — Fish School Search optimizing a neural network's weights

The RL agents (linear, tile, DQN) learn through interaction with the environment via reward signals. The metaheuristic agents (ABC, BAT, GWO, Firefly, FSS) all share the same feedforward neural network architecture (`classifier/neural_network.py`) and instead search directly over the network's weights, using the agent's in-game score as the fitness function.

## Repository structure

```
game/          game engine (SurvivalGame), configuration and agent wrappers
classifier/    feedforward neural network used by the metaheuristic agents
heuristics/    the eight algorithms (q_learning, q_learning_tile, dqn, abc, bat, gwo, firefly, fss)
weights/       best weights obtained during training for each model
main.py        trains a model, plots the learning curve and runs a test evaluation
test_trained_agent.py   loads saved weights and evaluates an agent, no training
human_play.py  play the game yourself with the keyboard
graph.py       statistical comparison (t-test, Mann-Whitney) and boxplot across all agents
```

## Requirements

Python 3.9+ is recommended.

```
pip install numpy matplotlib scipy pygame
pip install torch --index-url https://download.pytorch.org/whl/cpu   # only needed for DQN
```

`pygame` is required even without rendering, since the game engine depends on it. Use `--render` on any script to actually open the game window.

## Training a model

Training is done through `main.py`. Every run trains the model, saves a plot of the learning curve, and finishes with a 30-episode evaluation using the best weights found.

```
python main.py --model linear      # Q-Learning, linear features
python main.py --model tile        # Q-Learning, tile coding
python main.py --model dqn         # Deep Q-Network
python main.py --model abc         # Artificial Bee Colony
python main.py --model bat         # Bat Algorithm
python main.py --model gwo         # Grey Wolf Optimizer
python main.py --model firefly     # Firefly Algorithm
python main.py --model fss         # Fish School Search
```

Useful options:

```
--episodes N     override the number of training episodes (RL models only)
--iterations N   override the number of iterations (metaheuristic models only)
--render         open the Pygame window during training (much slower)
--test-only      skip training, load the saved weights and just run the 30-episode evaluation
```

Weights are saved to the repository root (for example `q_weights.npy`, `dqn_weights.pth`, `abc_weights.npy`), a training log is written to a `training_results_*.txt` file, and a `*.png` plot of the learning curve is generated.

## Testing a trained model

`test_trained_agent.py` loads weights from disk and evaluates the agent without any training step. This is the script to use to reproduce results with the weights already provided in `weights/`, which are the best ones obtained during training for the thesis.

```
python test_trained_agent.py --model linear  --weights weights/q_weights.npy
python test_trained_agent.py --model tile    --weights weights/q_weights_tile.npy
python test_trained_agent.py --model dqn     --weights weights/dqn_weights.pth
python test_trained_agent.py --model abc     --weights weights/abc_weights.npy
python test_trained_agent.py --model bat     --weights weights/bat_weights.npy
python test_trained_agent.py --model gwo     --weights weights/gwo_weights.npy
python test_trained_agent.py --model firefly --weights weights/firefly_weights.npy
python test_trained_agent.py --model fss     --weights weights/fss_weights.npy
```



If `--weights` is omitted, the script looks for the default filename in the current directory instead (for example `q_weights.npy`), not inside `weights/`.

Other options:

```
--n N       number of test episodes (default: 30)
--render    watch the agent play in the Pygame window
```

## Comparing results

`graph.py` recomputes statistical tests (Welch's t-test and Mann-Whitney U) between pairs of agents and produces a boxplot comparing all of them, including a rule-based baseline and human play scores collected for the thesis. Score arrays are hardcoded at the top of the file; edit them if you want to compare a different run.

```
python graph.py
```

## Playing manually

```
python human_play.py
```

Use the up and down arrow keys to move and avoid the obstacles.