"""
Q-Learning with Linear Function Approximation
=============================================
Q(s, a) = w_a · features(s)

For each action a, we maintain a separate weight vector w_a.
Update rule:
    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

Which translates to:
    w_a ← w_a + α · δ · features(s)
    where δ = r + γ max_a' Q(s',a') − Q(s,a)
"""

import numpy as np


# ─── Hyperparameters ────────────────────────────────────────────────────────

ALPHA       = 0.001   # learning rate
GAMMA       = 0.99    # discount factor
EPS_START   = 1.0     # initial exploration rate
EPS_END     = 0.05    # minimum exploration rate
EPS_DECAY   = 0.9995  # multiplicative decay per episode
N_ACTIONS   = 3       # 0=noop, 1=up, 2=down


# ─── Feature Engineering ────────────────────────────────────────────────────

def build_features(state: np.ndarray) -> np.ndarray:
    """
    Build a feature vector from the raw game state.

    The raw state is already a 1-D numpy array (~27 floats):
        - 25 sensor grid cells (5x5, binary occupancy)
        - player_y normalized
        - game speed normalized

    We add a bias term and a few interaction features to help
    the linear approximator capture non-linear patterns cheaply.

    Returns
    -------
    np.ndarray, shape (n_features,)
    """
    # Bias term
    bias = np.array([1.0])

    # Raw state features
    raw = state.astype(np.float32)

    # Squared values (capture non-linearity)
    squared = raw ** 2

    return np.concatenate([bias, raw, squared])


# ─── Q-Learning Agent ────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Q-Learning agent using linear function approximation.

    Q(s, a) = w_a · phi(s)

    One weight vector per action; all updated via semi-gradient TD(0).

    Parameters
    ----------
    n_features : int
        Dimensionality of phi(s) returned by build_features().
    n_actions : int
        Number of discrete actions.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration probability (epsilon-greedy).
    eps_decay : float
        Multiplicative decay applied to epsilon after each episode.
    eps_end : float
        Minimum value of epsilon.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int   = N_ACTIONS,
        alpha: float     = ALPHA,
        gamma: float     = GAMMA,
        epsilon: float   = EPS_START,
        eps_decay: float = EPS_DECAY,
        eps_end: float   = EPS_END,
    ):
        self.n_features = n_features
        self.n_actions  = n_actions
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.eps_decay  = eps_decay
        self.eps_end    = eps_end

        # One weight vector per action, initialized to small random values
        self.weights = np.random.randn(n_actions, n_features).astype(np.float32) * 0.01

    # ── Q-value helpers ──────────────────────────────────────────────────────

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """Return Q(s,·) for all actions given feature vector phi."""
        return self.weights @ phi  # shape (n_actions,)

    def q_value(self, phi: np.ndarray, action: int) -> float:
        """Return Q(s, a) for a single action."""
        return float(self.weights[action] @ phi)

    # ── Action selection ─────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        During training epsilon > 0 allows exploration.
        At test time, load weights and set epsilon=0 for pure exploitation.
        """
        phi = build_features(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(phi)))

    # ── TD update ────────────────────────────────────────────────────────────

    def update(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> float:
        """
        Semi-gradient TD(0) update for linear Q-learning.

        δ = r + γ · max_a' Q(s', a') − Q(s, a)   (if not done)
        δ = r − Q(s, a)                             (if done)

        w_a ← w_a + α · δ · phi(s)

        Returns the TD error δ for logging.
        """
        phi      = build_features(state)
        phi_next = build_features(next_state)

        q_sa = self.q_value(phi, action)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_values(phi_next))

        delta = target - q_sa  # TD error

        # Gradient step only on the weight vector for the chosen action
        self.weights[action] += self.alpha * delta * phi

        return delta

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Call once per episode to anneal exploration."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "q_weights.npy"):
        """Save weight matrix to disk."""
        np.save(path, self.weights)
        print(f"[QLearning] Weights saved to '{path}'.")

    def load(self, path: str = "q_weights.npy"):
        """Load weight matrix from disk."""
        self.weights = np.load(path).astype(np.float32)
        print(f"[QLearning] Weights loaded from '{path}'.")


# ─── Reward Shaping ──────────────────────────────────────────────────────────

def compute_reward(
    prev_alive: bool,
    curr_alive: bool,
    step:       int,
) -> float:
    """
    Derive reward from environment transitions.

    The game environment does not return an explicit reward signal,
    so we engineer one from observable state changes:

    +0.1  per step survived       — encourages staying alive
    -50   on death                — strong penalty for dying
    """
    if prev_alive and not curr_alive:
        return -50.0        # death penalty
    if curr_alive:
        return 0.1          # survival reward per step
    return 0.0              # agent was already dead


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_q_learning(
    agent,
    GameClass,
    GameConfigClass,
    n_episodes:  int   = 2000,
    render:      bool  = False,
    save_path:   str   = "q_weights.npy",
    print_every: int   = 100,
) -> list:
    """
    Train the QLearningAgent against the SurvivalGame.

    Parameters
    ----------
    agent          : QLearningAgent instance (pre-constructed).
    GameClass      : SurvivalGame class.
    GameConfigClass: GameConfig class.
    n_episodes     : Total training episodes.
    render         : Whether to render Pygame frames.
    save_path      : File to save best weights.
    print_every    : Log interval (episodes).

    Returns
    -------
    score_history : list of float  (score per episode)
    """
    config = GameConfigClass(num_players=1, fps=0 if not render else 60)
    score_history = []
    best_score    = -np.inf

    for ep in range(1, n_episodes + 1):
        game  = GameClass(config=config, render=render)
        state = game.get_state(0, include_internals=True)

        total_reward = 0.0
        prev_alive   = True

        while not game.all_players_dead():
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()

            curr_alive = game.players[0].alive
            reward     = compute_reward(prev_alive, curr_alive, game.frame_count)
            total_reward += reward

            next_state = game.get_state(0, include_internals=True)
            done       = not curr_alive

            agent.update(state, action, reward, next_state, done)

            state      = next_state
            prev_alive = curr_alive

        episode_score = game.players[0].score
        score_history.append(episode_score)

        # Decay exploration
        agent.decay_epsilon()

        # Save if best so far
        if episode_score > best_score:
            best_score = episode_score
            agent.save(save_path)

        if ep % print_every == 0:
            recent = score_history[-print_every:]
            print(
                f"Episode {ep:>5}/{n_episodes} | "
                f"Score: {episode_score:>7.2f} | "
                f"Avg(last {print_every}): {np.mean(recent):>7.2f} | "
                f"Best: {best_score:>7.2f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    return score_history
