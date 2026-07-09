"""
Q-Learning with Linear Value Function Approximation
=====================================================
Semi-gradient TD(0) with replay buffer and ε-greedy policy.
The value function is Q(s,a) = w_a · φ(s), where φ(s) is a 105-dimensional
feature vector built from the game's raw 27-dimensional state.

Reference: Sutton & Barto (2018), ch. 9-10.
"""

import time

import numpy as np
from collections import deque


# ─── Hyperparameters ──────────────────────────────────────────────────────────

MAX_HOURS    = 8.5
ALPHA        = 0.0005
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 0.995
BUFFER_SIZE  = 20_000
BATCH_SIZE   = 128
LEARN_EVERY  = 4
N_ACTIONS    = 3


# ─── Feature engineering ──────────────────────────────────────────────────────

def build_features(state: np.ndarray) -> np.ndarray:
    """Builds the 105-dimensional φ(s) vector from the raw 27-dim state.

    Components: [bias | state | state² | grid×y | grid×speed]
    """
    s     = state.astype(np.float32)
    grid  = s[:25]
    y_pos = s[25]
    speed = s[26]
    return np.concatenate([
        np.array([1.0], dtype=np.float32),
        s,
        s ** 2,
        grid * y_pos,
        grid * speed,
    ])


# ─── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Stores a transition in the buffer."""
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> list:
        """Returns a mini-batch of randomly sampled transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ─── Reward Function ───────────────────────────────────────────────────────────

def compute_reward(prev_alive: bool, curr_alive: bool, state: np.ndarray) -> float:
    """Computes the instantaneous reward.

    Death: -100. Obstacle in the center column: -0.5. Survival: +1.0.
    """
    if prev_alive and not curr_alive:
        return -100.0
    if not curr_alive:
        return 0.0
    reward = 1.0
    center_col = [2, 7, 12, 17, 22]
    if any(state[i] > 0 for i in center_col):
        reward -= 0.5
    return reward


# ─── Linear Q-Learning Agent ───────────────────────────────────────────────────

class QLearningAgent:
    """Q-Learning agent with linear value approximation and replay buffer.

    Implements the predict(state: np.ndarray) -> int interface required by the game.
    Weights w have shape (n_actions, n_features); Q(s,a) = w[a] · φ(s).
    """

    def __init__(
        self,
        n_features:  int   = None,
        n_actions:   int   = N_ACTIONS,
        alpha:       float = ALPHA,
        gamma:       float = GAMMA,
        epsilon:     float = EPS_START,
        eps_decay:   float = EPS_DECAY,
        eps_end:     float = EPS_END,
        buffer_size: int   = BUFFER_SIZE,
        batch_size:  int   = BATCH_SIZE,
    ):
        if n_features is None:
            n_features = len(build_features(np.zeros(27, dtype=np.float32)))

        self.n_features  = n_features
        self.n_actions   = n_actions
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.eps_decay   = eps_decay
        self.eps_end     = eps_end
        self.batch_size  = batch_size
        self.replay      = ReplayBuffer(buffer_size)
        self.total_steps = 0
        self.weights     = np.random.randn(n_actions, n_features).astype(np.float32) * 0.01

    # ── Value computation ─────────────────────────────────────────────────────

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """Returns Q(s,·) for all actions. phi: (n_features,)"""
        return self.weights @ phi

    def q_value(self, phi: np.ndarray, action: int) -> float:
        """Returns Q(s, a) for a specific action."""
        return float(self.weights[action] @ phi)

    # ── Agent interface ───────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """Selects an action via ε-greedy policy. Set epsilon=0 for evaluation."""
        phi = build_features(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(phi)))

    # ── Learning ───────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Adds a transition to the replay buffer."""
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """Semi-gradient TD(0) update on a mini-batch from the buffer.

        Returns the mean absolute TD error, or 0.0 if the buffer is empty.
        """
        if len(self.replay) < self.batch_size:
            return 0.0

        batch  = self.replay.sample(self.batch_size)
        errors = []

        for state, action, reward, next_state, done in batch:
            phi      = build_features(state)
            phi_next = build_features(next_state)
            q_sa     = self.q_value(phi, action)
            target   = reward if done else reward + self.gamma * np.max(self.q_values(phi_next))
            delta    = target - q_sa
            self.weights[action] += self.alpha * delta * phi
            errors.append(abs(delta))

        self.total_steps += 1
        return float(np.mean(errors))

    def decay_epsilon(self):
        """Multiplicatively decays ε. Call once per episode."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "q_weights.npy"):
        """Saves the weight matrix to disk."""
        np.save(path, self.weights)
        print(f"[QLearning] Pesos salvos → '{path}'")

    def load(self, path: str = "q_weights.npy"):
        """Loads the weight matrix from disk."""
        self.weights = np.load(path).astype(np.float32)
        print(f"[QLearning] Pesos carregados ← '{path}'")


# ─── Training Loop ──────────────────────────────────────────────────────────────

def train_q_learning(
    agent,
    GameClass,
    GameConfigClass,
    n_episodes:  int   = 999_999,
    max_hours:   float = MAX_HOURS,
    learn_every: int   = LEARN_EVERY,
    render:      bool  = False,
    save_path:   str   = "q_weights.npy",
    print_every: int   = 100,
) -> list:
    """Trains the linear Q-Learning agent until the time or episode limit is reached.

    Returns the score history per episode.
    """
    config        = GameConfigClass(num_players=1, fps=0 if not render else 60)
    score_history = []
    best_score    = -np.inf
    step_count    = 0
    start_time    = time.time()
    max_seconds   = max_hours * 3600

    for ep in range(1, n_episodes + 1):
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n[Q-Linear] Limite de tempo atingido ({elapsed/3600:.2f}h) — encerrando.")
            break

        game       = GameClass(config=config, render=render)
        state      = game.get_state(0, include_internals=True)
        prev_alive = True
        ep_errors  = []

        while not game.all_players_dead():
            action     = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()

            curr_alive = game.players[0].alive
            reward     = compute_reward(prev_alive, curr_alive, state)
            next_state = game.get_state(0, include_internals=True)
            done       = not curr_alive

            agent.store(state, action, reward, next_state, done)
            step_count += 1
            if step_count % learn_every == 0:
                err = agent.learn()
                if err > 0:
                    ep_errors.append(err)

            state      = next_state
            prev_alive = curr_alive

        episode_score = game.players[0].score
        score_history.append(episode_score)
        agent.decay_epsilon()

        if episode_score > best_score:
            best_score = episode_score
            agent.save(save_path)

        if ep % print_every == 0:
            recent    = score_history[-print_every:]
            avg_err   = float(np.mean(ep_errors)) if ep_errors else 0.0
            elapsed   = time.time() - start_time
            remaining = (max_seconds - elapsed) / 3600
            print(
                f"Ep {ep:>6} | "
                f"Score: {episode_score:>7.2f} | "
                f"Avg({print_every}): {np.mean(recent):>7.2f} | "
                f"Melhor: {best_score:>7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Erro TD: {avg_err:.4f} | "
                f"Restante: {remaining:.2f}h"
            )

    return score_history