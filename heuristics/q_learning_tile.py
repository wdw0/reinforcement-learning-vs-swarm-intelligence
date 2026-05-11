import numpy as np
from collections import deque


# ─── Hyperparameters ─────────────────────────────────────────────────────────

N_TILINGS    = 8        # number of overlapping grids
N_BINS       = 8        # bins per dimension per tiling
ALPHA        = 0.0001   # learning rate (smaller — more parameters now)
GAMMA        = 0.99     # discount factor
EPS_START    = 1.0      # initial exploration rate
EPS_END      = 0.05     # minimum exploration rate
EPS_DECAY    = 0.997    # slower decay — more episodes needed to fill tile space
BUFFER_SIZE  = 30_000   # larger buffer — richer feature space needs more diversity
BATCH_SIZE   = 128      # mini-batch size per update
LEARN_EVERY  = 4        # update every N environment steps
N_ACTIONS    = 3        # 0=noop, 1=up, 2=down

# Continuous state dimensions and their ranges
# state[25] = player_y in [0, 1]
# state[26] = game_speed in [0, 1]
CONT_DIMS    = [25, 26]
CONT_RANGES  = [(0.0, 1.0), (0.0, 1.0)]


# ─── Tile Coder ───────────────────────────────────────────────────────────────

class TileCoder:
    def __init__(self, n_tilings, n_bins, cont_dims, cont_ranges):
        self.n_tilings   = n_tilings
        self.n_bins      = n_bins
        self.cont_dims   = cont_dims
        self.cont_ranges = cont_ranges
        self.n_cont      = len(cont_dims)

        # Total number of tile features
        # Each tiling has n_bins^n_cont tiles; exactly one fires per tiling
        self.n_tiles = n_tilings * (n_bins ** self.n_cont)

        # Precompute offsets: each tiling is shifted by offset_i in each dim
        # Standard offset: i / (n_tilings * n_bins) of the full range
        self._offsets = np.array([
            [
                i / (n_tilings * n_bins) * (hi - lo)
                for (lo, hi) in cont_ranges
            ]
            for i in range(n_tilings)
        ], dtype=np.float32)   # shape (n_tilings, n_cont)

    def _get_tile_indices(self, cont_values: np.ndarray) -> np.ndarray:
        indices = np.zeros(self.n_tilings, dtype=np.int32)

        for t in range(self.n_tilings):
            tile_idx = 0
            for d in range(self.n_cont):
                lo, hi   = self.cont_ranges[d]
                val      = cont_values[d]
                # Shift value by tiling offset, then clamp to range
                shifted  = val + self._offsets[t, d]
                shifted  = np.clip(shifted, lo, hi - 1e-8)
                # Map to bin index
                bin_idx  = int((shifted - lo) / (hi - lo) * self.n_bins)
                bin_idx  = np.clip(bin_idx, 0, self.n_bins - 1)
                # Combine dimensions into single flat tile index (row-major)
                tile_idx = tile_idx * self.n_bins + bin_idx

            # Offset by tiling number so each tiling occupies its own region
            indices[t] = t * (self.n_bins ** self.n_cont) + tile_idx

        return indices

    def encode(self, state: np.ndarray) -> tuple:
        cont_values  = np.array([state[d] for d in self.cont_dims], dtype=np.float32)
        active_tiles = self._get_tile_indices(cont_values)

        tile_features           = np.zeros(self.n_tiles, dtype=np.float32)
        tile_features[active_tiles] = 1.0

        # Normalized bin indices for cross terms (one per continuous dim)
        tile_indices = np.zeros(self.n_cont, dtype=np.float32)
        for d in range(self.n_cont):
            lo, hi      = self.cont_ranges[d]
            val         = np.clip(cont_values[d], lo, hi - 1e-8)
            tile_indices[d] = int((val - lo) / (hi - lo) * self.n_bins) / self.n_bins

        return tile_features, tile_indices


# ─── Feature Vector ───────────────────────────────────────────────────────────

# Build a global tile coder (shared across all calls to build_features)
_tile_coder = TileCoder(N_TILINGS, N_BINS, CONT_DIMS, CONT_RANGES)


def build_features(state: np.ndarray) -> np.ndarray:
    s              = state.astype(np.float32)
    grid           = s[:25]                          # sensor grid (binary)

    tile_features, tile_indices = _tile_coder.encode(s)

    y_idx          = tile_indices[0]                 # normalized y bin
    spd_idx        = tile_indices[1]                 # normalized speed bin

    cross_y        = grid * y_idx                    # grid × y position
    cross_speed    = grid * spd_idx                  # grid × speed
    bias           = np.array([1.0], dtype=np.float32)

    return np.concatenate([grid, tile_features, cross_y, cross_speed, bias])


def feature_dim() -> int:
    """Return the dimensionality of the feature vector."""
    dummy = np.zeros(27, dtype=np.float32)
    return len(build_features(dummy))


# ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer for (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done)
        ))

    def sample(self, batch_size: int) -> list:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ─── Reward Function ──────────────────────────────────────────────────────────

def compute_reward(prev_alive: bool, curr_alive: bool, state: np.ndarray) -> float:
    if prev_alive and not curr_alive:
        return -100.0
    if not curr_alive:
        return 0.0

    reward = 1.0
    center_col = [2, 7, 12, 17, 22]
    if any(state[i] > 0 for i in center_col):
        reward -= 0.5

    return reward


# ─── Q-Learning Agent ─────────────────────────────────────────────────────────

class QLearningTileAgent:

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
            n_features = feature_dim()

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

        # Weight matrix: (n_actions, n_features)
        # Optimistic initialisation: small positive values encourage the agent
        # to try all actions early on rather than defaulting to argmax of zeros
        self.weights = np.full(
            (n_actions, n_features), 0.01, dtype=np.float32
        )

    # ── Q-value helpers ───────────────────────────────────────────────────────

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """Q(s,·) for all actions. phi shape: (n_features,)"""
        return self.weights @ phi

    def q_value(self, phi: np.ndarray, action: int) -> float:
        return float(self.weights[action] @ phi)

    # ── Agent interface ───────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        Set epsilon=0 before testing for pure exploitation.
        """
        phi = build_features(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(phi)))

    # ── Learning ──────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        
        if len(self.replay) < self.batch_size:
            return 0.0

        batch  = self.replay.sample(self.batch_size)
        errors = []

        for state, action, reward, next_state, done in batch:
            phi      = build_features(state)
            phi_next = build_features(next_state)

            q_sa  = self.q_value(phi, action)
            target = reward if done else reward + self.gamma * np.max(self.q_values(phi_next))

            delta = target - q_sa
            self.weights[action] += self.alpha * delta * phi
            errors.append(abs(delta))

        self.total_steps += 1
        return float(np.mean(errors))

    def decay_epsilon(self):
        """Anneal exploration — call once per episode."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "q_weights_tile.npy"):
        np.save(path, self.weights)
        print(f"[QLearning-Tile] Weights saved → '{path}'")

    def load(self, path: str = "q_weights_tile.npy"):
        self.weights = np.load(path).astype(np.float32)
        print(f"[QLearning-Tile] Weights loaded ← '{path}'")


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_q_learning_tile(
    agent,
    GameClass,
    GameConfigClass,
    n_episodes:  int  = 8000,
    learn_every: int  = LEARN_EVERY,
    render:      bool = False,
    save_path:   str  = "q_weights_tile.npy",
    print_every: int  = 100,
) -> list:
  

    config        = GameConfigClass(num_players=1, fps=0 if not render else 60)
    score_history = []
    best_score    = -np.inf
    step_count    = 0

    for ep in range(1, n_episodes + 1):
        game       = GameClass(config=config, render=render)
        state      = game.get_state(0, include_internals=True)
        prev_alive = True
        ep_errors  = []

        while not game.all_players_dead():
            action = agent.predict(state)
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
            recent  = score_history[-print_every:]
            avg_err = float(np.mean(ep_errors)) if ep_errors else 0.0
            print(
                f"Ep {ep:>5}/{n_episodes} | "
                f"Score: {episode_score:>7.2f} | "
                f"Avg({print_every}): {np.mean(recent):>7.2f} | "
                f"Best: {best_score:>7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"TD err: {avg_err:.4f} | "
                f"Buffer: {len(agent.replay)}"
            )

    return score_history
