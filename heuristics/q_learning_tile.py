"""
Q-Learning com Tile Coding
==========================
Semi-gradiente TD(0) com representação por tile coding nas dimensões contínuas
do estado (posição y do jogador e velocidade do jogo). As 25 células binárias do
grid sensor permanecem como features diretas. Vetor φ(s): 588 dimensões.

Resultado: negativo — tile coding enriqueceu dimensões que não eram o gargalo.
Ver DEC-008 e EXP-003 em FOR_PAPER.md para a análise completa.

Referência: Sutton & Barto (2018), cap. 9-10.
"""

import time

import numpy as np
from collections import deque


# ─── Hiperparâmetros ──────────────────────────────────────────────────────────

MAX_HOURS    = 8.5
N_TILINGS    = 8
N_BINS       = 8
ALPHA        = 0.0001
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 0.997
BUFFER_SIZE  = 30_000
BATCH_SIZE   = 128
LEARN_EVERY  = 4
N_ACTIONS    = 3

# Dimensões contínuas do estado: player_y ∈ [0,1] e game_speed ∈ [0,1]
CONT_DIMS    = [25, 26]
CONT_RANGES  = [(0.0, 1.0), (0.0, 1.0)]


# ─── Tile Coder ───────────────────────────────────────────────────────────────

class TileCoder:
    """Codificação por tiles sobrepostos para dimensões contínuas do estado.

    Cada tiling é deslocado levemente em relação ao anterior, garantindo que
    estados próximos ativem diferentes combinações de tiles. Para cada tiling,
    exatamente um tile é ativado por dimensão contínua.
    """

    def __init__(self, n_tilings, n_bins, cont_dims, cont_ranges):
        self.n_tilings   = n_tilings
        self.n_bins      = n_bins
        self.cont_dims   = cont_dims
        self.cont_ranges = cont_ranges
        self.n_cont      = len(cont_dims)
        self.n_tiles     = n_tilings * (n_bins ** self.n_cont)

        # Deslocamento de cada tiling em cada dimensão contínua
        self._offsets = np.array([
            [i / (n_tilings * n_bins) * (hi - lo) for (lo, hi) in cont_ranges]
            for i in range(n_tilings)
        ], dtype=np.float32)

    def _get_tile_indices(self, cont_values: np.ndarray) -> np.ndarray:
        """Retorna os índices dos tiles ativos (um por tiling)."""
        indices = np.zeros(self.n_tilings, dtype=np.int32)
        for t in range(self.n_tilings):
            tile_idx = 0
            for d in range(self.n_cont):
                lo, hi  = self.cont_ranges[d]
                shifted = np.clip(cont_values[d] + self._offsets[t, d], lo, hi - 1e-8)
                bin_idx = int((shifted - lo) / (hi - lo) * self.n_bins)
                bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
                tile_idx = tile_idx * self.n_bins + bin_idx
            indices[t] = t * (self.n_bins ** self.n_cont) + tile_idx
        return indices

    def encode(self, state: np.ndarray) -> tuple:
        """Codifica as dimensões contínuas em vetor de tiles binário.

        Retorna (tile_features, tile_indices):
          tile_features: vetor one-hot com os tiles ativos
          tile_indices:  índice normalizado de bin por dimensão (para cross-terms)
        """
        cont_values  = np.array([state[d] for d in self.cont_dims], dtype=np.float32)
        active_tiles = self._get_tile_indices(cont_values)

        tile_features = np.zeros(self.n_tiles, dtype=np.float32)
        tile_features[active_tiles] = 1.0

        tile_indices = np.zeros(self.n_cont, dtype=np.float32)
        for d in range(self.n_cont):
            lo, hi = self.cont_ranges[d]
            val    = np.clip(cont_values[d], lo, hi - 1e-8)
            tile_indices[d] = int((val - lo) / (hi - lo) * self.n_bins) / self.n_bins

        return tile_features, tile_indices


# ─── Vetor de Features ────────────────────────────────────────────────────────

_tile_coder = TileCoder(N_TILINGS, N_BINS, CONT_DIMS, CONT_RANGES)


def build_features(state: np.ndarray) -> np.ndarray:
    """Constrói o vetor φ(s) de 588 dimensões.

    Componentes: [grid | tiles | grid×y | grid×velocidade | bias]
    """
    s    = state.astype(np.float32)
    grid = s[:25]

    tile_features, tile_indices = _tile_coder.encode(s)
    y_idx   = tile_indices[0]
    spd_idx = tile_indices[1]

    return np.concatenate([
        grid,
        tile_features,
        grid * y_idx,
        grid * spd_idx,
        np.array([1.0], dtype=np.float32),
    ])


def feature_dim() -> int:
    """Retorna a dimensionalidade do vetor de features."""
    return len(build_features(np.zeros(27, dtype=np.float32)))


# ─── Buffer de Replay ─────────────────────────────────────────────────────────

class ReplayBuffer:
    """Buffer circular de transições (s, a, r, s', done)."""

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Armazena uma transição no buffer."""
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> list:
        """Retorna um mini-batch de transições amostradas aleatoriamente."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ─── Função de Recompensa ─────────────────────────────────────────────────────

def compute_reward(prev_alive: bool, curr_alive: bool, state: np.ndarray) -> float:
    """Calcula a recompensa instantânea.

    Morte: -100. Obstáculo na coluna central: -0.5. Sobrevivência: +1.0.
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


# ─── Agente Q-Learning com Tile Coding ───────────────────────────────────────

class QLearningTileAgent:
    """Agente Q-Learning com representação por tile coding.

    Usa semi-gradiente TD(0) e buffer de replay, idêntico ao agente linear,
    mas com um vetor φ(s) de 588 dimensões baseado em tile coding.
    Interface: predict(state: np.ndarray) -> int.
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
        # Inicialização otimista: pequeno valor positivo incentiva exploração inicial
        self.weights     = np.full((n_actions, n_features), 0.01, dtype=np.float32)

    # ── Cálculo de valor ──────────────────────────────────────────────────────

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """Retorna Q(s,·) para todas as ações. phi: (n_features,)"""
        return self.weights @ phi

    def q_value(self, phi: np.ndarray, action: int) -> float:
        """Retorna Q(s, a) para uma ação específica."""
        return float(self.weights[action] @ phi)

    # ── Interface do agente ───────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """Seleciona ação via política ε-greedy. Defina epsilon=0 na avaliação."""
        phi = build_features(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(phi)))

    # ── Aprendizado ───────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Adiciona uma transição ao buffer de replay."""
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """Atualização de semi-gradiente TD(0) em um mini-batch do buffer.

        Retorna o erro TD médio absoluto, ou 0.0 se o buffer estiver insuficiente.
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
        """Reduz ε multiplicativamente. Chamar uma vez por episódio."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ── Persistência ──────────────────────────────────────────────────────────

    def save(self, path: str = "q_weights_tile.npy"):
        """Salva a matriz de pesos em disco."""
        np.save(path, self.weights)
        print(f"[QLearning-Tile] Pesos salvos → '{path}'")

    def load(self, path: str = "q_weights_tile.npy"):
        """Carrega a matriz de pesos do disco."""
        self.weights = np.load(path).astype(np.float32)
        print(f"[QLearning-Tile] Pesos carregados ← '{path}'")


# ─── Loop de Treinamento ──────────────────────────────────────────────────────

def train_q_learning_tile(
    agent,
    GameClass,
    GameConfigClass,
    n_episodes:  int   = 999_999,
    max_hours:   float = MAX_HOURS,
    learn_every: int   = LEARN_EVERY,
    render:      bool  = False,
    save_path:   str   = "q_weights_tile.npy",
    print_every: int   = 100,
) -> list:
    """Treina o agente Q-Tile até atingir o limite de tempo ou de episódios.

    Retorna o histórico de pontuações por episódio.
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
            print(f"\n[Q-Tile] Limite de tempo atingido ({elapsed/3600:.2f}h) — encerrando.")
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
