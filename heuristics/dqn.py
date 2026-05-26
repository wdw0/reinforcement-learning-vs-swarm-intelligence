import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─── Hyperparameters ─────────────────────────────────────────────────────────

LAYER_SIZES        = [27, 64, 64, 3]  # expanded — more capacity for grid patterns
LR                 = 0.001
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY          = 0.9985           # slow — ε hits floor ~ep 1900
BUFFER_SIZE        = 50_000           # large — diverse transitions during exploration
BATCH_SIZE         = 64
LEARN_EVERY        = 4
TARGET_UPDATE_FREQ = 300              # slightly more frequent than v1
N_ACTIONS          = 3
N_EPISODES         = 5000
REWARD_SCALE       = 50.0             # stronger signal — /50 instead of /100


# ─── Neural Network ───────────────────────────────────────────────────────────

class QNetwork(nn.Module):

    def __init__(self, layer_sizes: list = LAYER_SIZES):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.layers = nn.ModuleList(layers)

        # Xavier initialisation — good default for tanh activations
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)   # tanh on hidden layers, linear on output
        return x


# ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) transitions.
    Rewards normalised at storage time by dividing by REWARD_SCALE.
    """

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward) / REWARD_SCALE,   # normalise at storage time
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int, device: torch.device):
        """Return random mini-batch as tensors on the given device."""
        batch       = random.sample(self.buffer, batch_size)
        states      = torch.tensor(
            np.stack([t[0] for t in batch]), dtype=torch.float32, device=device)
        actions     = torch.tensor(
            [t[1] for t in batch], dtype=torch.long, device=device)
        rewards     = torch.tensor(
            [t[2] for t in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(
            np.stack([t[3] for t in batch]), dtype=torch.float32, device=device)
        dones       = torch.tensor(
            [t[4] for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

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


# ─── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:

    def __init__(
        self,
        layer_sizes:        list  = LAYER_SIZES,
        lr:                 float = LR,
        gamma:              float = GAMMA,
        epsilon:            float = EPS_START,
        eps_decay:          float = EPS_DECAY,
        eps_end:            float = EPS_END,
        buffer_size:        int   = BUFFER_SIZE,
        batch_size:         int   = BATCH_SIZE,
        target_update_freq: int   = TARGET_UPDATE_FREQ,
        n_actions:          int   = N_ACTIONS,
    ):
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.eps_decay          = eps_decay
        self.eps_end            = eps_end
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.n_actions          = n_actions
        self.total_steps        = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Device: {self.device}")

        self.online_net = QNetwork(layer_sizes).to(self.device)
        self.target_net = QNetwork(layer_sizes).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(buffer_size)

        n_params = sum(p.numel() for p in self.online_net.parameters())
        print(f"[DQN] Network: {layer_sizes} — {n_params:,} parameters")

    # ── Agent interface ───────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """Epsilon-greedy. Set epsilon=0 before testing."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s      = torch.tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            q_vals = self.online_net(s)
        return int(q_vals.argmax(dim=1).item())

    # ── Learning ──────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        """Reward is normalised inside ReplayBuffer.push."""
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """
        One gradient descent step.

        Loss: MSE( Q(s,a;θ),  r + γ·max Q(s',·;θ⁻) )

        Returns scalar loss for logging.
        """
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay.sample(self.batch_size, self.device)

        # Q(s, a; θ) for chosen actions
        q_current = self.online_net(states)
        q_sa      = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target = r + γ · max_a' Q(s', a'; θ⁻)
        with torch.no_grad():
            q_next     = self.target_net(next_states)
            max_q_next = q_next.max(dim=1).values
            targets    = rewards + self.gamma * max_q_next * (1.0 - dones)

        loss = F.mse_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1

        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "dqn_weights.pth"):
        torch.save(self.online_net.state_dict(), path)
        print(f"[DQN] Weights saved → '{path}'")

    def load(self, path: str = "dqn_weights.pth"):
        state_dict = torch.load(path, map_location=self.device,
                                weights_only=True)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.online_net.eval()
        self.target_net.eval()
        print(f"[DQN] Weights loaded ← '{path}'")


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_dqn(
    agent,
    GameClass,
    GameConfigClass,
    n_episodes:  int  = N_EPISODES,
    learn_every: int  = LEARN_EVERY,
    render:      bool = False,
    save_path:   str  = "dqn_weights.pth",
    print_every: int  = 50,
) -> list:
  
    config        = GameConfigClass(num_players=1, fps=0 if not render else 60)
    score_history = []
    best_score    = -np.inf
    step_count    = 0

    for ep in range(1, n_episodes + 1):
        game       = GameClass(config=config, render=render)
        state      = game.get_state(0, include_internals=True)
        prev_alive = True
        ep_losses  = []

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
                loss = agent.learn()
                if loss > 0:
                    ep_losses.append(loss)

            state      = next_state
            prev_alive = curr_alive

        episode_score = game.players[0].score
        score_history.append(episode_score)
        agent.decay_epsilon()

        if episode_score > best_score:
            best_score = episode_score
            agent.save(save_path)

        if ep % print_every == 0:
            recent   = score_history[-print_every:]
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            print(
                f"Ep {ep:>5}/{n_episodes} | "
                f"Score: {episode_score:>7.2f} | "
                f"Avg({print_every}): {np.mean(recent):>7.2f} | "
                f"Best: {best_score:>7.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Buffer: {len(agent.replay)}"
            )

    return score_history
