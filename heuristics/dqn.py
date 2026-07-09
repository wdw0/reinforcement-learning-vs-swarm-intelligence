"""
Deep Q-Network (DQN) — v8  "Rainbow-lite + backbone wider"
===========================================================
Double DQN + Dueling Architecture + n-step Returns (n=10)
+ Prioritized Experience Replay (PER) + 300k buffer

20,356 parameters. He/Kaiming initialization (correct for LeakyReLU).
Training budget: 8.5h (same budget as all agents).

References:
  Mnih et al. (2015). Human-level control through deep RL. Nature 518.
  Wang et al. (2016). Dueling network architectures for DRL. ICML.
  Schaul et al. (2016). Prioritized experience replay. ICLR.
  Hasselt et al. (2016). Deep RL with Double Q-learning. AAAI.
"""

import numpy as np
from collections import deque
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─── Hyperparameters ──────────────────────────────────────────────────────────

N_STEP             = 10
LR                 = 0.0001
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY_STEPS    = 1_500_000
BUFFER_SIZE        = 300_000
BATCH_SIZE         = 64
LEARN_EVERY        = 4
TARGET_UPDATE_FREQ = 1_000
WARMUP_STEPS       = 10_000
N_ACTIONS          = 3
N_EPISODES         = 999_999
MAX_HOURS          = 8.5
GRAD_CLIP          = 1.0

# PER — Prioritized Experience Replay
PER_ALPHA          = 0.6
PER_BETA_START     = 0.4
PER_BETA_END       = 1.0
PER_BETA_STEPS     = 6_000_000
PER_EPS            = 1e-6


# ─── Reward Function ───────────────────────────────────────────────────────────

def compute_reward(prev_alive: bool, curr_alive: bool, state: np.ndarray) -> float:
    """Survival signal: +0.05 per step, -5.0 on death."""
    if prev_alive and not curr_alive:
        return -5.0
    if not curr_alive:
        return 0.0
    return 0.05


# ─── SumTree ─────────────────────────────────────────────────────────────────

class SumTree:
    """Binary tree where leaves store priorities and internal nodes store sums.

    Supports insertion, update, and priority-proportional sampling in O(log n).
    """

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = [None] * capacity
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] \
               else self._retrieve(right, s - self.tree[left])

    def add(self, priority: float, data):
        """Inserts new data with the given priority."""
        leaf = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        """Updates a leaf's priority and propagates the difference."""
        self._propagate(leaf_idx, priority - self.tree[leaf_idx])
        self.tree[leaf_idx] = priority

    def get(self, s: float):
        """Returns (leaf_idx, priority, data) for the cumulative value s."""
        leaf     = self._retrieve(0, s)
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    @property
    def total(self) -> float:
        """Total sum of all priorities."""
        return float(self.tree[0])


# ─── Prioritized Replay Buffer (PER) ──────────────────────────────────────────

class PrioritizedReplayBuffer:
    """Replay buffer with priority-proportional sampling (Schaul et al. 2016).

    Priority ∝ |TD error|^α. Importance sampling (IS) weights correct sampling bias.
    """

    def __init__(
        self,
        capacity:   int   = BUFFER_SIZE,
        alpha:      float = PER_ALPHA,
        beta_start: float = PER_BETA_START,
        beta_end:   float = PER_BETA_END,
        beta_steps: int   = PER_BETA_STEPS,
    ):
        self.tree       = SumTree(capacity)
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._step      = 0
        self.max_prio   = 1.0

    @property
    def beta(self) -> float:
        """Current β value, linearly interpolated from beta_start to beta_end."""
        frac = min(1.0, self._step / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, state, action, reward, next_state, done):
        """Inserts a transition with the current maximum priority."""
        self.tree.add(self.max_prio, (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int, device: torch.device):
        """Samples a priority-proportional mini-batch and computes IS weights."""
        self._step += 1
        indices, priorities, batch = [], [], []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total)
                idx, prio, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(max(prio, PER_EPS))
            batch.append(data)

        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        probs   = np.clip(probs, 1e-10, 1.0)
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        states      = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32, device=device)
        actions     = torch.tensor([t[1] for t in batch],           dtype=torch.long,    device=device)
        rewards     = torch.tensor([t[2] for t in batch],           dtype=torch.float32, device=device)
        next_states = torch.tensor(np.stack([t[3] for t in batch]), dtype=torch.float32, device=device)
        dones       = torch.tensor([t[4] for t in batch],           dtype=torch.float32, device=device)
        weights_t   = torch.tensor(weights,                          dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones, indices, weights_t

    def update_priorities(self, indices, td_errors: np.ndarray):
        """Updates the priorities of sampled transitions based on TD errors."""
        for idx, err in zip(indices, td_errors):
            prio = (float(abs(err)) + PER_EPS) ** self.alpha
            self.tree.update(idx, prio)
            self.max_prio = max(self.max_prio, prio)

    def __len__(self) -> int:
        return self.tree.n_entries


# ─── n-step Buffer ─────────────────────────────────────────────────────────────

class NStepBuffer:
    """Accumulates n consecutive transitions and computes the n-step return.

    G = r_t + γ r_{t+1} + ... + γ^{n-1} r_{t+n-1}

    If any step within the horizon is terminal, the return is truncated there.
    """

    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self.buf   = deque()

    def push(self, state, action, reward, next_state, done):
        """Adds a transition to the n-step buffer."""
        self.buf.append((state, action, reward, next_state, done))

    def ready(self) -> bool:
        """Returns True when the buffer has n accumulated transitions."""
        return len(self.buf) >= self.n

    def get(self):
        """Removes and returns (s, a, G_n, s_{t+n}, done) for the oldest transition."""
        state, action = self.buf[0][0], self.buf[0][1]
        G = 0.0
        term_ns, term_done = self.buf[-1][3], self.buf[-1][4]
        for i in range(self.n):
            _, _, r, ns, d = self.buf[i]
            G += (self.gamma ** i) * r
            if d:
                term_ns, term_done = ns, True
                break
        self.buf.popleft()
        return state, action, G, term_ns, term_done

    def flush(self) -> list:
        """Drains the buffer at the end of an episode with reduced horizons."""
        results = []
        while self.buf:
            state, action = self.buf[0][0], self.buf[0][1]
            G = 0.0
            term_ns, term_done = self.buf[-1][3], self.buf[-1][4]
            for i, (_, _, r, ns, d) in enumerate(self.buf):
                G += (self.gamma ** i) * r
                if d:
                    term_ns, term_done = ns, True
                    break
            results.append((state, action, G, term_ns, term_done))
            self.buf.popleft()
        return results


# ─── Dueling Q Network ─────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """DQN network with a Dueling architecture (Wang et al. 2016).

    Shared backbone → Value stream V(s) + Advantage stream A(s,a).
    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
    """

    def __init__(self):
        super().__init__()
        self.feature   = nn.Sequential(nn.Linear(27, 128), nn.LeakyReLU(0.01))
        self.value     = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Linear(64, N_ACTIONS))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


# ─── DQN Agent ───────────────────────────────────────────────────────────────

class DQNAgent:
    """DQN v8 agent — Rainbow-lite (Double DQN + Dueling + n-step + PER).

    Interface: predict(state: np.ndarray) -> int.
    """

    def __init__(
        self,
        lr:                 float = LR,
        gamma:              float = GAMMA,
        epsilon:            float = EPS_START,
        eps_start:          float = EPS_START,
        eps_decay_steps:    int   = EPS_DECAY_STEPS,
        eps_end:            float = EPS_END,
        buffer_size:        int   = BUFFER_SIZE,
        batch_size:         int   = BATCH_SIZE,
        target_update_freq: int   = TARGET_UPDATE_FREQ,
        warmup_steps:       int   = WARMUP_STEPS,
        n_step:             int   = N_STEP,
        n_actions:          int   = N_ACTIONS,
    ):
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.eps_start          = eps_start
        self.eps_decay_steps    = eps_decay_steps
        self.eps_end            = eps_end
        self.eps_step           = 0
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps       = warmup_steps
        self.n_step             = n_step
        self.n_actions          = n_actions
        self.total_steps        = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] Dispositivo: {self.device}")

        self.online_net = DuelingQNetwork().to(self.device)
        self.target_net = DuelingQNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay    = PrioritizedReplayBuffer(buffer_size)
        self.nstep_buf = NStepBuffer(n_step, gamma)

        n_params = sum(p.numel() for p in self.online_net.parameters())
        print(f"[DQN] Rede Dueling: 27→128→(64→1 | 64→3) — {n_params:,} parâmetros")
        print(f"[DQN] n-step={n_step} | PER α={PER_ALPHA} β={PER_BETA_START}→{PER_BETA_END} over {PER_BETA_STEPS:,} steps")
        print(f"[DQN] Buffer={buffer_size:,} | Warmup={warmup_steps:,} | MaxTime={MAX_HOURS}h")
        print(f"[DQN] ε: {EPS_START}→{EPS_END} linear over {eps_decay_steps:,} env steps")

    def predict(self, state: np.ndarray) -> int:
        """Selects an action via ε-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online_net(s).argmax(dim=1).item())

    def step_epsilon(self):
        """Linear ε decay: eps_start → eps_end over eps_decay_steps steps."""
        self.eps_step += 1
        frac          = min(1.0, self.eps_step / self.eps_decay_steps)
        self.epsilon  = self.eps_start - frac * (self.eps_start - self.eps_end)

    def push(self, state, action, reward, next_state, done):
        """Feeds a transition through the n-step buffer and inserts into PER when ready."""
        self.nstep_buf.push(state, action, reward, next_state, done)
        if self.nstep_buf.ready():
            self.replay.push(*self.nstep_buf.get())
        if done:
            for trans in self.nstep_buf.flush():
                self.replay.push(*trans)

    def learn(self):
        """Performs a gradient update with Huber loss weighted by IS weights.

        Returns (loss, indices, td_errors) for PER priority updates.
        """
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return 0.0, None, None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            # Double DQN: online network selects the action, target network evaluates
            next_acts = self.online_net(next_states).argmax(dim=1, keepdim=True)
            q_next    = self.target_net(next_states).gather(1, next_acts).squeeze(1)
            gamma_n   = self.gamma ** self.n_step
            targets   = rewards + gamma_n * q_next * (1.0 - dones)

        q_sa    = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errs = (targets - q_sa).detach().cpu().numpy()

        elem_loss = F.huber_loss(q_sa, targets, reduction='none')
        loss      = (weights * elem_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item()), indices, td_errs

    def save(self, path: str = "dqn_weights.pth"):
        """Saves the online network's weights to disk."""
        torch.save(self.online_net.state_dict(), path)
        print(f"[DQN] Pesos salvos → '{path}'")

    def load(self, path: str = "dqn_weights.pth"):
        """Loads the online and target network weights from disk. Sets ε=0."""
        sd = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)
        self.online_net.eval()
        self.target_net.eval()
        self.epsilon = 0.0
        print(f"[DQN] Pesos carregados ← '{path}'")


# ─── Training Loop ──────────────────────────────────────────────────────────────

def train_dqn(
    agent,
    GameClass,
    GameConfigClass,
    n_episodes:  int  = N_EPISODES,
    learn_every: int  = LEARN_EVERY,
    render:      bool = False,
    save_path:   str  = "dqn_weights.pth",
    print_every: int  = 100,
) -> list:
    """Trains the DQN v8 agent until MAX_HOURS or n_episodes is reached.

    Returns the score history per episode.
    """
    config        = GameConfigClass(num_players=1, fps=0 if not render else 60)
    score_history = []
    best_score    = -np.inf
    step_count    = 0
    warming_up    = True
    max_seconds   = MAX_HOURS * 3600
    start_time    = time.time()

    for ep in range(1, n_episodes + 1):
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n[DQN] Limite de tempo atingido no ep {ep} ({elapsed/3600:.2f}h) — encerrando.")
            break

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

            agent.push(state, action, reward, next_state, done)
            agent.step_epsilon()
            step_count += 1

            if warming_up and len(agent.replay) >= agent.warmup_steps:
                warming_up = False
                print(f"[DQN] Warmup concluído no ep {ep} ({len(agent.replay):,} transições) — aprendizado iniciado.")

            if not warming_up and step_count % learn_every == 0:
                loss, indices, td_errs = agent.learn()
                if loss > 0 and indices is not None:
                    agent.replay.update_priorities(indices, td_errs)
                    ep_losses.append(loss)

            state      = next_state
            prev_alive = curr_alive

        episode_score = game.players[0].score
        score_history.append(episode_score)

        if episode_score > best_score:
            best_score = episode_score
            agent.save(save_path)

        if ep % print_every == 0:
            elapsed   = time.time() - start_time
            remaining = (max_seconds - elapsed) / 3600
            recent    = score_history[-print_every:]
            avg_loss  = float(np.mean(ep_losses)) if ep_losses else 0.0
            status    = "WARMUP" if warming_up else f"ε={agent.epsilon:.5f}"
            print(
                f"Ep {ep:>6} | "
                f"Score: {episode_score:>7.2f} | "
                f"Avg({print_every}): {np.mean(recent):>7.2f} | "
                f"Melhor: {best_score:>7.2f} | "
                f"{status} | "
                f"Loss: {avg_loss:.5f} | "
                f"Passos: {step_count:,} | "
                f"Restante: {remaining:.1f}h"
            )

    return score_history