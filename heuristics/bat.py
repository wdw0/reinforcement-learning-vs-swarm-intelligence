"""
Bat Algorithm (BA)
==================
Otimização de pesos de rede neural para o jogo de sobrevivência.

Inspirado no comportamento de ecolocalização de morcegos (Yang, 2010).
Cada morcego representa um vetor de pesos candidato com três parâmetros:
  frequência (f) — controla a atualização de velocidade
  volume     (A) — diminui conforme o morcego converge para boa solução
  taxa pulsada(r) — aumenta ao longo do tempo, favorecendo busca local

Regras de atualização (por iteração):
  f_i = f_min + (f_max − f_min) × β,   β ~ Uniforme(0, 1)
  v_i = W × v_i + (x_i − x*) × f_i    (W = inércia)
  x_i = x_i + v_i
  se rand > r_i: busca local em torno de x*
  se rand < A_i e f(x_new) > f(x_i): aceitar, atualizar A e r

Referência:
  Yang, X.S. (2010). A New Metaheuristic Bat-Inspired Algorithm.
  Nature Inspired Cooperative Strategies for Optimization (NICSO 2010).
  Springer. https://doi.org/10.1007/978-3-642-12538-6_6
"""

import numpy as np
import time
import os

from classifier.neural_network import NeuralNetwork
from game.agents import NeuralNetworkAgent


# ─── Arquitetura da rede ──────────────────────────────────────────────────────

LAYER_SIZES = [27, 64, 32, 16, 3]

# ─── Hiperparâmetros ──────────────────────────────────────────────────────────

N_BATS      = 100
N_ITER      = 1000
N_RUNS      = 5
F_MIN       = 0.0
F_MAX       = 1.0
A_INIT      = 0.9
R_INIT      = 0.9
ALPHA       = 0.9
GAMMA_BA    = 0.01
INERTIA     = 0.7
BOUNDS      = (-2.0, 2.0)
MAX_HOURS   = 8.5


# ─── Inicialização da população ───────────────────────────────────────────────

def _init_population_layerwise(layer_sizes, n_agents, lb=-2.0, ub=2.0):
    """Inicialização He (Kaiming) por camada: std = sqrt(2 / fan_in) por camada.

    Garante diversidade real desde a iteração 0.
    """
    nn_ref   = NeuralNetwork(layer_sizes)
    n_params = nn_ref.count_weights()
    pop      = np.zeros((n_agents, n_params), dtype=np.float32)

    idx = 0
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        out_sz = layer_sizes[i + 1]
        std    = np.sqrt(2.0 / fan_in)
        w_size = fan_in * out_sz
        b_size = out_sz
        weights = np.random.normal(0, std, (n_agents, w_size)).astype(np.float32)
        pop[:, idx : idx + w_size] = np.clip(weights, lb, ub)
        idx += w_size + b_size

    return pop


# ─── Função de Fitness ────────────────────────────────────────────────────────

def evaluate_population(
    pop:            np.ndarray,
    layer_sizes:    list,
    GameClass,
    GameConfigClass,
    n_runs:         int  = N_RUNS,
    render:         bool = False,
) -> np.ndarray:
    """Avalia todos os N agentes simultaneamente usando o modo multi-jogador do jogo.

    Todos os morcegos jogam na mesma instância (obstáculos compartilhados, posições
    y independentes). Um jogo fornece pontuações para todos os 100 morcegos ao mesmo
    tempo — ~100× mais rápido que 100 jogos individuais sequenciais.

    Retorna vetor de fitness de forma (n_agents,).
    """
    n_agents   = pop.shape[0]
    all_scores = np.zeros((n_runs, n_agents), dtype=np.float32)

    for run in range(n_runs):
        config = GameConfigClass(num_players=n_agents, fps=0 if not render else 60)
        game   = GameClass(config=config, render=render)

        nns    = [NeuralNetwork(layer_sizes) for _ in range(n_agents)]
        for i, nn in enumerate(nns):
            nn.set_weights(pop[i].astype(np.float32))
        agents = [NeuralNetworkAgent(nn) for nn in nns]

        while not game.all_players_dead():
            actions = []
            for i in range(n_agents):
                if game.players[i].alive:
                    state  = game.get_state(i, include_internals=True)
                    action = agents[i].predict(state)
                else:
                    action = 0
                actions.append(action)
            game.update(actions)
            if render:
                game.render_frame()

        for i in range(n_agents):
            all_scores[run, i] = game.players[i].score

    fitness = np.zeros(n_agents, dtype=np.float32)
    for i in range(n_agents):
        scores_i     = all_scores[:, i]
        avg_score    = float(np.mean(scores_i))
        median_score = float(np.median(scores_i))
        robust_score = 0.7 * avg_score + 0.3 * median_score
        consistency  = 1.0 / (1.0 + float(np.std(scores_i)) / max(avg_score, 1.0))
        f = robust_score * (0.85 + 0.15 * consistency)
        if robust_score > 40:  f *= 1.15
        if robust_score > 70:  f *= 1.25
        if robust_score > 120: f *= 1.40
        fitness[i] = f

    return fitness


# ─── Bat Algorithm ────────────────────────────────────────────────────────────

class BatAlgorithm:
    """Bat Algorithm para otimização de pesos de rede neural.

    Parâmetros
    ----------
    layer_sizes : topologia da rede neural
    n_bats      : tamanho da população (requisito: 100)
    n_iter      : máximo de iterações (requisito: 1000)
    n_runs      : jogos multi-jogador por avaliação de fitness
    f_min/f_max : faixa de frequência
    a_init      : volume inicial
    r_init      : taxa pulsada inicial
    alpha       : taxa de decaimento do volume
    gamma       : taxa de decaimento exponencial da taxa pulsada
    inertia     : amortecimento de velocidade (estilo PSO)
    bounds      : (lb, ub) para clipping dos pesos
    max_hours   : orçamento de treinamento
    """

    def __init__(
        self,
        layer_sizes: list  = LAYER_SIZES,
        n_bats:      int   = N_BATS,
        n_iter:      int   = N_ITER,
        n_runs:      int   = N_RUNS,
        f_min:       float = F_MIN,
        f_max:       float = F_MAX,
        a_init:      float = A_INIT,
        r_init:      float = R_INIT,
        alpha:       float = ALPHA,
        gamma:       float = GAMMA_BA,
        inertia:     float = INERTIA,
        bounds:      tuple = BOUNDS,
        max_hours:   float = MAX_HOURS,
    ):
        self.layer_sizes = layer_sizes
        self.n_bats      = n_bats
        self.n_iter      = n_iter
        self.n_runs      = n_runs
        self.f_min       = f_min
        self.f_max       = f_max
        self.a_init      = a_init
        self.r_init      = r_init
        self.alpha       = alpha
        self.gamma       = gamma
        self.inertia     = inertia
        self.lb, self.ub = bounds
        self.max_seconds = max_hours * 3600

        nn = NeuralNetwork(layer_sizes)
        self.n_params = nn.count_weights()

        self.history      = []
        self.best_weights = None
        self.best_score   = -np.inf

    def optimise(
        self,
        GameClass,
        GameConfigClass,
        render:      bool = False,
        save_path:   str  = "bat_weights.npy",
        print_every: int  = 10,
    ) -> tuple:
        """Executa o Bat Algorithm até o limite de tempo ou de iterações.

        Retorna
        -------
        best_weights, best_score, history
        """
        print(f"[BAT] Rede: {self.layer_sizes} | Params: {self.n_params:,}")
        print(f"[BAT] Morcegos={self.n_bats} | Iter={self.n_iter} | "
              f"Runs={self.n_runs} | MaxTime={self.max_seconds/3600:.1f}h")
        print("[BAT] Avaliação: batch multi-jogador (todos os morcegos em um jogo)")

        start_time = time.time()

        pos = _init_population_layerwise(self.layer_sizes, self.n_bats, self.lb, self.ub)
        vel = np.zeros((self.n_bats, self.n_params), dtype=np.float32)
        A   = np.full(self.n_bats, self.a_init, dtype=np.float32)
        r   = np.full(self.n_bats, self.r_init, dtype=np.float32)
        r0  = r.copy()

        print("[BAT] Avaliando população inicial...")
        fitness = evaluate_population(pos, self.layer_sizes, GameClass, GameConfigClass, self.n_runs, render)

        best_idx          = int(np.argmax(fitness))
        self.best_score   = float(fitness[best_idx])
        self.best_weights = pos[best_idx].copy()
        self.history      = [self.best_score]
        np.save(save_path, self.best_weights)

        print(f"[BAT] Melhor fitness inicial: {self.best_score:.4f}\n")

        for it in range(self.n_iter):

            if time.time() - start_time >= self.max_seconds:
                print(f"\n[BAT] Limite de tempo atingido na iteração {it+1}")
                break

            avg_A = float(np.mean(A))

            # ── Passo 1: calcular novas posições de todos os morcegos ─────────
            new_pos = np.empty_like(pos)
            for i in range(self.n_bats):
                beta  = np.random.rand()
                freq  = self.f_min + (self.f_max - self.f_min) * beta
                vel[i] = self.inertia * vel[i] + (pos[i] - self.best_weights) * freq
                x_new  = np.clip(pos[i] + vel[i], self.lb, self.ub).astype(np.float32)

                # Busca local: rand > r → explotar em torno do melhor + ruído gaussiano.
                # r começa em R_INIT≈0.9 (pouca busca local) e decai → mais busca local.
                if np.random.rand() > r[i]:
                    x_new = np.clip(
                        self.best_weights + avg_A * np.random.normal(0, 0.1, self.n_params),
                        self.lb, self.ub
                    ).astype(np.float32)

                new_pos[i] = x_new

            # ── Passo 2: avaliar todas as novas posições em batch ─────────────
            fitness_new = evaluate_population(new_pos, self.layer_sizes, GameClass, GameConfigClass,
                                              self.n_runs, render)

            # ── Passo 3: aceitar/rejeitar por morcego ─────────────────────────
            for i in range(self.n_bats):
                if fitness_new[i] > fitness[i] and np.random.rand() < A[i]:
                    pos[i]     = new_pos[i]
                    fitness[i] = fitness_new[i]
                    A[i]      *= self.alpha
                    r[i]       = r0[i] * np.exp(-self.gamma * (it + 1))

            # ── Passo 4: atualizar melhor global ──────────────────────────────
            best_idx = int(np.argmax(fitness))
            if fitness[best_idx] > self.best_score:
                old               = self.best_score
                self.best_score   = float(fitness[best_idx])
                self.best_weights = pos[best_idx].copy()
                np.save(save_path, self.best_weights)
                elapsed_h = (time.time() - start_time) / 3600
                print(f"[BAT] Iter {it+1:>5} ({elapsed_h:.2f}h) | NOVO MELHOR: {old:.4f} → {self.best_score:.4f} (salvo)")

            self.history.append(self.best_score)

            if (it + 1) % print_every == 0:
                elapsed   = time.time() - start_time
                remaining = (self.max_seconds - elapsed) / 3600
                print(
                    f"Iter {it+1:>5}/{self.n_iter} | "
                    f"Melhor: {self.best_score:>8.4f} | "
                    f"Avg: {np.mean(fitness):>7.4f} | "
                    f"AvgA: {avg_A:.3f} | "
                    f"Restante: {remaining:.1f}h"
                )

        elapsed = time.time() - start_time
        print(f"\n[BAT] Concluído. Tempo: {elapsed/3600:.2f}h | Melhor fitness: {self.best_score:.4f}")

        return self.best_weights, self.best_score, self.history
