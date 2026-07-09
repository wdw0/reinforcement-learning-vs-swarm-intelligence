"""
Grey Wolf Optimizer (GWO)
=========================
Otimização de pesos de rede neural para o jogo de sobrevivência.

Inspirado na hierarquia de liderança e comportamento de caça de lobos cinzentos
(Mirjalili, Mirjalili & Lewis, 2014).

Hierarquia de lobos:
  α (alpha) — melhor solução encontrada
  β (beta)  — segunda melhor solução
  δ (delta) — terceira melhor solução
  ω (omega) — demais lobos, guiados por α, β e δ

Atualização de posição para cada lobo ω:
  A₁,A₂,A₃ = 2a·r₁−a,  2a·r₂−a,  2a·r₃−a
  C₁,C₂,C₃ = 2·r₄,     2·r₅,     2·r₆
  D_α = |C₁·X_α − X|,  X₁ = X_α − A₁·D_α
  D_β = |C₂·X_β − X|,  X₂ = X_β − A₂·D_β
  D_δ = |C₃·X_δ − X|,  X₃ = X_δ − A₃·D_δ
  X(t+1) = (X₁ + X₂ + X₃) / 3

O parâmetro `a` decai linearmente de 2 a 0, transitando de exploração para exploração.

Referência:
  Mirjalili, S., Mirjalili, S.M. & Lewis, A. (2014). Grey Wolf Optimizer.
  Advances in Engineering Software, 69, 46-61.
  https://doi.org/10.1016/j.advengsoft.2013.12.007
"""

import numpy as np
import time
import os

from classifier.neural_network import NeuralNetwork
from game.agents import NeuralNetworkAgent


# ─── Arquitetura da rede ──────────────────────────────────────────────────────

LAYER_SIZES = [27, 64, 32, 16, 3]

# ─── Hiperparâmetros ──────────────────────────────────────────────────────────

N_WOLVES          = 100
N_ITER            = 1000
N_RUNS            = 5
BOUNDS            = (-2.0, 2.0)
MAX_HOURS         = 8.5
STAGNATION_LIMIT  = 80
REINIT_RATIO      = 0.30


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

    Todos os lobos jogam na mesma instância (obstáculos compartilhados, posições
    y independentes). Um jogo fornece pontuações para todos os 100 lobos de uma vez.

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


# ─── Grey Wolf Optimizer ──────────────────────────────────────────────────────

class GreyWolfOptimizer:
    """Grey Wolf Optimizer para otimização de pesos de rede neural.

    Parâmetros
    ----------
    layer_sizes      : topologia da rede neural
    n_wolves         : tamanho da população (requisito: 100)
    n_iter           : máximo de iterações (requisito: 1000)
    n_runs           : jogos multi-jogador por avaliação de fitness
    bounds           : (lb, ub) para clipping dos pesos
    max_hours        : orçamento de treinamento em horas
    stagnation_limit : iterações sem melhoria para acionar reinicialização parcial
    reinit_ratio     : fração dos piores lobos a reinicializar na estagnação
    """

    def __init__(
        self,
        layer_sizes:      list  = LAYER_SIZES,
        n_wolves:         int   = N_WOLVES,
        n_iter:           int   = N_ITER,
        n_runs:           int   = N_RUNS,
        bounds:           tuple = BOUNDS,
        max_hours:        float = MAX_HOURS,
        stagnation_limit: int   = STAGNATION_LIMIT,
        reinit_ratio:     float = REINIT_RATIO,
    ):
        self.layer_sizes      = layer_sizes
        self.n_wolves         = n_wolves
        self.n_iter           = n_iter
        self.n_runs           = n_runs
        self.lb, self.ub      = bounds
        self.max_seconds      = max_hours * 3600
        self.stagnation_limit = stagnation_limit
        self.reinit_ratio     = reinit_ratio

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
        save_path:   str  = "gwo_weights.npy",
        print_every: int  = 10,
    ) -> tuple:
        """Executa o GWO até o limite de tempo ou de iterações.

        Retorna
        -------
        best_weights, best_score, history
        """
        print(f"[GWO] Rede: {self.layer_sizes} | Params: {self.n_params:,}")
        print(f"[GWO] Lobos={self.n_wolves} | Iter={self.n_iter} | "
              f"Runs={self.n_runs} | MaxTime={self.max_seconds/3600:.1f}h")
        print("[GWO] Avaliação: batch multi-jogador (todos os lobos em um jogo)")

        start_time = time.time()

        pop = _init_population_layerwise(self.layer_sizes, self.n_wolves, self.lb, self.ub)

        print("[GWO] Avaliando população inicial...")
        fitness = evaluate_population(pop, self.layer_sizes, GameClass, GameConfigClass,
                                      self.n_runs, render)

        sorted_idx = np.argsort(fitness)[::-1]
        alpha_pos  = pop[sorted_idx[0]].copy()
        alpha_fit  = float(fitness[sorted_idx[0]])
        beta_pos   = pop[sorted_idx[1]].copy()
        delta_pos  = pop[sorted_idx[2]].copy()

        self.best_score   = alpha_fit
        self.best_weights = alpha_pos.copy()
        self.history      = [self.best_score]
        np.save(save_path, self.best_weights)

        print(f"[GWO] Melhor fitness inicial: {self.best_score:.4f}\n")

        stagnation_count = 0

        for it in range(self.n_iter):

            if time.time() - start_time >= self.max_seconds:
                print(f"\n[GWO] Limite de tempo atingido na iteração {it+1}")
                break

            # `a` decai linearmente de 2 a 0 — governa a transição exploração→exploração
            a = 2.0 - 2.0 * (it / self.n_iter)

            for i in range(self.n_wolves):
                r1 = np.random.rand(self.n_params)
                A1 = 2 * a * r1 - a
                C1 = 2 * np.random.rand(self.n_params)
                D_alpha = np.abs(C1 * alpha_pos - pop[i])
                X1 = alpha_pos - A1 * D_alpha

                r1 = np.random.rand(self.n_params)
                A2 = 2 * a * r1 - a
                C2 = 2 * np.random.rand(self.n_params)
                D_beta = np.abs(C2 * beta_pos - pop[i])
                X2 = beta_pos - A2 * D_beta

                r1 = np.random.rand(self.n_params)
                A3 = 2 * a * r1 - a
                C3 = 2 * np.random.rand(self.n_params)
                D_delta = np.abs(C3 * delta_pos - pop[i])
                X3 = delta_pos - A3 * D_delta

                pop[i] = np.clip((X1 + X2 + X3) / 3.0, self.lb, self.ub).astype(np.float32)

            # Elitismo: lobo 0 sempre mantém a posição alpha atual
            pop[0] = alpha_pos.copy()

            fitness = evaluate_population(pop, self.layer_sizes, GameClass, GameConfigClass,
                                          self.n_runs, render)
            fitness[0] = alpha_fit  # fitness do alpha é conhecido — não re-médiar

            sorted_idx = np.argsort(fitness)[::-1]
            if fitness[sorted_idx[0]] > alpha_fit:
                old       = alpha_fit
                alpha_pos = pop[sorted_idx[0]].copy()
                alpha_fit = float(fitness[sorted_idx[0]])
                beta_pos  = pop[sorted_idx[1]].copy()
                delta_pos = pop[sorted_idx[2]].copy()

                self.best_score   = alpha_fit
                self.best_weights = alpha_pos.copy()
                np.save(save_path, self.best_weights)
                elapsed_h = (time.time() - start_time) / 3600
                print(f"[GWO] Iter {it+1:>5} ({elapsed_h:.2f}h) | NOVO MELHOR: {old:.4f} → {self.best_score:.4f} (salvo)")
                stagnation_count = 0
            else:
                beta_pos  = pop[sorted_idx[1]].copy()
                delta_pos = pop[sorted_idx[2]].copy()
                stagnation_count += 1

            # Reinicialização parcial por estagnação: reinicializa os piores lobos
            if stagnation_count >= self.stagnation_limit:
                n_reinit  = max(3, int(self.n_wolves * self.reinit_ratio))
                worst_idx = np.argsort(fitness)[:n_reinit]
                fresh     = _init_population_layerwise(self.layer_sizes, n_reinit, self.lb, self.ub)
                pop[worst_idx] = fresh
                elapsed_h = (time.time() - start_time) / 3600
                print(f"[GWO] Iter {it+1:>5} ({elapsed_h:.2f}h) | Estagnação ({stagnation_count} iters) → reinit {n_reinit} lobos")
                stagnation_count = 0

            self.history.append(self.best_score)

            if (it + 1) % print_every == 0:
                elapsed   = time.time() - start_time
                remaining = (self.max_seconds - elapsed) / 3600
                print(
                    f"Iter {it+1:>5}/{self.n_iter} | "
                    f"Melhor: {self.best_score:>8.4f} | "
                    f"Avg: {np.mean(fitness):>7.4f} | "
                    f"a: {a:.3f} | "
                    f"Restante: {remaining:.1f}h"
                )

        elapsed = time.time() - start_time
        print(f"\n[GWO] Concluído. Tempo: {elapsed/3600:.2f}h | Melhor fitness: {self.best_score:.4f}")

        return self.best_weights, self.best_score, self.history
