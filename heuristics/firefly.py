"""
Firefly Algorithm (FA)
======================
Otimização de pesos de rede neural para o jogo de sobrevivência.

Inspirado no comportamento bioluminescente de vaga-lumes (Yang, 2009).

Regras principais:
  1. Todos os vaga-lumes são unissexos — qualquer um pode ser atraído por outro.
  2. A atratividade é proporcional ao brilho (fitness); vaga-lumes mais escuros
     se movem em direção aos mais brilhantes.
  3. Se nenhum vaga-lume for mais brilhante, o vaga-lume faz um caminho aleatório.

Atualização de posição (vaga-lume i atraído por j, brilho[j] > brilho[i]):
  x_i = x_i + β₀·exp(−γ·r²_ij)·(x_j − x_i) + α·ε_i

onde:
  β₀  — atratividade base (distância zero)
  γ   — coeficiente de absorção de luz (controla o decaimento da atração)
  r_ij — distância Euclidiana entre vaga-lumes i e j
  α   — tamanho do passo de aleatorização (decai ao longo do tempo)
  ε_i — perturbação aleatória ~ Uniforme(−0.5, 0.5)

Em espaços de alta dimensão (n_params ≈ 4451), a comparação O(n²) é gerenciada
amostrando N_COMPARE=20 parceiros aleatórios por vaga-lume por iteração.

Referência:
  Yang, X.S. (2009). Firefly Algorithms for Multimodal Optimization.
  Stochastic Algorithms: Foundations and Applications (SAGA 2009).
  Springer. https://doi.org/10.1007/978-3-642-04944-6_14
"""

import numpy as np
import time
import os

from classifier.neural_network import NeuralNetwork
from game.agents import NeuralNetworkAgent


# ─── Arquitetura da rede ──────────────────────────────────────────────────────

LAYER_SIZES = [27, 64, 32, 16, 3]

# ─── Hiperparâmetros ──────────────────────────────────────────────────────────

N_FIREFLIES = 100
N_ITER      = 1000
N_RUNS      = 8
BETA0       = 1.0
GAMMA_FA    = 0.01
ALPHA_INIT  = 0.5
ALPHA_MIN   = 0.01
ALPHA_DECAY = 0.97
N_COMPARE   = 20
BOUNDS      = (-2.0, 2.0)
MAX_HOURS   = 8.5
WARM_START_FILE  = "firefly_weights.npy"
WARM_START_RATIO = 0.20
WARM_START_NOISE = 0.15


# ─── Inicialização da população ───────────────────────────────────────────────

def _init_population_layerwise(layer_sizes, n_agents, lb=-2.0, ub=2.0):
    """Inicialização He (Kaiming) por camada: std = sqrt(2 / fan_in) por camada.

    Garante diversidade real desde a iteração 0. A inicialização Xavier global
    com divisor n_params=4451 produzia std≈0.021, fazendo todas as redes agirem
    uniformemente aleatórias (saídas softmax ≈ [0.333, 0.333, 0.333]).
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

def evaluate_weights(
    weights:        np.ndarray,
    layer_sizes:    list,
    GameClass,
    GameConfigClass,
    n_runs:         int  = N_RUNS,
    render:         bool = False,
) -> float:
    """Avalia um vetor de pesos em n_runs episódios do jogo.

    Retorna fitness combinando média, mediana, consistência e bônus progressivos.
    """
    nn    = NeuralNetwork(layer_sizes)
    nn.set_weights(weights.astype(np.float32))
    agent = NeuralNetworkAgent(nn)

    config = GameConfigClass(num_players=1, fps=0 if not render else 60)
    scores = []
    for _ in range(n_runs):
        game = GameClass(config=config, render=render)
        while not game.all_players_dead():
            state  = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
        scores.append(game.players[0].score)

    scores       = np.array(scores)
    avg_score    = np.mean(scores)
    median_score = np.median(scores)
    robust_score = 0.7 * avg_score + 0.3 * median_score
    consistency  = 1.0 / (1.0 + np.std(scores) / max(avg_score, 1.0))
    fitness      = robust_score * (0.85 + 0.15 * consistency)

    if robust_score > 40:  fitness *= 1.15
    if robust_score > 70:  fitness *= 1.25
    if robust_score > 120: fitness *= 1.40

    return float(fitness)


# ─── Firefly Algorithm ────────────────────────────────────────────────────────

class FireflyAlgorithm:
    """Firefly Algorithm para otimização de pesos de rede neural.

    Parâmetros
    ----------
    layer_sizes  : topologia da rede neural
    n_fireflies  : tamanho da população (requisito: 100)
    n_iter       : máximo de iterações (requisito: 1000)
    n_runs       : episódios por avaliação de fitness
    beta0        : atratividade base (distância zero)
    gamma        : coeficiente de absorção de luz
    alpha_init   : tamanho inicial do passo de aleatorização
    alpha_min    : tamanho mínimo do passo de aleatorização
    alpha_decay  : decaimento multiplicativo de alpha por iteração
    n_compare    : número de parceiros aleatórios comparados por vaga-lume
    bounds       : (lb, ub) para clipping dos pesos
    max_hours    : orçamento de treinamento em horas
    """

    def __init__(
        self,
        layer_sizes: list  = LAYER_SIZES,
        n_fireflies: int   = N_FIREFLIES,
        n_iter:      int   = N_ITER,
        n_runs:      int   = N_RUNS,
        beta0:       float = BETA0,
        gamma:       float = GAMMA_FA,
        alpha_init:  float = ALPHA_INIT,
        alpha_min:   float = ALPHA_MIN,
        alpha_decay: float = ALPHA_DECAY,
        n_compare:   int   = N_COMPARE,
        bounds:      tuple = BOUNDS,
        max_hours:   float = MAX_HOURS,
    ):
        self.layer_sizes = layer_sizes
        self.n_fireflies = n_fireflies
        self.n_iter      = n_iter
        self.n_runs      = n_runs
        self.beta0       = beta0
        self.gamma       = gamma
        self.alpha       = alpha_init
        self.alpha_min   = alpha_min
        self.alpha_decay = alpha_decay
        self.n_compare   = n_compare
        self.lb, self.ub = bounds
        self.max_seconds = max_hours * 3600

        nn = NeuralNetwork(layer_sizes)
        self.n_params = nn.count_weights()

        self.history      = []
        self.best_weights = None
        self.best_score   = -np.inf

    def _initialize(self):
        """Inicializa a população com He por camada e warm-start opcional."""
        pop = _init_population_layerwise(self.layer_sizes, self.n_fireflies, self.lb, self.ub)

        if os.path.exists(WARM_START_FILE):
            try:
                best = np.load(WARM_START_FILE).astype(np.float32)
                if best.size == self.n_params:
                    n_warm = int(self.n_fireflies * WARM_START_RATIO)
                    noise  = np.random.normal(0, WARM_START_NOISE,
                                              (n_warm, self.n_params)).astype(np.float32)
                    pop[:n_warm] = np.clip(best[None, :] + noise, self.lb, self.ub)
                    print(f"[FA] Warm-start: {n_warm} vaga-lumes de '{WARM_START_FILE}'")
            except Exception as e:
                print(f"[FA] Warm-start ignorado: {e}")

        return pop

    def _attract(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Move o vaga-lume i em direção ao vaga-lume j mais brilhante.

        A atratividade decai com o quadrado da distância normalizada.
        """
        diff    = xj - xi
        r_sq    = float(np.dot(diff, diff)) / self.n_params  # distância² normalizada
        beta    = self.beta0 * np.exp(-self.gamma * r_sq)
        epsilon = np.random.uniform(-0.5, 0.5, self.n_params).astype(np.float32)
        x_new   = xi + beta * diff + self.alpha * epsilon
        return np.clip(x_new, self.lb, self.ub).astype(np.float32)

    def optimise(
        self,
        GameClass,
        GameConfigClass,
        render:      bool = False,
        save_path:   str  = "firefly_weights.npy",
        print_every: int  = 25,
    ) -> tuple:
        """Executa o Firefly Algorithm até o limite de tempo ou de iterações.

        Retorna
        -------
        best_weights, best_score, history
        """
        print(f"[FA] Rede: {self.layer_sizes} | Params: {self.n_params:,}")
        print(f"[FA] Vaga-lumes={self.n_fireflies} | Iter={self.n_iter} | "
              f"Runs={self.n_runs} | MaxTime={self.max_seconds/3600:.1f}h")

        start_time = time.time()

        pop = self._initialize()

        print("[FA] Avaliando população inicial...")
        fitness = np.array([
            evaluate_weights(pop[i], self.layer_sizes, GameClass, GameConfigClass, self.n_runs, render)
            for i in range(self.n_fireflies)
        ])

        best_idx          = np.argmax(fitness)
        self.best_score   = fitness[best_idx]
        self.best_weights = pop[best_idx].copy()
        self.history      = [self.best_score]

        print(f"[FA] Melhor fitness inicial: {self.best_score:.4f}\n")

        for it in range(self.n_iter):

            if time.time() - start_time >= self.max_seconds:
                print(f"\n[FA] Limite de tempo atingido na iteração {it+1}")
                break

            moved = np.zeros(self.n_fireflies, dtype=bool)

            for i in range(self.n_fireflies):
                partners = np.random.choice(
                    [j for j in range(self.n_fireflies) if j != i],
                    size=min(self.n_compare, self.n_fireflies - 1),
                    replace=False
                )

                attracted = False
                for j in partners:
                    if fitness[j] > fitness[i]:
                        x_new = self._attract(pop[i], pop[j])
                        f_new = evaluate_weights(x_new, self.layer_sizes,
                                                 GameClass, GameConfigClass, self.n_runs, render)
                        if f_new > fitness[i]:
                            pop[i]     = x_new
                            fitness[i] = f_new
                            moved[i]   = True
                            attracted  = True
                            break  # move em direção ao primeiro parceiro que melhora

                # Nenhum vizinho mais brilhante encontrado: caminho aleatório
                if not attracted:
                    epsilon = np.random.uniform(-0.5, 0.5, self.n_params).astype(np.float32)
                    x_rnd   = np.clip(pop[i] + self.alpha * epsilon, self.lb, self.ub).astype(np.float32)
                    f_rnd   = evaluate_weights(x_rnd, self.layer_sizes,
                                               GameClass, GameConfigClass, self.n_runs, render)
                    if f_rnd > fitness[i]:
                        pop[i]     = x_rnd
                        fitness[i] = f_rnd

            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_score:
                old = self.best_score
                self.best_score   = fitness[best_idx]
                self.best_weights = pop[best_idx].copy()
                np.save(save_path, self.best_weights)
                elapsed_h = (time.time() - start_time) / 3600
                print(f"[FA] Iter {it+1:>5} ({elapsed_h:.2f}h) | NOVO MELHOR: {old:.4f} → {self.best_score:.4f} (salvo)")

            self.history.append(self.best_score)

            if (it + 1) % print_every == 0:
                elapsed   = time.time() - start_time
                remaining = (self.max_seconds - elapsed) / 3600
                print(
                    f"Iter {it+1:>5}/{self.n_iter} | "
                    f"Melhor: {self.best_score:>8.4f} | "
                    f"Avg: {np.mean(fitness):>7.4f} | "
                    f"α: {self.alpha:.4f} | "
                    f"Movidos: {moved.sum():>3} | "
                    f"Restante: {remaining:.1f}h"
                )

        elapsed = time.time() - start_time
        print(f"\n[FA] Concluído. Tempo: {elapsed/3600:.2f}h | Melhor fitness: {self.best_score:.4f}")

        return self.best_weights, self.best_score, self.history
