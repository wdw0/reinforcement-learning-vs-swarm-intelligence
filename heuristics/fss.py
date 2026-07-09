"""
Fish School Search (FSS)
========================
Otimização de pesos de rede neural para o jogo de sobrevivência.

Inspirado no comportamento coletivo de cardumes de peixes (Bastos-Filho et al., 2008).
Cada peixe tem posição x_i (vetor de pesos candidato) e peso w_i (ganho de fitness acumulado).

Três operadores por iteração:

1. Movimento individual (alimentação):
   Cada peixe dá um passo aleatório; se houver melhora, move e ganha peso:
     x_i_new = x_i + step_ind × Uniforme(0,1) × Uniforme(−1,1)
     Δf_i    = f(x_i_new) − f(x_i)
     se Δf_i > 0: aceitar, w_i += Δf_i / max(|Δf|)

2. Movimento instintivo coletivo:
   Todos os peixes se movem em direção à média ponderada dos movimentos bem-sucedidos:
     I = Σ(Δx_i × Δf_i) / Σ(Δf_i)   (apenas peixes bem-sucedidos contribuem)
     x_i += I

3. Movimento volitivo coletivo (expansão/contração do cardume):
   Baricentro B = Σ(x_i × w_i) / Σ(w_i)
   Se peso total aumentou (cardume comendo): mover peixes em direção a B (explorar)
   Se peso total diminuiu (cardume com fome): mover peixes para longe de B (explorar)
     x_i ± step_vol × Uniforme(0,1) × (x_i − B) / |x_i − B|

step_ind e step_vol decaem linearmente de seus valores iniciais a quase zero.

Referência:
  Bastos-Filho, C.J.A., Lima Neto, F.B., Lins, A.J.C.C., Nascimento, A.I.S.
  e Lima, M.P. (2008). A Novel Swarm Intelligence Based Optimization Method:
  Fish School Search. Proc. IEEE Int. Conf. on Hybrid Intelligent Systems.
  https://doi.org/10.1109/HIS.2008.31
"""

import numpy as np
import time
import os

from classifier.neural_network import NeuralNetwork
from game.agents import NeuralNetworkAgent


# ─── Arquitetura da rede ──────────────────────────────────────────────────────

LAYER_SIZES = [27, 64, 32, 16, 3]

# ─── Hiperparâmetros ──────────────────────────────────────────────────────────

N_FISH           = 100
N_ITER           = 1000
N_RUNS           = 5
STEP_IND_INIT    = 0.5
STEP_IND_FINAL   = 0.001
STEP_VOL_INIT    = 0.1
STEP_VOL_FINAL   = 0.005
W_MAX            = 2500.0
W_INIT           = W_MAX / 2
BOUNDS           = (-2.0, 2.0)
MAX_HOURS        = 8.5
WARM_START_FILE  = "fss_weights.npy"
WARM_START_RATIO = 0.10
WARM_START_NOISE = 0.20


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


# ─── Funções de Fitness ───────────────────────────────────────────────────────

def evaluate_population(
    pop:            np.ndarray,
    layer_sizes:    list,
    GameClass,
    GameConfigClass,
    n_runs:         int  = N_RUNS,
    render:         bool = False,
) -> np.ndarray:
    """Avalia todos os N agentes simultaneamente usando o modo multi-jogador do jogo.

    Todos os peixes jogam na mesma instância (obstáculos compartilhados, posições
    y independentes). Um jogo fornece pontuações para todos os 100 peixes ao mesmo tempo.

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
        fitness[i]   = 0.7 * avg_score + 0.3 * median_score

    return fitness


def evaluate_weights(
    weights:        np.ndarray,
    layer_sizes:    list,
    GameClass,
    GameConfigClass,
    n_runs:         int  = N_RUNS,
    render:         bool = False,
) -> float:
    """Avalia um único agente — usado apenas pelo operador de reprodução."""
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
    avg_score    = float(np.mean(scores))
    median_score = float(np.median(scores))
    return 0.7 * avg_score + 0.3 * median_score


# ─── Fish School Search ───────────────────────────────────────────────────────

class FishSchoolSearch:
    """Fish School Search para otimização de pesos de rede neural.

    Parâmetros
    ----------
    layer_sizes     : topologia da rede neural
    n_fish          : tamanho do cardume (requisito: 100)
    n_iter          : máximo de iterações (requisito: 1000)
    n_runs          : episódios de jogo por avaliação de fitness
    step_ind_init   : tamanho inicial do passo individual
    step_ind_final  : tamanho final do passo individual
    step_vol_init   : tamanho inicial do passo volitivo
    step_vol_final  : tamanho final do passo volitivo
    w_init          : peso inicial de cada peixe
    w_max           : peso máximo absoluto
    bounds          : (lb, ub) para clipping dos pesos
    max_hours       : orçamento de treinamento em horas
    """

    def __init__(
        self,
        layer_sizes:    list  = LAYER_SIZES,
        n_fish:         int   = N_FISH,
        n_iter:         int   = N_ITER,
        n_runs:         int   = N_RUNS,
        step_ind_init:  float = STEP_IND_INIT,
        step_ind_final: float = STEP_IND_FINAL,
        step_vol_init:  float = STEP_VOL_INIT,
        step_vol_final: float = STEP_VOL_FINAL,
        w_init:         float = W_INIT,
        w_max:          float = W_MAX,
        bounds:         tuple = BOUNDS,
        max_hours:      float = MAX_HOURS,
    ):
        self.layer_sizes    = layer_sizes
        self.n_fish         = n_fish
        self.n_iter         = n_iter
        self.n_runs         = n_runs
        self.step_ind_init  = step_ind_init
        self.step_ind_final = step_ind_final
        self.step_vol_init  = step_vol_init
        self.step_vol_final = step_vol_final
        self.w_init         = w_init
        self.w_max          = w_max
        self.lb, self.ub    = bounds
        self.max_seconds    = max_hours * 3600

        nn = NeuralNetwork(layer_sizes)
        self.n_params = nn.count_weights()

        self.history      = []
        self.best_weights = None
        self.best_score   = -np.inf

    def _step(self, it: int, init: float, final: float) -> float:
        """Decaimento linear de init a final ao longo de n_iter iterações."""
        progress = it / max(self.n_iter - 1, 1)
        return init + (final - init) * progress

    def _initialize(self):
        """Inicializa posições com He por camada e warm-start opcional."""
        pos = _init_population_layerwise(self.layer_sizes, self.n_fish, self.lb, self.ub)

        if os.path.exists(WARM_START_FILE):
            try:
                best = np.load(WARM_START_FILE).astype(np.float32)
                if best.size == self.n_params:
                    n_warm = int(self.n_fish * WARM_START_RATIO)
                    noise  = np.random.normal(0, WARM_START_NOISE,
                                              (n_warm, self.n_params)).astype(np.float32)
                    pos[:n_warm] = np.clip(best[None, :] + noise, self.lb, self.ub)
                    print(f"[FSS] Warm-start: {n_warm} peixes de '{WARM_START_FILE}'")
            except Exception as e:
                print(f"[FSS] Warm-start ignorado: {e}")

        weights = np.full(self.n_fish, self.w_init, dtype=np.float32)
        return pos, weights

    def _breeding(self, pos, fitness, w, GameClass, GameConfigClass, render):
        """Reprodução: dois peixes fortes geram um filho; substitui o peixe mais fraco."""
        strong = np.where(w >= self.w_max * 0.9)[0]
        if len(strong) < 2:
            return pos, fitness, w
        i = np.random.choice(strong)
        scores = []
        for j in strong:
            if j == i:
                continue
            dist = max(float(np.linalg.norm(pos[i] - pos[j])), 1e-9)
            scores.append((w[j] / dist, j))
        if not scores:
            return pos, fitness, w
        j         = max(scores, key=lambda x: x[0])[1]
        child_pos = ((pos[i] + pos[j]) / 2.0).astype(np.float32)
        child_fit = evaluate_weights(child_pos, self.layer_sizes,
                                     GameClass, GameConfigClass, self.n_runs, render)
        weakest        = int(np.argmin(w))
        pos[weakest]   = child_pos
        w[weakest]     = (w[i] + w[j]) / 2.0
        fitness[weakest] = child_fit
        return pos, fitness, w

    def optimise(
        self,
        GameClass,
        GameConfigClass,
        render:      bool = False,
        save_path:   str  = "fss_weights.npy",
        print_every: int  = 25,
    ) -> tuple:
        """Executa o Fish School Search até o limite de tempo ou de iterações.

        Retorna
        -------
        best_weights, best_score, history
        """
        print(f"[FSS] Rede: {self.layer_sizes} | Params: {self.n_params:,}")
        print(f"[FSS] Peixes={self.n_fish} | Iter={self.n_iter} | "
              f"Runs={self.n_runs} | MaxTime={self.max_seconds/3600:.1f}h")
        print("[FSS] Avaliação: batch multi-jogador (todos os peixes em um jogo)")

        start_time = time.time()

        pos, w = self._initialize()

        print("[FSS] Avaliando população inicial...")
        fitness = evaluate_population(pos, self.layer_sizes, GameClass, GameConfigClass,
                                      self.n_runs, render)

        best_idx          = np.argmax(fitness)
        self.best_score   = fitness[best_idx]
        self.best_weights = pos[best_idx].copy()
        self.history      = [self.best_score]

        print(f"[FSS] Melhor fitness inicial: {self.best_score:.4f}\n")

        w_total_prev = float(np.sum(w))

        for it in range(self.n_iter):

            if time.time() - start_time >= self.max_seconds:
                print(f"\n[FSS] Limite de tempo atingido na iteração {it+1}")
                break

            step_ind = self._step(it, self.step_ind_init, self.step_ind_final)
            step_vol = self._step(it, self.step_vol_init, self.step_vol_final)

            # ── 1. Movimento individual (alimentação) ─────────────────────────
            # Gerar todos os candidatos primeiro, depois avaliar em batch.
            delta_x  = np.zeros_like(pos)
            delta_f  = np.zeros(self.n_fish, dtype=np.float32)
            new_pos  = np.zeros_like(pos)
            new_fitness = fitness.copy()
            candidates  = np.zeros_like(pos)

            for i in range(self.n_fish):
                direction  = np.random.uniform(-1, 1, self.n_params).astype(np.float32)
                magnitude  = np.random.uniform(0, 1, self.n_params).astype(np.float32)
                step       = step_ind * magnitude * direction
                candidates[i] = np.clip(pos[i] + step, self.lb, self.ub).astype(np.float32)

            cand_fitness = evaluate_population(candidates, self.layer_sizes,
                                               GameClass, GameConfigClass, self.n_runs, render)

            for i in range(self.n_fish):
                if cand_fitness[i] > fitness[i]:
                    new_pos[i]     = candidates[i]
                    new_fitness[i] = cand_fitness[i]
                    delta_x[i]     = candidates[i] - pos[i]
                    delta_f[i]     = cand_fitness[i] - fitness[i]
                else:
                    new_pos[i] = pos[i]

            # Alimentação: atualizar pesos dos peixes com melhora
            max_df = np.max(np.abs(delta_f))
            if max_df > 1e-8:
                for i in range(self.n_fish):
                    if delta_f[i] > 0:
                        w[i] = min(w[i] + delta_f[i] / max_df, self.w_max)

            pos     = new_pos
            fitness = new_fitness

            # ── Atualização do melhor global ──────────────────────────────────
            # Deve ocorrer ANTES dos movimentos coletivos, pois pos[best_idx] após
            # os movimentos instintivo/volitivo não corresponde mais ao fitness medido.
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_score:
                old = self.best_score
                self.best_score   = fitness[best_idx]
                self.best_weights = pos[best_idx].copy()
                np.save(save_path, self.best_weights)
                elapsed_h = (time.time() - start_time) / 3600
                print(f"[FSS] Iter {it+1:>5} ({elapsed_h:.2f}h) | NOVO MELHOR: {old:.4f} → {self.best_score:.4f} (salvo)")

            # ── 2. Movimento instintivo coletivo ──────────────────────────────
            sum_df = np.sum(np.abs(delta_f))
            if sum_df > 1e-8:
                instinct = np.sum(delta_x * delta_f[:, None], axis=0) / sum_df
                pos = np.clip(pos + instinct, self.lb, self.ub).astype(np.float32)

            # ── 3. Movimento volitivo coletivo ────────────────────────────────
            # Compara w_total com a iteração ANTERIOR para decidir contração/expansão.
            w_total_curr = float(np.sum(w))
            barycenter   = np.sum(pos * w[:, None], axis=0) / w_total_curr

            # Fator aleatório por peixe — garante diversidade no passo volitivo
            rand_factors = np.random.rand(self.n_fish, 1).astype(np.float32)
            if w_total_curr > w_total_prev:
                # Cardume bem alimentado → contrair em direção ao baricentro (exploração)
                pos = np.clip(
                    pos - rand_factors * step_vol * (pos - barycenter), self.lb, self.ub
                ).astype(np.float32)
            else:
                # Cardume com fome → expandir para longe do baricentro (exploração)
                pos = np.clip(
                    pos + rand_factors * step_vol * (pos - barycenter), self.lb, self.ub
                ).astype(np.float32)

            w_total_prev = w_total_curr

            # ── 4. Reprodução ─────────────────────────────────────────────────
            pos, fitness, w = self._breeding(pos, fitness, w, GameClass, GameConfigClass, render)

            self.history.append(self.best_score)

            if (it + 1) % print_every == 0:
                elapsed   = time.time() - start_time
                remaining = (self.max_seconds - elapsed) / 3600
                print(
                    f"Iter {it+1:>5}/{self.n_iter} | "
                    f"Melhor: {self.best_score:>8.4f} | "
                    f"Avg: {np.mean(fitness):>7.4f} | "
                    f"PesoMédio: {np.mean(w):.2f} | "
                    f"StepInd: {step_ind:.4f} | "
                    f"Restante: {remaining:.1f}h"
                )

        elapsed = time.time() - start_time
        print(f"\n[FSS] Concluído. Tempo: {elapsed/3600:.2f}h | Melhor fitness: {self.best_score:.4f}")

        return self.best_weights, self.best_score, self.history
