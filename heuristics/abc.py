import numpy as np
import time
from classifier.neural_network import NeuralNetwork
from game.agents import NeuralNetworkAgent


# ─── Hyperparameters ─────────────────────────────────────────────────────────

LAYER_SIZES = [27, 32, 16, 3]   # same as reference agents (assignment spec)
N_BEES      = 100                # population size (assignment requirement)
N_ITER      = 1000               # max iterations (assignment requirement)
LIMIT       = 500                # abandonment counter threshold
N_RUNS      = 3                  # game episodes per fitness evaluation
BOUNDS      = (-1.0, 1.0)       # weight search space
K_CONST     = 8                  # iABC memory board duration constant
                                  # K=2^3=8, selected from Table 2 of paper


# ─── Fitness Function ─────────────────────────────────────────────────────────

def evaluate_fitness(
    population:     np.ndarray,
    layer_sizes:    list,
    GameClass,
    GameConfigClass,
    n_runs:         int = N_RUNS,
    render:         bool = False,
) -> np.ndarray:
    """
    Evaluate a population of weight vectors by playing the game.

    All bees play simultaneously as parallel players in the same game
    instance, matching the approach used by the Bat Algorithm for fair
    comparison. Fitness = average score over n_runs episodes.

    Parameters
    ----------
    population     : np.ndarray, shape (n_bees, n_params)
    layer_sizes    : neural network topology
    GameClass      : SurvivalGame class
    GameConfigClass: GameConfig class
    n_runs         : number of game episodes per evaluation
    render         : render Pygame window

    Returns
    -------
    fitness : np.ndarray, shape (n_bees,) — average scores (higher = better)
    """
    n_bees = population.shape[0]
    config = GameConfigClass(num_players=n_bees, fps=0 if not render else 60)

    # Build one neural network agent per bee
    agents = []
    for weights in population:
        nn = NeuralNetwork(layer_sizes)
        nn.set_weights(weights)
        agents.append(NeuralNetworkAgent(nn))

    total_scores = np.zeros(n_bees, dtype=np.float64)

    for _ in range(n_runs):
        game = GameClass(config=config, render=render)
        while not game.all_players_dead():
            actions = [
                agents[i].predict(game.get_state(i, include_internals=True))
                if game.players[i].alive else 0
                for i in range(n_bees)
            ]
            game.update(actions)
            if render:
                game.render_frame()
        for i, player in enumerate(game.players):
            total_scores[i] += player.score

    return total_scores / n_runs


def _fitness_value(score: float) -> float:
    """
    Convert game score to ABC fitness value (Equation 3 of paper).

    fit_i = 1 / (1 + |f_i|)  if f_i < 0
    fit_i = 1 + f_i           if f_i >= 0

    Since game scores are always >= 0, fitness = 1 + score.
    Higher fitness = better solution.
    """
    if score >= 0:
        return 1.0 + score
    return 1.0 / (1.0 + abs(score))


# ─── ABC / iABC Algorithm ─────────────────────────────────────────────────────

class ArtificialBeeColony:
    """
    Improved Artificial Bee Colony (iABC) optimiser.

    Optimises neural network weights to maximise game score.
    Implements the iABC improvement from Kiran & Babalik (2014):
    memory board for informed neighbourhood selection by onlooker bees.

    Parameters
    ----------
    layer_sizes    : neural network topology, e.g. [27, 32, 16, 3]
    n_bees         : number of employed bees (= number of onlooker bees)
    n_iter         : maximum number of iterations
    limit          : abandonment counter threshold before scout phase
    n_runs         : game episodes per fitness evaluation
    bounds         : (min, max) weight bounds
    k_const        : iABC memory board duration constant K
    use_iabc       : if True, use iABC improvement; if False, use basic ABC
    """

    def __init__(
        self,
        layer_sizes: list  = LAYER_SIZES,
        n_bees:      int   = N_BEES,
        n_iter:      int   = N_ITER,
        limit:       int   = LIMIT,
        n_runs:      int   = N_RUNS,
        bounds:      tuple = BOUNDS,
        k_const:     float = K_CONST,
        use_iabc:    bool  = True,
    ):
        self.layer_sizes = layer_sizes
        self.n_bees      = n_bees
        self.n_iter      = n_iter
        self.limit       = limit
        self.n_runs      = n_runs
        self.lb, self.ub = bounds
        self.k_const     = k_const
        self.use_iabc    = use_iabc

        # Determine number of weight parameters from network topology
        nn = NeuralNetwork(layer_sizes)
        self.n_params = nn.count_weights()

        self.history       = []   # best fitness per iteration
        self.best_weights  = None
        self.best_score    = -np.inf

    def _random_solution(self) -> np.ndarray:
        """Generate a random weight vector within bounds."""
        return np.random.uniform(self.lb, self.ub, self.n_params).astype(np.float32)

    def _perturb(
        self,
        source:     np.ndarray,
        neighbour:  np.ndarray,
    ) -> np.ndarray:
     
        candidate = source.copy()
        j         = np.random.randint(self.n_params)   # random dimension
        phi       = np.random.uniform(-1.0, 1.0)
        candidate[j] = source[j] + phi * (source[j] - neighbour[j])
        candidate[j] = np.clip(candidate[j], self.lb, self.ub)
        return candidate

    def _build_memory_board(
        self,
        population: np.ndarray,
        fitnesses:  np.ndarray,
        board:      dict,
        iteration:  int,
    ) -> dict:
    
        avg_fitness = np.mean(fitnesses)

        for i, fit in enumerate(fitnesses):
            if fit > avg_fitness:
                # Waiting time proportional to fitness (Equation 6)
                duration = int(self.k_const * fit)
                expiry   = iteration + max(1, duration)
                # Only update if not already on board or extends stay
                if i not in board or board[i] < expiry:
                    board[i] = expiry

        # Remove expired entries
        board = {i: exp for i, exp in board.items() if exp > iteration}
        return board

    def optimise(
        self,
        GameClass,
        GameConfigClass,
        render:      bool = False,
        save_path:   str  = "abc_weights.npy",
        print_every: int  = 10,
    ) -> tuple:
        """
        Run the iABC (or basic ABC) optimisation.

        Parameters
        ----------
        GameClass      : SurvivalGame class
        GameConfigClass: GameConfig class
        render         : render Pygame window (slow)
        save_path      : file to save best weights
        print_every    : console log interval (iterations)

        Returns
        -------
        best_weights : np.ndarray — best weight vector found
        best_score   : float — best average game score
        history      : list of float — best score per iteration
        """
        algo_name = "iABC" if self.use_iabc else "ABC"
        print(f"[{algo_name}] Starting optimisation")
        print(f"[{algo_name}] Bees={self.n_bees} | Iter={self.n_iter} | "
              f"Limit={self.limit} | Params={self.n_params:,}")

        # ── Initialisation ────────────────────────────────────────────────────
        # Random initial population (Equation 1)
        population = np.array([
            self._random_solution() for _ in range(self.n_bees)
        ])
        abandon_counters = np.zeros(self.n_bees, dtype=np.int32)

        # Evaluate initial fitness
        scores   = evaluate_fitness(
            population, self.layer_sizes, GameClass, GameConfigClass,
            self.n_runs, render
        )
        fitnesses = np.array([_fitness_value(s) for s in scores])

        # Track global best
        best_idx         = np.argmax(scores)
        self.best_score  = scores[best_idx]
        self.best_weights = population[best_idx].copy()
        self.history      = [self.best_score]

        # iABC memory board: {bee_index: expiry_iteration}
        memory_board: dict = {}

        # ── Main Loop ─────────────────────────────────────────────────────────
        for iteration in range(1, self.n_iter + 1):
            iter_start = time.time()

            # ── Employed Bee Phase ────────────────────────────────────────────
            # Each employed bee tries to improve its own solution
            for i in range(self.n_bees):
                # Pick a random neighbour (different from i)
                k = np.random.choice([x for x in range(self.n_bees) if x != i])
                candidate     = self._perturb(population[i], population[k])
                cand_scores   = evaluate_fitness(
                    candidate[None, :], self.layer_sizes,
                    GameClass, GameConfigClass, self.n_runs, render
                )
                cand_score    = cand_scores[0]
                cand_fitness  = _fitness_value(cand_score)

                if cand_fitness >= fitnesses[i]:
                    # Accept improvement
                    population[i]        = candidate
                    scores[i]            = cand_score
                    fitnesses[i]         = cand_fitness
                    abandon_counters[i]  = 0
                else:
                    abandon_counters[i] += 1

            # ── iABC: Update memory board after employed bee phase ─────────────
            if self.use_iabc:
                memory_board = self._build_memory_board(
                    population, fitnesses, memory_board, iteration
                )
                board_indices = list(memory_board.keys())

            # ── Onlooker Bee Phase ────────────────────────────────────────────
            # Select sources via roulette wheel; improve using neighbour
            # from memory board (iABC) or random (basic ABC)
            total_fitness = fitnesses.sum()
            probs         = fitnesses / total_fitness   # Equation 4

            for _ in range(self.n_bees):
                # Roulette wheel selection of employed bee to improve
                i = np.random.choice(self.n_bees, p=probs)

                # Neighbour selection: memory board (iABC) or random (ABC)
                if self.use_iabc and len(board_indices) > 0:
                    # Pick neighbour from memory board, excluding i if possible
                    candidates = [b for b in board_indices if b != i]
                    if len(candidates) == 0:
                        candidates = board_indices
                    k = np.random.choice(candidates)
                else:
                    # Basic ABC: fully random neighbour selection
                    k = np.random.choice([x for x in range(self.n_bees) if x != i])

                candidate    = self._perturb(population[i], population[k])
                cand_scores  = evaluate_fitness(
                    candidate[None, :], self.layer_sizes,
                    GameClass, GameConfigClass, self.n_runs, render
                )
                cand_score   = cand_scores[0]
                cand_fitness = _fitness_value(cand_score)

                if cand_fitness >= fitnesses[i]:
                    population[i]        = candidate
                    scores[i]            = cand_score
                    fitnesses[i]         = cand_fitness
                    abandon_counters[i]  = 0
                else:
                    abandon_counters[i] += 1

            # ── Scout Bee Phase ───────────────────────────────────────────────
            # Bees that failed to improve for LIMIT steps become scouts
            for i in range(self.n_bees):
                if abandon_counters[i] >= self.limit:
                    population[i]        = self._random_solution()
                    new_scores           = evaluate_fitness(
                        population[i][None, :], self.layer_sizes,
                        GameClass, GameConfigClass, self.n_runs, render
                    )
                    scores[i]            = new_scores[0]
                    fitnesses[i]         = _fitness_value(scores[i])
                    abandon_counters[i]  = 0

            # ── Global best update ────────────────────────────────────────────
            iter_best_idx = np.argmax(scores)
            if scores[iter_best_idx] > self.best_score:
                self.best_score   = scores[iter_best_idx]
                self.best_weights = population[iter_best_idx].copy()
                np.save(save_path, self.best_weights)
                print(f"[{algo_name}] New best → {self.best_score:.2f} "
                      f"(saved to '{save_path}')")

            self.history.append(self.best_score)

            if iteration % print_every == 0:
                iter_time = time.time() - iter_start
                print(
                    f"Iter {iteration:>5}/{self.n_iter} | "
                    f"Best: {self.best_score:>8.2f} | "
                    f"Iter best: {scores[iter_best_idx]:>8.2f} | "
                    f"Avg: {np.mean(scores):>7.2f} | "
                    f"Scouts: {np.sum(abandon_counters >= self.limit):>2} | "
                    f"Board: {len(memory_board):>3} | "
                    f"Time/iter: {iter_time:.1f}s"
                )

        print(f"\n[{algo_name}] Optimisation complete.")
        print(f"[{algo_name}] Best score: {self.best_score:.2f}")
        return self.best_weights, self.best_score, self.history
