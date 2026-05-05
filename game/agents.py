import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable
from game.core import GameConfig
import random

from classifier.neural_network import NeuralNetwork


class Agent(ABC):
    """Interface para todos os agentes de jogo."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Retorna a ação a ser executada baseado no estado atual."""
        pass


class HumanAgent(Agent):
    """Agente controlado manualmente (para debugging/jogador humano)."""
    def predict(self, state: np.ndarray) -> int:
        # Será sobrescrito pelo controlador de input no modo manual
        return 0


class NeuralNetworkAgent(Agent):
    """Agente baseado em rede neural feedforward, configurável."""
    def __init__(
        self,
        model: NeuralNetwork,
        action_space: List[int] = None,
        config: GameConfig = None,
        epsilon: float = 0.0,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = None
    ):
        self.model = model
        self.action_space = action_space if action_space is not None else [0, 1, 2]
        self.config = config
        self.epsilon = epsilon
        self.preprocess_fn = preprocess_fn or (lambda s: s.flatten().astype(np.float32))

    def predict(self, state: np.ndarray) -> int:
        x = self.preprocess_fn(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        logits = self.model.forward(x[None, :])
        idx = int(np.argmax(logits, axis=1)[0])
        return self.action_space[idx]


class QLearningAgent(Agent):
    """
    Q-Learning agent using linear function approximation.

    Delegates to heuristics.q_learning.QLearningAgent internally,
    but exposes the standard Agent interface expected by the game loop.

    This thin wrapper allows QLearningAgent to be used anywhere an
    Agent is expected (e.g., human_play.py, Graph.py comparisons).
    """

    def __init__(self, q_agent):
        """
        Parameters
        ----------
        q_agent : heuristics.q_learning.QLearningAgent
            A fully constructed (and optionally trained) Q-learning agent.
        """
        self._q = q_agent

    def predict(self, state: np.ndarray) -> int:
        return self._q.predict(state)
