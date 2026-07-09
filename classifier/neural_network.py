import numpy as np


class NeuralNetwork:
    """
    Configurable feedforward neural network.

    Default architecture for metaheuristics:
        Input(27) → Dense(N, LeakyReLU) → ... → Output(3, softmax)

    LeakyReLU (α=0.01) in the hidden layers prevents saturation: its
    derivative is always either 1 or 0.01, so ABC perturbations always
    produce measurable changes in the output (unlike tanh, which saturates
    for |z| > 2 when weights grow within the (-2, 2) range).

    Softmax in the output layer produces normalized probabilities; argmax
    provides deterministic action selection compatible with predict().

    Parameters
    ----------
    layer_sizes : list of int
        Layer sizes including input and output layers.
        Example: [27, 64, 32, 16, 3]
    """

    def __init__(self, layer_sizes: list):
        self.layer_sizes = layer_sizes
        self.num_layers  = len(layer_sizes) - 1

        self.weights = []
        self.biases  = []
        for i in range(self.num_layers):
            in_sz  = layer_sizes[i]
            out_sz = layer_sizes[i + 1]
            self.weights.append(np.zeros((in_sz, out_sz), dtype=np.float32))
            self.biases.append(np.zeros(out_sz, dtype=np.float32))

    @staticmethod
    def _leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-8)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.

        Hidden layers: LeakyReLU (α=0.01)
        Output layer: softmax

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, input_dim)

        Returns
        -------
        np.ndarray, shape (batch_size, output_dim)
        """
        a = x
        for i in range(self.num_layers):
            z = a.dot(self.weights[i]) + self.biases[i]
            if i < self.num_layers - 1:
                a = self._leaky_relu(z)   # hidden layers: LeakyReLU
            else:
                a = self._softmax(z)      # output layer: softmax
        return a

    def count_weights(self) -> int:
        """Return the total number of parameters (weights + biases)."""
        return sum(W.size + b.size
                   for W, b in zip(self.weights, self.biases))

    def get_weights(self) -> np.ndarray:
        """Return all parameters as a 1D vector."""
        flat = []
        for W, b in zip(self.weights, self.biases):
            flat.append(W.ravel())
            flat.append(b.ravel())
        return np.concatenate(flat)

    def set_weights(self, vector: np.ndarray) -> None:
        """
        Fill weights and biases from a 1D vector.

        Parameters
        ----------
        vector : np.ndarray, shape (n_params,)
        """
        expected = self.count_weights()
        if vector.size != expected:
            raise ValueError(
                f"Tamanho incorreto: esperado {expected}, recebido {vector.size}"
            )
        idx = 0
        for i in range(self.num_layers):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            self.weights[i] = vector[idx: idx + w_size].reshape(
                self.weights[i].shape).astype(np.float32)
            idx += w_size
            self.biases[i] = vector[idx: idx + b_size].reshape(
                self.biases[i].shape).astype(np.float32)
            idx += b_size