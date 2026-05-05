import numpy as np

class NeuralNetwork:
    """
    Rede neural feedforward configurável para classificação.

    Parâmetros
    ----------
    layer_sizes : list of int
        Tamanhos das camadas, incluindo dimensão de entrada e saída. Exemplo: [128, 64, 3].
    """
    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        # Inicializa pesos e vieses com zeros; serão configurados via set_weights
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            in_sz = layer_sizes[i]
            out_sz = layer_sizes[i+1]
            # pesos: matriz de forma (in_sz, out_sz)
            self.weights.append(np.zeros((in_sz, out_sz), dtype=np.float32))
            # vieses: vetor de tamanho (out_sz,)
            self.biases.append(np.zeros((out_sz,), dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propaga entrada pela rede.

        Parameters
        ----------
        x : np.ndarray, shape (batch_size, input_dim)
            Dados de entrada.

        Returns
        -------
        np.ndarray, shape (batch_size, output_dim)
            Saída bruta (logits).
        """
        a = x
        for i in range(self.num_layers):
            z = a.dot(self.weights[i]) + self.biases[i]
            if i < self.num_layers - 1:
                # Função de ativação para camadas ocultas: tanh
                a = np.tanh(z)
            else:
                # Camada de saída: identidade (logits)
                a = z
        return a

    def count_weights(self) -> int:
        """
        Retorna o número total de parâmetros (pesos + vieses).
        """
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total

    def get_weights(self) -> np.ndarray:
        """
        Retorna todos os parâmetros em um vetor 1D.
        """
        flat = []
        for W, b in zip(self.weights, self.biases):
            flat.append(W.ravel())
            flat.append(b.ravel())
        return np.concatenate(flat)

    def set_weights(self, vector: np.ndarray) -> None:
        """
        Preenche pesos e vieses a partir de um vetor 1D.

        Parameters
        ----------
        vector : np.ndarray, shape (n_params,)
            Vetor contendo todos os parâmetros na ordem definida em get_weights.
        """
        expected = self.count_weights()
        if vector.size != expected:
            raise ValueError(f"Tamanho de vetor incorreto: esperado {expected}, recebido {vector.size}")
        idx = 0
        for i in range(self.num_layers):
            W = self.weights[i]
            b = self.biases[i]
            w_size = W.size
            b_size = b.size
            # Extrai segmento para W
            W_flat = vector[idx: idx + w_size]
            self.weights[i] = W_flat.reshape(W.shape)
            idx += w_size
            # Extrai segmento para b
            b_flat = vector[idx: idx + b_size]
            self.biases[i] = b_flat.reshape(b.shape)
            idx += b_size
