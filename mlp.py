"""
mlp.py - Rede Neural MLP implementada do zero (apenas NumPy).

Arquitetura:  N entradas -> 8 neurônios ocultos (ReLU) -> 1 saída (Sigmoide)
Erro:         MSE (Erro Quadrático Médio)
Aprendizado:  Backpropagation + Gradiente Descendente (mini-batch)
"""
import numpy as np


class MLP:
    def __init__(self, n_entradas=4, n_ocultos=8, taxa_aprendizado=0.1, semente=42):
        self.lr = taxa_aprendizado
        rng = np.random.default_rng(semente)

        # Inicialização dos pesos
        # Camada oculta (ReLU): inicialização "He"  -> desvio = sqrt(2 / n_entradas)
        # Camada de saída (sigmoide): inicialização "Xavier" -> sqrt(1 / n_ocultos)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_entradas), size=(n_entradas, n_ocultos))
        self.b1 = np.zeros((1, n_ocultos))
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_ocultos), size=(n_ocultos, 1))
        self.b2 = np.zeros((1, 1))

    # ---------- Funções de ativação ----------
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivada(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoide(z):
        z = np.clip(z, -500, 500)          # evita overflow em exp()
        return 1.0 / (1.0 + np.exp(-z))

    # ---------- Passo para frente (forward) ----------
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1     # soma ponderada da camada oculta
        self.A1 = self.relu(self.Z1)        # ativação ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoide(self.Z2)    # saída: probabilidade entre 0 e 1
        return self.A2

    # ---------- Erro (MSE) ----------
    @staticmethod
    def mse(y_prev, y_real):
        return float(np.mean((y_prev - y_real) ** 2))

    # ---------- Backpropagation ----------
    def backward(self, y_real):
        n = y_real.shape[0]

        # 1) Derivada do erro em relação à saída:  dMSE/dŷ = 2(ŷ - y) / n
        dA2 = 2.0 * (self.A2 - y_real) / n
        # 2) Passa pela sigmoide:  σ'(z) = ŷ(1 - ŷ)
        dZ2 = dA2 * self.A2 * (1.0 - self.A2)
        # 3) Gradientes da camada de saída
        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)
        # 4) Propaga o erro para trás, até a camada oculta
        dA1 = dZ2 @ self.W2.T
        # 5) Passa pela ReLU:  ReLU'(z) = 1 se z > 0, senão 0
        dZ1 = dA1 * self.relu_derivada(self.Z1)
        # 6) Gradientes da camada oculta
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        # 7) Atualização dos pesos (gradiente descendente)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ---------- Treinamento ----------
    def treinar(self, X, y, X_val=None, y_val=None, epocas=100, tamanho_lote=64,
                semente=42, verbose=True):
        rng = np.random.default_rng(semente)
        historico = {"erro_treino": [], "erro_val": []}

        for epoca in range(1, epocas + 1):
            ordem = rng.permutation(len(X))            # embaralha a cada época
            for i in range(0, len(X), tamanho_lote):
                idx = ordem[i:i + tamanho_lote]
                self.forward(X[idx])
                self.backward(y[idx])

            erro_t = self.mse(self.prever_proba(X), y)
            historico["erro_treino"].append(erro_t)
            if X_val is not None:
                historico["erro_val"].append(self.mse(self.prever_proba(X_val), y_val))

            if verbose and (epoca == 1 or epoca % 10 == 0):
                msg = f"Época {epoca:3d} | MSE treino: {erro_t:.4f}"
                if X_val is not None:
                    msg += f" | MSE validação: {historico['erro_val'][-1]:.4f}"
                print(msg)
        return historico

    # ---------- Predição ----------
    def prever_proba(self, X):
        return self.forward(X)

    def prever(self, X, limiar=0.5):
        return (self.prever_proba(X) >= limiar).astype(int)

    # ---------- Salvar / carregar ----------
    def salvar(self, caminho, **extras):
        np.savez(caminho, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, **extras)

    @classmethod
    def carregar(cls, caminho):
        d = np.load(caminho, allow_pickle=True)
        rede = cls(n_entradas=d["W1"].shape[0], n_ocultos=d["W1"].shape[1])
        rede.W1, rede.b1, rede.W2, rede.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return rede, d
