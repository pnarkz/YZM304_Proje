# ─────────────────────────────────────────────────────────────────────────────
# src/models_numpy.py  —  NumPy MLP sınıfları (sıfırdan implementasyon)
# YZM304 Derin Öğrenme | Pima Indians Diabetes MLP
#
# Sınıf hiyerarşisi:
#   _BaseMLPNumpy
#   ├── MLP_1Hidden   (1 gizli katman)
#   └── MLP_2Hidden   (2 gizli katman)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.config import RANDOM_SEED


# ── Ortak taban sınıf ─────────────────────────────────────────────────────────
class _BaseMLPNumpy:
    """
    NumPy MLP alt sınıfları için paylaşılan metotlar.
    Alt sınıflar yalnızca mimariye özgü metotları (_initialize_parameters,
    _forward, _compute_cost, _backward, _update, fit) uygular.
    """

    # ── Private: sigmoid ──────────────────────────────────────────────────────
    @staticmethod
    def _sigmoid(Z):
        Z = np.clip(Z, -500, 500)  # numerik stabilite — overflow önlemi
        return 1.0 / (1.0 + np.exp(-Z))

    # ── Public: tahmin ────────────────────────────────────────────────────────
    def predict(self, X, threshold=0.5):
        """0 veya 1 ikili tahmin."""
        A_out, _ = self._forward(X)
        return (A_out > threshold).astype(int).flatten()

    def predict_proba(self, X):
        """Tahmin olasılıkları."""
        A_out, _ = self._forward(X)
        return A_out.flatten()

    # ── Public: skor ──────────────────────────────────────────────────────────
    def score(self, X, Y):
        """Doğruluk (accuracy) skoru."""
        return accuracy_score(Y.flatten(), self.predict(X))

    # ── Public: öğrenme eğrisi ────────────────────────────────────────────────
    def plot_learning_curve(self, title='Öğrenme Eğrisi'):
        plt.figure(figsize=(9, 4))
        plt.plot(self.cost_history, label='Train Loss', color='steelblue', lw=1.5)
        if self.val_cost_history:
            plt.plot(self.val_cost_history, label='Val Loss',
                     color='coral', linestyle='--', lw=1.5)
        plt.xlabel('Adım (Epoch)')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# ── 1 Gizli Katmanlı Model ────────────────────────────────────────────────────
class MLP_1Hidden(_BaseMLPNumpy):
    """
    1 Gizli Katmanlı MLP — NumPy.
    Mimari: Input(n_x) → Dense(n_h, tanh/sigmoid) → Output(n_y, sigmoid)
    Loss:   Binary Cross-Entropy + opsiyonel L2
    Opt:    SGD (tam-batch veya mini-batch, momentum=0)
    """

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self, n_x, n_h, n_y=1,
                 learning_rate=0.1,
                 n_steps=1000,
                 hidden_activation='tanh',
                 l2_lambda=0.0,
                 batch_size=None,
                 print_cost=False,
                 print_every=200):
        """
        Parametreler
        ------------
        n_x              : Giriş özellik sayısı
        n_h              : Gizli katman nöron sayısı
        n_y              : Çıkış boyutu (ikili sınıflandırma: 1)
        learning_rate    : SGD öğrenme hızı (η)
        n_steps          : Epoch sayısı
        hidden_activation: 'tanh' veya 'sigmoid'
        l2_lambda        : L2 regularizasyon katsayısı (0 = kapalı)
        batch_size       : Mini-batch boyutu (None = tam-batch)
        print_cost       : Eğitim sırasında kayıp değerini yazdır
        print_every      : Kaç adımda bir yazdırılacak
        """
        self.n_x               = n_x
        self.n_h               = n_h
        self.n_y               = n_y
        self.lr                = learning_rate
        self.n_steps           = n_steps
        self.hidden_activation = hidden_activation
        self.l2_lambda         = l2_lambda
        self.batch_size        = batch_size
        self.print_cost        = print_cost
        self.print_every       = print_every
        self.parameters        = {}
        self.cost_history      = []
        self.val_cost_history  = []
        self._initialize_parameters()

    # ── Private: ağırlık başlatma ─────────────────────────────────────────────
    def _initialize_parameters(self):
        np.random.seed(RANDOM_SEED)
        self.parameters = {
            'W1': np.random.randn(self.n_h, self.n_x) * 0.01,
            'b1': np.zeros((self.n_h, 1)),
            'W2': np.random.randn(self.n_y, self.n_h) * 0.01,
            'b2': np.zeros((self.n_y, 1))
        }

    # ── Private: aktivasyon fonksiyonları ─────────────────────────────────────
    def _hidden_act(self, Z):
        if self.hidden_activation == 'tanh':
            return np.tanh(Z)
        return self._sigmoid(Z)

    def _hidden_act_deriv(self, A):
        """Aktivasyon türevi — A aktive edilmiş değer."""
        if self.hidden_activation == 'tanh':
            return 1.0 - A ** 2
        return A * (1.0 - A)

    # ── Private: ileri yayılım ────────────────────────────────────────────────
    def _forward(self, X):
        """X: (m, n_x)  →  A2: (1, m), cache"""
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']

        Z1 = W1 @ X.T + b1          # (n_h, m)
        A1 = self._hidden_act(Z1)   # (n_h, m)
        Z2 = W2 @ A1 + b2           # (1, m)
        A2 = self._sigmoid(Z2)      # (1, m)

        return A2, {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    # ── Private: Binary Cross-Entropy maliyet ─────────────────────────────────
    def _compute_cost(self, A2, Y):
        """A2: (1, m)  |  Y: (m, 1)"""
        m   = A2.shape[1]
        eps = 1e-8
        ce  = -np.mean(Y.T * np.log(A2 + eps) + (1 - Y.T) * np.log(1 - A2 + eps))
        l2  = 0.0
        if self.l2_lambda > 0:
            l2 = (self.l2_lambda / (2 * m)) * (
                np.sum(self.parameters['W1'] ** 2) +
                np.sum(self.parameters['W2'] ** 2)
            )
        return float(ce + l2)

    # ── Private: geri yayılım ─────────────────────────────────────────────────
    def _backward(self, X, Y, cache):
        m  = X.shape[0]
        A1, A2 = cache['A1'], cache['A2']
        W2     = self.parameters['W2']

        dZ2 = A2 - Y.T
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * self._hidden_act_deriv(A1)
        dW1 = (dZ1 @ X) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        if self.l2_lambda > 0:
            dW1 += (self.l2_lambda / m) * self.parameters['W1']
            dW2 += (self.l2_lambda / m) * self.parameters['W2']

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    # ── Private: SGD parametre güncelleme ────────────────────────────────────
    def _update(self, grads):
        self.parameters['W1'] -= self.lr * grads['dW1']
        self.parameters['b1'] -= self.lr * grads['db1']
        self.parameters['W2'] -= self.lr * grads['dW2']
        self.parameters['b2'] -= self.lr * grads['db2']

    # ── Public: eğitim ────────────────────────────────────────────────────────
    def fit(self, X, Y, X_val=None, Y_val=None):
        """
        X, Y         : Eğitim verisi
        X_val, Y_val : Dev seti (opsiyonel — öğrenme eğrisi için)

        batch_size=None ise tam-batch SGD;
        batch_size=k ise her epoch'ta veri k'lik mini-batch'lere bölünür.
        """
        self.cost_history     = []
        self.val_cost_history = []
        m = X.shape[0]

        for i in range(self.n_steps):
            if self.batch_size is not None and self.batch_size < m:
                # Mini-batch SGD: veriyi karıştır ve parçala
                perm = np.random.permutation(m)
                X_shuf, Y_shuf = X[perm], Y[perm]
                for start in range(0, m, self.batch_size):
                    end   = min(start + self.batch_size, m)
                    X_mb  = X_shuf[start:end]
                    Y_mb  = Y_shuf[start:end]
                    A2, cache = self._forward(X_mb)
                    grads     = self._backward(X_mb, Y_mb, cache)
                    self._update(grads)
                # Epoch sonunda tam veri üzerinden maliyet hesapla
                A2_full, _ = self._forward(X)
                cost = self._compute_cost(A2_full, Y)
            else:
                # Tam-batch SGD (orijinal davranış)
                A2, cache = self._forward(X)
                cost      = self._compute_cost(A2, Y)
                grads     = self._backward(X, Y, cache)
                self._update(grads)

            self.cost_history.append(cost)

            if X_val is not None:
                A2_val, _ = self._forward(X_val)
                val_cost  = self._compute_cost(A2_val, Y_val)
                self.val_cost_history.append(val_cost)

            if self.print_cost and i % self.print_every == 0:
                msg = f"  Adım {i:5d} | Train Loss: {cost:.6f}"
                if X_val is not None:
                    msg += f" | Val Loss: {val_cost:.6f}"
                print(msg)
        return self


# ── 2 Gizli Katmanlı Model ────────────────────────────────────────────────────
class MLP_2Hidden(_BaseMLPNumpy):
    """
    2 Gizli Katmanlı MLP — NumPy.
    Mimari: Input(n_x) → Dense(n_h1, tanh) → Dense(n_h2, tanh) → Output(1, sigmoid)
    Loss:   Binary Cross-Entropy + opsiyonel L2
    Opt:    Tam-batch SGD (momentum=0)
    predict / score / plot_learning_curve → _BaseMLPNumpy'den miras alınır.
    """

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self, n_x, n_h1, n_h2, n_y=1,
                 learning_rate=0.1, n_steps=1000,
                 l2_lambda=0.0,
                 print_cost=False, print_every=200):
        self.n_x          = n_x
        self.n_h1         = n_h1
        self.n_h2         = n_h2
        self.n_y          = n_y
        self.lr           = learning_rate
        self.n_steps      = n_steps
        self.l2_lambda    = l2_lambda
        self.print_cost   = print_cost
        self.print_every  = print_every
        self.parameters   = {}
        self.cost_history = []
        self.val_cost_history = []
        self._initialize_parameters()

    # ── Private: ağırlık başlatma ─────────────────────────────────────────────
    def _initialize_parameters(self):
        np.random.seed(RANDOM_SEED)
        self.parameters = {
            'W1': np.random.randn(self.n_h1, self.n_x)  * 0.01,
            'b1': np.zeros((self.n_h1, 1)),
            'W2': np.random.randn(self.n_h2, self.n_h1) * 0.01,
            'b2': np.zeros((self.n_h2, 1)),
            'W3': np.random.randn(self.n_y,  self.n_h2) * 0.01,
            'b3': np.zeros((self.n_y, 1))
        }

    # ── Private: ileri yayılım ────────────────────────────────────────────────
    def _forward(self, X):
        p  = self.parameters
        Z1 = p['W1'] @ X.T + p['b1']; A1 = np.tanh(Z1)
        Z2 = p['W2'] @ A1  + p['b2']; A2 = np.tanh(Z2)
        Z3 = p['W3'] @ A2  + p['b3']; A3 = self._sigmoid(Z3)
        return A3, {'A1': A1, 'A2': A2, 'A3': A3}

    # ── Private: maliyet ──────────────────────────────────────────────────────
    def _compute_cost(self, A3, Y):
        m   = A3.shape[1]; eps = 1e-8
        ce  = -np.mean(Y.T * np.log(A3 + eps) + (1 - Y.T) * np.log(1 - A3 + eps))
        l2  = 0.0
        if self.l2_lambda > 0:
            l2 = (self.l2_lambda / (2 * m)) * sum(
                np.sum(self.parameters[k] ** 2) for k in ['W1', 'W2', 'W3'])
        return float(ce + l2)

    # ── Private: geri yayılım ─────────────────────────────────────────────────
    def _backward(self, X, Y, cache):
        m  = X.shape[0]
        p  = self.parameters
        A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']

        dZ3 = A3 - Y.T
        dW3 = (dZ3 @ A2.T) / m;  db3 = dZ3.sum(1, keepdims=True) / m
        dZ2 = (p['W3'].T @ dZ3) * (1 - A2 ** 2)
        dW2 = (dZ2 @ A1.T) / m;  db2 = dZ2.sum(1, keepdims=True) / m
        dZ1 = (p['W2'].T @ dZ2) * (1 - A1 ** 2)
        dW1 = (dZ1 @ X)   / m;   db1 = dZ1.sum(1, keepdims=True) / m

        if self.l2_lambda > 0:
            dW1 += (self.l2_lambda / m) * p['W1']
            dW2 += (self.l2_lambda / m) * p['W2']
            dW3 += (self.l2_lambda / m) * p['W3']

        return {'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2,
                'dW3': dW3, 'db3': db3}

    # ── Private: güncelleme ───────────────────────────────────────────────────
    def _update(self, grads):
        for k in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            self.parameters[k] -= self.lr * grads['d' + k]

    # ── Public: eğitim ────────────────────────────────────────────────────────
    def fit(self, X, Y, X_val=None, Y_val=None):
        self.cost_history, self.val_cost_history = [], []
        for i in range(self.n_steps):
            A3, cache = self._forward(X)
            cost      = self._compute_cost(A3, Y)
            grads     = self._backward(X, Y, cache)
            self._update(grads)
            self.cost_history.append(cost)
            if X_val is not None:
                A3v, _ = self._forward(X_val)
                self.val_cost_history.append(self._compute_cost(A3v, Y_val))
            if self.print_cost and i % self.print_every == 0:
                msg = f"  Adım {i:5d} | Loss: {cost:.6f}"
                if X_val is not None:
                    msg += f" | Val: {self.val_cost_history[-1]:.6f}"
                print(msg)
        return self
