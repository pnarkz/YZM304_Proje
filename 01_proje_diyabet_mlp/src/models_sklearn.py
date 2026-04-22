# ─────────────────────────────────────────────────────────────────────────────
# src/models_sklearn.py  —  Sklearn MLPClassifier ile framework karşılaştırması
# YZM304 Derin Öğrenme | Pima Indians Diabetes MLP
#
# Aynı mimari, aynı SGD (momentum=0, tam-batch), aynı random_state.
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from src.config import RANDOM_SEED


def build_sklearn_model(hidden_layer_sizes: tuple,
                        learning_rate: float,
                        n_steps: int,
                        batch_size: int,
                        l2_alpha: float = 0.0) -> MLPClassifier:
    """
    NumPy modeliyle karşılaştırılabilir Sklearn MLPClassifier oluşturur.

    Parametreler
    ------------
    hidden_layer_sizes : Gizli katman nöron sayıları, örn. (8,) veya (16, 8)
    learning_rate      : SGD öğrenme hızı
    n_steps            : Epoch sayısı (max_iter)
    batch_size         : Tam-batch için X_train.shape[0]
    l2_alpha           : L2 regularizasyon katsayısı (varsayılan 0)
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='tanh',
        solver='sgd',
        learning_rate_init=learning_rate,
        max_iter=n_steps,
        batch_size=batch_size,        # tam batch = full gradient descent
        momentum=0.0,
        nesterovs_momentum=False,
        alpha=l2_alpha,
        random_state=RANDOM_SEED,
        early_stopping=False,
        warm_start=False,
    )


def train_and_evaluate_sklearn(model: MLPClassifier,
                               X_train, y_train,
                               X_dev,   y_dev,
                               X_test,  y_test,
                               model_name: str = 'Sklearn MLP') -> dict:
    """
    Modeli eğit, öğrenme eğrisini çiz, metrikleri döndür.

    Döndürür: {'train_acc', 'dev_acc', 'test_acc'}
    """
    model.fit(X_train, y_train.flatten())

    # Öğrenme eğrisi (loss_curve_ özelliği)
    plt.figure(figsize=(9, 4))
    plt.plot(model.loss_curve_, label='Train Loss', color='steelblue', lw=1.5)
    plt.xlabel('Adım (Epoch)')
    plt.ylabel('Log-Loss (Binary Cross-Entropy)')
    plt.title(f'{model_name} — Öğrenme Eğrisi')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    tr = accuracy_score(y_train.flatten(), model.predict(X_train))
    dv = accuracy_score(y_dev.flatten(),   model.predict(X_dev))
    ts = accuracy_score(y_test.flatten(),  model.predict(X_test))

    print(f"{model_name} | Train: {tr:.4f} | Dev: {dv:.4f} | Test: {ts:.4f}")
    return {'train_acc': tr, 'dev_acc': dv, 'test_acc': ts}
