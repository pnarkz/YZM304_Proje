# ─────────────────────────────────────────────────────────────────────────────
# src/models_pytorch.py  —  PyTorch MLP ile framework karşılaştırması
# YZM304 Derin Öğrenme | Pima Indians Diabetes MLP
#
# Aynı mimari, aynı SGD (momentum=0, tam-batch), aynı başlangıç ağırlıkları.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.config import RANDOM_SEED


def to_tensor(arr):
    """NumPy dizisini FloatTensor'a çevirir."""
    return torch.FloatTensor(arr)


# ── 1 Gizli Katmanlı PyTorch Modeli ──────────────────────────────────────────
class PyTorchMLP_1Hidden(nn.Module):
    """
    1 Gizli Katman — PyTorch nn.Module
    Mimari: Input(n_x) → Linear(n_h) → Tanh → Linear(n_y) → Sigmoid
    """
    def __init__(self, n_x, n_h, n_y=1):
        super().__init__()
        self.fc1 = nn.Linear(n_x, n_h)
        self.fc2 = nn.Linear(n_h, n_y)
        torch.manual_seed(RANDOM_SEED)
        nn.init.normal_(self.fc1.weight, 0, 0.01); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0, 0.01); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x))))


# ── 2 Gizli Katmanlı PyTorch Modeli ──────────────────────────────────────────
class PyTorchMLP_2Hidden(nn.Module):
    """
    2 Gizli Katman — PyTorch nn.Module
    Mimari: Input(n_x) → Linear(n_h1) → Tanh → Linear(n_h2) → Tanh → Linear(n_y) → Sigmoid
    """
    def __init__(self, n_x, n_h1, n_h2, n_y=1):
        super().__init__()
        self.fc1 = nn.Linear(n_x,  n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.fc3 = nn.Linear(n_h2, n_y)
        torch.manual_seed(RANDOM_SEED)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return torch.sigmoid(self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x))))))


# ── Eğitim fonksiyonu ─────────────────────────────────────────────────────────
def train_pytorch(model, X_tr, y_tr, X_val, y_val,
                  lr=0.1, n_steps=1000, print_every=200):
    """
    Tam-batch SGD ile PyTorch modeli eğitir.

    Döndürür: (train_loss_history, val_loss_history)
    """
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    cost_hist, val_hist = [], []

    for i in range(n_steps):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        cost_hist.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
            val_hist.append(val_loss.item())

        if i % print_every == 0:
            print(f"  Adım {i:5d} | Train: {loss.item():.6f} | Val: {val_loss.item():.6f}")

    return cost_hist, val_hist


def plot_pytorch_curve(cost_hist, val_hist, title='PyTorch Öğrenme Eğrisi'):
    plt.figure(figsize=(9, 4))
    plt.plot(cost_hist, label='Train Loss', color='steelblue', lw=1.5)
    plt.plot(val_hist,  label='Val Loss',   color='coral', linestyle='--', lw=1.5)
    plt.xlabel('Adım'); plt.ylabel('BCE Loss')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def pt_predict(model, X):
    """Tensor X için 0/1 tahmin dizisi döndürür."""
    model.eval()
    with torch.no_grad():
        return (model(X) > 0.5).int().numpy().flatten()
