# ─────────────────────────────────────────────────────────────────────────────
# src/data_loader.py  —  Veri yükleme, ön işleme, bölme ve standardizasyon
# YZM304 Derin Öğrenme | Pima Indians Diabetes MLP
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, TEST_SIZE, DEV_SIZE


# ── Veri yükleme ──────────────────────────────────────────────────────────────
def load_data(data_path: str = None) -> pd.DataFrame:
    """
    diabetes.csv dosyasını yükler.
    data_path verilmezse data/diabetes.csv → diabetes.csv → ../data/diabetes.csv
    sırasıyla arar.
    """
    if data_path is None:
        candidates = [
            os.path.join('data', 'diabetes.csv'),
            'diabetes.csv',
            os.path.join('..', 'data', 'diabetes.csv'),
        ]
        data_path = next((p for p in candidates if os.path.exists(p)), None)
        if data_path is None:
            raise FileNotFoundError(
                "diabetes.csv bulunamadı. Lütfen 'data/diabetes.csv' yoluna yerleştirin."
            )

    df = pd.read_csv(data_path)
    print(f"Veri yüklendi: {data_path}  →  {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


# ── Keşifsel veri analizi ─────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    """Temel istatistikler, sıfır analizi, dağılım grafikleri."""

    print("=== İstatistiksel Özet ===")
    print(df.describe().round(3))

    print("\n=== Biyolojik Sıfır Değerleri (Gerçekte Eksik Veri) ===")
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        n_zero = (df[col] == 0).sum()
        pct = n_zero / len(df) * 100
        print(f"  {col:30s}: {n_zero:3d} sıfır ({pct:.1f}%)")

    counts = df['Outcome'].value_counts()
    print(f"\n=== Sınıf Dağılımı ===")
    print(f"  Diyabet yok (0): {counts[0]} ({counts[0]/len(df)*100:.1f}%)")
    print(f"  Diyabet var (1): {counts[1]} ({counts[1]/len(df)*100:.1f}%)")

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Özellik dağılımları
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        axes[i].hist(df[df['Outcome'] == 0][feat], alpha=0.6, bins=25,
                     color='steelblue', label='Diyabet Yok (0)')
        axes[i].hist(df[df['Outcome'] == 1][feat], alpha=0.6, bins=25,
                     color='coral', label='Diyabet Var (1)')
        axes[i].set_title(feat, fontweight='bold')
        axes[i].legend(fontsize=7)
        axes[i].grid(alpha=0.3)
    plt.suptitle('Özellik Dağılımları — Sınıfa Göre', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Korelasyon matrisi + sınıf dağılımı
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                ax=axes[0], square=True)
    axes[0].set_title('Korelasyon Matrisi')
    axes[1].bar(['Diyabet Yok (0)', 'Diyabet Var (1)'],
                counts.values, color=['steelblue', 'coral'],
                edgecolor='black', width=0.5)
    axes[1].set_title('Sınıf Dağılımı')
    axes[1].set_ylabel('Örnek Sayısı')
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Boxplot
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        df.boxplot(column=feat, by='Outcome', ax=axes[i])
        axes[i].set_title(feat)
        axes[i].set_xlabel('Outcome')
    plt.suptitle('Boxplot — Sınıfa Göre Özellik Dağılımları')
    plt.tight_layout()
    plt.show()


# ── Ön işleme ─────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    1. Biyolojik sıfırları medyanla doldur
    2. Karıştır
    3. X / y ayır
    Döndürür: X (numpy), y (numpy, shape=(m,1))
    """
    df_clean = df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        median_val = df_clean[col][df_clean[col] != 0].median()
        n_replaced  = (df_clean[col] == 0).sum()
        df_clean[col] = df_clean[col].replace(0, median_val)
        print(f"  {col:30s}: {n_replaced} sıfır → medyan {median_val:.2f} ile değiştirildi")

    df_clean = df_clean.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    X = df_clean.iloc[:, :-1].to_numpy()
    y = df_clean.iloc[:, -1].to_numpy().reshape(-1, 1)
    print(f"\nX shape: {X.shape}  |  y shape: {y.shape}")
    return X, y


# ── Bölme ve standardizasyon ──────────────────────────────────────────────────
def split_and_scale(X: np.ndarray, y: np.ndarray):
    """
    Train %70 / Dev %15 / Test %15 — stratified.
    Standardizasyon: yalnızca train üzerinde fit.

    Döndürür: X_train_s, X_dev_s, X_test_s,
              y_train, y_dev, y_test, scaler
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + DEV_SIZE),
        random_state=RANDOM_SEED, stratify=y
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]}  |  Dev: {X_dev.shape[0]}  |  Test: {X_test.shape[0]}")
    for name, arr in [('Train', y_train), ('Dev', y_dev), ('Test', y_test)]:
        u, c = np.unique(arr, return_counts=True)
        print(f"  {name}: {dict(zip(u.astype(int), c))}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s   = scaler.transform(X_dev)
    X_test_s  = scaler.transform(X_test)
    print("Standardizasyon tamamlandı.")

    return X_train_s, X_dev_s, X_test_s, y_train, y_dev, y_test, scaler
