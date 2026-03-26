# ─────────────────────────────────────────────────────────────────────────────
# src/evaluate.py  —  Model değerlendirme ve karşılaştırma araçları
# YZM304 Derin Öğrenme | Pima Indians Diabetes MLP
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)


def evaluate_model(model_name: str, y_true, y_pred) -> float:
    """
    Tek model için tüm metrikleri yazdırır ve confusion matrix çizer.
    Döndürür: accuracy (float)
    """
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print('='*55)
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=['Diyabet Yok (0)', 'Diyabet Var (1)'],
                                digits=4))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Yok (0)', 'Var (1)'],
                yticklabels=['Yok (0)', 'Var (1)'])
    ax.set_xlabel('Tahmin Edilen')
    ax.set_ylabel('Gerçek')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.show()
    return accuracy_score(y_true, y_pred)


def plot_overfit_summary(model_names, train_accs, dev_accs,
                         threshold=0.75, title='Train vs Dev Accuracy'):
    """
    NumPy modelleri için overfitting / underfitting özet çubuğu grafiği.
    """
    x = np.arange(len(model_names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, train_accs, w, label='Train Acc', color='steelblue', alpha=0.85)
    ax.bar(x + w/2, dev_accs,   w, label='Dev Acc',   color='coral',     alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\n(fark = overfitting göstergesi)')
    ax.legend()
    ax.axhline(threshold, linestyle='--', color='gray', alpha=0.5,
               label=f'%{int(threshold*100)} eşiği')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison_bar(comparison_df: pd.DataFrame):
    """
    Tüm framework'lerin Train / Dev / Test accuracy karşılaştırması.
    comparison_df: 'Model', 'Train', 'Dev', 'Test' sütunlarını içermeli.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(comparison_df)); w = 0.25
    ax.bar(x - w,   comparison_df['Train'], w, label='Train', color='steelblue', alpha=0.85)
    ax.bar(x,       comparison_df['Dev'],   w, label='Dev',   color='orange',    alpha=0.85)
    ax.bar(x + w,   comparison_df['Test'],  w, label='Test',  color='coral',     alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.55, 1.0)
    ax.axhline(0.75, linestyle='--', color='gray', alpha=0.5)
    ax.set_title('Tüm Modeller — Train / Dev / Test Accuracy Karşılaştırması',
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_data_fraction_experiment(fractions, train_accs, dev_accs, test_accs,
                                  model_label='Seçilen Model'):
    """
    Veri miktarı etkisi: %50 / %75 / %100 train verisi ile sonuçları çizer.
    """
    pct_labels = [f'%{int(f*100)}' for f in fractions]
    x = np.arange(len(fractions)); w = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w,   train_accs, w, label='Train Acc', color='steelblue', alpha=0.85)
    ax.bar(x,       dev_accs,   w, label='Dev Acc',   color='orange',    alpha=0.85)
    ax.bar(x + w,   test_accs,  w, label='Test Acc',  color='coral',     alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(pct_labels)
    ax.set_xlabel('Kullanılan Train Verisi Oranı')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.60, 1.0)
    ax.set_title(f'Veri Miktarı Etkisi — {model_label}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\n{'Oran':<8} {'Train':>8} {'Dev':>8} {'Test':>8}")
    print('-' * 36)
    for f, tr, dv, ts in zip(pct_labels, train_accs, dev_accs, test_accs):
        print(f"{f:<8} {tr:>8.4f} {dv:>8.4f} {ts:>8.4f}")


def plot_roc_auc(models_info: list, y_true):
    """
    Birden fazla model için ROC eğrisini tek grafikte çizer ve AUC skorlarını yazdırır.

    Parametreler
    ------------
    models_info : list of dict
        Her dict: {'name': str, 'y_proba': array-like (tahmin olasılıkları)}
    y_true      : gerçek etiketler (0/1)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_info)))

    print(f"\n{'Model':<35s} {'AUC':>6s}")
    print('-' * 43)

    for info, color in zip(models_info, colors):
        name   = info['name']
        proba  = np.array(info['y_proba']).flatten()
        y_flat = np.array(y_true).flatten()

        fpr, tpr, _ = roc_curve(y_flat, proba)
        auc_val     = roc_auc_score(y_flat, proba)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f'{name}  (AUC={auc_val:.4f})')
        print(f"  {name:<33s} {auc_val:.4f}")

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, label='Rastgele (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Eğrisi — Model Karşılaştırması', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

