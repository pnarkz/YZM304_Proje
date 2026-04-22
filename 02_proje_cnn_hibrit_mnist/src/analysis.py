from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_feature_projection_plot(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    reducer = PCA(n_components=2, random_state=42)
    projected = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.75,
    )
    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    legend = ax.legend(*scatter.legend_elements(), title="Sinif", loc="best", fontsize=8)
    ax.add_artist(legend)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_misclassified_grid(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    output_path: Path,
    title: str,
    max_items: int = 16,
) -> None:
    wrong_indices = np.where(labels != predictions)[0]
    if wrong_indices.size == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Yanlis siniflandirma yok", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    ranked = wrong_indices[np.argsort(confidences[wrong_indices])[::-1]]
    selected = ranked[:max_items]
    rows = int(np.ceil(len(selected) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, 4)

    for axis in axes.flat:
        axis.axis("off")

    for axis, index in zip(axes.flat, selected):
        axis.imshow(images[index, 0], cmap="gray")
        axis.set_title(
            f"Gercek: {labels[index]}\nTahmin: {predictions[index]}\nGuven: {confidences[index]:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_vs_complexity_plot(
    results: list[dict],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    parameter_counts = [item["parameter_count"] for item in results]
    accuracies = [item["test_accuracy"] for item in results]

    ax.scatter(parameter_counts, accuracies, s=120, c=np.arange(len(results)), cmap="viridis")
    for item in results:
        ax.annotate(
            item["model"],
            (item["parameter_count"], item["test_accuracy"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameter Count (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
