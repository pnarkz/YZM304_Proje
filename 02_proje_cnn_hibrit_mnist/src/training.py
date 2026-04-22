from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_accuracy: list[float]
    test_accuracy: list[float]


@dataclass
class EvaluationResult:
    accuracy: float
    predictions: np.ndarray
    labels: np.ndarray
    confidences: np.ndarray
    images: np.ndarray
    report: dict
    confusion_matrix: np.ndarray


def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> TrainingHistory:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainingHistory(train_loss=[], train_accuracy=[], test_accuracy=[])

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        test_accuracy = evaluate_model(model, test_loader, device).accuracy

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.test_accuracy.append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"test_acc={test_accuracy:.4f}"
        )

    return history


def evaluate_model(model: nn.Module, data_loader, device: torch.device) -> EvaluationResult:
    model.eval()
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    confidences_all: list[np.ndarray] = []
    images_all: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            predictions = predictions.cpu().numpy()
            predictions_all.append(predictions)
            labels_all.append(labels.numpy())
            confidences_all.append(confidences.cpu().numpy())
            images_all.append(images.cpu().numpy())

    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    confidences_np = np.concatenate(confidences_all)
    images_np = np.concatenate(images_all)
    accuracy = float((labels_np == predictions_np).mean())
    report = classification_report(labels_np, predictions_np, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels_np, predictions_np)

    return EvaluationResult(
        accuracy=accuracy,
        predictions=predictions_np,
        labels=labels_np,
        confidences=confidences_np,
        images=images_np,
        report=report,
        confusion_matrix=cm,
    )


def save_history_plot(history: TrainingHistory, output_path: Path, title: str) -> None:
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, marker="o", label="Train Loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.train_accuracy, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history.test_accuracy, marker="s", label="Test Accuracy")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            ax.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
