from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(model, data_loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            batch_features = model.forward_features(images).cpu().numpy()
            features.append(batch_features)
            labels_all.append(labels.numpy())

    return np.concatenate(features), np.concatenate(labels_all)


def save_feature_arrays(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
) -> None:
    np.save(output_dir / "train_features.npy", train_features)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_features.npy", test_features)
    np.save(output_dir / "test_labels.npy", test_labels)


def train_svm_classifier(train_features: np.ndarray, train_labels: np.ndarray) -> Pipeline:
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", gamma="scale")),
        ]
    )
    classifier.fit(train_features, train_labels)
    return classifier
