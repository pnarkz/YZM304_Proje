from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.analysis import (
    count_trainable_parameters,
    save_accuracy_vs_complexity_plot,
    save_feature_projection_plot,
    save_misclassified_grid,
)
from src.data import build_mnist_dataloaders
from src.hybrid import extract_features, save_feature_arrays, train_svm_classifier
from src.models import LeNet5Baseline, LeNet5Improved, ResNet18ForMNIST
from src.training import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_history_plot,
    save_json,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YZM304 Derin Ogrenme proje deneyleri: LeNet, gelistirilmis LeNet, ResNet18 ve hibrit CNN+SVM."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Tum CNN modelleri icin epoch sayisi.")
    parser.add_argument("--batch-size", type=int, default=64, help="Egitim batch size.")
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=256,
        help="Ozellik cikartma asamasi batch size.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam optimizer ogrenme orani.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Sonuclarin kaydedilecegi klasor.")
    parser.add_argument("--data-dir", type=str, default="data", help="Veri seti klasoru.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Hizli deneme icin egitim veri alt kumesi boyutu.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Hizli deneme icin test veri alt kumesi boyutu.",
    )
    return parser.parse_args()


def ensure_directories(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
        "metrics": root / "metrics",
        "features": root / "features",
        "reports": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_experiment() -> None:
    args = parse_args()
    output_paths = ensure_directories(Path(args.output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")
    print(
        f"Hyperparameters | epochs={args.epochs} | batch_size={args.batch_size} | "
        f"feature_batch_size={args.feature_batch_size} | learning_rate={args.learning_rate}"
    )

    bundle = build_mnist_dataloaders(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    model_specs = [
        ("model_1_lenet5_baseline", LeNet5Baseline(num_classes=bundle.num_classes)),
        ("model_2_lenet5_improved", LeNet5Improved(num_classes=bundle.num_classes)),
        ("model_3_resnet18", ResNet18ForMNIST(num_classes=bundle.num_classes)),
    ]

    metrics_summary: list[dict] = []
    trained_models: dict[str, torch.nn.Module] = {}
    feature_snapshots: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model in model_specs:
        print(f"\n=== {model_name} ===")
        parameter_count = count_trainable_parameters(model)
        history = train_model(
            model=model,
            train_loader=bundle.train_loader,
            test_loader=bundle.test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        evaluation = evaluate_model(model, bundle.test_loader, device)

        save_history_plot(
            history,
            output_paths["plots"] / f"{model_name}_history.png",
            title=model_name,
        )
        save_confusion_matrix_plot(
            evaluation.confusion_matrix,
            bundle.class_names,
            output_paths["plots"] / f"{model_name}_confusion_matrix.png",
            title=f"{model_name} confusion matrix",
        )
        save_json(
            {
                "accuracy": evaluation.accuracy,
                "classification_report": evaluation.report,
            },
            output_paths["metrics"] / f"{model_name}.json",
        )
        save_misclassified_grid(
            images=evaluation.images,
            labels=evaluation.labels,
            predictions=evaluation.predictions,
            confidences=evaluation.confidences,
            output_path=output_paths["plots"] / f"{model_name}_misclassified_grid.png",
            title=f"{model_name} en guvenli yanlis tahminler",
        )
        metrics_summary.append(
            {
                "model": model_name,
                "test_accuracy": round(evaluation.accuracy, 4),
                "parameter_count": parameter_count,
            }
        )
        trained_models[model_name] = model
        if hasattr(model, "forward_features"):
            test_features_snapshot, test_labels_snapshot = extract_features(
                model,
                bundle.feature_test_loader,
                device,
            )
            feature_snapshots[model_name] = {
                "features": test_features_snapshot,
                "labels": test_labels_snapshot,
            }
            save_feature_projection_plot(
                test_features_snapshot,
                test_labels_snapshot,
                output_paths["plots"] / f"{model_name}_feature_projection.png",
                title=f"{model_name} PCA ozellik izdusumu",
            )

    reference_model_name = "model_2_lenet5_improved"
    reference_model = trained_models[reference_model_name]

    print(f"\n=== model_4_hybrid_cnn_svm (feature extractor: {reference_model_name}) ===")
    train_features, train_labels = extract_features(
        reference_model,
        bundle.feature_train_loader,
        device,
    )
    test_features, test_labels = extract_features(
        reference_model,
        bundle.feature_test_loader,
        device,
    )

    print(f"train_features shape: {train_features.shape}")
    print(f"train_labels shape: {train_labels.shape}")
    print(f"test_features shape: {test_features.shape}")
    print(f"test_labels shape: {test_labels.shape}")

    save_feature_arrays(
        train_features,
        train_labels,
        test_features,
        test_labels,
        output_paths["features"],
    )

    svm_classifier = train_svm_classifier(train_features, train_labels)
    hybrid_predictions = svm_classifier.predict(test_features)
    hybrid_accuracy = float((hybrid_predictions == test_labels).mean())
    hybrid_report = classification_report(
        test_labels,
        hybrid_predictions,
        output_dict=True,
        zero_division=0,
    )
    hybrid_confusion = confusion_matrix(test_labels, hybrid_predictions)

    save_confusion_matrix_plot(
        hybrid_confusion,
        bundle.class_names,
        output_paths["plots"] / "model_4_hybrid_cnn_svm_confusion_matrix.png",
        title="model_4_hybrid_cnn_svm confusion matrix",
    )
    save_json(
        {
            "accuracy": hybrid_accuracy,
            "classification_report": hybrid_report,
            "reference_cnn_model": reference_model_name,
            "feature_shapes": {
                "train_features": list(train_features.shape),
                "train_labels": list(train_labels.shape),
                "test_features": list(test_features.shape),
                "test_labels": list(test_labels.shape),
            },
        },
        output_paths["metrics"] / "model_4_hybrid_cnn_svm.json",
    )
    metrics_summary.append(
        {
            "model": "model_4_hybrid_cnn_svm",
            "test_accuracy": round(hybrid_accuracy, 4),
            "parameter_count": 0,
        }
    )

    full_cnn_accuracy = next(
        item["test_accuracy"] for item in metrics_summary if item["model"] == reference_model_name
    )
    comparison = {
        "dataset": "MNIST",
        "reference_full_cnn": reference_model_name,
        "reference_full_cnn_accuracy": full_cnn_accuracy,
        "hybrid_model_accuracy": round(hybrid_accuracy, 4),
        "accuracy_gap": round(hybrid_accuracy - full_cnn_accuracy, 4),
    }

    summary_payload = {
        "device": str(device),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "feature_batch_size": args.feature_batch_size,
            "learning_rate": args.learning_rate,
        },
        "results": metrics_summary,
        "hybrid_vs_full_cnn": comparison,
        "project_extensions": [
            "Yanlis siniflandirilan ornekler icin guven skoru galerisi",
            "CNN ozellik uzayi icin PCA tabanli 2B izdusum",
            "Dogruluk-parametre karmasikligi analizi",
        ],
    }
    save_json(summary_payload, output_paths["root"] / "summary.json")
    save_accuracy_vs_complexity_plot(
        metrics_summary,
        output_paths["plots"] / "accuracy_vs_complexity.png",
        title="Model dogrulugu ve parametre karmasikligi",
    )

    report_lines = [
        "# Sonuc Ozeti",
        "",
        "| Model | Test Accuracy | Parametre Sayisi |",
        "| --- | ---: | ---: |",
    ]
    for item in metrics_summary:
        report_lines.append(
            f"| {item['model']} | {item['test_accuracy']:.4f} | {item['parameter_count']} |"
        )
    report_lines.extend(
        [
            "",
            f"- Hibrit karsilastirma referansi: `{comparison['reference_full_cnn']}`",
            f"- Tam CNN accuracy: `{comparison['reference_full_cnn_accuracy']:.4f}`",
            f"- Hibrit accuracy: `{comparison['hybrid_model_accuracy']:.4f}`",
            f"- Accuracy farki: `{comparison['accuracy_gap']:.4f}`",
        ]
    )
    (output_paths["reports"] / "results_table.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    for item in metrics_summary:
        print(f"{item['model']}: test_accuracy={item['test_accuracy']:.4f}")
    print(
        "Hybrid comparison | "
        f"full_cnn={comparison['reference_full_cnn_accuracy']:.4f} | "
        f"hybrid={comparison['hybrid_model_accuracy']:.4f} | "
        f"gap={comparison['accuracy_gap']:.4f}"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_experiment()
