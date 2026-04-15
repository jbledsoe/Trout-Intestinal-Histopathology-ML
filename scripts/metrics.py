"""Metrics and plotting helpers for histology tile classification."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    cohen_kappa_score,
)


def save_confusion_matrix_figure(
    matrix: np.ndarray,
    class_labels: List[str],
    destination: str | Path,
    title: str,
) -> None:
    """Save a simple confusion matrix heatmap."""
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(np.arange(len(class_labels)))
    axis.set_yticks(np.arange(len(class_labels)))
    axis.set_xticklabels(class_labels)
    axis.set_yticklabels(class_labels)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_pr_curves(
    targets: np.ndarray,
    scores: np.ndarray,
    roc_destination: str | Path,
    pr_destination: str | Path,
) -> Dict[str, float]:
    """Save ROC and precision-recall curves for binary predictions."""
    if len(np.unique(targets)) < 2:
        warnings.warn(
            "ROC/PR curves are undefined because only one class is present in the targets. Saving placeholder figures.",
            RuntimeWarning,
        )
        _save_placeholder_curve(
            destination=roc_destination,
            title="ROC Curve",
            message="Undefined: only one class present in targets",
        )
        _save_placeholder_curve(
            destination=pr_destination,
            title="Precision-Recall Curve",
            message="Undefined: only one class present in targets",
        )
        return {"auroc_curve": float("nan"), "auprc_curve": float("nan")}

    fpr, tpr, _ = roc_curve(targets, scores)
    precision, recall, _ = precision_recall_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
    roc_axis.plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.set_title("ROC Curve")
    roc_axis.legend(loc="lower right")
    roc_figure.tight_layout()
    roc_figure.savefig(roc_destination, dpi=200, bbox_inches="tight")
    plt.close(roc_figure)

    pr_figure, pr_axis = plt.subplots(figsize=(6, 5))
    pr_axis.plot(recall, precision, label=f"AUPRC = {pr_auc:.3f}")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title("Precision-Recall Curve")
    pr_axis.legend(loc="lower left")
    pr_figure.tight_layout()
    pr_figure.savefig(pr_destination, dpi=200, bbox_inches="tight")
    plt.close(pr_figure)

    return {"auroc_curve": roc_auc, "auprc_curve": pr_auc}


def _safe_binary_metric(metric_name: str, targets: np.ndarray, values: np.ndarray) -> float:
    """Return NaN with a warning when a binary rank metric is undefined."""
    if len(np.unique(targets)) < 2:
        warnings.warn(
            f"{metric_name} is undefined because only one class is present in the targets.",
            RuntimeWarning,
        )
        return float("nan")
    if metric_name == "auroc":
        return float(roc_auc_score(targets, values))
    if metric_name == "auprc":
        return float(average_precision_score(targets, values))
    raise ValueError(f"Unsupported binary metric: {metric_name}")


def _save_placeholder_curve(destination: str | Path, title: str, message: str) -> None:
    """Save a placeholder figure when a curve metric is undefined."""
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def compute_binary_metrics(
    targets: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float | List[List[int]]]:
    """Compute core binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(targets, predicted_labels, labels=[0, 1]).ravel()
    metrics: Dict[str, float | List[List[int]]] = {
        "auroc": _safe_binary_metric("auroc", targets, scores),
        "auprc": _safe_binary_metric("auprc", targets, scores),
        "f1": float(f1_score(targets, predicted_labels)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted_labels)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics


def compute_multiclass_metrics(
    targets: np.ndarray,
    predicted_labels: np.ndarray,
    labels: List[int],
) -> Dict[str, object]:
    """Compute multiclass metrics with per-class detail."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predicted_labels, labels=labels)
    report = classification_report(
        targets,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = []
    for index, label in enumerate(labels):
        per_class.append(
            {
                "class": int(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )

    return {
        "accuracy": float(accuracy_score(targets, predicted_labels)),
        "macro_f1": float(f1_score(targets, predicted_labels, average="macro")),
        "weighted_f1": float(f1_score(targets, predicted_labels, average="weighted")),
        "quadratic_weighted_kappa": (
            float(cohen_kappa_score(targets, predicted_labels, weights="quadratic"))
            if len(np.unique(targets)) > 1
            else float("nan")
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
        "classification_report": report,
    }


def collect_hard_examples(
    predictions: pd.DataFrame,
    destination: str | Path,
    top_n: int = 50,
) -> pd.DataFrame:
    """Save a qualitative review table of confident errors or hard cases."""
    if "prob_positive" in predictions.columns:
        predictions = predictions.copy()
        predictions["difficulty_score"] = np.where(
            predictions["target"] == 1,
            1.0 - predictions["prob_positive"],
            predictions["prob_positive"],
        )
    else:
        probability_columns = [col for col in predictions.columns if col.startswith("prob_class_")]
        predictions = predictions.copy()
        target_indices = predictions["target"].astype(int).to_numpy() - 1
        probs = predictions[probability_columns].to_numpy(dtype=float)
        predictions["difficulty_score"] = 1.0 - probs[np.arange(len(predictions)), target_indices]

    hardest = predictions.sort_values("difficulty_score", ascending=False).head(top_n)
    hardest.to_csv(destination, index=False)
    return hardest


def collect_top_positive_tiles(
    predictions: pd.DataFrame,
    destination: str | Path,
    top_n: int = 50,
) -> pd.DataFrame:
    """Save the highest-confidence positive enteritis tiles for review."""
    if "prob_positive" not in predictions.columns:
        raise ValueError("Top positive tile export is only valid for binary predictions.")
    top_positive = predictions.sort_values("prob_positive", ascending=False).head(top_n)
    top_positive.to_csv(destination, index=False)
    return top_positive
