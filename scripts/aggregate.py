"""Prediction aggregation utilities for weakly supervised histology models."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def aggregate_binary_predictions(
    predictions: pd.DataFrame,
    method: str = "topk_mean",
    top_k: int = 10,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate tile-level binary predictions to analysis-unit level."""
    rows: List[Dict[str, float | str | int]] = []
    grouped = predictions.groupby("analysis_unit_id", sort=False)

    for analysis_unit_id, group in grouped:
        probabilities = group["prob_positive"].to_numpy(dtype=float)
        labels = group["target"].to_numpy(dtype=int)
        binary_preds = (probabilities >= threshold).astype(int)

        if method == "mean":
            aggregated_score = float(np.mean(probabilities))
        elif method == "topk_mean":
            k = min(top_k, len(probabilities))
            aggregated_score = float(np.mean(np.sort(probabilities)[-k:]))
        elif method == "proportion_above_threshold":
            aggregated_score = float(np.mean(binary_preds))
        else:
            raise ValueError(f"Unsupported binary aggregation method: {method}")

        rows.append(
            {
                "analysis_unit_id": analysis_unit_id,
                "study_id": group["study_id"].iloc[0],
                "split": group["split"].iloc[0],
                "target": int(labels[0]),
                "tile_count": int(len(group)),
                "aggregated_score": aggregated_score,
                "pred_label": int(aggregated_score >= threshold),
            }
        )

    return pd.DataFrame(rows)


def aggregate_multiclass_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tile-level class probabilities to analysis-unit level."""
    probability_columns = sorted(
        column for column in predictions.columns if column.startswith("prob_class_")
    )
    rows: List[Dict[str, float | str | int]] = []

    for analysis_unit_id, group in predictions.groupby("analysis_unit_id", sort=False):
        mean_probabilities = group[probability_columns].mean(axis=0).to_numpy(dtype=float)
        pred_index = int(np.argmax(mean_probabilities))
        pred_score = pred_index + 1
        row: Dict[str, float | str | int] = {
            "analysis_unit_id": analysis_unit_id,
            "study_id": group["study_id"].iloc[0],
            "split": group["split"].iloc[0],
            "target": int(group["target"].iloc[0]),
            "tile_count": int(len(group)),
            "pred_label": pred_score,
        }
        for class_index, probability in enumerate(mean_probabilities, start=1):
            row[f"prob_score_{class_index}"] = float(probability)
        rows.append(row)

    return pd.DataFrame(rows)
