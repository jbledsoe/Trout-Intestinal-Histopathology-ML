#!/usr/bin/env python3
"""Run public-facing tile inference and aggregate to analysis-unit predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import amp
from torch.utils.data import DataLoader, Dataset

from datasets import build_transforms
from models import build_model
from utils import RunPaths, build_run_paths, configure_logging, get_device, load_yaml_config, save_config_snapshot


class InferenceTileDataset(Dataset):
    """Minimal dataset for manifest-driven tile inference."""

    def __init__(self, dataframe: pd.DataFrame, transform: Any) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        from PIL import Image

        with Image.open(row["resolved_tile_path"]) as image:
            image.load()
            image = image.convert("RGB").copy()

        return {
            "image": self.transform(image) if self.transform is not None else image,
            "resolved_tile_path": row["resolved_tile_path"],
            "tile_path": row["tile_path"],
            "filename": row["filename"],
            "analysis_unit_id": row["analysis_unit_id"],
            "study_id": row["study_id"],
            "split": row["split"],
            "metadata": {column: row[column] for column in row.index if column not in {"resolved_tile_path"}},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run histology inference from a tile manifest and checkpoint.")
    parser.add_argument("--config", required=True, help="Path to an inference YAML config.")
    return parser.parse_args()


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_mapping(mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    normalized: Dict[Any, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, str) and key.strip().lstrip("-").isdigit():
            normalized[int(key)] = value
            normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _prepare_label_series(dataframe: pd.DataFrame, task_cfg: Dict[str, Any]) -> pd.DataFrame:
    label_column = task_cfg.get("label_column")
    label_map = _normalize_mapping(task_cfg.get("label_map", {}))
    label_values = list(task_cfg.get("label_values", []))
    prepared = dataframe.copy()

    if not label_column or label_column not in prepared.columns:
        return prepared

    raw = pd.to_numeric(prepared[label_column], errors="coerce")
    if label_map:
        def _map_value(value: float) -> Any:
            if pd.isna(value):
                return np.nan
            as_int = int(value)
            if as_int in label_map:
                return label_map[as_int]
            if value in label_map:
                return label_map[value]
            if str(as_int) in label_map:
                return label_map[str(as_int)]
            if str(value) in label_map:
                return label_map[str(value)]
            return np.nan

        mapped = raw.map(_map_value)
        prepared["target"] = pd.to_numeric(mapped, errors="coerce")
    else:
        prepared["target"] = raw

    if label_values:
        prepared.loc[~prepared["target"].isin(label_values), "target"] = np.nan
    return prepared


def _resolve_tile_paths(
    dataframe: pd.DataFrame,
    data_root: Path,
    tile_path_column: str,
) -> pd.DataFrame:
    resolved = dataframe.copy()

    def _join_path(value: Any) -> str:
        path = Path(str(value))
        if path.is_absolute():
            return str(path)
        return str((data_root / path).resolve())

    resolved["resolved_tile_path"] = resolved[tile_path_column].map(_join_path)
    resolved["tile_path"] = resolved[tile_path_column].astype(str)
    if "filename" not in resolved.columns:
        resolved["filename"] = resolved["tile_path"].map(lambda value: Path(value).name)
    else:
        resolved["filename"] = resolved["filename"].fillna("").astype(str)
        missing = resolved["filename"] == ""
        resolved.loc[missing, "filename"] = resolved.loc[missing, "tile_path"].map(lambda value: Path(value).name)
    return resolved


def _apply_metadata_joins(
    dataframe: pd.DataFrame,
    join_cfgs: Sequence[Dict[str, Any]],
    base_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    merged = dataframe.copy()
    for join_cfg in join_cfgs:
        join_path = _resolve_path(join_cfg["path"], base_dir)
        join_on = join_cfg["on"]
        how = str(join_cfg.get("how", "left"))
        columns = list(join_cfg.get("columns", []))
        logger.info("Joining supplemental metadata: %s", join_path)
        extra_df = pd.read_csv(join_path, low_memory=False)
        required_columns = [join_on] + [column for column in columns if column != join_on]
        keep_columns = [column for column in required_columns if column in extra_df.columns]
        extra_df = extra_df[keep_columns].drop_duplicates(subset=[join_on])
        merged = merged.merge(extra_df, on=join_on, how=how)
    return merged


def load_inference_manifest(config: Dict[str, Any], config_path: Path, logger: logging.Logger) -> pd.DataFrame:
    data_cfg = config["data"]
    task_cfg = config["task"]
    manifest_path = _resolve_path(data_cfg["tile_manifest_path"], config_path.parent)
    data_root = _resolve_path(data_cfg.get("data_root", "."), config_path.parent)
    tile_path_column = str(data_cfg.get("tile_path_column", "tile_path"))
    analysis_unit_column = str(data_cfg.get("analysis_unit_id_column", "analysis_unit_id"))
    split_column = str(data_cfg.get("split_column", "split"))
    study_column = str(data_cfg.get("study_id_column", "study_key"))
    task_column = str(data_cfg.get("task_column", "task"))
    allowed_tasks = list(data_cfg.get("allowed_tasks", [task_cfg["name"]]))
    allowed_read_levels = list(data_cfg.get("allowed_read_levels", [0]))

    df = pd.read_csv(manifest_path, low_memory=False)
    if data_cfg.get("metadata_joins"):
        df = _apply_metadata_joins(df, data_cfg["metadata_joins"], config_path.parent, logger)

    required_columns = [tile_path_column, analysis_unit_column]
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(f"Manifest is missing required columns: {missing_required}")

    if task_column in df.columns and allowed_tasks:
        df = df.loc[df[task_column].astype(str).isin([str(value) for value in allowed_tasks])].copy()

    if "tile_exists" in df.columns and _normalize_bool(data_cfg.get("require_tile_exists", True)):
        mask = df["tile_exists"].map(_normalize_bool)
        df = df.loc[mask].copy()

    if "read_level" in df.columns and allowed_read_levels:
        numeric_levels = pd.to_numeric(df["read_level"], errors="coerce")
        df = df.loc[numeric_levels.isin(allowed_read_levels)].copy()

    if split_column not in df.columns:
        df["split"] = str(data_cfg.get("default_split", "inference"))
        split_column = "split"
    else:
        df[split_column] = df[split_column].fillna(str(data_cfg.get("default_split", "inference"))).astype(str)

    if study_column not in df.columns:
        fallback_columns = [column for column in ["study_key", "study_id", "slide_id"] if column in df.columns]
        if fallback_columns:
            study_column = fallback_columns[0]
        else:
            df["study_id"] = "unknown_study"
            study_column = "study_id"

    df = _resolve_tile_paths(df, data_root=data_root, tile_path_column=tile_path_column)
    df["analysis_unit_id"] = df[analysis_unit_column].astype(str)
    df["study_id"] = df[study_column].astype(str)
    df["split"] = df[split_column].astype(str)

    df = _prepare_label_series(df, task_cfg)

    drop_missing_files = _normalize_bool(data_cfg.get("drop_missing_tiles", True))
    exists_mask = df["resolved_tile_path"].map(lambda value: Path(value).exists())
    missing_count = int((~exists_mask).sum())
    if missing_count:
        message = f"{missing_count} tile files referenced by the manifest do not exist on disk."
        if drop_missing_files:
            logger.warning("%s Dropping missing rows.", message)
            df = df.loc[exists_mask].copy()
        else:
            raise FileNotFoundError(message)

    metadata_columns = list(dict.fromkeys(data_cfg.get("metadata_columns", [])))
    passthrough = [
        column
        for column in [
            "slide_id",
            "sample_id",
            "section_label",
            "study_key",
            "tissue_fraction",
            "tile_x_level0",
            "tile_y_level0",
            "manual_qc_status",
            "reviewer_notes",
        ]
        if column in df.columns and column not in metadata_columns
    ]
    for column in passthrough:
        metadata_columns.append(column)
    keep_columns = [
        "resolved_tile_path",
        "tile_path",
        "filename",
        "analysis_unit_id",
        "study_id",
        "split",
    ] + [column for column in metadata_columns if column in df.columns]
    if "target" in df.columns:
        keep_columns.append("target")
    if task_cfg.get("label_column") in df.columns:
        keep_columns.append(task_cfg["label_column"])

    manifest = df[keep_columns].copy()
    if manifest.empty:
        raise ValueError("No manifest rows remain after inference filtering.")
    return manifest


def load_checkpoint_weights(
    checkpoint_path: Path,
    checkpoint_key: str,
    map_location: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if checkpoint_key == "auto":
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError("Unsupported checkpoint format for auto loading.")
    if checkpoint_key == "none":
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a state_dict-like mapping when checkpoint_key=none.")
        return checkpoint
    if checkpoint_key not in checkpoint:
        raise KeyError(f"Checkpoint key '{checkpoint_key}' not found in {checkpoint_path}.")
    return checkpoint[checkpoint_key]


def build_inference_loader(config: Dict[str, Any], manifest_df: pd.DataFrame) -> DataLoader:
    augmentation_cfg = config["augmentation"]
    inference_cfg = config["inference"]
    dataset = InferenceTileDataset(
        dataframe=manifest_df,
        transform=build_transforms(augmentation_cfg, is_training=False),
    )
    return DataLoader(
        dataset,
        batch_size=int(inference_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(inference_cfg.get("num_workers", config["data"].get("num_workers", 4))),
        pin_memory=bool(inference_cfg.get("pin_memory", config["data"].get("pin_memory", True))),
        drop_last=False,
    )


def _class_labels(task_cfg: Dict[str, Any]) -> List[int]:
    label_values = task_cfg.get("label_values")
    if label_values:
        return [int(value) for value in label_values]
    return list(range(int(task_cfg["num_classes"])))


def _class_names(task_cfg: Dict[str, Any], class_labels: Sequence[int]) -> List[str]:
    class_names = task_cfg.get("class_names")
    if class_names:
        return [str(value) for value in class_names]
    return [str(value) for value in class_labels]


def predict_tiles(
    model: torch.nn.Module,
    loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> pd.DataFrame:
    task_cfg = config["task"]
    class_labels = _class_labels(task_cfg)
    class_names = _class_names(task_cfg, class_labels)
    positive_class_index = int(task_cfg.get("positive_class_index", 1))

    model.eval()
    rows: List[Dict[str, Any]] = []
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with amp.autocast(amp_device_type, enabled=use_amp):
                probabilities = torch.softmax(model(images), dim=1).cpu().numpy()

            batch_size = images.size(0)
            for idx in range(batch_size):
                row = {
                    "analysis_unit_id": batch["analysis_unit_id"][idx],
                    "study_id": batch["study_id"][idx],
                    "tile_path": batch["tile_path"][idx],
                    "resolved_tile_path": batch["resolved_tile_path"][idx],
                    "filename": batch["filename"][idx],
                    "split": batch["split"][idx],
                }
                metadata = batch["metadata"]
                for key, values in metadata.items():
                    row[key] = values[idx]

                if "target" in row and pd.notna(row["target"]):
                    row["target"] = int(float(row["target"]))

                if str(task_cfg["mode"]).lower() == "binary":
                    negative_index = 1 - positive_class_index
                    prob_positive = float(probabilities[idx, positive_class_index])
                    prob_negative = float(probabilities[idx, negative_index])
                    pred_index = int(prob_positive >= float(config["aggregation"].get("threshold", 0.5)))
                    pred_label = class_labels[pred_index]
                    row.update(
                        {
                            "prob_negative": prob_negative,
                            "prob_positive": prob_positive,
                            "confidence": float(max(prob_negative, prob_positive)),
                            "pred_label": int(pred_label),
                            "pred_class_name": class_names[pred_index],
                        }
                    )
                else:
                    pred_index = int(np.argmax(probabilities[idx]))
                    pred_label = class_labels[pred_index]
                    row["confidence"] = float(np.max(probabilities[idx]))
                    row["pred_label"] = int(pred_label)
                    row["pred_class_name"] = class_names[pred_index]
                    for class_index, label_value in enumerate(class_labels):
                        row[f"prob_class_{label_value}"] = float(probabilities[idx, class_index])
                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_predictions(tile_predictions: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    task_cfg = config["task"]
    aggregation_cfg = config["aggregation"]
    mode = str(task_cfg["mode"]).lower()
    class_labels = _class_labels(task_cfg)
    class_names = _class_names(task_cfg, class_labels)
    label_to_name = dict(zip(class_labels, class_names))

    rows: List[Dict[str, Any]] = []
    for analysis_unit_id, group in tile_predictions.groupby("analysis_unit_id", sort=False):
        base_row: Dict[str, Any] = {
            "analysis_unit_id": analysis_unit_id,
            "study_id": group["study_id"].iloc[0],
            "split": group["split"].iloc[0],
            "tile_count": int(len(group)),
        }
        if "target" in group.columns and group["target"].notna().any():
            base_row["target"] = int(group["target"].dropna().iloc[0])

        passthrough_columns = [
            column
            for column in group.columns
            if column
            not in {
                "analysis_unit_id",
                "study_id",
                "resolved_tile_path",
                "tile_path",
                "filename",
                "split",
                "target",
                "prob_negative",
                "prob_positive",
                "confidence",
                "pred_label",
                "pred_class_name",
            }
            and not column.startswith("prob_class_")
        ]
        for column in passthrough_columns:
            base_row[column] = group[column].iloc[0]

        if mode == "binary":
            probabilities = group["prob_positive"].to_numpy(dtype=float)
            threshold = float(aggregation_cfg.get("threshold", 0.5))
            method = str(aggregation_cfg.get("method", "topk_mean"))
            if method == "mean":
                aggregated_score = float(np.mean(probabilities))
            elif method == "topk_mean":
                top_k = min(int(aggregation_cfg.get("top_k", 10)), len(probabilities))
                aggregated_score = float(np.mean(np.sort(probabilities)[-top_k:]))
            elif method == "proportion_above_threshold":
                aggregated_score = float(np.mean(probabilities >= threshold))
            else:
                raise ValueError(f"Unsupported binary aggregation method: {method}")

            pred_index = int(aggregated_score >= threshold)
            base_row.update(
                {
                    "aggregated_score": aggregated_score,
                    "confidence": float(max(aggregated_score, 1.0 - aggregated_score)),
                    "pred_label": int(class_labels[pred_index]),
                    "pred_class_name": label_to_name[class_labels[pred_index]],
                }
            )
        else:
            probability_columns = [f"prob_class_{label}" for label in class_labels]
            mean_probabilities = group[probability_columns].mean(axis=0).to_numpy(dtype=float)
            pred_index = int(np.argmax(mean_probabilities))
            pred_label = class_labels[pred_index]
            base_row.update(
                {
                    "pred_label": int(pred_label),
                    "pred_class_name": label_to_name[pred_label],
                    "confidence": float(np.max(mean_probabilities)),
                }
            )
            for label_value, probability in zip(class_labels, mean_probabilities):
                base_row[f"prob_score_{label_value}"] = float(probability)

        rows.append(base_row)

    return pd.DataFrame(rows)


def export_top_tiles(tile_predictions: pd.DataFrame, config: Dict[str, Any], run_paths: RunPaths) -> None:
    output_cfg = config["output"]
    top_n = int(output_cfg.get("top_tiles_global_n", 50))
    per_unit_n = int(output_cfg.get("top_tiles_per_analysis_unit", 3))
    mode = str(config["task"]["mode"]).lower()

    if mode == "binary":
        global_top = tile_predictions.sort_values("prob_positive", ascending=False).head(top_n)
        per_unit_top = (
            tile_predictions.sort_values(["analysis_unit_id", "prob_positive"], ascending=[True, False])
            .groupby("analysis_unit_id", sort=False)
            .head(per_unit_n)
        )
    else:
        global_top = tile_predictions.sort_values("confidence", ascending=False).head(top_n)
        per_unit_top = (
            tile_predictions.sort_values(["analysis_unit_id", "confidence"], ascending=[True, False])
            .groupby("analysis_unit_id", sort=False)
            .head(per_unit_n)
        )

    global_top.to_csv(run_paths.predictions_dir / "top_scoring_tiles_global.csv", index=False)
    per_unit_top.to_csv(run_paths.predictions_dir / "top_scoring_tiles_per_analysis_unit.csv", index=False)


def export_summary_figure(analysis_predictions: pd.DataFrame, config: Dict[str, Any], run_paths: RunPaths) -> None:
    if not _normalize_bool(config["output"].get("save_summary_figures", True)):
        return

    mode = str(config["task"]["mode"]).lower()
    if mode == "binary":
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(analysis_predictions["aggregated_score"], bins=20, color="#3a7ca5", edgecolor="white")
        axis.set_xlabel("Analysis-unit positive score")
        axis.set_ylabel("Count")
        axis.set_title("Analysis-unit score distribution")
        figure.tight_layout()
        figure.savefig(run_paths.figures_dir / "analysis_unit_score_histogram.png", dpi=200, bbox_inches="tight")
        plt.close(figure)
        return

    counts = analysis_predictions["pred_label"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([str(index) for index in counts.index], counts.values, color="#3a7ca5")
    axis.set_xlabel("Predicted analysis-unit class")
    axis.set_ylabel("Count")
    axis.set_title("Predicted class counts")
    figure.tight_layout()
    figure.savefig(run_paths.figures_dir / "analysis_unit_predicted_class_counts.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def export_run_summary(
    manifest_df: pd.DataFrame,
    tile_predictions: pd.DataFrame,
    analysis_predictions: pd.DataFrame,
    config: Dict[str, Any],
    run_paths: RunPaths,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "experiment_name": config["experiment_name"],
                "task_name": config["task"]["name"],
                "task_mode": config["task"]["mode"],
                "tile_count": len(tile_predictions),
                "analysis_unit_count": analysis_predictions["analysis_unit_id"].nunique(),
                "studies_observed": manifest_df["study_id"].nunique(),
                "has_ground_truth_labels": int("target" in tile_predictions.columns),
                "checkpoint_path": config["model"]["checkpoint_path"],
            }
        ]
    )
    summary.to_csv(run_paths.metrics_dir / "inference_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    run_paths = build_run_paths(config["output"]["root_dir"], config["experiment_name"])
    logger = configure_logging(run_paths.logs_dir / "infer.log")
    save_config_snapshot(config, run_paths.run_dir / "config_snapshot.yaml")

    manifest_df = load_inference_manifest(config, config_path=config_path, logger=logger)
    manifest_df.to_csv(run_paths.predictions_dir / "resolved_inference_manifest.csv", index=False)
    logger.info("Prepared manifest rows: %d", len(manifest_df))
    logger.info("Analysis units: %d", manifest_df["analysis_unit_id"].nunique())

    device = get_device(config["inference"].get("device"))
    use_amp = bool(config["inference"].get("use_amp", True) and device.type == "cuda")
    logger.info("Device: %s | AMP: %s", device, use_amp)

    model = build_model(
        backbone=str(config["model"]["backbone"]),
        num_classes=int(config["task"]["num_classes"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    ).to(device)

    checkpoint_path = _resolve_path(config["model"]["checkpoint_path"], config_path.parent)
    checkpoint_key = str(config["model"].get("checkpoint_key", "auto"))
    state_dict = load_checkpoint_weights(checkpoint_path, checkpoint_key=checkpoint_key, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(config["model"].get("strict", True)))
    if missing:
        logger.warning("Missing checkpoint keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected checkpoint keys: %s", unexpected)

    loader = build_inference_loader(config, manifest_df)
    tile_predictions = predict_tiles(model, loader, config, device=device, use_amp=use_amp)
    if tile_predictions.empty:
        raise ValueError("Inference produced no tile predictions.")
    tile_predictions.to_csv(run_paths.predictions_dir / "tile_predictions.csv", index=False)

    analysis_predictions = aggregate_predictions(tile_predictions, config)
    analysis_predictions.to_csv(run_paths.predictions_dir / "analysis_unit_predictions.csv", index=False)

    if _normalize_bool(config["output"].get("export_top_tiles", True)):
        export_top_tiles(tile_predictions, config, run_paths)
    export_summary_figure(analysis_predictions, config, run_paths)
    export_run_summary(manifest_df, tile_predictions, analysis_predictions, config, run_paths)
    logger.info("Inference complete. Outputs written to %s", run_paths.run_dir)


if __name__ == "__main__":
    main()
