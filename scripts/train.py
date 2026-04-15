"""Main training entry point for weakly supervised histology tile models.

The release manifest remains the source of truth for filtering, labels, and
split bookkeeping. Tiles are used as training instances, each tile inherits the
label of its parent analysis unit, and validation is reported primarily after
aggregating tile predictions back to the analysis-unit level.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from aggregate import aggregate_binary_predictions, aggregate_multiclass_predictions
from datasets import (
    HistologyTileDataset,
    build_task_manifest,
    build_transforms,
    compute_class_weights,
    build_weighted_sampler,
)
from losses import build_loss
from metrics import (
    collect_hard_examples,
    collect_top_positive_tiles,
    compute_binary_metrics,
    compute_multiclass_metrics,
    save_confusion_matrix_figure,
    save_roc_pr_curves,
)
from models import build_model
from splits import build_fixed_enteritis_split, build_group_stratified_cv, build_group_stratified_split
from utils import (
    RunPaths,
    build_run_paths,
    configure_logging,
    count_parameters,
    format_class_weights,
    get_device,
    is_metric_improved,
    save_checkpoint,
    save_config_snapshot,
    save_json,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train histology tile classifiers.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build splits/loaders and run a single train + val batch for smoke testing.",
    )
    return parser.parse_args()


def prepare_splits(config: Dict[str, object], manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Build train/val splits for the requested task."""
    task_name = config["task"]["name"]
    label_column = config["task"]["label_column"]
    split_cfg = config["split"]

    if task_name == "enteritis":
        artifacts = build_fixed_enteritis_split(manifest_df, label_column)
    else:
        artifacts = build_group_stratified_split(
            manifest_df,
            label_column=label_column,
            val_fraction=float(split_cfg.get("val_fraction", 0.25)),
            random_seed=int(config["training"]["seed"]),
            allow_fallback_random=bool(split_cfg.get("allow_fallback_random", False)),
        )
    return artifacts.train_df, artifacts.val_df, artifacts.summary


def export_split_tables(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_paths: RunPaths,
    label_column: str,
) -> None:
    """Save train/val split tables and class summaries."""
    train_df.to_csv(run_paths.splits_dir / "train_split.csv", index=False)
    val_df.to_csv(run_paths.splits_dir / "val_split.csv", index=False)

    summary = []
    for split_name, frame in [("train", train_df), ("val", val_df)]:
        label_counts = frame["target"].value_counts().sort_index()
        unit_counts = (
            frame.groupby("analysis_unit_id")["target"].first().value_counts().sort_index()
        )
        for label, count in label_counts.items():
            summary.append(
                {
                    "split": split_name,
                    "target": int(label),
                    "tile_count": int(count),
                    "analysis_unit_count": int(unit_counts.get(label, 0)),
                }
            )
    pd.DataFrame(summary).to_csv(run_paths.splits_dir / "class_distribution_summary.csv", index=False)


def build_dataloaders(
    config: Dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_labels: List[int],
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and validation dataloaders."""
    augmentation_config = config["augmentation"]
    data_config = config["data"]
    train_dataset = HistologyTileDataset(
        dataframe=train_df,
        transform=build_transforms(augmentation_config, is_training=True),
    )
    val_dataset = HistologyTileDataset(
        dataframe=val_df,
        transform=build_transforms(augmentation_config, is_training=False),
    )

    sampler = None
    shuffle = True
    if bool(config["training"].get("use_weighted_sampler", False)):
        sampler = build_weighted_sampler(train_df, class_labels)
        shuffle = False

    common_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 8)),
        "pin_memory": bool(data_config.get("pin_memory", True)),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def get_amp_device_type(device: torch.device) -> str:
    """Resolve the autocast/scaler device type."""
    return "cuda" if device.type == "cuda" else "cpu"


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: amp.GradScaler,
    use_amp: bool,
    logger: logging.Logger,
    amp_device_type: str,
    dry_run: bool = False,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    samples_seen = 0

    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(amp_device_type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        samples_seen += batch_size
        progress.set_postfix(loss=f"{running_loss / max(samples_seen, 1):.4f}")

        if dry_run and step >= 0:
            logger.info("Dry run enabled: stopping after one training batch.")
            break

    return {"loss": running_loss / max(samples_seen, 1)}


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_name: str,
    use_amp: bool,
    amp_device_type: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Collect tile-level validation predictions."""
    model.eval()
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="val", leave=False)
        for step, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)

            with amp.autocast(amp_device_type, enabled=use_amp):
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            for idx in range(images.size(0)):
                row: Dict[str, object] = {
                    "analysis_unit_id": batch["analysis_unit_id"][idx],
                    "study_id": batch["study_id"][idx],
                    "tile_path": batch["tile_path"][idx],
                    "filename": batch["filename"][idx],
                    "split": batch["split"][idx],
                    "target": int(batch["human_label"][idx]),
                }
                if task_name == "enteritis":
                    row["prob_negative"] = float(probabilities[idx, 0])
                    row["prob_positive"] = float(probabilities[idx, 1])
                    row["pred_label"] = int(probabilities[idx, 1] >= 0.5)
                else:
                    for class_index in range(probabilities.shape[1]):
                        row[f"prob_class_{class_index + 1}"] = float(probabilities[idx, class_index])
                    row["pred_label"] = int(np.argmax(probabilities[idx]) + 1)
                rows.append(row)

            if dry_run and step >= 0:
                break

    return pd.DataFrame(rows)


def evaluate_predictions(
    config: Dict[str, object],
    tile_predictions: pd.DataFrame,
    run_paths: RunPaths,
) -> Dict[str, float | object]:
    """Compute tile-level and analysis-unit metrics, then save artifacts."""
    task_name = config["task"]["name"]
    task_cfg = config["task"]
    aggregation_cfg = config["aggregation"]

    tile_predictions.to_csv(run_paths.predictions_dir / "tile_predictions.csv", index=False)
    collect_hard_examples(
        tile_predictions,
        destination=run_paths.predictions_dir / "hard_examples.csv",
        top_n=int(config["output"].get("hard_example_count", 50)),
    )
    if task_name == "enteritis":
        collect_top_positive_tiles(
            tile_predictions,
            destination=run_paths.predictions_dir / "top_positive_tiles.csv",
            top_n=int(config["output"].get("hard_example_count", 50)),
        )

    metrics: Dict[str, float | object] = {}
    if task_name == "enteritis":
        tile_targets = tile_predictions["target"].to_numpy(dtype=int)
        tile_scores = tile_predictions["prob_positive"].to_numpy(dtype=float)
        tile_pred_labels = tile_predictions["pred_label"].to_numpy(dtype=int)
        tile_metrics = compute_binary_metrics(tile_targets, tile_pred_labels, tile_scores)
        aggregation_methods = list(aggregation_cfg.get("compare_methods", ["mean", "topk_mean"]))
        threshold = float(aggregation_cfg.get("threshold", 0.5))
        top_k = int(aggregation_cfg.get("top_k", 10))
        primary_method = str(aggregation_cfg.get("primary_method", "topk_mean"))
        aggregation_summary_rows = []
        analysis_predictions = None
        unit_metrics = None
        for method in aggregation_methods:
            aggregated = aggregate_binary_predictions(
                tile_predictions,
                method=method,
                top_k=top_k,
                threshold=threshold,
            )
            aggregated.to_csv(
                run_paths.predictions_dir / f"analysis_unit_predictions_{method}.csv",
                index=False,
            )
            unit_targets = aggregated["target"].to_numpy(dtype=int)
            unit_scores = aggregated["aggregated_score"].to_numpy(dtype=float)
            unit_pred_labels = aggregated["pred_label"].to_numpy(dtype=int)
            method_metrics = compute_binary_metrics(unit_targets, unit_pred_labels, unit_scores)
            aggregation_summary_rows.append({"method": method, **method_metrics})
            if method == primary_method:
                analysis_predictions = aggregated
                unit_metrics = method_metrics

        if analysis_predictions is None or unit_metrics is None:
            raise ValueError(f"Primary aggregation method '{primary_method}' was not evaluated.")

        pd.DataFrame(aggregation_summary_rows).to_csv(
            run_paths.metrics_dir / "enteritis_aggregation_comparison.csv",
            index=False,
        )
        analysis_predictions.to_csv(run_paths.predictions_dir / "analysis_unit_predictions.csv", index=False)

        metrics.update({f"tile_{key}": value for key, value in tile_metrics.items()})
        metrics.update({f"analysis_unit_{key}": value for key, value in unit_metrics.items()})
        metrics["analysis_unit_primary_aggregation"] = primary_method

        save_confusion_matrix_figure(
            np.array(tile_metrics["confusion_matrix"]),
            class_labels=["0", "1"],
            destination=run_paths.figures_dir / "tile_confusion_matrix.png",
            title="Tile-Level Confusion Matrix",
        )
        save_confusion_matrix_figure(
            np.array(unit_metrics["confusion_matrix"]),
            class_labels=["0", "1"],
            destination=run_paths.figures_dir / "analysis_unit_confusion_matrix.png",
            title="Analysis-Unit Confusion Matrix",
        )
        curve_metrics = save_roc_pr_curves(
            analysis_predictions["target"].to_numpy(dtype=int),
            analysis_predictions["aggregated_score"].to_numpy(dtype=float),
            roc_destination=run_paths.figures_dir / "analysis_unit_roc_curve.png",
            pr_destination=run_paths.figures_dir / "analysis_unit_pr_curve.png",
        )
        metrics.update(curve_metrics)
    else:
        class_labels = [1, 2, 3, 4, 5]
        tile_targets = tile_predictions["target"].to_numpy(dtype=int)
        tile_pred_labels = tile_predictions["pred_label"].to_numpy(dtype=int)
        tile_metrics = compute_multiclass_metrics(tile_targets, tile_pred_labels, class_labels)
        analysis_predictions = aggregate_multiclass_predictions(tile_predictions)
        analysis_predictions.to_csv(
            run_paths.predictions_dir / "analysis_unit_predictions.csv",
            index=False,
        )
        unit_targets = analysis_predictions["target"].to_numpy(dtype=int)
        unit_pred_labels = analysis_predictions["pred_label"].to_numpy(dtype=int)
        unit_metrics = compute_multiclass_metrics(unit_targets, unit_pred_labels, class_labels)

        metrics.update(
            {
                "tile_accuracy": tile_metrics["accuracy"],
                "tile_macro_f1": tile_metrics["macro_f1"],
                "tile_weighted_f1": tile_metrics["weighted_f1"],
                "tile_quadratic_weighted_kappa": tile_metrics["quadratic_weighted_kappa"],
                "tile_confusion_matrix": tile_metrics["confusion_matrix"],
                "tile_per_class": tile_metrics["per_class"],
                "analysis_unit_accuracy": unit_metrics["accuracy"],
                "analysis_unit_macro_f1": unit_metrics["macro_f1"],
                "analysis_unit_weighted_f1": unit_metrics["weighted_f1"],
                "analysis_unit_quadratic_weighted_kappa": unit_metrics["quadratic_weighted_kappa"],
                "analysis_unit_confusion_matrix": unit_metrics["confusion_matrix"],
                "analysis_unit_per_class": unit_metrics["per_class"],
            }
        )
        save_confusion_matrix_figure(
            np.array(tile_metrics["confusion_matrix"]),
            class_labels=[str(label) for label in class_labels],
            destination=run_paths.figures_dir / "tile_confusion_matrix.png",
            title="Tile-Level Confusion Matrix",
        )
        save_confusion_matrix_figure(
            np.array(unit_metrics["confusion_matrix"]),
            class_labels=[str(label) for label in class_labels],
            destination=run_paths.figures_dir / "analysis_unit_confusion_matrix.png",
            title="Analysis-Unit Confusion Matrix",
        )

    pd.DataFrame([metrics]).to_csv(run_paths.metrics_dir / "metrics_latest.csv", index=False)
    save_json(metrics, run_paths.metrics_dir / "metrics_latest.json")
    return metrics


def run_dry_run_smoke_test(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: amp.GradScaler,
    use_amp: bool,
    amp_device_type: str,
    logger: logging.Logger,
) -> None:
    """Run a minimal smoke test without full-epoch metric reporting."""
    logger.info("Dry run: verifying one training batch and one validation batch.")
    train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        use_amp=use_amp,
        logger=logger,
        amp_device_type=amp_device_type,
        dry_run=True,
    )
    logger.info("Dry run training batch succeeded | loss=%.4f", train_metrics["loss"])

    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with amp.autocast(amp_device_type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        logger.info(
            "Dry run validation batch succeeded | batch_size=%d | loss=%.4f | logits_shape=%s",
            images.size(0),
            float(loss.item()),
            tuple(logits.shape),
        )
    logger.info(
        "Dry run complete. Config, filtering, splits, dataloading, forward pass, loss, and backward pass all succeeded."
    )


def maybe_export_cv_manifest(config: Dict[str, object], manifest_df: pd.DataFrame, run_paths: RunPaths) -> None:
    """Optionally save CV folds for later experimentation."""
    cv_cfg = config["split"].get("cross_validation", {})
    if not bool(cv_cfg.get("enabled", False)):
        return
    folds = build_group_stratified_cv(
        manifest_df=manifest_df,
        label_column=config["task"]["label_column"],
        n_splits=int(cv_cfg.get("n_splits", 5)),
        random_seed=int(config["training"]["seed"]),
    )
    for fold_index, (train_df, val_df) in enumerate(folds, start=1):
        train_df.to_csv(run_paths.splits_dir / f"cv_fold_{fold_index}_train.csv", index=False)
        val_df.to_csv(run_paths.splits_dir / f"cv_fold_{fold_index}_val.csv", index=False)


def main() -> None:
    args = parse_args()
    from utils import load_yaml_config  # Imported lazily to keep imports centralized.

    config = load_yaml_config(args.config)
    experiment_name = config["experiment_name"]
    run_paths = build_run_paths(config["output"]["root_dir"], experiment_name)
    logger = configure_logging(run_paths.logs_dir / "train.log")
    save_config_snapshot(config, run_paths.run_dir / "config_snapshot.yaml")

    seed = int(config["training"]["seed"])
    set_global_seed(seed)
    device = get_device(config["training"].get("device"))
    use_amp = bool(config["training"].get("use_amp", True) and device.type == "cuda")
    amp_device_type = get_amp_device_type(device)

    logger.info("Starting experiment: %s", experiment_name)
    logger.info("Device: %s | AMP: %s", device, use_amp)

    dataset_artifacts = build_task_manifest(
        manifest_path=config["data"]["manifest_path"],
        data_root=config["data"]["data_root"],
        task_name=config["task"]["name"],
        include_suspect=bool(config["data"].get("include_suspect", True)),
        usable_quality_values=config["data"].get("usable_slide_quality_status", ["evaluable", "usable"]),
    )
    logger.info("Filtered manifest rows: %d", len(dataset_artifacts.manifest_df))
    dataset_artifacts.class_distribution.to_csv(
        run_paths.splits_dir / "manifest_class_distribution.csv",
        index=False,
    )

    maybe_export_cv_manifest(config, dataset_artifacts.manifest_df, run_paths)

    train_df, val_df, split_summary = prepare_splits(config, dataset_artifacts.manifest_df)
    export_split_tables(train_df, val_df, run_paths, dataset_artifacts.label_column)
    save_json(split_summary, run_paths.splits_dir / "split_summary.json")
    logger.info("Split summary: %s", split_summary)
    train_class_weights = compute_class_weights(train_df, dataset_artifacts.class_labels)
    logger.info("Training split class weights: %s", format_class_weights(train_class_weights))

    model = build_model(
        backbone=config["model"]["backbone"],
        num_classes=int(config["task"]["num_classes"]),
        pretrained=bool(config["model"].get("pretrained", True)),
    ).to(device)
    logger.info("Model parameters: %d", count_parameters(model))

    train_class_weights = np.array(train_class_weights, dtype=np.float32, copy=True)
    class_weights = torch.as_tensor(train_class_weights, dtype=torch.float32, device=device)
    criterion = build_loss(
        loss_name=config["training"]["loss"],
        class_weights=class_weights,
        focal_gamma=float(config["training"].get("focal_gamma", 2.0)),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if bool(config["training"].get("maximize_metric", True)) else "min",
        factor=float(config["training"].get("lr_scheduler_factor", 0.5)),
        patience=int(config["training"].get("lr_scheduler_patience", 2)),
    )
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    maximize_metric = bool(config["training"].get("maximize_metric", True))
    best_metric_value = float("nan")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric_value = float(checkpoint.get("best_metric_value", best_metric_value))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        logger.info("Resumed from checkpoint: %s", args.resume)

    train_loader, val_loader = build_dataloaders(
        config=config,
        train_df=train_df,
        val_df=val_df,
        class_labels=dataset_artifacts.class_labels,
    )

    if args.dry_run:
        run_dry_run_smoke_test(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            amp_device_type=amp_device_type,
            logger=logger,
        )
        return

    monitor_metric = str(config["training"]["monitor_metric"])
    max_epochs = int(config["training"]["max_epochs"])
    early_stopping_patience = int(config["training"]["early_stopping_patience"])
    logger.info("Primary selection metric: %s | maximize=%s", monitor_metric, maximize_metric)

    history_rows: List[Dict[str, float | int | object]] = []
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        logger.info("Epoch %d/%d", epoch + 1, max_epochs)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            logger=logger,
            amp_device_type=amp_device_type,
        )
        tile_predictions = predict(
            model=model,
            loader=val_loader,
            device=device,
            task_name=config["task"]["name"],
            use_amp=use_amp,
            amp_device_type=amp_device_type,
        )
        eval_metrics = evaluate_predictions(config, tile_predictions, run_paths)
        current_metric = float(eval_metrics.get(monitor_metric, float("nan")))
        if np.isnan(current_metric):
            logger.warning(
                "Monitored metric %s is NaN for epoch %d; skipping scheduler update for this epoch.",
                monitor_metric,
                epoch + 1,
            )
        else:
            scheduler.step(current_metric)

        is_best = is_metric_improved(current_metric, best_metric_value, maximize_metric)
        if is_best:
            best_metric_value = current_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric_value": best_metric_value,
            "epochs_without_improvement": epochs_without_improvement,
            "config": config,
        }
        save_checkpoint(
            checkpoint_state,
            checkpoint_path=run_paths.checkpoints_dir / "last.pt",
            best_checkpoint_path=run_paths.checkpoints_dir / "best.pt",
            is_best=is_best,
        )

        epoch_row: Dict[str, float | int | object] = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "elapsed_seconds": round(time.time() - epoch_start, 2),
            monitor_metric: current_metric,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epochs_without_improvement": epochs_without_improvement,
        }
        history_rows.append(epoch_row)
        pd.DataFrame(history_rows).to_csv(run_paths.metrics_dir / "training_history.csv", index=False)

        logger.info(
            "Epoch %d complete | train_loss=%.4f | %s=%.4f | best=%.4f | lr=%.6g | no_improve=%d/%d",
            epoch + 1,
            train_metrics["loss"],
            monitor_metric,
            current_metric,
            best_metric_value,
            optimizer.param_groups[0]["lr"],
            epochs_without_improvement,
            early_stopping_patience,
        )
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Early stopping triggered after %d consecutive epochs without improvement on %s.",
                epochs_without_improvement,
                monitor_metric,
            )
            break


if __name__ == "__main__":
    main()
