"""Utility helpers for weakly supervised histology tile training.

This project trains tile-level CNNs using analysis-unit labels from the
filtered manifest. Tiles inherit their parent analysis-unit label during
training, and model outputs are aggregated back to the analysis-unit level for
primary validation and reporting.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


LOGGER_NAME = "histology_training"


@dataclass
class RunPaths:
    """Resolved output locations for a single training run."""

    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    figures_dir: Path
    splits_dir: Path


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_run_paths(output_root: str | Path, experiment_name: str) -> RunPaths:
    """Create and return standard output folders for a run."""
    run_dir = ensure_dir(Path(output_root) / experiment_name)
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=ensure_dir(run_dir / "checkpoints"),
        logs_dir=ensure_dir(run_dir / "logs"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        predictions_dir=ensure_dir(run_dir / "predictions"),
        figures_dir=ensure_dir(run_dir / "figures"),
        splits_dir=ensure_dir(run_dir / "splits"),
    )


def configure_logging(log_file: str | Path) -> logging.Logger:
    """Configure console and file logging for the training pipeline."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible training and splitting."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON payload with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_config_snapshot(config: Dict[str, Any], destination: str | Path) -> None:
    """Persist the active YAML configuration for reproducibility."""
    with Path(destination).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str | Path,
    best_checkpoint_path: Optional[str | Path] = None,
    is_best: bool = False,
) -> None:
    """Save a last checkpoint and optionally refresh the best checkpoint."""
    torch.save(state, checkpoint_path)
    if is_best and best_checkpoint_path is not None:
        shutil.copy2(checkpoint_path, best_checkpoint_path)


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Resolve the requested torch device."""
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def detach_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Move a tensor to CPU and convert it to numpy."""
    return tensor.detach().cpu().numpy()


def format_class_weights(weights: Iterable[float]) -> str:
    """Format class weights for concise logging."""
    return ", ".join(f"{weight:.4f}" for weight in weights)


def flatten_dict(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested dict one level deep for metrics exports."""
    flattened: Dict[str, Any] = {}
    for key, value in payload.items():
        flattened_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flattened[f"{flattened_key}_{child_key}"] = child_value
        else:
            flattened[flattened_key] = value
    return flattened


def bool_from_config(value: Any) -> bool:
    """Normalize common bool-like config values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division-by-zero crashes in reporting code."""
    return float(numerator / denominator) if denominator else 0.0


def is_metric_improved(
    current_value: float,
    best_value: float,
    maximize: bool,
) -> bool:
    """Compare monitored metrics while treating NaN as non-improving."""
    if np.isnan(current_value):
        return False
    if np.isnan(best_value):
        return True
    return current_value > best_value if maximize else current_value < best_value
