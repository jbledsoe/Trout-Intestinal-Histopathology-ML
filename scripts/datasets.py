"""Dataset and manifest helpers for weakly supervised histology training.

Tiles are sampled from the release manifest, filtered to evaluable material,
and assigned the analysis-unit label of their parent slide/sample. Validation
and reporting aggregate tile outputs back to the analysis-unit level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, PngImagePlugin, ImageFile
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# Allow larger PNG metadata chunks (some tiles contain oversized iCCP/text chunks)
PngImagePlugin.MAX_TEXT_CHUNK = 200 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 400 * 1024 * 1024

# Be tolerant of slightly imperfect PNG files
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_COLUMN_MAP = {
    "enteritis": "enteritis_bin",
    "mononuclear": "mononuclearinfiltration",
}


@dataclass
class DatasetArtifacts:
    """Task-ready manifest subsets and training metadata."""

    manifest_df: pd.DataFrame
    label_column: str
    class_labels: List[int]
    class_weights: np.ndarray
    class_distribution: pd.DataFrame


class OptionalGaussianBlur:
    """Apply a light Gaussian blur with a configurable probability."""

    def __init__(self, probability: float, radius_min: float = 0.1, radius_max: float = 1.0) -> None:
        self.probability = probability
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if np.random.rand() > self.probability:
            return image
        radius = float(np.random.uniform(self.radius_min, self.radius_max))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class HistologyTileDataset(Dataset):
    """PyTorch dataset for manifest-driven tile classification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[transforms.Compose],
        image_mode: str = "RGB",
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.dataframe.iloc[index]
        with Image.open(row["resolved_tile_path"]) as img:
            img.load()
            image = img.convert(self.image_mode).copy()
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": int(row["target_index"]),
            "analysis_unit_id": row["analysis_unit_id"],
            "study_id": row["study_id"],
            "tile_path": row["resolved_tile_path"],
            "filename": row["filename"],
            "split": row["split"],
            "human_label": int(row["target"]),
        }


def _normalize_review_category(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _normalize_quality_status(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(manifest_path, low_memory=False)
    dataframe["task"] = dataframe["task"].fillna("").astype(str)
    dataframe["split"] = dataframe["split"].fillna("").astype(str).str.lower()
    dataframe["analysis_unit_id"] = dataframe["analysis_unit_id"].astype(str)
    dataframe["study_id"] = dataframe["study_id"].astype(str)
    dataframe["filename"] = dataframe["filename"].astype(str)
    dataframe["review_category"] = _normalize_review_category(dataframe["review_category"])
    dataframe["slide_quality_status"] = _normalize_quality_status(dataframe["slide_quality_status"])
    dataframe["is_suspect"] = pd.to_numeric(dataframe["is_suspect"], errors="coerce").fillna(0).astype(int)
    dataframe["is_evaluable"] = pd.to_numeric(dataframe["is_evaluable"], errors="coerce").fillna(0).astype(int)
    dataframe["tile_exists"] = (
        dataframe["tile_exists"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )
    dataframe["read_level"] = pd.to_numeric(dataframe["read_level"], errors="coerce")
    return dataframe


def _resolve_tile_paths(df: pd.DataFrame, data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    resolved = df.copy()
    resolved["resolved_tile_path"] = resolved["tile_path"].map(lambda value: str((data_root / value).resolve()))
    return resolved


def _base_metadata_filter(
    df: pd.DataFrame,
    include_suspect: bool,
    usable_quality_values: Sequence[str],
) -> pd.DataFrame:
    filtered = df.loc[df["is_evaluable"] == 1].copy()
    filtered = filtered.loc[filtered["slide_quality_status"].isin([value.lower() for value in usable_quality_values])]
    filtered = filtered.loc[filtered["review_category"] != "exclude"]
    filtered = filtered.loc[filtered["tile_exists"]]
    filtered = filtered.loc[filtered["read_level"] == 0]
    filtered = filtered.loc[filtered["tile_path"].str.contains("/level_0/", regex=False)]
    if not include_suspect:
        filtered = filtered.loc[filtered["review_category"] != "suspect"]
        filtered = filtered.loc[filtered["is_suspect"] != 1]
    return filtered


def _prepare_targets(df: pd.DataFrame, task_name: str, label_column: str) -> Tuple[pd.DataFrame, List[int]]:
    prepared = df.copy()
    prepared[label_column] = pd.to_numeric(prepared[label_column], errors="coerce")

    if task_name == "enteritis":
        prepared = prepared.loc[prepared[label_column].notna()].copy()
        prepared["target"] = prepared[label_column].astype(int)
        class_labels = [0, 1]
    elif task_name == "mononuclear":
        prepared = prepared.loc[prepared[label_column].notna()].copy()
        prepared = prepared.loc[prepared[label_column].between(1, 5)].copy()
        prepared["target"] = prepared[label_column].astype(int)
        class_labels = [1, 2, 3, 4, 5]
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    prepared["target_index"] = prepared["target"] if task_name == "enteritis" else prepared["target"] - 1
    return prepared, class_labels


def compute_class_weights(df: pd.DataFrame, class_labels: Sequence[int]) -> np.ndarray:
    """Compute inverse-frequency class weights from tile counts."""
    counts = df["target"].value_counts().reindex(class_labels, fill_value=0).astype(float)
    non_zero = counts.replace(0, np.nan)
    weights = len(df) / (len(class_labels) * non_zero)
    weights = weights.fillna(0.0)
    return weights.to_numpy(dtype=np.float32)


def build_task_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    task_name: str,
    include_suspect: bool,
    usable_quality_values: Sequence[str],
) -> DatasetArtifacts:
    """Load and filter the release manifest for a specific training task."""
    label_column = LABEL_COLUMN_MAP[task_name]
    manifest_df = _load_manifest(manifest_path)
    manifest_df = manifest_df.loc[manifest_df["task"] == "enteritis"].copy()
    manifest_df = _base_metadata_filter(
        manifest_df,
        include_suspect=include_suspect,
        usable_quality_values=usable_quality_values,
    )
    manifest_df = _resolve_tile_paths(manifest_df, data_root)
    manifest_df, class_labels = _prepare_targets(manifest_df, task_name, label_column)

    if manifest_df.empty:
        raise ValueError(f"No rows remain after filtering for task '{task_name}'.")

    class_distribution = (
        manifest_df.groupby(["split", "target"])
        .size()
        .reset_index(name="tile_count")
        .sort_values(["split", "target"])
    )
    class_weights = compute_class_weights(manifest_df, class_labels)
    return DatasetArtifacts(
        manifest_df=manifest_df,
        label_column=label_column,
        class_labels=class_labels,
        class_weights=class_weights,
        class_distribution=class_distribution,
    )


def build_transforms(config: Dict[str, object], is_training: bool) -> transforms.Compose:
    """Construct task-appropriate augmentation and normalization."""
    image_size = int(config["image_size"])
    normalize = transforms.Normalize(
        mean=config.get("normalize_mean", [0.485, 0.456, 0.406]),
        std=config.get("normalize_std", [0.229, 0.224, 0.225]),
    )

    if not is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    crop_scale = tuple(config.get("random_resized_crop_scale", [0.9, 1.0]))
    jitter = config.get("color_jitter", {})
    train_transforms: List[object] = [
        transforms.RandomResizedCrop(image_size, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=float(config.get("horizontal_flip_prob", 0.5))),
        transforms.RandomVerticalFlip(p=float(config.get("vertical_flip_prob", 0.5))),
        transforms.RandomRotation(degrees=float(config.get("rotation_degrees", 10.0))),
        transforms.ColorJitter(
            brightness=float(jitter.get("brightness", 0.05)),
            contrast=float(jitter.get("contrast", 0.05)),
            saturation=float(jitter.get("saturation", 0.05)),
            hue=float(jitter.get("hue", 0.0)),
        ),
    ]
    if float(config.get("gaussian_blur_prob", 0.0)) > 0:
        train_transforms.append(
            OptionalGaussianBlur(
                probability=float(config["gaussian_blur_prob"]),
                radius_min=float(config.get("gaussian_blur_radius_min", 0.1)),
                radius_max=float(config.get("gaussian_blur_radius_max", 0.75)),
            )
        )

    train_transforms.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(train_transforms)


def build_weighted_sampler(train_df: pd.DataFrame, class_labels: Sequence[int]) -> WeightedRandomSampler:
    """Construct an optional weighted sampler for imbalanced training."""
    counts = train_df["target"].value_counts().reindex(class_labels, fill_value=0).astype(float)
    inverse = (1.0 / counts.replace(0, np.nan)).fillna(0.0)
    weights = np.array(train_df["target"].map(inverse).to_numpy(dtype=np.float64), dtype=np.float64, copy=True)
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
