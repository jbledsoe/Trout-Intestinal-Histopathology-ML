#!/usr/bin/env python3
"""
Build analysis-unit train/validation splits and extract train/validation tiles.

Notes
-----
Splitting happens before tiling so no tiles from the same `analysis_unit_id`
can leak across train and validation. Tile labels are inherited from the parent
analysis unit (`is_evaluable` for the quality task, `enteritis_bin` for the
enteritis task), which is appropriate weak supervision for this stage.

Tiling uses the SVS pyramid directly. For each selected ROI, the script reads
tiles at requested OpenSlide read levels, preserves `read_level` in the
manifest, and records both the output pixel size and the corresponding level-0
footprint for every tile.

The manifests intentionally retain non-core metadata columns from the merged
master table so later ordinal-score or villus-length/width modeling can reuse
the same split and tile lineage without rebuilding the pipeline.

Example
-------
python build_splits_and_tiles.py \
  --metadata merged_master_model_table_clean.csv \
  --output-dir tile_output \
  --task both \
  --train-fraction 0.75 \
  --random-seed 42 \
  --read-levels 0 \
  --tile-size-px 224 \
  --tile-size-level0 256 \
  --stride-level0 128 \
  --min-tissue-fraction 0.6 \
  --max-tiles-per-analysis-unit-per-level 75 \
  --selection-strategy spatial_farthest \
  --spatial-bins-x 6 \
  --spatial-bins-y 4 \
  --image-format png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import openslide
    from openslide import OpenSlide
except ImportError:  # pragma: no cover - depends on local OpenSlide install
    openslide = None
    OpenSlide = Any  # type: ignore[misc,assignment]


REQUIRED_METADATA_COLUMNS = [
    "analysis_unit_id",
    "sampleid",
    "study_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "qc_status",
    "is_evaluable",
]

QUALITY_REQUIRED_COLUMNS = REQUIRED_METADATA_COLUMNS
ENTERITIS_REQUIRED_COLUMNS = REQUIRED_METADATA_COLUMNS + ["enteritis_bin"]

ROI_NUMERIC_COLUMNS = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
SPLIT_VALUES = {"train", "val"}
SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg"}
INTERNAL_SAMPLE_ID_COLUMN = "sampleid"
OUTPUT_SAMPLE_ID_COLUMN = "SampleID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create analysis-unit splits and extract tiles from SVS ROIs."
    )
    parser.add_argument("--metadata", required=True, help="Merged master metadata CSV/XLSX path.")
    parser.add_argument("--output-dir", required=True, help="Output directory root.")
    parser.add_argument(
        "--task",
        choices=["quality", "enteritis", "both"],
        default="both",
        help="Task to prepare. Default: both",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.75,
        help="Fraction of analysis units assigned to train. Default: 0.75",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for stratified splitting. Default: 42",
    )
    parser.add_argument(
        "--read-levels",
        nargs="+",
        type=int,
        default=[0],
        help="One or more OpenSlide read levels to tile. Default: 0",
    )
    parser.add_argument(
        "--tile-size-px",
        type=int,
        default=224,
        help="Final saved tile width/height in pixels. Default: 224",
    )
    parser.add_argument(
        "--tile-size-level0",
        type=int,
        default=256,
        help="Tile footprint width/height in level-0 pixels. Default: 256",
    )
    parser.add_argument(
        "--stride-level0",
        type=int,
        default=128,
        help="Grid stride in level-0 pixels. Default: 128",
    )
    parser.add_argument(
        "--min-tissue-fraction",
        type=float,
        default=0.6,
        help="Minimum tissue fraction required to keep a tile. Default: 0.6",
    )
    parser.add_argument(
        "--max-tiles-per-analysis-unit-per-level",
        type=int,
        default=75,
        help="Maximum retained tiles per analysis unit per level. Default: 75",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["scan_order", "spatial_bins", "spatial_farthest"],
        default="spatial_farthest",
        help=(
            "How to select final retained tiles from eligible candidates. "
            "Default: spatial_farthest"
        ),
    )
    parser.add_argument(
        "--spatial-bins-x",
        type=int,
        default=6,
        help="Number of spatial bins across ROI width when using spatial_bins. Default: 6",
    )
    parser.add_argument(
        "--spatial-bins-y",
        type=int,
        default=4,
        help="Number of spatial bins across ROI height when using spatial_bins. Default: 4",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes to use across analysis units. "
            "Use 1 for serial execution. Default: 1"
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=sorted(SUPPORTED_IMAGE_FORMATS),
        default="png",
        help="Saved tile image format. Default: png",
    )
    parser.add_argument(
        "--svs-root",
        default=None,
        help="Optional directory used to resolve parent_svs_name when parent_svs_path is missing.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def ensure_output_dirs(root: Path) -> Dict[str, Path]:
    directories = {
        "root": root,
        "splits": root / "splits",
        "manifests": root / "manifests",
        "tiles": root / "tiles",
        "logs": root / "logs",
        "qc": root / "qc",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def normalize_column_name(value: str) -> str:
    text = value.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [normalize_column_name(str(column)) for column in renamed.columns]
    return renamed


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return np.nan
    return value


def load_table(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
            df = pd.read_csv(handle, sep=delimiter)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path)
    else:
        raise ValueError("Metadata file must be .csv, .xlsx, or .xls")

    df = normalize_columns(df)
    df = df.map(normalize_scalar)
    logging.info("Loaded metadata with %d rows and %d columns", len(df), len(df.columns))
    return df


def resolve_parent_svs_columns(df: pd.DataFrame) -> pd.DataFrame:
    resolved = df.copy()
    if INTERNAL_SAMPLE_ID_COLUMN not in resolved.columns and "sample_id" in resolved.columns:
        resolved = resolved.rename(columns={"sample_id": INTERNAL_SAMPLE_ID_COLUMN})

    if "parent_svs_path" not in resolved.columns:
        resolved["parent_svs_path"] = np.nan
    if "parent_svs_name" not in resolved.columns:
        resolved["parent_svs_name"] = np.nan

    return resolved


def load_master_metadata(metadata_path: Path) -> pd.DataFrame:
    df = resolve_parent_svs_columns(load_table(metadata_path))
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    if df["analysis_unit_id"].isna().any():
        raise ValueError("Metadata contains missing analysis_unit_id values")
    duplicated_ids = df["analysis_unit_id"][df["analysis_unit_id"].duplicated()].astype(str).tolist()
    if duplicated_ids:
        preview = duplicated_ids[:10]
        raise ValueError(
            f"Metadata must contain one row per analysis_unit_id. Duplicate IDs found: {preview}"
        )

    for column in ROI_NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["is_evaluable"] = pd.to_numeric(df["is_evaluable"], errors="coerce")
    if "enteritis_bin" in df.columns:
        df["enteritis_bin"] = pd.to_numeric(df["enteritis_bin"], errors="coerce")

    logging.info("Metadata validation complete for %d analysis units", len(df))
    return df


def validate_cli_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    if args.tile_size_px <= 0:
        raise ValueError("--tile-size-px must be positive")
    if args.tile_size_level0 <= 0:
        raise ValueError("--tile-size-level0 must be positive")
    if args.stride_level0 <= 0:
        raise ValueError("--stride-level0 must be positive")
    if not 0.0 <= args.min_tissue_fraction <= 1.0:
        raise ValueError("--min-tissue-fraction must be between 0 and 1")
    if args.max_tiles_per_analysis_unit_per_level <= 0:
        raise ValueError("--max-tiles-per-analysis-unit-per-level must be positive")
    if any(level < 0 for level in args.read_levels):
        raise ValueError("--read-levels must be non-negative integers")
    if args.spatial_bins_x <= 0 or args.spatial_bins_y <= 0:
        raise ValueError("--spatial-bins-x and --spatial-bins-y must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")


def has_valid_roi(row: pd.Series) -> bool:
    values = [row.get(column) for column in ROI_NUMERIC_COLUMNS]
    if any(pd.isna(value) for value in values):
        return False
    x, y, w, h = values
    return x >= 0 and y >= 0 and w > 0 and h > 0


def record_issue(issues: List[Dict[str, Any]], task: str, row: pd.Series, issue_code: str, detail: str) -> None:
    issues.append(
        {
            "task": task,
            "analysis_unit_id": row.get("analysis_unit_id"),
            INTERNAL_SAMPLE_ID_COLUMN: row.get(INTERNAL_SAMPLE_ID_COLUMN),
            "study_id": row.get("study_id"),
            "parent_svs_path": row.get("parent_svs_path"),
            "issue_code": issue_code,
            "detail": detail,
        }
    )


def prepare_quality_dataset(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    missing_columns = [column for column in QUALITY_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Cannot prepare quality dataset; missing columns: {missing_columns}")

    keep_mask = []
    for _, row in df.iterrows():
        if not has_valid_roi(row):
            record_issue(issues, "quality", row, "invalid_roi", "Missing or invalid ROI coordinates")
            keep_mask.append(False)
            continue
        if pd.isna(row.get("is_evaluable")):
            record_issue(issues, "quality", row, "missing_label", "Missing is_evaluable label")
            keep_mask.append(False)
            continue
        keep_mask.append(True)

    dataset = df.loc[np.asarray(keep_mask)].copy()
    dataset["task"] = "quality"
    dataset["label"] = dataset["is_evaluable"].astype(int)
    logging.info("Prepared quality dataset with %d analysis units", len(dataset))
    return dataset


def prepare_enteritis_dataset(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    missing_columns = [column for column in ENTERITIS_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Cannot prepare enteritis dataset; missing columns: {missing_columns}")

    keep_mask = []
    for _, row in df.iterrows():
        if not has_valid_roi(row):
            record_issue(issues, "enteritis", row, "invalid_roi", "Missing or invalid ROI coordinates")
            keep_mask.append(False)
            continue
        if pd.isna(row.get("is_evaluable")) or int(row.get("is_evaluable")) != 1:
            record_issue(
                issues,
                "enteritis",
                row,
                "not_evaluable",
                "Excluded because is_evaluable != 1",
            )
            keep_mask.append(False)
            continue
        if pd.isna(row.get("enteritis_bin")):
            record_issue(issues, "enteritis", row, "missing_label", "Missing enteritis_bin label")
            keep_mask.append(False)
            continue
        keep_mask.append(True)

    dataset = df.loc[np.asarray(keep_mask)].copy()
    dataset["task"] = "enteritis"
    dataset["label"] = dataset["enteritis_bin"].astype(int)
    logging.info("Prepared enteritis dataset with %d analysis units", len(dataset))
    return dataset


def validate_class_balance_for_split(dataset: pd.DataFrame, label_column: str, train_fraction: float) -> None:
    class_counts = dataset[label_column].value_counts().sort_index()
    if class_counts.empty:
        raise ValueError("No rows available for splitting")
    if len(class_counts) < 2:
        raise ValueError(
            f"Stratified splitting requires at least two classes; found {len(class_counts)} class"
        )
    if (class_counts < 2).any():
        too_small = class_counts[class_counts < 2].to_dict()
        raise ValueError(
            "Stratified splitting is impossible because some classes have fewer than 2 "
            f"analysis units: {too_small}"
        )

    n_samples = len(dataset)
    n_classes = len(class_counts)
    n_train = int(round(n_samples * train_fraction))
    n_val = n_samples - n_train
    if n_train < n_classes or n_val < n_classes:
        raise ValueError(
            "Train/validation sizes are too small to preserve all classes in both splits. "
            f"n_samples={n_samples}, n_train={n_train}, n_val={n_val}, n_classes={n_classes}"
        )


def create_stratified_split(
    dataset: pd.DataFrame,
    label_column: str,
    train_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    validate_class_balance_for_split(dataset, label_column, train_fraction)

    train_ids, val_ids = train_test_split(
        dataset["analysis_unit_id"],
        train_size=train_fraction,
        random_state=random_seed,
        stratify=dataset[label_column],
    )
    split_map = {analysis_unit_id: "train" for analysis_unit_id in train_ids}
    split_map.update({analysis_unit_id: "val" for analysis_unit_id in val_ids})

    split_df = dataset.copy()
    split_df["split_group"] = split_df["analysis_unit_id"]
    split_df["split"] = split_df["analysis_unit_id"].map(split_map)
    validate_split_integrity(split_df, label_column)
    return split_df


def validate_split_integrity(split_df: pd.DataFrame, label_column: str) -> None:
    if split_df["split"].isna().any():
        missing_ids = split_df.loc[split_df["split"].isna(), "analysis_unit_id"].tolist()[:10]
        raise ValueError(f"Split assignment missing for analysis_unit_id values: {missing_ids}")

    duplicate_map = split_df.groupby("analysis_unit_id")["split"].nunique()
    leaking_ids = duplicate_map[duplicate_map > 1].index.tolist()
    if leaking_ids:
        raise ValueError(f"Train/val leakage detected for analysis_unit_id values: {leaking_ids[:10]}")

    unique_splits = set(split_df["split"].unique())
    if not unique_splits.issubset(SPLIT_VALUES):
        raise ValueError(f"Unexpected split labels present: {sorted(unique_splits)}")

    class_by_split = split_df.groupby("split")[label_column].nunique().to_dict()
    if min(class_by_split.values()) < split_df[label_column].nunique():
        raise ValueError(
            f"At least one split is missing a class after stratification: {class_by_split}"
        )


def get_manifest_columns_first(df: pd.DataFrame, required_columns: Sequence[str]) -> List[str]:
    core = [column for column in required_columns if column in df.columns]
    extras = [column for column in df.columns if column not in core]
    return core + extras


def write_split_manifest(split_df: pd.DataFrame, task: str, split_path: Path) -> None:
    if task == "quality":
        required = [
            "analysis_unit_id",
            INTERNAL_SAMPLE_ID_COLUMN,
            "study_id",
            "split",
            "is_evaluable",
            "parent_svs_path",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "qc_status",
        ]
    else:
        required = [
            "analysis_unit_id",
            INTERNAL_SAMPLE_ID_COLUMN,
            "study_id",
            "split",
            "is_evaluable",
            "enteritis_bin",
            "parent_svs_path",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "qc_status",
        ]

    ordered = get_manifest_columns_first(split_df, required)
    export_df = split_df.loc[:, ordered].rename(
        columns={INTERNAL_SAMPLE_ID_COLUMN: OUTPUT_SAMPLE_ID_COLUMN}
    )
    export_df.to_csv(split_path, index=False)
    logging.info("Wrote %s split manifest: %s", task, split_path)


def write_tile_manifest(tile_records: List[Dict[str, Any]], task: str, output_path: Path) -> pd.DataFrame:
    tile_df = pd.DataFrame(tile_records)
    if tile_df.empty:
        tile_df = pd.DataFrame(
            columns=[
                "tile_path",
                "task",
                "split",
                "analysis_unit_id",
                OUTPUT_SAMPLE_ID_COLUMN,
                "study_id",
                "parent_svs_path",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "read_level",
                "tile_x_level0",
                "tile_y_level0",
                "tile_size_level0",
                "tile_width_px",
                "tile_height_px",
                "tile_width_level0",
                "tile_height_level0",
                "tissue_fraction",
                "is_evaluable",
                "enteritis_bin",
                "qc_status",
            ]
        )

    required = [
        "tile_path",
        "task",
        "split",
        "analysis_unit_id",
        OUTPUT_SAMPLE_ID_COLUMN,
        "study_id",
        "parent_svs_path",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "read_level",
        "tile_x_level0",
        "tile_y_level0",
        "tile_size_level0",
        "tile_width_px",
        "tile_height_px",
        "tile_width_level0",
        "tile_height_level0",
        "tissue_fraction",
        "is_evaluable",
        "enteritis_bin",
        "qc_status",
    ]
    export_df = tile_df.copy()
    if INTERNAL_SAMPLE_ID_COLUMN in export_df.columns:
        export_df = export_df.rename(columns={INTERNAL_SAMPLE_ID_COLUMN: OUTPUT_SAMPLE_ID_COLUMN})
    ordered = get_manifest_columns_first(export_df, required)
    export_df.loc[:, ordered].to_csv(output_path, index=False)
    logging.info("Wrote %s tile manifest: %s", task, output_path)
    return tile_df


def resolve_parent_svs_path(row: pd.Series, svs_root: Optional[Path]) -> Path:
    path_value = row.get("parent_svs_path")
    name_value = row.get("parent_svs_name")

    if isinstance(path_value, str) and path_value.strip():
        candidate = Path(path_value).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Parent SVS path does not exist: {candidate}")

    if isinstance(name_value, str) and name_value.strip() and svs_root is not None:
        candidate = (svs_root / name_value).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"SVS file not found under --svs-root: {candidate}")

    raise FileNotFoundError(
        "Missing parent_svs_path and no resolvable parent_svs_name/--svs-root fallback was available"
    )


def open_slide(slide_path: Path) -> OpenSlide:
    if openslide is None:
        raise ImportError(
            "openslide-python is required. Install it along with the OpenSlide shared library."
        )
    return openslide.OpenSlide(str(slide_path))


def get_level_downsample(slide: OpenSlide, read_level: int) -> float:
    if read_level < 0 or read_level >= slide.level_count:
        raise ValueError(
            f"Requested read_level {read_level} is not available; slide has levels 0-{slide.level_count - 1}"
        )
    return float(slide.level_downsamples[read_level])


def compute_level0_tile_extent(tile_size_level0: int, read_level: int, slide: OpenSlide) -> int:
    downsample = get_level_downsample(slide, read_level)
    level_pixels = max(1, int(math.ceil(tile_size_level0 / downsample)))
    return int(round(level_pixels * downsample))


def generate_positions(start: int, span: int, tile_size: int, stride: int) -> List[int]:
    if span <= tile_size:
        return [start]

    stop = start + span - tile_size
    positions = list(range(start, stop + 1, stride))
    if not positions:
        positions = [start]
    last_aligned = start + span - tile_size
    if positions[-1] != last_aligned:
        positions.append(last_aligned)
    return positions


def generate_tile_grid(
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    tile_size_level0: int,
    stride_level0: int,
) -> List[Tuple[int, int, int, int]]:
    x_positions = generate_positions(bbox_x, bbox_w, tile_size_level0, stride_level0)
    y_positions = generate_positions(bbox_y, bbox_h, tile_size_level0, stride_level0)
    grid: List[Tuple[int, int, int, int]] = []
    max_x = bbox_x + bbox_w
    max_y = bbox_y + bbox_h
    for y_pos in y_positions:
        for x_pos in x_positions:
            actual_w = min(tile_size_level0, max_x - x_pos)
            actual_h = min(tile_size_level0, max_y - y_pos)
            if actual_w > 0 and actual_h > 0:
                grid.append((x_pos, y_pos, actual_w, actual_h))
    return grid


def make_spatial_bin_key(
    tile_x_level0: int,
    tile_y_level0: int,
    tile_width_level0: int,
    tile_height_level0: int,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    bins_x: int,
    bins_y: int,
) -> Tuple[int, int]:
    center_x = tile_x_level0 + (tile_width_level0 / 2.0)
    center_y = tile_y_level0 + (tile_height_level0 / 2.0)

    rel_x = 0.0 if bbox_w <= 0 else (center_x - bbox_x) / float(bbox_w)
    rel_y = 0.0 if bbox_h <= 0 else (center_y - bbox_y) / float(bbox_h)
    rel_x = min(max(rel_x, 0.0), 0.999999)
    rel_y = min(max(rel_y, 0.0), 0.999999)

    bin_x = min(int(rel_x * bins_x), bins_x - 1)
    bin_y = min(int(rel_y * bins_y), bins_y - 1)
    return (bin_x, bin_y)


def select_spatially_diverse_candidates(
    candidates: List[Dict[str, Any]],
    max_tiles: int,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    bins_x: int,
    bins_y: int,
) -> List[Dict[str, Any]]:
    if len(candidates) <= max_tiles:
        return candidates

    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        key = make_spatial_bin_key(
            tile_x_level0=int(candidate["tile_x_level0"]),
            tile_y_level0=int(candidate["tile_y_level0"]),
            tile_width_level0=int(candidate["tile_width_level0"]),
            tile_height_level0=int(candidate["tile_height_level0"]),
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            bins_x=bins_x,
            bins_y=bins_y,
        )
        grouped.setdefault(key, []).append(candidate)

    for key in grouped:
        grouped[key] = sorted(
            grouped[key],
            key=lambda d: (
                -float(d["tissue_fraction"]),
                int(d["tile_y_level0"]),
                int(d["tile_x_level0"]),
            ),
        )

    ordered_keys = sorted(grouped.keys(), key=lambda k: (k[1], k[0]))
    selected: List[Dict[str, Any]] = []

    while len(selected) < max_tiles:
        added_this_round = False
        for key in ordered_keys:
            bucket = grouped[key]
            if bucket:
                selected.append(bucket.pop(0))
                added_this_round = True
                if len(selected) >= max_tiles:
                    break
        if not added_this_round:
            break

    return selected




def _candidate_center_normalized(
    candidate: Dict[str, Any],
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
) -> Tuple[float, float]:
    center_x = float(candidate["tile_x_level0"]) + (float(candidate["tile_width_level0"]) / 2.0)
    center_y = float(candidate["tile_y_level0"]) + (float(candidate["tile_height_level0"]) / 2.0)
    x_norm = 0.0 if bbox_w <= 0 else (center_x - bbox_x) / float(bbox_w)
    y_norm = 0.0 if bbox_h <= 0 else (center_y - bbox_y) / float(bbox_h)
    return (min(max(x_norm, 0.0), 1.0), min(max(y_norm, 0.0), 1.0))


def select_spatially_diverse_farthest_candidates(
    candidates: List[Dict[str, Any]],
    max_tiles: int,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    bins_x: int,
    bins_y: int,
) -> List[Dict[str, Any]]:
    if len(candidates) <= max_tiles:
        return candidates

    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        key = make_spatial_bin_key(
            tile_x_level0=int(candidate["tile_x_level0"]),
            tile_y_level0=int(candidate["tile_y_level0"]),
            tile_width_level0=int(candidate["tile_width_level0"]),
            tile_height_level0=int(candidate["tile_height_level0"]),
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            bins_x=bins_x,
            bins_y=bins_y,
        )
        grouped.setdefault(key, []).append(candidate)

    for key in grouped:
        grouped[key] = sorted(
            grouped[key],
            key=lambda d: (
                -float(d["tissue_fraction"]),
                int(d["tile_y_level0"]),
                int(d["tile_x_level0"]),
            ),
        )

    occupied_keys = sorted(grouped.keys(), key=lambda k: (k[1], k[0]))
    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    # Stage 1: seed one strong tile per occupied bin, ordered by best tissue fraction.
    bin_seeds = []
    for key in occupied_keys:
        if grouped[key]:
            top = grouped[key][0]
            bin_seeds.append((float(top["tissue_fraction"]), key, top))
    for _, key, top in sorted(bin_seeds, key=lambda x: (-x[0], x[1][1], x[1][0])):
        cid = id(top)
        if cid in selected_ids:
            continue
        selected.append(top)
        selected_ids.add(cid)
        if len(selected) >= max_tiles:
            return selected

    selected_centers = [
        _candidate_center_normalized(c, bbox_x, bbox_y, bbox_w, bbox_h)
        for c in selected
    ]

    remaining = [c for c in candidates if id(c) not in selected_ids]

    # Stage 2: repeatedly add the candidate farthest from the current selected set,
    # with tissue fraction as a secondary tie-breaker.
    while remaining and len(selected) < max_tiles:
        best_idx = None
        best_score = None
        for idx, candidate in enumerate(remaining):
            cx, cy = _candidate_center_normalized(candidate, bbox_x, bbox_y, bbox_w, bbox_h)
            min_dist = min(math.hypot(cx - sx, cy - sy) for sx, sy in selected_centers)
            score = (min_dist, float(candidate["tissue_fraction"]))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_centers.append(_candidate_center_normalized(chosen, bbox_x, bbox_y, bbox_w, bbox_h))

    return selected
def read_tile(
    slide: OpenSlide,
    tile_x_level0: int,
    tile_y_level0: int,
    tile_width_level0: int,
    tile_height_level0: int,
    read_level: int,
) -> Image.Image:
    downsample = get_level_downsample(slide, read_level)
    level_width = max(1, int(math.ceil(tile_width_level0 / downsample)))
    level_height = max(1, int(math.ceil(tile_height_level0 / downsample)))
    region = slide.read_region(
        (int(tile_x_level0), int(tile_y_level0)),
        read_level,
        (level_width, level_height),
    )
    return region.convert("RGB")


def resize_tile_for_output(tile_image: Image.Image, tile_size_px: int) -> Image.Image:
    if tile_image.size == (tile_size_px, tile_size_px):
        return tile_image
    return tile_image.resize((tile_size_px, tile_size_px), resample=Image.Resampling.BILINEAR)


def estimate_tissue_fraction(tile_rgb: np.ndarray) -> float:
    if cv2 is not None:
        hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        tissue_mask = np.logical_or(
            gray < 235,
            np.logical_and(hsv[:, :, 1] > 20, hsv[:, :, 2] < 250),
        )
        return float(np.mean(tissue_mask))

    tile_float = tile_rgb.astype(np.float32)
    gray = tile_float.mean(axis=2)
    channel_spread = tile_float.max(axis=2) - tile_float.min(axis=2)
    tissue_mask = np.logical_or(gray < 235.0, channel_spread > 12.0)
    return float(np.mean(tissue_mask))


def safe_token(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "unknown"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def build_tile_output_path(
    output_root: Path,
    task: str,
    split: str,
    read_level: int,
    image_format: str,
    row: pd.Series,
    tile_x_level0: int,
    tile_y_level0: int,
) -> Path:
    study_token = safe_token(row.get("study_id"))
    analysis_unit_token = safe_token(row.get("analysis_unit_id"))
    sample_token = safe_token(row.get(INTERNAL_SAMPLE_ID_COLUMN))
    filename = (
        f"{analysis_unit_token}__{sample_token}"
        f"__x{tile_x_level0}_y{tile_y_level0}"
        f"__l{read_level}.{image_format}"
    )
    return output_root / "tiles" / task / split / f"level_{read_level}" / study_token / filename


def build_tile_record(
    row: pd.Series,
    task: str,
    split: str,
    tile_path: Path,
    output_root: Path,
    read_level: int,
    tile_x_level0: int,
    tile_y_level0: int,
    tile_size_level0: int,
    tile_width_px: int,
    tile_height_px: int,
    tile_width_level0: int,
    tile_height_level0: int,
    tissue_fraction: float,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "tile_path": tile_path.relative_to(output_root).as_posix(),
        "task": task,
        "split": split,
        "analysis_unit_id": row.get("analysis_unit_id"),
        OUTPUT_SAMPLE_ID_COLUMN: row.get(INTERNAL_SAMPLE_ID_COLUMN),
        "study_id": row.get("study_id"),
        "parent_svs_path": row.get("parent_svs_path"),
        "bbox_x": row.get("bbox_x"),
        "bbox_y": row.get("bbox_y"),
        "bbox_w": row.get("bbox_w"),
        "bbox_h": row.get("bbox_h"),
        "read_level": read_level,
        "tile_x_level0": tile_x_level0,
        "tile_y_level0": tile_y_level0,
        "tile_size_level0": tile_size_level0,
        "tile_width_px": tile_width_px,
        "tile_height_px": tile_height_px,
        "tile_width_level0": tile_width_level0,
        "tile_height_level0": tile_height_level0,
        "tissue_fraction": tissue_fraction,
        "is_evaluable": row.get("is_evaluable"),
        "enteritis_bin": row.get("enteritis_bin"),
        "qc_status": row.get("qc_status"),
    }
    for column, value in row.items():
        if column == INTERNAL_SAMPLE_ID_COLUMN:
            continue
        if column not in record:
            record[column] = value
    return record


def extract_tiles_for_level(
    slide: OpenSlide,
    row: pd.Series,
    task: str,
    split: str,
    output_root: Path,
    read_level: int,
    tile_size_px: int,
    tile_size_level0: int,
    stride_level0: int,
    min_tissue_fraction: float,
    max_tiles_per_analysis_unit_per_level: int,
    selection_strategy: str,
    spatial_bins_x: int,
    spatial_bins_y: int,
    image_format: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bbox_x = int(row["bbox_x"])
    bbox_y = int(row["bbox_y"])
    bbox_w = int(row["bbox_w"])
    bbox_h = int(row["bbox_h"])
    grid = generate_tile_grid(
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        tile_size_level0=tile_size_level0,
        stride_level0=stride_level0,
    )

    eligible_candidates: List[Dict[str, Any]] = []
    considered = 0
    for tile_x_level0, tile_y_level0, tile_width_level0, tile_height_level0 in grid:
        considered += 1
        tile_image = read_tile(
            slide=slide,
            tile_x_level0=tile_x_level0,
            tile_y_level0=tile_y_level0,
            tile_width_level0=tile_width_level0,
            tile_height_level0=tile_height_level0,
            read_level=read_level,
        )
        resized = resize_tile_for_output(tile_image, tile_size_px)
        tissue_fraction = estimate_tissue_fraction(np.asarray(resized))
        if tissue_fraction < min_tissue_fraction:
            continue

        eligible_candidates.append(
            {
                "tile_x_level0": tile_x_level0,
                "tile_y_level0": tile_y_level0,
                "tile_width_level0": tile_width_level0,
                "tile_height_level0": tile_height_level0,
                "tissue_fraction": tissue_fraction,
            }
        )

    if selection_strategy == "scan_order":
        selected_candidates = eligible_candidates[:max_tiles_per_analysis_unit_per_level]
    elif selection_strategy == "spatial_bins":
        selected_candidates = select_spatially_diverse_candidates(
            candidates=eligible_candidates,
            max_tiles=max_tiles_per_analysis_unit_per_level,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            bins_x=spatial_bins_x,
            bins_y=spatial_bins_y,
        )
    else:
        selected_candidates = select_spatially_diverse_farthest_candidates(
            candidates=eligible_candidates,
            max_tiles=max_tiles_per_analysis_unit_per_level,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            bins_x=spatial_bins_x,
            bins_y=spatial_bins_y,
        )

    kept_records: List[Dict[str, Any]] = []
    for candidate in selected_candidates:
        tile_x_level0 = int(candidate["tile_x_level0"])
        tile_y_level0 = int(candidate["tile_y_level0"])
        tile_width_level0 = int(candidate["tile_width_level0"])
        tile_height_level0 = int(candidate["tile_height_level0"])
        tissue_fraction = float(candidate["tissue_fraction"])

        tile_image = read_tile(
            slide=slide,
            tile_x_level0=tile_x_level0,
            tile_y_level0=tile_y_level0,
            tile_width_level0=tile_width_level0,
            tile_height_level0=tile_height_level0,
            read_level=read_level,
        )
        resized = resize_tile_for_output(tile_image, tile_size_px)

        tile_path = build_tile_output_path(
            output_root=output_root,
            task=task,
            split=split,
            read_level=read_level,
            image_format=image_format,
            row=row,
            tile_x_level0=tile_x_level0,
            tile_y_level0=tile_y_level0,
        )
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: Dict[str, Any] = {}
        if image_format in {"jpg", "jpeg"}:
            save_kwargs["quality"] = 95
        resized.save(tile_path, **save_kwargs)

        kept_records.append(
            build_tile_record(
                row=row,
                task=task,
                split=split,
                tile_path=tile_path,
                output_root=output_root,
                read_level=read_level,
                tile_x_level0=tile_x_level0,
                tile_y_level0=tile_y_level0,
                tile_size_level0=tile_size_level0,
                tile_width_px=tile_size_px,
                tile_height_px=tile_size_px,
                tile_width_level0=tile_width_level0,
                tile_height_level0=tile_height_level0,
                tissue_fraction=tissue_fraction,
            )
        )

    summary = {
        "analysis_unit_id": row.get("analysis_unit_id"),
        "read_level": read_level,
        "selection_strategy": selection_strategy,
        "n_candidate_tiles": len(grid),
        "n_tiles_considered": considered,
        "n_tiles_eligible": len(eligible_candidates),
        "n_tiles_kept": len(kept_records),
    }
    return kept_records, summary


def extract_tiles_for_analysis_unit(
    row: pd.Series,
    task: str,
    output_root: Path,
    read_levels: Sequence[int],
    tile_size_px: int,
    tile_size_level0: int,
    stride_level0: int,
    min_tissue_fraction: float,
    max_tiles_per_analysis_unit_per_level: int,
    selection_strategy: str,
    spatial_bins_x: int,
    spatial_bins_y: int,
    image_format: str,
    svs_root: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split = str(row["split"])
    slide_path = resolve_parent_svs_path(row, svs_root)
    row = row.copy()
    row["parent_svs_path"] = str(slide_path)

    tile_records: List[Dict[str, Any]] = []
    extraction_summaries: List[Dict[str, Any]] = []
    slide = open_slide(slide_path)
    try:
        for read_level in read_levels:
            _ = compute_level0_tile_extent(tile_size_level0, read_level, slide)
            level_records, level_summary = extract_tiles_for_level(
                slide=slide,
                row=row,
                task=task,
                split=split,
                output_root=output_root,
                read_level=read_level,
                tile_size_px=tile_size_px,
                tile_size_level0=tile_size_level0,
                stride_level0=stride_level0,
                min_tissue_fraction=min_tissue_fraction,
                max_tiles_per_analysis_unit_per_level=max_tiles_per_analysis_unit_per_level,
                selection_strategy=selection_strategy,
                spatial_bins_x=spatial_bins_x,
                spatial_bins_y=spatial_bins_y,
                image_format=image_format,
            )
            tile_records.extend(level_records)
            extraction_summaries.append(level_summary)
    finally:
        slide.close()
    return tile_records, extraction_summaries


def summarize_task(
    task: str,
    split_df: pd.DataFrame,
    tile_df: pd.DataFrame,
    label_column: str,
    read_levels: Sequence[int],
) -> pd.DataFrame:
    class_counts_train = split_df.loc[split_df["split"] == "train", label_column].value_counts().sort_index()
    class_counts_val = split_df.loc[split_df["split"] == "val", label_column].value_counts().sort_index()
    rows: List[Dict[str, Any]] = []
    for read_level in read_levels:
        rows.append(
            {
                "task": task,
                "read_level": read_level,
                "n_analysis_units_total": int(len(split_df)),
                "n_analysis_units_train": int((split_df["split"] == "train").sum()),
                "n_analysis_units_val": int((split_df["split"] == "val").sum()),
                "class_counts_train": json.dumps({str(k): int(v) for k, v in class_counts_train.items()}),
                "class_counts_val": json.dumps({str(k): int(v) for k, v in class_counts_val.items()}),
                "n_tiles_train": int(
                    ((tile_df["split"] == "train") & (tile_df["read_level"] == read_level)).sum()
                    if not tile_df.empty
                    else 0
                ),
                "n_tiles_val": int(
                    ((tile_df["split"] == "val") & (tile_df["read_level"] == read_level)).sum()
                    if not tile_df.empty
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_worker_job(
    row: pd.Series,
    task: str,
    output_root: Path,
    args: argparse.Namespace,
    svs_root: Optional[Path],
) -> Dict[str, Any]:
    return {
        "row": row.to_dict(),
        "task": task,
        "output_root": str(output_root),
        "read_levels": list(args.read_levels),
        "tile_size_px": int(args.tile_size_px),
        "tile_size_level0": int(args.tile_size_level0),
        "stride_level0": int(args.stride_level0),
        "min_tissue_fraction": float(args.min_tissue_fraction),
        "max_tiles_per_analysis_unit_per_level": int(args.max_tiles_per_analysis_unit_per_level),
        "selection_strategy": str(args.selection_strategy),
        "spatial_bins_x": int(args.spatial_bins_x),
        "spatial_bins_y": int(args.spatial_bins_y),
        "image_format": str(args.image_format),
        "svs_root": str(svs_root) if svs_root is not None else None,
    }


def run_analysis_unit_job(job: Mapping[str, Any]) -> Dict[str, Any]:
    row = pd.Series(job["row"])
    output_root = Path(str(job["output_root"]))
    svs_root = Path(str(job["svs_root"])) if job.get("svs_root") else None
    task = str(job["task"])
    split = str(row["split"])
    analysis_unit_id = str(row["analysis_unit_id"])
    try:
        tile_records, summaries = extract_tiles_for_analysis_unit(
            row=row,
            task=task,
            output_root=output_root,
            read_levels=list(job["read_levels"]),
            tile_size_px=int(job["tile_size_px"]),
            tile_size_level0=int(job["tile_size_level0"]),
            stride_level0=int(job["stride_level0"]),
            min_tissue_fraction=float(job["min_tissue_fraction"]),
            max_tiles_per_analysis_unit_per_level=int(job["max_tiles_per_analysis_unit_per_level"]),
            selection_strategy=str(job["selection_strategy"]),
            spatial_bins_x=int(job["spatial_bins_x"]),
            spatial_bins_y=int(job["spatial_bins_y"]),
            image_format=str(job["image_format"]),
            svs_root=svs_root,
        )
        extraction_rows = [
            {"task": task, "split": split, **summary}
            for summary in summaries
        ]
        return {
            "ok": True,
            "analysis_unit_id": analysis_unit_id,
            "row": job["row"],
            "tile_records": tile_records,
            "extraction_log_rows": extraction_rows,
        }
    except Exception as exc:
        return {
            "ok": False,
            "analysis_unit_id": analysis_unit_id,
            "row": job["row"],
            "error": repr(exc),
        }


def sort_tile_records(tile_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        tile_records,
        key=lambda r: (
            str(r.get("task", "")),
            str(r.get("split", "")),
            str(r.get("analysis_unit_id", "")),
            int(r.get("read_level", -1)),
            int(r.get("tile_y_level0", -1)),
            int(r.get("tile_x_level0", -1)),
            str(r.get("tile_path", "")),
        ),
    )


def sort_extraction_rows(extraction_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        extraction_rows,
        key=lambda r: (
            str(r.get("task", "")),
            str(r.get("split", "")),
            str(r.get("analysis_unit_id", "")),
            int(r.get("read_level", -1)),
        ),
    )


def process_task(
    task: str,
    dataset: pd.DataFrame,
    output_dirs: Mapping[str, Path],
    args: argparse.Namespace,
    issues: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_column = "is_evaluable" if task == "quality" else "enteritis_bin"
    split_df = create_stratified_split(
        dataset=dataset,
        label_column=label_column,
        train_fraction=args.train_fraction,
        random_seed=args.random_seed,
    )

    split_manifest_path = output_dirs["splits"] / f"{task}_split_manifest.csv"
    write_split_manifest(split_df, task, split_manifest_path)

    svs_root = Path(args.svs_root).expanduser().resolve() if args.svs_root else None
    tile_records: List[Dict[str, Any]] = []
    extraction_log_rows: List[Dict[str, Any]] = []

    jobs = [
        build_worker_job(
            row=row,
            task=task,
            output_root=output_dirs["root"],
            args=args,
            svs_root=svs_root,
        )
        for _, row in split_df.iterrows()
    ]

    if args.num_workers == 1:
        results = [run_analysis_unit_job(job) for job in jobs]
    else:
        max_workers = min(int(args.num_workers), max(1, len(jobs)), os.cpu_count() or 1)
        logging.info("Running %s with %d worker processes", task, max_workers)
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_analysis_unit_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    result_by_id = {str(result["analysis_unit_id"]): result for result in results}

    for _, row in split_df.iterrows():
        analysis_unit_id = str(row["analysis_unit_id"])
        result = result_by_id[analysis_unit_id]
        if result["ok"]:
            tile_records.extend(result["tile_records"])
            for summary in result["extraction_log_rows"]:
                extraction_log_rows.append(summary)
                if summary["n_tiles_kept"] == 0:
                    record_issue(
                        issues,
                        task,
                        row,
                        "zero_tiles_retained",
                        f"No tiles passed the tissue filter at read_level={summary['read_level']}",
                    )
        else:
            record_issue(
                issues,
                task,
                row,
                "tile_extraction_failed",
                str(result["error"]),
            )
            logging.error("Failed tile extraction for %s (%s): %s", analysis_unit_id, task, result["error"])

    tile_records = sort_tile_records(tile_records)
    extraction_log_rows = sort_extraction_rows(extraction_log_rows)

    tile_manifest_path = output_dirs["manifests"] / f"{task}_tile_manifest.csv"
    tile_df = write_tile_manifest(tile_records, task, tile_manifest_path)

    extraction_log_df = pd.DataFrame(extraction_log_rows)
    extraction_log_path = output_dirs["qc"] / f"{task}_tile_extraction_summary.csv"
    extraction_log_df.to_csv(extraction_log_path, index=False)
    logging.info("Wrote %s extraction summary: %s", task, extraction_log_path)

    summary_df = summarize_task(task, split_df, tile_df, label_column, args.read_levels)
    summary_path = output_dirs["qc"] / f"{task}_dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Wrote %s dataset summary: %s", task, summary_path)
    return split_df, tile_df, summary_df


def main() -> None:
    args = parse_args()
    validate_cli_args(args)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_dirs = ensure_output_dirs(output_root)
    setup_logging(output_dirs["logs"] / "build_splits_and_tiles.log")

    metadata_path = Path(args.metadata).expanduser().resolve()
    metadata_df = load_master_metadata(metadata_path)
    issues: List[Dict[str, Any]] = []

    tasks_to_run: List[Tuple[str, pd.DataFrame]] = []
    if args.task in {"quality", "both"}:
        tasks_to_run.append(("quality", prepare_quality_dataset(metadata_df, issues)))
    if args.task in {"enteritis", "both"}:
        tasks_to_run.append(("enteritis", prepare_enteritis_dataset(metadata_df, issues)))

    overall_summaries: List[pd.DataFrame] = []
    for task_name, task_df in tasks_to_run:
        if task_df.empty:
            raise ValueError(f"No eligible rows remained for task '{task_name}' after filtering")
        _, _, summary_df = process_task(
            task=task_name,
            dataset=task_df,
            output_dirs=output_dirs,
            args=args,
            issues=issues,
        )
        overall_summaries.append(summary_df)

    issues_df = pd.DataFrame(issues)
    issues_path = output_dirs["qc"] / "data_issues.csv"
    issues_df.to_csv(issues_path, index=False)
    logging.info("Wrote issue log: %s", issues_path)

    if overall_summaries:
        combined_summary = pd.concat(overall_summaries, ignore_index=True)
        combined_summary_path = output_dirs["qc"] / "dataset_summary_all_tasks.csv"
        combined_summary.to_csv(combined_summary_path, index=False)
        logging.info("Wrote combined dataset summary: %s", combined_summary_path)

    logging.info("Completed split generation and tile extraction.")


if __name__ == "__main__":
    main()
