#!/usr/bin/env python3
"""Tile approved ROI boxes into a training/inference-ready tile manifest.

This script only processes rows where `include_for_tiling == 1` in the ROI
manifest. That makes the QC review step an explicit manual gate.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import openslide
import pandas as pd
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile approved ROIs from an ROI manifest.")
    parser.add_argument("--roi-manifest", required=True, help="CSV ROI manifest from ROI extraction.")
    parser.add_argument("--config", required=True, help="YAML tiling config.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tiles": output_dir / "tiles",
        "manifests": output_dir / "manifests",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def compute_tissue_fraction(image_rgb: np.ndarray, white_threshold: int) -> float:
    gray = np.mean(image_rgb, axis=2)
    return float(np.mean(gray < white_threshold))


def enumerate_tile_candidates(row: pd.Series, tile_size_level0: int, stride_level0: int) -> List[Tuple[int, int]]:
    x0 = int(row["bbox_x"])
    y0 = int(row["bbox_y"])
    width = int(row["bbox_w"])
    height = int(row["bbox_h"])
    x_values = range(x0, max(x0 + 1, x0 + width - tile_size_level0 + 1), stride_level0)
    y_values = range(y0, max(y0 + 1, y0 + height - tile_size_level0 + 1), stride_level0)
    return [(x, y) for y in y_values for x in x_values]


def select_spatial_farthest(candidates: List[Dict[str, Any]], max_tiles: int) -> List[Dict[str, Any]]:
    if len(candidates) <= max_tiles:
        return candidates

    selected = [max(candidates, key=lambda item: item["tissue_fraction"])]
    remaining = [item for item in candidates if item is not selected]

    while remaining and len(selected) < max_tiles:
        def min_distance(candidate: Dict[str, Any]) -> float:
            return min(
                math.hypot(candidate["center_x"] - chosen["center_x"], candidate["center_y"] - chosen["center_y"])
                for chosen in selected
            )

        next_item = max(remaining, key=lambda item: (min_distance(item), item["tissue_fraction"]))
        selected.append(next_item)
        remaining.remove(next_item)

    return selected


def select_candidates(candidates: List[Dict[str, Any]], strategy: str, max_tiles: int) -> List[Dict[str, Any]]:
    if strategy == "scan_order":
        return candidates[:max_tiles]
    if strategy == "spatial_farthest":
        return select_spatial_farthest(candidates, max_tiles)
    raise ValueError(f"Unsupported selection_strategy: {strategy}")


def save_tile(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    tile_size_level0: int,
    tile_size_px: int,
    destination: Path,
) -> None:
    image = slide.read_region((x, y), 0, (tile_size_level0, tile_size_level0)).convert("RGB")
    if tile_size_px != tile_size_level0:
        image = image.resize((tile_size_px, tile_size_px), Image.Resampling.BILINEAR)
    image.save(destination)


def main() -> None:
    args = parse_args()
    roi_manifest_path = Path(args.roi_manifest)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    dirs = ensure_dirs(output_dir)
    setup_logging(dirs["logs"] / "preprocess_tile_rois.log")

    roi_df = pd.read_csv(roi_manifest_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    tiling_cfg = config["tiling"]
    tile_size_level0 = int(tiling_cfg.get("tile_size_level0", 256))
    tile_size_px = int(tiling_cfg.get("tile_size_px", 224))
    stride_level0 = int(tiling_cfg.get("stride_level0", 128))
    min_tissue_fraction = float(tiling_cfg.get("min_tissue_fraction", 0.6))
    max_tiles_per_roi = int(tiling_cfg.get("max_tiles_per_roi", 75))
    white_threshold = int(tiling_cfg.get("white_threshold", 220))
    selection_strategy = str(tiling_cfg.get("selection_strategy", "spatial_farthest"))
    image_format = str(tiling_cfg.get("image_format", "png")).lower()

    approved_df = roi_df.loc[roi_df["include_for_tiling"].apply(truthy)].copy()
    if approved_df.empty:
        raise ValueError(
            "No ROI rows are approved for tiling. Review manifests/roi_manifest.csv and set include_for_tiling=1."
        )

    tile_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for _, row in approved_df.iterrows():
        slide_path = Path(str(row["svs_path"]))
        if not slide_path.exists():
            logging.warning("Skipping missing slide: %s", slide_path)
            continue

        analysis_unit_id = str(row["analysis_unit_id"])
        study_key = str(row.get("study_key", "default"))
        task_name = str(row.get("task", "generic"))
        split_name = str(row.get("split", "unspecified"))

        logging.info("Tiling %s", analysis_unit_id)
        with openslide.OpenSlide(str(slide_path)) as slide:
            candidates: List[Dict[str, Any]] = []
            for tile_x, tile_y in enumerate_tile_candidates(row, tile_size_level0, stride_level0):
                tile = slide.read_region((tile_x, tile_y), 0, (tile_size_level0, tile_size_level0)).convert("RGB")
                tile_array = np.array(tile)
                tissue_fraction = compute_tissue_fraction(tile_array, white_threshold)
                if tissue_fraction < min_tissue_fraction:
                    continue
                candidates.append(
                    {
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tissue_fraction": tissue_fraction,
                        "center_x": tile_x + tile_size_level0 / 2.0,
                        "center_y": tile_y + tile_size_level0 / 2.0,
                    }
                )

            chosen = select_candidates(candidates, selection_strategy, max_tiles_per_roi)
            tile_subdir = dirs["tiles"] / task_name / split_name / "level_0" / study_key
            tile_subdir.mkdir(parents=True, exist_ok=True)

            for tile_info in chosen:
                filename = (
                    f"{analysis_unit_id}__{row['slide_id']}__x{tile_info['tile_x']}_y{tile_info['tile_y']}__l0.{image_format}"
                )
                destination = tile_subdir / filename
                save_tile(
                    slide,
                    int(tile_info["tile_x"]),
                    int(tile_info["tile_y"]),
                    tile_size_level0,
                    tile_size_px,
                    destination,
                )

                tile_row = row.to_dict()
                tile_row.update(
                    {
                        "tile_path": str(destination.relative_to(output_dir)),
                        "filename": filename,
                        "read_level": 0,
                        "tile_x_level0": int(tile_info["tile_x"]),
                        "tile_y_level0": int(tile_info["tile_y"]),
                        "tile_size_level0": tile_size_level0,
                        "tile_width_px": tile_size_px,
                        "tile_height_px": tile_size_px,
                        "tissue_fraction": float(tile_info["tissue_fraction"]),
                        "tile_exists": True,
                    }
                )
                tile_rows.append(tile_row)

            summary_rows.append(
                {
                    "analysis_unit_id": analysis_unit_id,
                    "study_key": study_key,
                    "task": task_name,
                    "split": split_name,
                    "eligible_tiles": len(candidates),
                    "retained_tiles": len(chosen),
                }
            )

    tile_df = pd.DataFrame(tile_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not tile_df.empty:
        priority_columns = [
            "tile_path",
            "filename",
            "slide_id",
            "study_key",
            "sample_id",
            "analysis_unit_id",
            "section_label",
            "task",
            "split",
            "read_level",
            "tile_x_level0",
            "tile_y_level0",
            "tile_size_level0",
            "tile_width_px",
            "tile_height_px",
            "tissue_fraction",
            "tile_exists",
        ]
        ordered_columns = priority_columns + [col for col in tile_df.columns if col not in priority_columns]
        tile_df = tile_df.loc[:, ordered_columns]

    tile_df.to_csv(dirs["manifests"] / "tile_manifest.csv", index=False)
    summary_df.to_csv(dirs["manifests"] / "tiling_summary.csv", index=False)
    logging.info("Wrote %s tiles", len(tile_df))


if __name__ == "__main__":
    main()
