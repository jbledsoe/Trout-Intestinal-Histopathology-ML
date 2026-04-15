#!/usr/bin/env python3
"""Public-safe SVS ROI extraction with a manual QC pause before tiling.

This script is intentionally lightweight:
- slide-manifest driven
- YAML driven
- layout-aware across multiple studies
- configurable section selection and labeling
- writes QC overlays plus a review-ready ROI manifest

The output ROI manifest defaults `include_for_tiling` to 0 for every row so
users must manually review the QC outputs before running tiling.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import openslide
import pandas as pd
import yaml
from PIL import Image, ImageDraw


REQUIRED_MANIFEST_COLUMNS = {"slide_id", "study_key"}


@dataclass
class Component:
    rank: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    centroid_x: float
    centroid_y: float
    area_px: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract review-ready ROIs from SVS slides.")
    parser.add_argument("--slide-manifest", required=True, help="CSV or XLSX slide manifest.")
    parser.add_argument("--config", required=True, help="YAML extraction config.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--svs-root",
        default=None,
        help="Optional directory used when the manifest contains svs_filename instead of svs_path.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


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
        "qc_overlays": output_dir / "qc_overlays",
        "preview_crops": output_dir / "preview_crops",
        "manifests": output_dir / "manifests",
        "logs": output_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_slide_path(row: pd.Series, svs_root: Optional[Path]) -> Path:
    if pd.notna(row.get("svs_path", None)) and str(row["svs_path"]).strip():
        return Path(str(row["svs_path"]))
    if svs_root is not None and pd.notna(row.get("svs_filename", None)) and str(row["svs_filename"]).strip():
        return svs_root / str(row["svs_filename"])
    raise ValueError("Each row must provide svs_path or svs_filename with --svs-root.")


def build_thumbnail(slide: openslide.OpenSlide, max_dim: int) -> tuple[np.ndarray, float]:
    level0_w, level0_h = slide.dimensions
    scale = min(max_dim / max(level0_w, level0_h), 1.0)
    thumb_w = max(1, int(round(level0_w * scale)))
    thumb_h = max(1, int(round(level0_h * scale)))
    image = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    return np.array(image), scale


def detect_components(image_rgb: np.ndarray, config: Dict[str, Any], scale: float) -> List[Component]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    tissue_mask = (gray < int(config["white_threshold"])).astype(np.uint8) * 255

    kernel_size = int(config["morphology_kernel_size"])
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(tissue_mask, connectivity=8)

    components: List[Component] = []
    min_area_thumb = int(config["min_component_area_thumb"])
    level0_scale = 1.0 / max(scale, 1e-8)
    for label_index in range(1, num_labels):
        x, y, w, h, area = stats[label_index]
        if area < min_area_thumb:
            continue
        centroid_x, centroid_y = centroids[label_index]
        components.append(
            Component(
                rank=-1,
                bbox_x=int(round(x * level0_scale)),
                bbox_y=int(round(y * level0_scale)),
                bbox_w=max(1, int(round(w * level0_scale))),
                bbox_h=max(1, int(round(h * level0_scale))),
                centroid_x=float(centroid_x * level0_scale),
                centroid_y=float(centroid_y * level0_scale),
                area_px=int(round(area * (level0_scale**2))),
            )
        )

    components.sort(key=lambda item: item.area_px, reverse=True)
    for rank, component in enumerate(components, start=1):
        component.rank = rank
    return components


def expand_bbox(component: Component, slide_width: int, slide_height: int, margin_ratio: float) -> Component:
    margin_x = int(round(component.bbox_w * margin_ratio))
    margin_y = int(round(component.bbox_h * margin_ratio))
    x0 = max(0, component.bbox_x - margin_x)
    y0 = max(0, component.bbox_y - margin_y)
    x1 = min(slide_width, component.bbox_x + component.bbox_w + margin_x)
    y1 = min(slide_height, component.bbox_y + component.bbox_h + margin_y)
    return Component(
        rank=component.rank,
        bbox_x=x0,
        bbox_y=y0,
        bbox_w=max(1, x1 - x0),
        bbox_h=max(1, y1 - y0),
        centroid_x=component.centroid_x,
        centroid_y=component.centroid_y,
        area_px=component.area_px,
    )


def select_components(
    components: List[Component],
    layout: Dict[str, Any],
    slide_width: int,
    slide_height: int,
) -> List[Component]:
    mode = str(layout["selection_mode"])
    if not components:
        return []

    if mode == "keep_largest":
        return [components[0]]

    if mode == "keep_n_left_to_right":
        count = int(layout.get("expected_n", len(layout.get("output_labels", [])) or 1))
        chosen = components[:count]
        return sorted(chosen, key=lambda item: item.centroid_x)

    if mode == "keep_representative":
        candidate_pool = int(layout.get("candidate_pool", min(6, len(components))))
        candidates = components[:candidate_pool]
        rule = str(layout.get("representative_rule", "bottom_left"))
        if rule == "bottom_left":
            candidates = sorted(candidates, key=lambda item: (-item.centroid_y, item.centroid_x))
        elif rule == "top_left":
            candidates = sorted(candidates, key=lambda item: (item.centroid_y, item.centroid_x))
        elif rule == "center_most":
            center_x = slide_width / 2.0
            center_y = slide_height / 2.0
            candidates = sorted(
                candidates,
                key=lambda item: math.hypot(item.centroid_x - center_x, item.centroid_y - center_y),
            )
        elif rule == "largest":
            candidates = sorted(candidates, key=lambda item: item.area_px, reverse=True)
        else:
            raise ValueError(f"Unsupported representative_rule: {rule}")
        return [candidates[0]]

    raise ValueError(f"Unsupported selection_mode: {mode}")


def assign_labels(selected: List[Component], layout: Dict[str, Any]) -> List[str]:
    labels = list(layout.get("output_labels", []))
    if labels:
        if len(labels) < len(selected):
            labels.extend([f"section_{index}" for index in range(len(labels) + 1, len(selected) + 1)])
        return labels[: len(selected)]
    if len(selected) == 1:
        return ["whole"]
    return [f"section_{index}" for index in range(1, len(selected) + 1)]


def build_analysis_unit_id(row: pd.Series, label: str, naming_mode: str) -> str:
    base = str(row.get("sample_id") or row.get("slide_id"))
    if naming_mode == "no_suffix":
        return base
    if naming_mode == "suffix_labels":
        return f"{base}_{label}"
    raise ValueError(f"Unsupported naming_mode: {naming_mode}")


def save_preview(slide: openslide.OpenSlide, component: Component, destination: Path, max_dim: int) -> None:
    image = slide.read_region((component.bbox_x, component.bbox_y), 0, (component.bbox_w, component.bbox_h)).convert("RGB")
    image.thumbnail((max_dim, max_dim))
    image.save(destination)


def save_qc_overlay(
    thumbnail_rgb: np.ndarray,
    scale: float,
    selected: List[Component],
    all_components: List[Component],
    labels: List[str],
    destination: Path,
) -> None:
    image = Image.fromarray(thumbnail_rgb.copy())
    draw = ImageDraw.Draw(image)

    for component in all_components:
        x0 = int(round(component.bbox_x * scale))
        y0 = int(round(component.bbox_y * scale))
        x1 = int(round((component.bbox_x + component.bbox_w) * scale))
        y1 = int(round((component.bbox_y + component.bbox_h) * scale))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 180, 0), width=2)

    for label, component in zip(labels, selected):
        x0 = int(round(component.bbox_x * scale))
        y0 = int(round(component.bbox_y * scale))
        x1 = int(round((component.bbox_x + component.bbox_w) * scale))
        y1 = int(round((component.bbox_y + component.bbox_h) * scale))
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 120), width=4)
        draw.text((x0 + 6, max(0, y0 - 16)), label, fill=(0, 255, 120))

    image.save(destination)


def main() -> None:
    args = parse_args()
    slide_manifest_path = Path(args.slide_manifest)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    svs_root = Path(args.svs_root) if args.svs_root else None

    dirs = ensure_dirs(output_dir)
    setup_logging(dirs["logs"] / "preprocess_extract_rois.log")

    manifest_df = load_table(slide_manifest_path)
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"Slide manifest missing required columns: {sorted(missing_columns)}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    global_cfg = config.get("global", {})
    study_layouts = config.get("study_layouts", {})
    default_layout = config.get("default_layout", {})
    if not study_layouts and not default_layout:
        raise ValueError("Config must contain default_layout or study_layouts.")

    roi_rows: List[Dict[str, Any]] = []
    issue_rows: List[Dict[str, Any]] = []

    for _, row in manifest_df.iterrows():
        slide_id = str(row["slide_id"])
        study_key = str(row["study_key"])
        layout = dict(default_layout)
        layout.update(study_layouts.get(study_key, {}))
        if not layout:
            issue_rows.append(
                {
                    "slide_id": slide_id,
                    "study_key": study_key,
                    "issue_code": "missing_layout",
                    "message": "No layout config was provided for this study_key.",
                }
            )
            continue

        slide_path = resolve_slide_path(row, svs_root)
        if not slide_path.exists():
            issue_rows.append(
                {
                    "slide_id": slide_id,
                    "study_key": study_key,
                    "issue_code": "missing_slide",
                    "message": f"Slide file not found: {slide_path}",
                }
            )
            continue

        logging.info("Processing %s (%s)", slide_id, study_key)
        with openslide.OpenSlide(str(slide_path)) as slide:
            slide_width, slide_height = slide.dimensions
            thumbnail_rgb, scale = build_thumbnail(slide, int(global_cfg.get("thumbnail_max_dim", 2048)))
            components = detect_components(
                thumbnail_rgb,
                {
                    "white_threshold": int(global_cfg.get("white_threshold", 220)),
                    "morphology_kernel_size": int(global_cfg.get("morphology_kernel_size", 7)),
                    "min_component_area_thumb": int(global_cfg.get("min_component_area_thumb", 5000)),
                },
                scale,
            )
            expanded_components = [
                expand_bbox(component, slide_width, slide_height, float(global_cfg.get("bbox_margin_ratio", 0.02)))
                for component in components
            ]
            selected = select_components(expanded_components, layout, slide_width, slide_height)
            labels = assign_labels(selected, layout)

            qc_destination = dirs["qc_overlays"] / f"{slide_id}_qc.png"
            save_qc_overlay(thumbnail_rgb, scale, selected, expanded_components, labels, qc_destination)

            if not selected:
                issue_rows.append(
                    {
                        "slide_id": slide_id,
                        "study_key": study_key,
                        "issue_code": "no_components_selected",
                        "message": "No components passed the selection rules.",
                    }
                )
                continue

            for label, component in zip(labels, selected):
                preview_name = f"{slide_id}_{label}_preview.png"
                preview_destination = dirs["preview_crops"] / preview_name
                save_preview(slide, component, preview_destination, int(global_cfg.get("preview_max_dim", 1024)))

                roi_row = row.to_dict()
                roi_row.update(
                    {
                        "svs_path": str(slide_path),
                        "svs_name": slide_path.name,
                        "analysis_unit_id": build_analysis_unit_id(
                            row,
                            label,
                            str(layout.get("naming_mode", "suffix_labels")),
                        ),
                        "section_label": label,
                        "selection_mode": str(layout["selection_mode"]),
                        "representative_rule": str(layout.get("representative_rule", "")),
                        "expected_sections": layout.get("expected_n", layout.get("candidate_pool", "")),
                        "component_rank": int(component.rank),
                        "bbox_x": int(component.bbox_x),
                        "bbox_y": int(component.bbox_y),
                        "bbox_w": int(component.bbox_w),
                        "bbox_h": int(component.bbox_h),
                        "centroid_x": float(component.centroid_x),
                        "centroid_y": float(component.centroid_y),
                        "area_px": int(component.area_px),
                        "auto_qc_status": "pending_manual_review",
                        "manual_qc_status": "",
                        "include_for_tiling": 0,
                        "reviewer_notes": "",
                        "preview_image_path": str(Path("preview_crops") / preview_name),
                        "qc_overlay_path": str(Path("qc_overlays") / qc_destination.name),
                    }
                )
                roi_rows.append(roi_row)

    roi_df = pd.DataFrame(roi_rows)
    if not roi_df.empty:
        priority_columns = [
            "slide_id",
            "study_key",
            "sample_id",
            "svs_path",
            "analysis_unit_id",
            "section_label",
            "selection_mode",
            "representative_rule",
            "component_rank",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "centroid_x",
            "centroid_y",
            "area_px",
            "auto_qc_status",
            "manual_qc_status",
            "include_for_tiling",
            "reviewer_notes",
            "preview_image_path",
            "qc_overlay_path",
        ]
        ordered_columns = priority_columns + [col for col in roi_df.columns if col not in priority_columns]
        roi_df = roi_df.loc[:, ordered_columns]
    roi_df.to_csv(dirs["manifests"] / "roi_manifest.csv", index=False)
    pd.DataFrame(issue_rows).to_csv(dirs["manifests"] / "roi_issues.csv", index=False)

    logging.info("Wrote %s ROI rows", len(roi_df))
    logging.info("QC pause: review %s and then edit manifests/roi_manifest.csv", dirs["qc_overlays"])


if __name__ == "__main__":
    main()
