#!/usr/bin/env python3
"""
Study-aware ROI extraction for Aperio `.svs` histology slides.

README-style notes
------------------
This script reads a slide manifest (`.csv` or `.xlsx`) plus a YAML study config
and produces a unified ROI manifest across studies. It is designed for ROI
extraction and QC only: it does not tile, augment, or build training data.

Core workflow
-------------
1. Resolve each slide from the manifest.
2. Open the `.svs` with OpenSlide.
3. Create a thumbnail and detect tissue components on the thumbnail.
4. Scale selected ROIs back to level-0 coordinates.
5. Apply the study-specific selection rule from YAML.
6. Save a QC overlay, optional preview crop(s), and optional TIFF export(s).
7. Write a single ROI manifest CSV plus an issues CSV.

Downstream tiling note
----------------------
Downstream tiling can use the output ROI manifest directly by reading
`parent_svs_path` and the level-0 ROI box columns (`bbox_x`, `bbox_y`,
`bbox_w`, `bbox_h`) to tile only within selected regions while preserving
traceability to `study_id`, `sample_base_id`, and `analysis_unit_id`.

Example
-------
python extract_svs_rois.py \
  --manifest MasterHistoManifest.xlsx \
  --config study_rules.yaml \
  --svs-dir rawSVS \
  --output-dir fullextract_output \
  --export-previews \
  --min-component-area 10000 \
  --thumbnail-max-size 2048 \
  --margin-ratio 0.002 \
  --workers 6  
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import openslide
import pandas as pd
import yaml
from openslide import OpenSlide
from PIL import Image, ImageDraw
from skimage import measure, morphology

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROI_MANIFEST_COLUMNS = [
    "parent_svs_path",
    "parent_svs_name",
    "study_id",
    "sample_base_id",
    "section_label",
    "analysis_unit_id",
    "selection_mode",
    "expected_sections",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "centroid_x",
    "centroid_y",
    "area_px",
    "component_rank",
    "qc_status",
    "overlap_corrected",
    "preview_image_path",
    "export_image_path",
]

ISSUES_COLUMNS = [
    "parent_svs_path",
    "parent_svs_name",
    "study_id",
    "sample_base_id",
    "selection_mode",
    "severity",
    "issue_code",
    "message",
]

SUPPORTED_SELECTION_MODES = {
    "keep_n_left_to_right",
    "keep_largest",
    "keep_bottom_left_among_major_components",
}


@dataclass(frozen=True)
class StudyRule:
    selection_mode: str
    output_labels: List[str]
    naming_mode: str
    expected_n: Optional[int] = None
    expected_n_major: Optional[int] = None


@dataclass(frozen=True)
class DetectionParams:
    thumbnail_max_size: int
    min_component_area: int
    morphology_kernel_size: int
    margin_ratio: float


@dataclass
class Component:
    component_id: int
    bbox_thumb: Tuple[int, int, int, int]
    centroid_thumb: Tuple[float, float]
    area_thumb_px: int
    bbox_level0: Tuple[int, int, int, int]
    centroid_level0: Tuple[float, float]
    area_level0_px: int
    rank: int = -1


@dataclass
class SlideContext:
    manifest_row: pd.Series
    svs_path: Path
    study_id: str
    sample_base_id: str
    expected_sections: Optional[int]
    selection_mode: str
    study_rule: StudyRule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study-aware ROI extraction from Aperio SVS slides."
    )
    parser.add_argument("--manifest", required=True, help="Slide manifest CSV or XLSX.")
    parser.add_argument("--config", required=True, help="Study YAML config path.")
    parser.add_argument(
        "--svs-dir",
        help="Directory containing SVS files when manifest has svs_filename instead of svs_path.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory root.")
    parser.add_argument(
        "--export-previews",
        action="store_true",
        help="Export preview PNG crops for selected ROI(s).",
    )
    parser.add_argument(
        "--export-tiffs",
        action="store_true",
        help="Export TIFF crops for selected ROI(s) at level 0.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=5000,
        help="Minimum connected-component area in thumbnail pixels. Default: 5000",
    )
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=2048,
        help="Maximum thumbnail width or height used for detection. Default: 2048",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.02,
        help="Fractional bbox expansion applied in level-0 coordinates. Default: 0.02",
    )
    parser.add_argument(
        "--morphology-kernel-size",
        type=int,
        default=7,
        help="Odd kernel size for morphology cleanup on the thumbnail mask. Default: 7",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use across slides. Default: 1",
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
    output_dirs = {
        "root": root,
        "qc": root / "qc_overlays",
        "previews": root / "preview_crops",
        "exports": root / "tiff_exports",
        "manifests": root / "manifests",
        "logs": root / "logs",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def normalize_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def safe_stem(value: str) -> str:
    text = value.strip()
    if not text:
        return "unknown"
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def standardize_manifest_columns(manifest_df: pd.DataFrame) -> pd.DataFrame:
    normalized_columns = [normalize_column_name(str(column)) for column in manifest_df.columns]
    manifest_df = manifest_df.copy()
    manifest_df.columns = normalized_columns
    return manifest_df


def validate_manifest_columns(manifest_df: pd.DataFrame) -> None:
    required_columns = {"study_id", "sample_base_id", "expected_sections"}
    missing = sorted(required_columns.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    has_svs_path = "svs_path" in manifest_df.columns
    has_svs_filename = "svs_filename" in manifest_df.columns
    if not (has_svs_path or has_svs_filename):
        raise ValueError("Manifest must include either 'svs_path' or 'svs_filename'")


def load_manifest_csv(manifest_path: Path) -> pd.DataFrame:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return pd.read_csv(handle, sep=dialect.delimiter)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        manifest_df = load_manifest_csv(manifest_path)
    elif suffix in {".xlsx", ".xls"}:
        manifest_df = pd.read_excel(manifest_path)
    else:
        raise ValueError("Manifest must be .csv, .xlsx, or .xls")

    manifest_df = standardize_manifest_columns(manifest_df)
    validate_manifest_columns(manifest_df)
    logging.info("Loaded manifest with %d rows and columns: %s", len(manifest_df), list(manifest_df.columns))
    return manifest_df


def load_study_config(config_path: Path) -> Dict[str, StudyRule]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, Mapping):
        raise ValueError("Study config must be a mapping of study_id -> rule")

    study_rules: Dict[str, StudyRule] = {}
    for study_id, payload in raw_config.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"Study config for '{study_id}' must be a mapping")

        selection_mode = str(payload.get("selection_mode", "")).strip()
        output_labels = list(payload.get("output_labels", []))
        naming_mode = str(payload.get("naming_mode", "")).strip()

        if selection_mode not in SUPPORTED_SELECTION_MODES:
            raise ValueError(
                f"Unsupported selection_mode '{selection_mode}' for study '{study_id}'"
            )
        if not output_labels:
            raise ValueError(f"Study '{study_id}' must define non-empty output_labels")
        if naming_mode not in {"suffix_labels", "no_suffix"}:
            raise ValueError(f"Unsupported naming_mode '{naming_mode}' for study '{study_id}'")

        expected_n = payload.get("expected_n")
        expected_n_major = payload.get("expected_n_major")
        study_rules[str(study_id)] = StudyRule(
            selection_mode=selection_mode,
            output_labels=[str(label) for label in output_labels],
            naming_mode=naming_mode,
            expected_n=int(expected_n) if expected_n is not None else None,
            expected_n_major=int(expected_n_major) if expected_n_major is not None else None,
        )

    return study_rules


def resolve_svs_path(
    row: pd.Series,
    svs_dir: Optional[Path],
) -> Path:
    svs_path_value = normalize_token(row.get("svs_path"))
    svs_filename_value = normalize_token(row.get("svs_filename"))

    if svs_path_value:
        candidate = Path(svs_path_value).expanduser()
    elif svs_filename_value:
        if svs_dir is None:
            logging.error(
                "Missing --svs-dir for manifest row with svs_filename='%s'",
                svs_filename_value,
            )
            raise FileNotFoundError(
                "Manifest row contains svs_filename but no --svs-dir was provided"
            )
        candidate = svs_dir / svs_filename_value
    else:
        logging.error("Manifest row does not contain svs_path or svs_filename")
        raise FileNotFoundError("Manifest row does not contain svs_path or svs_filename")

    candidate = candidate.resolve()
    if not candidate.exists():
        logging.error("Missing SVS file: %s", candidate)
        raise FileNotFoundError(f"SVS file not found: {candidate}")
    return candidate


def load_slide_thumbnail(slide: OpenSlide, max_size: int) -> np.ndarray:
    width, height = slide.dimensions
    scale = min(max_size / width, max_size / height)
    thumb_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
    return np.asarray(thumbnail)


def detect_tissue_mask(
    thumbnail_rgb: np.ndarray,
    min_component_area: int,
    morphology_kernel_size: int,
) -> np.ndarray:
    gray = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)

    _, otsu_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    saturation_mask = (hsv[:, :, 1] > 18).astype(np.uint8) * 255
    value_mask = (hsv[:, :, 2] < 245).astype(np.uint8) * 255

    mask = np.logical_or(otsu_mask > 0, np.logical_and(saturation_mask > 0, value_mask > 0))
    mask = morphology.remove_small_objects(mask, max_size=max(16, min_component_area // 5))
    mask = morphology.remove_small_holes(mask, max_size=max(64, min_component_area // 3))

    kernel_size = max(3, morphology_kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    return mask_uint8


def extract_connected_components(
    mask: np.ndarray,
    slide_dimensions: Tuple[int, int],
    margin_ratio: float,
    min_component_area: int,
) -> List[Component]:
    thumb_h, thumb_w = mask.shape[:2]
    slide_w, slide_h = slide_dimensions
    scale_x = slide_w / float(thumb_w)
    scale_y = slide_h / float(thumb_h)

    label_image = measure.label(mask > 0, connectivity=2)
    region_props = measure.regionprops(label_image)

    components: List[Component] = []
    for idx, region in enumerate(region_props, start=1):
        if region.area < min_component_area:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        bbox_thumb = (
            int(min_col),
            int(min_row),
            int(max_col - min_col),
            int(max_row - min_row),
        )
        centroid_thumb = (float(region.centroid[1]), float(region.centroid[0]))

        bbox_level0 = scale_bbox_to_level0(
            bbox_thumb=bbox_thumb,
            scale_x=scale_x,
            scale_y=scale_y,
            slide_dimensions=slide_dimensions,
            margin_ratio=margin_ratio,
        )
        centroid_level0 = (
            centroid_thumb[0] * scale_x,
            centroid_thumb[1] * scale_y,
        )
        area_level0 = max(1, int(round(region.area * scale_x * scale_y)))

        components.append(
            Component(
                component_id=idx,
                bbox_thumb=bbox_thumb,
                centroid_thumb=centroid_thumb,
                area_thumb_px=int(region.area),
                bbox_level0=bbox_level0,
                centroid_level0=centroid_level0,
                area_level0_px=area_level0,
            )
        )

    components.sort(key=lambda component: component.area_thumb_px, reverse=True)
    for rank, component in enumerate(components, start=1):
        component.rank = rank
    return components


def scale_bbox_to_level0(
    bbox_thumb: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    slide_dimensions: Tuple[int, int],
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_thumb
    x0 = int(math.floor(x * scale_x))
    y0 = int(math.floor(y * scale_y))
    x1 = int(math.ceil((x + w) * scale_x))
    y1 = int(math.ceil((y + h) * scale_y))

    expand_x = int(round((x1 - x0) * margin_ratio))
    expand_y = int(round((y1 - y0) * margin_ratio))

    slide_w, slide_h = slide_dimensions
    x0 = max(0, x0 - expand_x)
    y0 = max(0, y0 - expand_y)
    x1 = min(slide_w, x1 + expand_x)
    y1 = min(slide_h, y1 + expand_y)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def level0_bbox_to_thumb(
    bbox_level0: Tuple[int, int, int, int],
    slide_dimensions: Tuple[int, int],
    thumbnail_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_level0
    slide_w, slide_h = slide_dimensions
    thumb_h, thumb_w = thumbnail_shape
    scale_x = thumb_w / float(slide_w)
    scale_y = thumb_h / float(slide_h)
    thumb_x = int(round(x * scale_x))
    thumb_y = int(round(y * scale_y))
    thumb_w_box = max(1, int(round(w * scale_x)))
    thumb_h_box = max(1, int(round(h * scale_y)))
    return thumb_x, thumb_y, thumb_w_box, thumb_h_box


def select_components_keep_n_left_to_right(
    components: Sequence[Component],
    expected_n: int,
) -> Tuple[List[Component], List[str]]:
    warnings: List[str] = []
    if not components:
        return [], ["no_valid_components"]

    selected = list(components[:expected_n])
    if len(selected) < expected_n:
        warnings.append(f"expected_{expected_n}_components_found_{len(selected)}")
    selected.sort(key=lambda component: component.centroid_level0[0])
    return selected, warnings


def select_component_keep_largest(
    components: Sequence[Component],
) -> Tuple[List[Component], List[str]]:
    if not components:
        return [], ["no_valid_components"]
    return [components[0]], []


def select_component_bottom_left_among_major(
    components: Sequence[Component],
    expected_n_major: int,
) -> Tuple[List[Component], List[str]]:
    if not components:
        return [], ["no_valid_components"]

    major_components = list(components[:expected_n_major])
    warnings: List[str] = []
    if len(major_components) < expected_n_major:
        warnings.append(
            f"expected_{expected_n_major}_major_components_found_{len(major_components)}"
        )

    selected = sorted(
        major_components,
        key=lambda component: (-component.centroid_level0[1], component.centroid_level0[0]),
    )[0]
    return [selected], warnings


def enforce_non_overlapping_boxes(
    selected_components: Sequence[Component],
    slide_context: SlideContext,
) -> Tuple[bool, List[Dict[str, Any]]]:
    if slide_context.selection_mode != "keep_n_left_to_right" or len(selected_components) < 2:
        return False, []

    overlap_corrected = False
    issues: List[Dict[str, Any]] = []

    for index in range(len(selected_components) - 1):
        left_component = selected_components[index]
        right_component = selected_components[index + 1]

        left_x, left_y, left_w, left_h = left_component.bbox_level0
        right_x, right_y, right_w, right_h = right_component.bbox_level0
        left_right = left_x + left_w
        right_right = right_x + right_w

        if left_right <= right_x:
            continue

        split_x = int(round(
            (left_component.centroid_level0[0] + right_component.centroid_level0[0]) / 2.0
        ))
        split_x = max(left_x + 1, min(split_x, right_right - 1))

        new_left_w = split_x - left_x
        new_right_x = split_x
        new_right_w = right_right - new_right_x

        if new_left_w <= 0 or new_right_w <= 0:
            issues.append(
                build_issue_record(
                    slide_context=slide_context,
                    severity="warning",
                    issue_code="overlap_correction_failed",
                    message=(
                        f"Could not correct overlap between boxes {index} and {index + 1} "
                        f"for {slide_context.svs_path.name}"
                    ),
                )
            )
            logging.warning(
                "[%s] Overlap correction failed between adjacent selected boxes",
                slide_context.sample_base_id or slide_context.svs_path.name,
            )
            continue

        left_component.bbox_level0 = (left_x, left_y, new_left_w, left_h)
        right_component.bbox_level0 = (new_right_x, right_y, new_right_w, right_h)
        overlap_corrected = True
        logging.info(
            "[%s] Corrected overlap between adjacent ROI boxes at split_x=%d",
            slide_context.sample_base_id or slide_context.svs_path.name,
            split_x,
        )

    for index in range(len(selected_components) - 1):
        left_component = selected_components[index]
        right_component = selected_components[index + 1]
        left_right = left_component.bbox_level0[0] + left_component.bbox_level0[2]
        right_left = right_component.bbox_level0[0]
        if left_right > right_left:
            issues.append(
                build_issue_record(
                    slide_context=slide_context,
                    severity="warning",
                    issue_code="overlap_persisted_after_correction",
                    message=(
                        f"Overlap persisted after correction between selected boxes {index} "
                        f"and {index + 1}"
                    ),
                )
            )

    return overlap_corrected, issues


def build_analysis_unit_id(
    sample_base_id: str,
    section_label: str,
    naming_mode: str,
) -> str:
    if naming_mode == "suffix_labels":
        return f"{sample_base_id}-{section_label}"
    if naming_mode == "no_suffix":
        return sample_base_id
    raise ValueError(f"Unsupported naming mode: {naming_mode}")


def create_placeholder_qc_image(
    output_path: Path,
    title: str,
    message: str,
) -> None:
    image = Image.new("RGB", (1200, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), title, fill=(0, 0, 0))
    draw.text((40, 120), message, fill=(180, 0, 0))
    image.save(output_path)


def save_qc_overlay(
    thumbnail_rgb: np.ndarray,
    components: Sequence[Component],
    selected_components: Sequence[Component],
    selected_labels: Sequence[str],
    slide_dimensions: Tuple[int, int],
    output_path: Path,
    slide_title: str,
    qc_status: str,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 12))
    axis.imshow(thumbnail_rgb)
    axis.set_title(f"{slide_title}\nQC: {qc_status}")
    axis.axis("off")

    for component in components:
        x, y, w, h = component.bbox_thumb
        rect = Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            edgecolor="#4c78a8",
            facecolor="none",
        )
        axis.add_patch(rect)
        axis.text(
            x,
            max(0, y - 4),
            f"rank={component.rank}",
            color="#4c78a8",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    for label, component in zip(selected_labels, selected_components):
        x, y, w, h = level0_bbox_to_thumb(
            bbox_level0=component.bbox_level0,
            slide_dimensions=slide_dimensions,
            thumbnail_shape=thumbnail_rgb.shape[:2],
        )
        rect = Rectangle(
            (x, y),
            w,
            h,
            linewidth=2.5,
            edgecolor="#e45756",
            facecolor="none",
        )
        axis.add_patch(rect)
        axis.text(
            x,
            max(14, y - 12),
            f"{label} | rank={component.rank}",
            color="#e45756",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def export_preview_crop(
    slide: OpenSlide,
    component: Component,
    output_path: Path,
    max_preview_size: int = 1024,
) -> Path:
    x, y, w, h = component.bbox_level0
    region = slide.read_region((x, y), 0, (w, h)).convert("RGB")
    region.thumbnail((max_preview_size, max_preview_size))
    region.save(output_path)
    return output_path


def export_tiff_crop(
    slide: OpenSlide,
    component: Component,
    output_path: Path,
) -> Path:
    x, y, w, h = component.bbox_level0
    region = slide.read_region((x, y), 0, (w, h)).convert("RGB")
    region.save(output_path, format="TIFF")
    return output_path


def write_roi_manifest(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    manifest_df = pd.DataFrame(records, columns=ROI_MANIFEST_COLUMNS)
    manifest_df.to_csv(output_path, index=False)


def write_issues_manifest(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    issues_df = pd.DataFrame(records, columns=ISSUES_COLUMNS)
    issues_df.to_csv(output_path, index=False)


def coerce_expected_sections(value: Any) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    return int(value)


def build_slide_context(
    row: pd.Series,
    svs_path: Path,
    study_rules: Mapping[str, StudyRule],
) -> SlideContext:
    study_id = normalize_token(row["study_id"])
    if study_id not in study_rules:
        raise KeyError(f"study_id '{study_id}' not present in YAML config")

    study_rule = study_rules[study_id]
    override_mode = normalize_token(row.get("selection_mode_override"))
    selection_mode = override_mode or study_rule.selection_mode
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(f"Unsupported effective selection_mode '{selection_mode}'")

    return SlideContext(
        manifest_row=row,
        svs_path=svs_path,
        study_id=study_id,
        sample_base_id=normalize_token(row["sample_base_id"]),
        expected_sections=coerce_expected_sections(row["expected_sections"]),
        selection_mode=selection_mode,
        study_rule=study_rule,
    )


def select_components_for_slide(
    slide_context: SlideContext,
    components: Sequence[Component],
) -> Tuple[List[Component], str, List[Dict[str, Any]]]:
    selection_mode = slide_context.selection_mode
    issues: List[Dict[str, Any]] = []

    if selection_mode == "keep_n_left_to_right":
        expected_n = slide_context.study_rule.expected_n or slide_context.expected_sections or 0
        selected_components, warnings = select_components_keep_n_left_to_right(
            components=components,
            expected_n=expected_n,
        )
    elif selection_mode == "keep_largest":
        selected_components, warnings = select_component_keep_largest(components)
    elif selection_mode == "keep_bottom_left_among_major_components":
        expected_n_major = slide_context.study_rule.expected_n_major or 0
        selected_components, warnings = select_component_bottom_left_among_major(
            components=components,
            expected_n_major=expected_n_major,
        )
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    if not selected_components:
        qc_status = "fail_no_selected_roi"
    elif warnings:
        qc_status = "warn_" + "__".join(warnings)
    else:
        qc_status = "ok"

    for warning in warnings:
        logging.warning(
            "Insufficient detected components for %s: %s",
            slide_context.svs_path.name,
            warning,
        )
        issues.append(
            build_issue_record(
                slide_context=slide_context,
                severity="warning",
                issue_code=warning,
                message=f"Selection produced warning: {warning}",
            )
        )
    if not selected_components:
        logging.error("No ROI selected for %s", slide_context.svs_path.name)
        issues.append(
            build_issue_record(
                slide_context=slide_context,
                severity="error",
                issue_code="no_selected_roi",
                message="No ROI could be selected for this slide.",
            )
        )

    return selected_components, qc_status, issues


def check_expected_sections_mismatch(
    slide_context: SlideContext,
    selected_components: Sequence[Component],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if slide_context.expected_sections is None:
        return issues

    selected_count = len(selected_components)
    if slide_context.expected_sections != selected_count:
        message = (
            f"expected_sections={slide_context.expected_sections} but selected_outputs={selected_count}"
        )
        logging.warning("Selection count mismatch for %s: %s", slide_context.svs_path.name, message)
        issues.append(
            build_issue_record(
                slide_context=slide_context,
                severity="warning",
                issue_code="expected_sections_mismatch",
                message=message,
            )
        )
    return issues


def build_issue_record(
    slide_context: SlideContext,
    severity: str,
    issue_code: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "parent_svs_path": str(slide_context.svs_path),
        "parent_svs_name": slide_context.svs_path.name,
        "study_id": slide_context.study_id,
        "sample_base_id": slide_context.sample_base_id,
        "selection_mode": slide_context.selection_mode,
        "severity": severity,
        "issue_code": issue_code,
        "message": message,
    }


def build_roi_records(
    slide_context: SlideContext,
    selected_components: Sequence[Component],
    qc_status: str,
    overlap_corrected: bool,
    preview_paths: Mapping[str, Optional[Path]],
    export_paths: Mapping[str, Optional[Path]],
) -> List[Dict[str, Any]]:
    labels = list(slide_context.study_rule.output_labels)
    if slide_context.selection_mode == "keep_n_left_to_right":
        labels = labels[: max(len(selected_components), len(labels))]
    elif selected_components:
        labels = [labels[0]]
    else:
        labels = [labels[0]]

    records: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        component = selected_components[index] if index < len(selected_components) else None
        bbox_x, bbox_y, bbox_w, bbox_h = component.bbox_level0 if component else (None, None, None, None)
        records.append(
            {
                "parent_svs_path": str(slide_context.svs_path),
                "parent_svs_name": slide_context.svs_path.name,
                "study_id": slide_context.study_id,
                "sample_base_id": slide_context.sample_base_id,
                "section_label": label,
                "analysis_unit_id": build_analysis_unit_id(
                    sample_base_id=slide_context.sample_base_id,
                    section_label=label,
                    naming_mode=slide_context.study_rule.naming_mode,
                ),
                "selection_mode": slide_context.selection_mode,
                "expected_sections": slide_context.expected_sections,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "centroid_x": round(component.centroid_level0[0], 2) if component else None,
                "centroid_y": round(component.centroid_level0[1], 2) if component else None,
                "area_px": component.area_level0_px if component else None,
                "component_rank": component.rank if component else None,
                "qc_status": qc_status,
                "overlap_corrected": overlap_corrected,
                "preview_image_path": str(preview_paths.get(label) or ""),
                "export_image_path": str(export_paths.get(label) or ""),
            }
        )
    return records


def build_placeholder_roi_records(
    slide_context: SlideContext,
    qc_status: str,
) -> List[Dict[str, Any]]:
    return build_roi_records(
        slide_context=slide_context,
        selected_components=[],
        qc_status=qc_status,
        overlap_corrected=False,
        preview_paths={},
        export_paths={},
    )


def process_slide(
    slide_context: SlideContext,
    detection_params: DetectionParams,
    output_dirs: Mapping[str, Path],
    export_previews: bool,
    export_tiffs: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_token = safe_stem(slide_context.sample_base_id or slide_context.svs_path.stem)
    qc_path = output_dirs["qc"] / f"{output_token}_qc.png"
    issues: List[Dict[str, Any]] = []

    try:
        slide = openslide.OpenSlide(str(slide_context.svs_path))
    except Exception as exc:
        message = f"Failed to open slide: {exc}"
        logging.exception(message)
        create_placeholder_qc_image(qc_path, slide_context.svs_path.name, message)
        issues.append(
            build_issue_record(
                slide_context=slide_context,
                severity="error",
                issue_code="slide_open_error",
                message=message,
            )
        )
        return build_placeholder_roi_records(slide_context, "slide_open_error"), issues

    try:
        thumbnail_rgb = load_slide_thumbnail(slide, detection_params.thumbnail_max_size)
        tissue_mask = detect_tissue_mask(
            thumbnail_rgb=thumbnail_rgb,
            min_component_area=detection_params.min_component_area,
            morphology_kernel_size=detection_params.morphology_kernel_size,
        )
        components = extract_connected_components(
            mask=tissue_mask,
            slide_dimensions=slide.dimensions,
            margin_ratio=detection_params.margin_ratio,
            min_component_area=detection_params.min_component_area,
        )

        selected_components, qc_status, selection_issues = select_components_for_slide(
            slide_context=slide_context,
            components=components,
        )
        issues.extend(selection_issues)
        overlap_corrected, overlap_issues = enforce_non_overlapping_boxes(
            selected_components=selected_components,
            slide_context=slide_context,
        )
        issues.extend(overlap_issues)
        issues.extend(check_expected_sections_mismatch(slide_context, selected_components))
        label_sequence = (
            slide_context.study_rule.output_labels[: len(selected_components)]
            if slide_context.selection_mode == "keep_n_left_to_right"
            else slide_context.study_rule.output_labels[:1]
        )
        if overlap_corrected and qc_status == "ok":
            qc_status = "ok_overlap_corrected"
        elif overlap_corrected:
            qc_status = f"{qc_status}__overlap_corrected"

        save_qc_overlay(
            thumbnail_rgb=thumbnail_rgb,
            components=components,
            selected_components=selected_components,
            selected_labels=label_sequence,
            slide_dimensions=slide.dimensions,
            output_path=qc_path,
            slide_title=f"{slide_context.svs_path.name} | {slide_context.study_id}",
            qc_status=qc_status,
        )

        preview_paths: Dict[str, Optional[Path]] = {}
        export_paths: Dict[str, Optional[Path]] = {}

        for label, component in zip(label_sequence, selected_components):
            preview_output: Optional[Path] = None
            export_output: Optional[Path] = None

            if export_previews:
                preview_output = (
                    output_dirs["previews"] / f"{output_token}_{safe_stem(label)}_preview.png"
                )
                export_preview_crop(slide, component, preview_output)

            if export_tiffs:
                export_output = (
                    output_dirs["exports"] / f"{output_token}_{safe_stem(label)}.tiff"
                )
                export_tiff_crop(slide, component, export_output)

            preview_paths[label] = preview_output
            export_paths[label] = export_output

        roi_records = build_roi_records(
            slide_context=slide_context,
            selected_components=selected_components,
            qc_status=qc_status,
            overlap_corrected=overlap_corrected,
            preview_paths=preview_paths,
            export_paths=export_paths,
        )

        if not roi_records:
            logging.warning(
                "No ROI records created for %s (%s)",
                slide_context.svs_path.name,
                qc_status,
            )
        else:
            logging.info(
                "Processed %s | detected=%d | selected=%d | qc=%s",
                slide_context.svs_path.name,
                len(components),
                len(selected_components),
                qc_status,
            )
        return roi_records, issues
    except Exception as exc:
        message = f"Processing error: {exc}"
        logging.exception("Failed while processing %s", slide_context.svs_path)
        create_placeholder_qc_image(qc_path, slide_context.svs_path.name, message)
        issues.append(
            build_issue_record(
                slide_context=slide_context,
                severity="error",
                issue_code="processing_error",
                message=message,
            )
        )
        return build_placeholder_roi_records(slide_context, "processing_error"), issues
    finally:
        slide.close()


def iterate_manifest_rows(manifest_df: pd.DataFrame) -> Iterable[pd.Series]:
    for _, row in manifest_df.iterrows():
        yield row


def build_placeholder_roi_records_from_row(
    row: pd.Series,
    study_rules: Mapping[str, StudyRule],
    qc_status: str,
    svs_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    study_id = normalize_token(row.get("study_id"))
    sample_base_id = normalize_token(row.get("sample_base_id"))
    rule = study_rules.get(study_id)
    if not rule:
        return []

    effective_svs_path = svs_path or Path(
        normalize_token(row.get("svs_path") or row.get("svs_filename") or "")
    )
    slide_context = SlideContext(
        manifest_row=row,
        svs_path=effective_svs_path,
        study_id=study_id,
        sample_base_id=sample_base_id,
        expected_sections=coerce_expected_sections(row.get("expected_sections")),
        selection_mode=normalize_token(row.get("selection_mode_override")) or rule.selection_mode,
        study_rule=rule,
    )
    return build_placeholder_roi_records(slide_context, qc_status)


def process_manifest_row(
    row_dict: Dict[str, Any],
    study_rules_payload: Dict[str, Dict[str, Any]],
    detection_params_payload: Dict[str, Any],
    svs_dir_str: Optional[str],
    output_dirs_payload: Dict[str, str],
    export_previews: bool,
    export_tiffs: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    study_rules = {
        study_id: StudyRule(**payload) for study_id, payload in study_rules_payload.items()
    }
    detection_params = DetectionParams(**detection_params_payload)
    output_dirs = {name: Path(path_str) for name, path_str in output_dirs_payload.items()}
    row = pd.Series(row_dict)
    svs_dir = Path(svs_dir_str) if svs_dir_str else None

    svs_path: Optional[Path] = None
    try:
        svs_path = resolve_svs_path(row, svs_dir)
        slide_context = build_slide_context(row, svs_path, study_rules)
        roi_records, issue_records = process_slide(
            slide_context=slide_context,
            detection_params=detection_params,
            output_dirs=output_dirs,
            export_previews=export_previews,
            export_tiffs=export_tiffs,
        )
        summary = (
            f"{slide_context.sample_base_id or slide_context.svs_path.name}: "
            f"roi_rows={len(roi_records)} issues={len(issue_records)}"
        )
        return roi_records, issue_records, summary
    except Exception as exc:
        roi_records = build_placeholder_roi_records_from_row(
            row=row,
            study_rules=study_rules,
            qc_status="manifest_or_path_error",
            svs_path=svs_path,
        )
        issue_records = [
            {
                "parent_svs_path": str(svs_path) if svs_path else normalize_token(row.get("svs_path")),
                "parent_svs_name": Path(
                    normalize_token(row.get("svs_path") or row.get("svs_filename"))
                ).name,
                "study_id": normalize_token(row.get("study_id")),
                "sample_base_id": normalize_token(row.get("sample_base_id")),
                "selection_mode": normalize_token(row.get("selection_mode_override")),
                "severity": "error",
                "issue_code": "manifest_or_path_error",
                "message": str(exc),
            }
        ]
        summary = (
            f"{normalize_token(row.get('sample_base_id')) or normalize_token(row.get('svs_filename'))}: "
            f"failed before slide processing ({exc})"
        )
        return roi_records, issue_records, summary


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_dirs = ensure_output_dirs(output_root)
    setup_logging(output_dirs["logs"] / "extract_svs_rois.log")

    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    svs_dir = Path(args.svs_dir).expanduser().resolve() if args.svs_dir else None

    detection_params = DetectionParams(
        thumbnail_max_size=args.thumbnail_max_size,
        min_component_area=args.min_component_area,
        morphology_kernel_size=args.morphology_kernel_size,
        margin_ratio=args.margin_ratio,
    )

    logging.info("Manifest: %s", manifest_path)
    logging.info("Config: %s", config_path)
    logging.info("SVS dir: %s", svs_dir if svs_dir else "<manifest paths only>")
    logging.info("Output dir: %s", output_root)
    logging.info("Workers: %d", args.workers)

    manifest_df = load_manifest(manifest_path)
    study_rules = load_study_config(config_path)
    row_dicts = manifest_df.to_dict(orient="records")

    roi_records: List[Dict[str, Any]] = []
    issue_records: List[Dict[str, Any]] = []
    study_rules_payload = {
        study_id: {
            "selection_mode": rule.selection_mode,
            "output_labels": rule.output_labels,
            "naming_mode": rule.naming_mode,
            "expected_n": rule.expected_n,
            "expected_n_major": rule.expected_n_major,
        }
        for study_id, rule in study_rules.items()
    }
    detection_params_payload = {
        "thumbnail_max_size": detection_params.thumbnail_max_size,
        "min_component_area": detection_params.min_component_area,
        "morphology_kernel_size": detection_params.morphology_kernel_size,
        "margin_ratio": detection_params.margin_ratio,
    }
    output_dirs_payload = {name: str(path) for name, path in output_dirs.items()}

    if args.workers == 1:
        for row_dict in row_dicts:
            slide_roi_records, slide_issue_records, summary = process_manifest_row(
                row_dict=row_dict,
                study_rules_payload=study_rules_payload,
                detection_params_payload=detection_params_payload,
                svs_dir_str=str(svs_dir) if svs_dir else None,
                output_dirs_payload=output_dirs_payload,
                export_previews=args.export_previews,
                export_tiffs=args.export_tiffs,
            )
            logging.info("%s", summary)
            roi_records.extend(slide_roi_records)
            issue_records.extend(slide_issue_records)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_manifest_row,
                    row_dict,
                    study_rules_payload,
                    detection_params_payload,
                    str(svs_dir) if svs_dir else None,
                    output_dirs_payload,
                    args.export_previews,
                    args.export_tiffs,
                )
                for row_dict in row_dicts
            ]
            for future in as_completed(futures):
                slide_roi_records, slide_issue_records, summary = future.result()
                logging.info("%s", summary)
                roi_records.extend(slide_roi_records)
                issue_records.extend(slide_issue_records)

    roi_manifest_path = output_dirs["manifests"] / "roi_manifest.csv"
    issues_manifest_path = output_dirs["manifests"] / "roi_issues.csv"
    root_roi_manifest_path = output_root / "roi_manifest.csv"
    root_issues_manifest_path = output_root / "roi_issues.csv"
    write_roi_manifest(roi_records, roi_manifest_path)
    write_issues_manifest(issue_records, issues_manifest_path)
    write_roi_manifest(roi_records, root_roi_manifest_path)
    write_issues_manifest(issue_records, root_issues_manifest_path)

    logging.info("Wrote ROI manifest: %s", root_roi_manifest_path)
    logging.info("Wrote issues manifest: %s", root_issues_manifest_path)
    logging.info("Completed ROI extraction. Selected ROI rows: %d", len(roi_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
