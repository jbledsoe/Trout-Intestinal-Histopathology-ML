#!/usr/bin/env python3
"""
Detect three tissue sections from Aperio SVS slides using thumbnail-resolution images.

Outputs per slide:
1. QC JPEG with labeled bounding boxes
2. Three low-resolution JPEG crops, one per section when available
3. Master CSV with detection metadata

This script is intentionally structured so thumbnail detections can later be
scaled back to level 0 for full-resolution export.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import openslide
    from openslide import OpenSlide
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "openslide-python is required. Install it along with the OpenSlide "
        "system library before running this script."
    ) from exc


SECTION_LABELS = ["L", "C", "R"]
CSV_COLUMNS = [
    "svs_path",
    "svs_basename",
    "SvsFile",
    "SlideNumber",
    "SampleID",
    "section_id",
    "section_label",
    "x_thumb",
    "y_thumb",
    "w_thumb",
    "h_thumb",
    "thumb_width",
    "thumb_height",
    "output_crop_path",
    "qc_path",
    "status",
]


@dataclass
class DetectionBox:
    x: int
    y: int
    w: int
    h: int
    area: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect three tissue sections per SVS slide from thumbnail images."
    )
    parser.add_argument("--svs_dir", required=True, help="Directory containing .svs files.")
    parser.add_argument("--metadata_csv", required=True, help="Metadata CSV path.")
    parser.add_argument("--output_dir", required=True, help="Output directory root.")
    parser.add_argument(
        "--thumbnail_max_dim",
        type=int,
        default=2500,
        help="Maximum width or height for the processing thumbnail. Default: 2500",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding in thumbnail pixels applied to each bounding box. Default: 20",
    )
    parser.add_argument(
        "--min_area_frac",
        type=float,
        default=0.002,
        help=(
            "Minimum connected-component area as a fraction of thumbnail area. "
            "Default: 0.002"
        ),
    )
    parser.add_argument(
        "--white_thresh",
        type=int,
        default=220,
        help="Upper grayscale threshold used to classify near-white background. Default: 220",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality for QC and crop exports. Default: 95",
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


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    qc_dir = output_dir / "qc"
    sections_dir = output_dir / "sections"
    logs_dir = output_dir / "logs"
    for path in (output_dir, qc_dir, sections_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"qc": qc_dir, "sections": sections_dir, "logs": logs_dir}


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "NA"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe.strip("_") or "NA"


def list_svs_files(svs_dir: Path) -> List[Path]:
    return sorted([path for path in svs_dir.iterdir() if path.is_file() and path.suffix.lower() == ".svs"])


def build_svs_index(svs_files: Sequence[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in svs_files:
        index[path.name.lower()] = path
        index[path.stem.lower()] = path
    return index


def match_svs_file(
    row: pd.Series,
    svs_files: Sequence[Path],
    svs_index: Dict[str, Path],
) -> Optional[Path]:
    raw_svs_file = str(row.get("SvsFile", "")).strip()
    slide_number = str(row.get("SlideNumber", "")).strip()

    candidates = []
    if raw_svs_file:
        svs_name = Path(raw_svs_file).name.lower()
        svs_stem = Path(raw_svs_file).stem.lower()
        candidates.extend([svs_name, svs_stem])

    for candidate in candidates:
        if candidate in svs_index:
            return svs_index[candidate]

    if raw_svs_file:
        raw_lower = raw_svs_file.lower()
        exact_substring_matches = [
            path for path in svs_files if raw_lower in path.name.lower() or raw_lower in path.stem.lower()
        ]
        if len(exact_substring_matches) == 1:
            return exact_substring_matches[0]

    if slide_number:
        slide_lower = slide_number.lower()
        slide_matches = [
            path for path in svs_files if slide_lower in path.name.lower() or slide_lower in path.stem.lower()
        ]
        if len(slide_matches) == 1:
            return slide_matches[0]
        if len(slide_matches) > 1:
            logging.warning(
                "Multiple SVS candidates found for SlideNumber '%s': %s",
                slide_number,
                ", ".join(path.name for path in slide_matches),
            )
            return None

    return None


def load_thumbnail(slide: OpenSlide, thumbnail_max_dim: int) -> np.ndarray:
    width, height = slide.dimensions
    max_dim = max(width, height)
    scale = thumbnail_max_dim / float(max_dim)
    thumb_w = max(1, int(round(width * scale)))
    thumb_h = max(1, int(round(height * scale)))
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_rgb = np.array(thumbnail)
    return thumb_rgb


def segment_tissue_mask(thumb_rgb: np.ndarray, white_thresh: int) -> np.ndarray:
    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)

    # Tissue tends to be less bright and slightly more saturated than the slide background.
    non_white_gray = gray < white_thresh
    non_white_sat = hsv[:, :, 1] > 10
    mask = np.logical_or(non_white_gray, non_white_sat).astype(np.uint8) * 255

    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_detection_boxes(
    mask: np.ndarray,
    min_area_frac: float,
    padding: int,
) -> List[DetectionBox]:
    thumb_h, thumb_w = mask.shape[:2]
    min_area = max(1, int(round(thumb_h * thumb_w * min_area_frac)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections: List[DetectionBox] = []

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area:
            continue

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(thumb_w, x + w + padding)
        y1 = min(thumb_h, y + h + padding)
        detections.append(DetectionBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0, area=int(area)))

    detections.sort(key=lambda box: box.area, reverse=True)
    detections = detections[:3]
    detections.sort(key=lambda box: box.x)
    return detections


def save_jpeg(image_bgr: np.ndarray, output_path: Path, jpeg_quality: int) -> None:
    success = cv2.imwrite(
        str(output_path),
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not success:
        raise IOError(f"Failed to save image: {output_path}")


def build_qc_image(
    thumb_rgb: np.ndarray,
    boxes: Sequence[DetectionBox],
    slide_number: str,
    sample_id: str,
) -> np.ndarray:
    qc_bgr = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR)
    red = (0, 0, 255)
    text_bg = (255, 255, 255)

    overlay_text = f"SlideNumber: {slide_number} | SampleID: {sample_id}"
    cv2.rectangle(qc_bgr, (10, 10), (min(qc_bgr.shape[1] - 10, 700), 50), text_bg, -1)
    cv2.putText(qc_bgr, overlay_text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    for idx, box in enumerate(boxes):
        label = SECTION_LABELS[idx]
        cv2.rectangle(qc_bgr, (box.x, box.y), (box.x + box.w, box.y + box.h), red, 3)
        label_y = max(25, box.y - 10)
        cv2.putText(
            qc_bgr,
            label,
            (box.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            red,
            2,
        )

    return qc_bgr


def build_csv_row(
    row: pd.Series,
    svs_path: Optional[Path],
    qc_path: Optional[Path],
    output_crop_path: Optional[Path],
    section_id: int,
    thumb_shape: Optional[Tuple[int, int]],
    detection: Optional[DetectionBox],
    status: str,
) -> Dict[str, object]:
    thumb_width = thumb_shape[1] if thumb_shape else None
    thumb_height = thumb_shape[0] if thumb_shape else None
    return {
        "svs_path": str(svs_path) if svs_path else None,
        "svs_basename": svs_path.name if svs_path else None,
        "SvsFile": row.get("SvsFile"),
        "SlideNumber": row.get("SlideNumber"),
        "SampleID": row.get("SampleID"),
        "section_id": section_id,
        "section_label": SECTION_LABELS[section_id - 1],
        "x_thumb": detection.x if detection else None,
        "y_thumb": detection.y if detection else None,
        "w_thumb": detection.w if detection else None,
        "h_thumb": detection.h if detection else None,
        "thumb_width": thumb_width,
        "thumb_height": thumb_height,
        "output_crop_path": str(output_crop_path) if output_crop_path else None,
        "qc_path": str(qc_path) if qc_path else None,
        "status": status,
    }


def process_slide(
    row: pd.Series,
    svs_path: Path,
    output_dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    slide_number = sanitize_token(row.get("SlideNumber"))
    sample_id = sanitize_token(row.get("SampleID"))
    qc_path = output_dirs["qc"] / f"{sample_id}_{slide_number}_QC.jpg"
    records: List[Dict[str, object]] = []

    try:
        slide = openslide.OpenSlide(str(svs_path))
    except Exception as exc:
        logging.exception("Failed to open slide %s: %s", svs_path, exc)
        for section_id in range(1, 4):
            records.append(
                build_csv_row(
                    row=row,
                    svs_path=svs_path,
                    qc_path=None,
                    output_crop_path=None,
                    section_id=section_id,
                    thumb_shape=None,
                    detection=None,
                    status="open_error",
                )
            )
        return records

    try:
        thumb_rgb = load_thumbnail(slide, args.thumbnail_max_dim)
        thumb_shape = thumb_rgb.shape[:2]
        mask = segment_tissue_mask(thumb_rgb, args.white_thresh)
        boxes = find_detection_boxes(mask, args.min_area_frac, args.padding)

        qc_bgr = build_qc_image(thumb_rgb, boxes, slide_number, sample_id)
        save_jpeg(qc_bgr, qc_path, args.jpeg_quality)

        status = "ok" if len(boxes) == 3 else "insufficient_components"

        for section_id in range(1, 4):
            detection = boxes[section_id - 1] if section_id <= len(boxes) else None
            crop_path: Optional[Path] = None

            if detection is not None:
                crop_rgb = thumb_rgb[
                    detection.y : detection.y + detection.h,
                    detection.x : detection.x + detection.w,
                ]
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                crop_path = (
                    output_dirs["sections"]
                    / f"{sample_id}_{slide_number}_{SECTION_LABELS[section_id - 1]}.jpg"
                )
                save_jpeg(crop_bgr, crop_path, args.jpeg_quality)

            records.append(
                build_csv_row(
                    row=row,
                    svs_path=svs_path,
                    qc_path=qc_path,
                    output_crop_path=crop_path,
                    section_id=section_id,
                    thumb_shape=thumb_shape,
                    detection=detection,
                    status=status,
                )
            )

        logging.info(
            "Processed %s | found %d component(s) | QC: %s",
            svs_path.name,
            len(boxes),
            qc_path.name,
        )
        return records

    except Exception as exc:
        logging.exception("Failed while processing slide %s: %s", svs_path, exc)
        for section_id in range(1, 4):
            records.append(
                build_csv_row(
                    row=row,
                    svs_path=svs_path,
                    qc_path=qc_path if qc_path.exists() else None,
                    output_crop_path=None,
                    section_id=section_id,
                    thumb_shape=None,
                    detection=None,
                    status="processing_error",
                )
            )
        return records
    finally:
        slide.close()


def validate_metadata_columns(df: pd.DataFrame) -> None:
    required = {"SvsFile", "SlideNumber", "SampleID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dirs = ensure_output_dirs(output_dir)
    setup_logging(output_dirs["logs"] / "run.log")

    svs_dir = Path(args.svs_dir).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()

    logging.info("SVS directory: %s", svs_dir)
    logging.info("Metadata CSV: %s", metadata_csv)
    logging.info("Output directory: %s", output_dir)

    if not svs_dir.exists():
        raise FileNotFoundError(f"SVS directory does not exist: {svs_dir}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")

    metadata_df = pd.read_csv(metadata_csv)
    validate_metadata_columns(metadata_df)

    svs_files = list_svs_files(svs_dir)
    if not svs_files:
        raise FileNotFoundError(f"No .svs files found in {svs_dir}")

    svs_index = build_svs_index(svs_files)
    matched_svs_paths = set()
    csv_rows: List[Dict[str, object]] = []

    for _, row in metadata_df.iterrows():
        svs_path = match_svs_file(row, svs_files, svs_index)
        if svs_path is None:
            logging.warning(
                "No SVS match found for metadata row with SlideNumber='%s', SampleID='%s', SvsFile='%s'",
                row.get("SlideNumber"),
                row.get("SampleID"),
                row.get("SvsFile"),
            )
            for section_id in range(1, 4):
                csv_rows.append(
                    build_csv_row(
                        row=row,
                        svs_path=None,
                        qc_path=None,
                        output_crop_path=None,
                        section_id=section_id,
                        thumb_shape=None,
                        detection=None,
                        status="unmatched_metadata",
                    )
                )
            continue

        matched_svs_paths.add(svs_path)
        csv_rows.extend(process_slide(row, svs_path, output_dirs, args))

    unused_svs_files = [path for path in svs_files if path not in matched_svs_paths]
    for path in unused_svs_files:
        logging.warning("SVS file was not matched to metadata and was skipped: %s", path.name)

    output_csv = output_dir / "detected_sections.csv"
    output_df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
    output_df.to_csv(output_csv, index=False)
    logging.info("Wrote master CSV: %s", output_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
