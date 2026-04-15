# Preprocessing Guide

This guide covers the path from local whole-slide images to a tile manifest suitable for training or inference.

## Scope

The preprocessing workflow is split into two stages:

1. ROI extraction
2. tiling of manually approved ROIs

This split is intentional. It keeps the review step explicit and prevents automatic tiling of low-quality or out-of-scope regions.

## Expected Input Material

- Tissue: distal intestine
- Stain: H&E
- Section format: cross-sectional whole-slide histology
- Current trained-model context: rainbow trout

The preprocessing scripts are configurable and may be adaptable to related salmonid histology datasets, but the public examples here were prepared for the rainbow trout setting above.

## Required Inputs

### Slide Manifest

Start from [`manifests_templates/slide_manifest_template.csv`](../manifests_templates/slide_manifest_template.csv).

Required columns:

- `slide_id`
- `study_key`
- `svs_path` or `svs_filename`

Typical optional columns:

- `sample_id`
- `task`
- `split`
- label columns used later for model training
- public-safe metadata fields to carry into ROI and tile manifests

### Layout Configuration

Start from [`configs/preprocessing_multistudy_template.yaml`](../configs/preprocessing_multistudy_template.yaml).

The layout configuration controls:

- thumbnail generation
- tissue masking
- component filtering
- ROI bounding-box expansion
- section selection rules by study layout
- analysis-unit naming rules

Reference layouts:

- `configs/layout_multisection_left_to_right.yaml`
- `configs/layout_single_section_whole.yaml`
- `configs/layout_representative_section.yaml`

## Step 1: Extract Review-Ready ROIs

```bash
python scripts/preprocess_extract_rois.py \
  --slide-manifest manifests_templates/slide_manifest_template.csv \
  --config configs/preprocessing_multistudy_template.yaml \
  --output-dir outputs/roi_extraction
```

Optional:

```bash
python scripts/preprocess_extract_rois.py \
  --slide-manifest manifests_templates/slide_manifest_template.csv \
  --config configs/preprocessing_multistudy_template.yaml \
  --output-dir outputs/roi_extraction \
  --svs-root /path/to/local/slides
```

The script:

- reads the slide manifest
- resolves slide paths
- generates thumbnails
- detects tissue components
- applies study-specific layout rules
- writes QC overlays and ROI preview crops
- writes a review-ready ROI manifest

Primary outputs:

- `outputs/roi_extraction/qc_overlays/`
- `outputs/roi_extraction/preview_crops/`
- `outputs/roi_extraction/manifests/roi_manifest.csv`
- `outputs/roi_extraction/manifests/roi_issues.csv`

## Step 2: Manual QC

Manual review is required before tiling.

The ROI manifest is initialized with:

- `auto_qc_status = pending_manual_review`
- `include_for_tiling = 0`

Recommended review process:

1. inspect the QC overlays
2. inspect the ROI preview crops
3. update `manual_qc_status`
4. set `include_for_tiling = 1` only for approved ROIs
5. record short notes in `reviewer_notes` if needed

Public example images are included in `figures/workflow_examples/`. These remain representative biological images and should receive a final manual release check before public push.

## Step 3: Tile Approved ROIs

```bash
python scripts/preprocess_tile_rois.py \
  --roi-manifest outputs/roi_extraction/manifests/roi_manifest.csv \
  --config configs/tiling_public_template.yaml \
  --output-dir outputs/tiling
```

Key tiling parameters:

- `tile_size_level0`
- `tile_size_px`
- `stride_level0`
- `min_tissue_fraction`
- `max_tiles_per_roi`
- `selection_strategy`

The public template uses `selection_strategy: spatial_farthest`, which provides a simple way to avoid retaining only dense local clusters.

## Outputs

ROI extraction outputs:

- QC overlays
- preview crops
- ROI manifest
- issue table

Tiling outputs:

- tile image files
- `manifests/tile_manifest.csv`
- `manifests/tiling_summary.csv`
- log files

The tile manifest is the handoff artifact for both training and inference.

## Public Release Notes

- Keep real slide files outside the repository.
- Do not commit real ROI or tile manifests.
- Do not commit tile datasets.
- Keep representative workflow images under manual review until final release approval.
