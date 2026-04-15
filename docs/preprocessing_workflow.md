# Public Preprocessing Workflow

This workflow is the public-facing preprocessing path for whole-slide histology images in this repository.

It is designed to be:

- YAML driven
- manifest driven
- configurable for multiple study layouts
- safe to share without any private project data

## Scripts

- `scripts/preprocess_extract_rois.py`
  - reads a slide manifest plus a layout YAML
  - detects tissue components on slide thumbnails
  - selects sections according to study-specific rules
  - writes QC overlays, preview crops, and a review-ready ROI manifest
- `scripts/preprocess_tile_rois.py`
  - reads the reviewed ROI manifest
  - tiles only rows where `include_for_tiling == 1`
  - writes tiles plus a tile manifest for later inference or training

## Required User Inputs

### 1. Slide Manifest

Start from `manifests_templates/slide_manifest_template.csv`.

Minimum required columns:

- `slide_id`
- `study_key`
- `svs_path` or `svs_filename`

Common optional columns:

- `sample_id`
- `task`
- `split`
- `target_label`
- any extra metadata you want carried into the ROI and tile manifests

## 2. Layout YAML

Start from `configs/preprocessing_multistudy_template.yaml`.

The main editable parts are:

- `global`
  - thumbnail size
  - white threshold for tissue masking
  - morphology cleanup size
  - minimum component area
  - ROI bbox expansion margin
- `default_layout`
  - fallback layout for studies that do not have a custom entry
- `study_layouts`
  - one entry per `study_key`
  - selection and labeling rules for that layout

Supported selection modes:

- `keep_n_left_to_right`
  - keeps the top N detected components and assigns labels in left-to-right order
- `keep_largest`
  - keeps the single largest component and usually labels it `whole`
- `keep_representative`
  - picks one representative component from the top major components using a configurable rule

Supported representative rules:

- `bottom_left`
- `top_left`
- `center_most`
- `largest`

Supported naming modes:

- `suffix_labels`
  - produces IDs like `SAMPLE_001_L`
- `no_suffix`
  - produces IDs like `SAMPLE_001`

## Example Layouts

- `configs/layout_multisection_left_to_right.yaml`
  - analogous to slides that contain several clearly separated sections that should become labeled outputs such as `L`, `C`, `R`
- `configs/layout_single_section_whole.yaml`
  - analogous to slides where one dominant section should inherit the slide identity
- `configs/layout_representative_section.yaml`
  - analogous to slides with several candidate components where one representative component should be kept

## QC Pause Point

The manual pause happens after ROI extraction and before tiling.

Run extraction first:

```bash
python scripts/preprocess_extract_rois.py \
  --slide-manifest manifests_templates/slide_manifest_template.csv \
  --config configs/preprocessing_multistudy_template.yaml \
  --output-dir outputs/roi_extraction
```

This creates:

- `qc_overlays/`
- `preview_crops/`
- `manifests/roi_manifest.csv`
- `manifests/roi_issues.csv`

The ROI manifest deliberately sets every row to:

- `auto_qc_status = pending_manual_review`
- `include_for_tiling = 0`

Before tiling, review the QC outputs and then edit `manifests/roi_manifest.csv`:

- set `manual_qc_status` to something like `approved` or `reject`
- set `include_for_tiling = 1` only for approved ROIs
- optionally add comments in `reviewer_notes`

## Tiling

Start from `configs/tiling_public_template.yaml`.

Key editable fields:

- `tile_size_level0`
- `tile_size_px`
- `stride_level0`
- `min_tissue_fraction`
- `max_tiles_per_roi`
- `selection_strategy`

Run tiling only after manual QC:

```bash
python scripts/preprocess_tile_rois.py \
  --roi-manifest outputs/roi_extraction/manifests/roi_manifest.csv \
  --config configs/tiling_public_template.yaml \
  --output-dir outputs/tiling
```

## Outputs

ROI extraction outputs:

- per-slide QC overlays
- per-selected-ROI preview crops
- ROI manifest with manual QC columns
- issues table for missing slides or layout problems

Tiling outputs:

- tiles saved in `tiles/<task>/<split>/level_0/<study_key>/`
- `manifests/tile_manifest.csv`
- `manifests/tiling_summary.csv`

## Recommended Manual Break

The simplest practical review loop is:

1. Run ROI extraction.
2. Open `qc_overlays/` and `preview_crops/`.
3. Edit `manifests/roi_manifest.csv`.
4. Mark only approved ROIs with `include_for_tiling = 1`.
5. Run tiling.

This is intentionally low-tech and easy to explain to collaborators.
