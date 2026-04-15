# Public Inference Workflow

This repository includes a lightweight inference path for users who have already:

1. run ROI extraction
2. manually reviewed ROIs
3. run tiling
4. obtained trained checkpoint files outside this repository

The inference workflow is intentionally:

- config driven
- manifest driven
- checkpoint agnostic
- compatible with multiple trained models

## Script

- `scripts/infer.py`
  - reads a tile manifest produced by `scripts/preprocess_tile_rois.py`
  - optionally joins extra preprocessing metadata from additional CSVs
  - loads a trained checkpoint
  - runs tile inference
  - aggregates tile predictions to analysis-unit predictions
  - writes CSV outputs and a few lightweight review artifacts

## Supported Public Model Profiles

- `configs/inference_enteritis_public.yaml`
  - binary enteritis
- `configs/inference_mononuclear_binary_public.yaml`
  - collapsed mononuclear infiltration
  - low (`1-2`) vs. moderate-severe (`3-5`)
- `configs/inference_mononuclear_5level_public.yaml`
  - 5-level mononuclear infiltration classifier

## Inputs

Minimum practical input is:

- `outputs/tiling/manifests/tile_manifest.csv`
- a local checkpoint file such as `best.pt`

Optional supplemental preprocessing tables can be merged in through `data.metadata_joins`, for example:

- ROI manifest fields like `manual_qc_status`
- reviewer notes
- other public-safe columns keyed by `analysis_unit_id`

## Typical Usage

Example enteritis run:

```bash
python scripts/infer.py \
  --config configs/inference_enteritis_public.yaml
```

Example mononuclear binary run:

```bash
python scripts/infer.py \
  --config configs/inference_mononuclear_binary_public.yaml
```

Example mononuclear 5-level run:

```bash
python scripts/infer.py \
  --config configs/inference_mononuclear_5level_public.yaml
```

Before running, update at least:

- `model.checkpoint_path`
- `data.tile_manifest_path`
- `data.data_root`

## Output Layout

Each run writes to:

- `results/<experiment_name>/predictions/`
- `results/<experiment_name>/figures/`
- `results/<experiment_name>/metrics/`
- `results/<experiment_name>/logs/`

Core outputs:

- `predictions/resolved_inference_manifest.csv`
- `predictions/tile_predictions.csv`
- `predictions/analysis_unit_predictions.csv`
- `predictions/top_scoring_tiles_global.csv`
- `predictions/top_scoring_tiles_per_analysis_unit.csv`
- `metrics/inference_summary.csv`

Optional figures:

- binary: `figures/analysis_unit_score_histogram.png`
- multiclass: `figures/analysis_unit_predicted_class_counts.png`

## Example Output Schema

### Tile Prediction CSV

Common columns:

- `analysis_unit_id`
- `study_id`
- `tile_path`
- `resolved_tile_path`
- `filename`
- `split`
- `pred_label`
- `pred_class_name`
- `confidence`

Binary-specific columns:

- `prob_negative`
- `prob_positive`

Multiclass-specific columns:

- `prob_class_1` ... `prob_class_5`

If label columns are present in the manifest and mapped in config, `target` is also carried through for reference.

### Analysis-Unit Prediction CSV

Common columns:

- `analysis_unit_id`
- `study_id`
- `tile_count`
- `pred_label`
- `pred_class_name`
- `confidence`

Binary-specific columns:

- `aggregated_score`

Multiclass-specific columns:

- `prob_score_1` ... `prob_score_5`

Additional public-safe metadata columns from the tile manifest can also be carried through.

## QC and Interpretation Responsibilities

Users are still responsible for:

- confirming ROI approval before tiling
- checking that the checkpoint matches the config backbone and class count
- verifying the tile manifest task matches the intended model
- reviewing top-scoring tiles for obvious artifacts or out-of-scope morphology
- treating predictions as model outputs, not a substitute for pathology review

## Practical Limitations

- No checkpoints are shipped in this public repo.
- The script does not calibrate probabilities.
- Aggregation is intentionally simple and config controlled.
- The workflow assumes tiles already exist and does not rerun preprocessing.
- If the manifest mixes tasks, users should filter with `data.allowed_tasks`.
