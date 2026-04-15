# Inference Guide

This guide covers theinference path for users who already have tiled data and a local trained checkpoint.

## Scope

Inference starts from a tile manifest and produces analysis-unit predictions plus lightweight summary outputs.

The current public configs are organized around the same three task settings used in the staged result summaries:

- enteritis binary
- mononuclear infiltration, five levels
- mononuclear infiltration, collapsed binary

## Public Configurations

- [`configs/inference_enteritis_public.yaml`](../configs/inference_enteritis_public.yaml)
- [`configs/inference_mononuclear_5level_public.yaml`](../configs/inference_mononuclear_5level_public.yaml)
- [`configs/inference_mononuclear_binary_public.yaml`](../configs/inference_mononuclear_binary_public.yaml)

## Required Inputs

- a tile manifest, typically `outputs/tiling/manifests/tile_manifest.csv`
- a local checkpoint supplied by the user

Before running, update:

- `model.checkpoint_path`
- `data.tile_manifest_path`
- `data.data_root`

No model weights are included in this repository.

## Basic Commands

Enteritis:

```bash
python scripts/infer.py \
  --config configs/inference_enteritis_public.yaml
```

Mononuclear, five levels:

```bash
python scripts/infer.py \
  --config configs/inference_mononuclear_5level_public.yaml
```

Mononuclear, collapsed binary:

```bash
python scripts/infer.py \
  --config configs/inference_mononuclear_binary_public.yaml
```

## What The Script Does

The inference script:

- loads the tile manifest
- filters rows by task and read level
- resolves tile paths
- optionally joins supplemental metadata
- loads a checkpoint
- scores tiles
- aggregates predictions to the analysis-unit level
- writes summary outputs

## Typical Outputs

Each run writes under `results/<experiment_name>/`:

- `predictions/`
- `metrics/`
- `figures/`
- `logs/`

These run directories are useful locally but should not be committed to the public repository unless they are reduced to summary-level outputs first.

## Interpretation Notes

- Predictions are model outputs, not clinical or diagnostic calls.
- Probability values are not automatically calibrated.
- Aggregation choice matters. Binary tasks in this repository commonly use `topk_mean`.
- The current trained-model context is rainbow trout distal intestine H&E histology.

## Public Release Notes

Do not publish:

- checkpoints
- tile-level predictions
- top-tile exports
- local file paths
- private slide provenance
