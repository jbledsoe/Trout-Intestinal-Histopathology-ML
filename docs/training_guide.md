# Training Guide

This guide describes the training side of the public workflow.

## Training Context

The current examples are based on rainbow trout distal intestine H&E whole-slide histology. Training is weakly supervised at the slide or analysis-unit level and tile-based at model input time.

Each tile inherits the label of its parent analysis unit. Validation is then summarized at both the tile level and the analysis-unit level.

## Public Training Configurations

- [`configs/enteritis_resnet18_public.yaml`](../configs/enteritis_resnet18_public.yaml)
- [`configs/mononuclear_resnet18_public.yaml`](../configs/mononuclear_resnet18_public.yaml)
- [`configs/mononuclear_collapsed_public.yaml`](../configs/mononuclear_collapsed_public.yaml)

These configs define:

- task labels
- model backbone
- augmentation settings
- split strategy
- aggregation settings
- output layout

## Input Requirements

Training expects a tile manifest produced outside the public repository or generated locally from the preprocessing workflow.

At minimum, the manifest should contain:

- `tile_path`
- `analysis_unit_id`
- task label columns
- split-related metadata

The public schema reference is:

- [`manifests_templates/tile_manifest_template.csv`](../manifests_templates/tile_manifest_template.csv)

## Basic Commands

Enteritis:

```bash
python scripts/train.py \
  --config configs/enteritis_resnet18_public.yaml
```

Mononuclear, five levels:

```bash
python scripts/train.py \
  --config configs/mononuclear_resnet18_public.yaml
```

Mononuclear, collapsed binary:

```bash
python scripts/train.py \
  --config configs/mononuclear_collapsed_public.yaml
```

Dry run:

```bash
python scripts/train.py \
  --config configs/enteritis_resnet18_public.yaml \
  --dry-run
```

Resume with a local checkpoint:

```bash
python scripts/train.py \
  --config configs/enteritis_resnet18_public.yaml \
  --resume /path/to/local_checkpoint.pt
```

## Current Public Result Structure

The staged public summaries support three main interpretations:

### Enteritis Binary Baseline

- The binary enteritis endpoint produced a real but modest signal under weak supervision.
- This is a useful proof-of-principle result, but the limitations are visible in the public metrics and confusion matrices.

### Mononuclear Infiltration, Five Levels

- The five-level classifier appears biologically informative.
- Performance is constrained by sparse representation of the higher-severity classes.

### Mononuclear Infiltration, Collapsed Binary

- The collapsed binary formulation uses low scores `1-2` versus moderate-to-severe scores `3-5`.
- This formulation produced the strongest practical result in the staged public-safe summaries.

## Limits Of The Current Training Data

The main limitations appear to come from:

- weak slide-level supervision
- phenotype imbalance
- sparse severe cases

The public results do not suggest that the computational workflow itself is the main bottleneck. The larger limitation is that the available training data are not yet ideal for the harder phenotype definitions.

## Public Release Boundaries

- No checkpoints are distributed here.
- No split tables from the real runs are included.
- No tile-level prediction exports are included.
- No hard-example exports are included.

Public presentation should stay at the level of summary metrics, confusion matrices, and selected aggregate plots.
