# MLHistologyAnalysis Public Repository

![Repository header](figures/RepHeader.png)

The current trained models are specific to rainbow trout distal intestine cross-sectional whole-slide histology. The workflow may be adaptable to other salmonid histology datasets, but that has not been established here.

## Project Overview

This repository documents a proof-of-principle workflow for:

1. identifying tissue regions on distal intestine H&E whole-slide images
2. extracting review-ready regions of interest
3. applying manual QC before tiling
4. generating tile manifests for training or inference
5. training weakly supervised image classifiers
6. aggregating tile predictions back to the analysis-unit level

## Workflow Summary

`slide -> ROI extraction -> manual QC -> tiling -> training/inference -> analysis-unit prediction`

- Preprocessing is manifest-driven and layout-aware.
- Manual review is a required gate before tiling.
- Training and inference both start from tile manifests.

See:

- [`docs/workflow_overview.md`](docs/workflow_overview.md)
- [`docs/preprocessing_guide.md`](docs/preprocessing_guide.md)
- [`docs/training_guide.md`](docs/training_guide.md)
- [`docs/inference_guide.md`](docs/inference_guide.md)
- [`docs/histology_methods.md`](docs/histology_methods.md)
- [`docs/scoring_rubric.md`](docs/scoring_rubric.md)
- [`docs/limitations_and_future_directions.md`](docs/limitations_and_future_directions.md)

## Current Model Summary

Three proof-of-principle model settings are represented in the staged public-safe results:

### 1. Enteritis Binary Baseline

- Target: binary enteritis endpoint under weak supervision
- Public interpretation: modest but real signal at the analysis-unit level
- Public-safe summary: analysis-unit AUROC `0.787`, analysis-unit AUPRC `0.392`, with `topk_mean` used as the primary aggregation method
- Main limitation: the endpoint is broad, while the supervision remains weak and slide-level

### 2. Mononuclear Infiltration 5-Level Classifier

- Target: five-level mononuclear infiltration score
- Public interpretation: biologically informative exploratory result
- Public-safe summary: analysis-unit accuracy `0.587`, weighted F1 `0.596`, quadratic weighted kappa `0.355`
- Main limitation: sparse representation of higher-severity classes limited learnability and evaluation

### 3. Mononuclear Infiltration Collapsed Binary Classifier

- Target: low infiltration (`1-2`) versus moderate-to-severe infiltration (`3-5`)
- Public interpretation: strongest practical proof-of-principle result among the staged model variants
- Public-safe summary: analysis-unit AUROC `0.821`, analysis-unit AUPRC `0.619`, with `topk_mean` aggregation
- Rationale: collapsing the scale reduces class sparsity and aligns better with the current amount of supervision

These results should be read as workflow-level proof of principle. Current limitations appear to be driven more by phenotype imbalance, sparse severe cases, and weak slide-level supervision than by the computational workflow itself.

## What Is Included

- Python scripts for ROI extraction, tiling, training, evaluation, and inference
- Public-safe YAML configuration templates
- Slurm templates with placeholders
- Template manifests with synthetic example rows only
- Summary result tables for the staged model runs
- Curated workflow, QC, screening, and model-result figures
- Placeholder documentation for methods and scoring sections

## What Is Not Included

- Raw slide data such as `.svs` files
- Full tile datasets
- Private metadata tables or real manifests
- Local absolute file paths
- Model checkpoints or packaged weights
- Detailed prediction tables, hard examples, full logs, or unpublished private artifacts

## Quick Start

Create an environment and install the Python requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run ROI extraction:

```bash
python scripts/preprocess_extract_rois.py \
  --slide-manifest manifests_templates/slide_manifest_template.csv \
  --config configs/preprocessing_multistudy_template.yaml \
  --output-dir outputs/roi_extraction
```

After manual QC, run tiling:

```bash
python scripts/preprocess_tile_rois.py \
  --roi-manifest outputs/roi_extraction/manifests/roi_manifest.csv \
  --config configs/tiling_public_template.yaml \
  --output-dir outputs/tiling
```

Run a training configuration:

```bash
python scripts/train.py \
  --config configs/enteritis_resnet18_public.yaml
```

Run inference with a user-supplied checkpoint:

```bash
python scripts/infer.py \
  --config configs/inference_enteritis_public.yaml
```

## Example Repository Tree

```text
MLHistologyAnalysis_public/
├── README.md
├── configs/
├── docs/
├── figures/
│   ├── model_results/
│   │   ├── enteritis_binary/
│   │   ├── mononuclear_5level/
│   │   └── mononuclear_collapsed_binary/
│   ├── qc_review/
│   ├── screening/
│   └── workflow_examples/
├── manifests_templates/
├── models/
├── results_summary/
├── scripts/
└── slurm/
```

## Raw Data And Model Scope

Raw slide data are not public in this repository.

Current trained models are rainbow trout specific. The code structure should be adaptable to other fish datasets, but any such reuse would require new data review, new model training, and new validation.
