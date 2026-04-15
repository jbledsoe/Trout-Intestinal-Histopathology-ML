# Limitations And Future Directions

## Current Limits

### Data Availability

- Raw `.svs` slides are not included.
- Tile datasets are not included.
- Real manifests and private metadata are not included.
- Model checkpoints are not included.

### Supervision Structure

The current workflow is weakly supervised. Labels are broader than the tile-level visual patterns the network sees during training. That limitation is especially relevant for the binary enteritis endpoint.

### Phenotype Imbalance

Higher-severity mononuclear classes are sparsely represented in the staged results. That limits both model learning and evaluation stability for the five-level classifier.

### Scope Of Current Models

The current trained models are specific to rainbow trout distal intestine H&E whole-slide histology. The code may be adaptable to related salmonid datasets, but the current weights and result summaries should not be treated as validated outside this setting.

### Interpretation

The current staged results are proof-of-principle only. They show that the workflow can recover useful signal from whole-slide histology under weak supervision, but they do not replace expert pathology review.

## Practical Reading Of The Current Results

- Enteritis: modest but real signal
- Mononuclear, five levels: biologically informative but limited by sparse higher-severity classes
- Mononuclear, collapsed binary: strongest practical result among the staged experiments

This pattern suggests that the main constraint is the training signal available for the phenotype definitions, not the general structure of the preprocessing and modeling workflow.

## Future Directions

### Data

- expand representation of severe phenotypes
- improve balance across score levels
- prepare a governed release pathway for any future public data package

### Labels

- add stronger region-level or tile-level supervision where feasible
- refine endpoint definitions for tasks that are currently broad at the slide level

### Public Documentation

- add a fuller methods section once the project is ready for manuscript-facing release
- add a data dictionary for manifest fields
- add a small synthetic example dataset if release constraints allow
