# Results Summary

## Model Progression

### Enteritis Binary

- File: `enteritis_metrics_latest.csv`
- Use: summary metrics for the binary enteritis proof-of-principle baseline
- Reading: the result is modest but real under weak supervision

### Mononuclear Infiltration, Five Levels

- File: `mononuclear_metrics_latest.csv`
- Use: summary metrics for the five-level mononuclear classifier
- Reading: the result is biologically informative, but higher-severity classes are sparse

### Mononuclear Infiltration, Collapsed Binary

- File: `mononuclear_collapsed_binary_metrics_latest.csv`
- Use: summary metrics for low (`1-2`) versus moderate-to-severe (`3-5`) infiltration
- Reading: this is the strongest practical result among the staged model settings

## Supporting Screening And QC Tables

- `candidate_target_shortlist.csv`
- `ordinal_screen_results.csv`
- `quant_screen_results.csv`
- `overall_review_counts.csv`
- `tile_filter_summary.csv`

These files provide context for phenotype selection, QC review, and task framing.

