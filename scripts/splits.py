"""Split creation and validation utilities for histology tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class SplitArtifacts:
    """Container for split data frames and summary metadata."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    summary: Dict[str, object]


def validate_no_group_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_column: str = "analysis_unit_id",
) -> None:
    """Raise if the same group appears in both train and validation."""
    overlap = set(train_df[group_column].unique()) & set(val_df[group_column].unique())
    if overlap:
        raise ValueError(
            f"Detected {len(overlap)} leaked {group_column} values between train and validation."
        )


def validate_binary_class_presence(train_df: pd.DataFrame, val_df: pd.DataFrame, label_column: str) -> None:
    """Ensure both binary classes are represented in train and val."""
    train_labels = set(train_df[label_column].astype(int).unique().tolist())
    val_labels = set(val_df[label_column].astype(int).unique().tolist())
    expected = {0, 1}
    if train_labels != expected or val_labels != expected:
        raise ValueError(
            f"Binary split is invalid. Train labels={sorted(train_labels)}, val labels={sorted(val_labels)}."
        )


def build_fixed_enteritis_split(manifest_df: pd.DataFrame, label_column: str) -> SplitArtifacts:
    """Use the manifest's existing train/val split for the enteritis task."""
    train_df = manifest_df.loc[manifest_df["split"] == "train"].copy()
    val_df = manifest_df.loc[manifest_df["split"] == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Enteritis split requires non-empty manifest train and val subsets.")
    if not train_df["tile_path"].str.contains("/train/", regex=False).all():
        raise ValueError("Manifest train rows are inconsistent with enteritis train tile paths.")
    if not val_df["tile_path"].str.contains("/val/", regex=False).all():
        raise ValueError("Manifest val rows are inconsistent with enteritis val tile paths.")

    validate_no_group_leakage(train_df, val_df)
    validate_binary_class_presence(train_df, val_df, label_column)

    summary = {
        "split_strategy": "fixed_manifest_split",
        "train_tiles": int(len(train_df)),
        "val_tiles": int(len(val_df)),
        "train_analysis_units": int(train_df["analysis_unit_id"].nunique()),
        "val_analysis_units": int(val_df["analysis_unit_id"].nunique()),
        "train_class_counts": train_df[label_column].astype(int).value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_column].astype(int).value_counts().sort_index().to_dict(),
    }
    return SplitArtifacts(train_df=train_df, val_df=val_df, summary=summary)


def _group_level_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    grouped = (
        df.groupby("analysis_unit_id", as_index=False)
        .agg(
            {
                label_column: "first",
                "study_id": "first",
                "review_category": "first",
                "split": "first",
            }
        )
        .rename(columns={label_column: "group_label"})
    )
    return grouped


def build_group_stratified_split(
    manifest_df: pd.DataFrame,
    label_column: str,
    val_fraction: float,
    random_seed: int,
    allow_fallback_random: bool = False,
) -> SplitArtifacts:
    """Build a group-level stratified split on analysis-unit labels."""
    grouped = _group_level_labels(manifest_df, label_column)
    label_counts = grouped["group_label"].astype(int).value_counts().sort_index()
    if label_counts.min() < 2:
        if not allow_fallback_random:
            raise ValueError(
                "Group-level stratified splitting is unstable because at least one class "
                "has fewer than two analysis units. Enable fallback or revise the cohort."
            )
        shuffled = grouped.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        val_groups = set(shuffled.iloc[:val_count]["analysis_unit_id"].tolist())
        strategy = "group_random_fallback"
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=random_seed,
        )
        train_index, val_index = next(
            splitter.split(grouped["analysis_unit_id"], grouped["group_label"].astype(int))
        )
        val_groups = set(grouped.iloc[val_index]["analysis_unit_id"].tolist())
        strategy = "group_stratified_shuffle_split"

    val_df = manifest_df.loc[manifest_df["analysis_unit_id"].isin(val_groups)].copy()
    train_df = manifest_df.loc[~manifest_df["analysis_unit_id"].isin(val_groups)].copy()
    validate_no_group_leakage(train_df, val_df)

    summary = {
        "split_strategy": strategy,
        "train_tiles": int(len(train_df)),
        "val_tiles": int(len(val_df)),
        "train_analysis_units": int(train_df["analysis_unit_id"].nunique()),
        "val_analysis_units": int(val_df["analysis_unit_id"].nunique()),
        "train_class_counts": train_df[label_column].astype(int).value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_column].astype(int).value_counts().sort_index().to_dict(),
        "group_class_counts": label_counts.to_dict(),
    }
    return SplitArtifacts(train_df=train_df, val_df=val_df, summary=summary)


def build_group_stratified_cv(
    manifest_df: pd.DataFrame,
    label_column: str,
    n_splits: int,
    random_seed: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create optional group-aware stratified CV folds on analysis units."""
    grouped = _group_level_labels(manifest_df, label_column)
    label_counts = grouped["group_label"].astype(int).value_counts()
    if label_counts.min() < n_splits:
        raise ValueError(
            "Cannot build stratified CV because at least one class has fewer analysis units "
            f"than the requested number of folds ({n_splits})."
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_index, val_index in splitter.split(grouped["analysis_unit_id"], grouped["group_label"].astype(int)):
        train_groups = set(grouped.iloc[train_index]["analysis_unit_id"].tolist())
        val_groups = set(grouped.iloc[val_index]["analysis_unit_id"].tolist())
        train_df = manifest_df.loc[manifest_df["analysis_unit_id"].isin(train_groups)].copy()
        val_df = manifest_df.loc[manifest_df["analysis_unit_id"].isin(val_groups)].copy()
        validate_no_group_leakage(train_df, val_df)
        folds.append((train_df, val_df))
    return folds
