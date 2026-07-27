#!/usr/bin/env python3
"""Run reproducible exploratory data analysis for the merged FIQA dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    GROUP_COLUMN,
    IDENTITY_COLUMN,
    IMAGE_COLUMN,
    TARGET,
    available_features,
    find_merged_data,
    group_order,
    load_runtime_config,
    prepare_numeric_columns,
    print_saved_files,
    save_json,
    stage_paths,
    validate_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create descriptive tables and figures for the merged FIQA dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/runtime_config.json"),
        help="Runtime JSON created by setup_project.py.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override for the merged CSV file.",
    )
    parser.add_argument(
        "--top-correlations",
        type=int,
        default=15,
        help="Number of target correlations shown in the bar plot.",
    )
    return parser.parse_args()


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    try:
        if args.top_correlations <= 0:
            raise ValueError("--top-correlations must be greater than zero.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "02_eda")

        df = pd.read_csv(data_file)
        validate_columns(df, [TARGET], context="Merged dataset")

        numeric_features = available_features(df, CONTINUOUS_FEATURES)
        binary_features = available_features(df, BINARY_FEATURES)
        df = prepare_numeric_columns(
            df,
            numeric_features=numeric_features,
            binary_features=binary_features,
            target=TARGET,
        )

        if df[TARGET].notna().sum() == 0:
            raise ValueError(f"Target column '{TARGET}' contains no usable values.")

        overview = pd.DataFrame(
            {
                "metric": [
                    "rows",
                    "columns",
                    "identities",
                    "demographic_groups",
                    "missing_target",
                    "duplicate_image_indices",
                ],
                "value": [
                    len(df),
                    df.shape[1],
                    df[IDENTITY_COLUMN].nunique(dropna=True)
                    if IDENTITY_COLUMN in df.columns
                    else np.nan,
                    df[GROUP_COLUMN].nunique(dropna=True)
                    if GROUP_COLUMN in df.columns
                    else np.nan,
                    int(df[TARGET].isna().sum()),
                    int(df[IMAGE_COLUMN].duplicated().sum())
                    if IMAGE_COLUMN in df.columns
                    else np.nan,
                ],
            }
        )
        overview.to_csv(paths["tables"] / "dataset_overview.csv", index=False)

        missing = pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": [int(df[column].isna().sum()) for column in df.columns],
                "missing_fraction": [float(df[column].isna().mean()) for column in df.columns],
                "dtype": [str(df[column].dtype) for column in df.columns],
                "unique_non_missing": [int(df[column].nunique(dropna=True)) for column in df.columns],
            }
        ).sort_values(["missing_fraction", "column"], ascending=[False, True])
        missing.to_csv(paths["tables"] / "missing_values.csv", index=False)

        analysis_numeric = available_features(
            df, [*CONTINUOUS_FEATURES, *BINARY_FEATURES, TARGET]
        )
        descriptive = df[analysis_numeric].describe().T.reset_index(names="feature")
        descriptive.to_csv(paths["tables"] / "descriptive_statistics.csv", index=False)

        support = pd.DataFrame(
            {
                "feature": analysis_numeric,
                "non_missing": [int(df[feature].notna().sum()) for feature in analysis_numeric],
                "missing": [int(df[feature].isna().sum()) for feature in analysis_numeric],
                "unique_values": [int(df[feature].nunique(dropna=True)) for feature in analysis_numeric],
                "minimum": [float(df[feature].min()) for feature in analysis_numeric],
                "maximum": [float(df[feature].max()) for feature in analysis_numeric],
            }
        )
        support.to_csv(paths["tables"] / "feature_support.csv", index=False)

        correlation_features = [
            feature
            for feature in available_features(df, [*CONTINUOUS_FEATURES, *BINARY_FEATURES])
            if df[feature].nunique(dropna=True) > 1
        ]
        correlation_rows = []
        for feature in correlation_features:
            pair = df[[feature, TARGET]].dropna()
            if len(pair) < 3:
                continue
            value = pair[feature].corr(pair[TARGET], method="pearson")
            correlation_rows.append(
                {
                    "feature": feature,
                    "pearson_correlation": float(value),
                    "absolute_correlation": float(abs(value)),
                    "n": len(pair),
                }
            )
        correlations = pd.DataFrame(correlation_rows)
        if not correlations.empty:
            correlations = correlations.sort_values(
                "absolute_correlation", ascending=False
            ).reset_index(drop=True)
        correlations.to_csv(
            paths["tables"] / "correlations_with_target.csv", index=False
        )

        target_values = df[TARGET].dropna()
        plt.figure(figsize=(9, 5))
        plt.hist(target_values, bins=40, edgecolor="black", alpha=0.8)
        plt.xlabel("CR-FIQA score")
        plt.ylabel("Number of images")
        plt.title("Distribution of CR-FIQA Scores")
        save_figure(paths["figures"] / "fiqa_score_distribution.png")

        plt.figure(figsize=(9, 2.8))
        plt.boxplot(target_values, vert=False)
        plt.xlabel("CR-FIQA score")
        plt.title("CR-FIQA Score Boxplot")
        save_figure(paths["figures"] / "fiqa_score_boxplot.png")

        group_counts = pd.DataFrame()
        group_summary = pd.DataFrame()
        if GROUP_COLUMN in df.columns and df[GROUP_COLUMN].notna().any():
            order = group_order(df[GROUP_COLUMN])
            group_counts = (
                df[GROUP_COLUMN]
                .value_counts(dropna=False)
                .rename_axis(GROUP_COLUMN)
                .reset_index(name="count")
            )
            group_counts["fraction"] = group_counts["count"] / len(df)
            group_counts.to_csv(paths["tables"] / "group_counts.csv", index=False)

            group_summary = (
                df.groupby(GROUP_COLUMN, dropna=False)[TARGET]
                .agg(count="count", mean="mean", std="std", median="median", minimum="min", maximum="max")
                .reset_index()
            )
            group_summary.to_csv(
                paths["tables"] / "group_score_summary.csv", index=False
            )

            count_lookup = group_counts.set_index(GROUP_COLUMN)["count"]
            ordered_counts = [int(count_lookup.get(group, 0)) for group in order]
            plt.figure(figsize=(10, 5))
            plt.bar(order, ordered_counts)
            plt.ylabel("Number of images")
            plt.title("Number of Images per Demographic Group")
            plt.xticks(rotation=35, ha="right")
            save_figure(paths["figures"] / "group_distribution.png")

            grouped_values = [
                df.loc[df[GROUP_COLUMN] == group, TARGET].dropna().to_numpy()
                for group in order
            ]
            plt.figure(figsize=(11, 6))
            plt.boxplot(grouped_values, tick_labels=order, showfliers=False)
            plt.ylabel("CR-FIQA score")
            plt.title("CR-FIQA Score Distribution by Demographic Group")
            plt.xticks(rotation=35, ha="right")
            save_figure(paths["figures"] / "fiqa_scores_by_group.png")

        if not correlations.empty:
            top = correlations.head(args.top_correlations).sort_values(
                "pearson_correlation"
            )
            plt.figure(figsize=(9, max(5, 0.35 * len(top))))
            plt.barh(top["feature"], top["pearson_correlation"])
            plt.axvline(0, linewidth=1)
            plt.xlabel("Pearson correlation with CR-FIQA score")
            plt.title("Strongest Feature Correlations with CR-FIQA")
            save_figure(paths["figures"] / "top_correlations_with_fiqa.png")

        metadata = {
            "data_file": str(data_file),
            "rows": len(df),
            "columns": df.shape[1],
            "continuous_features_found": numeric_features,
            "binary_features_found": binary_features,
            "target_non_missing": int(df[TARGET].notna().sum()),
        }
        save_json(metadata, paths["root"] / "run_metadata.json")

        print("EDA completed successfully.")
        print(f"Dataset: {data_file}")
        print(f"Results: {paths['root']}")
        print_saved_files(paths["root"])
        return 0

    except (FileNotFoundError, KeyError, ValueError, OSError, pd.errors.ParserError) as error:
        print(f"EDA failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
