#!/usr/bin/env python3
"""RQ2: univariate associations between image characteristics and CR-FIQA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pipeline_common import (
    BINARY_FEATURES,
    IDENTITY_COLUMN,
    IMAGE_COLUMN,
    GROUP_COLUMN,
    PRIMARY_CONTINUOUS_FEATURES,
    TARGET,
    available_features,
    find_merged_data,
    load_runtime_config,
    prepare_numeric_columns,
    print_saved_files,
    safe_fdr,
    save_json,
    stage_paths,
    validate_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Pearson, Spearman, point-biserial, binary group tests, "
            "effect sizes, and FDR-adjusted RQ2 summaries."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-plots", type=int, default=3)
    return parser.parse_args()


def correlation_table(
    frame: pd.DataFrame,
    features: list[str],
    method: str,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        pair = frame[[feature, TARGET]].dropna()
        if len(pair) < 3 or pair[feature].nunique() <= 1 or pair[TARGET].nunique() <= 1:
            coefficient = np.nan
            p_value = np.nan
        elif method == "pearson":
            coefficient, p_value = stats.pearsonr(pair[feature], pair[TARGET])
        elif method == "spearman":
            coefficient, p_value = stats.spearmanr(pair[feature], pair[TARGET])
        else:
            raise ValueError(f"Unsupported correlation method: {method}")

        rows.append(
            {
                "Feature": feature,
                "Coefficient": float(coefficient) if np.isfinite(coefficient) else np.nan,
                "Absolute_Coefficient": float(abs(coefficient)) if np.isfinite(coefficient) else np.nan,
                "P_Value": float(p_value) if np.isfinite(p_value) else np.nan,
                "N": len(pair),
                "Unique_Values": int(pair[feature].nunique()),
            }
        )

    result = pd.DataFrame(rows)
    adjusted, rejected = safe_fdr(result["P_Value"], alpha=alpha)
    result["Adjusted_P_Value_FDR"] = adjusted
    result["Significant_FDR"] = rejected
    return result.sort_values("Absolute_Coefficient", ascending=False, na_position="last").reset_index(drop=True)


def pooled_standard_deviation(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    numerator = (n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)
    denominator = n_a + n_b - 2
    if denominator <= 0:
        return np.nan
    value = np.sqrt(numerator / denominator)
    return float(value)


def binary_comparison_table(
    frame: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for feature in features:
        subset = frame[[feature, TARGET]].dropna()
        state_0 = subset.loc[subset[feature] == 0, TARGET].to_numpy(dtype=float)
        state_1 = subset.loc[subset[feature] == 1, TARGET].to_numpy(dtype=float)

        if len(state_0) < 2 or len(state_1) < 2:
            t_stat = t_p = u_stat = u_p = np.nan
            cohens_d = np.nan
        else:
            t_stat, t_p = stats.ttest_ind(state_1, state_0, equal_var=False, nan_policy="omit")
            u_stat, u_p = stats.mannwhitneyu(state_1, state_0, alternative="two-sided")
            pooled_sd = pooled_standard_deviation(state_1, state_0)
            cohens_d = (
                (float(np.mean(state_1)) - float(np.mean(state_0))) / pooled_sd
                if np.isfinite(pooled_sd) and pooled_sd > 0
                else np.nan
            )

        rows.append(
            {
                "Feature": feature,
                "N_0": len(state_0),
                "N_1": len(state_1),
                "Mean_0": float(np.mean(state_0)) if len(state_0) else np.nan,
                "Mean_1": float(np.mean(state_1)) if len(state_1) else np.nan,
                "Median_0": float(np.median(state_0)) if len(state_0) else np.nan,
                "Median_1": float(np.median(state_1)) if len(state_1) else np.nan,
                "Mean_Difference_1_minus_0": (
                    float(np.mean(state_1) - np.mean(state_0))
                    if len(state_0) and len(state_1)
                    else np.nan
                ),
                "Cohens_D": float(cohens_d) if np.isfinite(cohens_d) else np.nan,
                "Welch_T": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "Welch_P_Value": float(t_p) if np.isfinite(t_p) else np.nan,
                "Mann_Whitney_U": float(u_stat) if np.isfinite(u_stat) else np.nan,
                "Mann_Whitney_P_Value": float(u_p) if np.isfinite(u_p) else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    welch_adjusted, welch_rejected = safe_fdr(result["Welch_P_Value"], alpha=alpha)
    mw_adjusted, mw_rejected = safe_fdr(result["Mann_Whitney_P_Value"], alpha=alpha)
    result["Welch_FDR_P"] = welch_adjusted
    result["Welch_Significant_FDR"] = welch_rejected
    result["Mann_Whitney_FDR_P"] = mw_adjusted
    result["Mann_Whitney_Significant_FDR"] = mw_rejected
    result["Absolute_Cohens_D"] = result["Cohens_D"].abs()
    return result.sort_values("Absolute_Cohens_D", ascending=False, na_position="last").reset_index(drop=True)


def point_biserial_table(
    frame: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        subset = frame[[feature, TARGET]].dropna()
        if len(subset) < 3 or subset[feature].nunique() != 2:
            coefficient = p_value = np.nan
        else:
            coefficient, p_value = stats.pointbiserialr(subset[feature], subset[TARGET])
        rows.append(
            {
                "Feature": feature,
                "Coefficient": float(coefficient) if np.isfinite(coefficient) else np.nan,
                "Absolute_Coefficient": float(abs(coefficient)) if np.isfinite(coefficient) else np.nan,
                "P_Value": float(p_value) if np.isfinite(p_value) else np.nan,
                "N": len(subset),
            }
        )
    result = pd.DataFrame(rows)
    adjusted, rejected = safe_fdr(result["P_Value"], alpha=alpha)
    result["Adjusted_P_Value_FDR"] = adjusted
    result["Significant_FDR"] = rejected
    return result.sort_values("Absolute_Coefficient", ascending=False, na_position="last").reset_index(drop=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    try:
        if not 0 < args.alpha < 1:
            raise ValueError("--alpha must be between 0 and 1.")
        if args.top_plots < 0:
            raise ValueError("--top-plots cannot be negative.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "05_rq2_correlation_analysis")

        df = pd.read_csv(data_file)
        validate_columns(df, [TARGET], context="Merged dataset")

        continuous_features = available_features(df, PRIMARY_CONTINUOUS_FEATURES)
        binary_features = available_features(df, BINARY_FEATURES)
        protected_features = ["age"] if "age" in df.columns and df["age"].notna().any() else []
        all_numeric = [*continuous_features, *binary_features, *protected_features]

        df = prepare_numeric_columns(
            df,
            numeric_features=[*continuous_features, *protected_features],
            binary_features=binary_features,
            target=TARGET,
        )

        feature_support = pd.DataFrame(
            {
                "Feature": all_numeric,
                "Non_Missing": [int(df[feature].notna().sum()) for feature in all_numeric],
                "Missing": [int(df[feature].isna().sum()) for feature in all_numeric],
                "Unique_Values": [int(df[feature].nunique(dropna=True)) for feature in all_numeric],
            }
        )
        valid_primary = [
            feature
            for feature in [*continuous_features, *binary_features]
            if df[feature].nunique(dropna=True) > 1 and df[[feature, TARGET]].dropna().shape[0] >= 3
        ]
        valid_all = [
            feature
            for feature in all_numeric
            if df[feature].nunique(dropna=True) > 1 and df[[feature, TARGET]].dropna().shape[0] >= 3
        ]
        valid_binary = [feature for feature in binary_features if set(df[feature].dropna().unique()).issubset({0.0, 1.0}) and df[feature].nunique(dropna=True) == 2]

        descriptive = df[[*valid_all, TARGET]].describe().T.reset_index(names="Feature")
        binary_prevalence = pd.DataFrame(
            {
                "Feature": valid_binary,
                "N": [int(df[feature].notna().sum()) for feature in valid_binary],
                "Count_0": [int((df[feature] == 0).sum()) for feature in valid_binary],
                "Count_1": [int((df[feature] == 1).sum()) for feature in valid_binary],
                "Prevalence_1": [float(df[feature].mean()) for feature in valid_binary],
            }
        )

        pearson_all = correlation_table(df, valid_all, "pearson", args.alpha)
        spearman_all = correlation_table(df, valid_all, "spearman", args.alpha)
        pearson_primary = pearson_all.loc[pearson_all["Feature"].isin(valid_primary)].reset_index(drop=True)
        spearman_primary = spearman_all.loc[spearman_all["Feature"].isin(valid_primary)].reset_index(drop=True)
        point_biserial = point_biserial_table(df, valid_binary, args.alpha)
        binary_comparisons = binary_comparison_table(df, valid_binary, args.alpha)

        correlation_comparison = (
            pearson_primary[["Feature", "Coefficient", "P_Value", "Adjusted_P_Value_FDR", "Significant_FDR", "N"]]
            .rename(
                columns={
                    "Coefficient": "Pearson_R",
                    "P_Value": "Pearson_P_Value",
                    "Adjusted_P_Value_FDR": "Pearson_FDR_P",
                    "Significant_FDR": "Pearson_Significant_FDR",
                    "N": "Pearson_N",
                }
            )
            .merge(
                spearman_primary[["Feature", "Coefficient", "P_Value", "Adjusted_P_Value_FDR", "Significant_FDR", "N"]]
                .rename(
                    columns={
                        "Coefficient": "Spearman_Rho",
                        "P_Value": "Spearman_P_Value",
                        "Adjusted_P_Value_FDR": "Spearman_FDR_P",
                        "Significant_FDR": "Spearman_Significant_FDR",
                        "N": "Spearman_N",
                    }
                ),
                on="Feature",
                how="outer",
                validate="one_to_one",
            )
        )
        correlation_comparison["Same_Direction"] = np.sign(correlation_comparison["Pearson_R"]) == np.sign(correlation_comparison["Spearman_Rho"])
        correlation_comparison["Both_Significant_FDR"] = correlation_comparison["Pearson_Significant_FDR"] & correlation_comparison["Spearman_Significant_FDR"]
        correlation_comparison["Maximum_Absolute_Correlation"] = correlation_comparison[["Pearson_R", "Spearman_Rho"]].abs().max(axis=1)
        correlation_comparison = correlation_comparison.sort_values("Maximum_Absolute_Correlation", ascending=False, na_position="last").reset_index(drop=True)

        age_results = pearson_all.loc[pearson_all["Feature"] == "age"].merge(
            spearman_all.loc[spearman_all["Feature"] == "age"],
            on="Feature",
            how="outer",
            suffixes=("_Pearson", "_Spearman"),
        )

        focused_continuous = correlation_comparison.loc[
            correlation_comparison["Feature"].isin(continuous_features)
        ].copy()
        focused_binary = binary_comparisons.copy()

        rq2_summary = correlation_comparison.copy()
        rq2_summary["Robust_Association"] = rq2_summary["Same_Direction"] & rq2_summary["Both_Significant_FDR"]
        rq2_summary["Interpretation"] = np.select(
            [
                rq2_summary["Robust_Association"],
                rq2_summary[["Pearson_Significant_FDR", "Spearman_Significant_FDR"]].any(axis=1),
            ],
            [
                "Pearson and Spearman agree in direction and remain significant after FDR correction.",
                "Significant after FDR correction in one method only.",
            ],
            default="No statistically significant association after FDR correction.",
        )

        tables = {
            "feature_support.csv": feature_support,
            "descriptive_statistics.csv": descriptive,
            "binary_feature_prevalence.csv": binary_prevalence,
            "pearson_correlations_all_features.csv": pearson_all,
            "pearson_correlations_primary.csv": pearson_primary,
            "spearman_correlations_all_features.csv": spearman_all,
            "spearman_correlations_primary.csv": spearman_primary,
            "point_biserial_correlations.csv": point_biserial,
            "pearson_spearman_comparison.csv": correlation_comparison,
            "binary_feature_comparisons.csv": binary_comparisons,
            "protected_age_correlation.csv": age_results,
            "focused_continuous_summary.csv": focused_continuous,
            "focused_binary_summary.csv": focused_binary,
            "rq2_summary.csv": rq2_summary,
        }
        for filename, table in tables.items():
            table.to_csv(paths["tables"] / filename, index=False)

        if not correlation_comparison.empty:
            top = correlation_comparison.head(15).sort_values("Maximum_Absolute_Correlation")
            plt.figure(figsize=(10, max(5, len(top) * 0.36)))
            plt.barh(top["Feature"], top["Spearman_Rho"])
            plt.axvline(0, linewidth=1)
            plt.xlabel("Spearman correlation with CR-FIQA")
            plt.title("RQ2 Feature–FIQA Associations")
            save_figure(paths["figures"] / "spearman_correlations_primary.png")

        for feature in focused_continuous["Feature"].head(args.top_plots):
            pair = df[[feature, TARGET]].dropna()
            plt.figure(figsize=(7, 5))
            plt.scatter(pair[feature], pair[TARGET], alpha=0.35, s=14)
            if len(pair) >= 2:
                coefficients = np.polyfit(pair[feature], pair[TARGET], 1)
                x_values = np.linspace(pair[feature].min(), pair[feature].max(), 100)
                plt.plot(x_values, coefficients[0] * x_values + coefficients[1])
            plt.xlabel(feature)
            plt.ylabel("CR-FIQA score")
            plt.title(f"{feature} and CR-FIQA Score")
            save_figure(paths["figures"] / f"scatter_{feature}.png")

        if not binary_comparisons.empty:
            plot_data = binary_comparisons.sort_values("Cohens_D")
            plt.figure(figsize=(9, max(5, len(plot_data) * 0.4)))
            plt.barh(plot_data["Feature"], plot_data["Cohens_D"])
            plt.axvline(0, linewidth=1)
            plt.xlabel("Cohen's d (state 1 minus state 0)")
            plt.title("Binary Feature Effect Sizes")
            save_figure(paths["figures"] / "binary_feature_effect_sizes.png")

        metadata = {
            "alpha": args.alpha,
            "fdr_method": "Benjamini-Hochberg",
            "data_file": str(data_file),
            "continuous_features": continuous_features,
            "binary_features": binary_features,
            "valid_primary_features": valid_primary,
            "robust_associations": rq2_summary.loc[rq2_summary["Robust_Association"], "Feature"].tolist(),
        }
        save_json(metadata, paths["root"] / "run_metadata.json")

        print("RQ2 correlation analysis completed successfully.")
        print(f"Robust associations: {metadata['robust_associations']}")
        print_saved_files(paths["root"])
        return 0

    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"Correlation analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
