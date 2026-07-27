#!/usr/bin/env python3
"""Assess demographic consistency of feature–CR-FIQA relationships.

The primary evidence for heterogeneity is the FDR-adjusted joint test of all
feature-by-demographic-group interaction terms. Group-specific heatmaps are
provided as descriptive summaries, not as stand-alone significance tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline_common import (
    BINARY_FEATURES,
    DEMOGRAPHIC_CONTINUOUS_FEATURES,
    FACIAL_HAIR_FEATURES,
    GROUP_COLUMN,
    IDENTITY_COLUMN,
    TARGET,
    available_features,
    find_merged_data,
    group_order,
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
            "Estimate group-wise univariate and adjusted effects, formal global "
            "feature-by-group interaction tests, and male-only facial-hair analyses."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-state-count", type=int, default=15)
    parser.add_argument("--min-continuous-count", type=int, default=30)
    parser.add_argument("--min-nonzero-hair", type=int, default=15)
    parser.add_argument("--max-vif", type=float, default=10.0)
    return parser.parse_args()


def robust_fit(X: pd.DataFrame, y: pd.Series, clusters: pd.Series | None = None):
    design = sm.add_constant(X.astype(float), has_constant="add")
    ols = sm.OLS(y.astype(float), design).fit()
    if clusters is not None and clusters.notna().sum() == len(clusters) and clusters.nunique() >= 2:
        robust = ols.get_robustcov_results(cov_type="cluster", groups=clusters)
        covariance = "clustered by identity"
    else:
        robust = ols.get_robustcov_results(cov_type="HC3")
        covariance = "HC3"
    return ols, robust, design, covariance


def robust_coefficient_table(ols, robust, covariance: str) -> pd.DataFrame:
    names = list(ols.model.exog_names)
    confidence = np.asarray(robust.conf_int(), dtype=float)
    return pd.DataFrame(
        {
            "term": names,
            "coefficient": np.asarray(robust.params, dtype=float),
            "standard_error": np.asarray(robust.bse, dtype=float),
            "t_value": np.asarray(robust.tvalues, dtype=float),
            "p_value": np.asarray(robust.pvalues, dtype=float),
            "ci_lower_95": confidence[:, 0],
            "ci_upper_95": confidence[:, 1],
            "covariance": covariance,
        }
    )


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["term", "vif"])
    values = frame.astype(float)
    rows = []
    for position, term in enumerate(values.columns):
        try:
            value = variance_inflation_factor(values.to_numpy(), position)
        except Exception:
            value = np.nan
        rows.append({"term": term, "vif": float(value) if np.isfinite(value) else value})
    return pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last").reset_index(drop=True)


def vif_reduce(frame: pd.DataFrame, max_vif: float) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    predictors = list(frame.columns)
    history = []
    while len(predictors) > 1:
        complete = frame[predictors].dropna()
        vif = calculate_vif(complete)
        finite = vif.replace([np.inf, -np.inf], np.nan).dropna(subset=["vif"])
        if finite.empty:
            break
        highest = finite.iloc[0]
        if float(highest["vif"]) <= max_vif:
            break
        removed = str(highest["term"])
        history.append({"removed_term": removed, "vif_at_removal": float(highest["vif"])})
        predictors.remove(removed)
    final_complete = frame[predictors].dropna()
    return predictors, calculate_vif(final_complete), pd.DataFrame(history)


def feature_name(term: str) -> str:
    return term.removeprefix("z_")


def build_interaction_design(
    frame: pd.DataFrame,
    predictors: list[str],
    groups: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]], str]:
    reference = groups[0]
    design = frame[predictors].copy()
    interaction_columns: dict[str, list[str]] = {predictor: [] for predictor in predictors}

    for group in groups[1:]:
        dummy_name = f"group::{group}"
        dummy = (frame[GROUP_COLUMN] == group).astype(float)
        design[dummy_name] = dummy
        for predictor in predictors:
            interaction_name = f"interaction::{predictor}::{group}"
            design[interaction_name] = frame[predictor] * dummy
            interaction_columns[predictor].append(interaction_name)

    return design, interaction_columns, reference


def joint_interaction_tests(
    robust,
    exog_names: list[str],
    interaction_columns: dict[str, list[str]],
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for predictor, terms in interaction_columns.items():
        positions = [exog_names.index(term) for term in terms if term in exog_names]
        if not positions:
            statistic = p_value = np.nan
            degrees = 0
        else:
            restrictions = np.zeros((len(positions), len(exog_names)))
            for row, position in enumerate(positions):
                restrictions[row, position] = 1.0
            test = robust.wald_test(restrictions, scalar=True)
            statistic = float(np.asarray(test.statistic).reshape(-1)[0])
            p_value = float(np.asarray(test.pvalue).reshape(-1)[0])
            degrees = len(positions)
        rows.append(
            {
                "feature": feature_name(predictor),
                "predictor_term": predictor,
                "wald_statistic": statistic,
                "degrees_of_freedom": degrees,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    adjusted, rejected = safe_fdr(result["p_value"], alpha=alpha)
    result["fdr_p_value"] = adjusted
    result["significant_fdr"] = rejected
    return result.sort_values("fdr_p_value", na_position="last").reset_index(drop=True)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    if matrix.empty:
        return
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")
    plt.figure(figsize=(max(8, 1.2 * len(matrix.columns)), max(5, 0.45 * len(matrix))))
    image = plt.imshow(values, aspect="auto", cmap=cmap)
    plt.colorbar(image, label=colorbar_label)
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    plt.yticks(np.arange(len(matrix.index)), matrix.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def group_specific_models(
    frame: pd.DataFrame,
    groups: list[str],
    predictors: list[str],
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for group in groups:
        subset_columns = [TARGET, *predictors]
        if IDENTITY_COLUMN in frame.columns:
            subset_columns.append(IDENTITY_COLUMN)
        subset = frame.loc[frame[GROUP_COLUMN] == group, subset_columns].dropna().copy()
        if len(subset) <= len(predictors) + 2:
            continue
        clusters = subset[IDENTITY_COLUMN] if IDENTITY_COLUMN in subset.columns else None
        ols, robust, _, covariance = robust_fit(subset[predictors], subset[TARGET], clusters)
        table = robust_coefficient_table(ols, robust, covariance)
        table = table.loc[table["term"] != "const"].copy()
        table["group"] = group
        table["feature"] = table["term"].map(feature_name)
        table["n"] = len(subset)
        table["identities"] = int(clusters.nunique()) if clusters is not None else np.nan
        rows.append(table)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    adjusted, rejected = safe_fdr(result["p_value"], alpha=alpha)
    result["fdr_p_value"] = adjusted
    result["significant_fdr"] = rejected
    return result


def main() -> int:
    args = parse_args()

    try:
        if not 0 < args.alpha < 1:
            raise ValueError("--alpha must be between 0 and 1.")
        if args.min_state_count < 1 or args.min_continuous_count < 3 or args.min_nonzero_hair < 1:
            raise ValueError("Support thresholds must be positive.")
        if args.max_vif <= 1:
            raise ValueError("--max-vif must be greater than 1.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "08_demographic_consistency")

        df = pd.read_csv(data_file)
        continuous = available_features(df, DEMOGRAPHIC_CONTINUOUS_FEATURES)
        binary = available_features(df, BINARY_FEATURES)
        facial_hair = available_features(df, FACIAL_HAIR_FEATURES)
        validate_columns(df, [TARGET, GROUP_COLUMN], context="Merged dataset")

        df = prepare_numeric_columns(
            df,
            numeric_features=[*continuous, *facial_hair],
            binary_features=binary,
            target=TARGET,
        )
        df = df.dropna(subset=[TARGET, GROUP_COLUMN]).copy()
        df[GROUP_COLUMN] = df[GROUP_COLUMN].astype(str)
        groups = group_order(df[GROUP_COLUMN])
        if len(groups) < 2:
            raise ValueError("At least two demographic groups are required.")

        score_summary = (
            df.groupby(GROUP_COLUMN)[TARGET]
            .agg(count="count", mean="mean", std="std", median="median", minimum="min", maximum="max")
            .reindex(groups)
            .reset_index()
        )
        score_summary.to_csv(paths["tables"] / "demographic_score_summary.csv", index=False)

        grouped_scores = [df.loc[df[GROUP_COLUMN] == group, TARGET].dropna() for group in groups]
        plt.figure(figsize=(11, 6))
        plt.boxplot(grouped_scores, tick_labels=groups, showfliers=False)
        plt.ylabel("CR-FIQA score")
        plt.title("CR-FIQA Scores by Demographic Group")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(paths["figures"] / "fiqa_scores_by_group.png", dpi=300, bbox_inches="tight")
        plt.close()

        support_rows = []
        for group in groups:
            group_df = df.loc[df[GROUP_COLUMN] == group]
            for feature in continuous:
                values = group_df[feature].dropna()
                support_rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "feature_type": "continuous",
                        "non_missing": len(values),
                        "unique_values": int(values.nunique()),
                        "n_zero": np.nan,
                        "n_one_or_nonzero": np.nan,
                        "sufficient_support": len(values) >= args.min_continuous_count and values.nunique() >= 3,
                    }
                )
            for feature in binary:
                values = group_df[feature].dropna()
                n_zero = int((values == 0).sum())
                n_one = int((values == 1).sum())
                support_rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "feature_type": "binary",
                        "non_missing": len(values),
                        "unique_values": int(values.nunique()),
                        "n_zero": n_zero,
                        "n_one_or_nonzero": n_one,
                        "sufficient_support": n_zero >= args.min_state_count and n_one >= args.min_state_count,
                    }
                )
        support = pd.DataFrame(support_rows)
        support.to_csv(paths["tables"] / "feature_support_by_group.csv", index=False)

        supported_continuous = [
            feature
            for feature in continuous
            if support.loc[(support["feature"] == feature) & (support["feature_type"] == "continuous"), "sufficient_support"].all()
        ]
        supported_binary = [
            feature
            for feature in binary
            if support.loc[(support["feature"] == feature) & (support["feature_type"] == "binary"), "sufficient_support"].all()
        ]

        for feature in supported_continuous:
            mean = df[feature].mean()
            std = df[feature].std(ddof=0)
            if not np.isfinite(std) or std == 0:
                continue
            df[f"z_{feature}"] = (df[feature] - mean) / std

        candidate_predictors = [f"z_{feature}" for feature in supported_continuous] + supported_binary
        if not candidate_predictors:
            raise ValueError(
                "No common predictors meet the support thresholds in every demographic group. "
                "Inspect feature_support_by_group.csv or lower the thresholds deliberately."
            )

        complete_for_vif = df[candidate_predictors].dropna()
        final_predictors, final_vif, vif_history = vif_reduce(complete_for_vif, args.max_vif)
        final_vif["feature"] = final_vif["term"].map(feature_name)
        final_vif.to_csv(paths["tables"] / "final_common_predictor_vif.csv", index=False)
        vif_history.to_csv(paths["tables"] / "vif_removal_history.csv", index=False)

        univariate_rows = []
        for group in groups:
            group_df = df.loc[df[GROUP_COLUMN] == group]
            for feature in supported_continuous:
                pair = group_df[[feature, TARGET]].dropna()
                if len(pair) >= 3 and pair[feature].nunique() > 1:
                    effect, p_value = stats.spearmanr(pair[feature], pair[TARGET])
                else:
                    effect = p_value = np.nan
                univariate_rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "effect_type": "spearman_rho",
                        "effect": effect,
                        "p_value": p_value,
                        "n": len(pair),
                    }
                )
            for feature in supported_binary:
                subset = group_df[[feature, TARGET]].dropna()
                zero = subset.loc[subset[feature] == 0, TARGET]
                one = subset.loc[subset[feature] == 1, TARGET]
                if len(zero) >= 2 and len(one) >= 2:
                    _, p_value = stats.ttest_ind(one, zero, equal_var=False)
                    effect = float(one.mean() - zero.mean())
                else:
                    effect = p_value = np.nan
                univariate_rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "effect_type": "mean_difference_1_minus_0",
                        "effect": effect,
                        "p_value": p_value,
                        "n": len(subset),
                    }
                )
        univariate = pd.DataFrame(univariate_rows)
        adjusted, rejected = safe_fdr(univariate["p_value"], alpha=args.alpha)
        univariate["fdr_p_value"] = adjusted
        univariate["significant_fdr"] = rejected
        univariate.to_csv(paths["tables"] / "groupwise_univariate_effects.csv", index=False)

        group_coefficients = group_specific_models(df, groups, final_predictors, args.alpha)
        group_coefficients.to_csv(paths["tables"] / "groupwise_adjusted_coefficients.csv", index=False)

        pooled_columns = [TARGET, GROUP_COLUMN, *final_predictors]
        if IDENTITY_COLUMN in df.columns:
            pooled_columns.append(IDENTITY_COLUMN)
        pooled = df[pooled_columns].dropna().copy()
        interaction_design, interaction_columns, reference_group = build_interaction_design(
            pooled, final_predictors, groups
        )
        clusters = pooled[IDENTITY_COLUMN] if IDENTITY_COLUMN in pooled.columns else None
        pooled_ols, pooled_robust, _, pooled_covariance = robust_fit(
            interaction_design, pooled[TARGET], clusters
        )
        pooled_coefficients = robust_coefficient_table(
            pooled_ols, pooled_robust, pooled_covariance
        )
        pooled_coefficients.to_csv(paths["tables"] / "pooled_interaction_coefficients.csv", index=False)
        interaction_tests = joint_interaction_tests(
            pooled_robust,
            list(pooled_ols.model.exog_names),
            interaction_columns,
            args.alpha,
        )
        interaction_tests.to_csv(paths["tables"] / "global_interaction_tests.csv", index=False)

        hair_support_rows = []
        male_groups = [group for group in groups if "Man" in group]
        for group in male_groups:
            group_df = df.loc[df[GROUP_COLUMN] == group]
            for feature in facial_hair:
                values = group_df[feature].dropna()
                n_zero = int((values == 0).sum())
                n_nonzero = int((values > 0).sum())
                hair_support_rows.append(
                    {
                        "group": group,
                        "feature": feature,
                        "n_zero": n_zero,
                        "n_nonzero": n_nonzero,
                        "n_unique": int(values.nunique()),
                        "sufficient_support": (
                            n_zero >= args.min_state_count
                            and n_nonzero >= args.min_nonzero_hair
                            and values.nunique() >= 3
                        ),
                    }
                )
        hair_support = pd.DataFrame(hair_support_rows)
        hair_support.to_csv(paths["tables"] / "facial_hair_support.csv", index=False)

        supported_hair = []
        if not hair_support.empty:
            supported_hair = [
                feature
                for feature in facial_hair
                if hair_support.loc[hair_support["feature"] == feature, "sufficient_support"].all()
            ]

        hair_group_coefficients = pd.DataFrame()
        hair_interaction_tests = pd.DataFrame()
        if supported_hair and len(male_groups) >= 2:
            male_df = df.loc[df[GROUP_COLUMN].isin(male_groups)].copy()
            for feature in supported_hair:
                mean = male_df[feature].mean()
                std = male_df[feature].std(ddof=0)
                if np.isfinite(std) and std > 0:
                    male_df[f"z_{feature}"] = (male_df[feature] - mean) / std
            hair_predictors = [f"z_{feature}" for feature in supported_hair if f"z_{feature}" in male_df]
            hair_group_coefficients = group_specific_models(
                male_df, male_groups, hair_predictors, args.alpha
            )
            hair_group_coefficients.to_csv(
                paths["tables"] / "facial_hair_group_coefficients.csv", index=False
            )
            hair_columns = [TARGET, GROUP_COLUMN, *hair_predictors]
            if IDENTITY_COLUMN in male_df.columns:
                hair_columns.append(IDENTITY_COLUMN)
            male_complete = male_df[hair_columns].dropna().copy()
            hair_design, hair_interaction_columns, _ = build_interaction_design(
                male_complete, hair_predictors, male_groups
            )
            hair_clusters = male_complete[IDENTITY_COLUMN] if IDENTITY_COLUMN in male_complete.columns else None
            hair_ols, hair_robust, _, _ = robust_fit(
                hair_design, male_complete[TARGET], hair_clusters
            )
            hair_interaction_tests = joint_interaction_tests(
                hair_robust,
                list(hair_ols.model.exog_names),
                hair_interaction_columns,
                args.alpha,
            )
            hair_interaction_tests.to_csv(
                paths["tables"] / "facial_hair_interaction_tests.csv", index=False
            )
        else:
            pd.DataFrame().to_csv(paths["tables"] / "facial_hair_group_coefficients.csv", index=False)
            pd.DataFrame().to_csv(paths["tables"] / "facial_hair_interaction_tests.csv", index=False)

        coefficient_summary = pd.DataFrame()
        if not group_coefficients.empty:
            coefficient_summary = (
                group_coefficients.groupby("feature", as_index=False)
                .agg(
                    minimum_group_coefficient=("coefficient", "min"),
                    maximum_group_coefficient=("coefficient", "max"),
                    coefficient_standard_deviation=("coefficient", "std"),
                    groups_with_significant_fdr=("significant_fdr", "sum"),
                )
            )
            sign_changes = (
                group_coefficients.groupby("feature")["coefficient"]
                .apply(lambda values: bool((values < 0).any() and (values > 0).any()))
                .rename("descriptive_sign_change")
                .reset_index()
            )
            coefficient_summary = coefficient_summary.merge(sign_changes, on="feature", how="left")

        univariate_summary = (
            univariate.groupby("feature", as_index=False)
            .agg(
                minimum_univariate_effect=("effect", "min"),
                maximum_univariate_effect=("effect", "max"),
                groups_univariate_significant_fdr=("significant_fdr", "sum"),
            )
        )
        paper_ready = interaction_tests.merge(coefficient_summary, on="feature", how="left").merge(
            univariate_summary, on="feature", how="left"
        )
        paper_ready["heterogeneity_evidence"] = np.where(
            paper_ready["significant_fdr"],
            "Global feature-by-group interaction is significant after FDR correction.",
            "No significant global feature-by-group interaction after FDR correction.",
        )
        paper_ready.to_csv(paths["tables"] / "paper_ready_evidence.csv", index=False)

        univariate_matrix = univariate.pivot(index="feature", columns="group", values="effect").reindex(columns=groups)
        save_heatmap(
            univariate_matrix,
            paths["figures"] / "groupwise_univariate_effect_heatmap.png",
            "Group-wise Univariate Feature Effects",
            "Effect",
        )
        if not group_coefficients.empty:
            coefficient_matrix = group_coefficients.pivot(
                index="feature", columns="group", values="coefficient"
            ).reindex(columns=groups)
            save_heatmap(
                coefficient_matrix,
                paths["figures"] / "groupwise_adjusted_coefficient_heatmap.png",
                "Adjusted Feature Effects by Demographic Group",
                "Adjusted coefficient",
            )
        if not hair_group_coefficients.empty:
            hair_matrix = hair_group_coefficients.pivot(
                index="feature", columns="group", values="coefficient"
            ).reindex(columns=male_groups)
            save_heatmap(
                hair_matrix,
                paths["figures"] / "facial_hair_adjusted_coefficient_heatmap.png",
                "Male-group Facial-hair Effects",
                "Adjusted coefficient",
            )

        interaction_plot = interaction_tests.dropna(subset=["fdr_p_value"]).copy()
        if not interaction_plot.empty:
            interaction_plot["minus_log10_fdr"] = -np.log10(
                interaction_plot["fdr_p_value"].clip(lower=np.finfo(float).tiny)
            )
            interaction_plot = interaction_plot.sort_values("minus_log10_fdr")
            plt.figure(figsize=(9, max(5, 0.4 * len(interaction_plot))))
            plt.barh(interaction_plot["feature"], interaction_plot["minus_log10_fdr"])
            plt.axvline(-np.log10(args.alpha), linestyle="--", label=f"FDR {args.alpha}")
            plt.xlabel("−log10(FDR-adjusted interaction p-value)")
            plt.title("Global Feature × Demographic-group Interaction Tests")
            plt.legend()
            plt.tight_layout()
            plt.savefig(paths["figures"] / "global_interaction_tests.png", dpi=300, bbox_inches="tight")
            plt.close()

        metadata = {
            "data_file": str(data_file),
            "groups": groups,
            "reference_group": reference_group,
            "alpha": args.alpha,
            "minimum_state_count": args.min_state_count,
            "minimum_continuous_count": args.min_continuous_count,
            "minimum_nonzero_hair": args.min_nonzero_hair,
            "maximum_vif": args.max_vif,
            "supported_continuous_features": supported_continuous,
            "supported_binary_features": supported_binary,
            "final_common_predictors": final_predictors,
            "supported_facial_hair_features": supported_hair,
            "significant_global_interactions_fdr": interaction_tests.loc[
                interaction_tests["significant_fdr"], "feature"
            ].tolist(),
        }
        save_json(metadata, paths["root"] / "run_metadata.json")

        print("Demographic consistency analysis completed successfully.")
        print(f"Groups: {groups}")
        print(f"Final common predictors: {final_predictors}")
        print(
            "Significant global interactions after FDR:",
            metadata["significant_global_interactions_fdr"],
        )
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
        print(f"Demographic consistency analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
