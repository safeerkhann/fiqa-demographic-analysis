#!/usr/bin/env python3
"""Explain the final tuned FIQA model with permutation importance and SHAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from pipeline_common import (
    GROUP_COLUMN,
    IDENTITY_COLUMN,
    IMAGE_COLUMN,
    TARGET,
    find_merged_data,
    load_runtime_config,
    load_split_assignments,
    original_feature_from_transformed,
    prepare_numeric_columns,
    print_saved_files,
    reconstruct_split,
    regression_metrics,
    save_json,
    stage_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce final test metrics and compute permutation and SHAP "
            "importance for the tuned FIQA model."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="Optional model override.")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--max-shap-rows", type=int, default=1500)
    parser.add_argument("--shap-background-rows", type=int, default=200)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--dependence-plots", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError(
            "The selected model must be a scikit-learn Pipeline with "
            "'preprocessor' and 'model' steps."
        )
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    transformed_frame = pd.DataFrame(transformed, index=X.index, columns=names)
    return transformed_frame, estimator


def compute_shap_explanation(estimator, background: pd.DataFrame, sample: pd.DataFrame):
    import shap

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(sample)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, background)
        explanation = explainer(sample)
    else:
        explainer = shap.Explainer(estimator.predict, background)
        explanation = explainer(sample)

    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional SHAP matrix for regression, received {values.shape}."
        )

    base_values = np.asarray(explanation.base_values)
    return shap, explanation, values, base_values


def main() -> int:
    args = parse_args()

    try:
        if args.permutation_repeats <= 0:
            raise ValueError("--permutation-repeats must be greater than zero.")
        if args.max_shap_rows <= 0 or args.shap_background_rows <= 0:
            raise ValueError("SHAP row limits must be greater than zero.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "07_model_explainability")
        training_paths = stage_paths(config, "03_ml_models", include_models=True)
        optimization_paths = stage_paths(config, "04_model_optimization", include_models=True)
        rq2_paths = stage_paths(config, "05_rq2_correlation_analysis")
        rq3_paths = stage_paths(config, "06_rq3_regression_analysis")

        training_metadata_file = training_paths["root"] / "run_metadata.json"
        split_file = training_paths["tables"] / "identity_aware_split.csv"
        optimization_summary_file = optimization_paths["root"] / "model_optimization_summary.json"
        default_model_file = optimization_paths["models"] / "best_tuned_model.joblib"
        model_file = args.model.expanduser().resolve() if args.model else default_model_file

        for path, label in [
            (training_metadata_file, "training metadata"),
            (split_file, "identity-aware split"),
            (model_file, "final model"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Required {label} is missing: {path}")

        model_metadata = json.loads(training_metadata_file.read_text(encoding="utf-8"))
        optimization_summary = (
            json.loads(optimization_summary_file.read_text(encoding="utf-8"))
            if optimization_summary_file.exists()
            else {}
        )
        feature_columns = list(model_metadata["feature_columns"])
        continuous_features = list(model_metadata["continuous_features"])
        binary_features = list(model_metadata["binary_features"])

        df = pd.read_csv(data_file)
        df = prepare_numeric_columns(
            df,
            numeric_features=continuous_features,
            binary_features=binary_features,
            target=TARGET,
        )
        split = load_split_assignments(split_file)
        _, test_df = reconstruct_split(df, split, feature_columns)
        X_test = test_df[feature_columns].copy()
        y_test = test_df[TARGET].astype(float)

        final_model = joblib.load(model_file)
        if not isinstance(final_model, Pipeline):
            raise TypeError("The final model must be a fitted scikit-learn Pipeline.")

        prediction = final_model.predict(X_test)
        metrics = regression_metrics(y_test, prediction)
        prediction_table = test_df[
            [IMAGE_COLUMN, IDENTITY_COLUMN, GROUP_COLUMN, TARGET, "source_dataframe_index"]
        ].copy()
        prediction_table["prediction"] = prediction
        prediction_table["residual"] = y_test.to_numpy() - prediction
        prediction_table["absolute_error"] = prediction_table["residual"].abs()
        prediction_table.to_csv(paths["tables"] / "test_predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv(paths["tables"] / "test_metrics.csv", index=False)

        permutation = permutation_importance(
            final_model,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=args.permutation_repeats,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        permutation_table = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Permutation_Importance_Mean": permutation.importances_mean,
                "Permutation_Importance_Std": permutation.importances_std,
            }
        )
        permutation_table["Permutation_Rank"] = (
            permutation_table["Permutation_Importance_Mean"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        permutation_table = permutation_table.sort_values(
            "Permutation_Importance_Mean", ascending=False
        ).reset_index(drop=True)
        permutation_table.to_csv(
            paths["tables"] / "permutation_importance.csv", index=False
        )

        top_permutation = permutation_table.head(args.top_features).sort_values(
            "Permutation_Importance_Mean"
        )
        plt.figure(figsize=(9, max(5, 0.4 * len(top_permutation))))
        plt.barh(
            top_permutation["Feature"],
            top_permutation["Permutation_Importance_Mean"],
            xerr=top_permutation["Permutation_Importance_Std"],
        )
        plt.xlabel("Increase in RMSE after permutation")
        plt.title("Repeated Permutation Importance")
        save_figure(paths["figures"] / "permutation_importance.png")

        transformed_shap = pd.DataFrame()
        original_shap = pd.DataFrame()
        shap_rows_used = 0
        shap_error: str | None = None

        if not args.skip_shap:
            try:
                import shap  # noqa: F401

                transformed_test, estimator = transform_features(final_model, X_test)
                shap_sample = transformed_test.sample(
                    n=min(args.max_shap_rows, len(transformed_test)),
                    random_state=args.random_state,
                )
                background = transformed_test.sample(
                    n=min(args.shap_background_rows, len(transformed_test)),
                    random_state=args.random_state,
                )
                shap_rows_used = len(shap_sample)
                shap_module, explanation, shap_values, base_values = compute_shap_explanation(
                    estimator, background, shap_sample
                )

                transformed_shap = pd.DataFrame(
                    {
                        "Transformed_Feature": shap_sample.columns,
                        "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0),
                        "Mean_Signed_SHAP": shap_values.mean(axis=0),
                    }
                )
                transformed_shap["Original_Feature"] = transformed_shap[
                    "Transformed_Feature"
                ].map(
                    lambda name: original_feature_from_transformed(name, feature_columns)
                )
                transformed_shap["Transformed_SHAP_Rank"] = (
                    transformed_shap["Mean_Absolute_SHAP"]
                    .rank(method="min", ascending=False)
                    .astype(int)
                )
                transformed_shap = transformed_shap.sort_values(
                    "Mean_Absolute_SHAP", ascending=False
                ).reset_index(drop=True)

                original_shap = (
                    transformed_shap.groupby("Original_Feature", as_index=False)
                    .agg(
                        Mean_Absolute_SHAP=("Mean_Absolute_SHAP", "sum"),
                        Mean_Signed_SHAP=("Mean_Signed_SHAP", "sum"),
                        Transformed_Columns=("Transformed_Feature", "count"),
                    )
                    .rename(columns={"Original_Feature": "Feature"})
                )
                original_shap["SHAP_Rank"] = (
                    original_shap["Mean_Absolute_SHAP"]
                    .rank(method="min", ascending=False)
                    .astype(int)
                )
                original_shap = original_shap.sort_values(
                    "Mean_Absolute_SHAP", ascending=False
                ).reset_index(drop=True)

                transformed_shap.to_csv(
                    paths["tables"] / "transformed_shap_importance.csv", index=False
                )
                original_shap.to_csv(
                    paths["tables"] / "original_feature_shap_importance.csv", index=False
                )

                shap_module.summary_plot(
                    shap_values,
                    shap_sample,
                    show=False,
                    max_display=args.top_features,
                )
                save_figure(paths["figures"] / "shap_beeswarm.png")

                shap_module.summary_plot(
                    shap_values,
                    shap_sample,
                    plot_type="bar",
                    show=False,
                    max_display=args.top_features,
                )
                save_figure(paths["figures"] / "shap_global_bar.png")

                top_transformed = transformed_shap["Transformed_Feature"].head(
                    args.dependence_plots
                )
                for feature in top_transformed:
                    shap_module.dependence_plot(
                        feature,
                        shap_values,
                        shap_sample,
                        show=False,
                        interaction_index=None,
                    )
                    safe_name = feature.replace("/", "_").replace("__", "_")
                    save_figure(paths["figures"] / f"shap_dependence_{safe_name}.png")

                sample_predictions = estimator.predict(shap_sample)
                sample_indices = shap_sample.index.to_numpy()
                test_lookup = prediction_table.set_index(X_test.index)
                actual_for_sample = test_lookup.loc[sample_indices, TARGET].to_numpy()
                errors = np.abs(actual_for_sample - sample_predictions)
                local_cases = {
                    "well_predicted": int(np.argmin(errors)),
                    "poorly_predicted": int(np.argmax(errors)),
                }
                for label, position in local_cases.items():
                    if base_values.ndim == 0:
                        base = float(base_values)
                    elif base_values.size == 1:
                        base = float(base_values.reshape(-1)[0])
                    else:
                        base = float(base_values.reshape(-1)[position])
                    local_explanation = shap_module.Explanation(
                        values=shap_values[position],
                        base_values=base,
                        data=shap_sample.iloc[position].to_numpy(),
                        feature_names=shap_sample.columns.tolist(),
                    )
                    shap_module.plots.waterfall(
                        local_explanation,
                        max_display=args.top_features,
                        show=False,
                    )
                    save_figure(paths["figures"] / f"shap_local_{label}.png")

            except (ImportError, RuntimeError, TypeError, ValueError) as error:
                shap_error = str(error)
                print(
                    f"Warning: SHAP analysis was skipped because it failed: {error}",
                    file=sys.stderr,
                )

        focused = permutation_table.copy()
        if not original_shap.empty:
            focused = focused.merge(original_shap, on="Feature", how="outer")

        rq2_file = rq2_paths["tables"] / "pearson_spearman_comparison.csv"
        if rq2_file.exists():
            rq2 = pd.read_csv(rq2_file)
            rq2_columns = [
                column
                for column in [
                    "Feature",
                    "Pearson_R",
                    "Pearson_FDR_P",
                    "Pearson_Significant_FDR",
                    "Spearman_Rho",
                    "Spearman_FDR_P",
                    "Spearman_Significant_FDR",
                ]
                if column in rq2.columns
            ]
            focused = focused.merge(rq2[rq2_columns], on="Feature", how="outer")

        rq3_file = rq3_paths["tables"] / "hc3_robust_coefficients.csv"
        if rq3_file.exists():
            rq3 = pd.read_csv(rq3_file)
            rq3 = rq3.loc[rq3["Feature"] != "const", [
                "Feature",
                "Coefficient",
                "Adjusted_P_Value_FDR",
                "Significant_FDR",
            ]].rename(
                columns={
                    "Coefficient": "RQ3_HC3_Coefficient",
                    "Adjusted_P_Value_FDR": "RQ3_FDR_P",
                    "Significant_FDR": "RQ3_Significant_FDR",
                }
            )
            focused = focused.merge(rq3, on="Feature", how="outer")

        if "SHAP_Rank" in focused.columns:
            focused["SHAP_Rank"] = focused["SHAP_Rank"].astype("Int64")
        focused["Permutation_Rank"] = focused["Permutation_Rank"].astype("Int64")
        focused.to_csv(
            paths["tables"] / "focused_explainability_summary.csv", index=False
        )

        rank_correlation = np.nan
        if not original_shap.empty:
            rank_data = permutation_table[["Feature", "Permutation_Rank"]].merge(
                original_shap[["Feature", "SHAP_Rank"]],
                on="Feature",
                how="inner",
            )
            if len(rank_data) >= 2:
                rank_correlation = stats.spearmanr(
                    rank_data["Permutation_Rank"], rank_data["SHAP_Rank"]
                ).statistic

        metadata = {
            "random_state": args.random_state,
            "permutation_repeats": args.permutation_repeats,
            "maximum_shap_rows": args.max_shap_rows,
            "shap_rows_used": shap_rows_used,
            "shap_error": shap_error,
            "final_model_name": optimization_summary.get("final_model"),
            "model_file": str(model_file),
            "test_rows": len(X_test),
            "feature_columns": feature_columns,
            "test_metrics": metrics,
            "shap_permutation_rank_spearman": (
                None if not np.isfinite(rank_correlation) else float(rank_correlation)
            ),
        }
        save_json(metadata, paths["root"] / "run_metadata.json")

        print("Model explainability completed successfully.")
        print(f"Test metrics: {metrics}")
        if shap_error:
            print(f"SHAP warning: {shap_error}")
        print_saved_files(paths["root"])
        return 0

    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"Explainability failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
