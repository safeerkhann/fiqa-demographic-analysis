#!/usr/bin/env python3
"""Tune non-linear FIQA regressors using identity-aware cross-validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from pipeline_common import (
    GROUP_COLUMN,
    IDENTITY_COLUMN,
    IMAGE_COLUMN,
    TARGET,
    find_merged_data,
    load_runtime_config,
    load_split_assignments,
    make_preprocessor,
    prepare_numeric_columns,
    print_saved_files,
    reconstruct_split,
    regression_metrics,
    save_json,
    stage_paths,
    transformed_feature_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize Random Forest, Gradient Boosting, and XGBoost with "
            "identity-aware cross-validation while preserving the test split."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="Quick uses 12 samples/model; full uses 60 samples/model.",
    )
    parser.add_argument("--n-iter", type=int, default=None, help="Override iterations per model.")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("random_forest", "gradient_boosting", "xgboost"),
        default=("random_forest", "gradient_boosting", "xgboost"),
    )
    return parser.parse_args()


def model_candidates(random_state: int, n_jobs: int) -> dict[str, tuple[object, dict]]:
    candidates: dict[str, tuple[object, dict]] = {
        "random_forest": (
            RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
            {
                "model__n_estimators": randint(200, 1001),
                "model__max_depth": [None, 5, 10, 15, 20, 30, 40],
                "model__min_samples_split": randint(2, 21),
                "model__min_samples_leaf": randint(1, 11),
                "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
                "model__bootstrap": [True, False],
            },
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": randint(100, 701),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(2, 6),
                "model__min_samples_split": randint(2, 21),
                "model__min_samples_leaf": randint(1, 11),
                "model__subsample": uniform(0.6, 0.4),
                "model__max_features": [None, "sqrt", "log2"],
            },
        ),
    }

    try:
        from xgboost import XGBRegressor

        candidates["xgboost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
            {
                "model__n_estimators": randint(200, 1001),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(2, 9),
                "model__min_child_weight": randint(1, 11),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_alpha": loguniform(1e-6, 1.0),
                "model__reg_lambda": loguniform(1e-3, 10.0),
            },
        )
    except ImportError:
        pass

    return candidates


def display_name(key: str) -> str:
    return {
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
    }[key]


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def importance_table(model_name: str, pipeline: Pipeline) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()
    names = transformed_feature_names(pipeline)
    values = np.asarray(estimator.feature_importances_, dtype=float)
    if len(names) != len(values):
        raise RuntimeError(
            f"Feature-name count ({len(names)}) does not match model importance count ({len(values)})."
        )
    return (
        pd.DataFrame({"feature": names, "importance": values, "model": model_name})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()

    try:
        if args.cv_splits < 2:
            raise ValueError("--cv-splits must be at least 2.")
        n_iter = args.n_iter if args.n_iter is not None else (12 if args.mode == "quick" else 60)
        if n_iter <= 0:
            raise ValueError("--n-iter must be greater than zero.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "04_model_optimization", include_models=True)
        baseline_paths = stage_paths(config, "03_ml_models", include_models=True)

        metadata_file = baseline_paths["root"] / "run_metadata.json"
        split_file = baseline_paths["tables"] / "identity_aware_split.csv"
        comparison_file = baseline_paths["tables"] / "model_comparison.csv"

        for path, label in [
            (metadata_file, "training metadata"),
            (split_file, "identity-aware split"),
            (comparison_file, "baseline comparison"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Required {label} is missing: {path}. Run train_models.py first.")

        training_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        feature_columns = list(training_metadata["feature_columns"])
        continuous_features = list(training_metadata["continuous_features"])
        binary_features = list(training_metadata["binary_features"])
        categorical_features = list(training_metadata["categorical_features"])

        df = pd.read_csv(data_file)
        df = prepare_numeric_columns(
            df,
            numeric_features=continuous_features,
            binary_features=binary_features,
            target=TARGET,
        )
        split = load_split_assignments(split_file)
        train_df, test_df = reconstruct_split(df, split, feature_columns)

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET].astype(float)
        groups_train = train_df[IDENTITY_COLUMN]
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET].astype(float)

        unique_training_identities = int(groups_train.nunique())
        if unique_training_identities < args.cv_splits:
            raise ValueError(
                f"Only {unique_training_identities} training identities are available, "
                f"but --cv-splits={args.cv_splits}."
            )

        candidates = model_candidates(args.random_state, args.n_jobs)
        requested = []
        for key in args.models:
            if key == "xgboost" and key not in candidates:
                print("Warning: xgboost is unavailable and will be skipped.", file=sys.stderr)
                continue
            requested.append(key)
        if not requested:
            raise ValueError("No requested optimization model is available.")

        cv = GroupKFold(n_splits=args.cv_splits)
        summary_rows: list[dict[str, object]] = []
        fitted: dict[str, Pipeline] = {}
        predictions = test_df[
            [IMAGE_COLUMN, IDENTITY_COLUMN, GROUP_COLUMN, TARGET, "source_dataframe_index"]
        ].copy()

        for key in requested:
            name = display_name(key)
            estimator, parameter_distributions = candidates[key]
            pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        make_preprocessor(
                            continuous_features,
                            binary_features,
                            categorical_features,
                        ),
                    ),
                    ("model", estimator),
                ]
            )
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameter_distributions,
                n_iter=n_iter,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                refit=True,
                return_train_score=True,
                verbose=1,
                error_score="raise",
            )

            print(f"Optimizing {name} ({n_iter} parameter samples)...")
            start = time.perf_counter()
            search.fit(X_train, y_train, groups=groups_train)
            runtime = time.perf_counter() - start

            best_pipeline = search.best_estimator_
            fitted[name] = best_pipeline
            train_pred = best_pipeline.predict(X_train)
            test_pred = best_pipeline.predict(X_test)
            train_metrics = regression_metrics(y_train, train_pred)
            test_metrics = regression_metrics(y_test, test_pred)
            cv_rmse = -float(search.best_score_)

            summary_rows.append(
                {
                    "model": name,
                    "cv_rmse": cv_rmse,
                    "train_rmse": train_metrics["rmse"],
                    "test_rmse": test_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "test_mae": test_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "test_r2": test_metrics["r2"],
                    "search_seconds": runtime,
                    "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                }
            )

            predictions[f"prediction_{key}"] = test_pred
            cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
            cv_results.to_csv(paths["tables"] / f"{key}_search_results.csv", index=False)
            joblib.dump(best_pipeline, paths["models"] / f"tuned_{key}.joblib")
            save_json(search.best_params_, paths["root"] / f"{key}_best_parameters.json")

        tuned_summary = pd.DataFrame(summary_rows).sort_values("cv_rmse").reset_index(drop=True)
        final_model_name = str(tuned_summary.iloc[0]["model"])
        final_key = next(key for key in requested if display_name(key) == final_model_name)
        final_model = fitted[final_model_name]
        predictions["final_model"] = final_model_name
        predictions["prediction"] = predictions[f"prediction_{final_key}"]
        predictions["residual"] = predictions[TARGET] - predictions["prediction"]
        predictions["absolute_error"] = predictions["residual"].abs()

        baseline = pd.read_csv(comparison_file)
        nonlinear_names = [display_name(key) for key in requested]
        baseline_subset = baseline.loc[baseline["model"].isin(nonlinear_names)].copy()
        baseline_vs_tuned = tuned_summary.merge(
            baseline_subset[["model", "test_rmse", "test_mae", "test_r2"]].rename(
                columns={
                    "test_rmse": "baseline_test_rmse",
                    "test_mae": "baseline_test_mae",
                    "test_r2": "baseline_test_r2",
                }
            ),
            on="model",
            how="left",
        )
        baseline_vs_tuned["rmse_improvement"] = (
            baseline_vs_tuned["baseline_test_rmse"] - baseline_vs_tuned["test_rmse"]
        )
        baseline_vs_tuned["mae_improvement"] = (
            baseline_vs_tuned["baseline_test_mae"] - baseline_vs_tuned["test_mae"]
        )
        baseline_vs_tuned["r2_improvement"] = (
            baseline_vs_tuned["test_r2"] - baseline_vs_tuned["baseline_test_r2"]
        )

        tuned_summary.to_csv(paths["tables"] / "tuned_model_comparison.csv", index=False)
        baseline_vs_tuned.to_csv(paths["tables"] / "baseline_vs_tuned.csv", index=False)
        predictions.to_csv(paths["tables"] / "tuned_model_predictions.csv", index=False)
        joblib.dump(final_model, paths["models"] / "best_tuned_model.joblib")

        importance = importance_table(final_model_name, final_model)
        if not importance.empty:
            importance.to_csv(
                paths["tables"] / "best_model_feature_importance.csv", index=False
            )
            top = importance.head(15).sort_values("importance")
            plt.figure(figsize=(9, 6))
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Feature importance")
            plt.title(f"Final Tuned Model Importance — {final_model_name}")
            save_figure(paths["figures"] / "best_model_feature_importance.png")

        ordered = tuned_summary.sort_values("test_rmse", ascending=False)
        plt.figure(figsize=(9, 5))
        plt.barh(ordered["model"], ordered["test_rmse"])
        plt.xlabel("Test RMSE")
        plt.title("Tuned Model Test Performance")
        save_figure(paths["figures"] / "tuned_model_test_rmse.png")

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, predictions["prediction"], alpha=0.45, s=18)
        lower = min(float(y_test.min()), float(predictions["prediction"].min()))
        upper = max(float(y_test.max()), float(predictions["prediction"].max()))
        plt.plot([lower, upper], [lower, upper], linestyle="--")
        plt.xlabel("Actual CR-FIQA score")
        plt.ylabel("Predicted CR-FIQA score")
        plt.title(f"Final Tuned Model — {final_model_name}")
        save_figure(paths["figures"] / "final_model_actual_vs_predicted.png")

        final_row = tuned_summary.iloc[0]
        optimization_summary = {
            "selection_rule": "lowest identity-aware cross-validation RMSE",
            "final_model": final_model_name,
            "final_model_key": final_key,
            "run_mode": args.mode,
            "search_iterations_per_model": n_iter,
            "cv_splits": args.cv_splits,
            "random_state": args.random_state,
            "feature_columns": feature_columns,
            "training_rows": len(train_df),
            "test_rows": len(test_df),
            "final_cv_rmse": float(final_row["cv_rmse"]),
            "final_test_rmse": float(final_row["test_rmse"]),
            "final_test_mae": float(final_row["test_mae"]),
            "final_test_r2": float(final_row["test_r2"]),
            "data_file": str(data_file),
        }
        save_json(optimization_summary, paths["root"] / "model_optimization_summary.json")

        print("Model optimization completed successfully.")
        print(tuned_summary.to_string(index=False))
        print(f"Final model selected by CV RMSE: {final_model_name}")
        print_saved_files(paths["root"])
        print("Saved tuned models:")
        print_saved_files(paths["models"])
        return 0

    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print(f"Model optimization failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
