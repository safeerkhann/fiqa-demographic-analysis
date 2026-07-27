#!/usr/bin/env python3
"""Train identity-aware baseline regression models for CR-FIQA prediction."""

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

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from pipeline_common import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    GROUP_COLUMN,
    IDENTITY_COLUMN,
    IMAGE_COLUMN,
    TARGET,
    available_features,
    find_merged_data,
    load_runtime_config,
    make_preprocessor,
    prepare_numeric_columns,
    print_saved_files,
    regression_metrics,
    save_json,
    stage_paths,
    transformed_feature_names,
    validate_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare identity-aware regression baselines for "
            "predicting CR-FIQA scores."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Do not train XGBoost even when the package is installed.",
    )
    return parser.parse_args()


def build_models(random_state: int, n_jobs: int, skip_xgboost: bool) -> dict[str, object]:
    models: dict[str, object] = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=20000, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }

    if not skip_xgboost:
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except ImportError:
            print(
                "Warning: xgboost is not installed; the XGBoost baseline is skipped.",
                file=sys.stderr,
            )

    return models


def safe_filename(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def extract_interpretation_table(model_name: str, fitted_model: object) -> pd.DataFrame:
    if not isinstance(fitted_model, Pipeline):
        return pd.DataFrame()

    estimator = fitted_model.named_steps["model"]
    names = transformed_feature_names(fitted_model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if len(values) != len(names):
            return pd.DataFrame()
        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance": values,
                    "model": model_name,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        if len(values) != len(names):
            return pd.DataFrame()
        table = pd.DataFrame(
            {
                "feature": names,
                "coefficient": values,
                "absolute_coefficient": np.abs(values),
                "model": model_name,
            }
        )
        return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)

    return pd.DataFrame()


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    try:
        if not 0 < args.test_size < 1:
            raise ValueError("--test-size must be between 0 and 1.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "03_ml_models", include_models=True)

        df = pd.read_csv(data_file)
        validate_columns(
            df,
            [TARGET, IDENTITY_COLUMN, IMAGE_COLUMN, GROUP_COLUMN],
            context="Merged dataset",
        )

        continuous_features = available_features(df, CONTINUOUS_FEATURES)
        binary_features = available_features(df, BINARY_FEATURES)
        categorical_features = available_features(df, CATEGORICAL_FEATURES)
        feature_columns = [*continuous_features, *binary_features, *categorical_features]

        if not feature_columns:
            raise ValueError("No supported model predictors were found in the merged dataset.")

        df = prepare_numeric_columns(
            df,
            numeric_features=continuous_features,
            binary_features=binary_features,
            target=TARGET,
        )
        df["source_dataframe_index"] = df.index.astype(int)

        model_df = df.dropna(subset=[TARGET, IDENTITY_COLUMN, IMAGE_COLUMN]).copy()
        if model_df.empty:
            raise ValueError("No rows remain after removing missing target/identity/image values.")
        if model_df[IDENTITY_COLUMN].nunique() < 2:
            raise ValueError("At least two identities are required for an identity-aware split.")

        X = model_df[feature_columns].copy()
        y = model_df[TARGET].astype(float)
        groups = model_df[IDENTITY_COLUMN]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_positions, test_positions = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_positions].copy()
        X_test = X.iloc[test_positions].copy()
        y_train = y.iloc[train_positions].copy()
        y_test = y.iloc[test_positions].copy()
        groups_train = groups.iloc[train_positions]
        groups_test = groups.iloc[test_positions]

        overlap = set(groups_train.unique()).intersection(groups_test.unique())
        if overlap:
            raise RuntimeError(f"Identity leakage detected for {len(overlap)} identities.")

        preprocessor = make_preprocessor(
            continuous_features,
            binary_features,
            categorical_features,
        )
        estimators = build_models(args.random_state, args.n_jobs, args.skip_xgboost)

        fitted_models: dict[str, object] = {}
        result_rows: list[dict[str, object]] = []
        prediction_table = model_df.iloc[test_positions][
            [IMAGE_COLUMN, IDENTITY_COLUMN, GROUP_COLUMN, TARGET, "source_dataframe_index"]
        ].copy()

        for model_name, estimator in estimators.items():
            print(f"Training: {model_name}")
            start = time.perf_counter()

            if model_name == "Mean Baseline":
                fitted = estimator.fit(X_train, y_train)
            else:
                fitted = Pipeline(
                    [
                        ("preprocessor", make_preprocessor(
                            continuous_features,
                            binary_features,
                            categorical_features,
                        )),
                        ("model", estimator),
                    ]
                ).fit(X_train, y_train)

            runtime = time.perf_counter() - start
            train_prediction = fitted.predict(X_train)
            test_prediction = fitted.predict(X_test)
            train_metrics = regression_metrics(y_train, train_prediction)
            test_metrics = regression_metrics(y_test, test_prediction)

            fitted_models[model_name] = fitted
            prediction_column = f"prediction_{safe_filename(model_name)}"
            prediction_table[prediction_column] = test_prediction

            result_rows.append(
                {
                    "model": model_name,
                    "train_rmse": train_metrics["rmse"],
                    "test_rmse": test_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "test_mae": test_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "test_r2": test_metrics["r2"],
                    "runtime_seconds": runtime,
                }
            )

        model_comparison = pd.DataFrame(result_rows).sort_values("test_rmse").reset_index(drop=True)
        best_model_name = str(model_comparison.iloc[0]["model"])
        best_model = fitted_models[best_model_name]
        best_prediction_column = f"prediction_{safe_filename(best_model_name)}"
        prediction_table["best_model"] = best_model_name
        prediction_table["prediction"] = prediction_table[best_prediction_column]
        prediction_table["residual"] = prediction_table[TARGET] - prediction_table["prediction"]
        prediction_table["absolute_error"] = prediction_table["residual"].abs()

        split_table = model_df[
            [IMAGE_COLUMN, IDENTITY_COLUMN, GROUP_COLUMN, "source_dataframe_index"]
        ].copy()
        split_table["split"] = "train"
        split_table.iloc[test_positions, split_table.columns.get_loc("split")] = "test"

        model_comparison.to_csv(paths["tables"] / "model_comparison.csv", index=False)
        prediction_table.to_csv(paths["tables"] / "model_predictions.csv", index=False)
        split_table.to_csv(paths["tables"] / "identity_aware_split.csv", index=False)

        for model_name, fitted in fitted_models.items():
            filename = f"{safe_filename(model_name)}.joblib"
            joblib.dump(fitted, paths["models"] / filename)
            table = extract_interpretation_table(model_name, fitted)
            if not table.empty:
                suffix = "feature_importance" if "importance" in table.columns else "coefficients"
                table.to_csv(
                    paths["tables"] / f"{safe_filename(model_name)}_{suffix}.csv",
                    index=False,
                )

        joblib.dump(best_model, paths["models"] / "best_model.joblib")

        plt.figure(figsize=(10, 5))
        ordered = model_comparison.sort_values("test_rmse", ascending=False)
        plt.barh(ordered["model"], ordered["test_rmse"])
        plt.xlabel("Test RMSE")
        plt.title("Baseline Model Comparison")
        save_figure(paths["figures"] / "model_comparison_rmse.png")

        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, prediction_table["prediction"], alpha=0.45, s=18)
        lower = min(float(y_test.min()), float(prediction_table["prediction"].min()))
        upper = max(float(y_test.max()), float(prediction_table["prediction"].max()))
        plt.plot([lower, upper], [lower, upper], linestyle="--")
        plt.xlabel("Actual CR-FIQA score")
        plt.ylabel("Predicted CR-FIQA score")
        plt.title(f"Actual vs Predicted — {best_model_name}")
        save_figure(paths["figures"] / "best_model_actual_vs_predicted.png")

        plt.figure(figsize=(8, 5))
        plt.scatter(prediction_table["prediction"], prediction_table["residual"], alpha=0.45, s=18)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted CR-FIQA score")
        plt.ylabel("Residual")
        plt.title(f"Residuals — {best_model_name}")
        save_figure(paths["figures"] / "best_model_residuals.png")

        best_row = model_comparison.iloc[0]
        metadata = {
            "random_state": args.random_state,
            "test_size": args.test_size,
            "target": TARGET,
            "identity_column": IDENTITY_COLUMN,
            "image_column": IMAGE_COLUMN,
            "group_column": GROUP_COLUMN,
            "continuous_features": continuous_features,
            "binary_features": binary_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "best_model": best_model_name,
            "best_test_rmse": float(best_row["test_rmse"]),
            "best_test_mae": float(best_row["test_mae"]),
            "best_test_r2": float(best_row["test_r2"]),
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "training_identities": int(groups_train.nunique()),
            "test_identities": int(groups_test.nunique()),
            "data_file": str(data_file),
        }
        save_json(metadata, paths["root"] / "run_metadata.json")

        print("Model training completed successfully.")
        print(model_comparison.to_string(index=False))
        print(f"Best model: {best_model_name}")
        print_saved_files(paths["root"])
        print("Saved model files:")
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
        print(f"Model training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
