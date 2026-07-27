#!/usr/bin/env python3
"""RQ3: multiple regression with HC3 inference and diagnostics."""

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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline_common import (
    BINARY_FEATURES,
    GROUP_COLUMN,
    IDENTITY_COLUMN,
    IMAGE_COLUMN,
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
            "Fit the RQ3 multiple linear regression, report HC3 robust "
            "coefficients, FDR correction, diagnostics, and sensitivity models."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/runtime_config.json"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--include-age-primary",
        action="store_true",
        help="Include age in the primary model. The project default excludes age.",
    )
    parser.add_argument(
        "--skip-age-sensitivity",
        action="store_true",
        help="Skip the separate sensitivity model including age.",
    )
    parser.add_argument(
        "--skip-influence-sensitivity",
        action="store_true",
        help="Skip refitting after excluding observations above Cook's 4/N threshold.",
    )
    return parser.parse_args()


def coefficient_table(
    ols_results,
    robust_results,
    alpha: float,
    covariance_label: str,
) -> pd.DataFrame:
    names = list(ols_results.model.exog_names)
    confidence = np.asarray(robust_results.conf_int(alpha=alpha), dtype=float)
    result = pd.DataFrame(
        {
            "Feature": names,
            "Coefficient": np.asarray(robust_results.params, dtype=float),
            "Standard_Error": np.asarray(robust_results.bse, dtype=float),
            "T_Value": np.asarray(robust_results.tvalues, dtype=float),
            "P_Value": np.asarray(robust_results.pvalues, dtype=float),
            "CI_Lower_95": confidence[:, 0],
            "CI_Upper_95": confidence[:, 1],
            "Covariance": covariance_label,
        }
    )
    result["Adjusted_P_Value_FDR"] = np.nan
    result["Significant_FDR"] = False
    mask = result["Feature"] != "const"
    adjusted, rejected = safe_fdr(result.loc[mask, "P_Value"], alpha=alpha)
    result.loc[mask, "Adjusted_P_Value_FDR"] = adjusted
    result.loc[mask, "Significant_FDR"] = rejected
    result["Absolute_Coefficient"] = result["Coefficient"].abs()
    return result.sort_values(
        ["Feature"], key=lambda s: s.ne("const") if s.name == "Feature" else s
    ).reset_index(drop=True)


def fit_hc3(X: pd.DataFrame, y: pd.Series, alpha: float, label: str):
    X_with_constant = sm.add_constant(X.astype(float), has_constant="add")
    ols = sm.OLS(y.astype(float), X_with_constant).fit()
    robust = ols.get_robustcov_results(cov_type="HC3")
    table = coefficient_table(ols, robust, alpha, label)
    return ols, robust, table, X_with_constant


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    values = X.astype(float)
    rows = []
    for position, feature in enumerate(values.columns):
        try:
            value = variance_inflation_factor(values.to_numpy(), position)
        except Exception:
            value = np.nan
        rows.append({"Feature": feature, "VIF": float(value) if np.isfinite(value) else value})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False, na_position="last").reset_index(drop=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    try:
        if not 0 < args.alpha < 1:
            raise ValueError("--alpha must be between 0 and 1.")

        config = load_runtime_config(args.config)
        data_file = find_merged_data(config, args.data)
        paths = stage_paths(config, "06_rq3_regression_analysis")
        rq2_paths = stage_paths(config, "05_rq2_correlation_analysis")
        rq2_file = rq2_paths["tables"] / "pearson_spearman_comparison.csv"
        if not rq2_file.exists():
            raise FileNotFoundError(
                f"RQ2 comparison table is missing: {rq2_file}. Run correlation_analysis.py first."
            )

        df = pd.read_csv(data_file)
        validate_columns(df, [TARGET], context="Merged dataset")

        continuous_features = available_features(df, PRIMARY_CONTINUOUS_FEATURES)
        binary_features = available_features(df, BINARY_FEATURES)
        protected_features = ["age"] if "age" in df.columns and df["age"].notna().any() else []
        primary_features = [*continuous_features, *binary_features]
        if args.include_age_primary and protected_features:
            primary_features.append("age")

        if not primary_features:
            raise ValueError("No supported RQ3 predictors were found.")

        df = prepare_numeric_columns(
            df,
            numeric_features=[*continuous_features, *protected_features],
            binary_features=binary_features,
            target=TARGET,
        )

        valid_primary = [
            feature
            for feature in primary_features
            if df[feature].nunique(dropna=True) > 1
        ]
        if not valid_primary:
            raise ValueError("All candidate predictors are constant or missing.")

        metadata_columns = [
            column
            for column in [IMAGE_COLUMN, IDENTITY_COLUMN, GROUP_COLUMN]
            if column in df.columns
        ]
        model_df = df[[TARGET, *valid_primary, *metadata_columns]].dropna(
            subset=[TARGET, *valid_primary]
        ).copy()
        if len(model_df) <= len(valid_primary) + 2:
            raise ValueError("The complete-case sample is too small for the requested regression.")

        continuous_in_model = [
            feature
            for feature in [*continuous_features, "age"]
            if feature in valid_primary
        ]
        X = model_df[valid_primary].copy()
        scaler = StandardScaler()
        if continuous_in_model:
            X[continuous_in_model] = scaler.fit_transform(X[continuous_in_model])
        y = model_df[TARGET].astype(float)

        ols, robust, hc3_table, X_design = fit_hc3(
            X, y, args.alpha, "HC3"
        )

        # Fully standardized model for comparable coefficient magnitudes.
        standardized_X = pd.DataFrame(
            StandardScaler().fit_transform(model_df[valid_primary]),
            columns=valid_primary,
            index=model_df.index,
        )
        standardized_y = pd.Series(
            StandardScaler().fit_transform(y.to_numpy().reshape(-1, 1)).reshape(-1),
            index=y.index,
            name=TARGET,
        )
        std_ols, std_robust, standardized_table, _ = fit_hc3(
            standardized_X,
            standardized_y,
            args.alpha,
            "HC3 fully standardized",
        )

        vif_results = calculate_vif(X)
        fitted = ols.fittedvalues
        residuals = ols.resid
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))
        model_fit_summary = pd.DataFrame(
            {
                "Metric": [
                    "N",
                    "R_squared",
                    "Adjusted_R_squared",
                    "F_statistic",
                    "F_test_p_value",
                    "AIC",
                    "BIC",
                    "RMSE",
                    "MAE",
                ],
                "Value": [
                    int(ols.nobs),
                    float(ols.rsquared),
                    float(ols.rsquared_adj),
                    float(ols.fvalue),
                    float(ols.f_pvalue),
                    float(ols.aic),
                    float(ols.bic),
                    rmse,
                    mae,
                ],
            }
        )

        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, X_design)
        diagnostic_tests = pd.DataFrame(
            {
                "Test": [
                    "Breusch-Pagan LM",
                    "Breusch-Pagan F",
                    "Jarque-Bera",
                ],
                "Statistic": [bp_lm, bp_f, stats.jarque_bera(residuals).statistic],
                "P_Value": [bp_lm_p, bp_f_p, stats.jarque_bera(residuals).pvalue],
            }
        )

        influence = ols.get_influence()
        cooks_distance = np.asarray(influence.cooks_distance[0], dtype=float)
        leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
        studentized = np.asarray(influence.resid_studentized_external, dtype=float)
        cooks_threshold = 4.0 / len(model_df)
        influence_table = pd.DataFrame(
            {
                "regression_row": np.arange(len(model_df)),
                "source_dataframe_index": model_df.index.to_numpy(),
                "fitted_value": np.asarray(fitted),
                "residual": np.asarray(residuals),
                "studentized_residual": studentized,
                "leverage": leverage,
                "cooks_distance": cooks_distance,
                "Above_Cooks_4_over_N": cooks_distance > cooks_threshold,
            }
        )
        for column in metadata_columns:
            influence_table[column] = model_df[column].to_numpy()
        influence_table = influence_table.sort_values(
            "cooks_distance", ascending=False
        ).reset_index(drop=True)

        rq2 = pd.read_csv(rq2_file)
        rq3_source = (
            hc3_table.loc[hc3_table["Feature"] != "const", [
                "Feature",
                "Coefficient",
                "P_Value",
                "Adjusted_P_Value_FDR",
                "Significant_FDR",
            ]]
            .rename(
                columns={
                    "Coefficient": "RQ3_HC3_Coefficient",
                    "P_Value": "RQ3_P_Value",
                    "Adjusted_P_Value_FDR": "RQ3_FDR_P",
                    "Significant_FDR": "RQ3_Significant_FDR",
                }
            )
        )
        rq2_rq3 = rq2.merge(rq3_source, on="Feature", how="inner", validate="one_to_one")
        rq2_rq3["Pearson_and_RQ3_Same_Direction"] = (
            np.sign(rq2_rq3["Pearson_R"]) == np.sign(rq2_rq3["RQ3_HC3_Coefficient"])
        )
        rq2_rq3["Significant_in_RQ2_and_RQ3"] = (
            rq2_rq3["Pearson_Significant_FDR"] & rq2_rq3["RQ3_Significant_FDR"]
        )

        age_table = pd.DataFrame()
        age_comparison = pd.DataFrame()
        if not args.skip_age_sensitivity and "age" in protected_features and "age" not in valid_primary:
            age_features = [*valid_primary, "age"]
            age_df = df[[TARGET, *age_features]].dropna().copy()
            age_X = age_df[age_features].copy()
            age_continuous = [feature for feature in [*continuous_features, "age"] if feature in age_features]
            if age_continuous:
                age_X[age_continuous] = StandardScaler().fit_transform(age_X[age_continuous])
            _, _, age_table, _ = fit_hc3(age_X, age_df[TARGET], args.alpha, "HC3 with age")
            age_comparison = (
                hc3_table.loc[~hc3_table["Feature"].isin(["const"]), [
                    "Feature", "Coefficient", "Adjusted_P_Value_FDR", "Significant_FDR"
                ]]
                .rename(
                    columns={
                        "Coefficient": "Primary_Coefficient",
                        "Adjusted_P_Value_FDR": "Primary_FDR_P",
                        "Significant_FDR": "Primary_Significant_FDR",
                    }
                )
                .merge(
                    age_table.loc[~age_table["Feature"].isin(["const", "age"]), [
                        "Feature", "Coefficient", "Adjusted_P_Value_FDR", "Significant_FDR"
                    ]].rename(
                        columns={
                            "Coefficient": "Age_Adjusted_Coefficient",
                            "Adjusted_P_Value_FDR": "Age_Adjusted_FDR_P",
                            "Significant_FDR": "Age_Adjusted_Significant_FDR",
                        }
                    ),
                    on="Feature",
                    how="inner",
                    validate="one_to_one",
                )
            )
            age_comparison["Same_Direction"] = (
                np.sign(age_comparison["Primary_Coefficient"])
                == np.sign(age_comparison["Age_Adjusted_Coefficient"])
            )
            age_comparison["Coefficient_Change"] = (
                age_comparison["Age_Adjusted_Coefficient"]
                - age_comparison["Primary_Coefficient"]
            )

        reduced_table = pd.DataFrame()
        influence_comparison = pd.DataFrame()
        above = cooks_distance > cooks_threshold
        if not args.skip_influence_sensitivity and above.any() and (~above).sum() > len(valid_primary) + 2:
            reduced_X = X.loc[~above].copy()
            reduced_y = y.loc[~above].copy()
            _, _, reduced_table, _ = fit_hc3(
                reduced_X, reduced_y, args.alpha, "HC3 excluding Cook's-distance cases"
            )
            influence_comparison = (
                hc3_table.loc[hc3_table["Feature"] != "const", [
                    "Feature", "Coefficient", "Adjusted_P_Value_FDR", "Significant_FDR"
                ]]
                .rename(
                    columns={
                        "Coefficient": "Full_Coefficient",
                        "Adjusted_P_Value_FDR": "Full_FDR_P",
                        "Significant_FDR": "Full_Significant_FDR",
                    }
                )
                .merge(
                    reduced_table.loc[reduced_table["Feature"] != "const", [
                        "Feature", "Coefficient", "Adjusted_P_Value_FDR", "Significant_FDR"
                    ]].rename(
                        columns={
                            "Coefficient": "Reduced_Coefficient",
                            "Adjusted_P_Value_FDR": "Reduced_FDR_P",
                            "Significant_FDR": "Reduced_Significant_FDR",
                        }
                    ),
                    on="Feature",
                    how="inner",
                    validate="one_to_one",
                )
            )
            influence_comparison["Coefficient_Change"] = (
                influence_comparison["Reduced_Coefficient"]
                - influence_comparison["Full_Coefficient"]
            )
            influence_comparison["Same_Direction"] = (
                np.sign(influence_comparison["Full_Coefficient"])
                == np.sign(influence_comparison["Reduced_Coefficient"])
            )

        rq3_summary = hc3_table.loc[hc3_table["Feature"] != "const"].copy()
        rq3_summary["Direction"] = np.where(
            rq3_summary["Coefficient"] > 0,
            "positive",
            np.where(rq3_summary["Coefficient"] < 0, "negative", "zero"),
        )
        rq3_summary["Interpretation"] = np.where(
            rq3_summary["Significant_FDR"],
            "Adjusted association remains significant after FDR correction.",
            "No statistically significant adjusted association after FDR correction.",
        )
        rq3_summary = rq3_summary.sort_values(
            "Absolute_Coefficient", ascending=False
        ).reset_index(drop=True)

        output_tables = {
            "hc3_robust_coefficients.csv": hc3_table,
            "standardized_coefficients.csv": standardized_table,
            "vif_results.csv": vif_results,
            "model_fit_summary.csv": model_fit_summary,
            "diagnostic_tests.csv": diagnostic_tests,
            "influence_diagnostics.csv": influence_table,
            "rq2_rq3_comparison.csv": rq2_rq3,
            "age_sensitivity_coefficients.csv": age_table,
            "age_sensitivity_comparison.csv": age_comparison,
            "influence_sensitivity_coefficients.csv": reduced_table,
            "influence_sensitivity_comparison.csv": influence_comparison,
            "rq3_summary.csv": rq3_summary,
        }
        for filename, table in output_tables.items():
            table.to_csv(paths["tables"] / filename, index=False)

        plot_coefficients = hc3_table.loc[hc3_table["Feature"] != "const"].sort_values("Coefficient")
        lower_error = plot_coefficients["Coefficient"] - plot_coefficients["CI_Lower_95"]
        upper_error = plot_coefficients["CI_Upper_95"] - plot_coefficients["Coefficient"]
        plt.figure(figsize=(10, max(6, len(plot_coefficients) * 0.42)))
        plt.errorbar(
            plot_coefficients["Coefficient"],
            np.arange(len(plot_coefficients)),
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o",
            capsize=3,
        )
        plt.axvline(0, linestyle="--")
        plt.yticks(np.arange(len(plot_coefficients)), plot_coefficients["Feature"])
        plt.xlabel("Coefficient with 95% HC3 confidence interval")
        plt.title("RQ3 Adjusted Regression Coefficients")
        save_figure(paths["figures"] / "hc3_coefficient_plot.png")

        standardized_plot = standardized_table.loc[
            standardized_table["Feature"] != "const"
        ].sort_values("Coefficient")
        plt.figure(figsize=(10, max(6, len(standardized_plot) * 0.42)))
        plt.barh(standardized_plot["Feature"], standardized_plot["Coefficient"])
        plt.axvline(0, linewidth=1)
        plt.xlabel("Fully standardized HC3 coefficient")
        plt.title("Fully Standardized Regression Coefficients")
        save_figure(paths["figures"] / "standardized_coefficients.png")

        plt.figure(figsize=(8, 5))
        plt.scatter(fitted, residuals, alpha=0.4, s=16)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Fitted value")
        plt.ylabel("Residual")
        plt.title("Residuals vs Fitted Values")
        save_figure(paths["figures"] / "residuals_vs_fitted.png")

        plt.figure(figsize=(6, 6))
        sm.qqplot(residuals, line="45", fit=True, ax=plt.gca())
        plt.title("Normal Q-Q Plot of Residuals")
        save_figure(paths["figures"] / "residual_qq_plot.png")

        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(len(cooks_distance)), cooks_distance, alpha=0.45, s=16)
        plt.axhline(cooks_threshold, linestyle="--", label="4/N threshold")
        plt.xlabel("Regression-sample observation")
        plt.ylabel("Cook's distance")
        plt.title("Cook's Distance")
        plt.legend()
        save_figure(paths["figures"] / "cooks_distance.png")

        metadata = {
            "alpha": args.alpha,
            "data_file": str(data_file),
            "primary_features": valid_primary,
            "continuous_standardized": continuous_in_model,
            "age_in_primary_model": args.include_age_primary,
            "observations": int(ols.nobs),
            "r_squared": float(ols.rsquared),
            "adjusted_r_squared": float(ols.rsquared_adj),
            "maximum_vif": float(vif_results["VIF"].replace([np.inf, -np.inf], np.nan).max()),
            "cooks_threshold": cooks_threshold,
            "observations_above_cooks_threshold": int(above.sum()),
            "significant_features_fdr": rq3_summary.loc[rq3_summary["Significant_FDR"], "Feature"].tolist(),
        }
        save_json(metadata, paths["root"] / "run_metadata.json")

        print("RQ3 regression analysis completed successfully.")
        print(f"Adjusted R²: {ols.rsquared_adj:.4f}")
        print(f"Significant features after FDR: {metadata['significant_features_fdr']}")
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
        print(f"Regression analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
