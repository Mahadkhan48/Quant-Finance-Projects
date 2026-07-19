# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:56:15 2026

@author: mahad
"""

# file: 07_Run_ML_GB_Volatility_Models.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


def check_required_columns(df: pd.DataFrame, required_cols: list[str], file_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_name}: {missing}")


def mse_rmse_mae(actual: pd.Series, forecast: pd.Series) -> tuple[float, float, float]:
    actual = pd.Series(actual, dtype="float64").reset_index(drop=True)
    forecast = pd.Series(forecast, dtype="float64").reset_index(drop=True)

    errors = actual - forecast
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    return mse, rmse, mae


def get_current_log_rv(df: pd.DataFrame) -> pd.Series:
    if "Log Realized Volatility" in df.columns:
        return pd.to_numeric(df["Log Realized Volatility"], errors="coerce")

    if "Annualized Realized Volatility (%)" in df.columns:
        ann_vol = pd.to_numeric(df["Annualized Realized Volatility (%)"], errors="coerce")
        ann_vol = ann_vol.where(ann_vol > 0)
        return np.log(ann_vol)

    if "Annualized Realized Volatility Decimal" in df.columns:
        ann_vol_pct = pd.to_numeric(df["Annualized Realized Volatility Decimal"], errors="coerce") * 100
        ann_vol_pct = ann_vol_pct.where(ann_vol_pct > 0)
        return np.log(ann_vol_pct)

    if "Monthly Realized Volatility Decimal" in df.columns:
        ann_vol_pct = pd.to_numeric(df["Monthly Realized Volatility Decimal"], errors="coerce") * np.sqrt(12) * 100
        ann_vol_pct = ann_vol_pct.where(ann_vol_pct > 0)
        return np.log(ann_vol_pct)

    raise ValueError(
        "Could not find or create current-month Log Realized Volatility. "
        "Please include one of these columns: "
        "'Log Realized Volatility', 'Annualized Realized Volatility (%)', "
        "'Annualized Realized Volatility Decimal', or 'Monthly Realized Volatility Decimal'."
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Log Realized Volatility"] = get_current_log_rv(df)
    df["Log RV Lag 1"] = df["Log Realized Volatility"].shift(1)
    df["Log RV Lag 2"] = df["Log Realized Volatility"].shift(2)
    df["Log RV MA 3"] = df["Log Realized Volatility"].rolling(window=3, min_periods=3).mean()
    df["Log RV MA 6"] = df["Log Realized Volatility"].rolling(window=6, min_periods=6).mean()

    df["Aggregate Climate Risk"] = pd.to_numeric(df["Aggregate Climate Risk"], errors="coerce")
    df["PCA Climate Risk Index"] = pd.to_numeric(df["PCA Climate Risk Index"], errors="coerce")
    df["Next Month Log Realized Volatility"] = pd.to_numeric(
        df["Next Month Log Realized Volatility"], errors="coerce"
    )

    return df


def get_feature_sets() -> dict[str, list[str]]:
    benchmark_features = [
        "Log Realized Volatility",
        "Log RV Lag 1",
        "Log RV Lag 2",
        "Log RV MA 3",
        "Log RV MA 6",
    ]

    return {
        "ML Benchmark": benchmark_features,
        "ML + Aggregate CRI": benchmark_features + ["Aggregate Climate Risk"],
        "ML + PCA CRI": benchmark_features + ["PCA Climate Risk Index"],
    }


def tune_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
) -> tuple[dict, float, int, int]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    n_splits = min(5, max(2, len(train_df) // 24))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    param_dist = {
        "n_estimators": [50, 100, 150, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.10],
        "max_depth": [1, 2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4, 6],
        "subsample": [0.6, 0.8, 1.0],
        "max_features": [None, "sqrt", "log2"],
    }

    base_model = GradientBoostingRegressor(random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_mean_squared_error",
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    return search.best_params_, float(search.best_score_), n_splits, 40


def recursive_oos_forecasts(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_params: dict,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oos_df = df[df["Sample Split"] == "Out-of-Sample"].copy()
    oos_df = oos_df.sort_values("Target Month").reset_index(drop=True)

    forecast_rows = []
    importance_rows = []

    for _, current_row in oos_df.iterrows():
        current_target_month = current_row["Target Month"]

        train_df = df[df["Target Month"] < current_target_month].copy()
        train_df = train_df.dropna(subset=feature_cols + [target_col]).copy()
        train_df = train_df.sort_values("Target Month").reset_index(drop=True)

        if len(train_df) < 30:
            continue

        if current_row[feature_cols].isna().any():
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        model = GradientBoostingRegressor(random_state=42, **best_params)
        model.fit(X_train, y_train)

        X_test = current_row[feature_cols].to_frame().T
        forecast = float(model.predict(X_test)[0])
        actual = float(current_row[target_col])

        row = {
            "Month": current_row["Month"],
            "Target Month": current_row["Target Month"],
            "Sample Split": current_row["Sample Split"],
            "Actual Next Month Log Realized Volatility": actual,
            f"{model_name} Forecast": forecast,
            f"{model_name} Forecast Error": actual - forecast,
            f"{model_name} Squared Error": (actual - forecast) ** 2,
            f"{model_name} Absolute Error": abs(actual - forecast),
            f"{model_name} Forecast Bias": forecast - actual,
        }

        for col in feature_cols:
            row[col] = current_row[col]

        forecast_rows.append(row)

        for feature_name, importance_value in zip(feature_cols, model.feature_importances_):
            importance_rows.append(
                {
                    "Target Month": current_target_month,
                    "Model name": model_name,
                    "Feature": feature_name,
                    "Importance": float(importance_value),
                }
            )

    forecasts_df = pd.DataFrame(forecast_rows)
    importance_df = pd.DataFrame(importance_rows)

    return forecasts_df, importance_df


def merge_forecast_tables(
    benchmark_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    feature_cols_all: list[str],
) -> pd.DataFrame:
    base_cols = ["Month", "Target Month", "Sample Split", "Actual Next Month Log Realized Volatility"]

    merged = benchmark_df.copy()

    for df_part in [agg_df, pca_df]:
        keep_cols = [col for col in df_part.columns if col not in base_cols or col == "Target Month"]
        merged = merged.merge(
            df_part[keep_cols],
            on="Target Month",
            how="outer",
            suffixes=("", "_dup"),
        )

        dup_cols = [col for col in merged.columns if col.endswith("_dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

    feature_cols_present = [col for col in feature_cols_all if col in merged.columns]
    ordered_cols = [
        "Month",
        "Target Month",
        "Sample Split",
        "Actual Next Month Log Realized Volatility",
        "ML Benchmark Forecast",
        "ML + Aggregate CRI Forecast",
        "ML + PCA CRI Forecast",
        "ML Benchmark Forecast Error",
        "ML + Aggregate CRI Forecast Error",
        "ML + PCA CRI Forecast Error",
        "ML Benchmark Squared Error",
        "ML + Aggregate CRI Squared Error",
        "ML + PCA CRI Squared Error",
        "ML Benchmark Absolute Error",
        "ML + Aggregate CRI Absolute Error",
        "ML + PCA CRI Absolute Error",
        "ML Benchmark Forecast Bias",
        "ML + Aggregate CRI Forecast Bias",
        "ML + PCA CRI Forecast Bias",
    ] + feature_cols_present

    ordered_cols = [col for col in ordered_cols if col in merged.columns]
    merged = merged[ordered_cols].sort_values("Target Month").reset_index(drop=True)
    return merged


def build_performance_table(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Benchmark
    bench_df = forecasts_df.dropna(
        subset=["Actual Next Month Log Realized Volatility", "ML Benchmark Forecast"]
    ).copy()

    if len(bench_df) > 0:
        bench_mse, bench_rmse, bench_mae = mse_rmse_mae(
            bench_df["Actual Next Month Log Realized Volatility"],
            bench_df["ML Benchmark Forecast"],
        )
    else:
        bench_mse, bench_rmse, bench_mae = np.nan, np.nan, np.nan

    rows.append(
        {
            "Model name": "ML Benchmark",
            "Number of OOS observations": len(bench_df),
            "MSE / MSFE": bench_mse,
            "RMSE": bench_rmse,
            "MAE": bench_mae,
            "R2_OS": np.nan,
            "R2_OS (%)": np.nan,
            "Benchmark MSFE used for R2_OS": np.nan,
            "Same-row benchmark observation count": np.nan,
        }
    )

    # Aggregate
    agg_df = forecasts_df.dropna(
        subset=[
            "Actual Next Month Log Realized Volatility",
            "ML Benchmark Forecast",
            "ML + Aggregate CRI Forecast",
        ]
    ).copy()

    if len(agg_df) > 0:
        agg_mse, agg_rmse, agg_mae = mse_rmse_mae(
            agg_df["Actual Next Month Log Realized Volatility"],
            agg_df["ML + Aggregate CRI Forecast"],
        )
        agg_bench_mse, _, _ = mse_rmse_mae(
            agg_df["Actual Next Month Log Realized Volatility"],
            agg_df["ML Benchmark Forecast"],
        )
        agg_r2_os = 1 - (agg_mse / agg_bench_mse) if agg_bench_mse != 0 else np.nan
    else:
        agg_mse, agg_rmse, agg_mae = np.nan, np.nan, np.nan
        agg_bench_mse, agg_r2_os = np.nan, np.nan

    rows.append(
        {
            "Model name": "ML + Aggregate CRI",
            "Number of OOS observations": len(agg_df),
            "MSE / MSFE": agg_mse,
            "RMSE": agg_rmse,
            "MAE": agg_mae,
            "R2_OS": agg_r2_os,
            "R2_OS (%)": agg_r2_os * 100 if pd.notna(agg_r2_os) else np.nan,
            "Benchmark MSFE used for R2_OS": agg_bench_mse,
            "Same-row benchmark observation count": len(agg_df),
        }
    )

    # PCA
    pca_df = forecasts_df.dropna(
        subset=[
            "Actual Next Month Log Realized Volatility",
            "ML Benchmark Forecast",
            "ML + PCA CRI Forecast",
        ]
    ).copy()

    if len(pca_df) > 0:
        pca_mse, pca_rmse, pca_mae = mse_rmse_mae(
            pca_df["Actual Next Month Log Realized Volatility"],
            pca_df["ML + PCA CRI Forecast"],
        )
        pca_bench_mse, _, _ = mse_rmse_mae(
            pca_df["Actual Next Month Log Realized Volatility"],
            pca_df["ML Benchmark Forecast"],
        )
        pca_r2_os = 1 - (pca_mse / pca_bench_mse) if pca_bench_mse != 0 else np.nan
    else:
        pca_mse, pca_rmse, pca_mae = np.nan, np.nan, np.nan
        pca_bench_mse, pca_r2_os = np.nan, np.nan

    rows.append(
        {
            "Model name": "ML + PCA CRI",
            "Number of OOS observations": len(pca_df),
            "MSE / MSFE": pca_mse,
            "RMSE": pca_rmse,
            "MAE": pca_mae,
            "R2_OS": pca_r2_os,
            "R2_OS (%)": pca_r2_os * 100 if pd.notna(pca_r2_os) else np.nan,
            "Benchmark MSFE used for R2_OS": pca_bench_mse,
            "Same-row benchmark observation count": len(pca_df),
        }
    )

    return pd.DataFrame(rows)


def main():
    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")
    input_file = base_path / "GARCH_Monthly_Target_Dataset.xlsx"

    forecasts_output_file = base_path / "ML_GB_Model_Forecasts.xlsx"
    performance_output_file = base_path / "ML_GB_Model_Performance.xlsx"
    hyperparams_output_file = base_path / "ML_GB_Model_Hyperparameters.xlsx"
    feature_importance_output_file = base_path / "ML_GB_Feature_Importance.xlsx"
    summary_output_file = base_path / "ML_GB_Model_Results.txt"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # --------------------------------------------------
    # READ DATA
    # --------------------------------------------------
    print("Reading input file...")

    df = pd.read_excel(input_file)
    df.columns = [str(col).strip() for col in df.columns]

    print("\nAvailable columns:")
    for col in df.columns:
        print(f"- {col}")

    required_cols = [
        "Month",
        "Target Month",
        "Next Month Log Realized Volatility",
        "Aggregate Climate Risk",
        "PCA Climate Risk Index",
        "Sample Split",
    ]
    check_required_columns(df, required_cols, "GARCH_Monthly_Target_Dataset.xlsx")

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Target Month"] = pd.to_datetime(df["Target Month"], errors="coerce")
    df["Sample Split"] = df["Sample Split"].astype(str).str.strip()

    df = df.dropna(subset=["Month", "Target Month", "Sample Split"]).copy()
    df = df.sort_values("Target Month").reset_index(drop=True)

    # --------------------------------------------------
    # SANITY CHECK: EXCLUDE GARCH/GJR FORECAST COLUMNS
    # --------------------------------------------------
    forbidden_cols = [
        col
        for col in df.columns
        if ("garch" in col.lower() or "gjr" in col.lower()) and "target" not in col.lower()
    ]

    print("\nSanity check:")
    print("The following GARCH / GJR-GARCH related columns exist but will NOT be used as predictors:")
    if forbidden_cols:
        for col in forbidden_cols:
            print(f"- {col}")
    else:
        print("- None found")

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    print("\nBuilding features...")

    df = build_features(df)

    feature_sets = get_feature_sets()
    all_feature_cols = sorted({col for cols in feature_sets.values() for col in cols})

    target_col = "Next Month Log Realized Volatility"

    df_model = df.dropna(subset=[target_col] + all_feature_cols).copy()
    df_model = df_model.sort_values("Target Month").reset_index(drop=True)

    in_sample_df = df_model[df_model["Sample Split"].str.lower() == "in-sample"].copy()
    oos_df = df_model[df_model["Sample Split"].str.lower() == "out-of-sample"].copy()

    print(f"In-sample observations: {len(in_sample_df)}")
    print(f"Out-of-sample observations: {len(oos_df)}")
    print(f"OOS start: {oos_df['Target Month'].min() if len(oos_df) > 0 else 'N/A'}")
    print(f"OOS end: {oos_df['Target Month'].max() if len(oos_df) > 0 else 'N/A'}")
    print("No data leakage: tuning uses in-sample only, and OOS forecasts use recursive expanding windows.")

    # --------------------------------------------------
    # HYPERPARAMETER TUNING
    # --------------------------------------------------
    print("\nTuning hyperparameters on in-sample data...")

    hyperparam_rows = []
    best_params_dict = {}

    for model_name, feature_cols in feature_sets.items():
        print(f"Tuning {model_name}...")
        tune_df = in_sample_df.dropna(subset=feature_cols + [target_col]).copy()

        if len(tune_df) < 40:
            raise ValueError(f"Not enough in-sample observations to tune {model_name}.")

        best_params, best_score, cv_splits, n_iter = tune_model(
            tune_df,
            feature_cols,
            target_col,
            random_state=42,
        )

        best_params_dict[model_name] = best_params

        hyperparam_row = {
            "Model name": model_name,
            "Best CV score": best_score,
            "Number of CV splits": cv_splits,
            "Number of randomized search iterations": n_iter,
        }
        hyperparam_row.update(best_params)
        hyperparam_rows.append(hyperparam_row)

    hyperparams_df = pd.DataFrame(hyperparam_rows)

    # --------------------------------------------------
    # RECURSIVE OOS FORECASTS
    # --------------------------------------------------
    print("\nRunning recursive expanding-window OOS forecasts...")

    benchmark_forecasts, benchmark_importance = recursive_oos_forecasts(
        df_model,
        feature_sets["ML Benchmark"],
        target_col,
        best_params_dict["ML Benchmark"],
        "ML Benchmark",
    )

    agg_forecasts, agg_importance = recursive_oos_forecasts(
        df_model,
        feature_sets["ML + Aggregate CRI"],
        target_col,
        best_params_dict["ML + Aggregate CRI"],
        "ML + Aggregate CRI",
    )

    pca_forecasts, pca_importance = recursive_oos_forecasts(
        df_model,
        feature_sets["ML + PCA CRI"],
        target_col,
        best_params_dict["ML + PCA CRI"],
        "ML + PCA CRI",
    )

    print(f"Benchmark forecasts generated: {len(benchmark_forecasts)}")
    print(f"Aggregate CRI forecasts generated: {len(agg_forecasts)}")
    print(f"PCA CRI forecasts generated: {len(pca_forecasts)}")

    # --------------------------------------------------
    # MERGE FORECASTS
    # --------------------------------------------------
    forecasts_df = merge_forecast_tables(
        benchmark_forecasts,
        agg_forecasts,
        pca_forecasts,
        all_feature_cols,
    )

    feature_importance_df = pd.concat(
        [benchmark_importance, agg_importance, pca_importance],
        ignore_index=True,
    )

    avg_feature_importance_df = (
        feature_importance_df.groupby(["Model name", "Feature"], as_index=False)["Importance"]
        .mean()
        .sort_values(["Model name", "Importance"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------
    print("\nCalculating performance...")
    performance_df = build_performance_table(forecasts_df)

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------
    print("\nSaving output files...")

    forecasts_df.to_excel(forecasts_output_file, index=False)
    performance_df.to_excel(performance_output_file, index=False)
    hyperparams_df.to_excel(hyperparams_output_file, index=False)
    avg_feature_importance_df.to_excel(feature_importance_output_file, index=False)

    # --------------------------------------------------
    # TEXT SUMMARY
    # --------------------------------------------------
    if len(performance_df) > 0 and performance_df["MSE / MSFE"].notna().any():
        best_msfe_model = performance_df.loc[
            performance_df["MSE / MSFE"].idxmin(), "Model name"
        ]
        best_rmse_model = performance_df.loc[
            performance_df["RMSE"].idxmin(), "Model name"
        ]
    else:
        best_msfe_model = np.nan
        best_rmse_model = np.nan

    summary_text = f"""
ML GRADIENT BOOSTING VOLATILITY FORECASTING RESULTS
==================================================

Input file:
{input_file}

Target variable:
Next Month Log Realized Volatility

Model structure:
1. ML Benchmark
   Predictors:
   - Log Realized Volatility
   - Log RV Lag 1
   - Log RV Lag 2
   - Log RV MA 3
   - Log RV MA 6

2. ML + Aggregate CRI
   Benchmark predictors + Aggregate Climate Risk

3. ML + PCA CRI
   Benchmark predictors + PCA Climate Risk Index

Cross-validation method:
TimeSeriesSplit

Hyperparameter tuning:
RandomizedSearchCV
- n_iter = 40
- scoring = neg_mean_squared_error
- random_state = 42
- n_jobs = -1

Important checks:
- GARCH and GJR-GARCH forecast columns were not used as predictors
- No random train-test split was used
- No shuffled cross-validation was used
- Hyperparameter tuning used in-sample data only
- OOS forecasts used recursive expanding-window estimation

OOS period:
Start: {oos_df['Target Month'].min() if len(oos_df) > 0 else np.nan}
End: {oos_df['Target Month'].max() if len(oos_df) > 0 else np.nan}

Number of successful forecasts:
- ML Benchmark: {len(benchmark_forecasts)}
- ML + Aggregate CRI: {len(agg_forecasts)}
- ML + PCA CRI: {len(pca_forecasts)}

Performance table:
{performance_df.to_string(index=False)}

Best model by MSFE:
{best_msfe_model}

Best model by RMSE:
{best_rmse_model}

Interpretation guide:
- Lower MSFE, RMSE, and MAE indicate better forecast performance.
- Positive R2_OS means the climate-augmented model improves on the ML Benchmark.
- R2_OS is calculated using the benchmark MSFE on the exact same OOS rows as the extended model.
"""

    with open(summary_output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nFinal performance table:")
    print(performance_df.to_string(index=False))
    print("\nAll output files saved.")


if __name__ == "__main__":
    main()