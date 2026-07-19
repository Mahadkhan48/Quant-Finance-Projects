# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 00:05:48 2026

@author: mahad
"""

# file: 08_Run_AR_80_20_Model.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from scipy import stats


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def check_required_columns(df, required_cols, file_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_name}: {missing}")


def compute_metrics(actual, forecast):
    actual = pd.Series(actual, dtype="float64").reset_index(drop=True)
    forecast = pd.Series(forecast, dtype="float64").reset_index(drop=True)

    errors = actual - forecast
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    return mse, mse, rmse, mae


def clark_west_test(actual, f_bench, f_model):
    actual = pd.Series(actual, dtype="float64").reset_index(drop=True)
    f_bench = pd.Series(f_bench, dtype="float64").reset_index(drop=True)
    f_model = pd.Series(f_model, dtype="float64").reset_index(drop=True)

    if len(actual) < 5:
        return np.nan, np.nan

    e_b = actual - f_bench
    e_m = actual - f_model
    adjustment = (f_bench - f_model) ** 2

    cw_series = e_b ** 2 - (e_m ** 2 - adjustment)

    X = np.ones((len(cw_series), 1))
    model = sm.OLS(cw_series, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 1},
    )

    stat = float(model.tvalues[0])
    pval = float(1 - stats.norm.cdf(stat))

    return stat, pval


def run_ols(y, X):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 1},
    )
    return model


def get_current_log_rv(df):
    """
    Gets current-month Log Realized Volatility.

    Preferred column:
    - Log Realized Volatility

    Fallback columns:
    - Annualized Realized Volatility (%)
    - Annualized Realized Volatility Decimal
    - Monthly Realized Volatility Decimal
    """
    if "Log Realized Volatility" in df.columns:
        return pd.to_numeric(df["Log Realized Volatility"], errors="coerce")

    if "Annualized Realized Volatility (%)" in df.columns:
        ann_vol = pd.to_numeric(df["Annualized Realized Volatility (%)"], errors="coerce")
        ann_vol = ann_vol.where(ann_vol > 0)
        return np.log(ann_vol)

    if "Annualized Realized Volatility Decimal" in df.columns:
        ann_vol_pct = pd.to_numeric(
            df["Annualized Realized Volatility Decimal"],
            errors="coerce",
        ) * 100
        ann_vol_pct = ann_vol_pct.where(ann_vol_pct > 0)
        return np.log(ann_vol_pct)

    if "Monthly Realized Volatility Decimal" in df.columns:
        ann_vol_pct = (
            pd.to_numeric(df["Monthly Realized Volatility Decimal"], errors="coerce")
            * np.sqrt(12)
            * 100
        )
        ann_vol_pct = ann_vol_pct.where(ann_vol_pct > 0)
        return np.log(ann_vol_pct)

    raise ValueError(
        "Could not find or create current-month Log Realized Volatility. "
        "Please include one of these columns: "
        "'Log Realized Volatility', "
        "'Annualized Realized Volatility (%)', "
        "'Annualized Realized Volatility Decimal', or "
        "'Monthly Realized Volatility Decimal'."
    )


def add_constant_and_align_for_prediction(model, X_test):
    X_test = sm.add_constant(X_test, has_constant="add")

    for col in model.model.exog_names:
        if col not in X_test.columns:
            X_test[col] = 1.0

    X_test = X_test[model.model.exog_names]
    return X_test


def fit_and_predict(train_df, test_row, predictors, target_col):
    """
    Fits AR-style OLS model and predicts one OOS observation.
    """
    train_sub = train_df.dropna(subset=predictors + [target_col]).copy()

    if len(train_sub) < 20:
        return np.nan, None, "Fewer than 20 training observations"

    if test_row[predictors].isna().any():
        return np.nan, None, "Missing current predictor value"

    y_train = train_sub[target_col]
    X_train = train_sub[predictors]

    try:
        model = run_ols(y_train, X_train)

        X_test = test_row[predictors].to_frame().T
        X_test = add_constant_and_align_for_prediction(model, X_test)

        forecast = float(model.predict(X_test).iloc[0])
        return forecast, model, ""

    except Exception as e:
        return np.nan, None, str(e)


def fit_and_predict_with_train_standardized_climate(
    train_df,
    test_row,
    base_predictor,
    climate_col,
    climate_std_col,
    target_col,
):
    """
    Standardizes climate risk using training-sample mean and std only.
    This avoids look-ahead bias.
    """
    required_cols = [base_predictor, climate_col, target_col]
    train_sub = train_df.dropna(subset=required_cols).copy()

    if len(train_sub) < 20:
        return np.nan, None, np.nan, np.nan, np.nan, "Fewer than 20 training observations"

    if pd.isna(test_row[base_predictor]) or pd.isna(test_row[climate_col]):
        return np.nan, None, np.nan, np.nan, np.nan, "Missing current predictor value"

    climate_mean = train_sub[climate_col].mean()
    climate_std = train_sub[climate_col].std()

    if pd.isna(climate_std) or climate_std == 0:
        return np.nan, None, climate_mean, climate_std, np.nan, "Climate std is zero or missing"

    train_sub[climate_std_col] = (train_sub[climate_col] - climate_mean) / climate_std
    current_climate_std = (test_row[climate_col] - climate_mean) / climate_std

    predictors = [base_predictor, climate_std_col]
    y_train = train_sub[target_col]
    X_train = train_sub[predictors]

    try:
        model = run_ols(y_train, X_train)

        X_test = pd.DataFrame(
            [
                {
                    base_predictor: test_row[base_predictor],
                    climate_std_col: current_climate_std,
                }
            ]
        )
        X_test = add_constant_and_align_for_prediction(model, X_test)

        forecast = float(model.predict(X_test).iloc[0])

        return (
            forecast,
            model,
            climate_mean,
            climate_std,
            current_climate_std,
            "",
        )
    except Exception as e:
        return np.nan, None, climate_mean, climate_std, current_climate_std, str(e)


# ==================================================
# MAIN SCRIPT
# ==================================================
def main():
    print("Running final AR model with 80/20 in-sample and out-of-sample split...")

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    input_file = base / "GARCH_Monthly_Target_Dataset.xlsx"

    in_sample_results_file = base / "AR_80_20_InSample_Results.xlsx"
    in_sample_coef_file = base / "AR_80_20_InSample_Coefficients.xlsx"
    oos_forecasts_file = base / "AR_80_20_OutOfSample_Forecasts.xlsx"
    oos_performance_file = base / "AR_80_20_OutOfSample_Performance.xlsx"
    summary_file = base / "AR_80_20_Model_Results.txt"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # --------------------------------------------------
    # READ DATA
    # --------------------------------------------------
    print("Reading data...")

    df = pd.read_excel(input_file)
    df.columns = df.columns.map(str).str.strip()

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

    # --------------------------------------------------
    # CLEAN DATES
    # --------------------------------------------------
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df["Target Month"] = pd.to_datetime(df["Target Month"], errors="coerce")

    df = df.dropna(subset=["Month", "Target Month"]).copy()

    df["Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()
    df["Target Month"] = df["Target Month"].dt.to_period("M").dt.to_timestamp()

    # --------------------------------------------------
    # CLEAN NUMERIC COLUMNS
    # --------------------------------------------------
    df["Log Realized Volatility"] = get_current_log_rv(df)

    numeric_cols = [
        "Log Realized Volatility",
        "Next Month Log Realized Volatility",
        "Aggregate Climate Risk",
        "PCA Climate Risk Index",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Sample Split"] = df["Sample Split"].astype(str).str.strip()

    # --------------------------------------------------
    # FINAL CLEAN DATASET
    # --------------------------------------------------
    df_model = df.dropna(
        subset=[
            "Month",
            "Target Month",
            "Log Realized Volatility",
            "Next Month Log Realized Volatility",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
            "Sample Split",
        ]
    ).copy()

    df_model = df_model.sort_values("Target Month").reset_index(drop=True)

    in_sample_df = df_model[df_model["Sample Split"].str.lower() == "in-sample"].copy()
    oos_df = df_model[df_model["Sample Split"].str.lower() == "out-of-sample"].copy()

    if len(in_sample_df) == 0:
        raise ValueError("No in-sample rows found. Check the 'Sample Split' column.")

    if len(oos_df) == 0:
        raise ValueError("No out-of-sample rows found. Check the 'Sample Split' column.")

    print("\nSanity check:")
    print(f"Total usable rows: {len(df_model)}")
    print(f"In-sample rows: {len(in_sample_df)}")
    print(f"Out-of-sample rows: {len(oos_df)}")
    print(f"In-sample target start: {in_sample_df['Target Month'].min()}")
    print(f"In-sample target end: {in_sample_df['Target Month'].max()}")
    print(f"OOS target start: {oos_df['Target Month'].min()}")
    print(f"OOS target end: {oos_df['Target Month'].max()}")
    print("OOS months are taken from Sample Split, not from hard-coded dates.")
    print("Climate variables are standardized using training data only during recursive OOS forecasting.")

    # --------------------------------------------------
    # PART A: IN-SAMPLE MODELS
    # --------------------------------------------------
    print("\nRunning in-sample AR models...")

    in_train = in_sample_df.copy()

    agg_mean_in = in_train["Aggregate Climate Risk"].mean()
    agg_std_in = in_train["Aggregate Climate Risk"].std()

    pca_mean_in = in_train["PCA Climate Risk Index"].mean()
    pca_std_in = in_train["PCA Climate Risk Index"].std()

    if agg_std_in == 0 or pd.isna(agg_std_in):
        raise ValueError("Aggregate Climate Risk in-sample standard deviation is zero or missing.")

    if pca_std_in == 0 or pd.isna(pca_std_in):
        raise ValueError("PCA Climate Risk Index in-sample standard deviation is zero or missing.")

    in_train["Aggregate Climate Risk Std"] = (
        in_train["Aggregate Climate Risk"] - agg_mean_in
    ) / agg_std_in

    in_train["PCA Climate Risk Index Std"] = (
        in_train["PCA Climate Risk Index"] - pca_mean_in
    ) / pca_std_in

    target_col = "Next Month Log Realized Volatility"

    in_sample_models = {
        "AR Benchmark": ["Log Realized Volatility"],
        "AR + Aggregate CRI": ["Log Realized Volatility", "Aggregate Climate Risk Std"],
        "AR + PCA CRI": ["Log Realized Volatility", "PCA Climate Risk Index Std"],
    }

    in_summary_rows = []
    in_coef_rows = []

    for model_name, predictors in in_sample_models.items():
        sub = in_train.dropna(subset=predictors + [target_col]).copy()

        y = sub[target_col]
        X = sub[predictors]

        model = run_ols(y, X)

        in_summary_rows.append(
            {
                "Model name": model_name,
                "N": int(model.nobs),
                "R2": model.rsquared,
                "Adj R2": model.rsquared_adj,
                "AIC": model.aic,
                "BIC": model.bic,
                "Climate mean used": agg_mean_in if "Aggregate" in model_name else pca_mean_in if "PCA" in model_name else np.nan,
                "Climate std used": agg_std_in if "Aggregate" in model_name else pca_std_in if "PCA" in model_name else np.nan,
            }
        )

        for var in model.params.index:
            in_coef_rows.append(
                {
                    "Model name": model_name,
                    "Variable": var,
                    "Coefficient": model.params[var],
                    "t-stat HAC": model.tvalues[var],
                    "p-value HAC": model.pvalues[var],
                }
            )

    df_in_summary = pd.DataFrame(in_summary_rows)
    df_in_coef = pd.DataFrame(in_coef_rows)

    # --------------------------------------------------
    # PART B: RECURSIVE OUT-OF-SAMPLE FORECASTING
    # --------------------------------------------------
    print("\nRunning recursive OOS forecasting...")

    forecasts = []

    target_months = oos_df["Target Month"].sort_values().unique()

    for i, t in enumerate(target_months):
        current_target_month = pd.Timestamp(t)
        print(f"Forecast {i + 1}/{len(target_months)}: {current_target_month.date()}")

        train = df_model[df_model["Target Month"] < current_target_month].copy()
        test = df_model[df_model["Target Month"] == current_target_month].copy()

        if len(train) < 20 or len(test) == 0:
            continue

        test_row = test.iloc[0]
        actual = test_row[target_col]

        row = {
            "Forecast Origin Month": test_row["Month"],
            "Target Month": current_target_month,
            "Sample Split": test_row["Sample Split"],
            "Actual Next Month Log Realized Volatility": actual,
            "Log Realized Volatility": test_row["Log Realized Volatility"],
            "Aggregate Climate Risk": test_row["Aggregate Climate Risk"],
            "PCA Climate Risk Index": test_row["PCA Climate Risk Index"],
            "AR Benchmark Forecast": np.nan,
            "AR Benchmark Forecast Error": np.nan,
            "AR Benchmark Squared Error": np.nan,
            "AR Benchmark Absolute Error": np.nan,
            "AR Benchmark Forecast Bias": np.nan,
            "AR Benchmark Training Observations": np.nan,
            "AR Benchmark Error Message": "",
            "AR + Aggregate CRI Forecast": np.nan,
            "AR + Aggregate CRI Forecast Error": np.nan,
            "AR + Aggregate CRI Squared Error": np.nan,
            "AR + Aggregate CRI Absolute Error": np.nan,
            "AR + Aggregate CRI Forecast Bias": np.nan,
            "Aggregate CRI Training Mean": np.nan,
            "Aggregate CRI Training Std": np.nan,
            "Aggregate CRI Standardized Current Value": np.nan,
            "AR + Aggregate CRI Training Observations": np.nan,
            "AR + Aggregate CRI Error Message": "",
            "AR + PCA CRI Forecast": np.nan,
            "AR + PCA CRI Forecast Error": np.nan,
            "AR + PCA CRI Squared Error": np.nan,
            "AR + PCA CRI Absolute Error": np.nan,
            "AR + PCA CRI Forecast Bias": np.nan,
            "PCA CRI Training Mean": np.nan,
            "PCA CRI Training Std": np.nan,
            "PCA CRI Standardized Current Value": np.nan,
            "AR + PCA CRI Training Observations": np.nan,
            "AR + PCA CRI Error Message": "",
        }

        # ---------------------------
        # MODEL 1: AR BENCHMARK
        # ---------------------------
        bench_train = train.dropna(subset=["Log Realized Volatility", target_col]).copy()
        row["AR Benchmark Training Observations"] = len(bench_train)

        bench_forecast, bench_model, bench_error = fit_and_predict(
            train_df=train,
            test_row=test_row,
            predictors=["Log Realized Volatility"],
            target_col=target_col,
        )

        if pd.notna(bench_forecast):
            row["AR Benchmark Forecast"] = bench_forecast
            row["AR Benchmark Forecast Error"] = actual - bench_forecast
            row["AR Benchmark Squared Error"] = (actual - bench_forecast) ** 2
            row["AR Benchmark Absolute Error"] = abs(actual - bench_forecast)
            row["AR Benchmark Forecast Bias"] = bench_forecast - actual
        else:
            row["AR Benchmark Error Message"] = bench_error

        # ---------------------------
        # MODEL 2: AR + AGGREGATE CRI
        # ---------------------------
        agg_train = train.dropna(
            subset=["Log Realized Volatility", "Aggregate Climate Risk", target_col]
        ).copy()
        row["AR + Aggregate CRI Training Observations"] = len(agg_train)

        (
            agg_forecast,
            agg_model,
            agg_mean,
            agg_std,
            current_agg_std,
            agg_error,
        ) = fit_and_predict_with_train_standardized_climate(
            train_df=train,
            test_row=test_row,
            base_predictor="Log Realized Volatility",
            climate_col="Aggregate Climate Risk",
            climate_std_col="Aggregate Climate Risk Std",
            target_col=target_col,
        )

        row["Aggregate CRI Training Mean"] = agg_mean
        row["Aggregate CRI Training Std"] = agg_std
        row["Aggregate CRI Standardized Current Value"] = current_agg_std

        if pd.notna(agg_forecast):
            row["AR + Aggregate CRI Forecast"] = agg_forecast
            row["AR + Aggregate CRI Forecast Error"] = actual - agg_forecast
            row["AR + Aggregate CRI Squared Error"] = (actual - agg_forecast) ** 2
            row["AR + Aggregate CRI Absolute Error"] = abs(actual - agg_forecast)
            row["AR + Aggregate CRI Forecast Bias"] = agg_forecast - actual
        else:
            row["AR + Aggregate CRI Error Message"] = agg_error

        # ---------------------------
        # MODEL 3: AR + PCA CRI
        # ---------------------------
        pca_train = train.dropna(
            subset=["Log Realized Volatility", "PCA Climate Risk Index", target_col]
        ).copy()
        row["AR + PCA CRI Training Observations"] = len(pca_train)

        (
            pca_forecast,
            pca_model,
            pca_mean,
            pca_std,
            current_pca_std,
            pca_error,
        ) = fit_and_predict_with_train_standardized_climate(
            train_df=train,
            test_row=test_row,
            base_predictor="Log Realized Volatility",
            climate_col="PCA Climate Risk Index",
            climate_std_col="PCA Climate Risk Index Std",
            target_col=target_col,
        )

        row["PCA CRI Training Mean"] = pca_mean
        row["PCA CRI Training Std"] = pca_std
        row["PCA CRI Standardized Current Value"] = current_pca_std

        if pd.notna(pca_forecast):
            row["AR + PCA CRI Forecast"] = pca_forecast
            row["AR + PCA CRI Forecast Error"] = actual - pca_forecast
            row["AR + PCA CRI Squared Error"] = (actual - pca_forecast) ** 2
            row["AR + PCA CRI Absolute Error"] = abs(actual - pca_forecast)
            row["AR + PCA CRI Forecast Bias"] = pca_forecast - actual
        else:
            row["AR + PCA CRI Error Message"] = pca_error

        forecasts.append(row)

    df_forecast = pd.DataFrame(forecasts)

    # --------------------------------------------------
    # PART C: PERFORMANCE
    # --------------------------------------------------
    print("\nCalculating OOS performance...")

    actual_col = "Actual Next Month Log Realized Volatility"
    bench_col = "AR Benchmark Forecast"
    agg_col = "AR + Aggregate CRI Forecast"
    pca_col = "AR + PCA CRI Forecast"

    perf_rows = []

    # Benchmark
    bench_df = df_forecast.dropna(subset=[actual_col, bench_col]).copy()

    if len(bench_df) > 0:
        bench_mse, bench_msfe, bench_rmse, bench_mae = compute_metrics(
            bench_df[actual_col],
            bench_df[bench_col],
        )
    else:
        bench_mse = bench_msfe = bench_rmse = bench_mae = np.nan

    perf_rows.append(
        {
            "Model name": "AR Benchmark",
            "Number of OOS observations": len(bench_df),
            "MSE / MSFE": bench_msfe,
            "RMSE": bench_rmse,
            "MAE": bench_mae,
            "R2_OS": np.nan,
            "R2_OS (%)": np.nan,
            "Clark-West statistic": np.nan,
            "Clark-West p-value": np.nan,
            "Benchmark MSFE used for R2_OS": np.nan,
            "Same-row benchmark observation count": np.nan,
        }
    )

    # Aggregate CRI
    agg_df = df_forecast.dropna(subset=[actual_col, bench_col, agg_col]).copy()

    if len(agg_df) > 0:
        agg_mse, agg_msfe, agg_rmse, agg_mae = compute_metrics(
            agg_df[actual_col],
            agg_df[agg_col],
        )
        _, agg_bench_msfe, _, _ = compute_metrics(
            agg_df[actual_col],
            agg_df[bench_col],
        )
        agg_r2os = 1 - (agg_msfe / agg_bench_msfe) if agg_bench_msfe != 0 else np.nan
        agg_cw_stat, agg_cw_p = clark_west_test(
            agg_df[actual_col],
            agg_df[bench_col],
            agg_df[agg_col],
        )
    else:
        agg_msfe = agg_rmse = agg_mae = agg_bench_msfe = agg_r2os = np.nan
        agg_cw_stat = agg_cw_p = np.nan

    perf_rows.append(
        {
            "Model name": "AR + Aggregate CRI",
            "Number of OOS observations": len(agg_df),
            "MSE / MSFE": agg_msfe,
            "RMSE": agg_rmse,
            "MAE": agg_mae,
            "R2_OS": agg_r2os,
            "R2_OS (%)": agg_r2os * 100 if pd.notna(agg_r2os) else np.nan,
            "Clark-West statistic": agg_cw_stat,
            "Clark-West p-value": agg_cw_p,
            "Benchmark MSFE used for R2_OS": agg_bench_msfe,
            "Same-row benchmark observation count": len(agg_df),
        }
    )

    # PCA CRI
    pca_df = df_forecast.dropna(subset=[actual_col, bench_col, pca_col]).copy()

    if len(pca_df) > 0:
        pca_mse, pca_msfe, pca_rmse, pca_mae = compute_metrics(
            pca_df[actual_col],
            pca_df[pca_col],
        )
        _, pca_bench_msfe, _, _ = compute_metrics(
            pca_df[actual_col],
            pca_df[bench_col],
        )
        pca_r2os = 1 - (pca_msfe / pca_bench_msfe) if pca_bench_msfe != 0 else np.nan
        pca_cw_stat, pca_cw_p = clark_west_test(
            pca_df[actual_col],
            pca_df[bench_col],
            pca_df[pca_col],
        )
    else:
        pca_msfe = pca_rmse = pca_mae = pca_bench_msfe = pca_r2os = np.nan
        pca_cw_stat = pca_cw_p = np.nan

    perf_rows.append(
        {
            "Model name": "AR + PCA CRI",
            "Number of OOS observations": len(pca_df),
            "MSE / MSFE": pca_msfe,
            "RMSE": pca_rmse,
            "MAE": pca_mae,
            "R2_OS": pca_r2os,
            "R2_OS (%)": pca_r2os * 100 if pd.notna(pca_r2os) else np.nan,
            "Clark-West statistic": pca_cw_stat,
            "Clark-West p-value": pca_cw_p,
            "Benchmark MSFE used for R2_OS": pca_bench_msfe,
            "Same-row benchmark observation count": len(pca_df),
        }
    )

    df_perf = pd.DataFrame(perf_rows)

    # --------------------------------------------------
    # PART D: SAVE OUTPUTS
    # --------------------------------------------------
    print("\nSaving outputs...")

    df_in_summary.to_excel(in_sample_results_file, index=False)
    df_in_coef.to_excel(in_sample_coef_file, index=False)
    df_forecast.to_excel(oos_forecasts_file, index=False)
    df_perf.to_excel(oos_performance_file, index=False)

    # --------------------------------------------------
    # PART E: TEXT SUMMARY
    # --------------------------------------------------
    if len(df_perf) > 0 and df_perf["MSE / MSFE"].notna().any():
        best_msfe_model = df_perf.loc[df_perf["MSE / MSFE"].idxmin(), "Model name"]
        best_rmse_model = df_perf.loc[df_perf["RMSE"].idxmin(), "Model name"]
    else:
        best_msfe_model = np.nan
        best_rmse_model = np.nan

    summary_text = f"""
AR MODEL RESULTS WITH FINAL 80/20 SAMPLE SPLIT
=============================================

Input file:
{input_file}

Output files created:
1. {in_sample_results_file}
2. {in_sample_coef_file}
3. {oos_forecasts_file}
4. {oos_performance_file}
5. {summary_file}

Sample period:
Forecast-origin Month: {df_model['Month'].min()} to {df_model['Month'].max()}
Target Month: {df_model['Target Month'].min()} to {df_model['Target Month'].max()}

In-sample period:
Forecast-origin Month: {in_sample_df['Month'].min()} to {in_sample_df['Month'].max()}
Target Month: {in_sample_df['Target Month'].min()} to {in_sample_df['Target Month'].max()}
In-sample observations: {len(in_sample_df)}

Out-of-sample period:
Forecast-origin Month: {oos_df['Month'].min()} to {oos_df['Month'].max()}
Target Month: {oos_df['Target Month'].min()} to {oos_df['Target Month'].max()}
OOS observations: {len(oos_df)}

Models:
Model 1: AR Benchmark
Next Month Log Realized Volatility = constant + beta1 × Log Realized Volatility

Model 2: AR + Aggregate CRI
Next Month Log Realized Volatility = constant + beta1 × Log Realized Volatility + beta2 × Aggregate Climate Risk Std

Model 3: AR + PCA CRI
Next Month Log Realized Volatility = constant + beta1 × Log Realized Volatility + beta2 × PCA Climate Risk Index Std

Important methodology notes:
- This final AR model uses the same 80/20 sample split as the GARCH, GJR-GARCH, and ML models.
- OOS months are selected from the Sample Split column.
- The OOS period is not hard-coded.
- Climate indices are standardized using training data only in the recursive OOS forecasting stage.
- This avoids look-ahead bias.
- Recursive expanding-window forecasting is used.
- Clark-West test is used for AR + climate models because the extended AR models nest the AR benchmark.

Successful OOS forecasts:
- AR Benchmark: {df_forecast[bench_col].notna().sum() if len(df_forecast) > 0 else 0}
- AR + Aggregate CRI: {df_forecast[agg_col].notna().sum() if len(df_forecast) > 0 else 0}
- AR + PCA CRI: {df_forecast[pca_col].notna().sum() if len(df_forecast) > 0 else 0}

In-sample results:
{df_in_summary.to_string(index=False)}

Out-of-sample performance:
{df_perf.to_string(index=False)}

Lowest error models:
- Lowest MSFE: {best_msfe_model}
- Lowest RMSE: {best_rmse_model}

Interpretation guide:
- Lower MSFE, RMSE, and MAE indicate better forecast performance.
- Positive R2_OS means the climate-augmented AR model improves over the AR Benchmark.
- Negative R2_OS means the climate-augmented AR model performs worse than the AR Benchmark.
- A Clark-West p-value below 0.05 suggests statistically significant improvement over the benchmark.
"""

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nFinal OOS performance table:")
    print(df_perf.to_string(index=False))
    print("\nDONE ✅")


if __name__ == "__main__":
    main()