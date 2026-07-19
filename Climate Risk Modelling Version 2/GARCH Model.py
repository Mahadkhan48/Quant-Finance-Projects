# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:44:29 2026

@author: mahad
"""

# file: 04_Run_TwoStep_GARCH_Models.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import norm

try:
    from arch import arch_model
except ImportError:
    raise ImportError("The 'arch' package is required. Install it using: pip install arch")


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def check_required_columns(df, required_cols, file_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_name}: {missing}")


def calculate_performance_metrics(actual, forecast):
    actual = pd.Series(actual, dtype="float64")
    forecast = pd.Series(forecast, dtype="float64")

    errors = actual - forecast
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    return mse, mse, rmse, mae


def clark_west_test(actual, benchmark_forecast, extended_forecast):
    actual = pd.Series(actual, dtype="float64").reset_index(drop=True)
    benchmark_forecast = pd.Series(benchmark_forecast, dtype="float64").reset_index(drop=True)
    extended_forecast = pd.Series(extended_forecast, dtype="float64").reset_index(drop=True)

    if len(actual) < 5:
        return np.nan, np.nan

    e_benchmark = actual - benchmark_forecast
    e_extended = actual - extended_forecast

    cw_series = e_benchmark ** 2 - (
        e_extended ** 2 - (benchmark_forecast - extended_forecast) ** 2
    )

    X = np.ones((len(cw_series), 1))
    model = sm.OLS(cw_series, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    t_stat = model.tvalues[0]
    p_value = 1 - norm.cdf(t_stat)

    return t_stat, p_value


def fit_ols_and_predict(train_df, y_col, predictor_cols, current_values, min_train_obs=30):
    """
    Fits OLS model on training data and predicts one current OOS observation.
    HAC/Newey-West standard errors are used for coefficient diagnostics.
    """
    required_cols = [y_col] + predictor_cols
    train_clean = train_df.dropna(subset=required_cols).copy()

    result = {
        "success": False,
        "forecast": np.nan,
        "nobs": len(train_clean),
        "r2_in_sample": np.nan,
        "params": {},
        "tvalues_hac": {},
        "pvalues_hac": {},
        "error_message": "",
    }

    if len(train_clean) < min_train_obs:
        result["error_message"] = f"Fewer than {min_train_obs} training observations"
        return result

    for col in predictor_cols:
        if col not in current_values:
            result["error_message"] = f"Missing current value for predictor: {col}"
            return result
        if pd.isna(current_values[col]):
            result["error_message"] = f"Current value is missing for predictor: {col}"
            return result

    y_train = train_clean[y_col]
    X_train = train_clean[predictor_cols]
    X_train = sm.add_constant(X_train, has_constant="add")

    try:
        model = sm.OLS(y_train, X_train).fit()
        hac_model = model.get_robustcov_results(cov_type="HAC", maxlags=1)

        X_current = pd.DataFrame([{col: current_values[col] for col in predictor_cols}])
        X_current = sm.add_constant(X_current, has_constant="add")

        # Align columns
        for col in X_train.columns:
            if col not in X_current.columns:
                X_current[col] = 1.0
        X_current = X_current[X_train.columns]

        forecast = float(model.predict(X_current).iloc[0])

        params = pd.Series(model.params, index=X_train.columns)
        tvalues_hac = pd.Series(hac_model.tvalues, index=X_train.columns)
        pvalues_hac = pd.Series(hac_model.pvalues, index=X_train.columns)

        result["success"] = True
        result["forecast"] = forecast
        result["nobs"] = int(model.nobs)
        result["r2_in_sample"] = model.rsquared
        result["params"] = params.to_dict()
        result["tvalues_hac"] = tvalues_hac.to_dict()
        result["pvalues_hac"] = pvalues_hac.to_dict()

    except Exception as e:
        result["error_message"] = str(e)

    return result


def estimate_raw_garch_forecast(training_returns, horizon_days):
    """
    Estimate benchmark GARCH(1,1) on daily returns and forecast target-month volatility.

    Daily returns are in percentage form. Therefore:
    - GARCH variance is in percent-squared units.
    - Monthly forecast variance = sum of daily forecast variances.
    - Annualized forecast volatility (%) = sqrt(monthly variance) * sqrt(12).
    - Log forecast volatility = log(annualized forecast volatility %).
    """
    result = {
        "success": False,
        "forecast_annualized_volatility": np.nan,
        "forecast_log_volatility": np.nan,
        "omega": np.nan,
        "alpha": np.nan,
        "beta": np.nan,
        "alpha_plus_beta": np.nan,
        "mu": np.nan,
        "log_likelihood": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "convergence_flag": np.nan,
        "error_message": "",
    }

    training_returns = pd.Series(training_returns, dtype="float64").dropna()

    if len(training_returns) < 500:
        result["error_message"] = "Fewer than 500 daily training observations"
        return result

    if pd.isna(horizon_days) or horizon_days <= 0:
        result["error_message"] = "Invalid forecast horizon"
        return result

    try:
        garch = arch_model(
            training_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        fitted = garch.fit(disp="off", show_warning=False)

        convergence_flag = getattr(fitted, "convergence_flag", np.nan)
        if pd.notna(convergence_flag) and int(convergence_flag) != 0:
            result["error_message"] = f"Non-zero convergence flag: {convergence_flag}"
            result["convergence_flag"] = convergence_flag
            return result

        forecast_obj = fitted.forecast(horizon=int(horizon_days), reindex=False)
        forecast_variances = forecast_obj.variance.iloc[-1].values

        monthly_forecast_variance = np.sum(forecast_variances)
        monthly_forecast_volatility = np.sqrt(monthly_forecast_variance)
        annualized_forecast_volatility = monthly_forecast_volatility * np.sqrt(12)

        if annualized_forecast_volatility <= 0 or not np.isfinite(annualized_forecast_volatility):
            result["error_message"] = "Invalid annualized forecast volatility"
            return result

        log_forecast_volatility = np.log(annualized_forecast_volatility)

        params = fitted.params

        result["success"] = True
        result["forecast_annualized_volatility"] = annualized_forecast_volatility
        result["forecast_log_volatility"] = log_forecast_volatility
        result["mu"] = params.get("mu", np.nan)
        result["omega"] = params.get("omega", np.nan)
        result["alpha"] = params.get("alpha[1]", np.nan)
        result["beta"] = params.get("beta[1]", np.nan)
        result["alpha_plus_beta"] = result["alpha"] + result["beta"]
        result["log_likelihood"] = fitted.loglikelihood
        result["aic"] = fitted.aic
        result["bic"] = fitted.bic
        result["convergence_flag"] = convergence_flag

    except Exception as e:
        result["error_message"] = str(e)

    return result


def add_coefficient_values(row_dict, model_result):
    """
    Adds OLS coefficients, HAC t-stats, and HAC p-values to a coefficient row.
    """
    for key, value in model_result["params"].items():
        row_dict[f"coef_{key}"] = value

    for key, value in model_result["tvalues_hac"].items():
        row_dict[f"tstat_HAC_{key}"] = value

    for key, value in model_result["pvalues_hac"].items():
        row_dict[f"pvalue_HAC_{key}"] = value

    return row_dict


# ==================================================
# MAIN SCRIPT
# ==================================================
def main():
    print("Running two-step GARCH forecast-augmentation models...")

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    daily_file = base_path / "GARCH_Daily_Model_Dataset.xlsx"
    monthly_file = base_path / "GARCH_Monthly_Target_Dataset.xlsx"

    forecasts_output_file = base_path / "GARCH_TwoStep_Model_Forecasts.xlsx"
    performance_output_file = base_path / "GARCH_TwoStep_Model_Performance.xlsx"
    coefficients_output_file = base_path / "GARCH_TwoStep_Model_Coefficients.xlsx"
    raw_garch_output_file = base_path / "GARCH_TwoStep_Raw_GARCH_Forecasts.xlsx"
    summary_output_file = base_path / "GARCH_TwoStep_Model_Results.txt"

    if not daily_file.exists():
        raise FileNotFoundError(f"Input file not found: {daily_file}")

    if not monthly_file.exists():
        raise FileNotFoundError(f"Input file not found: {monthly_file}")

    # --------------------------------------------------
    # READ DATA
    # --------------------------------------------------
    print("Reading input files...")

    df_daily = pd.read_excel(daily_file)
    df_monthly = pd.read_excel(monthly_file)

    df_daily.columns = [str(col).strip() for col in df_daily.columns]
    df_monthly.columns = [str(col).strip() for col in df_monthly.columns]

    required_daily_cols = [
        "Date",
        "Month",
        "Daily Return (%)",
    ]
    required_monthly_cols = [
        "Month",
        "Target Month",
        "Next Month Log Realized Volatility",
        "Next Month Annualized Realized Volatility (%)",
        "Aggregate Climate Risk",
        "PCA Climate Risk Index",
        "Number of Trading Days in Target Month",
        "Sample Split",
    ]

    check_required_columns(df_daily, required_daily_cols, "GARCH_Daily_Model_Dataset.xlsx")
    check_required_columns(df_monthly, required_monthly_cols, "GARCH_Monthly_Target_Dataset.xlsx")

    df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce")
    df_daily["Month"] = pd.to_datetime(df_daily["Month"], errors="coerce")
    df_monthly["Month"] = pd.to_datetime(df_monthly["Month"], errors="coerce")
    df_monthly["Target Month"] = pd.to_datetime(df_monthly["Target Month"], errors="coerce")

    df_daily["Daily Return (%)"] = pd.to_numeric(df_daily["Daily Return (%)"], errors="coerce")

    numeric_monthly_cols = [
        "Next Month Log Realized Volatility",
        "Next Month Annualized Realized Volatility (%)",
        "Aggregate Climate Risk",
        "PCA Climate Risk Index",
        "Number of Trading Days in Target Month",
    ]
    for col in numeric_monthly_cols:
        df_monthly[col] = pd.to_numeric(df_monthly[col], errors="coerce")

    df_monthly["Sample Split"] = df_monthly["Sample Split"].astype(str).str.strip()

    df_daily = df_daily.dropna(subset=["Date", "Month", "Daily Return (%)"]).copy()
    df_monthly = df_monthly.dropna(
        subset=[
            "Month",
            "Target Month",
            "Next Month Log Realized Volatility",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
            "Number of Trading Days in Target Month",
            "Sample Split",
        ]
    ).copy()

    df_daily = df_daily.sort_values("Date").reset_index(drop=True)
    df_monthly = df_monthly.sort_values("Target Month").reset_index(drop=True)

    # --------------------------------------------------
    # STEP 1: CREATE RAW GARCH FORECASTS
    # --------------------------------------------------
    print("Step 1: Estimating raw benchmark GARCH forecasts...")

    raw_garch_rows = []

    for i, row in df_monthly.iterrows():
        origin_month = row["Month"]
        target_month = row["Target Month"]
        horizon_days = int(row["Number of Trading Days in Target Month"])

        print(f"Raw GARCH {i + 1}/{len(df_monthly)}: target month {target_month.date()}")

        raw_row = {
            "Month": origin_month,
            "Target Month": target_month,
            "Sample Split": row["Sample Split"],
            "Number of Trading Days in Target Month": horizon_days,
            "Actual Next Month Log Realized Volatility": row["Next Month Log Realized Volatility"],
            "Actual Next Month Annualized Realized Volatility (%)": row["Next Month Annualized Realized Volatility (%)"],
            "Raw GARCH Forecast Annualized Volatility (%)": np.nan,
            "Raw GARCH Forecast Log Volatility": np.nan,
            "Raw GARCH Forecast Error": np.nan,
            "Daily Training Observations": np.nan,
            "GARCH mu": np.nan,
            "GARCH omega": np.nan,
            "GARCH alpha": np.nan,
            "GARCH beta": np.nan,
            "GARCH alpha + beta": np.nan,
            "GARCH log likelihood": np.nan,
            "GARCH AIC": np.nan,
            "GARCH BIC": np.nan,
            "GARCH convergence flag": np.nan,
            "Raw GARCH Error Message": "",
        }

        origin_daily = df_daily[df_daily["Month"] == origin_month].copy()

        if len(origin_daily) == 0:
            raw_row["Raw GARCH Error Message"] = "No daily returns found for origin month"
            raw_garch_rows.append(raw_row)
            continue

        last_origin_date = origin_daily["Date"].max()

        train_returns = df_daily.loc[
            df_daily["Date"] <= last_origin_date,
            "Daily Return (%)",
        ].dropna()

        raw_row["Daily Training Observations"] = len(train_returns)

        garch_result = estimate_raw_garch_forecast(train_returns, horizon_days)

        if garch_result["success"]:
            raw_row["Raw GARCH Forecast Annualized Volatility (%)"] = garch_result["forecast_annualized_volatility"]
            raw_row["Raw GARCH Forecast Log Volatility"] = garch_result["forecast_log_volatility"]
            raw_row["Raw GARCH Forecast Error"] = (
                row["Next Month Log Realized Volatility"] - garch_result["forecast_log_volatility"]
            )
            raw_row["GARCH mu"] = garch_result["mu"]
            raw_row["GARCH omega"] = garch_result["omega"]
            raw_row["GARCH alpha"] = garch_result["alpha"]
            raw_row["GARCH beta"] = garch_result["beta"]
            raw_row["GARCH alpha + beta"] = garch_result["alpha_plus_beta"]
            raw_row["GARCH log likelihood"] = garch_result["log_likelihood"]
            raw_row["GARCH AIC"] = garch_result["aic"]
            raw_row["GARCH BIC"] = garch_result["bic"]
            raw_row["GARCH convergence flag"] = garch_result["convergence_flag"]
        else:
            raw_row["Raw GARCH Error Message"] = garch_result["error_message"]
            raw_row["GARCH convergence flag"] = garch_result["convergence_flag"]

        raw_garch_rows.append(raw_row)

    raw_garch_df = pd.DataFrame(raw_garch_rows)

    # Merge raw GARCH forecasts into monthly dataset
    model_df = df_monthly.merge(
        raw_garch_df[
            [
                "Month",
                "Target Month",
                "Raw GARCH Forecast Annualized Volatility (%)",
                "Raw GARCH Forecast Log Volatility",
                "Daily Training Observations",
                "GARCH mu",
                "GARCH omega",
                "GARCH alpha",
                "GARCH beta",
                "GARCH alpha + beta",
                "GARCH convergence flag",
                "Raw GARCH Error Message",
            ]
        ],
        on=["Month", "Target Month"],
        how="left",
    )

    # --------------------------------------------------
    # STEP 2: MONTHLY FORECAST-AUGMENTATION REGRESSIONS
    # --------------------------------------------------
    print("Step 2: Running recursive monthly forecast-augmentation models...")

    y_col = "Next Month Log Realized Volatility"
    raw_garch_col = "Raw GARCH Forecast Log Volatility"
    agg_col = "Aggregate Climate Risk"
    pca_col = "PCA Climate Risk Index"

    oos_df = model_df[model_df["Sample Split"] == "Out-of-Sample"].copy()
    oos_df = oos_df.sort_values("Target Month").reset_index(drop=True)

    forecast_rows = []
    coefficient_rows = []

    for i, row in oos_df.iterrows():
        current_target_month = row["Target Month"]
        print(f"Monthly model {i + 1}/{len(oos_df)}: target month {current_target_month.date()}")

        forecast_row = {
            "Month": row["Month"],
            "Target Month": row["Target Month"],
            "Sample Split": row["Sample Split"],
            "Actual Next Month Log Realized Volatility": row[y_col],
            "Actual Next Month Annualized Realized Volatility (%)": row["Next Month Annualized Realized Volatility (%)"],
            "Aggregate Climate Risk": row[agg_col],
            "PCA Climate Risk Index": row[pca_col],
            "Raw GARCH Forecast Annualized Volatility (%)": row["Raw GARCH Forecast Annualized Volatility (%)"],
            "Raw GARCH Forecast Log Volatility": row[raw_garch_col],
            "Two-Step Benchmark Forecast Log Volatility": np.nan,
            "Two-Step Benchmark Forecast Annualized Volatility (%)": np.nan,
            "Two-Step Benchmark Forecast Error": np.nan,
            "Two-Step Aggregate CRI Forecast Log Volatility": np.nan,
            "Two-Step Aggregate CRI Forecast Annualized Volatility (%)": np.nan,
            "Two-Step Aggregate CRI Forecast Error": np.nan,
            "Two-Step PCA CRI Forecast Log Volatility": np.nan,
            "Two-Step PCA CRI Forecast Annualized Volatility (%)": np.nan,
            "Two-Step PCA CRI Forecast Error": np.nan,
            "Aggregate CRI Standardized Current Value": np.nan,
            "PCA CRI Standardized Current Value": np.nan,
            "Benchmark Training Observations": np.nan,
            "Aggregate Training Observations": np.nan,
            "PCA Training Observations": np.nan,
            "Benchmark Error Message": "",
            "Aggregate CRI Error Message": "",
            "PCA CRI Error Message": "",
        }

        actual_value = row[y_col]

        # Recursive expanding-window training data
        train_base = model_df[model_df["Target Month"] < current_target_month].copy()

        # ------------------------------
        # MODEL 1: BENCHMARK TWO-STEP GARCH
        # ------------------------------
        benchmark_result = fit_ols_and_predict(
            train_df=train_base,
            y_col=y_col,
            predictor_cols=[raw_garch_col],
            current_values={raw_garch_col: row[raw_garch_col]},
            min_train_obs=30,
        )

        forecast_row["Benchmark Training Observations"] = benchmark_result["nobs"]

        if benchmark_result["success"]:
            benchmark_forecast = benchmark_result["forecast"]
            forecast_row["Two-Step Benchmark Forecast Log Volatility"] = benchmark_forecast
            forecast_row["Two-Step Benchmark Forecast Annualized Volatility (%)"] = np.exp(benchmark_forecast)
            forecast_row["Two-Step Benchmark Forecast Error"] = actual_value - benchmark_forecast

            coeff_row = {
                "Target Month": current_target_month,
                "Model": "Two-Step Benchmark GARCH",
                "Training Observations": benchmark_result["nobs"],
                "In-sample R2": benchmark_result["r2_in_sample"],
                "Forecast": benchmark_forecast,
                "Actual": actual_value,
                "Forecast Error": actual_value - benchmark_forecast,
                "Climate Mean Used": np.nan,
                "Climate Std Used": np.nan,
                "Current Climate Std Value": np.nan,
                "Success": True,
                "Error Message": "",
            }
            coeff_row = add_coefficient_values(coeff_row, benchmark_result)
            coefficient_rows.append(coeff_row)
        else:
            forecast_row["Benchmark Error Message"] = benchmark_result["error_message"]

        # ------------------------------
        # MODEL 2: GARCH + AGGREGATE CRI
        # ------------------------------
        train_agg = train_base.dropna(subset=[y_col, raw_garch_col, agg_col]).copy()

        if len(train_agg) >= 30:
            agg_mean = train_agg[agg_col].mean()
            agg_std = train_agg[agg_col].std()

            if pd.notna(agg_std) and agg_std != 0:
                train_agg["Aggregate CRI Std"] = (train_agg[agg_col] - agg_mean) / agg_std
                current_agg_std = (row[agg_col] - agg_mean) / agg_std
                forecast_row["Aggregate CRI Standardized Current Value"] = current_agg_std

                agg_result = fit_ols_and_predict(
                    train_df=train_agg,
                    y_col=y_col,
                    predictor_cols=[raw_garch_col, "Aggregate CRI Std"],
                    current_values={
                        raw_garch_col: row[raw_garch_col],
                        "Aggregate CRI Std": current_agg_std,
                    },
                    min_train_obs=30,
                )

                forecast_row["Aggregate Training Observations"] = agg_result["nobs"]

                if agg_result["success"]:
                    agg_forecast = agg_result["forecast"]
                    forecast_row["Two-Step Aggregate CRI Forecast Log Volatility"] = agg_forecast
                    forecast_row["Two-Step Aggregate CRI Forecast Annualized Volatility (%)"] = np.exp(agg_forecast)
                    forecast_row["Two-Step Aggregate CRI Forecast Error"] = actual_value - agg_forecast

                    coeff_row = {
                        "Target Month": current_target_month,
                        "Model": "Two-Step GARCH + Aggregate CRI",
                        "Training Observations": agg_result["nobs"],
                        "In-sample R2": agg_result["r2_in_sample"],
                        "Forecast": agg_forecast,
                        "Actual": actual_value,
                        "Forecast Error": actual_value - agg_forecast,
                        "Climate Mean Used": agg_mean,
                        "Climate Std Used": agg_std,
                        "Current Climate Std Value": current_agg_std,
                        "Success": True,
                        "Error Message": "",
                    }
                    coeff_row = add_coefficient_values(coeff_row, agg_result)
                    coefficient_rows.append(coeff_row)
                else:
                    forecast_row["Aggregate CRI Error Message"] = agg_result["error_message"]
            else:
                forecast_row["Aggregate CRI Error Message"] = "Aggregate CRI standard deviation is zero or missing"
        else:
            forecast_row["Aggregate CRI Error Message"] = "Fewer than 30 usable Aggregate CRI training observations"

        # ------------------------------
        # MODEL 3: GARCH + PCA CRI
        # ------------------------------
        train_pca = train_base.dropna(subset=[y_col, raw_garch_col, pca_col]).copy()

        if len(train_pca) >= 30:
            pca_mean = train_pca[pca_col].mean()
            pca_std = train_pca[pca_col].std()

            if pd.notna(pca_std) and pca_std != 0:
                train_pca["PCA CRI Std"] = (train_pca[pca_col] - pca_mean) / pca_std
                current_pca_std = (row[pca_col] - pca_mean) / pca_std
                forecast_row["PCA CRI Standardized Current Value"] = current_pca_std

                pca_result = fit_ols_and_predict(
                    train_df=train_pca,
                    y_col=y_col,
                    predictor_cols=[raw_garch_col, "PCA CRI Std"],
                    current_values={
                        raw_garch_col: row[raw_garch_col],
                        "PCA CRI Std": current_pca_std,
                    },
                    min_train_obs=30,
                )

                forecast_row["PCA Training Observations"] = pca_result["nobs"]

                if pca_result["success"]:
                    pca_forecast = pca_result["forecast"]
                    forecast_row["Two-Step PCA CRI Forecast Log Volatility"] = pca_forecast
                    forecast_row["Two-Step PCA CRI Forecast Annualized Volatility (%)"] = np.exp(pca_forecast)
                    forecast_row["Two-Step PCA CRI Forecast Error"] = actual_value - pca_forecast

                    coeff_row = {
                        "Target Month": current_target_month,
                        "Model": "Two-Step GARCH + PCA CRI",
                        "Training Observations": pca_result["nobs"],
                        "In-sample R2": pca_result["r2_in_sample"],
                        "Forecast": pca_forecast,
                        "Actual": actual_value,
                        "Forecast Error": actual_value - pca_forecast,
                        "Climate Mean Used": pca_mean,
                        "Climate Std Used": pca_std,
                        "Current Climate Std Value": current_pca_std,
                        "Success": True,
                        "Error Message": "",
                    }
                    coeff_row = add_coefficient_values(coeff_row, pca_result)
                    coefficient_rows.append(coeff_row)
                else:
                    forecast_row["PCA CRI Error Message"] = pca_result["error_message"]
            else:
                forecast_row["PCA CRI Error Message"] = "PCA CRI standard deviation is zero or missing"
        else:
            forecast_row["PCA CRI Error Message"] = "Fewer than 30 usable PCA CRI training observations"

        forecast_rows.append(forecast_row)

    forecasts_df = pd.DataFrame(forecast_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------
    print("Calculating performance metrics...")

    actual_col = "Actual Next Month Log Realized Volatility"
    benchmark_forecast_col = "Two-Step Benchmark Forecast Log Volatility"
    agg_forecast_col = "Two-Step Aggregate CRI Forecast Log Volatility"
    pca_forecast_col = "Two-Step PCA CRI Forecast Log Volatility"

    performance_rows = []

    # Benchmark performance
    benchmark_df = forecasts_df.dropna(subset=[actual_col, benchmark_forecast_col]).copy()

    if len(benchmark_df) > 0:
        _, benchmark_msfe, benchmark_rmse, benchmark_mae = calculate_performance_metrics(
            benchmark_df[actual_col],
            benchmark_df[benchmark_forecast_col],
        )
    else:
        benchmark_msfe = benchmark_rmse = benchmark_mae = np.nan

    performance_rows.append(
        {
            "Model name": "Two-Step Benchmark GARCH",
            "Number of OOS observations": len(benchmark_df),
            "MSE / MSFE": benchmark_msfe,
            "RMSE": benchmark_rmse,
            "MAE": benchmark_mae,
            "R2_OS": np.nan,
            "R2_OS (%)": np.nan,
            "Clark-West statistic": np.nan,
            "Clark-West p-value": np.nan,
            "Benchmark MSFE used for R2_OS": np.nan,
            "Same-row benchmark observation count": np.nan,
        }
    )

    # Aggregate CRI performance
    agg_df = forecasts_df.dropna(
        subset=[actual_col, benchmark_forecast_col, agg_forecast_col]
    ).copy()

    if len(agg_df) > 0:
        _, agg_msfe, agg_rmse, agg_mae = calculate_performance_metrics(
            agg_df[actual_col],
            agg_df[agg_forecast_col],
        )
        _, agg_benchmark_msfe, _, _ = calculate_performance_metrics(
            agg_df[actual_col],
            agg_df[benchmark_forecast_col],
        )
        agg_r2_os = 1 - (agg_msfe / agg_benchmark_msfe) if agg_benchmark_msfe != 0 else np.nan
        agg_cw_stat, agg_cw_p = clark_west_test(
            agg_df[actual_col],
            agg_df[benchmark_forecast_col],
            agg_df[agg_forecast_col],
        )
    else:
        agg_msfe = agg_rmse = agg_mae = agg_r2_os = agg_cw_stat = agg_cw_p = np.nan
        agg_benchmark_msfe = np.nan

    performance_rows.append(
        {
            "Model name": "Two-Step GARCH + Aggregate CRI",
            "Number of OOS observations": len(agg_df),
            "MSE / MSFE": agg_msfe,
            "RMSE": agg_rmse,
            "MAE": agg_mae,
            "R2_OS": agg_r2_os,
            "R2_OS (%)": agg_r2_os * 100 if pd.notna(agg_r2_os) else np.nan,
            "Clark-West statistic": agg_cw_stat,
            "Clark-West p-value": agg_cw_p,
            "Benchmark MSFE used for R2_OS": agg_benchmark_msfe,
            "Same-row benchmark observation count": len(agg_df),
        }
    )

    # PCA CRI performance
    pca_df = forecasts_df.dropna(
        subset=[actual_col, benchmark_forecast_col, pca_forecast_col]
    ).copy()

    if len(pca_df) > 0:
        _, pca_msfe, pca_rmse, pca_mae = calculate_performance_metrics(
            pca_df[actual_col],
            pca_df[pca_forecast_col],
        )
        _, pca_benchmark_msfe, _, _ = calculate_performance_metrics(
            pca_df[actual_col],
            pca_df[benchmark_forecast_col],
        )
        pca_r2_os = 1 - (pca_msfe / pca_benchmark_msfe) if pca_benchmark_msfe != 0 else np.nan
        pca_cw_stat, pca_cw_p = clark_west_test(
            pca_df[actual_col],
            pca_df[benchmark_forecast_col],
            pca_df[pca_forecast_col],
        )
    else:
        pca_msfe = pca_rmse = pca_mae = pca_r2_os = pca_cw_stat = pca_cw_p = np.nan
        pca_benchmark_msfe = np.nan

    performance_rows.append(
        {
            "Model name": "Two-Step GARCH + PCA CRI",
            "Number of OOS observations": len(pca_df),
            "MSE / MSFE": pca_msfe,
            "RMSE": pca_rmse,
            "MAE": pca_mae,
            "R2_OS": pca_r2_os,
            "R2_OS (%)": pca_r2_os * 100 if pd.notna(pca_r2_os) else np.nan,
            "Clark-West statistic": pca_cw_stat,
            "Clark-West p-value": pca_cw_p,
            "Benchmark MSFE used for R2_OS": pca_benchmark_msfe,
            "Same-row benchmark observation count": len(pca_df),
        }
    )

    performance_df = pd.DataFrame(performance_rows)

    # --------------------------------------------------
    # SAVE OUTPUT FILES
    # --------------------------------------------------
    print("Saving output files...")

    raw_garch_df.to_excel(raw_garch_output_file, index=False)

    with pd.ExcelWriter(forecasts_output_file, engine="openpyxl") as writer:
        forecasts_df.to_excel(writer, sheet_name="OOS_Forecasts", index=False)

    performance_df.to_excel(performance_output_file, index=False)
    coefficients_df.to_excel(coefficients_output_file, index=False)

    # --------------------------------------------------
    # SUMMARY TEXT
    # --------------------------------------------------
    if len(performance_df) > 0 and performance_df["MSE / MSFE"].notna().any():
        lowest_msfe_model = performance_df.loc[
            performance_df["MSE / MSFE"].idxmin(), "Model name"
        ]
        lowest_rmse_model = performance_df.loc[
            performance_df["RMSE"].idxmin(), "Model name"
        ]
    else:
        lowest_msfe_model = np.nan
        lowest_rmse_model = np.nan

    summary_text = f"""
TWO-STEP GARCH FORECAST-AUGMENTATION MODEL RESULTS
==================================================

Input files used:
1. {daily_file}
2. {monthly_file}

Output files created:
1. {raw_garch_output_file}
2. {forecasts_output_file}
3. {performance_output_file}
4. {coefficients_output_file}
5. {summary_output_file}

Model structure:

Step 1:
Estimate benchmark GARCH(1,1) using daily S&P 500 returns.

Raw GARCH forecast:
Daily GARCH variances are forecast for each trading day of the target month.
These daily variance forecasts are summed to get target-month forecast variance.
The square root gives monthly forecast volatility.
This is annualized by multiplying by sqrt(12).
Then log volatility is used in the monthly forecasting regression.

Step 2:
Run recursive monthly forecast regressions.

Model 1: Two-Step Benchmark GARCH
Next Month Log Realized Volatility = alpha + beta × Raw GARCH Forecast Log Volatility

Model 2: Two-Step GARCH + Aggregate CRI
Next Month Log Realized Volatility = alpha + beta × Raw GARCH Forecast Log Volatility + gamma × Aggregate CRI

Model 3: Two-Step GARCH + PCA CRI
Next Month Log Realized Volatility = alpha + beta × Raw GARCH Forecast Log Volatility + gamma × PCA CRI

Important methodology note:
- This is not Paper-1-style GARCH-CRI.
- Climate risk does not enter the daily GARCH variance equation.
- Instead, climate risk is used as an additional monthly predictor after the raw GARCH forecast has been created.
- This model tests whether climate risk adds incremental predictive information beyond benchmark GARCH.

Daily return unit:
- Daily Return (%) is used.
- Therefore GARCH variance forecasts are in percent-squared units.
- No extra multiplication by 100 is applied.

OOS period:
- OOS start target month: {forecasts_df['Target Month'].min() if len(forecasts_df) > 0 else np.nan}
- OOS end target month: {forecasts_df['Target Month'].max() if len(forecasts_df) > 0 else np.nan}
- OOS forecast rows: {len(forecasts_df)}

Successful OOS forecasts:
- Two-Step Benchmark GARCH: {forecasts_df[benchmark_forecast_col].notna().sum() if len(forecasts_df) > 0 else 0}
- Two-Step GARCH + Aggregate CRI: {forecasts_df[agg_forecast_col].notna().sum() if len(forecasts_df) > 0 else 0}
- Two-Step GARCH + PCA CRI: {forecasts_df[pca_forecast_col].notna().sum() if len(forecasts_df) > 0 else 0}

Performance table:
{performance_df.to_string(index=False)}

Lowest error models:
- Lowest MSFE: {lowest_msfe_model}
- Lowest RMSE: {lowest_rmse_model}

Interpretation guide:
- Positive R2_OS means the climate-augmented model improves over the two-step benchmark GARCH model.
- Negative R2_OS means the climate-augmented model performs worse than the two-step benchmark GARCH model.
- Clark-West p-value below 0.05 suggests statistically significant improvement over benchmark.
"""

    with open(summary_output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print("Two-step GARCH model completed successfully.")


if __name__ == "__main__":
    main()