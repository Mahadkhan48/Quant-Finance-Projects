
"""
Created on Sun Jun 28 02:01:57 2026

@author: mahad
"""
# file: 04_Run_GARCH_CRI_Models.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm


# ==================================================
# BASIC HELPER FUNCTIONS
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

    e_b = actual - benchmark_forecast
    e_e = actual - extended_forecast

    cw_t = e_b ** 2 - (e_e ** 2 - (benchmark_forecast - extended_forecast) ** 2)

    X = np.ones(len(cw_t))
    model = sm.OLS(cw_t, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    t_stat = model.tvalues[0]
    p_value = 1 - norm.cdf(t_stat)

    return t_stat, p_value


def safe_log_volatility(annualized_volatility):
    if pd.isna(annualized_volatility) or annualized_volatility <= 0:
        return np.nan
    return np.log(annualized_volatility)


# ==================================================
# CUSTOM GARCH / GARCH-CRI FUNCTIONS
# ==================================================

def calculate_garch_variance_path(params, returns, exog=None):
    """
    Benchmark GARCH:
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

    GARCH-CRI:
        h_t = omega + gamma * CRI_{t-1} + alpha * eps_{t-1}^2 + beta * h_{t-1}

    returns are in percentage form.
    Therefore h_t is in percent-squared units.
    """

    returns = np.asarray(returns, dtype="float64")
    n = len(returns)

    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]

    if exog is None:
        gamma = 0.0
    else:
        gamma = params[4]
        exog = np.asarray(exog, dtype="float64")

    eps = returns - mu
    h = np.zeros(n)

    initial_variance = np.var(returns)
    if pd.isna(initial_variance) or initial_variance <= 0:
        initial_variance = 1.0

    h[0] = initial_variance

    min_variance = 1e-8

    for t in range(1, n):
        h_t = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]

        if exog is not None:
            h_t = h_t + gamma * exog[t - 1]

        if pd.isna(h_t) or h_t <= min_variance or not np.isfinite(h_t):
            return None, eps

        h[t] = h_t

    return h, eps


def negative_log_likelihood(params, returns, exog=None):
    """
    Normal log-likelihood for GARCH-type model.
    """

    omega = params[1]
    alpha = params[2]
    beta = params[3]

    if omega <= 0:
        return 1e12

    if alpha < 0 or beta < 0:
        return 1e12

    if alpha + beta >= 0.999:
        return 1e12

    h, eps = calculate_garch_variance_path(params, returns, exog)

    if h is None:
        return 1e12

    if np.any(h <= 0) or np.any(~np.isfinite(h)):
        return 1e12

    nll = 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (eps ** 2 / h))

    if pd.isna(nll) or not np.isfinite(nll):
        return 1e12

    return nll


def estimate_garch_model(returns, exog=None):
    """
    Estimate benchmark GARCH or GARCH-CRI using scipy minimize.
    """

    returns = pd.Series(returns, dtype="float64").dropna().values

    if exog is not None:
        exog = pd.Series(exog, dtype="float64").values
        valid_mask = np.isfinite(returns) & np.isfinite(exog)
        returns = returns[valid_mask]
        exog = exog[valid_mask]
    else:
        valid_mask = np.isfinite(returns)
        returns = returns[valid_mask]

    result = {
        "success": False,
        "message": "",
        "mu": np.nan,
        "omega": np.nan,
        "alpha": np.nan,
        "beta": np.nan,
        "gamma": np.nan,
        "alpha_plus_beta": np.nan,
        "log_likelihood": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "last_h": np.nan,
        "last_eps_squared": np.nan,
        "n_obs": len(returns),
    }

    if len(returns) < 500:
        result["message"] = "Fewer than 500 daily observations"
        return result

    mean_return = np.mean(returns)
    var_return = np.var(returns)

    if pd.isna(var_return) or var_return <= 0:
        var_return = 1.0

    omega_start = max(var_return * 0.05, 1e-6)

    if exog is None:
        start_values_list = [
            np.array([mean_return, omega_start, 0.05, 0.90]),
            np.array([mean_return, omega_start, 0.10, 0.85]),
            np.array([mean_return, omega_start, 0.03, 0.94]),
        ]

        bounds = [
            (None, None),      # mu
            (1e-8, None),      # omega
            (1e-8, 0.999),     # alpha
            (1e-8, 0.999),     # beta
        ]

    else:
        start_values_list = [
            np.array([mean_return, omega_start, 0.05, 0.90, 0.00]),
            np.array([mean_return, omega_start, 0.10, 0.85, 0.01]),
            np.array([mean_return, omega_start, 0.10, 0.85, -0.01]),
            np.array([mean_return, omega_start, 0.03, 0.94, 0.00]),
        ]

        bounds = [
            (None, None),      # mu
            (1e-8, None),      # omega
            (1e-8, 0.999),     # alpha
            (1e-8, 0.999),     # beta
            (-10.0, 10.0),     # gamma
        ]

    constraints = [
        {
            "type": "ineq",
            "fun": lambda p: 0.999 - p[2] - p[3],
        }
    ]

    best_fit = None
    best_nll = np.inf

    for start_values in start_values_list:
        try:
            fit = minimize(
                negative_log_likelihood,
                start_values,
                args=(returns, exog),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": 1000,
                    "ftol": 1e-8,
                    "disp": False,
                },
            )

            if fit.fun < best_nll:
                best_nll = fit.fun
                best_fit = fit

        except Exception:
            continue

    if best_fit is None:
        result["message"] = "Optimization failed completely"
        return result

    params = best_fit.x
    h, eps = calculate_garch_variance_path(params, returns, exog)

    if h is None:
        result["message"] = "Estimated variance path became invalid"
        return result

    k = len(params)
    log_likelihood = -best_nll
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(len(returns)) * k - 2 * log_likelihood

    result["success"] = bool(best_fit.success)
    result["message"] = str(best_fit.message)
    result["mu"] = params[0]
    result["omega"] = params[1]
    result["alpha"] = params[2]
    result["beta"] = params[3]
    result["gamma"] = params[4] if exog is not None else np.nan
    result["alpha_plus_beta"] = params[2] + params[3]
    result["log_likelihood"] = log_likelihood
    result["aic"] = aic
    result["bic"] = bic
    result["last_h"] = h[-1]
    result["last_eps_squared"] = eps[-1] ** 2

    return result


def forecast_garch_variances(model_result, horizon_days, exog_forecast_value=None):
    """
    Forecast daily variances for target month.

    Benchmark:
        h_{t+1} = omega + alpha * eps_t^2 + beta * h_t

    GARCH-CRI:
        h_{t+1} = omega + gamma * CRI_t + alpha * eps_t^2 + beta * h_t

    For multi-step forecasts:
        expected eps^2 is replaced by forecast h.
    """

    if not pd.notna(model_result["omega"]):
        return None, "Missing estimated parameters"

    omega = model_result["omega"]
    alpha = model_result["alpha"]
    beta = model_result["beta"]
    gamma = model_result["gamma"] if pd.notna(model_result["gamma"]) else 0.0

    last_h = model_result["last_h"]
    last_eps_squared = model_result["last_eps_squared"]

    if pd.isna(last_h) or pd.isna(last_eps_squared):
        return None, "Missing last conditional variance or residual"

    if exog_forecast_value is None or pd.isna(exog_forecast_value):
        exog_forecast_value = 0.0

    forecast_variances = []
    min_variance = 1e-8

    for step in range(int(horizon_days)):
        if step == 0:
            h_next = (
                omega
                + gamma * exog_forecast_value
                + alpha * last_eps_squared
                + beta * last_h
            )
        else:
            h_next = (
                omega
                + gamma * exog_forecast_value
                + (alpha + beta) * forecast_variances[-1]
            )

        if pd.isna(h_next) or h_next <= min_variance or not np.isfinite(h_next):
            return None, "Forecast variance became non-positive or invalid"

        forecast_variances.append(h_next)

    return np.array(forecast_variances), ""


def convert_daily_variance_forecast_to_monthly_log_vol(forecast_variances):
    """
    Daily returns are in percentage form.

    So:
    - daily variance is in percent-squared units
    - monthly variance = sum of daily forecast variances
    - monthly volatility (%) = sqrt(monthly variance)
    - annualized volatility (%) = monthly volatility * sqrt(12)
    - log volatility = log(annualized volatility %)
    """

    monthly_forecast_variance = np.sum(forecast_variances)
    monthly_forecast_volatility = np.sqrt(monthly_forecast_variance)
    annualized_forecast_volatility = monthly_forecast_volatility * np.sqrt(12)
    log_forecast_volatility = safe_log_volatility(annualized_forecast_volatility)

    return annualized_forecast_volatility, log_forecast_volatility


def standardize_current_value(history_series, current_value):
    history_series = pd.Series(history_series, dtype="float64").dropna()

    if len(history_series) < 12:
        return np.nan, np.nan, np.nan

    mean_value = history_series.mean()
    std_value = history_series.std()

    if pd.isna(std_value) or std_value == 0:
        return np.nan, mean_value, std_value

    standardized_value = (current_value - mean_value) / std_value

    return standardized_value, mean_value, std_value


# ==================================================
# MAIN SCRIPT
# ==================================================

def main():
    print("Running Paper-1-style GARCH-CRI models...")

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    daily_file = base_path / "GARCH_Daily_Model_Dataset.xlsx"
    monthly_file = base_path / "GARCH_Monthly_Target_Dataset.xlsx"

    forecasts_output_file = base_path / "GARCH_CRI_Model_Forecasts.xlsx"
    performance_output_file = base_path / "GARCH_CRI_Model_Performance.xlsx"
    coefficients_output_file = base_path / "GARCH_CRI_Model_Coefficients.xlsx"
    summary_output_file = base_path / "GARCH_CRI_Model_Results.txt"

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
    # CREATE PREDETERMINED DAILY CLIMATE VARIABLES
    # --------------------------------------------------
    print("Mapping monthly CRI to daily returns without look-ahead bias...")

    # Important:
    # For daily returns in month M, use climate risk from month M-1.
    # This means climate risk is predetermined.
    df_daily["Exog Month"] = df_daily["Month"] - pd.DateOffset(months=1)

    monthly_exog = df_monthly[
        [
            "Month",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
        ]
    ].copy()

    monthly_exog = monthly_exog.rename(
        columns={
            "Month": "Exog Month",
            "Aggregate Climate Risk": "Aggregate Climate Risk Exog",
            "PCA Climate Risk Index": "PCA Climate Risk Index Exog",
        }
    )

    df_daily = df_daily.merge(monthly_exog, on="Exog Month", how="left")

    # --------------------------------------------------
    # RUN RECURSIVE OOS FORECASTS
    # --------------------------------------------------
    print("Estimating recursive benchmark GARCH and GARCH-CRI models...")

    oos_monthly = df_monthly[df_monthly["Sample Split"] == "Out-of-Sample"].copy()
    oos_monthly = oos_monthly.sort_values("Target Month").reset_index(drop=True)

    forecast_rows = []
    coefficient_rows = []

    min_train_obs = 500

    for i, row in oos_monthly.iterrows():
        origin_month = row["Month"]
        target_month = row["Target Month"]
        horizon_days = int(row["Number of Trading Days in Target Month"])

        print(f"Processing {i + 1}/{len(oos_monthly)}: target month {target_month.date()}")

        result_row = {
            "Month": origin_month,
            "Target Month": target_month,
            "Sample Split": row["Sample Split"],
            "Actual Next Month Log Realized Volatility": row["Next Month Log Realized Volatility"],
            "Actual Next Month Annualized Realized Volatility (%)": row["Next Month Annualized Realized Volatility (%)"],
            "Aggregate Climate Risk": row["Aggregate Climate Risk"],
            "PCA Climate Risk Index": row["PCA Climate Risk Index"],
            "Number of Trading Days in Target Month": horizon_days,

            "Benchmark GARCH Forecast Annualized Volatility (%)": np.nan,
            "Benchmark GARCH Forecast Log Volatility": np.nan,

            "GARCH-CRI Aggregate Forecast Annualized Volatility (%)": np.nan,
            "GARCH-CRI Aggregate Forecast Log Volatility": np.nan,

            "GARCH-CRI PCA Forecast Annualized Volatility (%)": np.nan,
            "GARCH-CRI PCA Forecast Log Volatility": np.nan,

            "Benchmark Forecast Error": np.nan,
            "Aggregate CRI Forecast Error": np.nan,
            "PCA CRI Forecast Error": np.nan,

            "Aggregate CRI Std Forecast Value": np.nan,
            "PCA CRI Std Forecast Value": np.nan,

            "Training Observations": np.nan,

            "Benchmark Error Message": "",
            "Aggregate CRI Error Message": "",
            "PCA CRI Error Message": "",
        }

        if pd.isna(horizon_days) or horizon_days <= 0:
            msg = "Missing or invalid target-month trading days"
            result_row["Benchmark Error Message"] = msg
            result_row["Aggregate CRI Error Message"] = msg
            result_row["PCA CRI Error Message"] = msg
            forecast_rows.append(result_row)
            continue

        origin_daily = df_daily[df_daily["Month"] == origin_month].copy()

        if len(origin_daily) == 0:
            msg = "No daily returns found for forecast-origin month"
            result_row["Benchmark Error Message"] = msg
            result_row["Aggregate CRI Error Message"] = msg
            result_row["PCA CRI Error Message"] = msg
            forecast_rows.append(result_row)
            continue

        last_origin_date = origin_daily["Date"].max()

        train_daily = df_daily[df_daily["Date"] <= last_origin_date].copy()

        # Historical monthly CRI available at forecast origin.
        hist_monthly = df_monthly[df_monthly["Month"] <= origin_month].copy()

        agg_std_forecast, agg_mean, agg_std = standardize_current_value(
            hist_monthly["Aggregate Climate Risk"],
            row["Aggregate Climate Risk"],
        )

        pca_std_forecast, pca_mean, pca_std = standardize_current_value(
            hist_monthly["PCA Climate Risk Index"],
            row["PCA Climate Risk Index"],
        )

        result_row["Aggregate CRI Std Forecast Value"] = agg_std_forecast
        result_row["PCA CRI Std Forecast Value"] = pca_std_forecast

        if pd.isna(agg_std_forecast) or pd.isna(pca_std_forecast):
            msg = "Climate index standardization failed"
            result_row["Benchmark Error Message"] = msg
            result_row["Aggregate CRI Error Message"] = msg
            result_row["PCA CRI Error Message"] = msg
            forecast_rows.append(result_row)
            continue

        # Standardize daily exogenous variables using monthly history only.
        train_daily["Aggregate CRI Exog Std"] = (
            (train_daily["Aggregate Climate Risk Exog"] - agg_mean) / agg_std
        )

        train_daily["PCA CRI Exog Std"] = (
            (train_daily["PCA Climate Risk Index Exog"] - pca_mean) / pca_std
        )

        # Use a common training sample so benchmark and extended models are comparable.
        train_common = train_daily.dropna(
            subset=[
                "Daily Return (%)",
                "Aggregate CRI Exog Std",
                "PCA CRI Exog Std",
            ]
        ).copy()

        if len(train_common) < min_train_obs:
            msg = f"Fewer than {min_train_obs} usable daily training observations"
            result_row["Benchmark Error Message"] = msg
            result_row["Aggregate CRI Error Message"] = msg
            result_row["PCA CRI Error Message"] = msg
            forecast_rows.append(result_row)
            continue

        result_row["Training Observations"] = len(train_common)

        returns_train = train_common["Daily Return (%)"].values
        agg_exog_train = train_common["Aggregate CRI Exog Std"].values
        pca_exog_train = train_common["PCA CRI Exog Std"].values

        actual_value = row["Next Month Log Realized Volatility"]

        # ------------------------------
        # MODEL 1: BENCHMARK GARCH
        # ------------------------------
        try:
            benchmark_model = estimate_garch_model(returns_train, exog=None)

            if not pd.notna(benchmark_model["omega"]):
                raise ValueError(benchmark_model["message"])

            bench_vars, bench_msg = forecast_garch_variances(
                benchmark_model,
                horizon_days,
                exog_forecast_value=None,
            )

            if bench_vars is None:
                raise ValueError(bench_msg)

            bench_ann_vol, bench_log_vol = convert_daily_variance_forecast_to_monthly_log_vol(
                bench_vars
            )

            result_row["Benchmark GARCH Forecast Annualized Volatility (%)"] = bench_ann_vol
            result_row["Benchmark GARCH Forecast Log Volatility"] = bench_log_vol
            result_row["Benchmark Forecast Error"] = actual_value - bench_log_vol

            coefficient_rows.append(
                {
                    "Target Month": target_month,
                    "Model": "Benchmark GARCH",
                    "Training Observations": len(train_common),
                    "mu": benchmark_model["mu"],
                    "omega": benchmark_model["omega"],
                    "alpha": benchmark_model["alpha"],
                    "beta": benchmark_model["beta"],
                    "gamma": np.nan,
                    "alpha + beta": benchmark_model["alpha_plus_beta"],
                    "log likelihood": benchmark_model["log_likelihood"],
                    "AIC": benchmark_model["aic"],
                    "BIC": benchmark_model["bic"],
                    "success": benchmark_model["success"],
                    "message": benchmark_model["message"],
                    "Climate variable mean": np.nan,
                    "Climate variable std": np.nan,
                }
            )

        except Exception as e:
            result_row["Benchmark Error Message"] = str(e)

        # ------------------------------
        # MODEL 2: GARCH-CRI AGGREGATE
        # ------------------------------
        try:
            agg_model = estimate_garch_model(returns_train, exog=agg_exog_train)

            if not pd.notna(agg_model["omega"]):
                raise ValueError(agg_model["message"])

            agg_vars, agg_msg = forecast_garch_variances(
                agg_model,
                horizon_days,
                exog_forecast_value=agg_std_forecast,
            )

            if agg_vars is None:
                raise ValueError(agg_msg)

            agg_ann_vol, agg_log_vol = convert_daily_variance_forecast_to_monthly_log_vol(
                agg_vars
            )

            result_row["GARCH-CRI Aggregate Forecast Annualized Volatility (%)"] = agg_ann_vol
            result_row["GARCH-CRI Aggregate Forecast Log Volatility"] = agg_log_vol
            result_row["Aggregate CRI Forecast Error"] = actual_value - agg_log_vol

            coefficient_rows.append(
                {
                    "Target Month": target_month,
                    "Model": "GARCH-CRI Aggregate",
                    "Training Observations": len(train_common),
                    "mu": agg_model["mu"],
                    "omega": agg_model["omega"],
                    "alpha": agg_model["alpha"],
                    "beta": agg_model["beta"],
                    "gamma": agg_model["gamma"],
                    "alpha + beta": agg_model["alpha_plus_beta"],
                    "log likelihood": agg_model["log_likelihood"],
                    "AIC": agg_model["aic"],
                    "BIC": agg_model["bic"],
                    "success": agg_model["success"],
                    "message": agg_model["message"],
                    "Climate variable mean": agg_mean,
                    "Climate variable std": agg_std,
                }
            )

        except Exception as e:
            result_row["Aggregate CRI Error Message"] = str(e)

        # ------------------------------
        # MODEL 3: GARCH-CRI PCA
        # ------------------------------
        try:
            pca_model = estimate_garch_model(returns_train, exog=pca_exog_train)

            if not pd.notna(pca_model["omega"]):
                raise ValueError(pca_model["message"])

            pca_vars, pca_msg = forecast_garch_variances(
                pca_model,
                horizon_days,
                exog_forecast_value=pca_std_forecast,
            )

            if pca_vars is None:
                raise ValueError(pca_msg)

            pca_ann_vol, pca_log_vol = convert_daily_variance_forecast_to_monthly_log_vol(
                pca_vars
            )

            result_row["GARCH-CRI PCA Forecast Annualized Volatility (%)"] = pca_ann_vol
            result_row["GARCH-CRI PCA Forecast Log Volatility"] = pca_log_vol
            result_row["PCA CRI Forecast Error"] = actual_value - pca_log_vol

            coefficient_rows.append(
                {
                    "Target Month": target_month,
                    "Model": "GARCH-CRI PCA",
                    "Training Observations": len(train_common),
                    "mu": pca_model["mu"],
                    "omega": pca_model["omega"],
                    "alpha": pca_model["alpha"],
                    "beta": pca_model["beta"],
                    "gamma": pca_model["gamma"],
                    "alpha + beta": pca_model["alpha_plus_beta"],
                    "log likelihood": pca_model["log_likelihood"],
                    "AIC": pca_model["aic"],
                    "BIC": pca_model["bic"],
                    "success": pca_model["success"],
                    "message": pca_model["message"],
                    "Climate variable mean": pca_mean,
                    "Climate variable std": pca_std,
                }
            )

        except Exception as e:
            result_row["PCA CRI Error Message"] = str(e)

        forecast_rows.append(result_row)

    forecasts_df = pd.DataFrame(forecast_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------
    print("Calculating performance metrics...")

    performance_rows = []

    actual_col = "Actual Next Month Log Realized Volatility"
    benchmark_col = "Benchmark GARCH Forecast Log Volatility"
    agg_col = "GARCH-CRI Aggregate Forecast Log Volatility"
    pca_col = "GARCH-CRI PCA Forecast Log Volatility"

    benchmark_df = forecasts_df.dropna(subset=[actual_col, benchmark_col]).copy()

    if len(benchmark_df) > 0:
        _, benchmark_msfe, benchmark_rmse, benchmark_mae = calculate_performance_metrics(
            benchmark_df[actual_col],
            benchmark_df[benchmark_col],
        )
    else:
        benchmark_msfe = benchmark_rmse = benchmark_mae = np.nan

    performance_rows.append(
        {
            "Model name": "Benchmark GARCH",
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
    agg_df = forecasts_df.dropna(subset=[actual_col, benchmark_col, agg_col]).copy()

    if len(agg_df) > 0:
        _, agg_msfe, agg_rmse, agg_mae = calculate_performance_metrics(
            agg_df[actual_col],
            agg_df[agg_col],
        )

        _, agg_benchmark_msfe, _, _ = calculate_performance_metrics(
            agg_df[actual_col],
            agg_df[benchmark_col],
        )

        agg_r2_os = 1 - (agg_msfe / agg_benchmark_msfe) if agg_benchmark_msfe != 0 else np.nan

        agg_cw_stat, agg_cw_p = clark_west_test(
            agg_df[actual_col],
            agg_df[benchmark_col],
            agg_df[agg_col],
        )
    else:
        agg_msfe = agg_rmse = agg_mae = agg_r2_os = agg_cw_stat = agg_cw_p = np.nan
        agg_benchmark_msfe = np.nan

    performance_rows.append(
        {
            "Model name": "GARCH-CRI Aggregate",
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
    pca_df = forecasts_df.dropna(subset=[actual_col, benchmark_col, pca_col]).copy()

    if len(pca_df) > 0:
        _, pca_msfe, pca_rmse, pca_mae = calculate_performance_metrics(
            pca_df[actual_col],
            pca_df[pca_col],
        )

        _, pca_benchmark_msfe, _, _ = calculate_performance_metrics(
            pca_df[actual_col],
            pca_df[benchmark_col],
        )

        pca_r2_os = 1 - (pca_msfe / pca_benchmark_msfe) if pca_benchmark_msfe != 0 else np.nan

        pca_cw_stat, pca_cw_p = clark_west_test(
            pca_df[actual_col],
            pca_df[benchmark_col],
            pca_df[pca_col],
        )
    else:
        pca_msfe = pca_rmse = pca_mae = pca_r2_os = pca_cw_stat = pca_cw_p = np.nan
        pca_benchmark_msfe = np.nan

    performance_rows.append(
        {
            "Model name": "GARCH-CRI PCA",
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

    with pd.ExcelWriter(forecasts_output_file, engine="openpyxl") as writer:
        forecasts_df.to_excel(writer, sheet_name="OOS_Forecasts", index=False)

    performance_df.to_excel(performance_output_file, index=False)
    coefficients_df.to_excel(coefficients_output_file, index=False)

    # --------------------------------------------------
    # SUMMARY TEXT
    # --------------------------------------------------
    if len(performance_df) > 0 and performance_df["MSE / MSFE"].notna().any():
        lowest_msfe_model = performance_df.loc[performance_df["MSE / MSFE"].idxmin(), "Model name"]
        lowest_rmse_model = performance_df.loc[performance_df["RMSE"].idxmin(), "Model name"]
    else:
        lowest_msfe_model = np.nan
        lowest_rmse_model = np.nan

    summary_text = f"""
PAPER-1-STYLE GARCH-CRI MODEL RESULTS
=====================================

Input files used:
1. {daily_file}
2. {monthly_file}

Output files created:
1. {forecasts_output_file}
2. {performance_output_file}
3. {coefficients_output_file}
4. {summary_output_file}

Model structure:

Benchmark GARCH:
h_t = omega + alpha * epsilon_(t-1)^2 + beta * h_(t-1)

GARCH-CRI:
h_t = omega + gamma * CRI_(t-1) + alpha * epsilon_(t-1)^2 + beta * h_(t-1)

Important methodology note:
- This is not the two-step GARCH forecast-regression model.
- Climate risk enters directly into the conditional variance equation.
- Monthly CRI is treated as a predetermined exogenous variable.
- For daily returns in month M, the model uses CRI from month M-1.
- For forecasting target month M+1, the model uses CRI from forecast-origin month M.
- This avoids look-ahead bias.

Daily return unit:
- Daily Return (%) is used.
- Therefore daily variance forecasts are in percent-squared units.
- Monthly forecast variance = sum of forecasted daily variances.
- Monthly forecast volatility (%) = sqrt(monthly forecast variance).
- Annualized forecast volatility (%) = monthly volatility * sqrt(12).
- No extra multiplication by 100 is applied.

OOS period:
- OOS start target month: {forecasts_df['Target Month'].min() if len(forecasts_df) > 0 else np.nan}
- OOS end target month: {forecasts_df['Target Month'].max() if len(forecasts_df) > 0 else np.nan}
- OOS forecast rows: {len(forecasts_df)}

Successful forecasts:
- Benchmark GARCH: {forecasts_df[benchmark_col].notna().sum() if len(forecasts_df) > 0 else 0}
- GARCH-CRI Aggregate: {forecasts_df[agg_col].notna().sum() if len(forecasts_df) > 0 else 0}
- GARCH-CRI PCA: {forecasts_df[pca_col].notna().sum() if len(forecasts_df) > 0 else 0}

Performance table:
{performance_df.to_string(index=False)}

Lowest error models:
- Lowest MSFE: {lowest_msfe_model}
- Lowest RMSE: {lowest_rmse_model}

Interpretation guide:
- Positive R2_OS means the GARCH-CRI model improves over benchmark GARCH.
- Negative R2_OS means the GARCH-CRI model performs worse than benchmark GARCH.
- Clark-West p-value below 0.05 suggests statistically significant improvement over benchmark.
"""

    with open(summary_output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()
