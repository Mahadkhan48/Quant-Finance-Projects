# file: 09_Run_GARCH_MIDAS_Models.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

try:
    from arch import arch_model
except ImportError:
    raise ImportError("The 'arch' package is required. Install it using: pip install arch")


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def normalize_month(x):
    return pd.Timestamp(x).to_period("M").to_timestamp()


def check_required_columns(df, required_cols, file_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_name}: {missing}")


def calculate_performance_metrics(actual, forecast):
    actual = pd.Series(actual, dtype="float64").reset_index(drop=True)
    forecast = pd.Series(forecast, dtype="float64").reset_index(drop=True)

    errors = actual - forecast
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    return mse, mse, rmse, mae


def beta_midas_weights(k_lags, omega_midas):
    k = np.arange(1, k_lags + 1, dtype=float)
    raw = (1.0 - k / (k_lags + 1.0)) ** (omega_midas - 1.0)

    if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        return None

    raw_sum = np.sum(raw)
    if raw_sum <= 0 or not np.isfinite(raw_sum):
        return None

    return raw / raw_sum


def standardize_lag_matrix(train_lag_df, current_lag_values):
    """
    Standardize using only the training sample.
    Uses one pooled mean/std across the full training lag matrix.
    """
    stacked = train_lag_df.to_numpy(dtype=float).ravel()
    stacked = stacked[np.isfinite(stacked)]

    if len(stacked) == 0:
        return None, None, None, None

    mean_val = float(np.mean(stacked))
    std_val = float(np.std(stacked, ddof=1))

    if not np.isfinite(std_val) or std_val == 0:
        return None, mean_val, std_val, None

    train_std = (train_lag_df - mean_val) / std_val
    current_std = (current_lag_values - mean_val) / std_val

    if np.any(~np.isfinite(train_std.to_numpy(dtype=float))) or np.any(~np.isfinite(current_std.to_numpy(dtype=float))):
        return None, mean_val, std_val, None

    return train_std, mean_val, std_val, current_std


def build_climate_lag_columns(df_monthly, climate_col, k_lags=12):
    """
    Training-period lags for month t:
    t-1, t-2, ..., t-12
    """
    df = df_monthly[["Month", climate_col]].copy()
    df = df.sort_values("Month").reset_index(drop=True)

    for lag in range(1, k_lags + 1):
        df[f"{climate_col} Lag {lag}"] = df[climate_col].shift(lag)

    lag_cols = [f"{climate_col} Lag {lag}" for lag in range(1, k_lags + 1)]
    return df[["Month"] + lag_cols].copy(), lag_cols


def get_forecast_climate_vector(df_monthly, origin_month, climate_col, k_lags):
    """
    Forecast-period lags for target month t+1 use origin-month information:
    t, t-1, ..., t-11
    """
    origin_month = normalize_month(origin_month)

    month_map = (
        df_monthly[["Month", climate_col]]
        .drop_duplicates(subset=["Month"])
        .set_index("Month")[climate_col]
        .to_dict()
    )

    values = []
    for j in range(k_lags):
        required_month = normalize_month(origin_month - pd.DateOffset(months=j))
        if required_month not in month_map:
            return None

        value = month_map[required_month]
        if pd.isna(value):
            return None

        values.append(float(value))

    return pd.Series(values, dtype="float64")


def estimate_simple_garch_forecast(training_returns, horizon_days):
    """
    Simple GARCH(1,1) benchmark forecast using Daily Return (%).
    Non-converged models are rejected.
    """
    result = {
        "success": False,
        "forecast_annualized_volatility": np.nan,
        "forecast_log_volatility": np.nan,
        "mu": np.nan,
        "omega": np.nan,
        "alpha": np.nan,
        "beta": np.nan,
        "alpha_plus_beta": np.nan,
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
        model = arch_model(
            training_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        fitted = model.fit(disp="off", show_warning=False)

        convergence_flag = getattr(fitted, "convergence_flag", np.nan)
        result["convergence_flag"] = convergence_flag

        if pd.notna(convergence_flag) and int(convergence_flag) != 0:
            result["success"] = False
            result["error_message"] = f"Simple GARCH did not converge. Convergence flag: {convergence_flag}"
            return result

        forecast_obj = fitted.forecast(horizon=int(horizon_days), reindex=False)
        daily_variances = forecast_obj.variance.iloc[-1].values

        monthly_variance = float(np.sum(daily_variances))
        monthly_volatility = float(np.sqrt(monthly_variance))
        annualized_volatility = float(monthly_volatility * np.sqrt(12))

        if annualized_volatility <= 0 or not np.isfinite(annualized_volatility):
            result["error_message"] = "Invalid annualized forecast volatility"
            return result

        result["success"] = True
        result["forecast_annualized_volatility"] = annualized_volatility
        result["forecast_log_volatility"] = float(np.log(annualized_volatility))
        result["mu"] = fitted.params.get("mu", np.nan)
        result["omega"] = fitted.params.get("omega", np.nan)
        result["alpha"] = fitted.params.get("alpha[1]", np.nan)
        result["beta"] = fitted.params.get("beta[1]", np.nan)
        result["alpha_plus_beta"] = (
            result["alpha"] + result["beta"]
            if pd.notna(result["alpha"]) and pd.notna(result["beta"])
            else np.nan
        )
        result["log_likelihood"] = fitted.loglikelihood
        result["aic"] = fitted.aic
        result["bic"] = fitted.bic

    except Exception as e:
        result["error_message"] = str(e)

    return result


def garch_midas_negative_log_likelihood(params, returns, tau_by_month, month_index):
    mu, m, theta, alpha, beta = params
    penalty = 1e12

    if alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return penalty

    if not np.all(np.isfinite(params)):
        return penalty

    tau = np.exp(m + theta * tau_by_month)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        return penalty

    n = len(returns)
    eps = returns - mu
    g = np.zeros(n, dtype=float)
    h = np.zeros(n, dtype=float)

    g[0] = 1.0
    h[0] = tau[month_index[0]] * g[0]

    if h[0] <= 0 or not np.isfinite(h[0]):
        return penalty

    for t in range(1, n):
        current_tau = tau[month_index[t]]
        g[t] = (1.0 - alpha - beta) + alpha * ((eps[t - 1] ** 2) / current_tau) + beta * g[t - 1]
        h[t] = current_tau * g[t]

        if g[t] <= 0 or h[t] <= 0 or not np.isfinite(g[t]) or not np.isfinite(h[t]):
            return penalty

    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (eps ** 2) / h)
    if not np.isfinite(ll):
        return penalty

    return -float(ll)


def fit_garch_midas_model(training_returns, training_months, month_tau_values, starting_values_list):
    """
    Estimate GARCH-MIDAS with scipy.optimize.minimize.
    """
    result = {
        "success": False,
        "params": None,
        "log_likelihood": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "optimizer_message": "",
        "error_message": "",
    }

    training_months = pd.Series(training_months)
    training_months = pd.to_datetime(training_months, errors="coerce").dt.to_period("M").dt.to_timestamp()

    if training_months.isna().any():
        result["error_message"] = "Training months contain invalid dates"
        return result

    month_tau_values_fixed = {}
    for k, v in month_tau_values.items():
        month_tau_values_fixed[normalize_month(k)] = v

    unique_months = pd.Index(training_months.unique()).sort_values()
    month_to_idx = {m: i for i, m in enumerate(unique_months)}

    try:
        month_index = np.array([month_to_idx[m] for m in training_months], dtype=int)
        tau_by_month = np.array([month_tau_values_fixed[m] for m in unique_months], dtype=float)
    except KeyError as e:
        result["error_message"] = f"Month mapping error in fit_garch_midas_model: {e}"
        return result

    if np.any(~np.isfinite(tau_by_month)):
        result["error_message"] = "Non-finite month_tau_values"
        return result

    returns = np.asarray(training_returns, dtype=float)

    bounds = [
        (-2.0, 2.0),    # mu
        (-10.0, 10.0),  # m
        (-5.0, 5.0),    # theta
        (1e-6, 0.50),   # alpha
        (1e-6, 0.999),  # beta
    ]

    best_opt = None
    best_fun = np.inf

    for start_vals in starting_values_list:
        try:
            opt = minimize(
                garch_midas_negative_log_likelihood,
                x0=np.array(start_vals, dtype=float),
                args=(returns, tau_by_month, month_index),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200},
            )

            if np.isfinite(opt.fun) and opt.fun < best_fun:
                best_fun = opt.fun
                best_opt = opt
        except Exception:
            continue

    if best_opt is None:
        result["error_message"] = "Optimization failed for all starting values"
        return result

    mu, m, theta, alpha, beta = best_opt.x
    log_likelihood = -best_opt.fun
    k_params = 5
    n_obs = len(returns)

    result["success"] = bool(best_opt.success)
    result["params"] = {
        "mu": float(mu),
        "m": float(m),
        "theta": float(theta),
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha + beta": float(alpha + beta),
    }
    result["log_likelihood"] = float(log_likelihood)
    result["aic"] = float(2 * k_params - 2 * log_likelihood)
    result["bic"] = float(np.log(n_obs) * k_params - 2 * log_likelihood)
    result["optimizer_message"] = str(best_opt.message)

    return result


def compute_last_state(training_returns, training_months, fitted_params, month_tau_values):
    """
    Compute last epsilon and last g from in-sample daily data using fitted GARCH-MIDAS params.
    """
    mu = fitted_params["mu"]
    alpha = fitted_params["alpha"]
    beta = fitted_params["beta"]
    m = fitted_params["m"]
    theta = fitted_params["theta"]

    training_months = pd.Series(training_months)
    training_months = pd.to_datetime(training_months, errors="coerce").dt.to_period("M").dt.to_timestamp()

    if training_months.isna().any():
        return np.nan, np.nan, "Training months contain invalid dates"

    month_tau_values_fixed = {}
    for k, v in month_tau_values.items():
        month_tau_values_fixed[normalize_month(k)] = v

    unique_months = pd.Index(training_months.unique()).sort_values()
    month_to_idx = {mth: i for i, mth in enumerate(unique_months)}

    try:
        month_index = np.array([month_to_idx[mth] for mth in training_months], dtype=int)
        tau_by_month = np.array(
            [np.exp(m + theta * month_tau_values_fixed[mth]) for mth in unique_months],
            dtype=float,
        )
    except KeyError as e:
        return np.nan, np.nan, f"Month mapping error in compute_last_state: {e}"

    returns = np.asarray(training_returns, dtype=float)
    eps = returns - mu

    g = np.zeros(len(returns), dtype=float)
    g[0] = 1.0

    for t in range(1, len(returns)):
        current_tau = tau_by_month[month_index[t]]
        g[t] = (1.0 - alpha - beta) + alpha * ((eps[t - 1] ** 2) / current_tau) + beta * g[t - 1]

        if g[t] <= 0 or not np.isfinite(g[t]):
            return np.nan, np.nan, "Invalid in-sample g recursion"

    return float(eps[-1]), float(g[-1]), ""


def forecast_garch_midas_month(fitted_params, current_midas_value, last_eps, last_g, horizon_days):
    mu = fitted_params["mu"]
    m = fitted_params["m"]
    theta = fitted_params["theta"]
    alpha = fitted_params["alpha"]
    beta = fitted_params["beta"]

    tau_forecast = np.exp(m + theta * current_midas_value)
    if tau_forecast <= 0 or not np.isfinite(tau_forecast):
        return np.nan, np.nan, "Invalid tau forecast"

    forecast_daily_variances = np.zeros(int(horizon_days), dtype=float)

    g_next = (1.0 - alpha - beta) + alpha * ((last_eps ** 2) / tau_forecast) + beta * last_g
    h_next = tau_forecast * g_next

    if g_next <= 0 or h_next <= 0 or not np.isfinite(g_next) or not np.isfinite(h_next):
        return np.nan, np.nan, "Invalid first forecast variance"

    forecast_daily_variances[0] = h_next
    g_prev = g_next

    for i in range(1, int(horizon_days)):
        g_curr = (1.0 - alpha - beta) + alpha * (forecast_daily_variances[i - 1] / tau_forecast) + beta * g_prev
        h_curr = tau_forecast * g_curr

        if g_curr <= 0 or h_curr <= 0 or not np.isfinite(g_curr) or not np.isfinite(h_curr):
            return np.nan, np.nan, f"Invalid variance at forecast step {i + 1}"

        forecast_daily_variances[i] = h_curr
        g_prev = g_curr

    monthly_variance = float(np.sum(forecast_daily_variances))
    monthly_volatility = float(np.sqrt(monthly_variance))
    annualized_volatility = float(monthly_volatility * np.sqrt(12))

    if annualized_volatility <= 0 or not np.isfinite(annualized_volatility):
        return np.nan, np.nan, "Invalid annualized forecast volatility"

    log_vol = float(np.log(annualized_volatility))
    return annualized_volatility, log_vol, ""


def make_starting_values(previous_params):
    """
    Smaller starting-value set for speed.
    Previous successful parameters are tried first.
    """
    base_values = [
        [0.00, 0.00, 0.10, 0.05, 0.90],
        [0.00, -1.00, 0.05, 0.10, 0.85],
    ]

    if previous_params is None:
        return base_values

    prev = [
        previous_params.get("mu", 0.0),
        previous_params.get("m", 0.0),
        previous_params.get("theta", 0.1),
        previous_params.get("alpha", 0.05),
        previous_params.get("beta", 0.90),
    ]

    values = [prev]
    for item in base_values:
        if not np.allclose(prev, item, equal_nan=True):
            values.append(item)
    return values


# ==================================================
# MAIN SCRIPT
# ==================================================
def main():
    print("Running GARCH-MIDAS forecast comparison models...")

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    daily_file = base_path / "GARCH_Daily_Model_Dataset.xlsx"
    monthly_file = base_path / "GARCH_Monthly_Target_Dataset.xlsx"

    forecasts_output_file = base_path / "GARCH_MIDAS_Model_Forecasts.xlsx"
    performance_output_file = base_path / "GARCH_MIDAS_Model_Performance.xlsx"
    coefficients_output_file = base_path / "GARCH_MIDAS_Model_Coefficients.xlsx"
    summary_output_file = base_path / "GARCH_MIDAS_Model_Results.txt"

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

    print("\nAvailable columns in daily file:")
    for col in df_daily.columns:
        print(f"- {col}")

    print("\nAvailable columns in monthly file:")
    for col in df_monthly.columns:
        print(f"- {col}")

    check_required_columns(
        df_daily,
        ["Date", "Month", "Daily Return (%)"],
        "GARCH_Daily_Model_Dataset.xlsx",
    )
    check_required_columns(
        df_monthly,
        [
            "Month",
            "Target Month",
            "Next Month Log Realized Volatility",
            "Next Month Annualized Realized Volatility (%)",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
            "Number of Trading Days in Target Month",
            "Sample Split",
        ],
        "GARCH_Monthly_Target_Dataset.xlsx",
    )

    df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce")
    df_daily["Month"] = pd.to_datetime(df_daily["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df_daily["Daily Return (%)"] = pd.to_numeric(df_daily["Daily Return (%)"], errors="coerce")

    df_monthly["Month"] = pd.to_datetime(df_monthly["Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df_monthly["Target Month"] = pd.to_datetime(df_monthly["Target Month"], errors="coerce").dt.to_period("M").dt.to_timestamp()

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
    df_daily = df_daily.sort_values("Date").reset_index(drop=True)

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
    df_monthly = df_monthly.sort_values("Target Month").reset_index(drop=True)

    print(f"\nNumber of daily observations: {len(df_daily)}")
    print(f"Number of monthly observations: {len(df_monthly)}")

    in_sample_df = df_monthly[df_monthly["Sample Split"].str.lower() == "in-sample"].copy()
    oos_df = df_monthly[df_monthly["Sample Split"].str.lower() == "out-of-sample"].copy()

    print(f"In-sample monthly rows: {len(in_sample_df)}")
    print(f"OOS monthly rows: {len(oos_df)}")
    print(f"OOS start target month: {oos_df['Target Month'].min() if len(oos_df) > 0 else np.nan}")
    print(f"OOS end target month: {oos_df['Target Month'].max() if len(oos_df) > 0 else np.nan}")

    # --------------------------------------------------
    # BUILD TRAINING CLIMATE LAGS
    # --------------------------------------------------
    k_lags = 12

    agg_lag_df, agg_lag_cols = build_climate_lag_columns(df_monthly, "Aggregate Climate Risk", k_lags=k_lags)
    pca_lag_df, pca_lag_cols = build_climate_lag_columns(df_monthly, "PCA Climate Risk Index", k_lags=k_lags)

    model_df = df_monthly.merge(agg_lag_df, on="Month", how="left")
    model_df = model_df.merge(pca_lag_df, on="Month", how="left")
    model_df = model_df.sort_values("Target Month").reset_index(drop=True)

    print("\nSanity checks:")
    print("- Training-period MIDAS climate lags for month t end at t-1.")
    print("- Forecast-period MIDAS climate lags for target month t+1 use climate information from origin month t back to t-11.")
    print("- Climate variables are standardized using training data only.")
    print("- No future climate information is used.")
    print("- Non-converged simple GARCH estimates are rejected.")
    print("- Non-successful GARCH-MIDAS optimizer results are rejected.")

    # --------------------------------------------------
    # SPEED IMPROVEMENTS
    # --------------------------------------------------
    omega_midas_candidates_base = [1.50, 2.00, 5.00]
    weights_cache = {omega: beta_midas_weights(k_lags, omega) for omega in omega_midas_candidates_base}

    forecast_vector_cache = {}
    for climate_col in ["Aggregate Climate Risk", "PCA Climate Risk Index"]:
        for m in df_monthly["Month"].drop_duplicates():
            forecast_vector_cache[(climate_col, normalize_month(m))] = get_forecast_climate_vector(
                df_monthly=df_monthly,
                origin_month=m,
                climate_col=climate_col,
                k_lags=k_lags,
            )

    prev_agg_params = None
    prev_pca_params = None
    prev_agg_omega = None
    prev_pca_omega = None

    # --------------------------------------------------
    # RECURSIVE OOS FORECASTS
    # --------------------------------------------------
    forecast_rows = []
    coefficient_rows = []

    for i, row in oos_df.sort_values("Target Month").reset_index(drop=True).iterrows():
        current_target_month = row["Target Month"]
        origin_month = row["Month"]
        horizon_days = int(row["Number of Trading Days in Target Month"])

        print(f"\nForecast {i + 1}/{len(oos_df)}: target month {current_target_month.date()}")

        forecast_row = {
            "Month": origin_month,
            "Target Month": current_target_month,
            "Sample Split": row["Sample Split"],
            "Actual Next Month Log Realized Volatility": row["Next Month Log Realized Volatility"],
            "Actual Next Month Annualized Realized Volatility (%)": row["Next Month Annualized Realized Volatility (%)"],

            "Simple GARCH Forecast Log Volatility": np.nan,
            "Simple GARCH Forecast Annualized Volatility (%)": np.nan,

            "GARCH-MIDAS Aggregate CRI Forecast Log Volatility": np.nan,
            "GARCH-MIDAS Aggregate CRI Forecast Annualized Volatility (%)": np.nan,

            "GARCH-MIDAS PCA CRI Forecast Log Volatility": np.nan,
            "GARCH-MIDAS PCA CRI Forecast Annualized Volatility (%)": np.nan,

            "Simple GARCH Forecast Error": np.nan,
            "Simple GARCH Squared Error": np.nan,
            "Simple GARCH Absolute Error": np.nan,
            "Simple GARCH Forecast Bias": np.nan,

            "GARCH-MIDAS Aggregate CRI Forecast Error": np.nan,
            "GARCH-MIDAS Aggregate CRI Squared Error": np.nan,
            "GARCH-MIDAS Aggregate CRI Absolute Error": np.nan,
            "GARCH-MIDAS Aggregate CRI Forecast Bias": np.nan,

            "GARCH-MIDAS PCA CRI Forecast Error": np.nan,
            "GARCH-MIDAS PCA CRI Squared Error": np.nan,
            "GARCH-MIDAS PCA CRI Absolute Error": np.nan,
            "GARCH-MIDAS PCA CRI Forecast Bias": np.nan,

            "Simple GARCH Error Message": "",
            "GARCH-MIDAS Aggregate CRI Error Message": "",
            "GARCH-MIDAS PCA CRI Error Message": "",
        }

        origin_daily = df_daily[df_daily["Month"] == origin_month].copy()
        if len(origin_daily) == 0:
            forecast_row["Simple GARCH Error Message"] = "No daily observations found for origin month"
            forecast_row["GARCH-MIDAS Aggregate CRI Error Message"] = "No daily observations found for origin month"
            forecast_row["GARCH-MIDAS PCA CRI Error Message"] = "No daily observations found for origin month"
            forecast_rows.append(forecast_row)
            continue

        last_origin_date = origin_daily["Date"].max()
        training_daily = df_daily[df_daily["Date"] <= last_origin_date].copy()

        # ------------------------------
        # MODEL 1: SIMPLE GARCH
        # ------------------------------
        simple_result = estimate_simple_garch_forecast(training_daily["Daily Return (%)"], horizon_days)

        coefficient_rows.append(
            {
                "Target Month": current_target_month,
                "Model name": "Simple GARCH(1,1)",
                "mu": simple_result["mu"],
                "m": np.nan,
                "theta": np.nan,
                "alpha": simple_result["alpha"],
                "beta": simple_result["beta"],
                "omega_midas": np.nan,
                "alpha + beta": simple_result["alpha_plus_beta"],
                "log likelihood": simple_result["log_likelihood"],
                "AIC": simple_result["aic"],
                "BIC": simple_result["bic"],
                "optimizer success status": simple_result["success"],
                "error message if any": simple_result["error_message"],
            }
        )

        if simple_result["success"]:
            forecast_row["Simple GARCH Forecast Log Volatility"] = simple_result["forecast_log_volatility"]
            forecast_row["Simple GARCH Forecast Annualized Volatility (%)"] = simple_result["forecast_annualized_volatility"]

            actual_log = row["Next Month Log Realized Volatility"]
            forecast_row["Simple GARCH Forecast Error"] = actual_log - simple_result["forecast_log_volatility"]
            forecast_row["Simple GARCH Squared Error"] = (actual_log - simple_result["forecast_log_volatility"]) ** 2
            forecast_row["Simple GARCH Absolute Error"] = abs(actual_log - simple_result["forecast_log_volatility"])
            forecast_row["Simple GARCH Forecast Bias"] = simple_result["forecast_log_volatility"] - actual_log
        else:
            forecast_row["Simple GARCH Error Message"] = simple_result["error_message"]

        # ------------------------------
        # MODEL 2: GARCH-MIDAS + AGGREGATE CRI
        # ------------------------------
        train_months_agg = model_df[model_df["Target Month"] < current_target_month].copy()
        train_months_agg = train_months_agg.dropna(subset=["Target Month", "Month"] + agg_lag_cols).copy()

        agg_error_message = ""
        agg_params_saved = {
            "mu": np.nan,
            "m": np.nan,
            "theta": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "omega_midas": np.nan,
            "alpha + beta": np.nan,
            "log likelihood": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "optimizer success status": np.nan,
            "error message if any": "",
        }

        if len(train_months_agg) < 24:
            agg_error_message = "Fewer than 24 monthly training rows with full climate lags"
        else:
            current_agg_vector = forecast_vector_cache.get(("Aggregate Climate Risk", normalize_month(origin_month)))

            if current_agg_vector is None:
                agg_error_message = "Current Aggregate CRI forecast climate vector is missing"
            else:
                agg_train_std, agg_mean, agg_std, current_agg_std = standardize_lag_matrix(
                    train_months_agg[agg_lag_cols],
                    current_agg_vector,
                )

                if agg_train_std is None:
                    agg_error_message = "Aggregate climate standardization failed"
                else:
                    best_agg_fit = None
                    best_agg_fun = np.inf
                    best_agg_omega = np.nan
                    best_agg_train_months = None

                    omega_candidates = omega_midas_candidates_base.copy()
                    if prev_agg_omega in omega_candidates:
                        omega_candidates = [prev_agg_omega] + [x for x in omega_candidates if x != prev_agg_omega]

                    for omega_midas in omega_candidates:
                        weights = weights_cache.get(omega_midas)
                        if weights is None:
                            continue

                        temp_train = train_months_agg.copy()
                        temp_train["MIDAS Climate Value"] = np.dot(
                            agg_train_std.to_numpy(dtype=float),
                            weights,
                        )

                        if np.any(~np.isfinite(temp_train["MIDAS Climate Value"].to_numpy(dtype=float))):
                            continue

                        month_tau_values = dict(zip(temp_train["Month"], temp_train["MIDAS Climate Value"]))

                        daily_train_agg = training_daily[training_daily["Month"].isin(temp_train["Month"])].copy()
                        if len(daily_train_agg) < 500:
                            continue

                        starting_values_list = make_starting_values(prev_agg_params)

                        fit_res = fit_garch_midas_model(
                            training_returns=daily_train_agg["Daily Return (%)"].to_numpy(dtype=float),
                            training_months=daily_train_agg["Month"].to_numpy(),
                            month_tau_values=month_tau_values,
                            starting_values_list=starting_values_list,
                        )

                        if fit_res["params"] is None or fit_res["success"] is not True:
                            continue

                        objective = -fit_res["log_likelihood"]
                        if np.isfinite(objective) and objective < best_agg_fun:
                            best_agg_fun = objective
                            best_agg_fit = fit_res
                            best_agg_omega = omega_midas
                            best_agg_train_months = temp_train.copy()

                    if best_agg_fit is None:
                        agg_error_message = "Aggregate GARCH-MIDAS optimization failed or no successful optimizer result"
                    else:
                        current_weights = weights_cache.get(best_agg_omega)
                        current_midas_value = float(np.dot(current_agg_std.to_numpy(dtype=float), current_weights))

                        month_tau_values_best = dict(
                            zip(
                                best_agg_train_months["Month"],
                                best_agg_train_months["MIDAS Climate Value"],
                            )
                        )

                        daily_train_best = training_daily[training_daily["Month"].isin(best_agg_train_months["Month"])].copy()

                        last_eps, last_g, state_err = compute_last_state(
                            daily_train_best["Daily Return (%)"].to_numpy(dtype=float),
                            daily_train_best["Month"].to_numpy(),
                            best_agg_fit["params"],
                            month_tau_values_best,
                        )

                        if state_err:
                            agg_error_message = state_err
                        else:
                            ann_vol, log_vol, fc_err = forecast_garch_midas_month(
                                best_agg_fit["params"],
                                current_midas_value,
                                last_eps,
                                last_g,
                                horizon_days,
                            )

                            if fc_err:
                                agg_error_message = fc_err
                            else:
                                forecast_row["GARCH-MIDAS Aggregate CRI Forecast Annualized Volatility (%)"] = ann_vol
                                forecast_row["GARCH-MIDAS Aggregate CRI Forecast Log Volatility"] = log_vol

                                actual_log = row["Next Month Log Realized Volatility"]
                                forecast_row["GARCH-MIDAS Aggregate CRI Forecast Error"] = actual_log - log_vol
                                forecast_row["GARCH-MIDAS Aggregate CRI Squared Error"] = (actual_log - log_vol) ** 2
                                forecast_row["GARCH-MIDAS Aggregate CRI Absolute Error"] = abs(actual_log - log_vol)
                                forecast_row["GARCH-MIDAS Aggregate CRI Forecast Bias"] = log_vol - actual_log

                                agg_params_saved = {
                                    "mu": best_agg_fit["params"]["mu"],
                                    "m": best_agg_fit["params"]["m"],
                                    "theta": best_agg_fit["params"]["theta"],
                                    "alpha": best_agg_fit["params"]["alpha"],
                                    "beta": best_agg_fit["params"]["beta"],
                                    "omega_midas": best_agg_omega,
                                    "alpha + beta": best_agg_fit["params"]["alpha + beta"],
                                    "log likelihood": best_agg_fit["log_likelihood"],
                                    "AIC": best_agg_fit["aic"],
                                    "BIC": best_agg_fit["bic"],
                                    "optimizer success status": best_agg_fit["success"],
                                    "error message if any": best_agg_fit["optimizer_message"],
                                }
                                prev_agg_params = best_agg_fit["params"].copy()
                                prev_agg_omega = best_agg_omega

        forecast_row["GARCH-MIDAS Aggregate CRI Error Message"] = agg_error_message
        coefficient_rows.append(
            {
                "Target Month": current_target_month,
                "Model name": "GARCH-MIDAS + Aggregate CRI",
                **agg_params_saved,
            }
        )

        # ------------------------------
        # MODEL 3: GARCH-MIDAS + PCA CRI
        # ------------------------------
        train_months_pca = model_df[model_df["Target Month"] < current_target_month].copy()
        train_months_pca = train_months_pca.dropna(subset=["Target Month", "Month"] + pca_lag_cols).copy()

        pca_error_message = ""
        pca_params_saved = {
            "mu": np.nan,
            "m": np.nan,
            "theta": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "omega_midas": np.nan,
            "alpha + beta": np.nan,
            "log likelihood": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "optimizer success status": np.nan,
            "error message if any": "",
        }

        if len(train_months_pca) < 24:
            pca_error_message = "Fewer than 24 monthly training rows with full climate lags"
        else:
            current_pca_vector = forecast_vector_cache.get(("PCA Climate Risk Index", normalize_month(origin_month)))

            if current_pca_vector is None:
                pca_error_message = "Current PCA CRI forecast climate vector is missing"
            else:
                pca_train_std, pca_mean, pca_std, current_pca_std = standardize_lag_matrix(
                    train_months_pca[pca_lag_cols],
                    current_pca_vector,
                )

                if pca_train_std is None:
                    pca_error_message = "PCA climate standardization failed"
                else:
                    best_pca_fit = None
                    best_pca_fun = np.inf
                    best_pca_omega = np.nan
                    best_pca_train_months = None

                    omega_candidates = omega_midas_candidates_base.copy()
                    if prev_pca_omega in omega_candidates:
                        omega_candidates = [prev_pca_omega] + [x for x in omega_candidates if x != prev_pca_omega]

                    for omega_midas in omega_candidates:
                        weights = weights_cache.get(omega_midas)
                        if weights is None:
                            continue

                        temp_train = train_months_pca.copy()
                        temp_train["MIDAS Climate Value"] = np.dot(
                            pca_train_std.to_numpy(dtype=float),
                            weights,
                        )

                        if np.any(~np.isfinite(temp_train["MIDAS Climate Value"].to_numpy(dtype=float))):
                            continue

                        month_tau_values = dict(zip(temp_train["Month"], temp_train["MIDAS Climate Value"]))

                        daily_train_pca = training_daily[training_daily["Month"].isin(temp_train["Month"])].copy()
                        if len(daily_train_pca) < 500:
                            continue

                        starting_values_list = make_starting_values(prev_pca_params)

                        fit_res = fit_garch_midas_model(
                            training_returns=daily_train_pca["Daily Return (%)"].to_numpy(dtype=float),
                            training_months=daily_train_pca["Month"].to_numpy(),
                            month_tau_values=month_tau_values,
                            starting_values_list=starting_values_list,
                        )

                        if fit_res["params"] is None or fit_res["success"] is not True:
                            continue

                        objective = -fit_res["log_likelihood"]
                        if np.isfinite(objective) and objective < best_pca_fun:
                            best_pca_fun = objective
                            best_pca_fit = fit_res
                            best_pca_omega = omega_midas
                            best_pca_train_months = temp_train.copy()

                    if best_pca_fit is None:
                        pca_error_message = "PCA GARCH-MIDAS optimization failed or no successful optimizer result"
                    else:
                        current_weights = weights_cache.get(best_pca_omega)
                        current_midas_value = float(np.dot(current_pca_std.to_numpy(dtype=float), current_weights))

                        month_tau_values_best = dict(
                            zip(
                                best_pca_train_months["Month"],
                                best_pca_train_months["MIDAS Climate Value"],
                            )
                        )

                        daily_train_best = training_daily[training_daily["Month"].isin(best_pca_train_months["Month"])].copy()

                        last_eps, last_g, state_err = compute_last_state(
                            daily_train_best["Daily Return (%)"].to_numpy(dtype=float),
                            daily_train_best["Month"].to_numpy(),
                            best_pca_fit["params"],
                            month_tau_values_best,
                        )

                        if state_err:
                            pca_error_message = state_err
                        else:
                            ann_vol, log_vol, fc_err = forecast_garch_midas_month(
                                best_pca_fit["params"],
                                current_midas_value,
                                last_eps,
                                last_g,
                                horizon_days,
                            )

                            if fc_err:
                                pca_error_message = fc_err
                            else:
                                forecast_row["GARCH-MIDAS PCA CRI Forecast Annualized Volatility (%)"] = ann_vol
                                forecast_row["GARCH-MIDAS PCA CRI Forecast Log Volatility"] = log_vol

                                actual_log = row["Next Month Log Realized Volatility"]
                                forecast_row["GARCH-MIDAS PCA CRI Forecast Error"] = actual_log - log_vol
                                forecast_row["GARCH-MIDAS PCA CRI Squared Error"] = (actual_log - log_vol) ** 2
                                forecast_row["GARCH-MIDAS PCA CRI Absolute Error"] = abs(actual_log - log_vol)
                                forecast_row["GARCH-MIDAS PCA CRI Forecast Bias"] = log_vol - actual_log

                                pca_params_saved = {
                                    "mu": best_pca_fit["params"]["mu"],
                                    "m": best_pca_fit["params"]["m"],
                                    "theta": best_pca_fit["params"]["theta"],
                                    "alpha": best_pca_fit["params"]["alpha"],
                                    "beta": best_pca_fit["params"]["beta"],
                                    "omega_midas": best_pca_omega,
                                    "alpha + beta": best_pca_fit["params"]["alpha + beta"],
                                    "log likelihood": best_pca_fit["log_likelihood"],
                                    "AIC": best_pca_fit["aic"],
                                    "BIC": best_pca_fit["bic"],
                                    "optimizer success status": best_pca_fit["success"],
                                    "error message if any": best_pca_fit["optimizer_message"],
                                }
                                prev_pca_params = best_pca_fit["params"].copy()
                                prev_pca_omega = best_pca_omega

        forecast_row["GARCH-MIDAS PCA CRI Error Message"] = pca_error_message
        coefficient_rows.append(
            {
                "Target Month": current_target_month,
                "Model name": "GARCH-MIDAS + PCA CRI",
                **pca_params_saved,
            }
        )

        forecast_rows.append(forecast_row)

    forecasts_df = pd.DataFrame(forecast_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------
    print("\nCalculating performance metrics...")

    actual_col = "Actual Next Month Log Realized Volatility"
    bench_col = "Simple GARCH Forecast Log Volatility"
    agg_col = "GARCH-MIDAS Aggregate CRI Forecast Log Volatility"
    pca_col = "GARCH-MIDAS PCA CRI Forecast Log Volatility"

    performance_rows = []

    benchmark_df = forecasts_df.dropna(subset=[actual_col, bench_col]).copy()
    if len(benchmark_df) > 0:
        _, benchmark_msfe, benchmark_rmse, benchmark_mae = calculate_performance_metrics(
            benchmark_df[actual_col],
            benchmark_df[bench_col],
        )
    else:
        benchmark_msfe = benchmark_rmse = benchmark_mae = np.nan

    performance_rows.append(
        {
            "Model name": "Simple GARCH(1,1)",
            "Number of OOS observations": len(benchmark_df),
            "MSE / MSFE": benchmark_msfe,
            "RMSE": benchmark_rmse,
            "MAE": benchmark_mae,
            "R2_OS": np.nan,
            "R2_OS (%)": np.nan,
            "Benchmark MSFE used for R2_OS": np.nan,
            "Same-row benchmark observation count": np.nan,
        }
    )

    agg_df = forecasts_df.dropna(subset=[actual_col, bench_col, agg_col]).copy()
    if len(agg_df) > 0:
        _, agg_msfe, agg_rmse, agg_mae = calculate_performance_metrics(
            agg_df[actual_col],
            agg_df[agg_col],
        )
        _, agg_benchmark_msfe, _, _ = calculate_performance_metrics(
            agg_df[actual_col],
            agg_df[bench_col],
        )
        agg_r2_os = 1 - (agg_msfe / agg_benchmark_msfe) if agg_benchmark_msfe != 0 else np.nan
    else:
        agg_msfe = agg_rmse = agg_mae = agg_benchmark_msfe = agg_r2_os = np.nan

    performance_rows.append(
        {
            "Model name": "GARCH-MIDAS + Aggregate CRI",
            "Number of OOS observations": len(agg_df),
            "MSE / MSFE": agg_msfe,
            "RMSE": agg_rmse,
            "MAE": agg_mae,
            "R2_OS": agg_r2_os,
            "R2_OS (%)": agg_r2_os * 100 if pd.notna(agg_r2_os) else np.nan,
            "Benchmark MSFE used for R2_OS": agg_benchmark_msfe,
            "Same-row benchmark observation count": len(agg_df),
        }
    )

    pca_df = forecasts_df.dropna(subset=[actual_col, bench_col, pca_col]).copy()
    if len(pca_df) > 0:
        _, pca_msfe, pca_rmse, pca_mae = calculate_performance_metrics(
            pca_df[actual_col],
            pca_df[pca_col],
        )
        _, pca_benchmark_msfe, _, _ = calculate_performance_metrics(
            pca_df[actual_col],
            pca_df[bench_col],
        )
        pca_r2_os = 1 - (pca_msfe / pca_benchmark_msfe) if pca_benchmark_msfe != 0 else np.nan
    else:
        pca_msfe = pca_rmse = pca_mae = pca_benchmark_msfe = pca_r2_os = np.nan

    performance_rows.append(
        {
            "Model name": "GARCH-MIDAS + PCA CRI",
            "Number of OOS observations": len(pca_df),
            "MSE / MSFE": pca_msfe,
            "RMSE": pca_rmse,
            "MAE": pca_mae,
            "R2_OS": pca_r2_os,
            "R2_OS (%)": pca_r2_os * 100 if pd.notna(pca_r2_os) else np.nan,
            "Benchmark MSFE used for R2_OS": pca_benchmark_msfe,
            "Same-row benchmark observation count": len(pca_df),
        }
    )

    performance_df = pd.DataFrame(performance_rows)

    # --------------------------------------------------
    # SAVE OUTPUT FILES
    # --------------------------------------------------
    print("\nSaving output files...")

    forecasts_df.to_excel(forecasts_output_file, index=False)
    performance_df.to_excel(performance_output_file, index=False)
    coefficients_df.to_excel(coefficients_output_file, index=False)

    # --------------------------------------------------
    # SUMMARY TEXT
    # --------------------------------------------------
    successful_simple = forecasts_df["Simple GARCH Forecast Log Volatility"].notna().sum() if len(forecasts_df) > 0 else 0
    successful_agg = forecasts_df["GARCH-MIDAS Aggregate CRI Forecast Log Volatility"].notna().sum() if len(forecasts_df) > 0 else 0
    successful_pca = forecasts_df["GARCH-MIDAS PCA CRI Forecast Log Volatility"].notna().sum() if len(forecasts_df) > 0 else 0

    if len(performance_df) > 0 and performance_df["MSE / MSFE"].notna().any():
        best_msfe_model = performance_df.loc[performance_df["MSE / MSFE"].idxmin(), "Model name"]
        best_rmse_model = performance_df.loc[performance_df["RMSE"].idxmin(), "Model name"]
    else:
        best_msfe_model = np.nan
        best_rmse_model = np.nan

    summary_text = f"""
GARCH-MIDAS MODEL RESULTS
=========================

Input files used:
1. {daily_file}
2. {monthly_file}

Output files created:
1. {forecasts_output_file}
2. {performance_output_file}
3. {coefficients_output_file}
4. {summary_output_file}

Model structure:
Model 1: Simple GARCH(1,1) Benchmark
Model 2: GARCH-MIDAS + Aggregate CRI
Model 3: GARCH-MIDAS + PCA CRI

Short-run component:
g_i,t = (1 - alpha - beta) + alpha * ((r_(i-1,t) - mu)^2 / tau_t) + beta * g_(i-1,t)

Long-run component:
tau_t = exp(m + theta * MIDAS_Climate_t)

MIDAS climate term:
- Uses K = 12 monthly climate lags
- Training-period MIDAS climate lags for month t end at t-1
- Forecast-period MIDAS climate lags for target month t+1 use climate information from origin month t back to t-11
- Uses normalized one-parameter decreasing beta-style weights
- The MIDAS weighting parameter omega_midas was selected over a fixed grid of candidate values.

Important methodology notes:
- Daily S&P 500 returns are used for the short-run volatility component
- Monthly climate risk is used for the long-run MIDAS component
- Daily Return (%) is used, so daily variances are in percent-squared units
- Forecasted annualized volatility (%) is monthly volatility * sqrt(12)
- No extra multiplication by 100 is used
- Out-of-sample rows come from Sample Split = Out-of-Sample
- Climate variables are standardized using training data only
- No future climate information is used
- All forecasts use only information available at the forecast-origin month
- Non-converged simple GARCH estimates are rejected
- Non-successful GARCH-MIDAS optimizer results are rejected

OOS period:
Start target month: {oos_df['Target Month'].min() if len(oos_df) > 0 else np.nan}
End target month: {oos_df['Target Month'].max() if len(oos_df) > 0 else np.nan}
OOS row count: {len(oos_df)}

Number of successful forecasts:
- Simple GARCH(1,1): {successful_simple}
- GARCH-MIDAS + Aggregate CRI: {successful_agg}
- GARCH-MIDAS + PCA CRI: {successful_pca}

Performance table:
{performance_df.to_string(index=False)}

Best model by MSFE:
{best_msfe_model}

Best model by RMSE:
{best_rmse_model}

Interpretation guide:
- Lower MSFE, RMSE, and MAE indicate better forecast performance.
- Positive R2_OS means the GARCH-MIDAS model improves over the simple GARCH benchmark.
- Negative R2_OS means the GARCH-MIDAS model performs worse than the simple GARCH benchmark.
- Benchmark MSFE for R2_OS is always calculated on the exact same OOS rows as the extended model.
"""

    with open(summary_output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nFinal performance table:")
    print(performance_df.to_string(index=False))
    print("\nAll output files saved.")
    print("\nSpeed improvements used:")
    print("- Reduced omega_midas candidate grid")
    print("- Reduced starting values")
    print("- Previous successful parameters used first")
    print("- Previous successful omega_midas used first")
    print("- Beta weights cached")
    print("- Forecast climate vectors cached")


if __name__ == "__main__":
    main()