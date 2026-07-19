# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:50:29 2026

@author: mahad
"""

# file: 10_AR_Utility_Benefit_Analysis.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy import stats


# ==================================================
# COLUMN SETTINGS
# Manually edit these only if auto-detection fails.
# Set to None to allow auto-detection.
# ==================================================
MONTH_COL = None
TARGET_MONTH_COL = None
ACTUAL_LOG_VOL_COL = None
AR_BENCHMARK_FORECAST_COL = None
AR_AGG_FORECAST_COL = None
AR_PCA_FORECAST_COL = None


# ==================================================
# USER SETTINGS
# ==================================================
SR = 0.4
GAMMA = 2.0
HAC_MAXLAGS = 1


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def find_column(columns, manual_name, candidate_names, label):
    if manual_name is not None:
        if manual_name in columns:
            return manual_name
        raise ValueError(
            f"Manual setting for '{label}' was not found: {manual_name}\n"
            f"Available columns:\n{list(columns)}"
        )

    normalized_map = {str(col).strip().lower(): col for col in columns}

    for candidate in candidate_names:
        candidate_lower = candidate.strip().lower()
        if candidate_lower in normalized_map:
            return normalized_map[candidate_lower]

    for col in columns:
        col_lower = str(col).strip().lower()
        for candidate in candidate_names:
            if candidate.strip().lower() in col_lower:
                return col

    raise ValueError(
        f"Could not find required column for '{label}'.\n"
        f"Tried candidates: {candidate_names}\n"
        f"Available columns:\n{list(columns)}"
    )


def compute_reported_utility(rv_actual, rv_forecast, sr=0.4, gamma=2.0):
    """
    Reported Utility_t = U_t * 100
    where:
    U_t = (SR^2 / gamma) * [ sqrt(RV_actual)/sqrt(RV_forecast) - RV_actual/(2*RV_forecast) ]
    """
    rv_actual = pd.Series(rv_actual, dtype="float64")
    rv_forecast = pd.Series(rv_forecast, dtype="float64")

    utility = (sr**2 / gamma) * (
        np.sqrt(rv_actual) / np.sqrt(rv_forecast)
        - rv_actual / (2.0 * rv_forecast)
    )

    return utility * 100.0


def dm_style_test_utility_diff(diff_series, maxlags=1):
    """
    DM-style test:
    Regress utility differential on a constant using HAC/Newey-West SE.
    """
    diff_series = pd.Series(diff_series, dtype="float64").dropna().reset_index(drop=True)

    if len(diff_series) < 5:
        return np.nan, np.nan, np.nan

    X = np.ones((len(diff_series), 1))
    model = sm.OLS(diff_series, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    stat = float(model.tvalues[0])
    one_sided_p = float(1 - stats.norm.cdf(stat))
    two_sided_p = float(2 * (1 - stats.norm.cdf(abs(stat))))

    return stat, one_sided_p, two_sided_p


def interpretation_text(diff_value):
    if pd.isna(diff_value):
        return "No valid comparison available."
    if diff_value > 0:
        return "Positive utility differential: climate-augmented AR model provides higher economic utility than AR Benchmark."
    if diff_value < 0:
        return "Negative utility differential: AR Benchmark provides higher economic utility than climate-augmented AR model."
    return "No utility difference relative to AR Benchmark."


# ==================================================
# MAIN SCRIPT
# ==================================================
def main():
    print("Reading AR out-of-sample forecast file...")

    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")
    input_file = base_path / "AR_80_20_OutOfSample_Forecasts.xlsx"

    details_output_file = base_path / "AR_Utility_Benefit_Details.xlsx"
    summary_output_file = base_path / "AR_Utility_Benefit_Summary.xlsx"
    text_output_file = base_path / "AR_Utility_Benefit_Results.txt"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_excel(input_file)
    df.columns = [str(col).strip() for col in df.columns]

    print("\nAvailable columns in input file:")
    for col in df.columns:
        print(f"- {col}")

    # --------------------------------------------------
    # AUTO-DETECT REQUIRED COLUMNS
    # --------------------------------------------------
    month_col = find_column(
        df.columns,
        MONTH_COL,
        [
            "Month",
            "Forecast Origin Month",
        ],
        "Month",
    )

    target_month_col = find_column(
        df.columns,
        TARGET_MONTH_COL,
        [
            "Target Month",
        ],
        "Target Month",
    )

    actual_log_vol_col = find_column(
        df.columns,
        ACTUAL_LOG_VOL_COL,
        [
            "Actual Next Month Log Realized Volatility",
            "Actual",
        ],
        "Actual Next Month Log Realized Volatility",
    )

    ar_benchmark_col = find_column(
        df.columns,
        AR_BENCHMARK_FORECAST_COL,
        [
            "AR Benchmark Forecast",
            "AR Benchmark forecast log volatility",
            "AR benchmark forecast log volatility",
        ],
        "AR Benchmark forecast column",
    )

    ar_agg_col = find_column(
        df.columns,
        AR_AGG_FORECAST_COL,
        [
            "AR + Aggregate CRI Forecast",
            "AR + Aggregate forecast",
            "AR Aggregate Forecast",
        ],
        "AR + Aggregate CRI forecast column",
    )

    ar_pca_col = find_column(
        df.columns,
        AR_PCA_FORECAST_COL,
        [
            "AR + PCA CRI Forecast",
            "AR + PCA forecast",
            "AR PCA Forecast",
        ],
        "AR + PCA CRI forecast column",
    )

    print("\nSelected columns:")
    print(f"- Month column: {month_col}")
    print(f"- Target Month column: {target_month_col}")
    print(f"- Actual log volatility column: {actual_log_vol_col}")
    print(f"- AR Benchmark forecast column: {ar_benchmark_col}")
    print(f"- AR + Aggregate CRI forecast column: {ar_agg_col}")
    print(f"- AR + PCA CRI forecast column: {ar_pca_col}")

    # --------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------
    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    df[target_month_col] = pd.to_datetime(df[target_month_col], errors="coerce")

    numeric_cols = [actual_log_vol_col, ar_benchmark_col, ar_agg_col, ar_pca_col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[month_col, target_month_col]).copy()
    df = df.sort_values(target_month_col).reset_index(drop=True)

    details_df = df[[month_col, target_month_col, actual_log_vol_col, ar_benchmark_col, ar_agg_col, ar_pca_col]].copy()
    details_df = details_df.dropna(subset=[actual_log_vol_col, ar_benchmark_col, ar_agg_col, ar_pca_col]).copy()
    details_df = details_df.sort_values(target_month_col).reset_index(drop=True)

    if len(details_df) == 0:
        raise ValueError("No valid OOS rows remain after cleaning actual and forecast log volatility columns.")

    # --------------------------------------------------
    # CONVERT LOG VOLATILITY TO LEVELS AND RV
    # --------------------------------------------------
    details_df["Actual Volatility Level"] = np.exp(details_df[actual_log_vol_col])
    details_df["Actual RV"] = details_df["Actual Volatility Level"] ** 2

    details_df["AR Benchmark Forecast Volatility Level"] = np.exp(details_df[ar_benchmark_col])
    details_df["AR Benchmark Forecast RV"] = details_df["AR Benchmark Forecast Volatility Level"] ** 2

    details_df["AR + Aggregate CRI Forecast Volatility Level"] = np.exp(details_df[ar_agg_col])
    details_df["AR + Aggregate CRI Forecast RV"] = details_df["AR + Aggregate CRI Forecast Volatility Level"] ** 2

    details_df["AR + PCA CRI Forecast Volatility Level"] = np.exp(details_df[ar_pca_col])
    details_df["AR + PCA CRI Forecast RV"] = details_df["AR + PCA CRI Forecast Volatility Level"] ** 2

    # --------------------------------------------------
    # CALCULATE REPORTED UTILITY
    # --------------------------------------------------
    details_df["AR Benchmark Utility"] = compute_reported_utility(
        details_df["Actual RV"],
        details_df["AR Benchmark Forecast RV"],
        sr=SR,
        gamma=GAMMA,
    )

    details_df["AR + Aggregate CRI Utility"] = compute_reported_utility(
        details_df["Actual RV"],
        details_df["AR + Aggregate CRI Forecast RV"],
        sr=SR,
        gamma=GAMMA,
    )

    details_df["AR + PCA CRI Utility"] = compute_reported_utility(
        details_df["Actual RV"],
        details_df["AR + PCA CRI Forecast RV"],
        sr=SR,
        gamma=GAMMA,
    )

    details_df["Aggregate Utility Differential vs Benchmark"] = (
        details_df["AR + Aggregate CRI Utility"] - details_df["AR Benchmark Utility"]
    )
    details_df["PCA Utility Differential vs Benchmark"] = (
        details_df["AR + PCA CRI Utility"] - details_df["AR Benchmark Utility"]
    )

    # --------------------------------------------------
    # OPTIONAL REFERENCE ERRORS
    # --------------------------------------------------
    details_df["AR Benchmark Forecast Error"] = details_df[actual_log_vol_col] - details_df[ar_benchmark_col]
    details_df["AR Benchmark Squared Error"] = details_df["AR Benchmark Forecast Error"] ** 2
    details_df["AR Benchmark Absolute Error"] = details_df["AR Benchmark Forecast Error"].abs()
    details_df["AR Benchmark Forecast Bias"] = details_df[ar_benchmark_col] - details_df[actual_log_vol_col]

    details_df["AR + Aggregate CRI Forecast Error"] = details_df[actual_log_vol_col] - details_df[ar_agg_col]
    details_df["AR + Aggregate CRI Squared Error"] = details_df["AR + Aggregate CRI Forecast Error"] ** 2
    details_df["AR + Aggregate CRI Absolute Error"] = details_df["AR + Aggregate CRI Forecast Error"].abs()
    details_df["AR + Aggregate CRI Forecast Bias"] = details_df[ar_agg_col] - details_df[actual_log_vol_col]

    details_df["AR + PCA CRI Forecast Error"] = details_df[actual_log_vol_col] - details_df[ar_pca_col]
    details_df["AR + PCA CRI Squared Error"] = details_df["AR + PCA CRI Forecast Error"] ** 2
    details_df["AR + PCA CRI Absolute Error"] = details_df["AR + PCA CRI Forecast Error"].abs()
    details_df["AR + PCA CRI Forecast Bias"] = details_df[ar_pca_col] - details_df[actual_log_vol_col]

    # --------------------------------------------------
    # DM-STYLE TESTS ON UTILITY DIFFERENTIALS
    # --------------------------------------------------
    agg_avg_utility = float(details_df["AR + Aggregate CRI Utility"].mean())
    pca_avg_utility = float(details_df["AR + PCA CRI Utility"].mean())
    bench_avg_utility = float(details_df["AR Benchmark Utility"].mean())

    agg_diff = float(details_df["Aggregate Utility Differential vs Benchmark"].mean())
    pca_diff = float(details_df["PCA Utility Differential vs Benchmark"].mean())

    agg_dm_stat, agg_dm_p_one, agg_dm_p_two = dm_style_test_utility_diff(
        details_df["Aggregate Utility Differential vs Benchmark"],
        maxlags=HAC_MAXLAGS,
    )
    pca_dm_stat, pca_dm_p_one, pca_dm_p_two = dm_style_test_utility_diff(
        details_df["PCA Utility Differential vs Benchmark"],
        maxlags=HAC_MAXLAGS,
    )

    summary_rows = [
        {
            "Model name": "AR Benchmark",
            "Number of OOS observations": len(details_df),
            "Average Utility": bench_avg_utility,
            "Utility Differential vs AR Benchmark": 0.0,
            "DM Statistic vs AR Benchmark": np.nan,
            "One-sided DM p-value": np.nan,
            "Two-sided DM p-value": np.nan,
            "Interpretation": "Benchmark model.",
        },
        {
            "Model name": "AR + Aggregate CRI",
            "Number of OOS observations": len(details_df),
            "Average Utility": agg_avg_utility,
            "Utility Differential vs AR Benchmark": agg_diff,
            "DM Statistic vs AR Benchmark": agg_dm_stat,
            "One-sided DM p-value": agg_dm_p_one,
            "Two-sided DM p-value": agg_dm_p_two,
            "Interpretation": interpretation_text(agg_diff),
        },
        {
            "Model name": "AR + PCA CRI",
            "Number of OOS observations": len(details_df),
            "Average Utility": pca_avg_utility,
            "Utility Differential vs AR Benchmark": pca_diff,
            "DM Statistic vs AR Benchmark": pca_dm_stat,
            "One-sided DM p-value": pca_dm_p_one,
            "Two-sided DM p-value": pca_dm_p_two,
            "Interpretation": interpretation_text(pca_diff),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    # --------------------------------------------------
    # RENAME / ORDER DETAIL COLUMNS
    # --------------------------------------------------
    details_df = details_df.rename(
        columns={
            month_col: "Month",
            target_month_col: "Target Month",
            actual_log_vol_col: "Actual Next Month Log Realized Volatility",
            ar_benchmark_col: "AR Benchmark Forecast Log Volatility",
            ar_agg_col: "AR + Aggregate CRI Forecast Log Volatility",
            ar_pca_col: "AR + PCA CRI Forecast Log Volatility",
        }
    )

    detail_column_order = [
        "Month",
        "Target Month",
        "Actual Next Month Log Realized Volatility",
        "Actual Volatility Level",
        "Actual RV",
        "AR Benchmark Forecast Log Volatility",
        "AR Benchmark Forecast Volatility Level",
        "AR Benchmark Forecast RV",
        "AR Benchmark Utility",
        "AR + Aggregate CRI Forecast Log Volatility",
        "AR + Aggregate CRI Forecast Volatility Level",
        "AR + Aggregate CRI Forecast RV",
        "AR + Aggregate CRI Utility",
        "AR + PCA CRI Forecast Log Volatility",
        "AR + PCA CRI Forecast Volatility Level",
        "AR + PCA CRI Forecast RV",
        "AR + PCA CRI Utility",
        "Aggregate Utility Differential vs Benchmark",
        "PCA Utility Differential vs Benchmark",
        "AR Benchmark Forecast Error",
        "AR Benchmark Squared Error",
        "AR Benchmark Absolute Error",
        "AR Benchmark Forecast Bias",
        "AR + Aggregate CRI Forecast Error",
        "AR + Aggregate CRI Squared Error",
        "AR + Aggregate CRI Absolute Error",
        "AR + Aggregate CRI Forecast Bias",
        "AR + PCA CRI Forecast Error",
        "AR + PCA CRI Squared Error",
        "AR + PCA CRI Absolute Error",
        "AR + PCA CRI Forecast Bias",
    ]
    details_df = details_df[detail_column_order]

    # --------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------
    print("\nSanity checks:")
    print(f"- Number of OOS observations used: {len(details_df)}")
    print(f"- Target month start: {details_df['Target Month'].min()}")
    print(f"- Target month end: {details_df['Target Month'].max()}")
    print("- Confirmed that log volatility values were converted using exp()")
    print("- Confirmed that RV values were calculated as volatility squared")
    print("- Confirmed that utility was multiplied by 100 for reporting")
    print(f"- Confirmed that SR = {SR} and gamma = {GAMMA}")

    perfect_fit_utility = ((SR**2 / GAMMA) * (1 - 0.5)) * 100
    print(f"- Sanity benchmark: if forecast equals actual exactly, reported utility should be about {perfect_fit_utility:.2f}")

    # --------------------------------------------------
    # SAVE OUTPUT FILES
    # --------------------------------------------------
    details_df.to_excel(details_output_file, index=False)
    summary_df.to_excel(summary_output_file, index=False)

    highest_utility_model = summary_df.loc[summary_df["Average Utility"].idxmax(), "Model name"]

    text_report = f"""
AR UTILITY BENEFIT RESULTS
==========================

Input file used:
{input_file}

Utility methodology:
Reported Utility_t = U_t * 100

U_t = (SR^2 / gamma) * [ sqrt(RV_actual_t) / sqrt(RV_forecast_t) - RV_actual_t / (2 * RV_forecast_t) ]

Where:
- Actual Volatility_t = exp(Actual Log Volatility_t)
- Forecast Volatility_t = exp(Forecast Log Volatility_t)
- RV_actual_t = Actual Volatility_t^2
- RV_forecast_t = Forecast Volatility_t^2

Settings:
- SR = {SR}
- gamma = {GAMMA}
- HAC maxlags = {HAC_MAXLAGS}

OOS period:
- Start Target Month: {details_df['Target Month'].min()}
- End Target Month: {details_df['Target Month'].max()}
- Number of observations: {len(details_df)}

Selected columns:
- Month: {month_col}
- Target Month: {target_month_col}
- Actual log volatility: {actual_log_vol_col}
- AR Benchmark forecast: {ar_benchmark_col}
- AR + Aggregate CRI forecast: {ar_agg_col}
- AR + PCA CRI forecast: {ar_pca_col}

Average utilities:
- AR Benchmark: {bench_avg_utility:.6f}
- AR + Aggregate CRI: {agg_avg_utility:.6f}
- AR + PCA CRI: {pca_avg_utility:.6f}

Utility differentials:
- Aggregate vs Benchmark: {agg_diff:.6f}
- PCA vs Benchmark: {pca_diff:.6f}

DM-style tests on monthly utility differentials:
Aggregate vs Benchmark:
- DM Statistic: {agg_dm_stat:.6f}
- One-sided p-value: {agg_dm_p_one:.6f}
- Two-sided p-value: {agg_dm_p_two:.6f}

PCA vs Benchmark:
- DM Statistic: {pca_dm_stat:.6f}
- One-sided p-value: {pca_dm_p_one:.6f}
- Two-sided p-value: {pca_dm_p_two:.6f}

Highest average utility model:
- {highest_utility_model}

Interpretation guide:
- Positive utility differential means the climate-augmented AR model provides higher economic value than the AR benchmark.
- Negative utility differential means the climate-augmented AR model provides lower economic value than the AR benchmark.
- A small one-sided p-value suggests the climate-augmented AR model delivers significantly higher utility than the benchmark.
"""

    with open(text_output_file, "w", encoding="utf-8") as f:
        f.write(text_report)

    # --------------------------------------------------
    # PRINT FINAL SUMMARY
    # --------------------------------------------------
    print("\nFinal utility summary table:")
    print(summary_df.to_string(index=False))

    print("\nInterpretation:")
    print("- Positive utility differential means the climate-augmented AR model provides higher economic value than the AR benchmark.")
    print("- Negative utility differential means the climate-augmented AR model provides lower economic value than the AR benchmark.")

    print("\nAll output files saved.")


if __name__ == "__main__":
    main()