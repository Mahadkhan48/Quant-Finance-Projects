# file: 05_GARCH_CRI_Diagnostics.py

import numpy as np
import pandas as pd
from pathlib import Path


def safe_read_excel(file_path, sheet_name=None):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if sheet_name is None:
        return pd.read_excel(file_path)

    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        return pd.read_excel(file_path)


def calculate_model_summary(df, model_name, forecast_log_col, forecast_vol_col=None):
    actual_log_col = "Actual Next Month Log Realized Volatility"
    actual_vol_col = "Actual Next Month Annualized Realized Volatility (%)"

    temp = df.dropna(subset=[actual_log_col, forecast_log_col]).copy()

    if len(temp) == 0:
        return {
            "Model": model_name,
            "N": 0,
            "Mean Actual Log Vol": np.nan,
            "Mean Forecast Log Vol": np.nan,
            "Forecast Bias Log": np.nan,
            "MSE / MSFE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Overprediction Count Log": np.nan,
            "Underprediction Count Log": np.nan,
            "Overprediction % Log": np.nan,
            "Mean Actual Annualized Vol (%)": np.nan,
            "Mean Forecast Annualized Vol (%)": np.nan,
            "Forecast Bias Annualized Vol (%)": np.nan,
        }

    actual = temp[actual_log_col]
    forecast = temp[forecast_log_col]

    error = actual - forecast
    forecast_bias = forecast - actual

    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    over_count = (forecast_bias > 0).sum()
    under_count = (forecast_bias < 0).sum()

    summary = {
        "Model": model_name,
        "N": len(temp),
        "Mean Actual Log Vol": actual.mean(),
        "Mean Forecast Log Vol": forecast.mean(),
        "Forecast Bias Log": forecast_bias.mean(),
        "MSE / MSFE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Overprediction Count Log": over_count,
        "Underprediction Count Log": under_count,
        "Overprediction % Log": over_count / len(temp) * 100,
        "Mean Actual Annualized Vol (%)": np.nan,
        "Mean Forecast Annualized Vol (%)": np.nan,
        "Forecast Bias Annualized Vol (%)": np.nan,
    }

    if forecast_vol_col is not None and forecast_vol_col in temp.columns and actual_vol_col in temp.columns:
        vol_temp = temp.dropna(subset=[actual_vol_col, forecast_vol_col]).copy()
        if len(vol_temp) > 0:
            summary["Mean Actual Annualized Vol (%)"] = vol_temp[actual_vol_col].mean()
            summary["Mean Forecast Annualized Vol (%)"] = vol_temp[forecast_vol_col].mean()
            summary["Forecast Bias Annualized Vol (%)"] = (
                vol_temp[forecast_vol_col] - vol_temp[actual_vol_col]
            ).mean()

    return summary


def calculate_extended_comparison(df, model_name, extended_col, benchmark_col):
    actual_col = "Actual Next Month Log Realized Volatility"

    temp = df.dropna(subset=[actual_col, benchmark_col, extended_col]).copy()

    if len(temp) == 0:
        return {
            "Model": model_name,
            "Same-row N": 0,
            "Benchmark MSFE Same Rows": np.nan,
            "Extended MSFE": np.nan,
            "R2_OS": np.nan,
            "R2_OS (%)": np.nan,
            "Mean Delta SE": np.nan,
            "Months Extended Better": np.nan,
            "Months Benchmark Better": np.nan,
            "Extended Better Hit Rate (%)": np.nan,
        }

    actual = temp[actual_col]
    benchmark = temp[benchmark_col]
    extended = temp[extended_col]

    benchmark_se = (actual - benchmark) ** 2
    extended_se = (actual - extended) ** 2

    benchmark_msfe = benchmark_se.mean()
    extended_msfe = extended_se.mean()

    r2_os = 1 - (extended_msfe / benchmark_msfe) if benchmark_msfe != 0 else np.nan

    delta_se = extended_se - benchmark_se

    months_extended_better = (delta_se < 0).sum()
    months_benchmark_better = (delta_se > 0).sum()

    return {
        "Model": model_name,
        "Same-row N": len(temp),
        "Benchmark MSFE Same Rows": benchmark_msfe,
        "Extended MSFE": extended_msfe,
        "R2_OS": r2_os,
        "R2_OS (%)": r2_os * 100 if pd.notna(r2_os) else np.nan,
        "Mean Delta SE": delta_se.mean(),
        "Months Extended Better": months_extended_better,
        "Months Benchmark Better": months_benchmark_better,
        "Extended Better Hit Rate (%)": months_extended_better / len(temp) * 100,
    }


def create_gamma_summary(coefficients_df):
    if "Model" not in coefficients_df.columns or "gamma" not in coefficients_df.columns:
        return pd.DataFrame()

    rows = []

    for model_name in ["GARCH-CRI Aggregate", "GARCH-CRI PCA"]:
        temp = coefficients_df[coefficients_df["Model"] == model_name].copy()

        if len(temp) == 0:
            continue

        gamma = pd.to_numeric(temp["gamma"], errors="coerce")

        row = {
            "Model": model_name,
            "N": len(temp),
            "Gamma Mean": gamma.mean(),
            "Gamma Median": gamma.median(),
            "Gamma Std": gamma.std(),
            "Gamma Min": gamma.min(),
            "Gamma Max": gamma.max(),
            "Gamma Positive Count": (gamma > 0).sum(),
            "Gamma Negative Count": (gamma < 0).sum(),
            "Gamma Positive %": (gamma > 0).sum() / gamma.notna().sum() * 100 if gamma.notna().sum() > 0 else np.nan,
        }

        if "alpha + beta" in temp.columns:
            alpha_beta = pd.to_numeric(temp["alpha + beta"], errors="coerce")
            row["Alpha + Beta Mean"] = alpha_beta.mean()
            row["Alpha + Beta Min"] = alpha_beta.min()
            row["Alpha + Beta Max"] = alpha_beta.max()

        if "success" in temp.columns:
            row["Optimizer Success Count"] = temp["success"].sum()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    forecasts_file = base_path / "GARCH_CRI_Model_Forecasts.xlsx"
    coefficients_file = base_path / "GARCH_CRI_Model_Coefficients.xlsx"
    performance_file = base_path / "GARCH_CRI_Model_Performance.xlsx"

    diagnostic_output_file = base_path / "GARCH_CRI_Diagnostics.xlsx"
    diagnostic_summary_file = base_path / "GARCH_CRI_Diagnostics_Summary.txt"

    print("Reading GARCH-CRI output files...")

    forecasts_df = safe_read_excel(forecasts_file, sheet_name="OOS_Forecasts")
    coefficients_df = safe_read_excel(coefficients_file)
    performance_df = safe_read_excel(performance_file)

    forecasts_df.columns = [str(col).strip() for col in forecasts_df.columns]
    coefficients_df.columns = [str(col).strip() for col in coefficients_df.columns]
    performance_df.columns = [str(col).strip() for col in performance_df.columns]

    if "Target Month" in forecasts_df.columns:
        forecasts_df["Target Month"] = pd.to_datetime(forecasts_df["Target Month"], errors="coerce")

    if "Target Month" in coefficients_df.columns:
        coefficients_df["Target Month"] = pd.to_datetime(coefficients_df["Target Month"], errors="coerce")

    # --------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------
    actual_log_col = "Actual Next Month Log Realized Volatility"
    actual_vol_col = "Actual Next Month Annualized Realized Volatility (%)"

    benchmark_log_col = "Benchmark GARCH Forecast Log Volatility"
    agg_log_col = "GARCH-CRI Aggregate Forecast Log Volatility"
    pca_log_col = "GARCH-CRI PCA Forecast Log Volatility"

    benchmark_vol_col = "Benchmark GARCH Forecast Annualized Volatility (%)"
    agg_vol_col = "GARCH-CRI Aggregate Forecast Annualized Volatility (%)"
    pca_vol_col = "GARCH-CRI PCA Forecast Annualized Volatility (%)"

    required_cols = [
        actual_log_col,
        benchmark_log_col,
        agg_log_col,
        pca_log_col,
    ]

    missing = [col for col in required_cols if col not in forecasts_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in forecast file: {missing}")

    # --------------------------------------------------
    # CREATE DETAILED ERROR TABLE
    # --------------------------------------------------
    print("Creating detailed forecast-error diagnostics...")

    detail = forecasts_df.copy()

    models = {
        "Benchmark": benchmark_log_col,
        "Aggregate": agg_log_col,
        "PCA": pca_log_col,
    }

    for model_name, forecast_col in models.items():
        detail[f"{model_name} Error"] = detail[actual_log_col] - detail[forecast_col]
        detail[f"{model_name} Forecast Bias"] = detail[forecast_col] - detail[actual_log_col]
        detail[f"{model_name} Squared Error"] = detail[f"{model_name} Error"] ** 2
        detail[f"{model_name} Absolute Error"] = detail[f"{model_name} Error"].abs()

    detail["Aggregate SE minus Benchmark SE"] = (
        detail["Aggregate Squared Error"] - detail["Benchmark Squared Error"]
    )
    detail["PCA SE minus Benchmark SE"] = (
        detail["PCA Squared Error"] - detail["Benchmark Squared Error"]
    )

    detail["Aggregate Better Than Benchmark"] = detail["Aggregate SE minus Benchmark SE"] < 0
    detail["PCA Better Than Benchmark"] = detail["PCA SE minus Benchmark SE"] < 0

    if benchmark_vol_col in detail.columns and agg_vol_col in detail.columns:
        detail["Aggregate Forecast Vol minus Benchmark Vol"] = (
            detail[agg_vol_col] - detail[benchmark_vol_col]
        )

    if benchmark_vol_col in detail.columns and pca_vol_col in detail.columns:
        detail["PCA Forecast Vol minus Benchmark Vol"] = (
            detail[pca_vol_col] - detail[benchmark_vol_col]
        )

    # --------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------
    print("Creating model bias summary...")

    model_summary_rows = [
        calculate_model_summary(
            detail,
            "Benchmark GARCH",
            benchmark_log_col,
            benchmark_vol_col if benchmark_vol_col in detail.columns else None,
        ),
        calculate_model_summary(
            detail,
            "GARCH-CRI Aggregate",
            agg_log_col,
            agg_vol_col if agg_vol_col in detail.columns else None,
        ),
        calculate_model_summary(
            detail,
            "GARCH-CRI PCA",
            pca_log_col,
            pca_vol_col if pca_vol_col in detail.columns else None,
        ),
    ]

    model_summary = pd.DataFrame(model_summary_rows)

    # --------------------------------------------------
    # EXTENDED MODEL COMPARISON
    # --------------------------------------------------
    extended_comparison = pd.DataFrame(
        [
            calculate_extended_comparison(
                detail,
                "GARCH-CRI Aggregate",
                agg_log_col,
                benchmark_log_col,
            ),
            calculate_extended_comparison(
                detail,
                "GARCH-CRI PCA",
                pca_log_col,
                benchmark_log_col,
            ),
        ]
    )

    # --------------------------------------------------
    # TOP WORSE / BETTER MONTHS
    # --------------------------------------------------
    top_worse_agg = detail.sort_values(
        "Aggregate SE minus Benchmark SE", ascending=False
    ).head(10)

    top_better_agg = detail.sort_values(
        "Aggregate SE minus Benchmark SE", ascending=True
    ).head(10)

    top_worse_pca = detail.sort_values(
        "PCA SE minus Benchmark SE", ascending=False
    ).head(10)

    top_better_pca = detail.sort_values(
        "PCA SE minus Benchmark SE", ascending=True
    ).head(10)

    # --------------------------------------------------
    # GAMMA COEFFICIENT DIAGNOSTICS
    # --------------------------------------------------
    print("Creating gamma coefficient diagnostics...")

    gamma_summary = create_gamma_summary(coefficients_df)

    gamma_by_month_cols = [
        "Target Month",
        "Model",
        "gamma",
        "alpha",
        "beta",
        "alpha + beta",
        "success",
        "message",
    ]

    available_gamma_cols = [col for col in gamma_by_month_cols if col in coefficients_df.columns]

    gamma_by_month = coefficients_df[
        coefficients_df["Model"].isin(["GARCH-CRI Aggregate", "GARCH-CRI PCA"])
    ][available_gamma_cols].copy()

    # --------------------------------------------------
    # CORRELATION DIAGNOSTICS
    # --------------------------------------------------
    correlation_rows = []

    if "Aggregate Climate Risk" in detail.columns:
        correlation_rows.append(
            {
                "Variable 1": "Aggregate Climate Risk",
                "Variable 2": "Aggregate SE minus Benchmark SE",
                "Correlation": detail["Aggregate Climate Risk"].corr(detail["Aggregate SE minus Benchmark SE"]),
            }
        )

        if "Aggregate Forecast Vol minus Benchmark Vol" in detail.columns:
            correlation_rows.append(
                {
                    "Variable 1": "Aggregate Climate Risk",
                    "Variable 2": "Aggregate Forecast Vol minus Benchmark Vol",
                    "Correlation": detail["Aggregate Climate Risk"].corr(detail["Aggregate Forecast Vol minus Benchmark Vol"]),
                }
            )

    if "PCA Climate Risk Index" in detail.columns:
        correlation_rows.append(
            {
                "Variable 1": "PCA Climate Risk Index",
                "Variable 2": "PCA SE minus Benchmark SE",
                "Correlation": detail["PCA Climate Risk Index"].corr(detail["PCA SE minus Benchmark SE"]),
            }
        )

        if "PCA Forecast Vol minus Benchmark Vol" in detail.columns:
            correlation_rows.append(
                {
                    "Variable 1": "PCA Climate Risk Index",
                    "Variable 2": "PCA Forecast Vol minus Benchmark Vol",
                    "Correlation": detail["PCA Climate Risk Index"].corr(detail["PCA Forecast Vol minus Benchmark Vol"]),
                }
            )

    correlation_diagnostics = pd.DataFrame(correlation_rows)

    # --------------------------------------------------
    # SAVE EXCEL FILE
    # --------------------------------------------------
    print("Saving diagnostic Excel file...")

    with pd.ExcelWriter(diagnostic_output_file, engine="openpyxl") as writer:
        model_summary.to_excel(writer, sheet_name="Model_Bias_Summary", index=False)
        extended_comparison.to_excel(writer, sheet_name="Extended_Comparison", index=False)
        performance_df.to_excel(writer, sheet_name="Original_Performance", index=False)
        detail.to_excel(writer, sheet_name="Forecast_Error_Detail", index=False)
        top_worse_agg.to_excel(writer, sheet_name="Top_Worse_Agg", index=False)
        top_better_agg.to_excel(writer, sheet_name="Top_Better_Agg", index=False)
        top_worse_pca.to_excel(writer, sheet_name="Top_Worse_PCA", index=False)
        top_better_pca.to_excel(writer, sheet_name="Top_Better_PCA", index=False)
        gamma_summary.to_excel(writer, sheet_name="Gamma_Summary", index=False)
        gamma_by_month.to_excel(writer, sheet_name="Gamma_By_Month", index=False)
        correlation_diagnostics.to_excel(writer, sheet_name="Correlations", index=False)

    # --------------------------------------------------
    # SAVE TEXT SUMMARY
    # --------------------------------------------------
    summary_text = f"""
GARCH-CRI DIAGNOSTIC SUMMARY

Input files:
1. {forecasts_file}
2. {coefficients_file}
3. {performance_file}

Output files:
1. {diagnostic_output_file}
2. {diagnostic_summary_file}

Purpose:
This diagnostic file checks why GARCH-CRI performed differently from Benchmark GARCH.

Main checks included:
1. Forecast bias
2. Overprediction / underprediction
3. Squared-error comparison against Benchmark GARCH
4. Top months where GARCH-CRI performed worse
5. Top months where GARCH-CRI performed better
6. Gamma coefficient stability
7. Correlation between climate risk and forecast-error deterioration

Model bias summary:
{model_summary.to_string(index=False)}

Extended comparison:
{extended_comparison.to_string(index=False)}

Gamma summary:
{gamma_summary.to_string(index=False) if len(gamma_summary) > 0 else "No gamma summary available."}

How to interpret:
- Forecast Bias Log > 0 means the model overpredicted log volatility on average.
- Forecast Bias Log < 0 means the model underpredicted log volatility on average.
- Aggregate SE minus Benchmark SE > 0 means Aggregate CRI was worse than Benchmark for that month.
- Aggregate SE minus Benchmark SE < 0 means Aggregate CRI was better than Benchmark for that month.
- PCA SE minus Benchmark SE > 0 means PCA CRI was worse than Benchmark for that month.
- PCA SE minus Benchmark SE < 0 means PCA CRI was better than Benchmark for that month.
- Positive R2_OS means the CRI model improved over Benchmark GARCH.
- Negative R2_OS means the CRI model performed worse than Benchmark GARCH.
"""

    with open(diagnostic_summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print("Diagnostic files created successfully.")


if __name__ == "__main__":
    main()