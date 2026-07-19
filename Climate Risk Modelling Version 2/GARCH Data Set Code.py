# file: 02_prepare_GARCH_dataset.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA


def check_required_columns(df, required_columns, file_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_name}: {missing}")


def main():
    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------
    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    sp500_file = base_path / "S&P500_Daily_Data.xlsx"
    climate_file = base_path / "Faccini Climate Risk Data.xlsx"

    daily_output_file = base_path / "GARCH_Daily_Model_Dataset.xlsx"
    monthly_output_file = base_path / "GARCH_Monthly_Target_Dataset.xlsx"
    pca_output_file = base_path / "GARCH_Recursive_PCA_Details.xlsx"
    summary_output_file = base_path / "GARCH_Dataset_Summary.txt"

    intended_start_date = pd.Timestamp("2000-01-01")
    intended_end_date = pd.Timestamp("2025-12-31")

    if not sp500_file.exists():
        raise FileNotFoundError(f"Input file not found: {sp500_file}")

    if not climate_file.exists():
        raise FileNotFoundError(f"Input file not found: {climate_file}")

    # --------------------------------------------------
    # PART A: READ S&P 500 DAILY DATA
    # --------------------------------------------------
    print("Reading S&P 500 daily data...")

    df_sp500 = pd.read_excel(sp500_file)
    df_sp500.columns = [str(col).strip().lower() for col in df_sp500.columns]

    check_required_columns(
        df_sp500,
        ["date", "adjusted_close"],
        "S&P500_Daily_Data.xlsx"
    )

    df_sp500["date"] = pd.to_datetime(df_sp500["date"], errors="coerce")
    df_sp500["adjusted_close"] = pd.to_numeric(df_sp500["adjusted_close"], errors="coerce")

    df_sp500 = df_sp500.dropna(subset=["date", "adjusted_close"]).copy()
    df_sp500 = df_sp500.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # CALCULATE DAILY RETURNS BEFORE DATE FILTERING
    # --------------------------------------------------
    print("Calculating daily returns...")

    df_sp500["Daily Return Decimal"] = np.log(
        df_sp500["adjusted_close"] / df_sp500["adjusted_close"].shift(1)
    )
    df_sp500["Daily Return (%)"] = df_sp500["Daily Return Decimal"] * 100

    # For this dissertation, GARCH will use Daily Return (%).
    #
    # If GARCH uses Daily Return (%), then:
    # - Forecasted daily variance is in percent-squared units
    # - Monthly forecast variance = sum of forecasted daily variances
    # - Monthly forecast volatility = sqrt(monthly forecast variance)
    # - Annualized forecast volatility (%) = monthly forecast volatility * sqrt(12)
    # - Do NOT multiply by 100 again
    #
    # If GARCH uses Daily Return Decimal, then:
    # - Annualized forecast volatility (%) = monthly forecast volatility * sqrt(12) * 100

    df_sp500 = df_sp500.dropna(subset=["Daily Return Decimal", "Daily Return (%)"]).copy()

    # Now apply the broad intended window after returns are created.
    df_sp500 = df_sp500[
        (df_sp500["date"] >= intended_start_date) &
        (df_sp500["date"] <= intended_end_date)
    ].copy()

    df_sp500["Month"] = df_sp500["date"].dt.to_period("M").dt.to_timestamp()

    # --------------------------------------------------
    # PART B: MONTHLY REALIZED VOLATILITY FROM DAILY DATA
    # --------------------------------------------------
    print("Creating monthly realized volatility...")

    monthly_vol = (
        df_sp500.groupby("Month")["Daily Return Decimal"]
        .apply(lambda x: np.sqrt(np.sum(x ** 2)))
        .reset_index(name="Monthly Realized Volatility Decimal")
    )

    monthly_vol["Annualized Realized Volatility Decimal"] = (
        monthly_vol["Monthly Realized Volatility Decimal"] * np.sqrt(12)
    )
    monthly_vol["Annualized Realized Volatility (%)"] = (
        monthly_vol["Annualized Realized Volatility Decimal"] * 100
    )
    monthly_vol["Log Realized Volatility"] = np.log(
        monthly_vol["Annualized Realized Volatility (%)"]
    )

    monthly_vol = monthly_vol.sort_values("Month").reset_index(drop=True)

    # --------------------------------------------------
    # PART C: CREATE NEXT-MONTH TARGET VARIABLES
    # --------------------------------------------------
    monthly_vol["Target Month"] = monthly_vol["Month"] + pd.offsets.MonthBegin(1)
    monthly_vol["Next Month Log Realized Volatility"] = monthly_vol["Log Realized Volatility"].shift(-1)
    monthly_vol["Next Month Annualized Realized Volatility (%)"] = (
        monthly_vol["Annualized Realized Volatility (%)"].shift(-1)
    )

    # --------------------------------------------------
    # PART D: READ FACCINI CLIMATE RISK DATA
    # --------------------------------------------------
    print("Reading Faccini climate risk data...")

    df_climate = pd.read_excel(climate_file)
    df_climate.columns = [str(col).strip() for col in df_climate.columns]

    check_required_columns(
        df_climate,
        ["date", "US climate policy", "International summits", "Global warming", "Natural disasters"],
        "Faccini Climate Risk Data.xlsx"
    )

    df_climate["date"] = pd.to_datetime(df_climate["date"], errors="coerce")
    df_climate = df_climate.dropna(subset=["date"]).copy()
    df_climate = df_climate.sort_values("date").reset_index(drop=True)

    climate_vars = [
        "US climate policy",
        "International summits",
        "Global warming",
        "Natural disasters",
    ]

    for col in climate_vars:
        df_climate[col] = pd.to_numeric(df_climate[col], errors="coerce")

    df_climate["Month"] = df_climate["date"].dt.to_period("M").dt.to_timestamp()

    # --------------------------------------------------
    # CREATE MONTHLY CLIMATE VARIABLES
    # --------------------------------------------------
    print("Creating monthly climate variables...")

    monthly_climate = (
        df_climate.groupby("Month")[climate_vars]
        .mean()
        .reset_index()
        .sort_values("Month")
        .reset_index(drop=True)
    )

    for col in climate_vars:
        monthly_climate[f"log_{col}"] = np.log1p(monthly_climate[col])
        monthly_climate[f"smoothed_{col}"] = (
            monthly_climate[f"log_{col}"].rolling(window=6, min_periods=6).mean()
        )

    smoothed_cols = [f"smoothed_{col}" for col in climate_vars]

    # --------------------------------------------------
    # PART E: RECURSIVE AGGREGATE CLIMATE RISK
    # --------------------------------------------------
    print("Creating Aggregate Climate Risk...")

    min_obs = 12
    aggregate_cri_values = []

    for i in range(len(monthly_climate)):
        current_row = monthly_climate.loc[i, smoothed_cols]

        if current_row.isna().any():
            aggregate_cri_values.append(np.nan)
            continue

        hist = monthly_climate.iloc[: i + 1].dropna(subset=smoothed_cols)

        if len(hist) < min_obs:
            aggregate_cri_values.append(np.nan)
            continue

        hist_means = hist[smoothed_cols].mean()
        hist_stds = hist[smoothed_cols].std()

        if hist_stds.isna().any() or (hist_stds == 0).any():
            aggregate_cri_values.append(np.nan)
            continue

        current_standardized = (current_row - hist_means) / hist_stds
        aggregate_cri_values.append(current_standardized.mean())

    monthly_climate["Aggregate Climate Risk"] = aggregate_cri_values

    # --------------------------------------------------
    # PART F: RECURSIVE PCA CLIMATE RISK INDEX
    # --------------------------------------------------
    print("Creating recursive PCA Climate Risk Index...")

    pca_values = []
    pca_details = []

    for i in range(len(monthly_climate)):
        current_row = monthly_climate.loc[i, smoothed_cols]

        if current_row.isna().any():
            pca_values.append(np.nan)
            continue

        hist = monthly_climate.iloc[: i + 1].dropna(subset=smoothed_cols)

        if len(hist) < min_obs:
            pca_values.append(np.nan)
            continue

        hist_means = hist[smoothed_cols].mean()
        hist_stds = hist[smoothed_cols].std()

        if hist_stds.isna().any() or (hist_stds == 0).any():
            pca_values.append(np.nan)
            continue

        hist_std = (hist[smoothed_cols] - hist_means) / hist_stds

        pca = PCA(n_components=1)
        pca.fit(hist_std)

        current_std = ((current_row - hist_means) / hist_stds).values.reshape(1, -1)
        current_pca_value = pca.transform(current_std)[0, 0]

        loadings = pca.components_[0].copy()

        if loadings.mean() < 0:
            current_pca_value = -current_pca_value
            loadings = -loadings

        pca_values.append(current_pca_value)

        pca_details.append(
            {
                "Month": monthly_climate.loc[i, "Month"],
                "Number of observations used": len(hist),
                "Explained variance ratio": pca.explained_variance_ratio_[0],
                "Loading - US climate policy": loadings[0],
                "Loading - International summits": loadings[1],
                "Loading - Global warming": loadings[2],
                "Loading - Natural disasters": loadings[3],
            }
        )

    monthly_climate["PCA Climate Risk Index"] = pca_values
    df_pca_details = pd.DataFrame(pca_details)

    # --------------------------------------------------
    # PART G: MERGE MONTHLY VOLATILITY AND CLIMATE DATA
    # --------------------------------------------------
    monthly_target = pd.merge(
        monthly_vol,
        monthly_climate,
        on="Month",
        how="inner"
    )

    monthly_target = monthly_target[
        (monthly_target["Month"] >= intended_start_date) &
        (monthly_target["Target Month"] <= pd.Timestamp("2025-12-01"))
    ].copy()

    monthly_target = monthly_target.dropna(
        subset=[
            "Next Month Log Realized Volatility",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
        ]
    ).copy()

    monthly_target = monthly_target.sort_values("Target Month").reset_index(drop=True)

    # --------------------------------------------------
    # PART H: CREATE TRADING-DAY COUNTS
    # --------------------------------------------------
    print("Creating target-month trading-day counts...")

    trading_days_month = (
        df_sp500.groupby("Month")
        .size()
        .reset_index(name="Number of Trading Days in Month")
    )

    trading_days_target = trading_days_month.rename(
        columns={
            "Month": "Target Month",
            "Number of Trading Days in Month": "Number of Trading Days in Target Month",
        }
    )

    monthly_target = monthly_target.merge(
        trading_days_month,
        on="Month",
        how="left"
    )

    monthly_target = monthly_target.merge(
        trading_days_target,
        on="Target Month",
        how="left"
    )

    # For GARCH forecasting, the modelling script should use:
    # Number of Trading Days in Target Month

    # --------------------------------------------------
    # PART I: CREATE 80/20 CHRONOLOGICAL SPLIT
    # --------------------------------------------------
    print("Creating 80/20 chronological split...")

    monthly_target = monthly_target.sort_values("Target Month").reset_index(drop=True)

    total_months = len(monthly_target)
    split_index = int(0.8 * total_months)

    monthly_target["Sample Split"] = "Out-of-Sample"
    monthly_target.loc[: split_index - 1, "Sample Split"] = "In-Sample"

    in_sample_df = monthly_target[monthly_target["Sample Split"] == "In-Sample"].copy()
    out_sample_df = monthly_target[monthly_target["Sample Split"] == "Out-of-Sample"].copy()

    # --------------------------------------------------
    # PART J: CREATE DAILY GARCH DATASET
    # --------------------------------------------------
    monthly_info_for_daily = monthly_target[
        [
            "Month",
            "Target Month",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
            "Log Realized Volatility",
            "Annualized Realized Volatility (%)",
            "Next Month Log Realized Volatility",
            "Next Month Annualized Realized Volatility (%)",
            "Number of Trading Days in Target Month",
            "Sample Split",
        ]
    ].copy()

    daily_garch = pd.merge(
        df_sp500,
        monthly_info_for_daily,
        on="Month",
        how="left"
    )

    daily_garch = daily_garch[
        [
            "date",
            "Month",
            "adjusted_close",
            "Daily Return Decimal",
            "Daily Return (%)",
            "Aggregate Climate Risk",
            "PCA Climate Risk Index",
            "Log Realized Volatility",
            "Annualized Realized Volatility (%)",
            "Next Month Log Realized Volatility",
            "Next Month Annualized Realized Volatility (%)",
            "Target Month",
            "Number of Trading Days in Target Month",
            "Sample Split",
        ]
    ].copy()

    daily_garch = daily_garch.rename(columns={"date": "Date"})
    daily_garch = daily_garch.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------
    # SAVE OUTPUT FILES
    # --------------------------------------------------
    print("Saving output files...")

    daily_garch.to_excel(daily_output_file, index=False)
    monthly_target.to_excel(monthly_output_file, index=False)
    df_pca_details.to_excel(pca_output_file, index=False)

    # --------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------
    daily_missing = daily_garch.isna().sum()
    monthly_missing = monthly_target.isna().sum()

    first_valid_agg = monthly_target["Aggregate Climate Risk"].first_valid_index()
    first_valid_pca = monthly_target["PCA Climate Risk Index"].first_valid_index()

    first_valid_agg_month = (
        monthly_target.loc[first_valid_agg, "Month"] if first_valid_agg is not None else np.nan
    )
    first_valid_pca_month = (
        monthly_target.loc[first_valid_pca, "Month"] if first_valid_pca is not None else np.nan
    )

    vol_series = monthly_target["Annualized Realized Volatility (%)"]

    summary_text = f"""
GARCH DATASET SUMMARY
=====================

Input files used:
1. {sp500_file}
2. {climate_file}

Output files created:
1. {daily_output_file}
2. {monthly_output_file}
3. {pca_output_file}
4. {summary_output_file}

Confirmation:
- AR_Monthly_Model_Dataset.xlsx was not used
- Recursive_PCA_Details.xlsx was not used

Sample note:
The intended broad sample window is Jan 2000 to Dec 2025. The actual usable sample may start later or end earlier depending on available S&P 500 data, Faccini climate data, smoothing requirements, recursive PCA minimum observations, and next-month target availability.

Daily dataset:
Intended daily start date: {intended_start_date}
Intended daily end date: {intended_end_date}
Actual daily start date: {daily_garch['Date'].min()}
Actual daily end date: {daily_garch['Date'].max()}
Number of daily observations: {len(daily_garch)}

Monthly dataset:
Actual monthly start month: {monthly_target['Month'].min()}
Actual monthly end month: {monthly_target['Month'].max()}
Actual first target month: {monthly_target['Target Month'].min()}
Actual last target month: {monthly_target['Target Month'].max()}
Number of monthly observations: {len(monthly_target)}

80/20 chronological split:
Number of in-sample monthly observations: {len(in_sample_df)}
Number of out-of-sample monthly observations: {len(out_sample_df)}
In-sample start period: {in_sample_df['Month'].min() if len(in_sample_df) > 0 else np.nan}
In-sample end target month: {in_sample_df['Target Month'].max() if len(in_sample_df) > 0 else np.nan}
Out-of-sample start target month: {out_sample_df['Target Month'].min() if len(out_sample_df) > 0 else np.nan}
Out-of-sample end target month: {out_sample_df['Target Month'].max() if len(out_sample_df) > 0 else np.nan}

Missing values by column in daily dataset:
{daily_missing.to_string()}

Missing values by column in monthly dataset:
{monthly_missing.to_string()}

Daily Return (%) statistics:
Mean: {daily_garch['Daily Return (%)'].mean()}
Min: {daily_garch['Daily Return (%)'].min()}
Max: {daily_garch['Daily Return (%)'].max()}
Standard deviation: {daily_garch['Daily Return (%)'].std()}

First valid Aggregate Climate Risk month:
{first_valid_agg_month}

First valid PCA Climate Risk Index month:
{first_valid_pca_month}

Unit confirmations:
- Daily Return (%) = Daily Return Decimal * 100
- GARCH model should use Daily Return (%)
- When GARCH uses Daily Return (%), forecasted annualized volatility should not be multiplied by 100 again
- Number of Trading Days in Target Month should be used as the GARCH forecast horizon

Monthly volatility statistics:
Mean Annualized Realized Volatility (%): {vol_series.mean()}
1st percentile: {vol_series.quantile(0.01)}
99th percentile: {vol_series.quantile(0.99)}
Standard deviation: {vol_series.std()}
Lag-1 autocorrelation: {vol_series.autocorr(lag=1)}
"""

    with open(summary_output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()