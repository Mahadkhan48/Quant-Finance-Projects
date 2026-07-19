# file: 00_prepare_monthly_AR_dataset_paper1_style.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA


def main():

    print("Starting dataset preparation...")

    base_path = Path(r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data_2")

    sp500_file = base_path / "S&P500_Daily_Data.xlsx"
    climate_file = base_path / "Faccini Climate Risk Data.xlsx"

    output_file = base_path / "AR_Monthly_Model_Dataset.xlsx"
    pca_file = base_path / "Recursive_PCA_Details.xlsx"
    summary_file = base_path / "AR_Monthly_Model_Dataset_Summary.txt"

    # ---------------------------
    # PART A: SP500 VOLATILITY
    # ---------------------------

    df = pd.read_excel(sp500_file)
    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["return"] = np.log(df["adjusted_close"] / df["adjusted_close"].shift(1))
    df = df.dropna()

    df["Month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    rv = df.groupby("Month")["return"].apply(lambda x: np.sqrt(np.sum(x**2))).reset_index()
    rv.columns = ["Month", "Monthly Realized Volatility Decimal"]

    rv["Annualized Realized Volatility Decimal"] = rv["Monthly Realized Volatility Decimal"] * np.sqrt(12)
    rv["Annualized Realized Volatility (%)"] = rv["Annualized Realized Volatility Decimal"] * 100
    rv["Log Realized Volatility"] = np.log(rv["Annualized Realized Volatility (%)"])

    # ---------------------------
    # PART B: CLIMATE DATA
    # ---------------------------

    df_c = pd.read_excel(climate_file)
    df_c.columns = df_c.columns.str.strip()

    df_c["date"] = pd.to_datetime(df_c["date"])
    df_c = df_c.sort_values("date")

    df_c["Month"] = df_c["date"].dt.to_period("M").dt.to_timestamp()

    climate_vars = [
        "US climate policy",
        "International summits",
        "Global warming",
        "Natural disasters"
    ]

    monthly_c = df_c.groupby("Month")[climate_vars].mean().reset_index()

    for col in climate_vars:
        monthly_c[f"log_{col}"] = np.log(1 + monthly_c[col])
        monthly_c[f"smoothed_{col}"] = monthly_c[f"log_{col}"].rolling(6, min_periods=6).mean()

    smoothed_cols = [f"smoothed_{c}" for c in climate_vars]

    # ---------------------------
    # PART C: AGGREGATE CRI (FIXED)
    # ---------------------------

    print("Creating Aggregate CRI...")

    agg_list = []
    min_obs = 12

    for i in range(len(monthly_c)):

        hist = monthly_c.iloc[:i+1].dropna(subset=smoothed_cols)

        if len(hist) < min_obs:
            agg_list.append(np.nan)
            continue

        means = hist[smoothed_cols].mean()
        stds = hist[smoothed_cols].std()

        if stds.isna().any() or (stds == 0).any():
            agg_list.append(np.nan)
            continue

        current = monthly_c.loc[i, smoothed_cols]

        if current.isna().any():
            agg_list.append(np.nan)
            continue

        z = (current - means) / stds
        agg_list.append(z.mean())

    monthly_c["Aggregate Climate Risk"] = agg_list

    # ---------------------------
    # PART D: PCA CRI (FIXED)
    # ---------------------------

    print("Creating PCA CRI...")

    pca_values = []
    pca_details = []

    for i in range(len(monthly_c)):

        hist = monthly_c.iloc[:i+1].dropna(subset=smoothed_cols)

        if len(hist) < min_obs:
            pca_values.append(np.nan)
            continue

        X = hist[smoothed_cols]

        means = X.mean()
        stds = X.std()

        if stds.isna().any() or (stds == 0).any():
            pca_values.append(np.nan)
            continue

        X_std = (X - means) / stds

        pca = PCA(n_components=1)
        pca.fit(X_std)

        current = monthly_c.loc[i, smoothed_cols]

        if current.isna().any():
            pca_values.append(np.nan)
            continue

        current_std = ((current - means) / stds).values.reshape(1, -1)
        val = pca.transform(current_std)[0, 0]

        loadings = pca.components_[0]

        if loadings.mean() < 0:
            val = -val
            loadings = -loadings

        pca_values.append(val)

        pca_details.append({
            "Month": monthly_c.loc[i, "Month"],
            "Obs": len(hist),
            "Explained Variance": pca.explained_variance_ratio_[0],
            "US climate policy": loadings[0],
            "International summits": loadings[1],
            "Global warming": loadings[2],
            "Natural disasters": loadings[3]
        })

    monthly_c["PCA Climate Risk Index"] = pca_values
    df_pca = pd.DataFrame(pca_details)

    # ---------------------------
    # PART E: MERGE + FILTER (FIXED)
    # ---------------------------

    df_final = pd.merge(rv, monthly_c, on="Month", how="inner")

    # ✅ FIX: restrict sample FIRST
    df_final = df_final[
        (df_final["Month"] >= "2005-01-01") &
        (df_final["Month"] <= "2022-12-01")
    ].copy()

    df_final = df_final.sort_values("Month").reset_index(drop=True)

    # ✅ THEN create targets
    df_final["Target Month"] = df_final["Month"] + pd.offsets.MonthBegin(1)
    df_final["Next Month Log Realized Volatility"] = df_final["Log Realized Volatility"].shift(-1)
    df_final["Next Month Annualized Realized Volatility (%)"] = df_final["Annualized Realized Volatility (%)"].shift(-1)

    df_final = df_final.dropna(subset=["Log Realized Volatility", "Next Month Log Realized Volatility"])

    # ---------------------------
    # SAVE
    # ---------------------------

    df_final.to_excel(output_file, index=False)
    df_pca.to_excel(pca_file, index=False)

    # ---------------------------
    # SUMMARY (FIXED)
    # ---------------------------

    vol = df_final["Annualized Realized Volatility (%)"]

    first_agg = df_final["Aggregate Climate Risk"].first_valid_index()
    first_pca = df_final["PCA Climate Risk Index"].first_valid_index()

    summary = f"""
Rows: {len(df_final)}
Start: {df_final['Month'].min()}
End: {df_final['Month'].max()}

Mean Vol (%): {vol.mean()}
1%: {vol.quantile(0.01)}
99%: {vol.quantile(0.99)}
Std: {vol.std()}
ACF1: {vol.autocorr(1)}

First Aggregate CRI Month: {df_final.loc[first_agg, 'Month'] if first_agg is not None else None}
First PCA CRI Month: {df_final.loc[first_pca, 'Month'] if first_pca is not None else None}

Missing Values:
{df_final.isna().sum()}
"""

    with open(summary_file, "w") as f:
        f.write(summary)

    print("DONE ✅")


if __name__ == "__main__":
    main()