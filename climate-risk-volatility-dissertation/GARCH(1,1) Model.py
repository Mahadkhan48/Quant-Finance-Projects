# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:45:48 2026

@author: mahad
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# 1) INPUT FILE PATH
# ============================================================

input_file = r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\S&P500_Daily_Data.xlsx"

# Output folder = same folder as input file
output_folder = os.path.dirname(input_file)

summary_output_file = os.path.join(
    output_folder,
    "GARCH_11_Benchmark_SP500_Results.txt"
)

vol_output_file = os.path.join(
    output_folder,
    "GARCH_11_SP500_Conditional_Volatility.xlsx"
)

# ============================================================
# 2) IMPORT ARCH PACKAGE WITH ERROR HANDLING
# ============================================================

try:
    from arch import arch_model
except ImportError:
    print("The 'arch' package is not installed.")
    print("Please install it using:")
    print("pip install arch")
    raise SystemExit

# ============================================================
# 3) READ EXCEL FILE
# ============================================================

try:
    df = pd.read_excel(input_file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise SystemExit

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# ============================================================
# 4) CHECK REQUIRED DATE COLUMN
# ============================================================

if "date" not in df.columns:
    print("Error: The Excel file must contain a 'date' column.")
    raise SystemExit

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# ============================================================
# 5) SELECT PRICE COLUMN
# ============================================================

if "adjusted_close" in df.columns:
    price_col = "adjusted_close"
elif "close" in df.columns:
    price_col = "close"
else:
    print("Error: The file must contain either 'adjusted_close' or 'close'.")
    raise SystemExit

# Make sure price column is numeric
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

# ============================================================
# 6) CALCULATE DAILY LOG RETURNS
# ============================================================

df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

# Drop missing return values
df = df.dropna(subset=["date", "log_return"]).reset_index(drop=True)

# Multiply returns by 100 for GARCH numerical stability
returns_for_garch = df["log_return"] * 100

# ============================================================
# 7) FIT GARCH(1,1) MODEL
# ============================================================

garch_model = arch_model(
    returns_for_garch,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

garch_result = garch_model.fit(disp="off")

# ============================================================
# 8) PRINT FULL MODEL SUMMARY
# ============================================================

print(garch_result.summary())

# ============================================================
# 9) EXTRACT KEY PARAMETERS
# ============================================================

params = garch_result.params

mu = params.get("mu", np.nan)
omega = params.get("omega", np.nan)
alpha_1 = params.get("alpha[1]", np.nan)
beta_1 = params.get("beta[1]", np.nan)
alpha_plus_beta = alpha_1 + beta_1

log_likelihood = garch_result.loglikelihood
aic = garch_result.aic
bic = garch_result.bic

print("\nExtracted GARCH(1,1) Results:")
print("mu:", mu)
print("omega:", omega)
print("alpha[1]:", alpha_1)
print("beta[1]:", beta_1)
print("alpha[1] + beta[1]:", alpha_plus_beta)
print("Log-Likelihood:", log_likelihood)
print("AIC:", aic)
print("BIC:", bic)

# ============================================================
# 10) SAVE FULL MODEL SUMMARY TO TEXT FILE
# ============================================================

with open(summary_output_file, "w", encoding="utf-8") as f:
    f.write(str(garch_result.summary()))
    f.write("\n\nExtracted GARCH(1,1) Results:\n")
    f.write(f"mu: {mu}\n")
    f.write(f"omega: {omega}\n")
    f.write(f"alpha[1]: {alpha_1}\n")
    f.write(f"beta[1]: {beta_1}\n")
    f.write(f"alpha[1] + beta[1]: {alpha_plus_beta}\n")
    f.write(f"Log-Likelihood: {log_likelihood}\n")
    f.write(f"AIC: {aic}\n")
    f.write(f"BIC: {bic}\n")

# ============================================================
# 11) SAVE CONDITIONAL VOLATILITY SERIES
# ============================================================

df["conditional_volatility"] = garch_result.conditional_volatility

vol_df = df[["date", "log_return", "conditional_volatility"]]

vol_df.to_excel(vol_output_file, index=False)

# ============================================================
# 12) FINAL MESSAGE
# ============================================================

print("\nFiles saved successfully:")
print("Model summary:", summary_output_file)
print("Conditional volatility:", vol_output_file)