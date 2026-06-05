# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:10:55 2026

@author: mahad
"""

# ============================================================
# PART 1: DCC-GARCH(1,1) for 5-dimensional system
#   1) Load daily data
#   2) Transform to log-returns
#   3) Estimate VAR(1) mean equation
#   4) Estimate univariate GARCH(1,1) for each residual series
#   5) Estimate DCC(1,1) on standardized residuals
#   6) Construct H_t = D_t R_t D_t
# ============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = r"C:\MAHAD\Master in Quant\SUBJECTS\Term 2\Financial Econometrics\Coursework\Question_3\market_data - Copy.xlsx"
sheet_name = "in"

raw = pd.read_excel(file_path, sheet_name=sheet_name)

# Keep only relevant columns
cols = ["Date", "SP500", "GBPUSD", "Brent_Spot", "ust10y_yield_pct", "corp_oas_pct"]
df = raw[cols].copy()

# Date handling
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)

# ============================================================
# ============================================================
# 2. CLEAN + TRANSFORM SERIES PROPERLY
# ============================================================

series_cols = ["SP500", "GBPUSD", "Brent_Spot", "ust10y_yield_pct", "corp_oas_pct"]

for c in series_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Interpolate small gaps if needed
df[series_cols] = df[series_cols].interpolate(method="linear", limit_direction="both")

# Proper finance transformations
ret = pd.DataFrame()
ret["Date"] = df["Date"]

# Prices / FX / commodity -> log returns (%)
ret["EQ"] = 100 * np.log(df["SP500"]).diff()
ret["FX"] = 100 * np.log(df["GBPUSD"]).diff()
ret["CMDTY"] = 100 * np.log(df["Brent_Spot"]).diff()

# Yield / spread -> first differences in basis points
ret["GOVYLD"] = 100 * df["ust10y_yield_pct"].diff()
ret["CORPSPR"] = 100 * df["corp_oas_pct"].diff()

ret = ret.dropna().reset_index(drop=True)
# ============================================================
# 3. ESTIMATE VAR(1) MEAN EQUATION
# ============================================================
# This follows lecture logic: MGARCH is applied to VAR residuals.

Y = ret[["EQ", "FX", "CMDTY", "GOVYLD", "CORPSPR"]].copy()

var_model = VAR(Y)
var_res = var_model.fit(maxlags=1, ic=None, trend="c")

print("\n================ VAR(1) SUMMARY ================\n")
print(var_res.summary())

# Residuals from VAR
eps = var_res.resid.copy()
eps.index = ret["Date"].iloc[var_res.k_ar:].values  # align dates after lag loss

# ============================================================
# 4. UNIVARIATE GARCH(1,1) ESTIMATION
# ============================================================

def garch11_negloglik(params, x):
    """
    Gaussian negative log-likelihood for GARCH(1,1):
        h_t = omega + alpha * x_{t-1}^2 + beta * h_{t-1}
    """
    omega, alpha, beta = params

    # Parameter constraints
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e12

    T = len(x)
    h = np.zeros(T)
    h[0] = np.var(x)

    for t in range(1, T):
        h[t] = omega + alpha * x[t-1]**2 + beta * h[t-1]
        if h[t] <= 0:
            return 1e12

    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (x**2) / h)
    return -ll  # minimize negative log-likelihood



def fit_garch11(x):
    x = np.asarray(x, dtype=float)
    var_x = np.var(x)

    # starting values
    x0 = np.array([0.01 * var_x, 0.05, 0.90])

    bounds = [
        (1e-8, None),   # omega
        (1e-8, 0.999),  # alpha
        (1e-8, 0.999)   # beta
    ]

    result = minimize(
        garch11_negloglik,
        x0=x0,
        args=(x,),
        method="L-BFGS-B",
        bounds=bounds
    )

    omega, alpha, beta = result.x

    # Conditional variances
    T = len(x)
    h = np.zeros(T)
    h[0] = np.var(x)

    for t in range(1, T):
        h[t] = omega + alpha * x[t-1]**2 + beta * h[t-1]

    z = x / np.sqrt(h)

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "loglik": -result.fun,
        "h": h,
        "z": z,
        "success": result.success
    }

garch_results = {}
std_resids = pd.DataFrame(index=eps.index)

for c in eps.columns:
    fit = fit_garch11(eps[c].values)
    garch_results[c] = fit
    std_resids[c] = fit["z"]

# Print univariate GARCH results
print("\n================ UNIVARIATE GARCH(1,1) RESULTS ================\n")
for c, res in garch_results.items():
    print(
        f"{c}: omega={res['omega']:.6f}, alpha={res['alpha']:.4f}, "
        f"beta={res['beta']:.4f}, alpha+beta={res['alpha'] + res['beta']:.4f}, "
        f"loglik={res['loglik']:.2f}, success={res['success']}"
    )

# ============================================================
# 5. DCC(1,1) ESTIMATION
# ============================================================
# Engle-style DCC:
#   Q_t = (1-a-b) Qbar + a z_{t-1}z'_{t-1} + b Q_{t-1}
#   R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

Z = std_resids.values
T, N = Z.shape
Qbar = np.cov(Z.T)

def dcc_negloglik(params, Z, Qbar):
    a, b = params

    # constraints
    if a < 0 or b < 0 or (a + b) >= 0.999:
        return 1e12

    T, N = Z.shape
    Qt = Qbar.copy()
    nll = 0.0

    for t in range(T):
        if t > 0:
            zlag = Z[t-1].reshape(-1, 1)
            Qt = (1 - a - b) * Qbar + a * (zlag @ zlag.T) + b * Qt

        diag_q = np.sqrt(np.diag(Qt))
        if np.any(diag_q <= 0):
            return 1e12

        Dinv = np.diag(1.0 / diag_q)
        Rt = Dinv @ Qt @ Dinv

        # Numerical stability
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0:
            return 1e12

        invRt = np.linalg.inv(Rt)
        zt = Z[t].reshape(-1, 1)

        # DCC correlation log-likelihood contribution
        nll += 0.5 * (logdet + float(zt.T @ (invRt - np.eye(N)) @ zt))

    return nll

dcc_start = np.array([0.02, 0.95])
dcc_bounds = [(1e-8, 0.999), (1e-8, 0.999)]

dcc_res = minimize(
    dcc_negloglik,
    x0=dcc_start,
    args=(Z, Qbar),
    method="L-BFGS-B",
    bounds=dcc_bounds
)

a_dcc, b_dcc = dcc_res.x

print("\n================ DCC(1,1) RESULTS ================\n")
print(f"a = {a_dcc:.6f}")
print(f"b = {b_dcc:.6f}")
print(f"a + b = {a_dcc + b_dcc:.6f}")
print(f"Optimization success = {dcc_res.success}")
print(f"DCC log-likelihood contribution = {-dcc_res.fun:.4f}")

# ============================================================
# 6. BUILD TIME-VARYING CORRELATION MATRICES R_t
#    AND CONDITIONAL COVARIANCE MATRICES H_t
# ============================================================

Qt_list = []
Rt_list = []
Ht_list = []

Qt = Qbar.copy()

# Collect conditional std devs from univariate GARCH
sigma_t = np.column_stack([
    np.sqrt(garch_results[c]["h"]) for c in eps.columns
])

for t in range(T):
    if t > 0:
        zlag = Z[t-1].reshape(-1, 1)
        Qt = (1 - a_dcc - b_dcc) * Qbar + a_dcc * (zlag @ zlag.T) + b_dcc * Qt

    diag_q = np.sqrt(np.diag(Qt))
    Dinv = np.diag(1.0 / diag_q)
    Rt = Dinv @ Qt @ Dinv

    Dt = np.diag(sigma_t[t, :])
    Ht = Dt @ Rt @ Dt

    Qt_list.append(Qt.copy())
    Rt_list.append(Rt.copy())
    Ht_list.append(Ht.copy())

# ============================================================
# 7. STORE SELECTED OUTPUTS
# ============================================================

# Example: save pairwise conditional correlations to a DataFrame
pair_names = []
corr_data = {}

asset_names = list(eps.columns)

for i in range(N):
    for j in range(i + 1, N):
        pair = f"corr_{asset_names[i]}_{asset_names[j]}"
        pair_names.append(pair)
        corr_data[pair] = [Rt_list[t][i, j] for t in range(T)]

dcc_corr_df = pd.DataFrame(corr_data, index=eps.index)
dcc_corr_df.index.name = "Date"

# Save conditional covariance matrices as flattened rows
cov_rows = []
for t, dt in enumerate(eps.index):
    row = {"Date": dt}
    H = Ht_list[t]
    for i in range(N):
        for j in range(N):
            row[f"H_{asset_names[i]}_{asset_names[j]}"] = H[i, j]
    cov_rows.append(row)

Ht_df = pd.DataFrame(cov_rows)

# Save standardized residuals too
std_resids_out = std_resids.copy()
std_resids_out.index.name = "Date"

# ============================================================
# 8. EXPORT RESULTS
# ============================================================

out_file = "C:/MAHAD/Master in Quant/SUBJECTS/Term 2/Financial Econometrics/Coursework/Question_3/DCC_GARCH_part1_results.xlsx"

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    ret.to_excel(writer, sheet_name="Returns", index=False)
    eps.reset_index().rename(columns={"index": "Date"}).to_excel(writer, sheet_name="VAR_Residuals", index=False)
    std_resids_out.reset_index().to_excel(writer, sheet_name="Std_Residuals", index=False)
    dcc_corr_df.reset_index().to_excel(writer, sheet_name="DCC_Correlations", index=False)
    Ht_df.to_excel(writer, sheet_name="Cond_Covariances", index=False)

    # Parameter summary
    param_rows = []
    for c, res in garch_results.items():
        param_rows.append({
            "Series": c,
            "omega": res["omega"],
            "alpha": res["alpha"],
            "beta": res["beta"],
            "alpha_plus_beta": res["alpha"] + res["beta"],
            "loglik": res["loglik"],
            "success": res["success"]
        })

    param_rows.append({
        "Series": "DCC",
        "omega": np.nan,
        "alpha": a_dcc,
        "beta": b_dcc,
        "alpha_plus_beta": a_dcc + b_dcc,
        "loglik": -dcc_res.fun,
        "success": dcc_res.success
    })

    pd.DataFrame(param_rows).to_excel(writer, sheet_name="Parameters", index=False)

print(f"\nResults saved to: {out_file}")