import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm

# =====================================================
# CODE 2: CREATE PORTFOLIOS
# =====================================================

prepared_file_name = "Q3_prepared_data.xlsx"

current_folder = Path.cwd()

possible_paths = [
    current_folder / prepared_file_name,
    current_folder.parent / prepared_file_name
]

prepared_file = None

for path in possible_paths:
    if path.exists():
        prepared_file = path
        break

if prepared_file is None:
    raise FileNotFoundError(
        f"Could not find {prepared_file_name}. Please run Code 1 first."
    )

output_folder = prepared_file.parent
portfolio_output_file = output_folder / "Q3_portfolio_results.xlsx"

print("Prepared data file found at:")
print(prepared_file)

print("\nPortfolio results will be saved at:")
print(portfolio_output_file)

# =====================================================
# 1. Load prepared return data
# =====================================================

returns = pd.read_excel(prepared_file, sheet_name="Log_Returns", index_col=0)
train_returns = pd.read_excel(prepared_file, sheet_name="Train_Returns", index_col=0)
test_returns = pd.read_excel(prepared_file, sheet_name="Test_Returns", index_col=0)

returns.index = pd.to_datetime(returns.index)
train_returns.index = pd.to_datetime(train_returns.index)
test_returns.index = pd.to_datetime(test_returns.index)

assets = returns.columns.tolist()
n_assets = len(assets)

# =====================================================
# 2. Basic inputs from training sample
# =====================================================

cov_matrix = train_returns.cov()
asset_vol_daily = train_returns.std()

z_95 = norm.ppf(0.95)

initial_weights = np.repeat(1 / n_assets, n_assets)

constraints = ({
    "type": "eq",
    "fun": lambda w: np.sum(w) - 1
})

bounds = tuple((0, 1) for _ in range(n_assets))

# =====================================================
# 3. Equal-Weighted Portfolio
# =====================================================

w_equal = np.repeat(1 / n_assets, n_assets)

# =====================================================
# 4. Parametric Component VaR function
# =====================================================

def parametric_component_var(weights, cov_matrix, z_value):
    weights = np.array(weights)

    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    marginal_var = z_value * (cov_matrix @ weights) / portfolio_vol

    component_var = weights * marginal_var

    total_var = component_var.sum()

    component_var_percent = component_var / total_var

    return component_var, component_var_percent, total_var


# =====================================================
# 5. Risk Parity Portfolio - Parametric Component VaR
# =====================================================

def risk_parity_parametric_objective(weights, cov_matrix, z_value):
    _, component_var_percent, _ = parametric_component_var(
        weights, cov_matrix, z_value
    )

    target = np.repeat(1 / len(weights), len(weights))

    return np.sum((component_var_percent - target) ** 2)


rp_parametric_result = minimize(
    risk_parity_parametric_objective,
    initial_weights,
    args=(cov_matrix, z_95),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not rp_parametric_result.success:
    raise RuntimeError("Parametric Risk Parity optimization failed.")

w_rp_parametric = rp_parametric_result.x

# =====================================================
# 6. Non-parametric Component VaR function
# =====================================================

def nonparametric_component_var(weights, return_data, alpha=0.95):
    weights = np.array(weights)

    portfolio_returns = return_data @ weights
    portfolio_losses = -portfolio_returns

    var_level = np.quantile(portfolio_losses, alpha)

    tail_days = portfolio_losses >= var_level

    asset_loss_contributions = -return_data.loc[tail_days].values * weights

    component_var = asset_loss_contributions.mean(axis=0)

    total_var = component_var.sum()

    component_var_percent = component_var / total_var

    return component_var, component_var_percent, total_var, var_level


# =====================================================
# 7. Risk Parity Portfolio - Non-parametric Component VaR
# =====================================================

def risk_parity_nonparametric_objective(weights, return_data, alpha=0.95):
    _, component_var_percent, _, _ = nonparametric_component_var(
        weights, return_data, alpha
    )

    target = np.repeat(1 / len(weights), len(weights))

    return np.sum((component_var_percent - target) ** 2)


rp_nonparametric_result = minimize(
    risk_parity_nonparametric_objective,
    initial_weights,
    args=(train_returns, 0.95),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not rp_nonparametric_result.success:
    raise RuntimeError("Non-parametric Risk Parity optimization failed.")

w_rp_nonparametric = rp_nonparametric_result.x

# =====================================================
# 8. Maximum Diversification Portfolio
# =====================================================

def diversification_ratio(weights, asset_vol, cov_matrix):
    weights = np.array(weights)

    weighted_average_vol = weights @ asset_vol

    portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    return weighted_average_vol / portfolio_vol


def negative_diversification_ratio(weights, asset_vol, cov_matrix):
    return -diversification_ratio(weights, asset_vol, cov_matrix)


mdp_result = minimize(
    negative_diversification_ratio,
    initial_weights,
    args=(asset_vol_daily.values, cov_matrix.values),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

if not mdp_result.success:
    raise RuntimeError("Maximum Diversification optimization failed.")

w_mdp = mdp_result.x

# =====================================================
# 9. Create weights table
# =====================================================

weights_table = pd.DataFrame({
    "Asset": assets,
    "Equal_Weighted": w_equal,
    "Risk_Parity_Parametric": w_rp_parametric,
    "Risk_Parity_NonParametric": w_rp_nonparametric,
    "Maximum_Diversification": w_mdp
})

# =====================================================
# 10. Component VaR tables
# =====================================================

rp_param_cvar, rp_param_cvar_pct, rp_param_total_var = parametric_component_var(
    w_rp_parametric, cov_matrix.values, z_95
)

rp_nonparam_cvar, rp_nonparam_cvar_pct, rp_nonparam_total_var, rp_nonparam_var_level = nonparametric_component_var(
    w_rp_nonparametric, train_returns, 0.95
)

component_var_table = pd.DataFrame({
    "Asset": assets,
    "RP_Parametric_Component_VaR": rp_param_cvar,
    "RP_Parametric_Component_VaR_%": rp_param_cvar_pct,
    "RP_NonParametric_Component_VaR": rp_nonparam_cvar,
    "RP_NonParametric_Component_VaR_%": rp_nonparam_cvar_pct
})

# =====================================================
# 11. Portfolio returns in test sample
# =====================================================

portfolio_returns_test = pd.DataFrame(index=test_returns.index)

portfolio_returns_test["Equal_Weighted"] = test_returns @ w_equal
portfolio_returns_test["Risk_Parity_Parametric"] = test_returns @ w_rp_parametric
portfolio_returns_test["Risk_Parity_NonParametric"] = test_returns @ w_rp_nonparametric
portfolio_returns_test["Maximum_Diversification"] = test_returns @ w_mdp

# =====================================================
# 12. Basic portfolio construction summary
# =====================================================

summary_table = pd.DataFrame({
    "Portfolio": [
        "Equal Weighted",
        "Risk Parity - Parametric Component VaR",
        "Risk Parity - Non-parametric Component VaR",
        "Maximum Diversification"
    ],
    "Description": [
        "Each stock receives equal weight.",
        "Weights chosen so that parametric Component VaR contributions are as equal as possible.",
        "Weights chosen so that historical tail Component VaR contributions are as equal as possible.",
        "Weights chosen to maximize the diversification ratio."
    ]
})

# =====================================================
# 13. Save outputs
# =====================================================

with pd.ExcelWriter(portfolio_output_file, engine="openpyxl", mode="w") as writer:
    returns.to_excel(writer, sheet_name="Log_Returns")
    train_returns.to_excel(writer, sheet_name="Train_Returns")
    test_returns.to_excel(writer, sheet_name="Test_Returns")
    cov_matrix.to_excel(writer, sheet_name="Train_Cov_Matrix")
    weights_table.to_excel(writer, sheet_name="Portfolio_Weights", index=False)
    component_var_table.to_excel(writer, sheet_name="Component_VaR", index=False)
    portfolio_returns_test.to_excel(writer, sheet_name="Portfolio_Test_Returns")
    summary_table.to_excel(writer, sheet_name="Portfolio_Methodology", index=False)

print("\nCode 2 completed successfully.")
print("All portfolio weights saved to:")
print(portfolio_output_file)

print("\nPortfolio Weights:")
print(weights_table)

print("\nComponent VaR Table:")
print(component_var_table)