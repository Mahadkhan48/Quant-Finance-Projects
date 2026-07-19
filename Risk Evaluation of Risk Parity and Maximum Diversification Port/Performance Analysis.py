import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# CODE 3: PORTFOLIO PERFORMANCE ANALYSIS
# Correct version for log-return methodology
# =====================================================

input_file_name = "Q3_portfolio_results.xlsx"

current_folder = Path.cwd()

possible_paths = [
    current_folder / input_file_name,
    current_folder.parent / input_file_name,
    current_folder.parent.parent / input_file_name
]

input_file = None

for path in possible_paths:
    if path.exists():
        input_file = path
        break

if input_file is None:
    raise FileNotFoundError(
        f"Could not find {input_file_name}. Please run Code 2 first."
    )

output_folder = input_file.parent
performance_output_file = output_folder / "Q3_performance_analysis.xlsx"

print("Portfolio results file found at:")
print(input_file)

print("\nPerformance analysis will be saved at:")
print(performance_output_file)

# =====================================================
# 1. Load required data
# =====================================================

train_log_returns = pd.read_excel(input_file, sheet_name="Train_Returns", index_col=0)
test_log_returns = pd.read_excel(input_file, sheet_name="Test_Returns", index_col=0)
weights_table = pd.read_excel(input_file, sheet_name="Portfolio_Weights")

train_log_returns.index = pd.to_datetime(train_log_returns.index)
test_log_returns.index = pd.to_datetime(test_log_returns.index)

assets = train_log_returns.columns.tolist()

# Set Asset column as index to ensure weights align correctly with return columns
weights_df = weights_table.set_index("Asset")
weights_df = weights_df.reindex(assets)

if weights_df.isna().any().any():
    raise ValueError("Weights table assets do not match return data columns.")

portfolio_names = weights_df.columns.tolist()

# =====================================================
# 2. Convert asset log returns to simple returns
# =====================================================

train_simple_returns = np.exp(train_log_returns) - 1
test_simple_returns = np.exp(test_log_returns) - 1

# =====================================================
# 3. Reconstruct portfolio returns correctly
# =====================================================
# Portfolio simple return:
# R_p,t = w1R1,t + w2R2,t + ... + wnRn,t
#
# Portfolio log return:
# r_p,t = log(1 + R_p,t)
# =====================================================

portfolio_train_simple_returns = pd.DataFrame(index=train_simple_returns.index)
portfolio_test_simple_returns = pd.DataFrame(index=test_simple_returns.index)

portfolio_train_log_returns = pd.DataFrame(index=train_log_returns.index)
portfolio_test_log_returns = pd.DataFrame(index=test_log_returns.index)

for portfolio in portfolio_names:
    weights = weights_df[portfolio].astype(float).values

    weight_sum = weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"Weights for {portfolio} do not sum to 1. Sum = {weight_sum}")

    portfolio_train_simple_returns[portfolio] = train_simple_returns @ weights
    portfolio_test_simple_returns[portfolio] = test_simple_returns @ weights

    portfolio_train_log_returns[portfolio] = np.log1p(
        portfolio_train_simple_returns[portfolio]
    )

    portfolio_test_log_returns[portfolio] = np.log1p(
        portfolio_test_simple_returns[portfolio]
    )

# =====================================================
# 4. Performance functions
# =====================================================

def annualized_return_from_log(log_returns):
    """
    Converts average daily log return into annualized simple return.
    """
    return np.exp(log_returns.mean() * 252) - 1


def annualized_volatility(log_returns):
    """
    Annualized volatility based on daily log returns.
    """
    return log_returns.std(ddof=1) * np.sqrt(252)


def sharpe_ratio(log_returns):
    """
    Sharpe Ratio assuming zero risk-free rate.
    Uses daily log returns.
    """
    vol = log_returns.std(ddof=1)

    if vol == 0:
        return np.nan

    return (log_returns.mean() / vol) * np.sqrt(252)


def sortino_ratio(log_returns):
    """
    Sortino Ratio using only negative daily log returns.
    Assumes zero risk-free rate.
    """
    downside_returns = log_returns[log_returns < 0]

    if len(downside_returns) < 2:
        return np.nan

    downside_vol = downside_returns.std(ddof=1)

    if downside_vol == 0:
        return np.nan

    return (log_returns.mean() / downside_vol) * np.sqrt(252)


def cumulative_value_from_simple(simple_returns):
    """
    Correct cumulative value from simple returns.
    """
    return (1 + simple_returns).cumprod()


def max_drawdown_from_simple(simple_returns):
    """
    Maximum drawdown calculated from cumulative portfolio value.
    """
    cumulative_value = cumulative_value_from_simple(simple_returns)
    running_max = cumulative_value.cummax()
    drawdown = cumulative_value / running_max - 1
    return drawdown.min()


def calmar_ratio(log_returns, simple_returns):
    """
    Annualized return divided by absolute maximum drawdown.
    """
    ann_return = annualized_return_from_log(log_returns)
    mdd = abs(max_drawdown_from_simple(simple_returns))

    if mdd == 0:
        return np.nan

    return ann_return / mdd


def historical_var_log(log_returns, alpha=0.95):
    """
    Historical VaR using log-return losses.
    Loss = -log return
    """
    losses = -log_returns
    return np.quantile(losses, alpha)


def count_var_violations(log_returns, var_value):
    """
    VaR violation occurs when actual loss exceeds VaR.
    """
    losses = -log_returns
    violations = losses > var_value
    return violations.sum(), violations.mean(), violations.astype(int)


# =====================================================
# 5. Calculate performance metrics
# =====================================================

alpha = 0.95
expected_violation_rate = 1 - alpha

performance_results = []
var_thresholds = []
var_violation_series = pd.DataFrame(index=portfolio_test_log_returns.index)

equal_weights = np.repeat(1 / len(assets), len(assets))

for portfolio in portfolio_names:

    train_log = portfolio_train_log_returns[portfolio]
    test_log = portfolio_test_log_returns[portfolio]

    test_simple = portfolio_test_simple_returns[portfolio]

    # Estimate 95% historical VaR from training sample
    var_95 = historical_var_log(train_log, alpha=alpha)

    # Count test-period VaR violations
    n_violations, violation_rate, violation_indicator = count_var_violations(
        test_log, var_95
    )

    expected_violations = len(test_log) * expected_violation_rate

    var_violation_series[portfolio + "_Violation"] = violation_indicator

    # Turnover from equal-weighted allocation
    weights = weights_df[portfolio].astype(float).values
    one_way_turnover_from_equal = 0.5 * np.sum(np.abs(weights - equal_weights))

    cumulative_value = cumulative_value_from_simple(test_simple)
    total_return = cumulative_value.iloc[-1] - 1

    performance_results.append({
        "Portfolio": portfolio,
        "Total Return": total_return,
        "Annualized Return": annualized_return_from_log(test_log),
        "Annualized Volatility": annualized_volatility(test_log),
        "Sharpe Ratio": sharpe_ratio(test_log),
        "Sortino Ratio": sortino_ratio(test_log),
        "Maximum Drawdown": max_drawdown_from_simple(test_simple),
        "Calmar Ratio": calmar_ratio(test_log, test_simple),
        "Historical VaR 95%": var_95,
        "Number of VaR Violations": int(n_violations),
        "Violation Rate": violation_rate,
        "Expected Violations at 95%": expected_violations,
        "Expected Violation Rate": expected_violation_rate,
        "One-Way Turnover from Equal Weight": one_way_turnover_from_equal,
        "Best Daily Log Return": test_log.max(),
        "Worst Daily Log Return": test_log.min()
    })

    var_thresholds.append({
        "Portfolio": portfolio,
        "VaR Confidence Level": alpha,
        "Historical VaR 95%": var_95,
        "Expected Violation Rate": expected_violation_rate,
        "Expected Violations": expected_violations
    })

performance_table = pd.DataFrame(performance_results)
var_thresholds_table = pd.DataFrame(var_thresholds)

# =====================================================
# 6. Cumulative values and drawdowns
# =====================================================

cumulative_values = pd.DataFrame(index=portfolio_test_simple_returns.index)
drawdown_series = pd.DataFrame(index=portfolio_test_simple_returns.index)

for portfolio in portfolio_names:
    cumulative_values[portfolio] = cumulative_value_from_simple(
        portfolio_test_simple_returns[portfolio]
    )

    running_max = cumulative_values[portfolio].cummax()
    drawdown_series[portfolio] = cumulative_values[portfolio] / running_max - 1

# =====================================================
# 7. Sanity checks
# =====================================================

sanity_checks = []

sanity_checks.append({
    "Check": "Number of training observations",
    "Result": len(train_log_returns)
})

sanity_checks.append({
    "Check": "Number of testing observations",
    "Result": len(test_log_returns)
})

sanity_checks.append({
    "Check": "Missing values in train log returns",
    "Result": int(train_log_returns.isna().sum().sum())
})

sanity_checks.append({
    "Check": "Missing values in test log returns",
    "Result": int(test_log_returns.isna().sum().sum())
})

sanity_checks.append({
    "Check": "Missing values in portfolio test log returns",
    "Result": int(portfolio_test_log_returns.isna().sum().sum())
})

for portfolio in portfolio_names:
    sanity_checks.append({
        "Check": f"{portfolio} weight sum",
        "Result": weights_df[portfolio].sum()
    })

    sanity_checks.append({
        "Check": f"{portfolio} minimum weight",
        "Result": weights_df[portfolio].min()
    })

    sanity_checks.append({
        "Check": f"{portfolio} maximum weight",
        "Result": weights_df[portfolio].max()
    })

sanity_checks_table = pd.DataFrame(sanity_checks)

# =====================================================
# 8. Save outputs
# =====================================================

with pd.ExcelWriter(performance_output_file, engine="openpyxl", mode="w") as writer:

    weights_df.to_excel(writer, sheet_name="Portfolio_Weights")

    portfolio_train_simple_returns.to_excel(
        writer, sheet_name="Portfolio_Train_Simple"
    )

    portfolio_test_simple_returns.to_excel(
        writer, sheet_name="Portfolio_Test_Simple"
    )

    portfolio_train_log_returns.to_excel(
        writer, sheet_name="Portfolio_Train_Log"
    )

    portfolio_test_log_returns.to_excel(
        writer, sheet_name="Portfolio_Test_Log"
    )

    cumulative_values.to_excel(
        writer, sheet_name="Cumulative_Values"
    )

    drawdown_series.to_excel(
        writer, sheet_name="Drawdowns"
    )

    var_violation_series.to_excel(
        writer, sheet_name="VaR_Violations"
    )

    var_thresholds_table.to_excel(
        writer, sheet_name="VaR_Thresholds", index=False
    )

    performance_table.to_excel(
        writer, sheet_name="Performance_Table", index=False
    )

    sanity_checks_table.to_excel(
        writer, sheet_name="Sanity_Checks", index=False
    )

print("\nCode 3 completed successfully.")
print("Performance analysis saved to:")
print(performance_output_file)

print("\nPerformance Table:")
print(performance_table)

print("\nSanity Checks:")
print(sanity_checks_table)