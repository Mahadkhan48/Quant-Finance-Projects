import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# CODE 1: DATA PREPARATION
# =====================================================

# Input file name
input_file_name = "six_stocks_adjusted_close_2014_2025.xlsx"

# Get current code folder
current_folder = Path.cwd()

# Search for file in current folder and parent folder
possible_paths = [
    current_folder / input_file_name,
    current_folder.parent / input_file_name
]

input_file = None

for path in possible_paths:
    if path.exists():
        input_file = path
        break

if input_file is None:
    raise FileNotFoundError(
        f"Could not find {input_file_name} in current folder or parent folder."
    )

# Output file saved in same folder as input data
output_folder = input_file.parent
prepared_output_file = output_folder / "Q3_prepared_data.xlsx"

print("Input file found at:")
print(input_file)

print("\nPrepared data will be saved at:")
print(prepared_output_file)

# =====================================================
# 1. Load adjusted close prices
# =====================================================

prices = pd.read_excel(input_file, sheet_name="Adj_Close")

prices["Date"] = pd.to_datetime(prices["Date"])
prices = prices.set_index("Date")
prices = prices.sort_index()

# =====================================================
# 2. Calculate daily log returns
# =====================================================

returns = np.log(prices / prices.shift(1)).dropna()

# =====================================================
# 3. Split data into training and testing samples
# =====================================================

split_point = len(returns) // 2

train_returns = returns.iloc[:split_point]
test_returns = returns.iloc[split_point:]

# =====================================================
# 4. Data checks
# =====================================================

summary = pd.DataFrame({
    "Item": [
        "Price Start Date",
        "Price End Date",
        "Return Start Date",
        "Return End Date",
        "Number of Price Observations",
        "Number of Return Observations",
        "Training Observations",
        "Testing Observations",
        "Number of Assets"
    ],
    "Value": [
        prices.index.min(),
        prices.index.max(),
        returns.index.min(),
        returns.index.max(),
        len(prices),
        len(returns),
        len(train_returns),
        len(test_returns),
        len(returns.columns)
    ]
})

missing_values = prices.isna().sum().reset_index()
missing_values.columns = ["Asset", "Missing Values"]

# =====================================================
# 5. Save prepared data
# =====================================================

with pd.ExcelWriter(prepared_output_file, engine="openpyxl", mode="w") as writer:
    prices.to_excel(writer, sheet_name="Prices")
    returns.to_excel(writer, sheet_name="Log_Returns")
    train_returns.to_excel(writer, sheet_name="Train_Returns")
    test_returns.to_excel(writer, sheet_name="Test_Returns")
    summary.to_excel(writer, sheet_name="Data_Summary", index=False)
    missing_values.to_excel(writer, sheet_name="Missing_Values", index=False)

print("\nCode 1 completed successfully.")
print("Prepared data saved to:")
print(prepared_output_file)

print("\nTraining sample:")
print(train_returns.index.min(), "to", train_returns.index.max())

print("\nTesting sample:")
print(test_returns.index.min(), "to", test_returns.index.max())