# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:42:04 2026

@author: mahad
"""

from pathlib import Path
import pandas as pd
import numpy as np

import statsmodels.api as sm

def main():
    # Defines the main function where the script runs.

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\SP500_Climate_Risk_Merged_Monthly_Data.xlsx"
    )
    # Stores the path of the input Excel file.

    output_file = input_file.parent / "AR_Aggregate_Climate_Risk_Model_Results.txt"
    # Creates the output text file path in the same folder as the input file.

    df = pd.read_excel(input_file)
    # Reads the Excel file into a pandas DataFrame.

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    # Converts the Month column into datetime format.

    df = df.sort_values("Month").reset_index(drop=True)
    # Sorts the DataFrame by Month in ascending order.

    df = df.dropna(
        subset=[
            "Monthly Realized Volatility",
            "Next Month Realized Volatility",
            "Aggregate Climate Risk",
        ]
    )
    # Drops rows with missing values in the required columns.

    y = df["Next Month Realized Volatility"]
    # Defines the dependent variable for the regression.

    X = df[["Monthly Realized Volatility", "Aggregate Climate Risk"]]
    # Defines the independent variables for the regression.

    X = sm.add_constant(X)
    # Adds a constant term so the regression includes an intercept.

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    # Runs the OLS regression and fits the model.

    print(model.summary())
    # Prints the full regression summary.

    rv_coefficient = model.params["Monthly Realized Volatility"]
    # Extracts the coefficient of Monthly Realized Volatility.

    rv_p_value = model.pvalues["Monthly Realized Volatility"]
    # Extracts the p-value of Monthly Realized Volatility.

    climate_coefficient = model.params["Aggregate Climate Risk"]
    # Extracts the coefficient of Aggregate Climate Risk.

    climate_p_value = model.pvalues["Aggregate Climate Risk"]
    # Extracts the p-value of Aggregate Climate Risk.

    r_squared = model.rsquared
    # Extracts the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Extracts the adjusted R-squared value.

    n_observations = int(model.nobs)
    # Extracts the number of observations used in the regression.

    print("\nKey Results:")
    # Prints a heading for the extracted results.

    print(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}")
    # Prints the coefficient of Monthly Realized Volatility.

    print(f"P-value of Monthly Realized Volatility: {rv_p_value}")
    # Prints the p-value of Monthly Realized Volatility.

    print(f"Coefficient of Aggregate Climate Risk: {climate_coefficient}")
    # Prints the coefficient of Aggregate Climate Risk.

    print(f"P-value of Aggregate Climate Risk: {climate_p_value}")
    # Prints the p-value of Aggregate Climate Risk.

    print(f"R-squared: {r_squared}")
    # Prints the R-squared value.

    print(f"Adjusted R-squared: {adjusted_r_squared}")
    # Prints the adjusted R-squared value.

    print(f"Number of observations: {n_observations}")
    # Prints the number of observations.

    with open(output_file, "w", encoding="utf-8") as file:
        # Opens the output text file in write mode.

        file.write("AR + Aggregate Climate Risk Model Results\n")
        # Writes a title to the text file.

        file.write("=" * 60 + "\n\n")
        # Writes a separator line.

        file.write(str(model.summary()))
        # Writes the full regression summary to the text file.

        file.write("\n\nKey Results:\n")
        # Writes a heading for the extracted results.

        file.write(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}\n")
        # Writes the coefficient of Monthly Realized Volatility.

        file.write(f"P-value of Monthly Realized Volatility: {rv_p_value}\n")
        # Writes the p-value of Monthly Realized Volatility.

        file.write(f"Coefficient of Aggregate Climate Risk: {climate_coefficient}\n")
        # Writes the coefficient of Aggregate Climate Risk.

        file.write(f"P-value of Aggregate Climate Risk: {climate_p_value}\n")
        # Writes the p-value of Aggregate Climate Risk.

        file.write(f"R-squared: {r_squared}\n")
        # Writes the R-squared value.

        file.write(f"Adjusted R-squared: {adjusted_r_squared}\n")
        # Writes the adjusted R-squared value.

        file.write(f"Number of observations: {n_observations}\n")
        # Writes the number of observations.

    print(f"\nRegression summary saved successfully at:\n{output_file}")
    # Prints where the text file was saved.


if __name__ == "__main__":
    # Checks if the script is being run directly.

    main()
    # Runs the main function.