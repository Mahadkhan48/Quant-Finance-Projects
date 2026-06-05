# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:47:15 2026

@author: mahad
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

def main():
    

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\SP500_Climate_Risk_Merged_Monthly_Data.xlsx"
    )
    # Stores the path of the input Excel file.

    output_file = input_file.parent / "AR_Benchmark_Model_Results.txt"
    # Creates the output text file path in the same folder as the input file.

    df = pd.read_excel(input_file)
    # Reads the Excel file into a pandas DataFrame.

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    # Converts the Month column into datetime format.

    df = df.sort_values("Month").reset_index(drop=True)
    # Sorts the DataFrame by Month in ascending order.

    df = df.dropna(
        subset=["Monthly Realized Volatility", "Next Month Realized Volatility"]
    )
    # Drops rows where either volatility column has missing values.

    y = df["Next Month Realized Volatility"]
    # Defines the dependent variable for the regression.

    X = df[["Monthly Realized Volatility"]]
    # Defines the independent variable for the regression.

    X = sm.add_constant(X)
    # Adds a constant term so the regression includes an intercept.

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    # Runs the OLS regression and fits the model.

    print(model.summary())
    # Prints the full regression summary.

    coefficient = model.params["Monthly Realized Volatility"]
    # Extracts the coefficient of Monthly Realized Volatility.

    p_value = model.pvalues["Monthly Realized Volatility"]
    # Extracts the p-value of Monthly Realized Volatility.

    r_squared = model.rsquared
    # Extracts the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Extracts the adjusted R-squared value.

    n_observations = int(model.nobs)
    # Extracts the number of observations used in the regression.

    print("\nKey Results:")
    # Prints a heading for the extracted results.

    print(f"Coefficient of Monthly Realized Volatility: {coefficient}")
    # Prints the regression coefficient.

    print(f"P-value: {p_value}")
    # Prints the p-value.

    print(f"R-squared: {r_squared}")
    # Prints the R-squared.

    print(f"Adjusted R-squared: {adjusted_r_squared}")
    # Prints the adjusted R-squared.

    print(f"Number of observations: {n_observations}")
    # Prints the number of observations.

    with open(output_file, "w", encoding="utf-8") as file:
        # Opens the output text file in write mode.

        file.write("AR Benchmark Model Results\n")
        # Writes a title to the text file.

        file.write("=" * 50 + "\n\n")
        # Writes a separator line.

        file.write(str(model.summary()))
        # Writes the full regression summary.

        file.write("\n\nKey Results:\n")
        # Writes a heading for the extracted results.

        file.write(f"Coefficient of Monthly Realized Volatility: {coefficient}\n")
        # Writes the coefficient.

        file.write(f"P-value: {p_value}\n")
        # Writes the p-value.

        file.write(f"R-squared: {r_squared}\n")
        # Writes the R-squared.

        file.write(f"Adjusted R-squared: {adjusted_r_squared}\n")
        # Writes the adjusted R-squared.

        file.write(f"Number of observations: {n_observations}\n")
        # Writes the number of observations.

    print(f"\nRegression summary saved successfully at:\n{output_file}")
    


if __name__ == "__main__":
   

    main()