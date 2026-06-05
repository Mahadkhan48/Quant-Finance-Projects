# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:11:12 2026

@author: mahad
"""

# file: run_ar_pca_climate_risk_model.py

from pathlib import Path
# Imports Path so file paths are easier to manage.

import pandas as pd
# Imports pandas for reading, cleaning, and saving data.

import statsmodels.api as sm
# Imports statsmodels for running the OLS regression.


def main():
    # Defines the main function where the script runs.

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\SP500_Climate_Risk_Merged_Monthly_Data_With_PCA.xlsx"
    )
    # Stores the path of the input Excel file.

    output_text_file = input_file.parent / "AR_PCA_Climate_Risk_Model_Results.txt"
    # Creates the output text file path in the same folder.

    output_excel_file = input_file.parent / "PCA_Climate_Risk_Regression_Summary.xlsx"
    # Creates the output Excel summary file path in the same folder.

    df = pd.read_excel(input_file)
    # Reads the Excel file into a pandas DataFrame.

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    # Converts the Month column into datetime format.

    df = df.sort_values("Month").reset_index(drop=True)
    # Sorts the dataset by Month in ascending order.

    df = df.dropna(
        subset=[
            "Monthly Realized Volatility",
            "Next Month Realized Volatility",
            "PCA Climate Risk Index",
        ]
    )
    # Drops rows with missing values in the required columns.

    y = df["Next Month Realized Volatility"]
    # Defines the dependent variable for the regression.

    X = df[["Monthly Realized Volatility", "PCA Climate Risk Index"]]
    # Defines the independent variables for the regression.

    X = sm.add_constant(X)
    # Adds a constant so the regression includes an intercept.

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    # Runs the OLS regression and fits the model.

    print(model.summary())
    # Prints the full regression summary.

    rv_coefficient = model.params["Monthly Realized Volatility"]
    # Extracts the coefficient of Monthly Realized Volatility.

    rv_p_value = model.pvalues["Monthly Realized Volatility"]
    # Extracts the p-value of Monthly Realized Volatility.

    pca_coefficient = model.params["PCA Climate Risk Index"]
    # Extracts the coefficient of PCA Climate Risk Index.

    pca_p_value = model.pvalues["PCA Climate Risk Index"]
    # Extracts the p-value of PCA Climate Risk Index.

    r_squared = model.rsquared
    # Extracts the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Extracts the adjusted R-squared value.

    number_of_observations = int(model.nobs)
    # Extracts the number of observations used in the regression.

    print("\nKey Results:")
    # Prints a heading for the extracted results.

    print(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}")
    # Prints the coefficient of Monthly Realized Volatility.

    print(f"P-value of Monthly Realized Volatility: {rv_p_value}")
    # Prints the p-value of Monthly Realized Volatility.

    print(f"Coefficient of PCA Climate Risk Index: {pca_coefficient}")
    # Prints the coefficient of PCA Climate Risk Index.

    print(f"P-value of PCA Climate Risk Index: {pca_p_value}")
    # Prints the p-value of PCA Climate Risk Index.

    print(f"R-squared: {r_squared}")
    # Prints the R-squared value.

    print(f"Adjusted R-squared: {adjusted_r_squared}")
    # Prints the adjusted R-squared value.

    print(f"Number of observations: {number_of_observations}")
    # Prints the number of observations.

    with open(output_text_file, "w", encoding="utf-8") as file:
        # Opens the output text file in write mode.

        file.write("AR + PCA Climate Risk Model Results\n")
        # Writes the title of the regression output.

        file.write("=" * 60 + "\n\n")
        # Writes a separator line.

        file.write(str(model.summary()))
        # Writes the full regression summary.

        file.write("\n\nKey Results:\n")
        # Writes a heading for the key results section.

        file.write(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}\n")
        # Writes the coefficient of Monthly Realized Volatility.

        file.write(f"P-value of Monthly Realized Volatility: {rv_p_value}\n")
        # Writes the p-value of Monthly Realized Volatility.

        file.write(f"Coefficient of PCA Climate Risk Index: {pca_coefficient}\n")
        # Writes the coefficient of PCA Climate Risk Index.

        file.write(f"P-value of PCA Climate Risk Index: {pca_p_value}\n")
        # Writes the p-value of PCA Climate Risk Index.

        file.write(f"R-squared: {r_squared}\n")
        # Writes the R-squared value.

        file.write(f"Adjusted R-squared: {adjusted_r_squared}\n")
        # Writes the adjusted R-squared value.

        file.write(f"Number of observations: {number_of_observations}\n")
        # Writes the number of observations.

    summary_df = pd.DataFrame(
        [
            {
                "Model Name": "AR + PCA Climate Risk",
                "Climate Variable": "PCA Climate Risk Index",
                "Climate Variable Coefficient": pca_coefficient,
                "Climate Variable P-value": pca_p_value,
                "Monthly Realized Volatility Coefficient": rv_coefficient,
                "Monthly Realized Volatility P-value": rv_p_value,
                "R-squared": r_squared,
                "Adjusted R-squared": adjusted_r_squared,
                "Number of Observations": number_of_observations,
            }
        ]
    )
    # Creates a one-row summary table with the main regression results.

    summary_df.to_excel(output_excel_file, index=False)
    # Saves the summary table to an Excel file.

    print(f"\nRegression summary text file saved successfully at:\n{output_text_file}")
    # Prints the location of the saved text file.

    print(f"\nRegression summary Excel file saved successfully at:\n{output_excel_file}")
    # Prints the location of the saved Excel summary file.


if __name__ == "__main__":
    # Checks if the script is being run directly.

    main()
    # Runs the main function.