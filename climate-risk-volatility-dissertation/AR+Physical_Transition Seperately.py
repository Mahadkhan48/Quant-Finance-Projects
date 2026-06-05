# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:36:23 2026

@author: mahad
"""

# file: run_ar_physical_transition_risk_models.py

from pathlib import Path

import pandas as pd

import statsmodels.api as sm


def save_model_results(model, risk_column, output_file):
    # This function saves the regression summary and key results to a text file.

    rv_coefficient = model.params["Monthly Realized Volatility"]
    # Gets the coefficient of Monthly Realized Volatility.

    rv_p_value = model.pvalues["Monthly Realized Volatility"]
    # Gets the p-value of Monthly Realized Volatility.

    risk_coefficient = model.params[risk_column]
    # Gets the coefficient of the climate risk variable.

    risk_p_value = model.pvalues[risk_column]
    # Gets the p-value of the climate risk variable.

    r_squared = model.rsquared
    # Gets the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Gets the adjusted R-squared value.

    n_observations = int(model.nobs)
    # Gets the number of observations used in the regression.

    with open(output_file, "w", encoding="utf-8") as file:
        # Opens the output text file in write mode.

        file.write(f"AR + {risk_column} Model Results\n")
        # Writes the model title.

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

        file.write(f"Coefficient of {risk_column}: {risk_coefficient}\n")
        # Writes the coefficient of the climate risk variable.

        file.write(f"P-value of {risk_column}: {risk_p_value}\n")
        # Writes the p-value of the climate risk variable.

        file.write(f"R-squared: {r_squared}\n")
        # Writes the R-squared value.

        file.write(f"Adjusted R-squared: {adjusted_r_squared}\n")
        # Writes the adjusted R-squared value.

        file.write(f"Number of observations: {n_observations}\n")
        # Writes the number of observations.


def run_model(df, risk_column):
    # This function prepares the regression inputs and fits one OLS model.

    y = df["Next Month Realized Volatility"]
    # Defines the dependent variable.

    X = df[["Monthly Realized Volatility", risk_column]]
    # Defines the independent variables.

    X = sm.add_constant(X)
    # Adds a constant so the regression includes an intercept.

    model = sm.OLS(y, X).fit()
    # Fits the OLS regression model.

    return model
    # Returns the fitted model.


def print_model_results(model, risk_column, model_name):
    # This function prints the full summary and key results for one model.

    rv_coefficient = model.params["Monthly Realized Volatility"]
    # Gets the coefficient of Monthly Realized Volatility.

    rv_p_value = model.pvalues["Monthly Realized Volatility"]
    # Gets the p-value of Monthly Realized Volatility.

    risk_coefficient = model.params[risk_column]
    # Gets the coefficient of the climate risk variable.

    risk_p_value = model.pvalues[risk_column]
    # Gets the p-value of the climate risk variable.

    r_squared = model.rsquared
    # Gets the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Gets the adjusted R-squared value.

    n_observations = int(model.nobs)
    # Gets the number of observations used in the regression.

    print(f"\n{model_name}")
    # Prints the model name.

    print("=" * 60)
    # Prints a separator line.

    print(model.summary())
    # Prints the full regression summary.

    print("\nKey Results:")
    # Prints a heading for the key results.

    print(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}")
    # Prints the coefficient of Monthly Realized Volatility.

    print(f"P-value of Monthly Realized Volatility: {rv_p_value}")
    # Prints the p-value of Monthly Realized Volatility.

    print(f"Coefficient of {risk_column}: {risk_coefficient}")
    # Prints the coefficient of the climate risk variable.

    print(f"P-value of {risk_column}: {risk_p_value}")
    # Prints the p-value of the climate risk variable.

    print(f"R-squared: {r_squared}")
    # Prints the R-squared value.

    print(f"Adjusted R-squared: {adjusted_r_squared}")
    # Prints the adjusted R-squared value.

    print(f"Number of observations: {n_observations}")
    # Prints the number of observations.


def main():
    # Defines the main function where the script runs.

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\SP500_Climate_Risk_Merged_Monthly_Data.xlsx"
    )
    # Stores the path of the input Excel file.

    output_file_physical = input_file.parent / "AR_Physical_Risk_Model_Results.txt"
    # Creates the output text file path for the Physical Risk model.

    output_file_transition = input_file.parent / "AR_Transition_Risk_Model_Results.txt"
    # Creates the output text file path for the Transition Risk model.

    df = pd.read_excel(input_file)
    # Reads the Excel file into a pandas DataFrame.

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    # Converts the Month column to datetime format.

    df = df.sort_values("Month").reset_index(drop=True)
    # Sorts the DataFrame by Month in ascending order.

    df = df.dropna(
        subset=[
            "Monthly Realized Volatility",
            "Next Month Realized Volatility",
            "Physical Risk",
            "Transition Risk",
        ]
    )
    # Drops rows with missing values in the required columns.

    physical_model = run_model(df, "Physical Risk")
    # Runs Model 1 with Physical Risk.

    transition_model = run_model(df, "Transition Risk")
    # Runs Model 2 with Transition Risk.

    print_model_results(
        physical_model,
        "Physical Risk",
        "Model 1: AR + Physical Risk",
    )
    # Prints the full summary and key results for the Physical Risk model.

    print_model_results(
        transition_model,
        "Transition Risk",
        "Model 2: AR + Transition Risk",
    )
    # Prints the full summary and key results for the Transition Risk model.

    save_model_results(physical_model, "Physical Risk", output_file_physical)
    # Saves the Physical Risk model results to a text file.

    save_model_results(transition_model, "Transition Risk", output_file_transition)
    # Saves the Transition Risk model results to a text file.

    print(f"\nPhysical Risk model summary saved successfully at:\n{output_file_physical}")
    # Prints the save location for the Physical Risk model file.

    print(f"\nTransition Risk model summary saved successfully at:\n{output_file_transition}")
    # Prints the save location for the Transition Risk model file.


if __name__ == "__main__":
    # Checks if the script is being run directly.

    main()
    # Runs the main function.