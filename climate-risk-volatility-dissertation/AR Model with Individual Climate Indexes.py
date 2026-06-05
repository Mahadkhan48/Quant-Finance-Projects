# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:14:36 2026

@author: mahad
"""

# file: run_individual_climate_factor_models.py

from pathlib import Path
# Imports Path so file paths are easier to manage.

import pandas as pd
# Imports pandas for reading, cleaning, and saving data.

import statsmodels.api as sm
# Imports statsmodels for running OLS regressions.


def run_regression(dataframe, climate_factor):
    # This function runs one OLS regression for one climate factor.

    y = dataframe["Next Month Realized Volatility"]
    # Sets the dependent variable.

    X = dataframe[["Monthly Realized Volatility", climate_factor]]
    # Sets the independent variables.

    X = sm.add_constant(X)
    # Adds a constant so the regression includes an intercept.

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    # Fits the OLS regression model.

    return model
    # Returns the fitted regression model.


def print_key_results(model, climate_factor, model_name):
    # This function prints the full summary and key results for one model.

    rv_coefficient = model.params["Monthly Realized Volatility"]
    # Gets the coefficient of Monthly Realized Volatility.

    rv_p_value = model.pvalues["Monthly Realized Volatility"]
    # Gets the p-value of Monthly Realized Volatility.

    factor_coefficient = model.params[climate_factor]
    # Gets the coefficient of the climate factor.

    factor_p_value = model.pvalues[climate_factor]
    # Gets the p-value of the climate factor.

    r_squared = model.rsquared
    # Gets the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Gets the adjusted R-squared value.

    n_observations = int(model.nobs)
    # Gets the number of observations used in the model.

    print(f"\n{model_name}")
    # Prints the model name.

    print("=" * 70)
    # Prints a separator line.

    print(model.summary())
    # Prints the full regression summary.

    print("\nKey Results:")
    # Prints a heading for the key results.

    print(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}")
    # Prints the coefficient of Monthly Realized Volatility.

    print(f"P-value of Monthly Realized Volatility: {rv_p_value}")
    # Prints the p-value of Monthly Realized Volatility.

    print(f"Coefficient of {climate_factor}: {factor_coefficient}")
    # Prints the coefficient of the climate factor.

    print(f"P-value of {climate_factor}: {factor_p_value}")
    # Prints the p-value of the climate factor.

    print(f"R-squared: {r_squared}")
    # Prints the R-squared value.

    print(f"Adjusted R-squared: {adjusted_r_squared}")
    # Prints the adjusted R-squared value.

    print(f"Number of observations: {n_observations}")
    # Prints the number of observations.


def save_model_results(model, climate_factor, model_name, output_file):
    # This function saves one model summary and its key results to a text file.

    rv_coefficient = model.params["Monthly Realized Volatility"]
    # Gets the coefficient of Monthly Realized Volatility.

    rv_p_value = model.pvalues["Monthly Realized Volatility"]
    # Gets the p-value of Monthly Realized Volatility.

    factor_coefficient = model.params[climate_factor]
    # Gets the coefficient of the climate factor.

    factor_p_value = model.pvalues[climate_factor]
    # Gets the p-value of the climate factor.

    r_squared = model.rsquared
    # Gets the R-squared value.

    adjusted_r_squared = model.rsquared_adj
    # Gets the adjusted R-squared value.

    n_observations = int(model.nobs)
    # Gets the number of observations used in the model.

    with open(output_file, "w", encoding="utf-8") as file:
        # Opens the output text file in write mode.

        file.write(f"{model_name}\n")
        # Writes the model name as the title.

        file.write("=" * 70 + "\n\n")
        # Writes a separator line.

        file.write(str(model.summary()))
        # Writes the full regression summary.

        file.write("\n\nKey Results:\n")
        # Writes a heading for the key results section.

        file.write(f"Coefficient of Monthly Realized Volatility: {rv_coefficient}\n")
        # Writes the coefficient of Monthly Realized Volatility.

        file.write(f"P-value of Monthly Realized Volatility: {rv_p_value}\n")
        # Writes the p-value of Monthly Realized Volatility.

        file.write(f"Coefficient of {climate_factor}: {factor_coefficient}\n")
        # Writes the coefficient of the climate factor.

        file.write(f"P-value of {climate_factor}: {factor_p_value}\n")
        # Writes the p-value of the climate factor.

        file.write(f"R-squared: {r_squared}\n")
        # Writes the R-squared value.

        file.write(f"Adjusted R-squared: {adjusted_r_squared}\n")
        # Writes the adjusted R-squared value.

        file.write(f"Number of observations: {n_observations}\n")
        # Writes the number of observations.


def create_summary_row(model, climate_factor, model_name):
    # This function creates one summary row for the Excel summary table.

    return {
        "Model Name": model_name,
        "Climate Factor": climate_factor,
        "Climate Factor Coefficient": model.params[climate_factor],
        "Climate Factor P-value": model.pvalues[climate_factor],
        "Monthly Realized Volatility Coefficient": model.params["Monthly Realized Volatility"],
        "Monthly Realized Volatility P-value": model.pvalues["Monthly Realized Volatility"],
        "R-squared": model.rsquared,
        "Adjusted R-squared": model.rsquared_adj,
        "Number of Observations": int(model.nobs),
    }
    # Returns the summary information as a dictionary.


def main():
    # This is the main function where the full script runs.

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\SP500_Climate_Risk_Merged_Monthly_Data.xlsx"
    )
    # Stores the path of the input Excel file.

    output_folder = input_file.parent
    # Stores the folder where all output files will be saved.

    summary_output_file = output_folder / "Individual_Climate_Factor_Regression_Summary.xlsx"
    # Stores the path of the Excel summary file.

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
            "US climate policy",
            "International summits",
            "Global warming",
            "Natural disasters",
        ]
    )
    # Drops rows with missing values in the required columns.

    model_settings = [
        {
            "model_name": "Model 1: AR + Natural Disasters",
            "climate_factor": "Natural disasters",
            "output_file": output_folder / "AR_Natural_Disasters_Model_Results.txt",
        },
        {
            "model_name": "Model 2: AR + Global Warming",
            "climate_factor": "Global warming",
            "output_file": output_folder / "AR_Global_Warming_Model_Results.txt",
        },
        {
            "model_name": "Model 3: AR + International Summits",
            "climate_factor": "International summits",
            "output_file": output_folder / "AR_International_Summits_Model_Results.txt",
        },
        {
            "model_name": "Model 4: AR + US Climate Policy",
            "climate_factor": "US climate policy",
            "output_file": output_folder / "AR_US_Climate_Policy_Model_Results.txt",
        },
    ]
    # Creates a list of model details for the 4 separate regressions.

    summary_rows = []
    # Creates an empty list to store the summary results for the Excel file.

    for setting in model_settings:
        # Loops through each model configuration.

        model_name = setting["model_name"]
        # Gets the model name.

        climate_factor = setting["climate_factor"]
        # Gets the climate factor for the current model.

        output_file = setting["output_file"]
        # Gets the text output file path for the current model.

        model = run_regression(df, climate_factor)
        # Runs the regression for the current climate factor.

        print_key_results(model, climate_factor, model_name)
        # Prints the full summary and key results for the current model.

        save_model_results(model, climate_factor, model_name, output_file)
        # Saves the current model summary to a text file.

        summary_rows.append(create_summary_row(model, climate_factor, model_name))
        # Adds one row of summary results to the list.

        print(f"\nModel results saved successfully at:\n{output_file}")
        # Prints the path where the current text file was saved.

    summary_df = pd.DataFrame(summary_rows)
    # Converts the list of summary rows into a DataFrame.

    summary_df.to_excel(summary_output_file, index=False)
    # Saves the summary DataFrame to an Excel file.

    print(f"\nExcel summary table saved successfully at:\n{summary_output_file}")
    # Prints the path where the Excel summary file was saved.


if __name__ == "__main__":
    # Checks if the script is being run directly.

    main()
    # Runs the main function.