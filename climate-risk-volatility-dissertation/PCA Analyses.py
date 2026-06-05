# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:23:56 2026

@author: mahad
"""

# file: create_pca_climate_risk_index.py

from pathlib import Path
# Imports Path so file paths are easier to handle.

import pandas as pd
# Imports pandas for reading, cleaning, and saving Excel data.

from sklearn.decomposition import PCA
# Imports PCA from sklearn to create the principal component index.

from sklearn.preprocessing import StandardScaler
# Imports StandardScaler to standardize the climate variables before PCA.


def main():
    # Defines the main function where the script runs.

    input_file = Path(
        r"C:\MAHAD\Master in Quant\Subjects\Research Paper\Data and Codes\Data\SP500_Climate_Risk_Merged_Monthly_Data.xlsx"
    )
    # Stores the path of the input Excel file.

    output_data_file = input_file.parent / "SP500_Climate_Risk_Merged_Monthly_Data_With_PCA.xlsx"
    # Creates the output file path for the updated dataset.

    output_details_file = input_file.parent / "PCA_Climate_Risk_Index_Details.xlsx"
    # Creates the output file path for the PCA details.

    df = pd.read_excel(input_file)
    # Reads the Excel file into a pandas DataFrame.

    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    # Converts the Month column into datetime format.

    df = df.sort_values("Month").reset_index(drop=True)
    # Sorts the dataset by Month in ascending order.

    climate_columns = [
        "US climate policy",
        "International summits",
        "Global warming",
        "Natural disasters",
    ]
    # Defines the four climate variables used for PCA.

    pca_df = df.dropna(subset=climate_columns).copy()
    # Creates a new DataFrame containing only rows with no missing values in the four climate variables.

    scaler = StandardScaler()
    # Creates a StandardScaler object to standardize the variables.

    standardized_data = scaler.fit_transform(pca_df[climate_columns])
    # Standardizes the four climate variables so they have comparable scales.

    pca = PCA(n_components=1)
    # Creates a PCA object that keeps only the first principal component.

    first_component = pca.fit_transform(standardized_data)
    # Applies PCA to the standardized data and extracts the first principal component.

    pca_df["PCA Climate Risk Index"] = first_component[:, 0]
    # Creates a new column in the PCA DataFrame with the first principal component values.

    df["PCA Climate Risk Index"] = pd.NA
    # Creates the new PCA Climate Risk Index column in the original dataset.

    df.loc[pca_df.index, "PCA Climate Risk Index"] = pca_df["PCA Climate Risk Index"]
    # Adds the PCA Climate Risk Index values back into the original dataset at the matching row indexes.

    loadings_df = pd.DataFrame(
        {
            "Climate Variable": climate_columns,
            "PCA Loading": pca.components_[0],
        }
    )
    # Creates a DataFrame showing the PCA loading for each climate variable.

    explained_variance_df = pd.DataFrame(
        {
            "Metric": ["Explained Variance Ratio of First Principal Component"],
            "Value": [pca.explained_variance_ratio_[0]],
        }
    )
    # Creates a DataFrame showing the explained variance ratio of the first principal component.

    print("PCA Loadings / Weights:")
    # Prints a title for the PCA loadings.

    print(loadings_df)
    # Prints the PCA loadings DataFrame.

    print("\nExplained Variance Ratio of the First Principal Component:")
    # Prints a title for the explained variance ratio.

    print(pca.explained_variance_ratio_[0])
    # Prints the explained variance ratio value.

    df.to_excel(output_data_file, index=False)
    # Saves the updated dataset with the PCA index to an Excel file.

    with pd.ExcelWriter(output_details_file) as writer:
        # Opens an Excel writer so multiple sheets can be saved in one Excel file.

        loadings_df.to_excel(writer, sheet_name="PCA Loadings", index=False)
        # Saves the PCA loadings to the first sheet.

        explained_variance_df.to_excel(writer, sheet_name="Explained Variance", index=False)
        # Saves the explained variance ratio to the second sheet.

    print(f"\nUpdated dataset saved successfully at:\n{output_data_file}")
    # Prints the location of the updated dataset file.

    print(f"\nPCA details file saved successfully at:\n{output_details_file}")
    # Prints the location of the PCA details file.


if __name__ == "__main__":
    # Checks if the script is being run directly.

    main()
    # Runs the main function.