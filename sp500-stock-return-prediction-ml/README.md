# S&P 500 Stock Return Prediction

This project applies machine learning techniques to predict S&P 500 stock return direction using market-based features and a leakage-safe modelling workflow.

## Objective

The aim is to test whether historical market data and engineered financial indicators can help predict future S&P 500 return direction. The project compares simple baseline performance against machine learning models to assess whether the signals have practical value for trading or investment decision-making.

## Methodology

- Collected and cleaned S&P 500 market data
- Created return-based and technical features
- Defined the prediction target as future return direction
- Split the data before fitting scalers or models to avoid data leakage
- Built a baseline model for comparison
- Tested machine learning models such as Logistic Regression, Random Forest and XGBoost or similar classifiers
- Evaluated performance using ROC-AUC, confusion matrix and classification metrics
- Interpreted model outputs using feature importance or coefficient analysis

## Tools Used

Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter Notebook

## Current Status

This is an ongoing Applied Machine Learning coursework project. The repository currently includes working code for data preparation, feature engineering, model training and model evaluation.
