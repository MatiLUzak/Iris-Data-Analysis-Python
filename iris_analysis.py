import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to load the data
def load_data(file_path):
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

# Function to calculate Pearson correlation coefficient using numpy
def calculate_pearson_correlation_np(data):
    columns = data.columns[:-1]  # Exclude the 'species' column
    results = {}

    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1, col2 = columns[i], columns[j]
            correlation_matrix = np.corrcoef(data[col1], data[col2])
            correlation = correlation_matrix[0, 1]
            results[(col1, col2)] = correlation

    return results

# Function to calculate linear regression equation for each pair of features
def calculate_linear_regression_equations(data):
    columns = data.columns[:-1]  # Exclude the 'species' column
    results = {}

    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1, col2 = columns[i], columns[j]
            X = data[col1].values.reshape(-1, 1)
            y = data[col2].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            equation = f"y = {slope:.2f}x + {intercept:.2f}"
            results[(col1, col2)] = equation

    return results

# Main function to perform all calculations
def analyze_iris_data(file_path):
    data = load_data(file_path)
    pearson_correlations = calculate_pearson_correlation_np(data)
    linear_regression_equations = calculate_linear_regression_equations(data)
    return pearson_correlations, linear_regression_equations

if __name__ == '__main__':
    # File path to the data.csv
    file_path = '../pythonProject3/data.csv'

    # Perform the analysis
    pearson_correlations, linear_regression_equations = analyze_iris_data(file_path)

    # Display the results
    print("Pearson Correlation Coefficients:")
    for pair, correlation in pearson_correlations.items():
        print(f"{pair}: {correlation:.3f}")

    print("\nLinear Regression Equations:")
    for pair, equation in linear_regression_equations.items():
        print(f"{pair}: {equation}")