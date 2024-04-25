import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        for j in range(i + 1, len(columns)):
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
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            X = data[col1].values.reshape(-1, 1)
            y = data[col2].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            equation = f"y = {slope:.2f}x + {intercept:.2f}"
            results[(col1, col2)] = equation

    return results


# Assuming the data is in 'data.csv'
file_path = '../pythonProject3/data.csv'
data = load_data(file_path)

# Calculate Pearson correlation and regression equations
pearson_correlations = calculate_pearson_correlation_np(data)
linear_regression_equations = calculate_linear_regression_equations(data)

## Create individual plots
for (col1, col2), corr in pearson_correlations.items():
    plt.figure(figsize=(6, 4))
    plt.scatter(data[col1], data[col2], color='blue')

    # Add regression line
    equation = linear_regression_equations[(col1, col2)]
    # We remove 'y = ' and then split by 'x +'
    slope, intercept = [float(term) for term in equation.replace('y = ', '').split('x + ')]

    x_values = np.linspace(min(data[col1]), max(data[col1]), 100)
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color='red')

    # Set title and labels with Pearson correlation and regression equation
    plt.title(f"{col1} vs {col2}\nr = {corr:.2f}; {equation}")
    plt.xlabel(f"{col1} (cm)")
    plt.ylabel(f"{col2} (cm)")

    # Show plot
    plt.show()
